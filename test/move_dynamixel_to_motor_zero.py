"""
Move the 7-DOF Dynamixel leader arm to a position target in position control mode.

Default behavior is a dry run: it reads present positions and prints the target.
Use --execute to actually switch to position control, enable torque, and command
the motors to the mapped target.

Examples:
    python test/move_dynamixel_to_motor_zero.py
    python test/move_dynamixel_to_motor_zero.py --robot-rad 0 0 0 -1.57 0 1.57 0 --execute
    python test/move_dynamixel_to_motor_zero.py --raw 2048 2048 2048 2048 2048 2048 2048 --execute
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np
from dynamixel_sdk import GroupSyncWrite, PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)


PROTOCOL_VERSION = 2.0

JOINT_IDS = (21, 22, 23, 24, 25, 26, 27)
Q_M0 = np.array(np.deg2rad([180, 180, 180, 180, 180, 180, 180]), dtype=float)
Q_R0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0], dtype=float)
SIGN = np.array([1, 1, 1, -1, 1, -1, 1], dtype=float)

ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_POSITION = 132

LEN_GOAL_POSITION = 4

TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
POSITION_CONTROL_MODE = 3

RAW_PER_RAD = 2048.0 / np.pi


@dataclass(frozen=True)
class MotorState:
    dxl_id: int
    raw_position: int
    motor_rad: float
    robot_rad: float


def windows_com_fallback(port: str) -> str | None:
    upper = port.upper()
    if not upper.startswith("COM"):
        return None
    try:
        number = int(upper[3:])
    except ValueError:
        return None
    if number <= 9:
        return None
    return "\\\\.\\" + upper


def open_port(port_name: str, baudrate: int) -> PortHandler:
    candidates = [port_name]
    fallback = windows_com_fallback(port_name)
    if fallback is not None:
        candidates.append(fallback)

    errors = []
    for candidate in candidates:
        port = PortHandler(candidate)
        try:
            opened = port.openPort()
        except Exception as exc:
            errors.append(f"{candidate}: openPort exception: {exc}")
            continue
        if not opened:
            errors.append(f"{candidate}: openPort failed")
            continue
        if not port.setBaudRate(baudrate):
            port.closePort()
            errors.append(f"{candidate}: setBaudRate({baudrate}) failed")
            continue
        print(f"Opened {candidate} at {baudrate} bps")
        return port

    raise RuntimeError("Could not open Dynamixel port. " + "; ".join(errors))


def check_comm(packet: PacketHandler, result: int, error: int, action: str, dxl_id: int) -> None:
    if result != COMM_SUCCESS:
        raise RuntimeError(f"ID {dxl_id}: {action} failed: {packet.getTxRxResult(result)}")
    if error != 0:
        raise RuntimeError(f"ID {dxl_id}: {action} failed: {packet.getRxPacketError(error)}")


def read_present_position(port: PortHandler, packet: PacketHandler, dxl_id: int) -> int:
    raw, result, error = packet.read4ByteTxRx(port, dxl_id, ADDR_PRESENT_POSITION)
    check_comm(packet, result, error, "read present position", dxl_id)
    if raw > 0x7FFFFFFF:
        raw -= 0x100000000
    return raw


def raw_to_motor_rad(raw: int) -> float:
    return raw / RAW_PER_RAD


def motor_to_robot_rad(q_motor: np.ndarray) -> np.ndarray:
    return Q_R0 + SIGN * (q_motor - Q_M0)


def robot_to_motor_rad(q_robot: np.ndarray) -> np.ndarray:
    return Q_M0 + SIGN * (q_robot - Q_R0)


def motor_rad_to_raw(q_motor: np.ndarray) -> np.ndarray:
    return np.rint(q_motor * RAW_PER_RAD).astype(int)


def read_states(port: PortHandler, packet: PacketHandler) -> list[MotorState]:
    raw_positions = np.array([read_present_position(port, packet, dxl_id) for dxl_id in JOINT_IDS])
    motor_rad = raw_positions / RAW_PER_RAD
    robot_rad = motor_to_robot_rad(motor_rad)
    return [
        MotorState(dxl_id, int(raw), float(qm), float(qr))
        for dxl_id, raw, qm, qr in zip(JOINT_IDS, raw_positions, motor_rad, robot_rad)
    ]


def write_1(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int, value: int, action: str) -> None:
    result, error = packet.write1ByteTxRx(port, dxl_id, address, value)
    check_comm(packet, result, error, action, dxl_id)


def write_4(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int, value: int, action: str) -> None:
    result, error = packet.write4ByteTxRx(port, dxl_id, address, value)
    check_comm(packet, result, error, action, dxl_id)


def print_states(states: list[MotorState], target_raw: np.ndarray) -> None:
    print("")
    print("Current state:")
    for state, raw_target in zip(states, target_raw):
        delta_raw = int(raw_target) - state.raw_position
        print(
            f"ID {state.dxl_id}: raw={state.raw_position:5d}, "
            f"motor={state.motor_rad:+.4f} rad, robot={state.robot_rad:+.4f} rad, "
            f"target_raw={int(raw_target):5d}, delta={delta_raw:+5d} ticks"
        )


def build_goal_param(raw_position: int) -> list[int]:
    raw_position &= 0xFFFFFFFF
    return [
        DXL_LOBYTE(DXL_LOWORD(raw_position)),
        DXL_HIBYTE(DXL_LOWORD(raw_position)),
        DXL_LOBYTE(DXL_HIWORD(raw_position)),
        DXL_HIBYTE(DXL_HIWORD(raw_position)),
    ]


def command_goal_position(port: PortHandler, packet: PacketHandler, target_raw: np.ndarray) -> None:
    sync_write = GroupSyncWrite(port, packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
    for dxl_id, raw in zip(JOINT_IDS, target_raw):
        param = build_goal_param(int(raw))
        if not sync_write.addParam(dxl_id, param):
            raise RuntimeError(f"ID {dxl_id}: failed to add goal position parameter")
    result = sync_write.txPacket()
    sync_write.clearParam()
    if result != COMM_SUCCESS:
        raise RuntimeError(f"sync write goal position failed: {packet.getTxRxResult(result)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Move leader Dynamixels to a robot-angle or direct raw target.")
    parser.add_argument("--port", default="COM21", help="Dynamixel serial port, default: COM21")
    parser.add_argument("--baudrate", type=int, default=57600, help="Bus baudrate, default: 57600")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--robot-rad",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Target robot joint angles in radians. Defaults to Q_R0 if omitted.",
    )
    target_group.add_argument(
        "--robot-deg",
        type=float,
        nargs=7,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Target robot joint angles in degrees. Defaults to Q_R0 if omitted.",
    )
    target_group.add_argument(
        "--raw",
        type=int,
        nargs=7,
        metavar=("M1", "M2", "M3", "M4", "M5", "M6", "M7"),
        help="Direct target motor raw Goal Position ticks for IDs 21-27. Bypasses robot angle mapping.",
    )
    parser.add_argument("--profile-velocity", type=int, default=20, help="Position profile velocity, default: 20")
    parser.add_argument("--profile-acceleration", type=int, default=5, help="Position profile acceleration, default: 5")
    parser.add_argument("--settle-sec", type=float, default=5.0, help="Time to wait after command, default: 5.0")
    parser.add_argument(
        "--max-delta-ticks",
        type=int,
        default=1200,
        help="Refuse --execute if any motor move exceeds this many ticks, default: 1200",
    )
    parser.add_argument("--force", action="store_true", help="Allow moves larger than --max-delta-ticks")
    parser.add_argument("--execute", action="store_true", help="Actually move the motors")
    args = parser.parse_args()

    direct_raw_mode = args.raw is not None
    if direct_raw_mode:
        target_raw = np.array(args.raw, dtype=int)
        target_motor_rad = target_raw / RAW_PER_RAD
        target_robot_rad = motor_to_robot_rad(target_motor_rad)
    elif args.robot_rad is not None:
        target_robot_rad = np.array(args.robot_rad, dtype=float)
        target_motor_rad = robot_to_motor_rad(target_robot_rad)
        target_raw = motor_rad_to_raw(target_motor_rad)
    elif args.robot_deg is not None:
        target_robot_rad = np.deg2rad(np.array(args.robot_deg, dtype=float))
        target_motor_rad = robot_to_motor_rad(target_robot_rad)
        target_raw = motor_rad_to_raw(target_motor_rad)
    else:
        target_robot_rad = Q_R0.copy()
        target_motor_rad = robot_to_motor_rad(target_robot_rad)
        target_raw = motor_rad_to_raw(target_motor_rad)

    if np.any(target_raw < 0) or np.any(target_raw > 4095):
        raise ValueError(
            "Target maps outside single-turn position range [0, 4095]. "
            f"target_raw={target_raw.tolist()}"
        )

    packet = PacketHandler(PROTOCOL_VERSION)
    port = open_port(args.port, args.baudrate)

    try:
        states = read_states(port, packet)
        print_states(states, target_raw)

        print("")
        if direct_raw_mode:
            print("Target mode:                direct motor raw ticks")
            print("Requested raw ticks:        " + np.array2string(target_raw))
            print("Equivalent robot rad:       " + np.array2string(target_robot_rad, precision=4))
            print("Equivalent robot deg:       " + np.array2string(np.rad2deg(target_robot_rad), precision=2))
        else:
            print("Target mode:                robot joint angle mapped to motor raw ticks")
            print("Requested robot target rad: " + np.array2string(target_robot_rad, precision=4))
            print("Requested robot target deg: " + np.array2string(np.rad2deg(target_robot_rad), precision=2))
        print("Mapped motor target rad:    " + np.array2string(target_motor_rad, precision=4))
        print("Mapped target raw ticks:    " + np.array2string(target_raw))
        print("Robot zero Q_R0 rad:        " + np.array2string(Q_R0, precision=4))
        print("Motor zero Q_M0 rad:        " + np.array2string(Q_M0, precision=4))
        print("Joint sign:                 " + np.array2string(SIGN, precision=1))

        current_raw = np.array([state.raw_position for state in states], dtype=int)
        max_delta = int(np.max(np.abs(target_raw - current_raw)))
        print(f"Max move: {max_delta} ticks")

        if not args.execute:
            print("")
            print("Dry run only. Re-run with --execute to move the arm.")
            return 0

        if max_delta > args.max_delta_ticks and not args.force:
            raise RuntimeError(
                f"Refusing to execute: max move {max_delta} ticks exceeds "
                f"--max-delta-ticks {args.max_delta_ticks}. Re-run with --force if this is intended."
            )

        print("")
        print("Switching to position control mode and commanding target.")
        for dxl_id in JOINT_IDS:
            write_1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE, "disable torque")
            write_1(port, packet, dxl_id, ADDR_OPERATING_MODE, POSITION_CONTROL_MODE, "set position mode")
            write_4(port, packet, dxl_id, ADDR_PROFILE_ACCELERATION, args.profile_acceleration, "set profile acceleration")
            write_4(port, packet, dxl_id, ADDR_PROFILE_VELOCITY, args.profile_velocity, "set profile velocity")
            write_1(port, packet, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE, "enable torque")

        command_goal_position(port, packet, target_raw)
        time.sleep(args.settle_sec)

        final_states = read_states(port, packet)
        print_states(final_states, target_raw)
        print("")
        print("Done. Torque remains enabled so the arm can hold the commanded position.")
        print("Use Dynamixel Wizard or a shutdown script to disable torque when safe.")
        return 0
    finally:
        port.closePort()
        print("")
        print("Closed Dynamixel port")


if __name__ == "__main__":
    raise SystemExit(main())
