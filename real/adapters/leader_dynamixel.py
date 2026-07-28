from __future__ import annotations

"""Dynamixel-based leader arm reader for real teleoperation."""

from dataclasses import dataclass
import time
from typing import Optional, Sequence

import numpy as np
from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import COMM_PORT_BUSY, COMM_SUCCESS
from dynamixel_sdk.robotis_def import DXL_HIBYTE, DXL_LOBYTE


PROTOCOL_VERSION = 2.0
DEFAULT_JOINT_IDS = (21, 22, 23, 24, 25, 26, 27)
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PROFILE_ACCELERATION = 108
ADDR_PROFILE_VELOCITY = 112
ADDR_GOAL_POSITION = 116
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132
LEN_GOAL_CURRENT = 2
LEN_GOAL_POSITION = 4
LEN_SYNC_READ = 10
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1
RAW_PER_RAD = 2048.0 / np.pi
VEL_RAW_TO_RAD_PER_SEC = 0.229 * 2.0 * np.pi / 60.0


def windows_com_fallback(port: str) -> Optional[str]:
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


def open_dynamixel_port(port_name: str, baudrate: int) -> PortHandler:
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
        print(f"Opened Dynamixel port {candidate} at {baudrate} bps")
        return port

    raise RuntimeError("Could not open Dynamixel port. " + "; ".join(errors))


def signed_from_u16(value: int) -> int:
    return value - 0x10000 if value > 0x7FFF else value


def signed_from_u32(value: int) -> int:
    return value - 0x100000000 if value > 0x7FFFFFFF else value


def to_goal_current_param(raw_current: int) -> list[int]:
    value = int(raw_current) & 0xFFFF
    return [DXL_LOBYTE(value), DXL_HIBYTE(value)]


def to_goal_position_param(raw_position: int) -> list[int]:
    value = int(raw_position) & 0xFFFFFFFF
    return [
        value & 0xFF,
        (value >> 8) & 0xFF,
        (value >> 16) & 0xFF,
        (value >> 24) & 0xFF,
    ]


def comm_ok(packet: PacketHandler, result: int, error: int, action: str, dxl_id: int | None = None) -> None:
    prefix = f"ID {dxl_id}: " if dxl_id is not None else ""
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{prefix}{action} failed: {packet.getTxRxResult(result)}")
    if error != 0:
        raise RuntimeError(f"{prefix}{action} failed: {packet.getRxPacketError(error)}")


@dataclass
class LeaderState:
    timestamp_monotonic: float
    raw_position: np.ndarray
    raw_velocity: np.ndarray
    raw_current: np.ndarray
    q_motor: np.ndarray
    dq_motor: np.ndarray


class LeaderDynamixelReader:
    """Read and control the 7-DoF Dynamixel leader arm."""

    def __init__(
        self,
        port_name: str,
        baudrate: int,
        read_retries: int = 3,
        retry_delay: float = 0.01,
        joint_ids: Sequence[int] = DEFAULT_JOINT_IDS,
        max_current_raw: Sequence[int] | None = None,
    ):
        self.port_name = str(port_name)
        self.baudrate = int(baudrate)
        self.read_retries = int(read_retries)
        self.retry_delay = float(retry_delay)
        self.joint_ids = tuple(int(x) for x in joint_ids)
        if max_current_raw is not None:
            self.max_current_raw = np.broadcast_to(
                np.asarray(max_current_raw, dtype=int), len(self.joint_ids)
            ).copy()
        else:
            self.max_current_raw = None
        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.port = open_dynamixel_port(self.port_name, self.baudrate)
        self.reader = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_CURRENT, LEN_SYNC_READ)
        self.writer = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
        for dxl_id in self.joint_ids:
            if not self.reader.addParam(dxl_id):
                raise RuntimeError(f"ID {dxl_id}: failed to add sync-read parameter")

    def close(self) -> None:
        try:
            self.port.closePort()
        finally:
            print("Closed Dynamixel port")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _recover_port(self) -> None:
        self.port.is_using = False
        try:
            self.port.clearPort()
        except Exception:
            pass

    def set_torque(self, enable: bool) -> None:
        value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        errors = []
        for dxl_id in self.joint_ids:
            try:
                result, error = self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, value)
                comm_ok(self.packet, result, error, "set torque", dxl_id)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise RuntimeError("; ".join(errors))

    def set_position_control_mode(self, profile_velocity: int, profile_acceleration: int) -> None:
        profile_velocity = int(profile_velocity)
        profile_acceleration = int(profile_acceleration)
        if profile_velocity <= 0:
            raise ValueError(f"profile_velocity must be positive, got {profile_velocity}")
        if profile_acceleration < 0:
            raise ValueError(f"profile_acceleration must be non-negative, got {profile_acceleration}")

        self.set_torque(False)
        time.sleep(0.05)
        for dxl_id in self.joint_ids:
            result, error = self.packet.write1ByteTxRx(
                self.port,
                dxl_id,
                ADDR_OPERATING_MODE,
                POSITION_CONTROL_MODE,
            )
            comm_ok(self.packet, result, error, "set position control mode", dxl_id)
            result, error = self.packet.write4ByteTxRx(
                self.port,
                dxl_id,
                ADDR_PROFILE_ACCELERATION,
                profile_acceleration,
            )
            comm_ok(self.packet, result, error, "set profile acceleration", dxl_id)
            result, error = self.packet.write4ByteTxRx(
                self.port,
                dxl_id,
                ADDR_PROFILE_VELOCITY,
                profile_velocity,
            )
            comm_ok(self.packet, result, error, "set profile velocity", dxl_id)
        time.sleep(0.2)

    def set_current_control_mode(self) -> None:
        self.set_torque(False)
        time.sleep(0.05)
        for dxl_id in self.joint_ids:
            result, error = self.packet.write1ByteTxRx(
                self.port,
                dxl_id,
                ADDR_OPERATING_MODE,
                CURRENT_CONTROL_MODE,
            )
            comm_ok(self.packet, result, error, "set current control mode", dxl_id)
        time.sleep(0.2)

    def write_goal_position(self, raw_positions: np.ndarray) -> None:
        values = np.asarray(raw_positions, dtype=int).reshape(-1)
        if values.shape != (len(self.joint_ids),):
            raise ValueError(f"raw_positions must have shape ({len(self.joint_ids)},), got {values.shape}")

        writer = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_POSITION, LEN_GOAL_POSITION)
        last_result = None
        for _ in range(2):
            writer.clearParam()
            for dxl_id, raw_position in zip(self.joint_ids, values):
                if not writer.addParam(dxl_id, to_goal_position_param(int(raw_position))):
                    raise RuntimeError(f"ID {dxl_id}: failed to add goal position parameter")
            result = writer.txPacket()
            writer.clearParam()
            if result == COMM_SUCCESS:
                return
            if result == COMM_PORT_BUSY:
                self._recover_port()
            last_result = result
            time.sleep(self.retry_delay)
        raise RuntimeError(f"sync write goal position failed: {self.packet.getTxRxResult(last_result)}")

    def home_to_raw_position(
        self,
        target_raw: Sequence[int],
        *,
        profile_velocity: int = 20,
        profile_acceleration: int = 5,
        position_tolerance_ticks: int = 15,
        velocity_tolerance_raw: int = 2,
        settle_time_s: float = 0.3,
        timeout_s: float = 15.0,
        max_delta_ticks: int = 1200,
        poll_hz: float = 20.0,
        verbose: bool = True,
    ) -> LeaderState:
        target = np.asarray(target_raw, dtype=np.float64).reshape(-1)
        expected_shape = (len(self.joint_ids),)
        if target.shape != expected_shape:
            raise ValueError(f"target_raw must have shape {expected_shape}, got {target.shape}")
        if not np.isfinite(target).all() or not np.equal(target, np.rint(target)).all():
            raise ValueError("target_raw must contain finite integer values")
        target = np.rint(target).astype(int)
        if np.any(target < 0) or np.any(target > 4095):
            raise ValueError(f"target_raw must stay within [0, 4095], got {target.tolist()}")

        position_tolerance_ticks = int(position_tolerance_ticks)
        velocity_tolerance_raw = int(velocity_tolerance_raw)
        max_delta_ticks = int(max_delta_ticks)
        settle_time_s = float(settle_time_s)
        timeout_s = float(timeout_s)
        poll_hz = float(poll_hz)
        if position_tolerance_ticks < 0:
            raise ValueError("position_tolerance_ticks must be non-negative")
        if velocity_tolerance_raw < 0:
            raise ValueError("velocity_tolerance_raw must be non-negative")
        if max_delta_ticks < 0:
            raise ValueError("max_delta_ticks must be non-negative")
        if settle_time_s < 0.0:
            raise ValueError("settle_time_s must be non-negative")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive")
        if poll_hz <= 0.0:
            raise ValueError("poll_hz must be positive")

        initial_state = self.read_state()
        initial_raw = np.rint(initial_state.raw_position).astype(int)
        delta = target - initial_raw
        largest_delta = int(np.max(np.abs(delta)))
        if verbose:
            print("[leader_home] current_raw =", initial_raw)
            print("[leader_home] target_raw  =", target)
            print(f"[leader_home] max move = {largest_delta} ticks")
        if largest_delta > max_delta_ticks:
            raise RuntimeError(
                f"Leader home refused: max move {largest_delta} ticks exceeds "
                f"max_delta_ticks={max_delta_ticks}"
            )

        final_state = initial_state
        settled_since: float | None = None
        poll_dt = 1.0 / poll_hz
        try:
            self.set_position_control_mode(profile_velocity, profile_acceleration)
            # XM430/XM540 copy Present Position into Goal Position when torque is
            # enabled in position mode, so the synchronized goal must follow it.
            self.set_torque(True)
            self.write_goal_position(target)
            deadline = time.monotonic() + timeout_s

            while True:
                now = time.monotonic()
                if now >= deadline:
                    error = np.abs(target - np.rint(final_state.raw_position).astype(int))
                    raise TimeoutError(
                        f"Leader home timed out after {timeout_s:.2f}s; "
                        f"position_error_ticks={error.tolist()}"
                    )

                final_state = self.read_state()
                position_error = np.abs(target - np.rint(final_state.raw_position).astype(int))
                velocity = np.abs(np.rint(final_state.raw_velocity).astype(int))
                is_settled = bool(
                    np.all(position_error <= position_tolerance_ticks)
                    and np.all(velocity <= velocity_tolerance_raw)
                )
                if is_settled:
                    if settled_since is None:
                        settled_since = now
                    if (now - settled_since) >= settle_time_s:
                        if verbose:
                            print("[leader_home] reached target; position_error_ticks =", position_error)
                        break
                else:
                    settled_since = None
                time.sleep(poll_dt)
        except BaseException:
            try:
                self.set_torque(False)
            except Exception as shutdown_exc:
                print(f"[leader_home] warning: failed to disable torque after error: {shutdown_exc}")
            raise

        # Position-hold torque must never leak into current-control startup.
        self.set_torque(False)
        return final_state

    def write_goal_current(self, raw_currents: np.ndarray) -> None:
        values = np.asarray(raw_currents, dtype=int).reshape(-1)
        if values.shape != (len(self.joint_ids),):
            raise ValueError(f"raw_currents must have shape ({len(self.joint_ids)},), got {values.shape}")
        if self.max_current_raw is not None:
            values = np.clip(values, -self.max_current_raw, self.max_current_raw)
        last_result = None
        for _ in range(2):
            self.writer.clearParam()
            for dxl_id, raw_current in zip(self.joint_ids, values):
                if not self.writer.addParam(dxl_id, to_goal_current_param(int(raw_current))):
                    raise RuntimeError(f"ID {dxl_id}: failed to add goal current parameter")
            result = self.writer.txPacket()
            self.writer.clearParam()
            if result == COMM_SUCCESS:
                return
            if result == COMM_PORT_BUSY:
                self._recover_port()
            last_result = result
            time.sleep(self.retry_delay)
        raise RuntimeError(f"sync write goal current failed: {self.packet.getTxRxResult(last_result)}")

    def write_goal_current_individual(self, raw_currents: np.ndarray) -> list[str]:
        values = np.asarray(raw_currents, dtype=int).reshape(-1)
        if values.shape != (len(self.joint_ids),):
            raise ValueError(f"raw_currents must have shape ({len(self.joint_ids)},), got {values.shape}")
        errors = []
        for dxl_id, raw_current in zip(self.joint_ids, values):
            value = int(raw_current) & 0xFFFF
            ok = False
            last_message = ""
            for _ in range(3):
                self._recover_port()
                result, error = self.packet.write2ByteTxRx(self.port, dxl_id, ADDR_GOAL_CURRENT, value)
                if result == COMM_SUCCESS and error == 0:
                    ok = True
                    break
                if result != COMM_SUCCESS:
                    last_message = self.packet.getTxRxResult(result)
                else:
                    last_message = self.packet.getRxPacketError(error)
                time.sleep(self.retry_delay)
            if not ok:
                errors.append(f"ID {dxl_id}: zero current write failed: {last_message}")
        return errors

    def safe_shutdown(self) -> None:
        zero = np.zeros(len(self.joint_ids), dtype=int)
        warnings = []
        time.sleep(0.05)
        self._recover_port()
        try:
            self.write_goal_current(zero)
        except Exception as exc:
            warnings.append(f"sync zero-current write failed: {exc}")
            time.sleep(0.1)
            self._recover_port()
            warnings.extend(self.write_goal_current_individual(zero))

        for dxl_id in self.joint_ids:
            try:
                self._recover_port()
                result, error = self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                comm_ok(self.packet, result, error, "disable torque", dxl_id)
            except Exception as exc:
                warnings.append(f"ID {dxl_id}: torque disable failed: {exc}")

        if warnings:
            print("Leader shutdown warnings:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("Wrote zero current and disabled leader torque.")

    def _sync_read_once(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        last_result = None
        for _ in range(self.read_retries + 1):
            result = self.reader.txRxPacket()
            if result == COMM_SUCCESS:
                break
            if result == COMM_PORT_BUSY:
                self._recover_port()
            last_result = result
            time.sleep(self.retry_delay)
        else:
            raise RuntimeError(f"sync read failed: {self.packet.getTxRxResult(last_result)}")

        if result != COMM_SUCCESS:
            raise RuntimeError(f"sync read failed: {self.packet.getTxRxResult(result)}")

        raw_current = []
        raw_velocity = []
        raw_position = []
        for dxl_id in self.joint_ids:
            for address, length, label in (
                (ADDR_PRESENT_CURRENT, 2, "present current"),
                (ADDR_PRESENT_VELOCITY, 4, "present velocity"),
                (ADDR_PRESENT_POSITION, 4, "present position"),
            ):
                if not self.reader.isAvailable(dxl_id, address, length):
                    raise RuntimeError(f"ID {dxl_id}: {label} not available")
            raw_current.append(self.reader.getData(dxl_id, ADDR_PRESENT_CURRENT, 2))
            raw_velocity.append(self.reader.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4))
            raw_position.append(self.reader.getData(dxl_id, ADDR_PRESENT_POSITION, 4))

        return (
            np.asarray([signed_from_u32(int(v)) for v in raw_position], dtype=np.float64),
            np.asarray([signed_from_u32(int(v)) for v in raw_velocity], dtype=np.float64),
            np.asarray([signed_from_u16(int(v)) for v in raw_current], dtype=np.float64),
        )

    def read_state(self) -> LeaderState:
        raw_position, raw_velocity, raw_current = self._sync_read_once()
        q_motor = raw_position / RAW_PER_RAD
        dq_motor = raw_velocity * VEL_RAW_TO_RAD_PER_SEC
        return LeaderState(
            timestamp_monotonic=time.monotonic(),
            raw_position=raw_position,
            raw_velocity=raw_velocity,
            raw_current=raw_current,
            q_motor=q_motor,
            dq_motor=dq_motor,
        )
