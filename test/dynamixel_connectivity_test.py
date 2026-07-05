"""
Connectivity test for the 7-DOF Dynamixel leader arm.

This script only opens the serial port, pings motor IDs 21-27, and reads a few
status registers. It does not enable torque and does not write motor settings.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from dynamixel_sdk import PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS


PROTOCOL_VERSION = 2.0

ADDR_MODEL_NUMBER = 0
ADDR_FIRMWARE_VERSION = 6
ADDR_ID = 7
ADDR_BAUD_RATE = 8
ADDR_RETURN_DELAY_TIME = 9
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE = 64
ADDR_PRESENT_POSITION = 132


@dataclass(frozen=True)
class MotorExpectation:
    dxl_id: int
    model_name: str


EXPECTED_MOTORS = [
    MotorExpectation(21, "XM430-W350"),
    MotorExpectation(22, "XM540-W270"),
    MotorExpectation(23, "XM430-W350"),
    MotorExpectation(24, "XM540-W270"),
    MotorExpectation(25, "XM430-W350"),
    MotorExpectation(26, "XM430-W350"),
    MotorExpectation(27, "XM430-W350"),
]


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


def open_port(port: str, baudrate: int) -> PortHandler:
    candidates = [port]
    fallback = windows_com_fallback(port)
    if fallback is not None:
        candidates.append(fallback)

    errors = []
    for candidate in candidates:
        handler = PortHandler(candidate)
        if not handler.openPort():
            errors.append(f"{candidate}: openPort failed")
            continue
        if not handler.setBaudRate(baudrate):
            handler.closePort()
            errors.append(f"{candidate}: setBaudRate({baudrate}) failed")
            continue
        print(f"Opened {candidate} at {baudrate} bps")
        return handler

    raise RuntimeError("Could not open Dynamixel port. " + "; ".join(errors))


def check_comm(packet: PacketHandler, result: int, error: int) -> str:
    if result != COMM_SUCCESS:
        return packet.getTxRxResult(result)
    if error != 0:
        return packet.getRxPacketError(error)
    return "OK"


def read_u8(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int):
    value, result, error = packet.read1ByteTxRx(port, dxl_id, address)
    return value, check_comm(packet, result, error)


def read_u4(port: PortHandler, packet: PacketHandler, dxl_id: int, address: int):
    value, result, error = packet.read4ByteTxRx(port, dxl_id, address)
    if value > 0x7FFFFFFF:
        value -= 0x100000000
    return value, check_comm(packet, result, error)


def main() -> int:
    parser = argparse.ArgumentParser(description="Ping and read status from Dynamixel IDs 21-27.")
    parser.add_argument("--port", default="COM21", help="Dynamixel serial port, default: COM21")
    parser.add_argument("--baudrate", type=int, default=57600, help="Bus baudrate, default: 57600")
    args = parser.parse_args()

    packet = PacketHandler(PROTOCOL_VERSION)
    port = open_port(args.port, args.baudrate)

    ok_count = 0
    try:
        print("")
        for expected in EXPECTED_MOTORS:
            dxl_id = expected.dxl_id
            model_number, result, error = packet.ping(port, dxl_id)
            status = check_comm(packet, result, error)
            if status != "OK":
                print(f"ID {dxl_id}: FAIL ping ({status}) expected={expected.model_name}")
                continue

            ok_count += 1
            firmware, firmware_status = read_u8(port, packet, dxl_id, ADDR_FIRMWARE_VERSION)
            id_reg, id_status = read_u8(port, packet, dxl_id, ADDR_ID)
            baud_reg, baud_status = read_u8(port, packet, dxl_id, ADDR_BAUD_RATE)
            delay_reg, delay_status = read_u8(port, packet, dxl_id, ADDR_RETURN_DELAY_TIME)
            mode, mode_status = read_u8(port, packet, dxl_id, ADDR_OPERATING_MODE)
            torque, torque_status = read_u8(port, packet, dxl_id, ADDR_TORQUE_ENABLE)
            position, position_status = read_u4(port, packet, dxl_id, ADDR_PRESENT_POSITION)

            print(
                f"ID {dxl_id}: OK expected={expected.model_name} model_no={model_number} "
                f"fw={firmware}({firmware_status}) id_reg={id_reg}({id_status}) "
                f"baud_reg={baud_reg}({baud_status}) return_delay={delay_reg}({delay_status}) "
                f"mode={mode}({mode_status}) torque={torque}({torque_status}) "
                f"present_position={position}({position_status})"
            )
    finally:
        port.closePort()
        print("")
        print("Closed Dynamixel port")

    print(f"Detected {ok_count}/{len(EXPECTED_MOTORS)} expected motors")
    return 0 if ok_count == len(EXPECTED_MOTORS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
