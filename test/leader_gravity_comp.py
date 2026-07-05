"""
Gravity compensation for the physical 7-DOF Dynamixel leader arm.

This is a Windows/no-ROS script that reuses the core FACTRTeleop gravity
compensation formula:

    tau_g = pin.rnea(model, data, q, dq, zeros)

Default behavior is a dry run. Use --execute to switch the motors to Current
Control Mode, enable torque, and stream Goal Current commands.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pinocchio as pin
import yaml
from dynamixel_sdk import GroupSyncRead, GroupSyncWrite, PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import COMM_PORT_BUSY, COMM_SUCCESS, DXL_HIBYTE, DXL_LOBYTE


PROTOCOL_VERSION = 2.0

JOINT_IDS = (21, 22, 23, 24, 25, 26, 27)
SERVO_TYPES = (
    "XM430-W350",
    "XM540-W270",
    "XM430-W350",
    "XM540-W270",
    "XM430-W350",
    "XM430-W350",
    "XM430-W350",
)
Q_M0 = np.array(np.deg2rad([180, 180, 180, 180, 180, 180, 180]), dtype=float)
Q_R0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
SIGN = np.array([1, 1, 1, 1, 1, 1, 1], dtype=float)

KT_NM_PER_AMP = {
    "XM430-W350": 1.783,
    "XM540-W270": 2.409,
}
GOAL_CURRENT_UNIT_AMP = 0.00269
KT = np.array([KT_NM_PER_AMP[servo] for servo in SERVO_TYPES], dtype=float)
DEFAULT_JOINT_GAIN = np.ones(7, dtype=float)
DEFAULT_CURRENT_DEADBAND_RAW = np.array([0, 0, 12, 0, 0, 0, 0], dtype=int)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("leader_comp_config.yaml")

ADDR_OPERATING_MODE = 11
ADDR_CURRENT_LIMIT = 38
ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
ADDR_PRESENT_CURRENT = 126
ADDR_PRESENT_VELOCITY = 128
ADDR_PRESENT_POSITION = 132

LEN_GOAL_CURRENT = 2
LEN_SYNC_READ = 10

CURRENT_CONTROL_MODE = 0
TORQUE_DISABLE = 0
TORQUE_ENABLE = 1

RAW_PER_RAD = 2048.0 / np.pi
VEL_RAW_TO_RAD_PER_SEC = 0.229 * 2.0 * np.pi / 60.0


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


def comm_ok(packet: PacketHandler, result: int, error: int, action: str, dxl_id: int | None = None) -> None:
    prefix = f"ID {dxl_id}: " if dxl_id is not None else ""
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{prefix}{action} failed: {packet.getTxRxResult(result)}")
    if error != 0:
        raise RuntimeError(f"{prefix}{action} failed: {packet.getRxPacketError(error)}")


def signed_from_u16(value: int) -> int:
    return value - 0x10000 if value > 0x7FFF else value


def signed_from_u32(value: int) -> int:
    return value - 0x100000000 if value > 0x7FFFFFFF else value


def to_goal_current_param(raw_current: int) -> list[int]:
    value = int(raw_current) & 0xFFFF
    return [DXL_LOBYTE(value), DXL_HIBYTE(value)]


class DynamixelCurrentController:
    def __init__(self, port_name: str, baudrate: int, read_retries: int, retry_delay: float):
        self.packet = PacketHandler(PROTOCOL_VERSION)
        self.port = open_dynamixel_port(port_name, baudrate)
        self.read_retries = read_retries
        self.retry_delay = retry_delay
        self.reader = GroupSyncRead(self.port, self.packet, ADDR_PRESENT_CURRENT, LEN_SYNC_READ)
        self.writer = GroupSyncWrite(self.port, self.packet, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
        for dxl_id in JOINT_IDS:
            if not self.reader.addParam(dxl_id):
                raise RuntimeError(f"ID {dxl_id}: failed to add sync-read parameter")

    def close(self) -> None:
        self.port.closePort()
        print("Closed Dynamixel port")

    def recover_port(self) -> None:
        self.port.is_using = False
        try:
            self.port.clearPort()
        except Exception:
            pass

    def set_torque(self, enable: bool) -> None:
        value = TORQUE_ENABLE if enable else TORQUE_DISABLE
        for dxl_id in JOINT_IDS:
            result, error = self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, value)
            comm_ok(self.packet, result, error, "set torque", dxl_id)

    def set_current_control_mode(self) -> None:
        self.set_torque(False)
        time.sleep(0.05)
        for dxl_id in JOINT_IDS:
            result, error = self.packet.write1ByteTxRx(
                self.port, dxl_id, ADDR_OPERATING_MODE, CURRENT_CONTROL_MODE
            )
            comm_ok(self.packet, result, error, "set current control mode", dxl_id)
        time.sleep(0.2)

    def read_current_limit(self) -> np.ndarray:
        values = []
        for dxl_id in JOINT_IDS:
            value, result, error = self.packet.read2ByteTxRx(self.port, dxl_id, ADDR_CURRENT_LIMIT)
            comm_ok(self.packet, result, error, "read current limit", dxl_id)
            values.append(value)
        return np.asarray(values, dtype=int)

    def read_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        read and compute q_robot(for gravity compensation), dq_robot(for gravity compensation), raw_position, raw_current from sync read of all joints
        """
        last_result = None
        for attempt in range(self.read_retries + 1):
            result = self.reader.txRxPacket()
            if result == COMM_SUCCESS:
                break
            if result == COMM_PORT_BUSY:
                self.recover_port()
            last_result = result
            time.sleep(self.retry_delay)
        else:
            raise RuntimeError(f"sync read failed: {self.packet.getTxRxResult(last_result)}")

        if result != COMM_SUCCESS:
            raise RuntimeError(f"sync read failed: {self.packet.getTxRxResult(result)}")

        raw_current = []
        raw_velocity = []
        raw_position = []
        for dxl_id in JOINT_IDS:
            for address, length, label in (
                (ADDR_PRESENT_CURRENT, 2, "present current"),
                (ADDR_PRESENT_VELOCITY, 4, "present velocity"),
                (ADDR_PRESENT_POSITION, 4, "present position"),
            ):
                if not self.reader.isAvailable(dxl_id, address, length):
                    raise RuntimeError(f"ID {dxl_id}: {label} not available")
            raw_current.append(signed_from_u16(self.reader.getData(dxl_id, ADDR_PRESENT_CURRENT, 2)))
            raw_velocity.append(signed_from_u32(self.reader.getData(dxl_id, ADDR_PRESENT_VELOCITY, 4)))
            raw_position.append(signed_from_u32(self.reader.getData(dxl_id, ADDR_PRESENT_POSITION, 4)))

        raw_current_arr = np.asarray(raw_current, dtype=float)
        raw_velocity_arr = np.asarray(raw_velocity, dtype=float)
        raw_position_arr = np.asarray(raw_position, dtype=float)

        q_motor = raw_position_arr / RAW_PER_RAD
        q_robot = Q_R0 + SIGN * (q_motor - Q_M0)
        dq_robot = SIGN * raw_velocity_arr * VEL_RAW_TO_RAD_PER_SEC
        return q_robot, dq_robot, raw_position_arr, raw_current_arr

    def write_goal_current(self, raw_currents: np.ndarray) -> None:
        last_result = None
        for _ in range(2):
            self.writer.clearParam()
            for dxl_id, raw_current in zip(JOINT_IDS, raw_currents):
                if not self.writer.addParam(dxl_id, to_goal_current_param(int(raw_current))):
                    raise RuntimeError(f"ID {dxl_id}: failed to add goal current parameter")
            result = self.writer.txPacket()
            self.writer.clearParam()
            if result == COMM_SUCCESS:
                return
            if result == COMM_PORT_BUSY:
                self.recover_port()
            last_result = result
            time.sleep(self.retry_delay)
        raise RuntimeError(f"sync write goal current failed: {self.packet.getTxRxResult(last_result)}")

    def write_goal_current_individual(self, raw_currents: np.ndarray) -> list[str]:
        errors = []
        for dxl_id, raw_current in zip(JOINT_IDS, raw_currents):
            value = int(raw_current) & 0xFFFF
            ok = False
            last_message = ""
            for _ in range(3):
                self.recover_port()
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
        zero = np.zeros(7, dtype=int)
        warnings = []
        time.sleep(0.05)
        self.recover_port()
        try:
            self.write_goal_current(zero)
        except Exception as exc:
            warnings.append(f"sync zero-current write failed: {exc}")
            time.sleep(0.1)
            self.recover_port()
            warnings.extend(self.write_goal_current_individual(zero))

        for dxl_id in JOINT_IDS:
            try:
                self.recover_port()
                result, error = self.packet.write1ByteTxRx(self.port, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_DISABLE)
                comm_ok(self.packet, result, error, "disable torque", dxl_id)
            except Exception as exc:
                warnings.append(f"ID {dxl_id}: torque disable failed: {exc}")

        if warnings:
            print("Shutdown warnings:")
            for warning in warnings:
                print(f"  {warning}")
        else:
            print("Wrote zero current and disabled torque.")


def load_pinocchio_model(urdf_path: Path) -> tuple[pin.Model, pin.Data]:
    model = pin.buildModelFromUrdf(str(urdf_path))
    if model.nq != 7 or model.nv != 7:
        raise RuntimeError(f"Expected 7-DOF URDF, got nq={model.nq}, nv={model.nv}")
    return model, model.createData()


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


def require_vector(config: dict, key: str, length: int, dtype: type) -> np.ndarray:
    value = config[key]
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise ValueError(f"{key} must have length {length}, got shape {array.shape}")
    return array


def optional_vector(config: dict, key: str, length: int, dtype: type, default: list[float]) -> np.ndarray:
    if key not in config:
        return np.asarray(default, dtype=dtype)
    return require_vector(config, key, length, dtype)


def gravity_torque(model: pin.Model, data: pin.Data, q: np.ndarray, dq: np.ndarray, gain: float) -> np.ndarray:
    tau_g = pin.rnea(model, data, q, np.zeros_like(dq), np.zeros_like(dq))
    return gain * tau_g


def static_friction_torque(
    dq: np.ndarray,
    tau_g: np.ndarray,
    enable_speed: float,
    gain: float,
    joint_gain: np.ndarray,
    min_torque: np.ndarray,
    max_torque: np.ndarray,
    dither_flag: np.ndarray,
) -> np.ndarray:
    tau_ss = np.zeros_like(tau_g)
    low_speed = np.abs(dq) < enable_speed
    gain_abs_tau_g = gain * joint_gain * np.abs(tau_g)
    gain_abs_tau_g = np.maximum(gain_abs_tau_g, min_torque)
    limit_mask = max_torque > 0.0
    gain_abs_tau_g[limit_mask] = np.minimum(gain_abs_tau_g[limit_mask], max_torque[limit_mask])
    tau_ss[low_speed & dither_flag] = gain_abs_tau_g[low_speed & dither_flag]
    tau_ss[low_speed & ~dither_flag] = -gain_abs_tau_g[low_speed & ~dither_flag]
    dither_flag[low_speed] = ~dither_flag[low_speed]
    return tau_ss


def kinetic_friction_torque(
    dq: np.ndarray,
    coulomb: np.ndarray,
    viscous: np.ndarray,
    velocity_deadband: float,
) -> np.ndarray:
    dq_active = dq.copy()
    dq_active[np.abs(dq_active) < velocity_deadband] = 0.0
    return coulomb * np.sign(dq_active) + viscous * dq_active


def robot_torque_to_goal_current(tau_robot: np.ndarray, max_current_raw: int, torque_sign: float) -> np.ndarray:
    tau_motor = torque_sign * SIGN * tau_robot
    raw = tau_motor / (KT * GOAL_CURRENT_UNIT_AMP)
    return np.rint(np.clip(raw, -max_current_raw, max_current_raw)).astype(int)


def apply_current_deadband(raw_current: np.ndarray, deadband_raw: np.ndarray) -> np.ndarray:
    output = raw_current.copy()
    output[np.abs(output) < deadband_raw] = 0
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamixel leader arm gravity compensation.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help=f"YAML config path, default: {DEFAULT_CONFIG_PATH}")
    parser.add_argument("--urdf", default=None, help="Override leader arm URDF path from YAML")
    parser.add_argument("--port", default=None, help="Override Dynamixel serial port from YAML")
    parser.add_argument("--baudrate", type=int, default=None, help="Override Dynamixel baudrate from YAML")
    parser.add_argument("--frequency", type=float, default=None, help="Override control frequency from YAML")
    parser.add_argument("--gain", type=float, default=None, help="Override gravity compensation gain from YAML")
    parser.add_argument("--max-current-raw", type=int, default=None, help="Override Goal Current raw limit from YAML")
    parser.add_argument("--ramp-sec", type=float, default=None, help="Override ramp time from YAML")
    parser.add_argument("--duration", type=float, default=None, help="Override run duration from YAML. 0 means until Ctrl+C")
    parser.add_argument("--torque-sign", type=float, choices=(-1.0, 1.0), default=None, help="Override torque sign from YAML")
    gravity_group = parser.add_mutually_exclusive_group()
    gravity_group.add_argument("--enable-gravity-comp", action="store_true", default=None, help="Override YAML and enable gravity compensation")
    gravity_group.add_argument("--disable-gravity-comp", action="store_false", dest="enable_gravity_comp", help="Override YAML and disable gravity compensation")
    friction_group = parser.add_mutually_exclusive_group()
    friction_group.add_argument("--enable-friction-comp", action="store_true", default=None, help="Override YAML and enable friction compensation")
    friction_group.add_argument("--disable-friction-comp", action="store_false", dest="enable_friction_comp", help="Override YAML and disable friction compensation")
    parser.add_argument(
        "--joint-gain",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override per-joint gravity gain multiplier from YAML",
    )
    parser.add_argument(
        "--current-deadband-raw",
        type=int,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override per-joint Goal Current raw deadband from YAML",
    )
    parser.add_argument("--friction-enable-speed", type=float, default=None, help="Override friction low-speed threshold from YAML")
    parser.add_argument("--friction-gain", type=float, default=None, help="Override friction compensation gain from YAML")
    parser.add_argument(
        "--friction-joint-gain",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override per-joint friction gain multiplier from YAML",
    )
    parser.add_argument(
        "--static-min-torque",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override per-joint static friction minimum torque in Nm from YAML",
    )
    parser.add_argument(
        "--static-max-torque",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override per-joint static friction max torque in Nm from YAML. Use 0 for no limit.",
    )
    kinetic_group = parser.add_mutually_exclusive_group()
    kinetic_group.add_argument("--enable-kinetic-friction-comp", action="store_true", default=None, help="Override YAML and enable kinetic friction compensation")
    kinetic_group.add_argument("--disable-kinetic-friction-comp", action="store_false", dest="enable_kinetic_friction_comp", help="Override YAML and disable kinetic friction compensation")
    parser.add_argument(
        "--coulomb-friction",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override Coulomb friction torque coefficients in Nm from YAML",
    )
    parser.add_argument(
        "--viscous-friction",
        type=float,
        nargs=7,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"),
        help="Override viscous friction coefficients in Nm/(rad/s) from YAML",
    )
    parser.add_argument("--kinetic-velocity-deadband", type=float, default=None, help="Override kinetic friction velocity deadband from YAML")
    parser.add_argument("--log-interval", type=float, default=None, help="Override console log interval from YAML")
    parser.add_argument("--read-retries", type=int, default=None, help="Override sync-read retries from YAML")
    parser.add_argument("--retry-delay", type=float, default=None, help="Override comm retry delay from YAML")
    parser.add_argument("--execute", action="store_true", help="Enable current mode and stream currents")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml_config(config_path)
    dynamixel_config = config["dynamixel"]
    controller_config = config["controller"]
    gravity_config = controller_config["gravity_comp"]
    friction_config = controller_config["friction_comp"]
    static_friction_config = friction_config.get("static", friction_config)
    kinetic_friction_config = friction_config.get("kinetic", {})

    urdf = args.urdf if args.urdf is not None else config["urdf"]
    port_name = args.port if args.port is not None else dynamixel_config["port"]
    baudrate = args.baudrate if args.baudrate is not None else dynamixel_config["baudrate"]
    frequency = args.frequency if args.frequency is not None else controller_config["frequency"]
    ramp_sec = args.ramp_sec if args.ramp_sec is not None else controller_config["ramp_sec"]
    duration = args.duration if args.duration is not None else controller_config["duration"]
    max_current_raw = (
        args.max_current_raw if args.max_current_raw is not None else controller_config["max_current_raw"]
    )
    torque_sign = args.torque_sign if args.torque_sign is not None else controller_config["torque_sign"]
    log_interval = args.log_interval if args.log_interval is not None else controller_config["log_interval"]
    read_retries = args.read_retries if args.read_retries is not None else dynamixel_config["read_retries"]
    retry_delay = args.retry_delay if args.retry_delay is not None else dynamixel_config["retry_delay"]

    enable_gravity_comp = (
        args.enable_gravity_comp
        if args.enable_gravity_comp is not None
        else bool(gravity_config["enable"])
    )
    enable_friction_comp = (
        args.enable_friction_comp
        if args.enable_friction_comp is not None
        else bool(friction_config["enable"])
    )
    gravity_gain = args.gain if args.gain is not None else gravity_config["gain"]
    joint_gain = (
        np.asarray(args.joint_gain, dtype=float)
        if args.joint_gain is not None
        else require_vector(gravity_config, "joint_gain", 7, float)
    )
    current_deadband_raw = (
        np.asarray(args.current_deadband_raw, dtype=int)
        if args.current_deadband_raw is not None
        else require_vector(controller_config, "current_deadband_raw", 7, int)
    )
    enable_static_friction_comp = bool(static_friction_config.get("enable", True))
    enable_kinetic_friction_comp = (
        args.enable_kinetic_friction_comp
        if args.enable_kinetic_friction_comp is not None
        else bool(kinetic_friction_config.get("enable", False))
    )
    friction_enable_speed = (
        args.friction_enable_speed
        if args.friction_enable_speed is not None
        else static_friction_config["enable_speed"]
    )
    friction_gain = args.friction_gain if args.friction_gain is not None else static_friction_config["gain"]
    friction_joint_gain = (
        np.asarray(args.friction_joint_gain, dtype=float)
        if args.friction_joint_gain is not None
        else require_vector(static_friction_config, "joint_gain", 7, float)
    )
    static_min_torque = (
        np.asarray(args.static_min_torque, dtype=float)
        if args.static_min_torque is not None
        else optional_vector(static_friction_config, "min_torque", 7, float, [0.0] * 7)
    )
    static_max_torque = (
        np.asarray(args.static_max_torque, dtype=float)
        if args.static_max_torque is not None
        else optional_vector(static_friction_config, "max_torque", 7, float, [0.0] * 7)
    )
    coulomb_friction = (
        np.asarray(args.coulomb_friction, dtype=float)
        if args.coulomb_friction is not None
        else optional_vector(kinetic_friction_config, "coulomb", 7, float, [0.0] * 7)
    )
    viscous_friction = (
        np.asarray(args.viscous_friction, dtype=float)
        if args.viscous_friction is not None
        else optional_vector(kinetic_friction_config, "viscous", 7, float, [0.0] * 7)
    )
    kinetic_velocity_deadband = (
        args.kinetic_velocity_deadband
        if args.kinetic_velocity_deadband is not None
        else float(kinetic_friction_config.get("velocity_deadband", 0.0))
    )
    stiction_dither_flag = np.ones(7, dtype=bool)

    model, data = load_pinocchio_model(Path(urdf).resolve())
    controller = DynamixelCurrentController(port_name, baudrate, read_retries, retry_delay)

    dt = 1.0 / frequency
    start = time.perf_counter()
    next_log = start
    current_limits = controller.read_current_limit()

    print(f"Loaded config: {config_path}")
    print(f"Loaded URDF: {Path(urdf).resolve()}")
    print(f"Pinocchio model: nq={model.nq}, nv={model.nv}, joints={list(model.names)}")
    print(f"Current limits raw: {current_limits.tolist()}")
    print(f"Kt Nm/A: {KT.tolist()}, current unit A: {GOAL_CURRENT_UNIT_AMP}")
    print(
        f"gravity_enable={enable_gravity_comp}, gravity_gain={gravity_gain}, "
        f"friction_enable={enable_friction_comp}, static_enable={enable_static_friction_comp}, "
        f"static_gain={friction_gain}, static_enable_speed={friction_enable_speed}, "
        f"kinetic_enable={enable_kinetic_friction_comp}, kinetic_velocity_deadband={kinetic_velocity_deadband}"
    )
    print(
        f"max_current_raw={max_current_raw}, torque_sign={torque_sign}, "
        f"frequency={frequency}, execute={args.execute}"
    )
    print("joint_gain=" + np.array2string(joint_gain, precision=3, separator=","))
    print("friction_joint_gain=" + np.array2string(friction_joint_gain, precision=3, separator=","))
    print("static_min_torque=" + np.array2string(static_min_torque, precision=4, separator=","))
    print("static_max_torque=" + np.array2string(static_max_torque, precision=4, separator=","))
    print("coulomb_friction=" + np.array2string(coulomb_friction, precision=4, separator=","))
    print("viscous_friction=" + np.array2string(viscous_friction, precision=4, separator=","))
    print("current_deadband_raw=" + np.array2string(current_deadband_raw, separator=","))

    try:
        if args.execute:
            print("Switching motors to Current Control Mode and enabling torque.")
            controller.set_torque(False)
            controller.set_current_control_mode()
            controller.set_torque(True)
            time.sleep(0.2)

        while True:
            loop_start = time.perf_counter()
            elapsed = loop_start - start
            if duration > 0 and elapsed >= duration:
                break

            q, dq, raw_pos, present_current = controller.read_state()
            tau_g = joint_gain * gravity_torque(model, data, q, dq, gravity_gain)
            tau_f = np.zeros(7, dtype=float)
            tau_f_static = np.zeros(7, dtype=float)
            tau_f_kinetic = np.zeros(7, dtype=float)
            tau = np.zeros(7, dtype=float)
            if enable_gravity_comp:
                tau += tau_g
            if enable_friction_comp:
                if enable_static_friction_comp:
                    tau_f_static = static_friction_torque(
                        dq,
                        tau_g,
                        friction_enable_speed,
                        friction_gain,
                        friction_joint_gain,
                        static_min_torque,
                        static_max_torque,
                        stiction_dither_flag,
                    )
                if enable_kinetic_friction_comp:
                    tau_f_kinetic = kinetic_friction_torque(
                        dq,
                        coulomb_friction,
                        viscous_friction,
                        kinetic_velocity_deadband,
                    )
                tau_f = tau_f_static + tau_f_kinetic
                tau += tau_f

            goal_current_before_deadband = robot_torque_to_goal_current(tau, max_current_raw, torque_sign)
            goal_current = apply_current_deadband(goal_current_before_deadband, current_deadband_raw)

            if args.execute:
                ramp = min(1.0, elapsed / max(ramp_sec, 1e-6))
                controller.write_goal_current(np.rint(ramp * goal_current).astype(int))

            now = time.perf_counter()
            if now >= next_log:
                print(
                    "q="
                    + np.array2string(q, precision=3, separator=",")
                    + " dq="
                    + np.array2string(dq, precision=3, separator=",")
                    + " tau_g="
                    + np.array2string(tau_g, precision=3, separator=",")
                    + " tau_f="
                    + np.array2string(tau_f, precision=3, separator=",")
                    + " tau_fs="
                    + np.array2string(tau_f_static, precision=3, separator=",")
                    + " tau_fk="
                    + np.array2string(tau_f_kinetic, precision=3, separator=",")
                    + " goal_raw="
                    + np.array2string(goal_current, separator=",")
                    + " goal_raw_pre_deadband="
                    + np.array2string(goal_current_before_deadband, separator=",")
                )
                next_log = now + log_interval

            sleep_time = dt - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        if args.execute:
            try:
                controller.safe_shutdown()
            finally:
                controller.close()
        else:
            controller.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
