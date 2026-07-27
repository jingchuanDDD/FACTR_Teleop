#!/usr/bin/env python3
"""Teleoperate an FR3 end effector and gripper with the D1 exoskeleton.

The exoskeleton is mapped to the D1 six-joint arm and converted to an
end-effector pose with D1Kinematics. Whenever teleoperation is enabled, the
current exoskeleton pose and FR3 pose are captured as fresh anchors. Every
command sent to the FR3 is an absolute pose computed from those anchors.

This file is intentionally control-only. Sensor and robot-state recording is
implemented separately in 11_data_collection.py.

Keyboard controls:
    B: toggle teleoperation
    Space: toggle the gripper
    Ctrl+C: exit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import math
import select
import sys
import termios
import time
import traceback
import tty
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from beingbeyond_d1_edu_sdk.exo import ExoDriver, format_servo_ids
from beingbeyond_d1_edu_sdk.pin_kinematics import D1Kinematics, D1KinematicsConfig
from beingbeyond_d1_edu_sdk.urdf_path import get_default_urdf_path

from exo_glove_teleop_common import (
    build_arm_exo_cfg,
    calibrate_arm_exo_zero,
    transform_arm_exo_to_arm_deg,
    validate_common_cfg,
)
from real_world.fr3_ee_control_client import (
    FR3EEClient,
    make_T_from_pos_quat_xyzw,
    rotmat_to_quat_xyzw,
    safe_space_clip,
)
from real_world.fr3_gripper import GripperClient


COORDINATE_MAPPING_IMPLEMENTED = True


def map_exo_delta_to_fr3(
    T_exo_delta: np.ndarray,
    axis_map: np.ndarray,
    translation_scale: float,
) -> np.ndarray:
    """Map an exoskeleton-base-frame delta into the FR3 base frame."""
    T_exo_delta = np.asarray(T_exo_delta, dtype=np.float64)
    axis_map = np.asarray(axis_map, dtype=np.float64)
    if T_exo_delta.shape != (4, 4):
        raise ValueError(f"T_exo_delta must be 4x4, got {T_exo_delta.shape}")
    if axis_map.shape != (3, 3):
        raise ValueError(f"axis_map must be 3x3, got {axis_map.shape}")
    mapped = np.eye(4, dtype=np.float64)
    mapped[:3, :3] = axis_map @ T_exo_delta[:3, :3] @ axis_map.T
    mapped[:3, 3] = translation_scale * (axis_map @ T_exo_delta[:3, 3])
    return mapped


@dataclass
class FR3TeleopCfg:
    arm_exo_port: str = "/dev/ttyUSB0"
    arm_exo_baudrate: int = 115200
    arm_exo_servo_ids: List[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 5, 6]
    )
    arm_exo_read_hz: float = 20.0
    arm_exo_command_delay_s: float = 0.008
    release_arm_exo_torque_on_start: bool = True

    head_deg: List[float] = field(default_factory=lambda: [-15.0, -60.0])
    arm_init_deg: List[float] = field(
        default_factory=lambda: [0.0, -90.0, 90.0, 0.0, 0.0, 0.0]
    )
    arm_sign: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1])
    arm_limit_low_deg: List[float] = field(
        default_factory=lambda: [-150.0, -90.0, -90.0, -150.0, -100.0, -150.0]
    )
    arm_limit_high_deg: List[float] = field(
        default_factory=lambda: [150.0, 90.0, 90.0, 150.0, 85.0, 150.0]
    )
    arm_min_valid: int = 4

    zero_calibration_timeout_s: float = 10.0
    zero_calibration_poll_s: float = 0.05
    zero_calibration_stable_s: float = 3.0
    zero_calibration_max_delta_deg: float = 3.0

    fr3_host: str = "192.168.20.8"
    fr3_ee_port: int = 5556
    fr3_gripper_port: int = 5558
    fr3_timeout_s: float = 2.0
    fr3_state_poll_hz: float = 30.0
    fr3_state_max_age_s: float = 0.2
    gripper_min_width: float = 0.0002
    gripper_max_width: float = 0.08
    gripper_speed: float = 0.05
    gripper_force: float = 0.1

    # Calibration confirmed in Isaac Gym: exoskeleton base XYZ == FR3 base XYZ.
    exo_to_fr3_axis_map: List[float] = field(
        default_factory=lambda: [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]
    )
    translation_scale: float = 2.0

    loop_hz: float = 30.0
    max_linear_speed_m_s: float = 0.10
    max_angular_speed_rad_s: float = math.radians(20.0)
    safe_space_mode: str = "raise"
    enable_motion: bool = False
    dbg: bool = False
    dbg_interval_s: float = 1.0


def validate_cfg(cfg: FR3TeleopCfg) -> None:
    validate_common_cfg(cfg)
    for name, values, size in (
        ("head_deg", cfg.head_deg, 2),
        ("arm_limit_low_deg", cfg.arm_limit_low_deg, 6),
        ("arm_limit_high_deg", cfg.arm_limit_high_deg, 6),
        ("exo_to_fr3_axis_map", cfg.exo_to_fr3_axis_map, 9),
    ):
        if len(values) != size:
            raise ValueError(f"{name} must contain {size} values")
    if cfg.gripper_min_width >= cfg.gripper_max_width:
        raise ValueError("gripper_min_width must be smaller than gripper_max_width")
    if cfg.gripper_speed <= 0.0:
        raise ValueError("gripper_speed must be positive")
    if cfg.gripper_force < 0.0:
        raise ValueError("gripper_force must be non-negative")
    axis_map = np.asarray(cfg.exo_to_fr3_axis_map, dtype=np.float64).reshape(3, 3)
    if not np.allclose(axis_map @ axis_map.T, np.eye(3), atol=1e-6):
        raise ValueError("exo_to_fr3_axis_map must be orthonormal")
    if cfg.translation_scale <= 0.0:
        raise ValueError("translation_scale must be positive")
    if cfg.dbg_interval_s <= 0.0:
        raise ValueError("dbg_interval_s must be positive")


class KeyboardInput:
    """Non-blocking single-key input with reliable terminal restoration."""

    def __init__(self) -> None:
        self.fd: Optional[int] = None
        self.old_settings = None

    def open(self) -> None:
        if not sys.stdin.isatty():
            raise RuntimeError("keyboard control requires an interactive terminal (TTY)")
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)

    def poll(self) -> List[str]:
        keys: List[str] = []
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            keys.append(key)
        return keys

    def close(self) -> None:
        if self.fd is not None and self.old_settings is not None:
            termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        self.fd = None
        self.old_settings = None


def limit_pose_step(
    T_previous: np.ndarray,
    T_desired: np.ndarray,
    dt: float,
    max_linear_speed: float,
    max_angular_speed: float,
) -> np.ndarray:
    """Limit one outgoing absolute-pose step without changing its frame."""
    out = np.asarray(T_desired, dtype=np.float64).copy()
    dp = out[:3, 3] - T_previous[:3, 3]
    max_dp = max_linear_speed * dt
    dp_norm = float(np.linalg.norm(dp))
    if dp_norm > max_dp > 0.0:
        out[:3, 3] = T_previous[:3, 3] + dp * (max_dp / dp_norm)

    relative_rotation = Rotation.from_matrix(
        T_previous[:3, :3].T @ out[:3, :3]
    )
    rotvec = relative_rotation.as_rotvec()
    angle = float(np.linalg.norm(rotvec))
    max_angle = max_angular_speed * dt
    if angle > max_angle > 0.0:
        step_rotation = Rotation.from_rotvec(
            rotvec * (max_angle / angle)
        ).as_matrix()
        out[:3, :3] = T_previous[:3, :3] @ step_rotation
    return out


class ExoPoseSource:
    def __init__(self, cfg: FR3TeleopCfg) -> None:
        self.cfg = cfg
        self.exo = ExoDriver(build_arm_exo_cfg(cfg))
        self.zero_deg: Optional[List[float]] = None
        self.arm_rad = np.deg2rad(
            np.asarray(cfg.arm_init_deg, dtype=np.float64)
        )

        urdf = get_default_urdf_path()
        self.kin = D1Kinematics(D1KinematicsConfig(
            urdf_path=urdf,
            base_link="link0",
            ee_link="ee",
            camera_link="camera",
        ))
        self.q_head = np.deg2rad(np.asarray(cfg.head_deg, dtype=np.float64))

    def open(self) -> None:
        self.exo.open()
        if self.cfg.release_arm_exo_torque_on_start:
            self.exo.release_torque()
        self.zero_deg = calibrate_arm_exo_zero(self.exo, self.cfg)

    def close(self) -> None:
        self.exo.close()

    def read(self) -> Dict[str, Any]:
        if self.zero_deg is None:
            raise RuntimeError("exo source is not calibrated")
        frame = self.exo.read_frame()
        valid = [read.ok and read.angle_deg is not None for read in frame.reads]
        raw_deg: List[Optional[float]] = [
            float(read.angle_deg) if read.angle_deg is not None else None
            for read in frame.reads
        ]
        if sum(valid) >= self.cfg.arm_min_valid:
            filled_deg = [
                float(read.angle_deg) if read.angle_deg is not None else zero
                for read, zero in zip(frame.reads, self.zero_deg)
            ]
            mapped_deg = transform_arm_exo_to_arm_deg(
                filled_deg,
                self.zero_deg,
                self.cfg.arm_init_deg,
                self.cfg.arm_sign,
                self.cfg.arm_limit_low_deg,
                self.cfg.arm_limit_high_deg,
            )
            candidate = np.deg2rad(np.asarray(mapped_deg, dtype=np.float64))
            for index, ok in enumerate(valid):
                if ok:
                    self.arm_rad[index] = candidate[index]

        T_exo = np.asarray(
            self.kin.ee_in_base(self.q_head, self.arm_rad),
            dtype=np.float64,
        )
        return {
            "timestamp_monotonic": time.monotonic(),
            "exo_valid": sum(valid) >= self.cfg.arm_min_valid,
            "exo_joint_valid": valid,
            "exo_raw_joint_deg": raw_deg,
            "exo_mapped_joint_rad": self.arm_rad.copy(),
            "exo_ee_transform": T_exo,
            "exo_ee_position": T_exo[:3, 3].copy(),
            "exo_ee_quaternion_xyzw": rotmat_to_quat_xyzw(T_exo[:3, :3]),
        }


class FR3ExoTeleop:
    def __init__(self, cfg: FR3TeleopCfg) -> None:
        self.cfg = cfg
        self.source = ExoPoseSource(cfg)
        self.ee = FR3EEClient(cfg.fr3_host, cfg.fr3_ee_port, cfg.fr3_timeout_s)
        self.gripper = GripperClient(
            host=cfg.fr3_host,
            port=cfg.fr3_gripper_port,
            timeout_s=cfg.fr3_timeout_s,
        )
        self.keyboard = KeyboardInput()
        self.axis_map = np.asarray(
            cfg.exo_to_fr3_axis_map,
            dtype=np.float64,
        ).reshape(3, 3)

    def close(self) -> None:
        self.keyboard.close()
        for resource in (self.gripper, self.ee, self.source):
            try:
                resource.close()
            except Exception:
                pass

    def run(self) -> None:
        validate_cfg(self.cfg)
        print("=== D1 exo -> FR3 Cartesian teleop ===")
        if not COORDINATE_MAPPING_IMPLEMENTED:
            raise RuntimeError(
                "COORDINATE_MAPPING_IMPLEMENTED=True after calibration"
            )
        if not self.cfg.enable_motion:
            print(
                "[DRY RUN] FR3 commands are disabled; "
                "use --enable-motion to control hardware."
            )

        self.source.open()
        self.ee.connect()
        self.ee.start_state_poller(
            self.cfg.fr3_state_poll_hz,
            self.cfg.fr3_state_max_age_s,
        )
        if not self.ee.wait_until_state_ready(self.cfg.fr3_timeout_s):
            raise RuntimeError("timed out waiting for the first FR3 state")
        self.gripper.connect()
        print("[Info] Gripper ping:", self.gripper.ping())
        self.keyboard.open()

        print("[Info] Arm exo ids:", format_servo_ids(self.cfg.arm_exo_servo_ids))
        print("[Info] Axis map:\n", self.axis_map)
        print(f"[Info] Translation scale: {self.cfg.translation_scale}")
        print("[Keys] B: teleop ON/OFF | Space: gripper | Ctrl+C: exit")
        print("[Teleop] OFF")

        dt = 1.0 / self.cfg.loop_hz
        teleop_active = False
        gripper_closed = False
        T_exo_anchor: Optional[np.ndarray] = None
        T_fr3_anchor: Optional[np.ndarray] = None
        T_last_command: Optional[np.ndarray] = None
        next_dbg_time = time.monotonic()

        try:
            while True:
                started = time.perf_counter()
                sample = self.source.read()
                T_exo = sample["exo_ee_transform"]

                for key in self.keyboard.poll():
                    key_lower = key.lower()
                    if key_lower == "b":
                        if teleop_active:
                            teleop_active = False
                            print("[Teleop] OFF")
                        else:
                            fr3_pos, fr3_quat, state_age = (
                                self.ee.get_ee_pose_cached(
                                    max_age_s=self.cfg.fr3_state_max_age_s,
                                    fallback_to_sync=True,
                                )
                            )
                            T_exo_anchor = T_exo.copy()
                            T_fr3_anchor = make_T_from_pos_quat_xyzw(
                                fr3_pos,
                                fr3_quat,
                            )
                            T_last_command = T_fr3_anchor.copy()
                            teleop_active = True
                            print(
                                "[Teleop] ON: anchors captured, "
                                f"FR3 state age={state_age:.3f}s"
                            )
                    elif key == " ":
                        requested_closed = not gripper_closed
                        requested_width = (
                            self.cfg.gripper_min_width
                            if requested_closed
                            else self.cfg.gripper_max_width
                        )
                        try:
                            if self.cfg.enable_motion:
                                self.gripper.send_gripper(
                                    width=requested_width,
                                    speed=self.cfg.gripper_speed,
                                    force=self.cfg.gripper_force,
                                )
                            gripper_closed = requested_closed
                        except Exception as exc:
                            print(f"[Warning] gripper command failed: {exc}")
                            continue
                        state = "CLOSED" if gripper_closed else "OPEN"
                        suffix = "" if self.cfg.enable_motion else " (dry run)"
                        print(
                            f"[Gripper] {state}: "
                            f"width={requested_width:.4f}m{suffix}"
                        )

                if not teleop_active:
                    sleep_s = dt - (time.perf_counter() - started)
                    if sleep_s > 0.0:
                        time.sleep(sleep_s)
                    continue

                assert T_exo_anchor is not None
                assert T_fr3_anchor is not None
                assert T_last_command is not None

                T_exo_delta = np.eye(4, dtype=np.float64)
                T_exo_delta[:3, :3] = (
                    T_exo[:3, :3] @ T_exo_anchor[:3, :3].T
                )
                T_exo_delta[:3, 3] = T_exo[:3, 3] - T_exo_anchor[:3, 3]
                T_fr3_delta = map_exo_delta_to_fr3(
                    T_exo_delta,
                    self.axis_map,
                    self.cfg.translation_scale,
                )
                T_target = np.eye(4, dtype=np.float64)
                T_target[:3, :3] = (
                    T_fr3_delta[:3, :3] @ T_fr3_anchor[:3, :3]
                )
                T_target[:3, 3] = (
                    T_fr3_anchor[:3, 3] + T_fr3_delta[:3, 3]
                )
                T_target[:3, 3] = safe_space_clip(
                    T_target[:3, 3],
                    mode=self.cfg.safe_space_mode,
                    name="fr3_target",
                )
                T_target = limit_pose_step(
                    T_last_command,
                    T_target,
                    dt,
                    self.cfg.max_linear_speed_m_s,
                    self.cfg.max_angular_speed_rad_s,
                )

                if self.cfg.enable_motion:
                    self.ee.send_ee_pose(
                        T_target[:3, 3].tolist(),
                        rotmat_to_quat_xyzw(T_target[:3, :3]).tolist(),
                    )
                T_last_command = T_target

                now = time.monotonic()
                if self.cfg.dbg and now >= next_dbg_time:
                    print(
                        "[DBG] exo_dp_base=", np.round(T_exo_delta[:3, 3], 4),
                        "fr3_dp_base=", np.round(T_fr3_delta[:3, 3], 4),
                        "target_pos=", np.round(T_target[:3, 3], 4),
                        "gripper=", "closed" if gripper_closed else "open",
                    )
                    next_dbg_time = now + self.cfg.dbg_interval_s

                sleep_s = dt - (time.perf_counter() - started)
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if self.cfg.enable_motion:
                try:
                    self.ee.stop()
                except Exception as exc:
                    print(f"[Warning] failed to stop FR3 controller: {exc}")
            self.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fr3-host", default="192.168.20.8")
    parser.add_argument("--ee-port", type=int, default=5556)
    parser.add_argument("--gripper-port", type=int, default=5558)
    parser.add_argument("--exo-port", default="/dev/ttyUSB0")
    parser.add_argument("--gripper-speed", type=float, default=0.05)
    parser.add_argument("--gripper-force", type=float, default=0.1)
    parser.add_argument("--enable-motion", action="store_true")
    parser.add_argument(
        "--axis-map",
        nargs=9,
        type=float,
        default=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        metavar=("R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"),
        help="Orthonormal exoskeleton-base to FR3-base axis mapping, row-major.",
    )
    parser.add_argument(
        "--translation-scale",
        type=float,
        default=1.0,
        help="FR3 translation / exoskeleton translation (default: 1.0).",
    )
    parser.add_argument("--dbg", action="store_true")
    parser.add_argument("--dbg-interval", type=float, default=1.0, metavar="SECONDS")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = FR3TeleopCfg(
        fr3_host=args.fr3_host,
        fr3_ee_port=args.ee_port,
        fr3_gripper_port=args.gripper_port,
        arm_exo_port=args.exo_port,
        gripper_speed=args.gripper_speed,
        gripper_force=args.gripper_force,
        exo_to_fr3_axis_map=list(args.axis_map),
        translation_scale=args.translation_scale,
        enable_motion=args.enable_motion,
        dbg=args.dbg,
        dbg_interval_s=args.dbg_interval,
    )
    teleop = FR3ExoTeleop(cfg)
    try:
        teleop.run()
    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        teleop.close()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
