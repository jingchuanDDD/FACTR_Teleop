from __future__ import annotations

"""Leader-side gravity, friction, force-feedback, and soft-limit compensation."""

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
from typing import Optional, Sequence

import numpy as np
import pinocchio as pin


SERVO_TYPES = (
    "XM430-W350",
    "XM540-W270",
    "XM430-W350",
    "XM540-W270",
    "XM430-W350",
    "XM430-W350",
    "XM430-W350",
)
KT_NM_PER_AMP = {
    "XM430-W350": 1.783,
    "XM540-W270": 2.409,
}
GOAL_CURRENT_UNIT_AMP = 0.00269
KT = np.asarray([KT_NM_PER_AMP[servo] for servo in SERVO_TYPES], dtype=np.float64)


def require_vector(config: dict, key: str, length: int, dtype: type) -> np.ndarray:
    value = config[key]
    array = np.asarray(value, dtype=dtype)
    if array.shape != (length,):
        raise ValueError(f"{key} must have length {length}, got shape {array.shape}")
    return array


def optional_vector(config: dict, key: str, length: int, dtype: type, default: Sequence[float]) -> np.ndarray:
    if key not in config:
        return np.asarray(default, dtype=dtype)
    return require_vector(config, key, length, dtype)


def read_urdf_revolute_limits(urdf_path: Path, joint_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(urdf_path).getroot()
    limits: dict[str, tuple[float, float]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if name not in joint_names:
            continue
        limit = joint.find("limit")
        if limit is None or "lower" not in limit.attrib or "upper" not in limit.attrib:
            raise ValueError(f"URDF joint {name} is missing lower/upper limits")
        limits[name] = (float(limit.attrib["lower"]), float(limit.attrib["upper"]))
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise ValueError(f"URDF is missing joint limits for: {missing}")
    q_min = np.asarray([limits[name][0] for name in joint_names], dtype=np.float64)
    q_max = np.asarray([limits[name][1] for name in joint_names], dtype=np.float64)
    return q_min, q_max


def follower_limits_to_leader_coords(
    follower_min: np.ndarray,
    follower_max: np.ndarray,
    q_r0: np.ndarray,
    sign: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = sign * (follower_min - q_r0)
    b = sign * (follower_max - q_r0)
    return np.minimum(a, b), np.maximum(a, b)


def load_pinocchio_model(urdf_path: Path) -> tuple[pin.Model, pin.Data]:
    model = pin.buildModelFromUrdf(str(urdf_path))
    if model.nq != 7 or model.nv != 7:
        raise RuntimeError(f"Expected 7-DOF URDF, got nq={model.nq}, nv={model.nv}")
    return model, model.createData()


def gravity_torque(model: pin.Model, data: pin.Data, q: np.ndarray, dq: np.ndarray, gain: float) -> np.ndarray:
    tau_g = pin.rnea(model, data, q, dq, np.zeros_like(dq))
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


def joint_limit_barrier_torque(
    tau: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    margin: np.ndarray,
    kp: np.ndarray,
    damping: np.ndarray,
    max_torque: np.ndarray,
) -> np.ndarray:
    barrier = np.zeros_like(tau)
    for idx in range(len(tau)):
        if margin[idx] <= 0.0:
            continue
        lower = q_min[idx] + margin[idx]
        upper = q_max[idx] - margin[idx]
        if lower >= upper:
            raise ValueError(f"Joint {idx + 1} soft-limit margin leaves no valid interval")
        if q[idx] > upper:
            barrier[idx] = -kp[idx] * (q[idx] - upper) - damping[idx] * dq[idx]
        elif q[idx] < lower:
            barrier[idx] = -kp[idx] * (q[idx] - lower) - damping[idx] * dq[idx]

    limit_mask = max_torque > 0.0
    barrier[limit_mask] = np.clip(barrier[limit_mask], -max_torque[limit_mask], max_torque[limit_mask])
    return tau + barrier


def joint_limit_status(
    q: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    margin: np.ndarray,
) -> np.ndarray:
    status = np.zeros_like(q, dtype=int)
    lower = q_min + margin
    upper = q_max - margin
    status[q < lower] = -1
    status[q > upper] = 1
    return status


def apply_current_deadband(raw_current: np.ndarray, deadband_raw: np.ndarray) -> np.ndarray:
    output = raw_current.copy()
    output[np.abs(output) < deadband_raw] = 0
    return output


@dataclass
class LeaderCompensationResult:
    q: np.ndarray
    dq: np.ndarray
    tau: np.ndarray
    tau_g: np.ndarray
    tau_static: np.ndarray
    tau_kinetic: np.ndarray
    tau_feedback: np.ndarray
    tau_limit: np.ndarray
    limit_status: np.ndarray
    goal_current: np.ndarray


class LeaderCompensation:
    def __init__(self, config: dict, max_current_raw: int):
        self.enable = bool(config.get("enable", False))
        self.q_m0 = np.deg2rad(require_vector(config, "q_m0_deg", 7, float))
        self.q_r0 = require_vector(config, "q_r0", 7, float)
        self.sign = require_vector(config, "sign", 7, float)
        if np.any(self.sign == 0.0):
            raise ValueError("leader_compensation.sign entries must be non-zero")

        controller_config = config["controller"]
        gravity_config = controller_config["gravity_comp"]
        friction_config = controller_config["friction_comp"]
        static_config = friction_config.get("static", friction_config)
        kinetic_config = friction_config.get("kinetic", {})

        self.enable_gravity = bool(gravity_config["enable"])
        self.enable_friction = bool(friction_config["enable"])
        self.enable_static_friction = bool(static_config.get("enable", True))
        self.enable_kinetic_friction = bool(kinetic_config.get("enable", False))

        self.gravity_gain = float(gravity_config["gain"])
        self.joint_gain = require_vector(gravity_config, "joint_gain", 7, float)
        self.current_deadband_raw = require_vector(controller_config, "current_deadband_raw", 7, int)
        self.ramp_sec = float(controller_config["ramp_sec"])
        self.torque_sign = float(controller_config["torque_sign"])
        self.max_torque_nm = max_current_raw * GOAL_CURRENT_UNIT_AMP * KT

        limit_config = controller_config.get("soft_limits", {})
        self.soft_limit_enable = bool(limit_config.get("enable", False))
        self.soft_limit_min, self.soft_limit_max = self._soft_limit_bounds(config, limit_config)
        if "margin_deg" in limit_config:
            self.soft_limit_margin = np.full(7, np.deg2rad(float(limit_config["margin_deg"])), dtype=np.float64)
        else:
            self.soft_limit_margin = optional_vector(limit_config, "margin", 7, float, [0.0] * 7)
        self.soft_limit_kp = optional_vector(limit_config, "kp", 7, float, [0.0] * 7)
        self.soft_limit_damping = optional_vector(limit_config, "damping", 7, float, [0.0] * 7)
        self.soft_limit_max_torque = optional_vector(limit_config, "max_torque", 7, float, [0.0] * 7)

        self.static_enable_speed = float(static_config["enable_speed"])
        self.static_gain = float(static_config["gain"])
        self.static_joint_gain = require_vector(static_config, "joint_gain", 7, float)
        self.static_min_torque = optional_vector(static_config, "min_torque", 7, float, [0.0] * 7)
        self.static_max_torque = optional_vector(static_config, "max_torque", 7, float, [0.0] * 7)

        self.coulomb_friction = optional_vector(kinetic_config, "coulomb", 7, float, [0.0] * 7)
        self.viscous_friction = optional_vector(kinetic_config, "viscous", 7, float, [0.0] * 7)
        self.kinetic_velocity_deadband = float(kinetic_config.get("velocity_deadband", 0.0))
        self.dither_flag = np.ones(7, dtype=bool)

        self.model, self.data = load_pinocchio_model(Path(config["urdf"]).resolve())

    def _soft_limit_bounds(self, config: dict, limit_config: dict) -> tuple[np.ndarray, np.ndarray]:
        configured_min = optional_vector(limit_config, "min", 7, float, [-np.inf] * 7)
        configured_max = optional_vector(limit_config, "max", 7, float, [np.inf] * 7)
        source = str(limit_config.get("source", "configured"))
        if source == "configured":
            return configured_min, configured_max
        if source != "leader_follower_intersection":
            raise ValueError("soft_limits.source must be configured or leader_follower_intersection")

        leader_min, leader_max = read_urdf_revolute_limits(
            Path(config["urdf"]).resolve(),
            tuple(f"joint_{idx}" for idx in range(1, 8)),
        )
        follower_min = require_vector(limit_config, "follower_min", 7, float)
        follower_max = require_vector(limit_config, "follower_max", 7, float)
        q_r0 = require_vector(limit_config, "teleop_q_r0", 7, float)
        sign = require_vector(limit_config, "teleop_sign", 7, float)
        follower_min_leader, follower_max_leader = follower_limits_to_leader_coords(
            follower_min,
            follower_max,
            q_r0,
            sign,
        )
        effective_min = np.maximum(leader_min, follower_min_leader)
        effective_max = np.minimum(leader_max, follower_max_leader)
        if np.any(effective_min >= effective_max):
            raise ValueError(
                "Leader/follower joint-limit intersection is empty: "
                f"min={effective_min.tolist()}, max={effective_max.tolist()}"
            )
        return effective_min, effective_max

    def motor_state_to_comp_state(self, q_motor: np.ndarray, dq_motor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_r0 + self.sign * (q_motor - self.q_m0)
        dq = self.sign * dq_motor
        return q, dq

    def compute(
        self,
        q_motor: np.ndarray,
        dq_motor: np.ndarray,
        tau_feedback: Optional[np.ndarray] = None,
    ) -> LeaderCompensationResult:
        q, dq = self.motor_state_to_comp_state(q_motor, dq_motor)
        tau_g = self.joint_gain * gravity_torque(self.model, self.data, q, dq, self.gravity_gain)
        tau_static = np.zeros(7, dtype=np.float64)
        tau_kinetic = np.zeros(7, dtype=np.float64)
        tau_feedback_safe = np.zeros(7, dtype=np.float64)
        tau_limit = np.zeros(7, dtype=np.float64)
        limit_status = np.zeros(7, dtype=int)
        tau = np.zeros(7, dtype=np.float64)

        if self.enable_gravity:
            tau += tau_g
        if self.enable_friction:
            if self.enable_static_friction:
                tau_static = static_friction_torque(
                    dq,
                    tau_g,
                    self.static_enable_speed,
                    self.static_gain,
                    self.static_joint_gain,
                    self.static_min_torque,
                    self.static_max_torque,
                    self.dither_flag,
                )
            if self.enable_kinetic_friction:
                tau_kinetic = kinetic_friction_torque(
                    dq,
                    self.coulomb_friction,
                    self.viscous_friction,
                    self.kinetic_velocity_deadband,
                )
            tau += tau_static + tau_kinetic

        if tau_feedback is not None:
            tau_feedback_safe = np.asarray(tau_feedback, dtype=np.float64)
            if tau_feedback_safe.shape != (7,):
                raise ValueError(f"tau_feedback must have shape (7,), got {tau_feedback_safe.shape}")
            if not np.isfinite(tau_feedback_safe).all():
                raise ValueError("tau_feedback contains NaN/Inf")
            tau += tau_feedback_safe

        if self.soft_limit_enable:
            tau_before_limit = tau.copy()
            tau = joint_limit_barrier_torque(
                tau_before_limit,
                q,
                dq,
                self.soft_limit_min,
                self.soft_limit_max,
                self.soft_limit_margin,
                self.soft_limit_kp,
                self.soft_limit_damping,
                self.soft_limit_max_torque,
            )
            tau_limit = tau - tau_before_limit
            limit_status = joint_limit_status(q, self.soft_limit_min, self.soft_limit_max, self.soft_limit_margin)

        tau_motor = self.torque_sign * self.sign * tau
        raw = tau_motor / (KT * GOAL_CURRENT_UNIT_AMP)
        goal_pre_deadband = np.rint(raw).astype(int)
        goal_current = apply_current_deadband(goal_pre_deadband, self.current_deadband_raw)
        return LeaderCompensationResult(
            q=q,
            dq=dq,
            tau=tau,
            tau_g=tau_g,
            tau_static=tau_static,
            tau_kinetic=tau_kinetic,
            tau_feedback=tau_feedback_safe,
            tau_limit=tau_limit,
            limit_status=limit_status,
            goal_current=goal_current,
        )
