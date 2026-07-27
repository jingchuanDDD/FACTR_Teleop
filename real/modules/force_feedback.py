from __future__ import annotations

"""Map FR3 external joint torque to safe Leader-side feedback torque."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _vector(config: dict, key: str, default: Sequence[float] | None = None) -> np.ndarray:
    value = config.get(key, default)
    if value is None:
        raise ValueError(f"force_feedback.{key} is required")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (7,):
        raise ValueError(f"force_feedback.{key} must have length 7, got shape {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"force_feedback.{key} contains NaN/Inf")
    return array


@dataclass(frozen=True)
class ForceFeedbackResult:
    tau_external: np.ndarray
    tau_raw: np.ndarray
    tau_applied: np.ndarray


class ForceFeedback:
    def __init__(self, config: dict, leader_max_torque: Sequence[float]):
        self.enable = bool(config.get("enable", False))
        self.joint_enable = _vector(config, "joint_enable", [1.0] * 7)
        self.joint_sign = _vector(config, "joint_sign", [1.0] * 7)
        self.joint_gain = _vector(config, "joint_gain", [1.0] * 7)
        self.follower_max_torque = _vector(config, "follower_max_torque")
        self.max_torque = _vector(config, "max_torque")
        self.damping = _vector(config, "damping", [0.0] * 7)
        self.scale = float(config.get("scale", 1.0))
        self.max_state_age_s = float(config.get("max_state_age_s", 0.1))

        leader_max_torque = np.asarray(leader_max_torque, dtype=np.float64)
        if leader_max_torque.shape != (7,) or not np.isfinite(leader_max_torque).all():
            raise ValueError("leader_max_torque must contain 7 finite values")
        if np.any(leader_max_torque <= 0.0):
            raise ValueError("leader_max_torque must be positive for every joint")
        if np.any(self.follower_max_torque <= 0.0):
            raise ValueError("force_feedback.follower_max_torque must be positive")
        if np.any(self.max_torque < 0.0):
            raise ValueError("force_feedback.max_torque must be non-negative")
        if np.any(self.joint_gain < 0.0):
            raise ValueError("force_feedback.joint_gain must be non-negative")
        if self.scale < 0.0 or not np.isfinite(self.scale):
            raise ValueError("force_feedback.scale must be finite and non-negative")
        if self.max_state_age_s <= 0.0 or not np.isfinite(self.max_state_age_s):
            raise ValueError("force_feedback.max_state_age_s must be finite and positive")

        self.leader_max_torque = leader_max_torque
        self.feedback_gain = self.scale * leader_max_torque / self.follower_max_torque

    def compute(self, tau_external: np.ndarray, dq_leader: np.ndarray) -> ForceFeedbackResult:
        tau_external = np.asarray(tau_external, dtype=np.float64)
        dq_leader = np.asarray(dq_leader, dtype=np.float64)
        if tau_external.shape != (7,) or not np.isfinite(tau_external).all():
            raise ValueError("tau_external must contain 7 finite values")
        if dq_leader.shape != (7,) or not np.isfinite(dq_leader).all():
            raise ValueError("dq_leader must contain 7 finite values")

        tau_raw = (
            self.joint_enable
            * self.joint_sign
            * self.joint_gain
            * self.feedback_gain
            * tau_external
            - self.damping * dq_leader
        )
        tau_applied = np.clip(tau_raw, -self.max_torque, self.max_torque)
        if not self.enable:
            tau_applied = np.zeros(7, dtype=np.float64)
        return ForceFeedbackResult(
            tau_external=tau_external.copy(),
            tau_raw=tau_raw,
            tau_applied=tau_applied,
        )
