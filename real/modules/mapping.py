from __future__ import annotations

"""Leader-to-Franka joint mapping helpers."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np


def _as_vec(values: Sequence[float], size: int, name: str) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    if vec.shape != (size,):
        raise ValueError(f"{name} must contain {size} values, got {vec.shape}")
    if not np.isfinite(vec).all():
        raise ValueError(f"{name} contains NaN/Inf: {vec}")
    return vec


@dataclass
class JointMapperConfig:
    q_m0_deg: Sequence[float]
    q_r0: Sequence[float]
    sign: Sequence[float]
    q_min: Optional[Sequence[float]] = None
    q_max: Optional[Sequence[float]] = None


class JointMapper:
    def __init__(self, cfg: JointMapperConfig):
        self.q_m0 = np.deg2rad(_as_vec(cfg.q_m0_deg, 7, "q_m0_deg"))
        self.q_r0 = _as_vec(cfg.q_r0, 7, "q_r0")
        self.sign = _as_vec(cfg.sign, 7, "sign")
        if np.any(self.sign == 0.0):
            raise ValueError("sign entries must be non-zero")
        self.q_min = None if cfg.q_min is None else _as_vec(cfg.q_min, 7, "q_min")
        self.q_max = None if cfg.q_max is None else _as_vec(cfg.q_max, 7, "q_max")
        if self.q_min is not None and self.q_max is not None and np.any(self.q_min > self.q_max):
            raise ValueError("q_min must not exceed q_max")

    def map_motor_q_to_franka_q(self, q_motor: Sequence[float]) -> np.ndarray:
        q_motor_vec = _as_vec(q_motor, 7, "q_motor")
        q_target = self.q_r0 + self.sign * (q_motor_vec - self.q_m0) 
        if self.q_min is not None and self.q_max is not None:
            q_target = np.clip(q_target, self.q_min, self.q_max)
        return q_target

    def map_raw_position_to_franka_q(self, raw_position: Sequence[float]) -> np.ndarray:
        raw_vec = _as_vec(raw_position, 7, "raw_position")
        q_motor = raw_vec / (2048.0 / np.pi)
        return self.map_motor_q_to_franka_q(q_motor)
