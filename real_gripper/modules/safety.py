from __future__ import annotations

"""Basic safety helpers for real teleoperation."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SafetyConfig:
    leader_state_max_age_s: float = 0.2
    follower_state_max_age_s: float = 0.5


class SafetyMonitor:
    def __init__(self, cfg: SafetyConfig):
        self.leader_state_max_age_s = float(cfg.leader_state_max_age_s)
        self.follower_state_max_age_s = float(cfg.follower_state_max_age_s)

    @staticmethod
    def validate_q(q: Sequence[float], name: str = "q") -> np.ndarray:
        vec = np.asarray(q, dtype=np.float64).reshape(-1)
        if vec.shape != (7,):
            raise ValueError(f"{name} must have shape (7,), got {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError(f"{name} contains NaN/Inf: {vec}")
        return vec

    def is_leader_fresh(self, age_s: float) -> bool:
        return age_s <= self.leader_state_max_age_s

    def is_follower_fresh(self, age_s: float) -> bool:
        return age_s <= self.follower_state_max_age_s
