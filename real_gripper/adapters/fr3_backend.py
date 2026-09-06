from __future__ import annotations

"""Thin wrapper around the FR3 joint control client."""

from typing import Optional, Sequence, Tuple

import numpy as np

try:
    from real.fr3_joint_control_client import FR3JointClient
except ModuleNotFoundError as exc:
    if exc.name != "real":
        raise
    from fr3_joint_control_client import FR3JointClient


class FR3Backend:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 2.0,
        retry_times: int = 1,
    ):
        self.client = FR3JointClient(
            host,
            port,
            timeout_s=timeout_s,
            retry_times=retry_times,
        )

    def __enter__(self):
        self.client.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.client.__exit__(exc_type, exc, tb)

    def connect(self) -> "FR3Backend":
        self.client.connect()
        return self

    def close(self) -> None:
        self.client.close()

    def ping(self) -> dict:
        return self.client.ping()

    def get_state(self) -> dict:
        return self.client.get_state()

    def get_joint_state_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[np.ndarray, float]:
        return self.client.get_joint_state_cached(
            max_age_s=max_age_s,
            fallback_to_sync=fallback_to_sync,
        )

    def get_external_joint_torque_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[np.ndarray, float]:
        state, age_s = self.client.get_state_cached(
            max_age_s=max_age_s,
            fallback_to_sync=fallback_to_sync,
        )
        tau_external = state.get("motor_torques_external")
        if tau_external is None:
            raise RuntimeError("cached FR3 state has no motor_torques_external")

        tau_external = np.asarray(tau_external, dtype=np.float64).reshape(-1)
        if tau_external.shape != (7,):
            raise RuntimeError(
                "cached motor_torques_external must have shape (7,), "
                f"got {tau_external.shape}"
            )
        if not np.isfinite(tau_external).all():
            raise RuntimeError("cached motor_torques_external contains NaN/Inf")
        return tau_external, age_s

    def start_state_poller(self, poll_hz: float, warn_stale_after_s: float) -> None:
        self.client.start_state_poller(
            poll_hz=poll_hz,
            warn_stale_after_s=warn_stale_after_s,
        )

    def stop_state_poller(self) -> None:
        self.client.stop_state_poller()

    def send_q(self, q: Sequence[float]) -> None:
        self.client.send_q(list(q))

    def move_to_q(
        self,
        target_q: Sequence[float],
        duration_s: float,
        hz: float,
        max_joint_vel_rad_s: Optional[Sequence[float]] = None,
        current_q_max_age_s: float = 0.5,
        fallback_to_sync: bool = True,
        verbose: bool = True,
    ) -> float:
        return self.client.move_to_q(
            target_q,
            duration_s=duration_s,
            hz=hz,
            max_joint_vel_rad_s=max_joint_vel_rad_s,
            current_q_max_age_s=current_q_max_age_s,
            fallback_to_sync=fallback_to_sync,
            verbose=verbose,
        )

    def stop(self) -> None:
        self.client.stop()
