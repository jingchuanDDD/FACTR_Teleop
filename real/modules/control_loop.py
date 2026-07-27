from __future__ import annotations

"""Main teleoperation loop for the real FR3 setup."""

from dataclasses import dataclass
import threading
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    try:
        from real.adapters.fr3_backend import FR3Backend
        from real.adapters.leader_dynamixel import LeaderDynamixelReader
        from real.modules.force_feedback import ForceFeedback
        from real.modules.leader_compensation import LeaderCompensation
        from real.modules.mapping import JointMapper
        from real.modules.safety import SafetyMonitor
    except ModuleNotFoundError as exc:
        if exc.name != "real":
            raise
        from adapters.fr3_backend import FR3Backend
        from adapters.leader_dynamixel import LeaderDynamixelReader
        from modules.force_feedback import ForceFeedback
        from modules.leader_compensation import LeaderCompensation
        from modules.mapping import JointMapper
        from modules.safety import SafetyMonitor


@dataclass
class TeleopLoopConfig:
    leader_hz: float = 20.0
    command_hz: float = 100.0
    enable: bool = True
    verbose: bool = True
    print_interval_s: float = 1.0


def _smooth_quintic(t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)


def run_teleop_loop(
    leader: "LeaderDynamixelReader",
    franka: "FR3Backend",
    mapper: "JointMapper",
    safety: "SafetyMonitor",
    cfg: TeleopLoopConfig,
    compensation: Optional["LeaderCompensation"] = None,
    force_feedback: Optional["ForceFeedback"] = None,
) -> None:
    leader_hz = float(cfg.leader_hz)
    command_hz = float(cfg.command_hz)
    if leader_hz <= 0.0:
        raise ValueError(f"leader_hz must be positive, got {leader_hz}")
    if command_hz <= 0.0:
        raise ValueError(f"command_hz must be positive, got {command_hz}")
    if command_hz < leader_hz:
        raise ValueError(
            f"command_hz must be >= leader_hz, got command_hz={command_hz}, leader_hz={leader_hz}"
        )
    command_dt = 1.0 / command_hz
    leader_dt = 1.0 / leader_hz
    interp_steps = command_hz / leader_hz

    last_print_t = 0.0
    stop_event = threading.Event()
    target_lock = threading.Lock()
    latest_target: Optional[np.ndarray] = None
    latest_target_t = 0.0
    latest_error: Optional[BaseException] = None
    latest_tau_external = np.zeros(7, dtype=np.float64)
    latest_tau_feedback_raw = np.zeros(7, dtype=np.float64)
    latest_tau_feedback = np.zeros(7, dtype=np.float64)
    latest_tau_gravity = np.zeros(7, dtype=np.float64)
    latest_tau_friction = np.zeros(7, dtype=np.float64)
    latest_feedback_age_s = 0.0
    latest_feedback_error: Optional[str] = None
    last_feedback_warn_t = 0.0
    compensation_configured = compensation is not None and compensation.enable
    feedback_enabled = force_feedback is not None and force_feedback.enable
    if feedback_enabled and not compensation_configured:
        raise ValueError("force feedback requires leader_compensation.enable=true")
    leader_output_active = compensation_configured
    leader_output_start_t = time.monotonic()
    current = None
    def leader_read_loop() -> None:
        nonlocal latest_target, latest_target_t, latest_error
        nonlocal latest_tau_external, latest_tau_feedback_raw, latest_tau_feedback
        nonlocal latest_tau_gravity, latest_tau_friction
        nonlocal latest_feedback_age_s, latest_feedback_error, last_feedback_warn_t
        nonlocal current
        next_read_t = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now < next_read_t:
                time.sleep(min(next_read_t - now, 0.001))
                continue
            next_read_t += leader_dt
            try:
                leader_state = leader.read_state()
                tau_external = np.zeros(7, dtype=np.float64)
                tau_feedback_raw = np.zeros(7, dtype=np.float64)
                tau_feedback = np.zeros(7, dtype=np.float64)
                tau_gravity = np.zeros(7, dtype=np.float64)
                tau_friction = np.zeros(7, dtype=np.float64)
                feedback_age_s = 0.0
                feedback_error = None

                if feedback_enabled:
                    try:
                        tau_external, feedback_age_s = franka.get_external_joint_torque_cached(
                            max_age_s=force_feedback.max_state_age_s,
                            fallback_to_sync=False,
                        )
                        _, dq_leader = compensation.motor_state_to_comp_state(
                            leader_state.q_motor,
                            leader_state.dq_motor,
                        )
                        feedback_result = force_feedback.compute(tau_external, dq_leader)
                        tau_feedback_raw = feedback_result.tau_raw
                        tau_feedback = feedback_result.tau_applied
                    except Exception as exc:
                        feedback_error = str(exc)
                        warn_t = time.monotonic()
                        if (warn_t - last_feedback_warn_t) >= 1.0:
                            print(f"[force_feedback] disabled for this cycle: {exc}")
                            last_feedback_warn_t = warn_t

                if compensation_configured:
                    comp_result = compensation.compute(
                        leader_state.q_motor,
                        leader_state.dq_motor,
                        tau_feedback=tau_feedback,
                    )
                    if compensation.enable_gravity:
                        tau_gravity = comp_result.tau_g
                    tau_friction = comp_result.tau_static + comp_result.tau_kinetic
                if leader_output_active:
                    ramp = min(
                        1.0,
                        (time.monotonic() - leader_output_start_t) / max(compensation.ramp_sec, 1e-6),
                    )
                    leader.write_goal_current(np.rint(ramp * comp_result.goal_current).astype(int))
                q_target = mapper.map_motor_q_to_franka_q(leader_state.q_motor)
                q_target = safety.validate_q(q_target, name="q_target")
                with target_lock:
                    latest_target = q_target
                    latest_target_t = time.monotonic()
                    latest_tau_external = tau_external
                    latest_tau_feedback_raw = tau_feedback_raw
                    latest_tau_feedback = tau_feedback
                    latest_tau_gravity = tau_gravity
                    latest_tau_friction = tau_friction
                    latest_feedback_age_s = feedback_age_s
                    latest_feedback_error = feedback_error
                    current = comp_result.goal_current if compensation_configured else None
            except BaseException as exc:
                latest_error = exc
                stop_event.set()

    try:
        if leader_output_active:
            print("Switching leader motors to Current Control Mode and enabling torque.")
            leader.set_current_control_mode()
            leader.write_goal_current(np.zeros(len(leader.joint_ids), dtype=int))
            leader.set_torque(True)
            time.sleep(0.2)
            leader_output_start_t = time.monotonic()
        else:
            leader.set_torque(False)

        leader_thread = threading.Thread(target=leader_read_loop, daemon=True)
        leader_thread.start()

        q_cmd, _ = franka.get_joint_state_cached(
            max_age_s=safety.follower_state_max_age_s,
            fallback_to_sync=True,
        )
        q_cmd = safety.validate_q(q_cmd, name="initial_franka_q")
        q_start = q_cmd.copy()
        q_goal = q_cmd.copy()
        segment_start_t = time.monotonic()
        consumed_target_t = 0.0
        next_command_t = segment_start_t

        while not stop_event.is_set():
            if latest_error is not None:
                raise RuntimeError("leader read loop failed") from latest_error

            now = time.monotonic()
            if now < next_command_t:
                time.sleep(min(next_command_t - now, 0.001))
                continue
            next_command_t += command_dt

            with target_lock:
                new_target = None if latest_target is None else latest_target.copy()
                new_target_t = latest_target_t
                tau_external_log = latest_tau_external.copy()
                tau_feedback_raw_log = latest_tau_feedback_raw.copy()
                tau_feedback_log = latest_tau_feedback.copy()
                tau_gravity_log = latest_tau_gravity.copy()
                tau_friction_log = latest_tau_friction.copy()
                feedback_age_s_log = latest_feedback_age_s
                feedback_error_log = latest_feedback_error

            if new_target is not None and new_target_t > consumed_target_t:
                q_start = q_cmd.copy()
                q_goal = new_target
                segment_start_t = now
                consumed_target_t = new_target_t

            alpha = (now - segment_start_t) / leader_dt
            s = _smooth_quintic(alpha)
            q_cmd = q_start + (q_goal - q_start) * s

            if cfg.enable:
                franka.send_q(q_cmd.tolist())

            if cfg.verbose and (time.monotonic() - last_print_t) >= cfg.print_interval_s:
                try:
                    q_actual, q_actual_age_s = franka.get_joint_state_cached(
                        max_age_s=safety.follower_state_max_age_s,
                        fallback_to_sync=False,
                    )
                    print("电流",current)
                    # print(
                    #     "[teleop] q_actual =",
                    #     np.array2string(q_actual, precision=5),
                    #     f"age_s={q_actual_age_s:.4f}",
                    #     "q_goal =",
                    #     np.array2string(q_goal, precision=5),
                    #     "q_cmd =",
                    #     np.array2string(q_cmd, precision=5),
                    # )
                except Exception as exc:
                    print(f"[teleop] unable to read current Franka q: {exc}")
                # if compensation_configured:
                #     print(
                #         "[leader_torque] tau_gravity =",
                #         np.array2string(tau_gravity_log, precision=4),
                #         "tau_friction =",
                #         np.array2string(tau_friction_log, precision=4),
                #         "tau_feedback =",
                #         np.array2string(tau_feedback_log, precision=4),
                #         "(Nm)",
                #     )
                if feedback_enabled:
                    if feedback_error_log is None:
                        print(
                            "[force_feedback] tau_external =",
                            np.array2string(tau_external_log, precision=4),
                            f"age_s={feedback_age_s_log:.4f}",
                            "tau_raw =",
                            np.array2string(tau_feedback_raw_log, precision=4),
                            "tau_applied =",
                            np.array2string(tau_feedback_log, precision=4),
                        )
                    else:
                        print(f"[force_feedback] tau_applied = 0, reason: {feedback_error_log}")
                last_print_t = time.monotonic()
    finally:
        stop_event.set()
        if "leader_thread" in locals():
            leader_thread.join(timeout=1.0)
        if leader_output_active:
            leader.safe_shutdown()
