from __future__ import annotations

"""Real-robot teleoperation entry point for FR3."""

import argparse
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

try:
    from real.modules.control_loop import TeleopLoopConfig, run_teleop_loop
    from real.modules.mapping import JointMapper, JointMapperConfig
    from real.modules.safety import SafetyConfig, SafetyMonitor
except ModuleNotFoundError as exc:
    if exc.name != "real":
        raise
    from modules.control_loop import TeleopLoopConfig, run_teleop_loop
    from modules.mapping import JointMapper, JointMapperConfig
    from modules.safety import SafetyConfig, SafetyMonitor


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs").joinpath("real_fr3.yaml")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return cfg


def _vec(config: Dict[str, Any], key: str, size: int, default=None):
    value = config.get(key, default)
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{key} must contain {size} values, got {arr.shape}")
    return arr.tolist()


def build_mapper(cfg: Dict[str, Any]) -> JointMapper:
    return JointMapper(
        JointMapperConfig(
            q_m0_deg=cfg["q_m0_deg"],
            q_r0=cfg["q_r0"],
            sign=cfg["sign"],
            q_min=_vec(cfg, "q_min", 7),
            q_max=_vec(cfg, "q_max", 7),
        )
    )


def build_safety(cfg: Dict[str, Any]) -> SafetyMonitor:
    return SafetyMonitor(
        SafetyConfig(
            leader_state_max_age_s=float(cfg.get("leader_state_max_age_s", 0.2)),
            follower_state_max_age_s=float(cfg.get("follower_state_max_age_s", 0.5)),
        )
    )


def resolve_relative_path(path_value: str, config_dir: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)

    real_dir = Path(__file__).resolve().parent
    repo_root = real_dir.parent
    candidates = (
        config_dir / path,
        config_dir.parent / path,
        repo_root / path,
        Path.cwd() / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((repo_root / path).resolve())


def build_leader_compensation(cfg: Dict[str, Any] | None, config_dir: Path, max_current_raw: int):
    if not cfg or not bool(cfg.get("enable", False)):
        return None
    cfg = dict(cfg)
    if "urdf" in cfg:
        cfg["urdf"] = resolve_relative_path(str(cfg["urdf"]), config_dir)
    try:
        from real.modules.leader_compensation import LeaderCompensation
    except ModuleNotFoundError as exc:
        if exc.name != "real":
            raise
        from modules.leader_compensation import LeaderCompensation

    return LeaderCompensation(cfg, max_current_raw)


def build_force_feedback(cfg: Dict[str, Any] | None, compensation):
    if not cfg:
        return None
    if compensation is None:
        return None
    try:
        from real.modules.force_feedback import ForceFeedback
    except ModuleNotFoundError as exc:
        if exc.name != "real":
            raise
        from modules.force_feedback import ForceFeedback

    return ForceFeedback(cfg, compensation.max_torque_nm)


def import_leader_adapter():
    try:
        from real.adapters.leader_dynamixel import LeaderDynamixelReader
    except ModuleNotFoundError as exc:
        if exc.name != "real":
            raise
        from adapters.leader_dynamixel import LeaderDynamixelReader
    return LeaderDynamixelReader


def import_fr3_adapter():
    try:
        from real.adapters.fr3_backend import FR3Backend
    except ModuleNotFoundError as exc:
        if exc.name != "real":
            raise
        from adapters.fr3_backend import FR3Backend
    return FR3Backend


def run_configured_leader_home(leader, cfg: Dict[str, Any]) -> None:
    print("[leader_home] moving leader arm to motor zero.")
    leader.home_to_raw_position(
        cfg["target_raw"],
        profile_velocity=int(cfg.get("profile_velocity", 20)),
        profile_acceleration=int(cfg.get("profile_acceleration", 5)),
        position_tolerance_ticks=int(cfg.get("position_tolerance_ticks", 15)),
        velocity_tolerance_raw=int(cfg.get("velocity_tolerance_raw", 2)),
        settle_time_s=float(cfg.get("settle_time_s", 0.3)),
        timeout_s=float(cfg.get("timeout_s", 15.0)),
        max_delta_ticks=int(cfg.get("max_delta_ticks", 1200)),
        poll_hz=float(cfg.get("poll_hz", 20.0)),
        verbose=bool(cfg.get("verbose", True)),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Real FR3 teleoperation entry point.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument(
        "--mode",
        choices=("status", "home", "leader-home", "teleop"),
        default="teleop",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml_config(config_path)

    leader_cfg = cfg["leader"]
    follower_cfg = cfg["follower"]
    mapping_cfg = cfg["mapping"]
    safety_cfg = cfg["safety"]
    teleop_cfg = cfg["teleop"]
    leader_comp_cfg = cfg.get("leader_compensation", {"enable": False})
    force_feedback_cfg = cfg.get("force_feedback", {"enable": False})
    home_cfg = cfg.get("home", {})
    leader_home_cfg = cfg.get("leader_home", {"enable": False})

    mapper = build_mapper(mapping_cfg)
    safety = build_safety(safety_cfg)
    teleop_enable = bool(
        teleop_cfg.get(
            "enable",
            not bool(teleop_cfg.get("dry_run", False)),
        )
    )
    teleop_loop_cfg = TeleopLoopConfig(
        leader_hz=float(teleop_cfg.get("leader_hz", 20.0)),
        command_hz=float(teleop_cfg.get("command_hz", 100.0)),
        enable=teleop_enable,
        verbose=bool(teleop_cfg.get("verbose", True)),
        print_interval_s=float(teleop_cfg.get("print_interval_s", 1.0)),
    )
    LeaderDynamixelReader = import_leader_adapter()
    if args.mode == "leader-home":
        if not bool(leader_home_cfg.get("enable", False)):
            print("[leader_home] disabled in config")
            return 0
        with LeaderDynamixelReader(
            port_name=leader_cfg["port"],
            baudrate=int(leader_cfg["baudrate"]),
            read_retries=int(leader_cfg.get("read_retries", 3)),
            retry_delay=float(leader_cfg.get("retry_delay", 0.01)),
            max_current_raw=int(leader_cfg["max_current_raw"]),
        ) as leader:
            run_configured_leader_home(leader, leader_home_cfg)
        return 0

    # construct leader and franka client
    FR3Backend = import_fr3_adapter()
    with LeaderDynamixelReader(
        port_name=leader_cfg["port"],
        baudrate=int(leader_cfg["baudrate"]),
        read_retries=int(leader_cfg.get("read_retries", 3)),
        retry_delay=float(leader_cfg.get("retry_delay", 0.01)),
        max_current_raw=int(leader_cfg["max_current_raw"]),
    ) as leader, FR3Backend(
        host=follower_cfg["host"],
        port=int(follower_cfg["port"]),
        timeout_s=float(follower_cfg.get("timeout_s", 2.0)),
        retry_times=int(follower_cfg.get("retry_times", 1)),
    ) as franka:
        print("[franka] ping ->", franka.ping())
        franka.start_state_poller(
            poll_hz=float(follower_cfg.get("poll_hz", 50.0)),
            warn_stale_after_s=float(safety.follower_state_max_age_s),
        )
        ready = franka.client.wait_until_state_ready(timeout_s=2.0)
        if not ready:
            raise RuntimeError("FR3 state poller not ready")

        state, age = franka.get_joint_state_cached(
            max_age_s=safety.follower_state_max_age_s,
            fallback_to_sync=True,
        )
        print("[franka] current q =", np.array2string(state, precision=5), "age_s=", f"{age:.4f}")

        if args.mode == "status":
            print("[status] leader and franka communication are ready.")
            return 0

        if args.mode == "home":
            if not bool(home_cfg.get("enable", True)):
                print("[home] disabled in config")
                return 0
            home_q = np.asarray(home_cfg["q"], dtype=np.float64).reshape(7)
            print("[home] move to =", np.array2string(home_q, precision=5))
            franka.move_to_q(
                home_q,
                duration_s=float(home_cfg.get("duration_s", 3.0)),
                hz=float(home_cfg.get("hz", 100.0)),
                current_q_max_age_s=float(safety.follower_state_max_age_s),
                fallback_to_sync=True,
                verbose=True,
            )
            return 0

        if bool(leader_home_cfg.get("enable", False)):
            run_configured_leader_home(leader, leader_home_cfg)

        leader_max_current_raw = int(leader_cfg["max_current_raw"])
        compensation = build_leader_compensation(leader_comp_cfg, config_path.parent, leader_max_current_raw)
        force_feedback = build_force_feedback(force_feedback_cfg, compensation)
        run_teleop_loop(
            leader,
            franka,
            mapper,
            safety,
            teleop_loop_cfg,
            compensation,
            force_feedback,
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
