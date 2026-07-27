from __future__ import annotations

"""Measure how long one Dynamixel leader read_state() call takes."""

import argparse
from pathlib import Path
import statistics
import time
from typing import Any, Dict

import numpy as np
import yaml

try:
    from real.adapters.leader_dynamixel import LeaderDynamixelReader
except ModuleNotFoundError as exc:
    if exc.name != "real":
        raise
    from adapters.leader_dynamixel import LeaderDynamixelReader


DEFAULT_CONFIG_PATH = Path(__file__).with_name("configs").joinpath("real_fr3.yaml")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return cfg


def percentile(values: list[float], p: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * p))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure leader read_state() latency.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="YAML config path")
    parser.add_argument("--samples", type=int, default=100, help="Number of timed reads")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup reads excluded from statistics")
    parser.add_argument("--sleep-s", type=float, default=0.0, help="Optional sleep between timed reads")
    args = parser.parse_args()

    if args.samples <= 0:
        raise ValueError("--samples must be positive")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative")
    if args.sleep_s < 0.0:
        raise ValueError("--sleep-s must be non-negative")

    cfg = load_yaml_config(Path(args.config).resolve())
    leader_cfg = cfg["leader"]

    durations_ms: list[float] = []
    last_q = None
    with LeaderDynamixelReader(
        port_name=leader_cfg["port"],
        baudrate=int(leader_cfg["baudrate"]),
        read_retries=int(leader_cfg.get("read_retries", 3)),
        retry_delay=float(leader_cfg.get("retry_delay", 0.01)),
    ) as leader:
        for _ in range(args.warmup):
            leader.read_state()

        for _ in range(args.samples):
            t0 = time.perf_counter()
            state = leader.read_state()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            durations_ms.append(elapsed_ms)
            last_q = state.q_motor
            if args.sleep_s > 0.0:
                time.sleep(args.sleep_s)

    mean_ms = statistics.fmean(durations_ms)
    median_ms = statistics.median(durations_ms)
    min_ms = min(durations_ms)
    max_ms = max(durations_ms)
    p95_ms = percentile(durations_ms, 0.95)
    p99_ms = percentile(durations_ms, 0.99)
    hz_from_mean = 1000.0 / mean_ms if mean_ms > 0.0 else float("inf")

    print("")
    print("leader read_state() latency")
    print(f"samples: {args.samples}, warmup: {args.warmup}, sleep_s: {args.sleep_s}")
    print(f"mean_ms:   {mean_ms:.3f}  (~{hz_from_mean:.1f} Hz)")
    print(f"median_ms: {median_ms:.3f}")
    print(f"p95_ms:    {p95_ms:.3f}")
    print(f"p99_ms:    {p99_ms:.3f}")
    print(f"min_ms:    {min_ms:.3f}")
    print(f"max_ms:    {max_ms:.3f}")
    if last_q is not None:
        print("last q_motor [rad]:", np.array2string(last_q, precision=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
