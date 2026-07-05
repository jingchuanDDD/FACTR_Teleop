"""
Automated MuJoCo Franka tracking tests.

This script isolates follower tracking from the physical Dynamixel leader. It
uses the same actuator_pd controller as leader_to_mujoco_franka.py and reports
joint and end-effector tracking metrics for:

1. Single-DOF step changes.
2. Multi-joint moves that produce end-effector motion.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


FRANKA_JOINT_NAMES = (
    "panda0_joint1",
    "panda0_joint2",
    "panda0_joint3",
    "panda0_joint4",
    "panda0_joint5",
    "panda0_joint6",
    "panda0_joint7",
)

Q0 = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0])
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -1.6573, -2.8973])
Q_MAX = np.array([2.8973, 1.7628, 2.8973, -0.4, 2.8973, 2.1127, 2.8973])

ACTUATOR_KP = np.array([420.0, 420.0, 380.0, 380.0, 180.0, 180.0, 150.0])
ACTUATOR_FORCE = np.array([87.0, 87.0, 87.0, 87.0, 28.0, 28.0, 22.0])
ACTUATOR_DAMPING = np.array([85.0, 85.0, 75.0, 75.0, 18.0, 18.0, 12.0])


@dataclass
class TrackingMetrics:
    name: str
    duration: float
    settle_time: float | None
    final_joint_inf: float
    max_joint_inf: float
    final_ee_error: float | None
    max_ee_error: float | None
    passed: bool


def mujoco_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj_type, name)
    if idx < 0:
        raise RuntimeError(f"Missing {obj_type} '{name}'")
    return idx


def qpos_addrs(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_qposadr[mujoco_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in FRANKA_JOINT_NAMES
    ]


def dof_addrs(model: mujoco.MjModel) -> list[int]:
    return [
        int(model.jnt_dofadr[mujoco_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])
        for name in FRANKA_JOINT_NAMES
    ]


def actuator_ids(model: mujoco.MjModel) -> list[int]:
    return [mujoco_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in FRANKA_JOINT_NAMES]


def configure_actuator_pd(
    model: mujoco.MjModel,
    arm_actuators: list[int],
    arm_dofs: list[int],
    kp_scale: float,
    force_scale: float,
    damping_scale: float,
) -> None:
    kp = kp_scale * ACTUATOR_KP
    force = force_scale * ACTUATOR_FORCE
    model.dof_damping[arm_dofs] = damping_scale * ACTUATOR_DAMPING
    model.actuator_gainprm[arm_actuators, 0] = kp
    model.actuator_biasprm[arm_actuators, 1] = -kp
    model.actuator_forcerange[arm_actuators, 0] = -force
    model.actuator_forcerange[arm_actuators, 1] = force


def set_arm_qpos(model: mujoco.MjModel, data: mujoco.MjData, arm_qpos: list[int], q: np.ndarray) -> None:
    data.qpos[arm_qpos] = q
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    data.qfrc_applied[:] = 0.0
    mujoco.mj_forward(model, data)


def ee_position_for_q(
    model: mujoco.MjModel,
    scratch: mujoco.MjData,
    arm_qpos: list[int],
    site_id: int,
    q: np.ndarray,
) -> np.ndarray:
    scratch.qpos[:] = 0.0
    scratch.qvel[:] = 0.0
    scratch.qpos[arm_qpos] = q
    mujoco.mj_forward(model, scratch)
    return scratch.site_xpos[site_id].copy()


def step_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_actuators: list[int],
    arm_dofs: list[int],
    q_target: np.ndarray,
    bias_comp_scale: float,
) -> None:
    data.ctrl[arm_actuators] = q_target
    data.qfrc_applied[arm_dofs] = bias_comp_scale * data.qfrc_bias[arm_dofs]
    mujoco.mj_step(model, data)


def run_step_test(
    model: mujoco.MjModel,
    arm_qpos: list[int],
    arm_dofs: list[int],
    arm_actuators: list[int],
    site_id: int,
    name: str,
    q_start: np.ndarray,
    q_target: np.ndarray,
    max_time: float,
    joint_tol: float,
    ee_tol: float,
    stable_time: float,
    bias_comp_scale: float,
) -> TrackingMetrics:
    data = mujoco.MjData(model)
    scratch = mujoco.MjData(model)
    set_arm_qpos(model, data, arm_qpos, q_start)
    data.ctrl[arm_actuators] = q_start
    mujoco.mj_forward(model, data)

    target_ee = ee_position_for_q(model, scratch, arm_qpos, site_id, q_target)
    stable_steps_required = max(1, int(round(stable_time / model.opt.timestep)))
    stable_steps = 0
    settle_time = None
    max_joint_inf = 0.0
    max_ee_error = 0.0

    steps = int(round(max_time / model.opt.timestep))
    for step in range(steps):
        step_controller(model, data, arm_actuators, arm_dofs, q_target, bias_comp_scale)
        q = data.qpos[arm_qpos].copy()
        joint_inf = float(np.max(np.abs(q_target - q)))
        ee_error = float(np.linalg.norm(target_ee - data.site_xpos[site_id]))
        max_joint_inf = max(max_joint_inf, joint_inf)
        max_ee_error = max(max_ee_error, ee_error)

        if joint_inf <= joint_tol and ee_error <= ee_tol:
            stable_steps += 1
            if stable_steps >= stable_steps_required and settle_time is None:
                settle_time = (step + 1 - stable_steps_required) * model.opt.timestep
        else:
            stable_steps = 0

    final_q = data.qpos[arm_qpos].copy()
    final_joint_inf = float(np.max(np.abs(q_target - final_q)))
    final_ee_error = float(np.linalg.norm(target_ee - data.site_xpos[site_id]))
    passed = settle_time is not None and final_joint_inf <= joint_tol and final_ee_error <= ee_tol
    return TrackingMetrics(
        name=name,
        duration=max_time,
        settle_time=settle_time,
        final_joint_inf=final_joint_inf,
        max_joint_inf=max_joint_inf,
        final_ee_error=final_ee_error,
        max_ee_error=max_ee_error,
        passed=passed,
    )


def print_metrics(metrics: list[TrackingMetrics]) -> None:
    print("")
    print("name,status,settle_s,final_joint_rad,max_joint_rad,final_ee_m,max_ee_m")
    for item in metrics:
        settle = "NA" if item.settle_time is None else f"{item.settle_time:.3f}"
        print(
            f"{item.name},{'PASS' if item.passed else 'FAIL'},"
            f"{settle},{item.final_joint_inf:.5f},{item.max_joint_inf:.5f},"
            f"{item.final_ee_error:.5f},{item.max_ee_error:.5f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Automated Franka follower tracking tests.")
    parser.add_argument("--xml", default="franka_sim/franka_panda.xml", help="MuJoCo Franka XML path")
    parser.add_argument("--joint-amplitude", type=float, default=0.25, help="Single-DOF step amplitude in rad")
    parser.add_argument("--max-time", type=float, default=1.5, help="Seconds per step test")
    parser.add_argument("--joint-tol", type=float, default=0.02, help="Joint settling tolerance in rad")
    parser.add_argument("--ee-tol", type=float, default=0.015, help="End-effector settling tolerance in meters")
    parser.add_argument("--stable-time", type=float, default=0.08, help="Required stable time in seconds")
    parser.add_argument("--actuator-kp-scale", type=float, default=1.0)
    parser.add_argument("--actuator-force-scale", type=float, default=1.0)
    parser.add_argument("--actuator-damping-scale", type=float, default=1.0)
    parser.add_argument("--bias-comp-scale", type=float, default=1.0)
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(Path(args.xml).resolve()))
    arm_qpos = qpos_addrs(model)
    arm_dofs = dof_addrs(model)
    arm_actuators = actuator_ids(model)
    site_id = mujoco_id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")
    configure_actuator_pd(
        model,
        arm_actuators,
        arm_dofs,
        args.actuator_kp_scale,
        args.actuator_force_scale,
        args.actuator_damping_scale,
    )

    print(f"Loaded {Path(args.xml).resolve()}")
    print(f"timestep={model.opt.timestep:.4f}s")
    print(
        "controller="
        f"kp_scale={args.actuator_kp_scale}, "
        f"force_scale={args.actuator_force_scale}, "
        f"damping_scale={args.actuator_damping_scale}, "
        f"bias_comp_scale={args.bias_comp_scale}"
    )

    metrics: list[TrackingMetrics] = []
    for idx in range(7):
        q_target = Q0.copy()
        q_target[idx] += args.joint_amplitude
        q_target = np.clip(q_target, Q_MIN, Q_MAX)
        metrics.append(
            run_step_test(
                model,
                arm_qpos,
                arm_dofs,
                arm_actuators,
                site_id,
                f"joint_{idx + 1}_positive",
                Q0,
                q_target,
                args.max_time,
                args.joint_tol,
                args.ee_tol,
                args.stable_time,
                args.bias_comp_scale,
            )
        )

        q_target = Q0.copy()
        q_target[idx] -= args.joint_amplitude
        q_target = np.clip(q_target, Q_MIN, Q_MAX)
        metrics.append(
            run_step_test(
                model,
                arm_qpos,
                arm_dofs,
                arm_actuators,
                site_id,
                f"joint_{idx + 1}_negative",
                Q0,
                q_target,
                args.max_time,
                args.joint_tol,
                args.ee_tol,
                args.stable_time,
                args.bias_comp_scale,
            )
        )

    mixed_targets = [
        Q0 + np.array([0.18, -0.18, 0.16, -0.18, 0.10, 0.12, -0.16]),
        Q0 + np.array([-0.20, 0.16, -0.18, 0.20, -0.12, -0.10, 0.18]),
        Q0 + np.array([0.12, 0.20, 0.14, -0.24, -0.16, 0.16, 0.20]),
    ]
    for idx, target in enumerate(mixed_targets, start=1):
        metrics.append(
            run_step_test(
                model,
                arm_qpos,
                arm_dofs,
                arm_actuators,
                site_id,
                f"mixed_ee_move_{idx}",
                Q0,
                np.clip(target, Q_MIN, Q_MAX),
                args.max_time,
                args.joint_tol,
                args.ee_tol,
                args.stable_time,
                args.bias_comp_scale,
            )
        )

    print_metrics(metrics)
    failed = [item for item in metrics if not item.passed]
    print("")
    print(f"Passed {len(metrics) - len(failed)}/{len(metrics)} tests")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
