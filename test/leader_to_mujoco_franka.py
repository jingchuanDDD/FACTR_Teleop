"""
Use the physical 7-DOF Dynamixel leader arm to teleoperate a MuJoCo Franka arm.

This is a Windows/no-ROS entry point. It reads Dynamixel IDs 21-27 on COM21,
maps motor positions to robot joint angles, and sends those angles to the first
seven MuJoCo qpos entries in franka_sim/franka_panda.xml.

直接设置 qpos：
python test\leader_to_mujoco_franka.py
"""
#TODO:讨论2号关节到限位附近很难遥操，6号关节摩擦补偿后非常松
from __future__ import annotations

import argparse
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.sax.saxutils import escape

import mujoco
import numpy as np
import yaml

from leader_gravity_comp import (
    DynamixelCurrentController,
    GOAL_CURRENT_UNIT_AMP,
    KT,
    apply_current_deadband,
    gravity_torque,
    kinetic_friction_torque,
    load_pinocchio_model,
    optional_vector,
    require_vector,
    robot_torque_to_goal_current,
    static_friction_torque,
)


try:
    import mujoco.viewer
except Exception:  # Viewer is optional for headless smoke tests.
    mujoco = mujoco


JOINT_IDS = (21, 22, 23, 24, 25, 26, 27)
DEFAULT_CONFIG_PATH = Path(__file__).with_name("leader_teleop_config.yaml")

FRANKA_JOINT_NAMES = (
    "panda0_joint1",
    "panda0_joint2",
    "panda0_joint3",
    "panda0_joint4",
    "panda0_joint5",
    "panda0_joint6",
    "panda0_joint7",
)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

RAW_PER_RAD = 2048.0 / np.pi
DXL_POS_RAW_PER_REV = 4096.0


def raw_to_franka_joints(raw: np.ndarray, q_m0: np.ndarray, q_r0: np.ndarray, sign: np.ndarray) -> np.ndarray:
    raw_near_zero = normalize_raw_position_near_zero(raw, q_m0 * RAW_PER_RAD)
    q_motor = raw_near_zero / RAW_PER_RAD
    return q_r0 + sign * (q_motor - q_m0)


def normalize_raw_position_near_zero(raw: np.ndarray, raw_zero: np.ndarray) -> np.ndarray:
    return raw_zero + ((raw - raw_zero + DXL_POS_RAW_PER_REV / 2.0) % DXL_POS_RAW_PER_REV - DXL_POS_RAW_PER_REV / 2.0)


def load_yaml_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {path}")
    return config


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
    q_min = np.asarray([limits[name][0] for name in joint_names], dtype=float)
    q_max = np.asarray([limits[name][1] for name in joint_names], dtype=float)
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


def format_limit_status(status: np.ndarray) -> str:
    entries = []
    for idx, value in enumerate(status):
        if value < 0:
            entries.append(f"J{idx + 1}:low")
        elif value > 0:
            entries.append(f"J{idx + 1}:high")
    return "none" if not entries else ",".join(entries)


class LeaderCompensation:
    def __init__(self, config: dict, config_label: str):
        self.config_label = config_label
        self.dynamixel_config = config["dynamixel"]
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
        self.max_current_raw = int(controller_config["max_current_raw"])
        self.current_deadband_raw = require_vector(controller_config, "current_deadband_raw", 7, int)
        self.ramp_sec = float(controller_config["ramp_sec"])
        self.torque_sign = float(controller_config["torque_sign"])
        self.max_torque_nm = self.max_current_raw * GOAL_CURRENT_UNIT_AMP * KT
        limit_config = controller_config.get("soft_limits", {})
        self.soft_limit_enable = bool(limit_config.get("enable", False))
        self.soft_limit_min, self.soft_limit_max = self._soft_limit_bounds(config, limit_config)
        if "margin_deg" in limit_config:
            self.soft_limit_margin = np.full(7, np.deg2rad(float(limit_config["margin_deg"])), dtype=float)
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

    def compute(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_feedback: np.ndarray | None = None,
        feedback_limit: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        tau_g = self.joint_gain * gravity_torque(self.model, self.data, q, dq, self.gravity_gain)
        tau_static = np.zeros(7, dtype=float)
        tau_kinetic = np.zeros(7, dtype=float)
        tau_feedback_safe = np.zeros(7, dtype=float)
        tau_limit = np.zeros(7, dtype=float)
        limit_status = np.zeros(7, dtype=int)
        tau = np.zeros(7, dtype=float)

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
            tau_feedback_safe = np.asarray(tau_feedback, dtype=float)
            if feedback_limit is not None:
                tau_feedback_safe = np.clip(tau_feedback_safe, -feedback_limit, feedback_limit)
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

        tau = np.clip(tau, -self.max_torque_nm, self.max_torque_nm)
        goal_pre_deadband = robot_torque_to_goal_current(tau, self.max_current_raw, self.torque_sign)
        goal_current = apply_current_deadband(goal_pre_deadband, self.current_deadband_raw)
        return tau, tau_g, tau_static, tau_kinetic, tau_feedback_safe, tau_limit, limit_status, goal_current


def joint_qpos_addr(model, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if idx < 0:
        raise RuntimeError(f"Missing joint '{name}' in MuJoCo model")
    return int(model.jnt_qposadr[idx])


def joint_dof_addr(model, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if idx < 0:
        raise RuntimeError(f"Missing joint '{name}' in MuJoCo model")
    return int(model.jnt_dofadr[idx])


def set_franka_qpos(model, data, q: np.ndarray) -> None:
    for joint_name, value in zip(FRANKA_JOINT_NAMES, q):
        data.qpos[joint_qpos_addr(model, joint_name)] = value
    mujoco.mj_forward(model, data)


def follower_external_joint_torque(data, dof_addrs: list[int]) -> np.ndarray:
    return data.qfrc_constraint[dof_addrs].copy()


def wall_contact_stats(model, data) -> tuple[int, float]:
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "teleop_collision_wall")
    if wall_id < 0:
        return 0, 0.0
    contact_count = 0
    min_contact_dist = float("inf")
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if int(contact.geom1) != wall_id and int(contact.geom2) != wall_id:
            continue
        contact_count += 1
        min_contact_dist = min(min_contact_dist, float(contact.dist))
    if contact_count == 0:
        return 0, 0.0
    return contact_count, min_contact_dist


def clamp_qpos_to_wall(
    model,
    data,
    q_safe: np.ndarray,
    q_target: np.ndarray,
    min_contact_dist: float,
    iterations: int,
) -> tuple[np.ndarray, bool, int, float]:
    set_franka_qpos(model, data, q_target)
    contact_count, contact_dist = wall_contact_stats(model, data)
    if contact_count == 0 or contact_dist >= min_contact_dist:
        return q_target, False, contact_count, contact_dist

    low = q_safe.copy()
    high = q_target.copy()
    best = q_safe.copy()
    best_count = 0
    best_dist = 0.0
    for _ in range(max(iterations, 1)):
        mid = 0.5 * (low + high)
        set_franka_qpos(model, data, mid)
        mid_count, mid_dist = wall_contact_stats(model, data)
        if mid_count == 0 or mid_dist >= min_contact_dist:
            best = mid
            best_count = mid_count
            best_dist = mid_dist
            low = mid
        else:
            high = mid

    set_franka_qpos(model, data, best)
    return best, True, best_count, best_dist


def collision_wall_contact_torque(
    model,
    data,
    site_idx: int,
    dof_addrs: list[int],
    axis: str,
    position: float,
    side: str,
    torque_scale: float,
    max_normal_force: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, int, float]:
    wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "teleop_collision_wall")
    if wall_id < 0:
        return np.zeros(7), np.zeros(3), data.site_xpos[site_idx].copy(), 0.0, 0

    normal_force = 0.0
    contact_count = 0
    min_contact_dist = float("inf")
    contact_force = np.zeros(6, dtype=float)
    for contact_idx in range(data.ncon):
        contact = data.contact[contact_idx]
        if int(contact.geom1) != wall_id and int(contact.geom2) != wall_id:
            continue
        min_contact_dist = min(min_contact_dist, float(contact.dist))
        mujoco.mj_contactForce(model, data, contact_idx, contact_force)
        normal_force += abs(float(contact_force[0]))
        contact_count += 1
    if max_normal_force > 0.0:
        normal_force = min(normal_force, max_normal_force)

    axis_idx = AXIS_INDEX[axis]
    ee_pos = data.site_xpos[site_idx].copy()
    signed_distance = ee_pos[axis_idx] - position
    penetration = signed_distance if side == "positive" else -signed_distance

    force = np.zeros(3, dtype=float)
    if contact_count > 0:
        normal_sign = -1.0 if side == "positive" else 1.0
        force[axis_idx] = normal_sign * normal_force

    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_idx)
    tau = torque_scale * (jacp[:, dof_addrs].T @ force)
    if contact_count == 0:
        min_contact_dist = 0.0
    return tau, force, ee_pos, penetration, contact_count, min_contact_dist


def collision_wall_xml_path(xml_path: Path, wall_config: dict) -> Path:
    axis = wall_config["axis"]
    axis_idx = AXIS_INDEX[axis]
    center = np.asarray(wall_config["center"], dtype=float)
    center[axis_idx] = float(wall_config["position"])
    half_size = float(wall_config["display_size"])
    half_thickness = 0.5 * float(wall_config["thickness"])
    size = np.array([half_size, half_size, half_size], dtype=float)
    size[axis_idx] = half_thickness
    group = int(wall_config.get("group", 0))
    rgba = str(wall_config.get("rgba", "1 0.15 0.05 0.55"))
    margin = float(wall_config.get("margin", 0.0))

    geom = (
        f'        <geom name="teleop_collision_wall" type="box" '
        f'pos="{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}" '
        f'size="{size[0]:.6f} {size[1]:.6f} {size[2]:.6f}" '
        f'rgba="{escape(rgba)}" contype="1" conaffinity="1" group="{group}" '
        f'margin="{margin:.6f}" '
        f'friction="{escape(str(wall_config["friction"]))}" '
        f'solref="{escape(str(wall_config["solref"]))}" '
        f'solimp="{escape(str(wall_config["solimp"]))}"/>\n'
    )
    source = xml_path.read_text(encoding="utf-8")
    if 'name="teleop_collision_wall"' in source:
        generated = source
    else:
        generated = source.replace("    </worldbody>", geom + "    </worldbody>", 1)
    generated_path = xml_path.with_name(xml_path.stem + "_teleop_wall.xml")
    generated_path.write_text(generated, encoding="utf-8")
    return generated_path


def site_id(model, name: str) -> int:
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if idx < 0:
        raise RuntimeError(f"Missing site '{name}' in MuJoCo model")
    return idx


def virtual_wall_joint_torque(
    model,
    data,
    site_idx: int,
    dof_addrs: list[int],
    axis: str,
    position: float,
    stiffness: float,
    damping: float,
    side: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    axis_idx = AXIS_INDEX[axis]
    ee_pos = data.site_xpos[site_idx].copy()
    ee_vel = np.zeros(3, dtype=float)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)
    mujoco.mj_jacSite(model, data, jacp, jacr, site_idx)
    ee_vel = jacp @ data.qvel

    signed_distance = ee_pos[axis_idx] - position
    penetration = signed_distance if side == "positive" else -signed_distance
    force = np.zeros(3, dtype=float)
    if penetration > 0.0:
        normal_sign = -1.0 if side == "positive" else 1.0
        force[axis_idx] = normal_sign * (stiffness * penetration + damping * ee_vel[axis_idx])

    tau = jacp[:, dof_addrs].T @ force
    return tau, force, ee_pos, penetration


def add_virtual_wall_marker(viewer, axis: str, position: float, size: float) -> None:
    if viewer is None:
        return
    scene = viewer.user_scn
    if scene.ngeom >= scene.maxgeom:
        return

    axis_idx = AXIS_INDEX[axis]
    pos = np.zeros(3, dtype=float)
    pos[axis_idx] = position
    if axis == "x":
        mat = np.array(
            [
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=float,
        ).reshape(-1)
    elif axis == "y":
        mat = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=float,
        ).reshape(-1)
    else:
        mat = np.eye(3, dtype=float).reshape(-1)

    rgba = np.array([1.0, 0.15, 0.05, 0.22], dtype=float)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_PLANE,
        np.array([size, size, 0.001], dtype=float),
        pos,
        mat,
        rgba,
    )
    scene.ngeom += 1


def geom_mesh_name(model, geom_id: int) -> str | None:
    mesh_id = int(model.geom_dataid[geom_id])
    if mesh_id < 0:
        return None
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id)


def geom_body_name(model, geom_id: int) -> str | None:
    body_id = int(model.geom_bodyid[geom_id])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)


def configure_collision_visualization(model, config: dict) -> None:
    if not bool(config.get("enable", False)):
        return

    collision_group = int(config.get("collision_group", 3))
    collision_rgba = np.asarray(config.get("collision_rgba", [0.1, 0.65, 1.0, 0.12]), dtype=float)
    hand_rgba = np.asarray(config.get("hand_collision_rgba", [0.0, 1.0, 0.25, 0.55]), dtype=float)
    wall_rgba = np.asarray(config.get("wall_rgba", [1.0, 0.15, 0.05, 0.55]), dtype=float)
    show_all_collision = bool(config.get("show_all_collision", False))

    for geom_id in range(model.ngeom):
        group = int(model.geom_group[geom_id])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        mesh_name = geom_mesh_name(model, geom_id)
        body_name = geom_body_name(model, geom_id) or ""

        if name == "teleop_collision_wall":
            model.geom_rgba[geom_id] = wall_rgba
            continue

        is_hand_collision = (
            mesh_name == "hand_col"
            or body_name in ("panda0_leftfinger", "panda0_rightfinger")
            or body_name == "panda0_gripper"
        )
        if is_hand_collision:
            model.geom_rgba[geom_id] = hand_rgba
        elif show_all_collision and group == collision_group:
            model.geom_rgba[geom_id] = collision_rgba


def configure_viewer_visibility(
    viewer,
    wall_group: int,
    extra_groups: list[int] | None = None,
    visualization_config: dict | None = None,
) -> None:
    if viewer is None:
        return
    if 0 <= wall_group < len(viewer.opt.geomgroup):
        viewer.opt.geomgroup[wall_group] = 1
    for group in extra_groups or []:
        if 0 <= group < len(viewer.opt.geomgroup):
            viewer.opt.geomgroup[group] = 1
    visualization_config = visualization_config or {}
    if bool(visualization_config.get("disable_shadows", True)):
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
    if bool(visualization_config.get("disable_reflections", True)):
        viewer.opt.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0


def feedback_gain_from_torque_ratio(
    leader_max_current_raw: int,
    follower_max_torque: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    leader_max_torque = leader_max_current_raw * GOAL_CURRENT_UNIT_AMP * KT
    if np.any(follower_max_torque <= 0.0):
        raise ValueError("force_feedback.follower_max_torque must be positive for every joint")
    return scale * leader_max_torque / follower_max_torque, leader_max_torque


def run_loop(args: argparse.Namespace) -> int:
    xml_path = Path(args.xml).resolve()
    if args.external_force_mode == "collision_wall":
        xml_path = collision_wall_xml_path(xml_path, args.wall_config)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    configure_collision_visualization(model, args.visualization_config)
    data = mujoco.MjData(model)

    qpos_addrs = [joint_qpos_addr(model, name) for name in FRANKA_JOINT_NAMES]
    dof_addrs = [joint_dof_addr(model, name) for name in FRANKA_JOINT_NAMES]
    ee_site_id = site_id(model, args.external_force_site)

    comp = LeaderCompensation(args.leader_compensation, args.config_path)
    port_name = comp.dynamixel_config["port"]
    baudrate = comp.dynamixel_config["baudrate"]
    controller = DynamixelCurrentController(
        port_name,
        baudrate,
        int(comp.dynamixel_config["read_retries"]),
        float(comp.dynamixel_config["retry_delay"]),
    )

    try:
        leader_comp_active = not args.disable_leader_comp
        if leader_comp_active:
            print("Switching leader motors to Current Control Mode and enabling torque.")
            controller.set_torque(False)
            controller.set_current_control_mode()
            controller.set_torque(True)
            time.sleep(0.2)
        else:
            controller.set_torque(False)
            print("Leader compensation disabled; leader torque is disabled.")

        q_comp, dq_comp, raw, present_current = controller.read_state()
        raw_map = normalize_raw_position_near_zero(raw, args.q_m0 * RAW_PER_RAD)
        q_leader = raw_to_franka_joints(raw, args.q_m0, args.q_r0, args.teleop_sign)
        q_target = np.clip(q_leader, args.q_min, args.q_max)
        set_franka_qpos(model, data, q_target)
        last_q_cmd = q_target.copy()

        dt = 1.0 / args.frequency
        next_log = time.perf_counter()
        start = time.perf_counter()
        last_tau = np.zeros(7, dtype=float)
        last_tau_g = np.zeros(7, dtype=float)
        last_tau_fs = np.zeros(7, dtype=float)
        last_tau_fk = np.zeros(7, dtype=float)
        last_tau_feedback_applied = np.zeros(7, dtype=float)
        last_tau_limit = np.zeros(7, dtype=float)
        last_limit_status = np.zeros(7, dtype=int)
        last_goal_current = np.zeros(7, dtype=int)
        last_tau_ext = np.zeros(7, dtype=float)
        last_tau_feedback = np.zeros(7, dtype=float)
        last_wall_force = np.zeros(3, dtype=float)
        last_ee_pos = np.zeros(3, dtype=float)
        last_wall_penetration = 0.0
        last_wall_contacts = 0
        last_wall_contact_dist = 0.0
        last_wall_clamped = False

        print(f"Loaded MuJoCo model: {xml_path}")
        print(f"MuJoCo timestep: {model.opt.timestep:.4f}s")
        print("Follower control mode: kinematic qpos write")
        print(
            f"External force mode={args.external_force_mode}, site={args.external_force_site}, "
            f"wall_axis={args.wall_axis}, wall_pos={args.wall_pos}, wall_side={args.wall_side}, "
            f"wall_k={args.wall_k}, wall_d={args.wall_d}"
        )
        print(f"Force feedback source={args.feedback_source}")
        print("Force feedback joint_enable: " + np.array2string(args.feedback_joint_enable.astype(int), separator=","))
        print("Force feedback joint_gain: " + np.array2string(args.feedback_joint_gain, precision=3, separator=","))
        print("Force feedback joint_sign: " + np.array2string(args.feedback_joint_sign, precision=1, separator=","))
        print(f"Config: {args.config_path}")
        print("Leader max torque Nm: " + np.array2string(args.leader_max_torque, precision=3, separator=","))
        print("Follower max torque Nm: " + np.array2string(args.follower_max_torque, precision=3, separator=","))
        print("Kf,p torque ratio: " + np.array2string(args.feedback_gain, precision=4, separator=","))
        print(
            f"Force feedback enabled={args.feedback_enable}, max torque Nm="
            + np.array2string(args.feedback_max_torque, precision=3, separator=",")
        )
        print(
            f"Leader compensation active={leader_comp_active}, "
            f"gravity={comp.enable_gravity}, friction={comp.enable_friction}, "
            f"static={comp.enable_static_friction}, kinetic={comp.enable_kinetic_friction}"
        )
        if comp.soft_limit_enable:
            print("Soft limit min: " + np.array2string(comp.soft_limit_min, precision=3, separator=","))
            print("Soft limit max: " + np.array2string(comp.soft_limit_max, precision=3, separator=","))
            print("Soft limit margin: " + np.array2string(comp.soft_limit_margin, precision=3, separator=","))
        print("Leader raw:     " + np.array2string(raw.astype(int)))
        print("Leader raw map: " + np.array2string(np.rint(raw_map).astype(int)))
        print("Leader q rad:   " + np.array2string(q_leader, precision=4))
        print("Initial target: " + np.array2string(q_target, precision=4))
        print("Press Ctrl+C in this terminal to stop.")

        def step_once() -> None:
            nonlocal next_log, last_tau, last_tau_g, last_tau_fs, last_tau_fk
            nonlocal last_tau_feedback_applied
            nonlocal last_tau_limit, last_limit_status
            nonlocal last_goal_current, last_tau_ext, last_tau_feedback
            nonlocal last_wall_force, last_ee_pos, last_wall_penetration
            nonlocal last_wall_contacts
            nonlocal last_wall_contact_dist
            nonlocal last_wall_clamped, last_q_cmd
            loop_start = time.perf_counter()
            q_comp_now, dq_comp_now, raw_now, present_current_now = controller.read_state()
            raw_map_now = normalize_raw_position_near_zero(raw_now, args.q_m0 * RAW_PER_RAD)
            q_leader_now = raw_to_franka_joints(raw_now, args.q_m0, args.q_r0, args.teleop_sign)

            q_candidate = np.clip(q_leader_now, args.q_min, args.q_max)
            q_cmd = q_candidate
            last_wall_clamped = False
            if args.external_force_mode == "collision_wall" and args.wall_clamp_enable:
                q_cmd, last_wall_clamped, last_wall_contacts, last_wall_contact_dist = clamp_qpos_to_wall(
                    model,
                    data,
                    last_q_cmd,
                    q_candidate,
                    args.wall_clamp_min_dist,
                    args.wall_clamp_iterations,
                )
            else:
                set_franka_qpos(model, data, q_cmd)
            last_q_cmd = q_cmd.copy()
            wall_tau = np.zeros(7, dtype=float)
            last_wall_force = np.zeros(3, dtype=float)
            last_ee_pos = data.site_xpos[ee_site_id].copy()
            last_wall_penetration = 0.0
            if not (args.external_force_mode == "collision_wall" and args.wall_clamp_enable):
                last_wall_contacts = 0
                last_wall_contact_dist = 0.0
            if args.external_force_mode == "virtual_wall":
                wall_tau, last_wall_force, last_ee_pos, last_wall_penetration = virtual_wall_joint_torque(
                    model,
                    data,
                    ee_site_id,
                    dof_addrs,
                    args.wall_axis,
                    args.wall_pos,
                    args.wall_k,
                    args.wall_d,
                    args.wall_side,
                )
                last_tau_ext = args.virtual_wall_torque_scale * wall_tau
            elif args.external_force_mode == "collision_wall" and args.feedback_source == "contact_normal":
                (
                    wall_tau,
                    last_wall_force,
                    last_ee_pos,
                    last_wall_penetration,
                    last_wall_contacts,
                    last_wall_contact_dist,
                ) = (
                    collision_wall_contact_torque(
                        model,
                        data,
                        ee_site_id,
                        dof_addrs,
                        args.wall_axis,
                        args.wall_pos,
                        args.wall_side,
                        args.virtual_wall_torque_scale,
                        args.feedback_contact_force_max,
                    )
                )
                last_tau_ext = wall_tau
            else:
                last_tau_ext = follower_external_joint_torque(data, dof_addrs)
            last_tau_feedback = (
                args.feedback_joint_enable
                * args.feedback_joint_sign
                * args.feedback_sign
                * args.feedback_joint_gain
                * args.feedback_gain
                * last_tau_ext
                - args.feedback_damping * dq_comp_now
            )
            if not args.feedback_enable:
                last_tau_feedback = np.zeros(7, dtype=float)

            (
                tau,
                tau_g,
                tau_fs,
                tau_fk,
                tau_feedback_safe,
                tau_limit,
                limit_status,
                goal_current,
            ) = comp.compute(
                q_comp_now,
                dq_comp_now,
                last_tau_feedback,
                args.feedback_max_torque,
            )
            if leader_comp_active:
                ramp = min(1.0, (loop_start - start) / max(comp.ramp_sec, 1e-6))
                controller.write_goal_current(np.rint(ramp * goal_current).astype(int))
            last_tau = tau
            last_tau_g = tau_g
            last_tau_fs = tau_fs
            last_tau_fk = tau_fk
            last_tau_feedback_applied = tau_feedback_safe
            last_tau_limit = tau_limit
            last_limit_status = limit_status
            last_goal_current = goal_current

            now = time.perf_counter()
            if now >= next_log:
                q_sim = data.qpos[qpos_addrs].copy()
                q_err = q_cmd - q_sim
                print(
                    # "raw="
                    # + np.array2string(raw_now.astype(int), separator=",")
                    # + " raw_map="
                    # + np.array2string(np.rint(raw_map_now).astype(int), separator=",")
                    # + " q_leader="
                    # + np.array2string(q_leader_now, precision=3, separator=",")
                    # + " q_cmd="
                    # + np.array2string(q_cmd, precision=3, separator=",")
                    # + " q_sim="
                    # + np.array2string(q_sim, precision=3, separator=",")
                    # + " q_err="
                    # + np.array2string(q_err, precision=3, separator=",")
                    "dq_leader="
                    + np.array2string(dq_comp_now, precision=3, separator=",")
                    + " leader_goal_raw="
                    + np.array2string(last_goal_current, separator=",")
                    # + " tau_leader="
                    # + np.array2string(last_tau, precision=3, separator=",")
                    +" tau_g="
                    + np.array2string(last_tau_g, precision=3, separator=",")
                    + " tau_fs="
                    + np.array2string(last_tau_fs, precision=3, separator=",")
                    # + " tau_fk="
                    # + np.array2string(last_tau_fk, precision=3, separator=",")

                    # +" tau_ext_sim="
                    # + np.array2string(last_tau_ext, precision=3, separator=",")
                    # + " tau_feedback_sim="
                    # + np.array2string(last_tau_feedback, precision=3, separator=",")
                    # + " tau_feedback_applied="
                    # + np.array2string(last_tau_feedback_applied, precision=3, separator=",")
                    # + " tau_limit="
                    # + np.array2string(last_tau_limit, precision=3, separator=",")
                    # + f" limit={format_limit_status(last_limit_status)}"
                    # + " ee_pos="
                    # + np.array2string(last_ee_pos, precision=3, separator=",")
                    # + " wall_force="
                    # + np.array2string(last_wall_force, precision=3, separator=",")
                    # +f" wall_contacts={last_wall_contacts}" 
                    # + f" wall_clamped={int(last_wall_clamped)}"
                    # + f" wall_dist={last_wall_contact_dist:.4f}"
                    # + f" wall_pen={last_wall_penetration:.4f}"
                )
                next_log = now + args.log_interval

            sleep_time = dt - (time.perf_counter() - loop_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        if args.no_viewer:
            while True:
                step_once()

        with mujoco.viewer.launch_passive(model, data) as viewer:
            configure_viewer_visibility(
                viewer,
                int(args.wall_config.get("group", 0)),
                [int(group) for group in args.visualization_config.get("visible_geom_groups", [])],
                args.visualization_config,
            )
            while viewer.is_running():
                step_once()
                viewer.user_scn.ngeom = 0
                if args.external_force_mode == "virtual_wall":
                    add_virtual_wall_marker(viewer, args.wall_axis, args.wall_pos, args.wall_size)
                viewer.sync()
        return 0
    finally:
        try:
            if not args.disable_leader_comp:
                controller.safe_shutdown()
            else:
                controller.set_torque(False)
        finally:
            controller.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Teleoperate MuJoCo Franka with physical Dynamixel leader.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=f"Teleop YAML config path, default: {DEFAULT_CONFIG_PATH}",
    )
    cli_args = parser.parse_args()

    config_path = Path(cli_args.config).resolve()
    config = load_yaml_config(config_path)
    mujoco_config = config["mujoco"]
    runtime_config = config["runtime"]
    mapping_config = config["teleop_mapping"]
    limits_config = config["franka_joint_limits"]
    external_config = config["external_force"]
    wall_config = external_config["virtual_wall"]
    feedback_config = config["force_feedback"]
    visualization_config = config.get("visualization", {})
    leader_controller_config = config["leader_compensation"]["controller"]
    if external_config["mode"] not in ("none", "virtual_wall", "collision_wall"):
        raise ValueError("external_force.mode must be one of: none, virtual_wall, collision_wall")
    feedback_source = str(feedback_config.get("source", "contact_normal"))
    if feedback_source not in ("contact_normal", "qfrc_constraint"):
        raise ValueError("force_feedback.source must be one of: contact_normal, qfrc_constraint")
    follower_max_torque = np.asarray(feedback_config["follower_max_torque"], dtype=float)
    feedback_max_torque = np.asarray(feedback_config["max_torque"], dtype=float)
    feedback_joint_enable = optional_vector(feedback_config, "joint_enable", 7, float, [1.0] * 7)
    feedback_joint_gain = optional_vector(feedback_config, "joint_gain", 7, float, [1.0] * 7)
    feedback_joint_sign = optional_vector(feedback_config, "joint_sign", 7, float, [1.0] * 7)
    if feedback_max_torque.shape != (7,):
        raise ValueError("force_feedback.max_torque must have length 7")
    feedback_gain, leader_max_torque = feedback_gain_from_torque_ratio(
        int(leader_controller_config["max_current_raw"]),
        follower_max_torque,
        float(feedback_config["scale"]),
    )

    args = argparse.Namespace(
        config_path=str(config_path),
        xml=mujoco_config["xml"],
        frequency=float(mujoco_config["frequency"]),
        no_viewer=bool(runtime_config["no_viewer"]),
        disable_leader_comp=bool(runtime_config["disable_leader_comp"]),
        log_interval=float(runtime_config["log_interval"]),
        q_m0=np.deg2rad(np.asarray(mapping_config["q_m0_deg"], dtype=float)),
        q_r0=np.asarray(mapping_config["q_r0"], dtype=float),
        teleop_sign=np.asarray(mapping_config["sign"], dtype=float),
        q_min=np.asarray(limits_config["min"], dtype=float),
        q_max=np.asarray(limits_config["max"], dtype=float),
        leader_compensation=config["leader_compensation"],
        external_force_mode=external_config["mode"],
        external_force_site=external_config["site"],
        wall_axis=wall_config["axis"],
        wall_pos=float(wall_config["position"]),
        wall_side=wall_config["side"],
        wall_k=float(wall_config["stiffness"]),
        wall_d=float(wall_config["damping"]),
        wall_size=float(wall_config["display_size"]),
        virtual_wall_torque_scale=float(wall_config["torque_scale"]),
        wall_clamp_enable=bool(wall_config.get("clamp_enable", True)),
        wall_clamp_min_dist=float(wall_config.get("clamp_min_dist", 0.0)),
        wall_clamp_iterations=int(wall_config.get("clamp_iterations", 8)),
        wall_config=wall_config,
        visualization_config=visualization_config,
        feedback_enable=bool(feedback_config["enable"]),
        feedback_source=feedback_source,
        feedback_joint_enable=feedback_joint_enable,
        feedback_joint_gain=feedback_joint_gain,
        feedback_joint_sign=feedback_joint_sign,
        feedback_contact_force_max=float(feedback_config.get("contact_force_max", 0.0)),
        feedback_gain=feedback_gain,
        feedback_max_torque=feedback_max_torque,
        leader_max_torque=leader_max_torque,
        follower_max_torque=follower_max_torque,
        feedback_damping=float(feedback_config["damping"]),
        feedback_sign=float(feedback_config["sign"]),
    )

    return run_loop(args)


if __name__ == "__main__":
    raise SystemExit(main())
