# FACTR Teleop Windows Experiment

This repository is a Windows/no-ROS experimental fork based on FACTR Teleop. The current focus is using a 7-DOF Dynamixel leader arm to teleoperate a MuJoCo Franka Panda follower while developing leader-side gravity compensation, friction compensation, force feedback, collision-wall interaction, and joint-limit protection.

Original FACTR resources:

- Project page: https://jasonjzliu.com/factr/
- Paper: https://arxiv.org/abs/2502.17432
- Upstream code: https://github.com/RaindragonD/factr/

## Current Hardware

Leader arm:

- 7 Dynamixel motors, IDs `21-27`
- Serial port: `COM21`
- Baudrate: `57600`
- Joint 2 and 4: `XM540-W270`
- Joint 1, 3, 5, 6, 7: `XM430-W350`

Motor/current constants:

- `XM430-W350`: `Kt = 1.783 Nm/A`
- `XM540-W270`: `Kt = 2.409 Nm/A`
- Dynamixel Goal Current unit: `2.69 mA/raw`

## Main Files

- `test/leader_to_mujoco_franka.py`: main teleoperation script. Reads the physical Dynamixel leader, writes MuJoCo Franka joint positions, and applies leader-side compensation and force feedback.
- `test/leader_teleop_config.yaml`: main runtime configuration for teleoperation, MuJoCo, compensation, wall contact, force feedback, visualization, and joint-limit barrier.
- `test/leader_gravity_comp.py`: standalone leader gravity/friction compensation test script.
- `test/leader_comp_config.yaml`: standalone compensation configuration.
- `franka_exo/robot.urdf`: leader arm URDF used by Pinocchio for inverse dynamics and gravity compensation.
- `franka_sim/franka_panda.xml`: base MuJoCo Franka model.
- `franka_sim/franka_panda_teleop_wall.xml`: generated MuJoCo model with the teleoperation collision wall.

## Environment

The current development environment is a conda environment named:

```powershell
ftservo
```

Core Python dependencies used in this workflow:

- `numpy`
- `pyyaml`
- `mujoco`
- `pinocchio`
- `dynamixel_sdk`
- `meshcat` for URDF visualization/debugging

ROS is not required for the Windows test scripts.

## Teleoperation

Run the main teleoperation script:

```powershell
python test\leader_to_mujoco_franka.py
```

The script:

1. Opens the Dynamixel leader on `COM21`.
2. Reads motor positions, velocities, and currents.
3. Maps motor raw positions to Franka joint angles.
4. Writes the follower state directly into MuJoCo `qpos`.
5. Computes leader-side gravity, friction, force feedback, and joint-limit torques.
6. Sends Dynamixel Goal Current commands in current-control mode.

The follower currently uses kinematic `qpos` writing for low-latency teleoperation. This is fast, but MuJoCo contact does not physically block motion unless explicit wall clamp logic is enabled.

## Joint Mapping

The teleoperation mapping is configured in `test/leader_teleop_config.yaml`:

```yaml
teleop_mapping:
  q_m0_deg: [180, 180, 180, 180, 180, 180, 180]
  q_r0: [0.0, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0]
  sign: [1, 1, 1, -1, 1, -1, 1]
```

The script normalizes Dynamixel multi-turn raw position readings back to the nearest single-turn value around the motor zero position. This prevents abnormal readings such as `-24660` on joint 7 from producing unrealistic joint angles.

## Gravity Compensation

Gravity compensation is computed with Pinocchio using:

```python
tau_g = pin.rnea(model, data, q, dq, zeros)
```

The leader URDF is:

```text
franka_exo/robot.urdf
```

The gravity compensation configuration is:

```yaml
gravity_comp:
  enable: true
  gain: ...
  joint_gain: [...]
```

The URDF mass parameters have been adjusted during calibration. Some masses were reduced, and the `motor_7` mesh mass was added into `link_5` because `motor_7` is a mesh part, not a separate URDF link.

## Friction Compensation

Two friction terms are implemented:

- Static friction compensation for low-speed stiction.
- Kinetic friction compensation with Coulomb and viscous terms.

Static friction compensation is configured with:

```yaml
static:
  enable: true
  enable_speed: ...
  gain: ...
  joint_gain: [...]
  min_torque: [...]
  max_torque: [...]
```

Kinetic friction compensation is configured with:

```yaml
kinetic:
  enable: true
  velocity_deadband: ...
  coulomb: [...]
  viscous: [...]
```

The tuning goal is:

- The leader is easier to start moving.
- It does not drift when untouched.
- It does not chatter or knock gear backlash.

## Force Feedback

The current force-feedback path uses a MuJoCo collision wall:

```text
wall contact normal force
-> end-effector translational Jacobian
-> follower joint torque estimate
-> scaled leader feedback torque
-> Dynamixel current command
```

Force feedback is configured in:

```yaml
force_feedback:
  enable: true
  source: contact_normal
  joint_enable: [...]
  joint_gain: [...]
  joint_sign: [...]
  contact_force_max: ...
  scale: ...
  max_torque: [...]
  damping: ...
```

Unlike upstream FACTR, this experiment keeps per-joint force-feedback switches, gains, and signs because the leader hardware, MuJoCo contact source, and joint mapping are being calibrated independently.

Important safety note: force feedback should be tuned conservatively. Start with one joint enabled, low `scale`, low `max_torque`, and nonzero damping.

## Collision Wall and Visualization

The wall is generated from the base Franka XML and inserted into the MuJoCo world as:

```text
teleop_collision_wall
```

Visualization options are configured in YAML:

```yaml
visualization:
  enable: true
  visible_geom_groups: [0, 3]
  show_all_collision: false
  hand_collision_rgba: [...]
  wall_rgba: [...]
  disable_shadows: true
  disable_reflections: true
```

Hand and finger collision geoms are highlighted to make it easier to debug contact.

The wall clamp option prevents kinematic `qpos` writing from pushing the follower through the wall:

```yaml
clamp_enable: true
clamp_min_dist: 0.0
clamp_iterations: 10
```

This is a kinematic guard, not a true dynamics simulation.

## Joint-Limit Barrier

The leader-side joint-limit barrier follows the FACTR-style repulsive torque:

```python
tau_limit = -kp * (q - limit) - kd * dq
```

The script compares:

1. Leader URDF joint limits.
2. Follower Franka joint limits transformed into leader coordinates.

It uses the intersection as the effective safe range and applies a 3 degree margin:

```yaml
soft_limits:
  enable: true
  source: leader_follower_intersection
  margin_deg: 3.0
```

Runtime logs include:

```text
limit=none
limit=J2:low
limit=J6:high
tau_limit=[...]
```

## Useful Debug Logs

The teleoperation script can print:

- `raw`
- `raw_map`
- `q_leader`
- `q_cmd`
- `q_sim`
- `tau_g`
- `tau_fs`
- `tau_fk`
- `tau_ext_sim`
- `tau_feedback_sim`
- `tau_feedback_applied`
- `tau_limit`
- `wall_contacts`
- `wall_clamped`
- `wall_dist`
- `wall_pen`
- `limit`

These logs are useful for separating hardware readout issues, joint mapping issues, MuJoCo contact issues, and leader current-control issues.

## Safety Notes

- Keep one hand near the emergency stop or power switch while testing force feedback.
- Start force feedback with one joint enabled.
- Use small current and torque limits before increasing gains.
- If a joint chatters, drifts, or hits backlash, reduce friction/feedback gains or add current deadband.
- If a joint is pushed toward its mechanical limit, verify `soft_limits` and `joint_sign`.
- Do not tune force feedback before gravity and friction compensation are stable.

## Development Status

Implemented:

- Dynamixel leader readout on Windows.
- MuJoCo Franka kinematic teleoperation.
- Gravity compensation.
- Static and kinetic friction compensation.
- Collision wall and contact-force extraction.
- Force-feedback torque path.
- Collision visualization.
- Joint-limit barrier.
- Current deadband.
- Dynamixel multi-turn raw normalization.

Known limitations:

- Force feedback still needs low-pass filtering and torque slew-rate limiting.
- Kinematic wall clamp is not a physically accurate contact simulation.
- URDF inertias may need recalibration after mass edits.
- Contact forces from MuJoCo do not exactly represent a real Franka external torque sensor.

## Attribution

This work is based on the FACTR Teleop codebase and the FACTR paper:

```bibtex
@article{factr,
  title={FACTR: Force-Attending Curriculum Training for Contact-Rich Policy Learning},
  author={Liu, Jason Jingzhou and Li, Yulong and Shaw, Kenneth and Tao, Tony and Salakhutdinov, Ruslan and Pathak, Deepak},
  journal={arXiv preprint arXiv:2502.17432},
  year={2025}
}
```
