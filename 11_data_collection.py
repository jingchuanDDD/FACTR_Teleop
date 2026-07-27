#!/usr/bin/env python3
"""Standalone dual-RealSense, dual-tactile, FR3 state, and action recorder.

This program does not import the exoskeleton SDK and never sends FR3 commands.
It only reads cached FR3 EE/gripper states and derives reached actions from
consecutive states. Run it in the Python environment where dual RealSense
streaming works, while running 9_exo_teleop_fr3.py separately for
teleoperation.

Controls (terminal or either OpenCV preview window):
    S: start/stop an episode
    Q or Esc: stop recording and exit
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import queue
import re
import select
import sys
import termios
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
import tty

import cv2
import numpy as np
import pyrealsense2 as rs

from real_world.fr3_ee_control_client import FR3EEClient
from real_world.fr3_gripper import GripperClient


DEFAULT_CAMERAS = (
    {"name": "wrist", "serial": "922612070441"},
    {"name": "third", "serial": "143322074106"},
)

DEFAULT_TACTILE_SENSORS = (
    {"name": "left", "dev_id": "L25480053", "pc_port": 60001},
    {"name": "right", "dev_id": "L25480055", "pc_port": 60002},
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


class TerminalKeyboard:
    """Non-blocking terminal keyboard input with reliable state restoration."""

    def __init__(self) -> None:
        self.fd: Optional[int] = None
        self.old_settings = None

    def open(self) -> bool:
        if not sys.stdin.isatty():
            print("[Keys] stdin is not a TTY; use the OpenCV windows for keys.")
            return False
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return True

    def poll(self) -> List[str]:
        keys: List[str] = []
        if self.fd is None:
            return keys
        while select.select([sys.stdin], [], [], 0.0)[0]:
            key = sys.stdin.read(1)
            if not key:
                break
            keys.append(key)
        return keys

    def close(self) -> None:
        if self.fd is not None and self.old_settings is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass
        self.fd = None
        self.old_settings = None


def _gripper_width_from_state(state: Dict[str, Any]) -> Optional[float]:
    payload = state.get("state", state)
    for key in ("width", "gripper_width", "hand_width"):
        value = payload.get(key)
        if value is not None:
            return float(value)
    match = re.search(r"\bwidth:\s*([-+0-9.eE]+)", str(payload.get("repr", "")))
    return float(match.group(1)) if match else None


def _quaternion_multiply_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply two quaternions stored in ``[x, y, z, w]`` order."""
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return np.asarray(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ],
        dtype=np.float64,
    )


def _rotation_delta_axis_angle_xyzw(
    previous_xyzw: np.ndarray,
    current_xyzw: np.ndarray,
) -> np.ndarray:
    """Return the shortest base-frame rotation from previous to current."""
    previous = np.asarray(previous_xyzw, dtype=np.float64)
    current = np.asarray(current_xyzw, dtype=np.float64)
    if previous.shape != (4,) or current.shape != (4,):
        raise ValueError("quaternions must have shape (4,)")
    previous_norm = float(np.linalg.norm(previous))
    current_norm = float(np.linalg.norm(current))
    if previous_norm <= 1e-12 or current_norm <= 1e-12:
        raise ValueError("quaternion norm must be non-zero")
    previous /= previous_norm
    current /= current_norm
    previous_inverse = np.asarray(
        [-previous[0], -previous[1], -previous[2], previous[3]],
        dtype=np.float64,
    )
    relative = _quaternion_multiply_xyzw(current, previous_inverse)
    # q and -q describe the same rotation. Pick the representation whose
    # angle lies in [0, pi] so discontinuous quaternion signs do not create
    # spurious 2*pi actions.
    if relative[3] < 0.0:
        relative = -relative
    vector_norm = float(np.linalg.norm(relative[:3]))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float64)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[3], -1.0, 1.0))
    return relative[:3] * (angle / vector_norm)


class FrankaActionCollector:
    """Derive a 7-DoF action from consecutive measured FR3 states.

    The action is ``[dx, dy, dz, dRx, dRy, dRz, gripper_width]``. Translation
    and axis-angle rotation are relative to the preceding valid sample and
    expressed in the FR3 base frame; gripper width is absolute in meters.
    """

    def __init__(self) -> None:
        self._previous: Optional[Dict[str, Any]] = None

    @staticmethod
    def _snapshot(franka_sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not franka_sample.get("valid", False):
            return None
        ee = franka_sample.get("ee", {})
        gripper = franka_sample.get("gripper", {})
        position = ee.get("position")
        quaternion = ee.get("quaternion_xyzw")
        width = gripper.get("width")
        if position is None or quaternion is None or width is None:
            return None
        position_array = np.asarray(position, dtype=np.float64)
        quaternion_array = np.asarray(quaternion, dtype=np.float64)
        if position_array.shape != (3,) or quaternion_array.shape != (4,):
            return None
        if not (
            np.all(np.isfinite(position_array))
            and np.all(np.isfinite(quaternion_array))
            and np.isfinite(width)
        ):
            return None
        return {
            "position": position_array.copy(),
            "quaternion_xyzw": quaternion_array.copy(),
            "gripper_width": float(width),
            "timestamp_monotonic": float(franka_sample["timestamp_monotonic"]),
        }

    def collect(self, franka_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Collect the action that moved the robot into ``franka_sample``."""
        current = self._snapshot(franka_sample)
        if current is None:
            return {
                "valid": False,
                "reason": "invalid_franka_state",
                "vector": None,
            }

        previous = self._previous
        self._previous = current
        if previous is None:
            return {
                "valid": False,
                "reason": "no_previous_state",
                "vector": None,
            }

        delta_position = current["position"] - previous["position"]
        try:
            delta_rotation = _rotation_delta_axis_angle_xyzw(
                previous["quaternion_xyzw"],
                current["quaternion_xyzw"],
            )
        except ValueError as exc:
            return {"valid": False, "reason": str(exc), "vector": None}
        vector = np.concatenate(
            [delta_position, delta_rotation, [current["gripper_width"]]]
        )
        return {
            "valid": True,
            "representation": "delta_ee_base_axis_angle_absolute_gripper_width",
            "delta_position": delta_position,
            "delta_rotation_axis_angle": delta_rotation,
            "gripper_width": current["gripper_width"],
            "dt_s": (
                current["timestamp_monotonic"]
                - previous["timestamp_monotonic"]
            ),
            "vector": vector,
        }


class FrankaStateSource:
    """Non-blocking cached FR3 EE and gripper state source."""

    def __init__(
        self,
        host: str,
        ee_port: int,
        gripper_port: int,
        timeout_s: float,
        poll_hz: float,
        max_age_s: float,
    ) -> None:
        self.host = host
        self.ee_port = ee_port
        self.gripper_port = gripper_port
        self.timeout_s = timeout_s
        self.poll_hz = poll_hz
        self.max_age_s = max_age_s
        self.ee = FR3EEClient(host, ee_port, timeout_s)
        self.gripper = GripperClient(host, gripper_port, timeout_s)
        self._gripper_stop = threading.Event()
        self._gripper_thread: Optional[threading.Thread] = None
        self._gripper_lock = threading.Lock()
        self._gripper_state: Optional[Dict[str, Any]] = None
        self._gripper_recv_mono = 0.0
        self._last_gripper_error_mono = 0.0

    def _gripper_poll_loop(self) -> None:
        period_s = 1.0 / self.poll_hz
        while not self._gripper_stop.is_set():
            started = time.monotonic()
            try:
                state = self.gripper.get_state()
                with self._gripper_lock:
                    self._gripper_state = state
                    self._gripper_recv_mono = time.monotonic()
            except Exception as exc:
                if self._gripper_stop.is_set():
                    break
                now = time.monotonic()
                if now - self._last_gripper_error_mono >= 1.0:
                    print(f"[Franka] Gripper state warning: {exc}")
                    self._last_gripper_error_mono = now
            wait_s = period_s - (time.monotonic() - started)
            if wait_s > 0.0:
                self._gripper_stop.wait(wait_s)

    def open(self) -> None:
        print(
            f"[Franka] Connecting EE {self.host}:{self.ee_port} and "
            f"gripper {self.host}:{self.gripper_port}..."
        )
        self.ee.connect()
        self.ee.start_state_poller(self.poll_hz, self.max_age_s)
        if not self.ee.wait_until_state_ready(self.timeout_s):
            raise RuntimeError("timed out waiting for the first FR3 EE state")

        self.gripper.connect()
        self._gripper_stop.clear()
        self._gripper_thread = threading.Thread(
            target=self._gripper_poll_loop,
            name="fr3-gripper-state",
            daemon=False,
        )
        self._gripper_thread.start()
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            with self._gripper_lock:
                ready = self._gripper_state is not None
            if ready:
                print("[Franka] EE and gripper states ready.")
                return
            time.sleep(0.01)
        raise RuntimeError("timed out waiting for the first FR3 gripper state")

    def read(self) -> Dict[str, Any]:
        captured_mono = time.monotonic()
        ee_valid = True
        ee_state: Dict[str, Any] = {}
        ee_age_s: Optional[float] = None
        try:
            ee_state, ee_age_s = self.ee.get_state_cached(
                max_age_s=self.max_age_s,
                fallback_to_sync=False,
            )
        except Exception:
            ee_valid = False

        with self._gripper_lock:
            gripper_state = (
                None if self._gripper_state is None else dict(self._gripper_state)
            )
            gripper_recv_mono = self._gripper_recv_mono
        gripper_age_s = (
            None
            if gripper_state is None
            else max(0.0, captured_mono - gripper_recv_mono)
        )
        gripper_valid = (
            gripper_state is not None
            and gripper_age_s is not None
            and gripper_age_s <= self.max_age_s
        )

        return {
            "valid": ee_valid and gripper_valid,
            "timestamp_wall": time.time(),
            "timestamp_monotonic": captured_mono,
            "ee": {
                "valid": ee_valid,
                "state_age_s": ee_age_s,
                "position": ee_state.get("ee_pos"),
                "quaternion_xyzw": ee_state.get("ee_quat"),
                "joint_position": ee_state.get("q"),
            },
            "gripper": {
                "valid": gripper_valid,
                "state_age_s": gripper_age_s,
                "width": (
                    _gripper_width_from_state(gripper_state)
                    if gripper_state is not None
                    else None
                ),
                "state": gripper_state,
            },
        }

    def close(self) -> None:
        self._gripper_stop.set()
        # Closing the socket unblocks a get_state() currently waiting on recv().
        try:
            self.gripper.close()
        except Exception:
            pass
        if self._gripper_thread is not None:
            self._gripper_thread.join(timeout=max(2.5, self.timeout_s + 0.5))
        self._gripper_thread = None
        try:
            self.ee.close()
        except Exception:
            pass


class RealSenseSource:
    """One RealSense color stream selected by hardware serial number."""

    def __init__(
        self,
        name: str,
        serial: str,
        width: int,
        height: int,
        fps: int,
        startup_timeout_ms: int,
        read_timeout_ms: int,
        warmup_frames: int,
        start_retries: int,
        retry_sleep_s: float,
    ) -> None:
        self.name = name
        self.serial = str(serial)
        self.width = width
        self.height = height
        self.fps = fps
        self.startup_timeout_ms = startup_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.warmup_frames = warmup_frames
        self.start_retries = start_retries
        self.retry_sleep_s = retry_sleep_s
        self.pipeline = None

    @staticmethod
    def available_devices() -> Dict[str, str]:
        devices: Dict[str, str] = {}
        for device in rs.context().query_devices():
            serial = device.get_info(rs.camera_info.serial_number)
            name = device.get_info(rs.camera_info.name)
            devices[str(serial)] = str(name)
        return devices

    def _new_pipeline(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color,
            self.width,
            self.height,
            rs.format.bgr8,
            self.fps,
        )
        return pipeline, config

    def open(self) -> None:
        available = self.available_devices()
        if self.serial not in available:
            raise RuntimeError(
                f"RealSense {self.name} serial={self.serial} not found; "
                f"available devices: {available}"
            )

        last_error: Optional[Exception] = None
        for attempt in range(1, self.start_retries + 1):
            pipeline, config = self._new_pipeline()
            try:
                print(
                    f"[Camera] Starting {self.name} serial={self.serial} "
                    f"({attempt}/{self.start_retries})..."
                )
                pipeline.start(config)
                for _ in range(self.warmup_frames):
                    pipeline.wait_for_frames(self.startup_timeout_ms)
                self.pipeline = pipeline
                print(
                    f"[Camera] {self.name} ready: "
                    f"{self.width}x{self.height}@{self.fps}"
                )
                return
            except Exception as exc:
                last_error = exc
                try:
                    pipeline.stop()
                except Exception:
                    pass
                print(f"[Camera] {self.name} start attempt failed: {exc}")
                if attempt < self.start_retries:
                    time.sleep(self.retry_sleep_s)

        raise RuntimeError(
            f"could not start RealSense {self.name} serial={self.serial}: "
            f"{last_error}"
        ) from last_error

    def read(self) -> Dict[str, Any]:
        if self.pipeline is None:
            raise RuntimeError(f"RealSense {self.name} is not open")
        try:
            frames = self.pipeline.wait_for_frames(self.read_timeout_ms)
        except Exception as exc:
            raise RuntimeError(
                f"failed reading RealSense {self.name} serial={self.serial}: {exc}"
            ) from exc
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(
                f"missing color frame from {self.name} serial={self.serial}"
            )
        return {
            "image": np.asanyarray(color_frame.get_data()).copy(),
            "frame_number": int(color_frame.get_frame_number()),
            "device_timestamp_ms": float(color_frame.get_timestamp()),
            "timestamp_wall": time.time(),
            "timestamp_monotonic": time.monotonic(),
        }

    def close(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
        self.pipeline = None


def _tactile_worker(
    sensor_config: Dict[str, Any],
    common_config: Dict[str, Any],
    data_queue,
    status_queue,
    stop_event,
) -> None:
    """Acquire deformation and depth for one tactile sensor."""
    name = str(sensor_config["name"])
    sensor = None
    try:
        from dmrobotics import Mode, Sensor, SensorOptions

        options = SensorOptions(
            dev_id=sensor_config["dev_id"],
            backend=common_config["backend"],
            mode=Mode.STANDARD,
            show_fps=False,
            max_fps=common_config["max_fps"],
            enable_raw=False,
            enable_deformation=True,
            enable_depth=True,
            enable_shear=False,
            enable_force=False,
            remote_addr=common_config["remote_addr"],
            pc_host=common_config["pc_host"],
            pc_port=sensor_config["pc_port"],
        )
        sensor = Sensor(options)
        status_queue.put(("ready", name, None))
        last_fid = -1

        while not stop_event.is_set():
            if sensor.getDevStatus() != 0:
                time.sleep(0.01)
                continue
            if not sensor.wait_for_new(last_fid, timeout_ms=200):
                continue

            deformation_fid, deformation = sensor.getDeformation2D()
            depth_fid, depth = sensor.getDepth()
            if deformation is None and depth is None:
                continue
            last_fid = depth_fid if depth is not None else deformation_fid
            sample = {
                "frame_id": last_fid,
                "timestamp_wall": time.time(),
                "timestamp_monotonic": time.monotonic(),
                "deformation": (
                    deformation.copy() if deformation is not None else None
                ),
                "depth": depth.copy() if depth is not None else None,
            }
            try:
                data_queue.put_nowait(sample)
            except queue.Full:
                pass
    except Exception as exc:
        try:
            status_queue.put(("error", name, repr(exc)))
        except Exception:
            pass
    finally:
        if sensor is not None:
            try:
                sensor.disconnect()
            except Exception:
                pass
        try:
            data_queue.close()
            data_queue.join_thread()
        except Exception:
            pass


class DualTactileSource:
    def __init__(
        self,
        sensor_configs: Tuple[Dict[str, Any], ...],
        backend: str,
        remote_addr: str,
        pc_host: str,
        max_fps: int,
        startup_timeout_s: float,
        max_age_s: float,
    ) -> None:
        self.sensor_configs = sensor_configs
        self.backend = backend
        self.remote_addr = remote_addr
        self.pc_host = pc_host
        self.max_fps = max_fps
        self.startup_timeout_s = startup_timeout_s
        self.max_age_s = max_age_s
        self.ctx = mp.get_context("spawn")
        self.stop_event = None
        self.status_queue = None
        self.data_queues: Dict[str, Any] = {}
        self.processes: Dict[str, Any] = {}
        self.latest: Dict[str, Dict[str, Any]] = {}

    def open(self) -> None:
        self.stop_event = self.ctx.Event()
        self.status_queue = self.ctx.Queue()
        common_config = {
            "backend": self.backend,
            "remote_addr": self.remote_addr,
            "pc_host": self.pc_host,
            "max_fps": self.max_fps,
        }
        for sensor_config in self.sensor_configs:
            name = str(sensor_config["name"])
            data_queue = self.ctx.Queue(maxsize=1)
            process = self.ctx.Process(
                target=_tactile_worker,
                args=(
                    dict(sensor_config), common_config, data_queue,
                    self.status_queue, self.stop_event,
                ),
                name=f"tactile-{name}",
            )
            self.data_queues[name] = data_queue
            self.processes[name] = process
            process.start()

        pending = set(self.processes)
        deadline = time.monotonic() + self.startup_timeout_s
        try:
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError(
                        f"timed out starting tactile sensors: {sorted(pending)}"
                    )
                try:
                    state, name, detail = self.status_queue.get(
                        timeout=min(0.2, remaining)
                    )
                except queue.Empty:
                    failed = [
                        name for name in pending
                        if not self.processes[name].is_alive()
                    ]
                    if failed:
                        raise RuntimeError(
                            f"tactile workers exited during startup: {failed}"
                        )
                    continue
                if state == "error":
                    raise RuntimeError(f"tactile sensor {name} failed: {detail}")
                if state == "ready":
                    pending.discard(name)
                    print(f"[Tactile] {name} ready")
        except Exception:
            self.close()
            raise

    def read(self) -> Dict[str, Any]:
        for name, data_queue in self.data_queues.items():
            while True:
                try:
                    self.latest[name] = data_queue.get_nowait()
                except queue.Empty:
                    break

        now = time.monotonic()
        sensors: Dict[str, Dict[str, Any]] = {}
        all_valid = True
        for config in self.sensor_configs:
            name = str(config["name"])
            sample = self.latest.get(name)
            if sample is None:
                all_valid = False
                continue
            age_s = max(0.0, now - sample["timestamp_monotonic"])
            valid = (
                age_s <= self.max_age_s
                and sample.get("deformation") is not None
                and sample.get("depth") is not None
            )
            sensors[name] = {**sample, "age_s": age_s, "valid": valid}
            all_valid = all_valid and valid
        if len(sensors) != len(self.sensor_configs):
            all_valid = False
        return {"valid": all_valid, "sensors": sensors}

    def _drain_queues(self) -> None:
        for data_queue in self.data_queues.values():
            while True:
                try:
                    data_queue.get_nowait()
                except (queue.Empty, EOFError, OSError):
                    break

    def close(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        deadline = time.monotonic() + 3.0
        while (
            any(process.is_alive() for process in self.processes.values())
            and time.monotonic() < deadline
        ):
            self._drain_queues()
            for process in self.processes.values():
                process.join(timeout=0.05)

        for process in self.processes.values():
            if process.is_alive():
                process.terminate()
            process.join(timeout=1.0)

        self._drain_queues()
        for data_queue in self.data_queues.values():
            try:
                data_queue.close()
                data_queue.join_thread()
            except Exception:
                pass
        if self.status_queue is not None:
            try:
                self.status_queue.close()
                self.status_queue.join_thread()
            except Exception:
                pass

        self.data_queues.clear()
        self.processes.clear()
        self.latest.clear()
        self.stop_event = None
        self.status_queue = None


class EpisodeWriter:
    """Write color videos, tactile NPZ files, actions, and per-step metadata."""

    def __init__(
        self,
        dataset_dir: str,
        camera_names: List[str],
        camera_fps: int,
        metadata: Dict[str, Any],
    ) -> None:
        root = Path(dataset_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        existing_indices = []
        existing_episode_count = 0
        for path in root.iterdir():
            if not path.is_dir():
                continue
            match = re.fullmatch(r"episode_(\d+)", path.name)
            if match:
                existing_indices.append(int(match.group(1)))
                existing_episode_count += 1
            elif path.name.startswith("sensor_episode_"):
                # Include episodes saved by older versions when choosing the
                # first sequential episode number.
                existing_episode_count += 1
        episode_index = max(
            max(existing_indices, default=-1) + 1,
            existing_episode_count,
        )
        name = f"episode_{episode_index}"
        episode_dir = root / name
        while episode_dir.exists():
            episode_index += 1
            name = f"episode_{episode_index}"
            episode_dir = root / name
        episode_dir.mkdir()
        (episode_dir / "videos").mkdir()
        (episode_dir / "tactile").mkdir()

        self.episode_dir = episode_dir
        self.camera_fps = camera_fps
        self.camera_names = camera_names
        self.video_writers: Dict[str, Any] = {}
        self.video_indices = {name: 0 for name in camera_names}
        self.steps_file = (episode_dir / "steps.jsonl").open("w", encoding="utf-8")
        self.started_wall = time.time()
        self.started_mono = time.monotonic()
        self.num_steps = 0
        self.action_collector = FrankaActionCollector()
        self.metadata = {
            **metadata,
            "format": "dual_camera_dual_tactile_action_v2",
            "episode_id": name,
            "created_at": datetime.now().astimezone().isoformat(),
            "success": None,
            "num_steps": 0,
            "duration_s": 0.0,
        }
        self._write_metadata()

    def _write_metadata(self) -> None:
        (self.episode_dir / "metadata.json").write_text(
            json.dumps(_jsonable(self.metadata), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def write_step(
        self,
        camera_samples: Dict[str, Dict[str, Any]],
        tactile_sample: Dict[str, Any],
        franka_sample: Dict[str, Any],
    ) -> None:
        camera_refs: Dict[str, Dict[str, Any]] = {}
        for name, sample in camera_samples.items():
            image = sample["image"]
            if name not in self.video_writers:
                height, width = image.shape[:2]
                path = self.episode_dir / "videos" / f"{name}.mp4"
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    self.camera_fps,
                    (width, height),
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not create video {path}")
                self.video_writers[name] = writer
            index = self.video_indices[name]
            self.video_writers[name].write(image)
            self.video_indices[name] = index + 1
            camera_refs[name] = {
                "video": f"videos/{name}.mp4",
                "frame_index": index,
                "device_frame_number": sample["frame_number"],
                "device_timestamp_ms": sample["device_timestamp_ms"],
                "timestamp_wall": sample["timestamp_wall"],
                "timestamp_monotonic": sample["timestamp_monotonic"],
            }

        tactile_arrays: Dict[str, np.ndarray] = {}
        tactile_refs: Dict[str, Dict[str, Any]] = {}
        for name, sample in tactile_sample["sensors"].items():
            channels: List[str] = []
            for channel in ("deformation", "depth"):
                value = sample.get(channel)
                if value is not None:
                    tactile_arrays[f"{name}_{channel}"] = np.asarray(value)
                    channels.append(channel)
            tactile_refs[name] = {
                "valid": sample["valid"],
                "frame_id": sample["frame_id"],
                "timestamp_wall": sample["timestamp_wall"],
                "timestamp_monotonic": sample["timestamp_monotonic"],
                "age_s": sample["age_s"],
                "channels": channels,
            }

        tactile_file: Optional[str] = None
        if tactile_arrays:
            relative_path = Path("tactile") / f"{self.num_steps:06d}.npz"
            np.savez(self.episode_dir / relative_path, **tactile_arrays)
            tactile_file = relative_path.as_posix()

        now_wall = time.time()
        now_mono = time.monotonic()
        action = self.action_collector.collect(franka_sample)
        step = {
            "step": self.num_steps,
            "timestamp_wall": now_wall,
            "timestamp_monotonic": now_mono,
            "elapsed_s": now_mono - self.started_mono,
            "valid": {
                "cameras": True,
                "tactile": tactile_sample["valid"],
                "franka": franka_sample["valid"],
                "action": action["valid"],
            },
            "cameras": camera_refs,
            "tactile": {
                "valid": tactile_sample["valid"],
                "file": tactile_file,
                "sensors": tactile_refs,
            },
            "franka": franka_sample,
            "action": action,
        }
        self.steps_file.write(
            json.dumps(_jsonable(step), ensure_ascii=False, separators=(",", ":"))
            + "\n"
        )
        ee = franka_sample["ee"]
        gripper = franka_sample["gripper"]
        print(
            f"[Record][Step {self.num_steps:06d}] "
            f"ee_pos={_jsonable(ee.get('position'))} "
            f"ee_quat_xyzw={_jsonable(ee.get('quaternion_xyzw'))} "
            f"gripper_width={gripper.get('width')} "
            f"action={_jsonable(action.get('vector'))} "
            f"gripper_state={_jsonable(gripper.get('state'))}"
        )
        self.num_steps += 1
        if self.num_steps % 10 == 0:
            self.steps_file.flush()

    def close(self, success: Optional[bool] = None) -> None:
        for writer in self.video_writers.values():
            try:
                writer.release()
            except Exception:
                pass
        self.video_writers.clear()
        self.steps_file.flush()
        self.steps_file.close()
        self.metadata["success"] = success
        self.metadata["num_steps"] = self.num_steps
        self.metadata["duration_s"] = time.monotonic() - self.started_mono
        self._write_metadata()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="datasets")
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--wrist-serial", default="922612070441")
    parser.add_argument("--third-serial", default="143322074106")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", "--high", dest="height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--camera-timeout-ms", type=int, default=10000)
    parser.add_argument("--camera-read-timeout-ms", type=int, default=1500)
    parser.add_argument("--camera-warmup-frames", type=int, default=10)
    parser.add_argument("--camera-start-retries", type=int, default=3)
    parser.add_argument("--camera-retry-sleep-s", type=float, default=1.0)
    parser.add_argument("--tactile-left-id", default="L25480053")
    parser.add_argument("--tactile-right-id", default="L25480055")
    parser.add_argument("--tactile-remote-addr", default="192.168.127.10:50051")
    parser.add_argument("--tactile-pc-host", default="192.168.127.99")
    parser.add_argument("--tactile-left-port", type=int, default=60001)
    parser.add_argument("--tactile-right-port", type=int, default=60002)
    parser.add_argument("--tactile-backend", default="CUDA")
    parser.add_argument("--tactile-max-fps", type=int, default=30)
    parser.add_argument("--fr3-host", default="192.168.20.8")
    parser.add_argument("--ee-port", type=int, default=5556)
    parser.add_argument("--gripper-port", type=int, default=5558)
    parser.add_argument("--fr3-timeout-s", type=float, default=2.0)
    parser.add_argument("--fr3-state-poll-hz", type=float, default=30.0)
    parser.add_argument("--fr3-state-max-age-s", type=float, default=0.5)
    return parser.parse_args()



def make_preview(image: np.ndarray, name: str, recording: bool) -> np.ndarray:
    preview = image.copy()
    status = "REC" if recording else "IDLE"
    color = (0, 0, 255) if recording else (0, 255, 0)
    cv2.putText(
        preview,
        f"{name}  {status}  [S] record  [Q] quit",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return preview


def main() -> int:
    args = parse_args()

    tactile_configs = (
        {
            "name": "left",
            "dev_id": args.tactile_left_id,
            "pc_port": args.tactile_left_port,
        },
        {
            "name": "right",
            "dev_id": args.tactile_right_id,
            "pc_port": args.tactile_right_port,
        },
    )
    tactile = DualTactileSource(
        sensor_configs=tactile_configs,
        backend=args.tactile_backend,
        remote_addr=args.tactile_remote_addr,
        pc_host=args.tactile_pc_host,
        max_fps=args.tactile_max_fps,
        startup_timeout_s=30.0,
        max_age_s=1.0,
    )
    franka = FrankaStateSource(
        host=args.fr3_host,
        ee_port=args.ee_port,
        gripper_port=args.gripper_port,
        timeout_s=args.fr3_timeout_s,
        poll_hz=args.fr3_state_poll_hz,
        max_age_s=args.fr3_state_max_age_s,
    )
    camera_configs = (
        {"name": "wrist", "serial": args.wrist_serial},
        {"name": "third", "serial": args.third_serial},
    )
    cameras = [
        RealSenseSource(
            name=config["name"],
            serial=config["serial"],
            width=args.width,
            height=args.height,
            fps=args.fps,
            startup_timeout_ms=args.camera_timeout_ms,
            read_timeout_ms=args.camera_read_timeout_ms,
            warmup_frames=args.camera_warmup_frames,
            start_retries=args.camera_start_retries,
            retry_sleep_s=args.camera_retry_sleep_s,
        )
        for config in camera_configs
    ]
    writer: Optional[EpisodeWriter] = None
    keyboard = TerminalKeyboard()

    def start_recording() -> EpisodeWriter:
        metadata = {
            "camera": {
                "type": "Intel RealSense color stream",
                "devices": camera_configs,
                "resolution": [args.width, args.height],
                "fps": args.fps,
            },
            "tactile": {
                "devices": tactile_configs,
                "channels": ["deformation", "depth"],
                "backend": args.tactile_backend,
                "max_fps": args.tactile_max_fps,
                "save_fps": args.fps,
                "remote_addr": args.tactile_remote_addr,
                "pc_host": args.tactile_pc_host,
            },
            "franka": {
                "host": args.fr3_host,
                "ee_port": args.ee_port,
                "gripper_port": args.gripper_port,
                "state_poll_hz": args.fr3_state_poll_hz,
                "state_max_age_s": args.fr3_state_max_age_s,
                "ee_quaternion_order": "xyzw",
                "position_unit": "meter",
                "gripper_width_unit": "meter",
            },
            "action": {
                "dimension": 7,
                "vector_order": [
                    "delta_x", "delta_y", "delta_z",
                    "delta_rx", "delta_ry", "delta_rz",
                    "gripper_width",
                ],
                "translation_frame": "fr3_base",
                "rotation_representation": "axis_angle",
                "translation_unit": "meter",
                "rotation_unit": "radian",
                "gripper_mode": "absolute_width",
                "gripper_width_unit": "meter",
                "alignment": "motion_from_previous_step_to_current_step",
            },
        }
        episode = EpisodeWriter(
            dataset_dir=args.dataset_dir,
            camera_names=[camera.name for camera in cameras],
            camera_fps=args.fps,
            metadata=metadata,
        )
        print(f"[Record] STARTED: {episode.episode_dir}")
        return episode

    def toggle_recording() -> None:
        nonlocal writer
        if writer is None:
            writer = start_recording()
            return
        finished = writer
        writer = None
        finished.close(success=None)
        print(
            f"[Record] STOPPED: {finished.num_steps} steps, "
            f"saved to {finished.episode_dir}"
        )

    try:
        # Spawn tactile workers before starting RealSense C++ background threads.
        tactile.open()
        franka.open()
        opened_cameras: List[RealSenseSource] = []
        try:
            for camera in cameras:
                camera.open()
                opened_cameras.append(camera)
        except Exception:
            for camera in reversed(opened_cameras):
                camera.close()
            raise

        keyboard.open()
        print(
            "[Keys] S: recording ON/OFF | Q/Esc: exit "
            "(terminal or OpenCV window)"
        )
        if args.record:
            writer = start_recording()

        while True:
            loop_started = time.perf_counter()

            terminal_keys = [key.lower() for key in keyboard.poll()]
            if "q" in terminal_keys or "\x1b" in terminal_keys:
                print("[Main] Exit requested from terminal.")
                break
            if "s" in terminal_keys:
                toggle_recording()

            camera_samples = {camera.name: camera.read() for camera in cameras}
            tactile_sample = tactile.read()
            franka_sample = franka.read()

            if writer is not None:
                writer.write_step(camera_samples, tactile_sample, franka_sample)

            for name, sample in camera_samples.items():
                cv2.imshow(
                    f"RealSense {name}",
                    make_preview(sample["image"], name, writer is not None),
                )
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("[Main] Exit requested from OpenCV window.")
                break
            if key == ord("s"):
                toggle_recording()

            period_s = 1.0 / args.fps
            sleep_s = period_s - (time.perf_counter() - loop_started)
            if sleep_s > 0.0:
                time.sleep(sleep_s)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted.")
    finally:
        keyboard.close()
        if writer is not None:
            writer.close(success=None)
            print(
                f"[Record] STOPPED: {writer.num_steps} steps, "
                f"saved to {writer.episode_dir}"
            )
        for camera in reversed(cameras):
            camera.close()
        cv2.destroyAllWindows()
        franka.close()
        tactile.close()
        print("[Main] All devices closed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
