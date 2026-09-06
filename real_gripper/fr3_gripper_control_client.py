#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TCP client for run_fr3_gripper_controller_server.py.

Examples:
python fr3_gripper_control_client.py --host 192.168.20.8 --port 5555 --ping
python fr3_gripper_control_client.py --host 192.168.20.8 --port 5555 --open
python fr3_gripper_control_client.py --host 192.168.20.8 --port 5555 --close
python fr3_gripper_control_client.py --host 192.168.20.8 --port 5555 --width 0.04

By default command messages are sent as:
  {"hand_qpos":[width]}

If your ControllerServer expects another key, use for example:
  --command_key gripper_width
  --command_key gripper
"""

import argparse
import json
import socket
import threading
import time
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple


def recv_line(
    sock: socket.socket,
    buf: bytearray,
    timeout_s: float,
) -> Tuple[Optional[str], bytearray]:
    old_timeout = sock.gettimeout()
    sock.settimeout(timeout_s)
    try:
        while True:
            nl = buf.find(b"\n")
            if nl != -1:
                line = bytes(buf[:nl]).decode("utf-8", errors="replace").strip()
                del buf[: nl + 1]
                if not line:
                    continue
                return line, buf

            try:
                chunk = sock.recv(4096)
            except socket.timeout as e:
                raise TimeoutError(
                    f"recv_line timeout after {timeout_s:.2f}s, buffered={len(buf)} bytes"
                ) from e

            if not chunk:
                return None, buf
            buf.extend(chunk)
    finally:
        sock.settimeout(old_timeout)


def send_json(sock: socket.socket, obj: dict) -> None:
    sock.sendall((json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8"))


def clip_width(width: float, min_width: float, max_width: float, mode: str) -> float:
    width = float(width)
    if min_width <= width <= max_width:
        return width

    clipped = min(max(width, min_width), max_width)
    msg = (
        f"gripper width {width:.6f} is outside "
        f"[{min_width:.6f}, {max_width:.6f}]"
    )
    if mode == "raise":
        raise ValueError(msg)
    if mode == "warn_clip":
        print(f"[WARNING] {msg}; clipped -> {clipped:.6f}")
    return clipped


class FR3GripperClient:
    def __init__(
        self,
        host: str,
        port: int = 5555,
        timeout_s: float = 5.0,
        retry_times: int = 1,
        retry_wait_s: float = 0.05,
        command_key: str = "hand_qpos",
        protocol: str = "zmq_torch",
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.retry_times = int(retry_times)
        self.retry_wait_s = float(retry_wait_s)
        self.command_key = command_key
        self.protocol = protocol

        self.cmd_sock: Optional[socket.socket] = None
        self.rpc_sock: Optional[socket.socket] = None
        self.rpc_buf = bytearray()
        self.zmq_context = None
        self.zmq_sock = None

        self._cmd_lock = threading.RLock()
        self._rpc_lock = threading.RLock()

        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_hz: float = 20.0

        self._state_lock = threading.Lock()
        self._cached_state: Optional[Dict[str, Any]] = None
        self._cached_state_recv_mono: float = 0.0
        self._last_poll_error_t: float = 0.0

    def _create_socket(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        return sock

    def _ensure_zmq_socket(self):
        if self.zmq_sock is not None:
            return

        import zmq

        self.zmq_context = zmq.Context()
        self.zmq_sock = self.zmq_context.socket(zmq.REQ)
        self.zmq_sock.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))
        self.zmq_sock.setsockopt(zmq.SNDTIMEO, int(self.timeout_s * 1000))
        self.zmq_sock.connect(f"tcp://{self.host}:{self.port}")

    @staticmethod
    def _torch_to_bytes(data: dict) -> bytes:
        import torch

        buffer = BytesIO()
        torch.save(data, buffer)
        return buffer.getvalue()

    @staticmethod
    def _torch_from_bytes(data: bytes) -> dict:
        import torch

        buffer = BytesIO(data)
        return torch.load(buffer, weights_only=False)

    def _request_zmq(self, endpoint: str, data: Optional[dict] = None) -> dict:
        import zmq

        last_err = None

        for attempt in range(self.retry_times + 1):
            try:
                self._ensure_zmq_socket()
                req: Dict[str, Any] = {"endpoint": endpoint}
                if data is not None:
                    req["data"] = data

                self.zmq_sock.send(self._torch_to_bytes(req))
                resp = self._torch_from_bytes(self.zmq_sock.recv())
                if "error" in resp:
                    raise RuntimeError(f"server error: {resp['error']}")
                return resp
            except zmq.Again as e:
                last_err = TimeoutError(
                    f"timeout waiting for ZMQ reply from {self.host}:{self.port} "
                    f"endpoint={endpoint} after {self.timeout_s:.2f}s"
                )
                self._close_zmq()
                if attempt < self.retry_times:
                    time.sleep(self.retry_wait_s)
            except Exception as e:
                last_err = e
                self._close_zmq()
                if attempt < self.retry_times:
                    time.sleep(self.retry_wait_s)

        raise RuntimeError(
            f"zmq request failed after retries, endpoint={endpoint}, err={last_err}"
        )

    def _close_zmq(self):
        if self.zmq_sock is not None:
            try:
                self.zmq_sock.close(linger=0)
            except Exception:
                pass
        self.zmq_sock = None

        if self.zmq_context is not None:
            try:
                self.zmq_context.term()
            except Exception:
                pass
        self.zmq_context = None

    def connect(self):
        if self.protocol == "zmq_torch":
            self._ensure_zmq_socket()
            return self

        with self._cmd_lock:
            if self.cmd_sock is None:
                self.cmd_sock = self._create_socket()

        with self._rpc_lock:
            if self.rpc_sock is None:
                self.rpc_sock = self._create_socket()
                self.rpc_buf = bytearray()

        return self

    def connect_command(self):
        if self.protocol == "zmq_torch":
            self._ensure_zmq_socket()
            return self

        with self._cmd_lock:
            if self.cmd_sock is None:
                self.cmd_sock = self._create_socket()
        return self

    def _close_cmd_locked(self):
        if self.cmd_sock is not None:
            try:
                self.cmd_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.cmd_sock.close()
            except Exception:
                pass
        self.cmd_sock = None

    def _close_rpc_locked(self):
        if self.rpc_sock is not None:
            try:
                self.rpc_sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
            try:
                self.rpc_sock.close()
            except Exception:
                pass
        self.rpc_sock = None
        self.rpc_buf = bytearray()

    def _reconnect_cmd_locked(self):
        self._close_cmd_locked()
        self.cmd_sock = self._create_socket()

    def _reconnect_rpc_locked(self):
        self._close_rpc_locked()
        self.rpc_sock = self._create_socket()
        self.rpc_buf = bytearray()

    def close(self):
        self.stop_state_poller()
        self._close_zmq()
        with self._cmd_lock:
            self._close_cmd_locked()
        with self._rpc_lock:
            self._close_rpc_locked()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def _make_gripper_request(self, width: float) -> dict:
        if self.command_key in ("hand_qpos", "qpos", "q"):
            return {self.command_key: [float(width)]}
        return {self.command_key: float(width)}

    def send_gripper_width(self, width: float) -> Optional[dict]:
        if self.protocol == "zmq_torch":
            import torch

            width_tensor = torch.tensor([float(width)], dtype=torch.float32)
            return self._request_zmq(
                "send_action",
                {
                    "arm_action": torch.zeros(7, dtype=torch.float32),
                    "hand_action": width_tensor,
                },
            )
            return

        req = self._make_gripper_request(float(width))
        last_err = None

        for attempt in range(self.retry_times + 1):
            with self._cmd_lock:
                try:
                    if self.cmd_sock is None:
                        self._reconnect_cmd_locked()
                    assert self.cmd_sock is not None
                    send_json(self.cmd_sock, req)
                    return None
                except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError) as e:
                    last_err = e
                    try:
                        self._reconnect_cmd_locked()
                    except Exception as reconnect_err:
                        last_err = reconnect_err

            if attempt < self.retry_times:
                time.sleep(self.retry_wait_s)

        raise RuntimeError(f"send_gripper_width failed after retries, err={last_err}")

    def open(self, width: float = 0.08) -> Optional[dict]:
        return self.send_gripper_width(width)

    def close_gripper(self, width: float = 0.0) -> Optional[dict]:
        return self.send_gripper_width(width)

    def send_normalized(
        self,
        value: float,
        min_width: float = 0.0,
        max_width: float = 0.08,
    ) -> float:
        value = min(max(float(value), 0.0), 1.0)
        width = min_width + value * (max_width - min_width)
        self.send_gripper_width(width)
        return width

    def _request_json_rpc(self, req: dict, expect_reply: bool = True) -> Optional[dict]:
        last_err = None

        for attempt in range(self.retry_times + 1):
            with self._rpc_lock:
                try:
                    if self.rpc_sock is None:
                        self._reconnect_rpc_locked()

                    assert self.rpc_sock is not None
                    send_json(self.rpc_sock, req)

                    if not expect_reply:
                        return None

                    line, self.rpc_buf = recv_line(
                        self.rpc_sock,
                        self.rpc_buf,
                        timeout_s=self.timeout_s,
                    )
                    if not line:
                        raise RuntimeError("peer closed connection")

                    return json.loads(line)

                except (
                    TimeoutError,
                    socket.timeout,
                    ConnectionResetError,
                    BrokenPipeError,
                    OSError,
                    json.JSONDecodeError,
                ) as e:
                    last_err = e
                    try:
                        self._reconnect_rpc_locked()
                    except Exception as reconnect_err:
                        last_err = reconnect_err

            if attempt < self.retry_times:
                time.sleep(self.retry_wait_s)

        raise RuntimeError(f"rpc request failed after retries, req={req}, err={last_err}")

    def get_state(self) -> dict:
        if self.protocol == "zmq_torch":
            return self._request_zmq("read_state")

        resp = self._request_json_rpc({"cmd": "get"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad get resp: {resp}")
        return resp

    def get_attributes(self) -> dict:
        if self.protocol == "zmq_torch":
            return self._request_zmq("get_attributes")
        raise RuntimeError("get_attributes is only supported by zmq_torch protocol")

    def ping(self) -> dict:
        if self.protocol == "zmq_torch":
            return self._request_zmq("ping")

        resp = self._request_json_rpc({"cmd": "ping"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad ping resp: {resp}")
        return resp

    def stop(self) -> None:
        if self.protocol == "zmq_torch":
            return

        resp = self._request_json_rpc({"cmd": "stop"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad stop resp: {resp}")

    def quit_server(self) -> None:
        resp = self._request_json_rpc({"cmd": "quit"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad quit resp: {resp}")

    def start_state_poller(self, poll_hz: float = 20.0):
        self.connect()
        self._poll_hz = float(poll_hz)

        if self._poll_thread is not None and self._poll_thread.is_alive():
            return

        self._poll_stop.clear()
        self._poll_thread = threading.Thread(target=self._state_poll_loop, daemon=True)
        self._poll_thread.start()

    def stop_state_poller(self):
        self._poll_stop.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.0)
        self._poll_thread = None

    def _state_poll_loop(self):
        dt = 1.0 / max(self._poll_hz, 1e-6)
        next_t = time.monotonic()

        while not self._poll_stop.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(next_t - now)
                continue
            next_t += dt

            try:
                st = self.get_state()
                with self._state_lock:
                    self._cached_state = st
                    self._cached_state_recv_mono = time.monotonic()
            except Exception as e:
                now2 = time.monotonic()
                if (now2 - self._last_poll_error_t) > 1.0:
                    print(f"[FR3GripperStatePoller] warn: get_state failed: {e}")
                    self._last_poll_error_t = now2
                next_t = time.monotonic() + dt

    def wait_until_state_ready(self, timeout_s: float = 2.0) -> bool:
        t0 = time.monotonic()
        while (time.monotonic() - t0) < timeout_s:
            with self._state_lock:
                ready = self._cached_state is not None
            if ready:
                return True
            time.sleep(0.01)
        return False

    def get_state_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[dict, float]:
        with self._state_lock:
            st = None if self._cached_state is None else dict(self._cached_state)
            recv_t = float(self._cached_state_recv_mono)

        if st is not None:
            age_s = max(0.0, time.monotonic() - recv_t)
            if (max_age_s is None) or (age_s <= max_age_s):
                return st, age_s

        if fallback_to_sync:
            st = self.get_state()
            with self._state_lock:
                self._cached_state = st
                self._cached_state_recv_mono = time.monotonic()
            return st, 0.0

        raise RuntimeError("cached FR3 gripper state unavailable or too stale")

    def get_gripper_width_from_state(self, state: dict) -> Optional[float]:
        for key in ("gripper_width", "width", "hand_width", "gripper"):
            if key in state and state[key] is not None:
                return float(state[key])

        for key in ("hand_qpos", "qpos", "q"):
            value = state.get(key, None)
            if value is not None and len(value) > 0:
                return float(value[0])

        return None

    def get_gripper_width_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[float, float]:
        st, age = self.get_state_cached(
            max_age_s=max_age_s,
            fallback_to_sync=fallback_to_sync,
        )
        width = self.get_gripper_width_from_state(st)
        if width is None:
            raise RuntimeError(f"state has no recognized gripper width field: {st}")
        return width, age


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", type=str, default="192.168.20.8")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--timeout_s", type=float, default=2.0)
    ap.add_argument("--retry_times", type=int, default=1)
    ap.add_argument("--retry_wait_s", type=float, default=0.05)
    ap.add_argument("--command_key", type=str, default="hand_qpos")
    ap.add_argument(
        "--protocol",
        type=str,
        default="zmq_torch",
        choices=["zmq_torch", "json_tcp"],
        help="Protocol used by the gripper controller server.",
    )
    ap.add_argument("--min_width", type=float, default=0.0)
    ap.add_argument("--max_width", type=float, default=0.08)
    ap.add_argument(
        "--clip_mode",
        type=str,
        default="raise",
        choices=["raise", "clip", "warn_clip"],
    )
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--get", action="store_true")
    ap.add_argument("--attrs", action="store_true")
    ap.add_argument("--open", action="store_true")
    ap.add_argument("--close", action="store_true")
    ap.add_argument("--width", type=float, default=None)
    ap.add_argument(
        "--normalized",
        type=float,
        default=None,
        help="Send normalized opening in [0, 1], mapped to min/max width.",
    )
    ap.add_argument("--demo", action="store_true", help="Open, wait, then close.")
    ap.add_argument("--demo_wait_s", type=float, default=1.0)
    ap.add_argument(
        "--print_response",
        action="store_true",
        help="Print command response when the selected protocol returns one.",
    )
    ap.add_argument(
        "--post_send_wait_s",
        type=float,
        default=0.1,
        help="Short wait after command-only sends before closing the TCP socket.",
    )
    args = ap.parse_args()

    c = FR3GripperClient(
        host=args.host,
        port=args.port,
        timeout_s=args.timeout_s,
        retry_times=args.retry_times,
        retry_wait_s=args.retry_wait_s,
        command_key=args.command_key,
        protocol=args.protocol,
    )
    try:
        if args.ping or args.get or args.attrs:
            c.connect()
        else:
            c.connect_command()

        if args.ping:
            print("[ping ]", c.ping())

        if args.get:
            st = c.get_state()
            print("[state]", st)
            width = c.get_gripper_width_from_state(st)
            if width is not None:
                print("[width]", width)

        if args.attrs:
            print("[attrs]", c.get_attributes())

        if args.demo:
            print(f"[send ] open width={args.max_width:.6f}")
            resp = c.open(args.max_width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            time.sleep(args.demo_wait_s)
            print(f"[send ] close width={args.min_width:.6f}")
            resp = c.close_gripper(args.min_width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            return

        if args.open:
            width = clip_width(args.max_width, args.min_width, args.max_width, args.clip_mode)
            print(f"[send ] open width={width:.6f}")
            resp = c.open(width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            time.sleep(args.post_send_wait_s)

        if args.close:
            width = clip_width(args.min_width, args.min_width, args.max_width, args.clip_mode)
            print(f"[send ] close width={width:.6f}")
            resp = c.close_gripper(width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            time.sleep(args.post_send_wait_s)

        if args.width is not None:
            width = clip_width(args.width, args.min_width, args.max_width, args.clip_mode)
            print(f"[send ] width={width:.6f}")
            resp = c.send_gripper_width(width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            time.sleep(args.post_send_wait_s)

        if args.normalized is not None:
            value = min(max(float(args.normalized), 0.0), 1.0)
            width = args.min_width + value * (args.max_width - args.min_width)
            print(f"[send ] normalized={value:.6f}, width={width:.6f}")
            resp = c.send_gripper_width(width)
            if args.print_response and resp is not None:
                print("[resp ]", resp)
            time.sleep(args.post_send_wait_s)
    finally:
        c.close()


if __name__ == "__main__":
    main()
