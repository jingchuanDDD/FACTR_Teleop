#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
FR3 joint control client.

功能：
1. 两条 TCP 连接分离
   - cmd_sock: 只发 {"q":[...]}，不等回复
   - rpc_sock: 只做 get/ping/stop/quit 等 request-reply

2. 支持后台状态轮询线程
   - start_state_poller()
   - get_state_cached()
   - get_joint_state_cached()

3. 支持平滑插值到目标关节角
   - move_to_q()

4. 支持三种模式：
   MODE = "status" : 只测试 ping/get_state
   MODE = "delta"  : 保留 q_des[3] += 0.1 这种通信测试
   MODE = "move"   : 直接给 TARGET_Q，然后平滑运动过去
"""

import json
import socket
import threading
import time
from typing import Optional, Tuple, List, Dict, Any, Sequence

import numpy as np

HOST = "192.168.50.2"
PORT = 5558
TIMEOUT_S = 2.0
RETRY_TIMES = 1

# 可选: "status", "delta", "move"
MODE = "move"

# ------------------------------------------------------------
# delta 模式：
# 从当前 q 出发，让某一个关节加一个小量，用来测试通信是否正常
# ------------------------------------------------------------
DELTA_JOINT_IDX = 3       # 0-based index; 3 表示 fr3_joint4
DELTA_RAD = 0.1           # q_des[3] = q0[3] + 0.1

# ------------------------------------------------------------
# move 模式：
# 直接给一个目标 q，程序会从当前 q 平滑插值过去
# ------------------------------------------------------------
TARGET_Q = np.array(
[-0.0797635689, -0.3948985040, 0.0658895746, -2.8056068420, 0.6789462566, 2.1837496758, 0.3903819323] # left side good cube pick
# [-0.2189386189, 0.3782366216, -0.6227233410, -2.0949795246, 0.0098963678, 2.6277432442, -2.1655480862] # right side
# [-0.0713522434, 0.3144021034, -0.7206720710, -1.9965538979, -0.0675622225, 2.3838295937, -2.0066637993] # right side higher tea picking
# [0.0976656303, -0.2882348597, -0.1106102094, -2.8132240772, 0.5152372122, 2.2561635971, 0.3973218501] # left side
# [0.4828683436, -0.3675746024, -0.6234995127, -2.7528100014, -0.2580889761, 2.2366662025, 0.2724474370] # mid side
# [-0.0800793543, -0.8847529888, -0.3024094403, -2.3091156483, 0.0410998538, 0.9188706279, 0.1403556615] # mid side closer
#  [-0.0228388999, -0.5265679359, -0.2807765007, -2.1093463898, 0.1291255653, 1.0906820297, 0.2001585513] # mid side higer to table
# [0.0951382741, 0.2137674838, -0.9429902434, -2.0333356857, -0.1329324841, 2.4110217094, -2.0914480686] # higher right side
,dtype=np.float64,
)

# 如果你想运行后在 terminal 手动输入 q，就设成 True。
# MODE = "move" 时生效。
INPUT_Q = False

# ------------------------------------------------------------
# 平滑运动参数
# ------------------------------------------------------------
DURATION_S = 3.0
HZ = 100.0

# None 表示默认每个关节 30 deg/s。
# 也可以写成一个数字，例如：
# MAX_JOINT_VEL_DEG_S = 15.0
# 或者每个关节单独设置：
# MAX_JOINT_VEL_DEG_S = [15, 15, 15, 15, 15, 15, 15]
MAX_JOINT_VEL_DEG_S = None

CURRENT_Q_MAX_AGE_S = 0.5
POLL_HZ = 50.0
WARN_STALE_AFTER_S = 0.2

# True 表示只打印，不真的 send_q
DRY_RUN = False


# ============================================================
# Utility functions
# ============================================================

def recv_line(sock: socket.socket, buf: bytearray, timeout_s: float) -> Tuple[Optional[str], bytearray]:
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


def format_q(q: Sequence[float], ndigits: int = 6) -> str:
    return "[" + ", ".join(f"{float(v):+.{ndigits}f}" for v in q) + "]"


def parse_q_from_text(text: str) -> np.ndarray:
    """
    支持这些格式：
      -0.1 0.2 -0.3 -2.0 0.1 2.3 -1.5
      [-0.1, 0.2, -0.3, -2.0, 0.1, 2.3, -1.5]
      -0.1,0.2,-0.3,-2.0,0.1,2.3,-1.5
    """
    s = text.strip()
    s = s.replace("[", " ").replace("]", " ")
    s = s.replace("(", " ").replace(")", " ")
    s = s.replace(",", " ")

    vals = [float(x) for x in s.split()]
    q = np.asarray(vals, dtype=np.float64).reshape(-1)

    if q.shape != (7,):
        raise ValueError(f"target q must have 7 values, got shape={q.shape}, q={q}")

    if not np.isfinite(q).all():
        raise ValueError(f"target q contains NaN/Inf: {q}")

    return q


def build_vmax_from_config(max_joint_vel_deg_s) -> Optional[np.ndarray]:
    """
    解析最大关节速度。

    支持：
      MAX_JOINT_VEL_DEG_S = None
      MAX_JOINT_VEL_DEG_S = 30.0
      MAX_JOINT_VEL_DEG_S = [20, 20, 20, 20, 20, 20, 20]
    """
    if max_joint_vel_deg_s is None:
        return None

    vals = np.asarray(max_joint_vel_deg_s, dtype=np.float64).reshape(-1)

    if vals.shape == (1,):
        vals = np.repeat(vals[0], 7)

    if vals.shape != (7,):
        raise ValueError(
            "MAX_JOINT_VEL_DEG_S must be either one value or seven values. "
            f"Got shape={vals.shape}, vals={vals}"
        )

    if not np.isfinite(vals).all() or np.any(vals <= 0):
        raise ValueError(f"invalid MAX_JOINT_VEL_DEG_S: {vals}")

    return np.deg2rad(vals)


def get_target_q_from_config() -> Optional[np.ndarray]:
    if INPUT_Q:
        print("\n请输入目标 q，单位 rad，7 个数。")
        print("支持格式示例：")
        print("  -0.1 0.2 -0.3 -2.0 0.1 2.3 -1.5")
        print("  [-0.1, 0.2, -0.3, -2.0, 0.1, 2.3, -1.5]")
        text = input("target_q > ")
        return parse_q_from_text(text)

    if TARGET_Q is None:
        return None

    q = np.asarray(TARGET_Q, dtype=np.float64).reshape(-1)
    if q.shape != (7,):
        raise ValueError(f"TARGET_Q expects 7 values, got shape={q.shape}, q={q}")
    if not np.isfinite(q).all():
        raise ValueError(f"TARGET_Q contains NaN/Inf: {q}")

    return q


# ============================================================
# FR3 client
# ============================================================

class FR3JointClient:
    def __init__(
        self,
        host: str,
        port: int,
        timeout_s: float = 5.0,
        retry_times: int = 1,
        retry_wait_s: float = 0.05,
    ):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.retry_times = int(retry_times)
        self.retry_wait_s = float(retry_wait_s)

        self.cmd_sock: Optional[socket.socket] = None
        self.rpc_sock: Optional[socket.socket] = None
        self.rpc_buf = bytearray()

        self._cmd_lock = threading.RLock()
        self._rpc_lock = threading.RLock()

        self._poll_thread: Optional[threading.Thread] = None
        self._poll_stop = threading.Event()
        self._poll_hz: float = 20.0
        self._warn_stale_after_s: float = 0.2

        self._state_lock = threading.Lock()
        self._cached_state: Optional[Dict[str, Any]] = None
        self._cached_state_recv_mono: float = 0.0
        self._last_poll_error_t: float = 0.0

    def _create_socket(self) -> socket.socket:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        return sock

    def connect(self):
        with self._cmd_lock:
            if self.cmd_sock is None:
                self.cmd_sock = self._create_socket()

        with self._rpc_lock:
            if self.rpc_sock is None:
                self.rpc_sock = self._create_socket()
                self.rpc_buf = bytearray()

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
        with self._cmd_lock:
            self._close_cmd_locked()
        with self._rpc_lock:
            self._close_rpc_locked()

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    # ---------------------------
    # command socket: q only
    # ---------------------------

    def send_q(self, q: List[float]) -> None:
        req = {"q": q}
        last_err = None

        for attempt in range(self.retry_times + 1):
            with self._cmd_lock:
                try:
                    if self.cmd_sock is None:
                        self._reconnect_cmd_locked()
                    assert self.cmd_sock is not None
                    send_json(self.cmd_sock, req)
                    return
                except (socket.timeout, ConnectionResetError, BrokenPipeError, OSError) as e:
                    last_err = e
                    try:
                        self._reconnect_cmd_locked()
                    except Exception as reconnect_err:
                        last_err = reconnect_err

            if attempt < self.retry_times:
                time.sleep(self.retry_wait_s)

        raise RuntimeError(f"send_q failed after retries, err={last_err}")

    # ---------------------------
    # rpc socket
    # ---------------------------

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

                    resp = json.loads(line)
                    return resp

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
        resp = self._request_json_rpc({"cmd": "get"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad get resp: {resp}")
        return resp

    def stop(self) -> None:
        resp = self._request_json_rpc({"cmd": "stop"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad stop resp: {resp}")

    def ping(self) -> dict:
        resp = self._request_json_rpc({"cmd": "ping"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad ping resp: {resp}")
        return resp

    def quit_server(self) -> None:
        resp = self._request_json_rpc({"cmd": "quit"}, expect_reply=True)
        if not resp.get("ok", False):
            raise RuntimeError(f"bad quit resp: {resp}")

    # ---------------------------
    # background state poller
    # ---------------------------

    def start_state_poller(self, poll_hz: float = 20.0, warn_stale_after_s: float = 0.2):
        self.connect()

        self._poll_hz = float(poll_hz)
        self._warn_stale_after_s = float(warn_stale_after_s)

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
                recv_t = time.monotonic()

                with self._state_lock:
                    self._cached_state = st
                    self._cached_state_recv_mono = recv_t

            except Exception as e:
                now2 = time.monotonic()
                if (now2 - self._last_poll_error_t) > 1.0:
                    print(f"[FR3StatePoller] warn: get_state failed: {e}")
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
            age_s = 0.0
            with self._state_lock:
                self._cached_state = st
                self._cached_state_recv_mono = time.monotonic()
            return st, age_s

        raise RuntimeError("cached FR3 state unavailable or too stale")

    def get_joint_state_cached(
        self,
        max_age_s: Optional[float] = None,
        fallback_to_sync: bool = False,
    ) -> Tuple[np.ndarray, float]:
        st, age = self.get_state_cached(
            max_age_s=max_age_s,
            fallback_to_sync=fallback_to_sync,
        )
        q = st.get("q", None)
        if q is None:
            raise RuntimeError(f"cached state has no q: {st}")

        return np.asarray(q, dtype=np.float64), age

    # ---------------------------
    # smooth joint motion
    # ---------------------------

    @staticmethod
    def _smooth_quintic(t: float) -> float:
        t = float(np.clip(t, 0.0, 1.0))
        return t * t * t * (10.0 - 15.0 * t + 6.0 * t * t)

    @staticmethod
    def _validate_q_vec(x, name: str = "q") -> np.ndarray:
        q = np.asarray(x, dtype=np.float64).reshape(-1)
        if q.shape != (7,):
            raise ValueError(f"{name} shape invalid: got {q.shape}, expected (7,)")
        if not np.isfinite(q).all():
            raise ValueError(f"{name} contains NaN/Inf: {q}")
        return q

    def move_to_q(
        self,
        target_q: Sequence[float],
        duration_s: float = 1.0,
        hz: float = 100.0,
        max_joint_vel_rad_s: Optional[Sequence[float]] = None,
        current_q_max_age_s: float = 0.5,
        fallback_to_sync: bool = True,
        atol: float = 1e-4,
        final_hold: bool = True,
        verbose: bool = True,
    ) -> float:
        target_q = self._validate_q_vec(target_q, name="target_q")

        q0, state_age = self.get_joint_state_cached(
            max_age_s=current_q_max_age_s,
            fallback_to_sync=fallback_to_sync,
        )
        q0 = self._validate_q_vec(q0, name="current_q")

        if np.allclose(q0, target_q, atol=atol, rtol=0.0):
            if final_hold:
                self.send_q(target_q.tolist())
            if verbose:
                print("[move_to_q] already near target, skip interpolation.")
            return 0.0

        if max_joint_vel_rad_s is None:
            vmax = np.deg2rad(np.array([30.0] * 7, dtype=np.float64))
        else:
            vmax = np.asarray(max_joint_vel_rad_s, dtype=np.float64).reshape(-1)
            if vmax.shape != (7,):
                raise ValueError(
                    f"max_joint_vel_rad_s shape invalid: got {vmax.shape}, expected (7,)"
                )
            if not np.isfinite(vmax).all() or np.any(vmax <= 0):
                raise ValueError(f"max_joint_vel_rad_s invalid: {vmax}")

        hz = float(hz)
        if hz <= 0:
            raise ValueError(f"hz must be > 0, got {hz}")

        dq = target_q - q0

        duration_min = float(np.max(np.abs(dq) / np.maximum(vmax, 1e-6)))
        duration = max(float(duration_s), duration_min, 0.2)

        n = max(int(round(duration * hz)), 1)
        dt = 1.0 / hz

        if verbose:
            print("[move_to_q] current_q [rad] =", format_q(q0, 6))
            print("[move_to_q] target_q  [rad] =", format_q(target_q, 6))
            print("[move_to_q] delta_q   [rad] =", format_q(dq, 6))
            print("[move_to_q] max |dq| [rad] =", float(np.max(np.abs(dq))))
            print(
                f"[move_to_q] start interp: duration={duration:.3f}s, "
                f"hz={hz:.1f}, steps={n}, state_age={state_age:.4f}s"
            )

        t0 = time.monotonic()

        for i in range(n + 1):
            s = self._smooth_quintic(i / n)
            q = q0 + dq * s
            self.send_q(q.tolist())

            next_t = t0 + (i + 1) * dt
            sleep_s = next_t - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        if final_hold:
            self.send_q(target_q.tolist())

        if verbose:
            print("[move_to_q] done.")

        return duration


def print_state_summary(c: FR3JointClient, max_age_s: float = 1.0):
    st, age = c.get_state_cached(max_age_s=max_age_s, fallback_to_sync=True)
    q = st.get("q", None)
    print(
        "[cached] client_cache_age =",
        f"{age:.4f}s",
        "server_age =",
        st.get("age_s", None),
    )
    print("[cached] q [rad] =", format_q(q, 6))


# ============================================================
# Main
# ============================================================

def main():
    if MODE not in ("status", "delta", "move"):
        raise ValueError(f"MODE must be one of status/delta/move, got {MODE}")

    if DELTA_JOINT_IDX < 0 or DELTA_JOINT_IDX >= 7:
        raise ValueError(f"DELTA_JOINT_IDX must be in [0, 6], got {DELTA_JOINT_IDX}")

    vmax = build_vmax_from_config(MAX_JOINT_VEL_DEG_S)

    with FR3JointClient(
        HOST,
        PORT,
        timeout_s=TIMEOUT_S,
        retry_times=RETRY_TIMES,
    ) as c:
        print("[ping]", c.ping())

        st0 = c.get_state()
        print("[sync get] age(server) =", st0.get("age_s", None))
        print("[sync get] q [rad] =", format_q(st0.get("q", []), 6))

        c.start_state_poller(
            poll_hz=float(POLL_HZ),
            warn_stale_after_s=float(WARN_STALE_AFTER_S),
        )

        ok = c.wait_until_state_ready(timeout_s=2.0)
        print("[poll] ready =", ok)

        if not ok:
            raise RuntimeError("state poller not ready")

        print_state_summary(c, max_age_s=1.0)

        if MODE == "status":
            print("[mode=status] only check communication and state. Done.")
            return

        if MODE == "delta":
            q0, age0 = c.get_joint_state_cached(
                max_age_s=float(CURRENT_Q_MAX_AGE_S),
                fallback_to_sync=True,
            )
            q0 = FR3JointClient._validate_q_vec(q0, name="current_q")

            q_des = q0.copy()
            q_des[DELTA_JOINT_IDX] = q_des[DELTA_JOINT_IDX] + float(DELTA_RAD)

            print("\n[mode=delta] communication test with smooth interpolation")
            print(f"[mode=delta] q_des[{DELTA_JOINT_IDX}] += {DELTA_RAD:+.6f} rad")
            print("[mode=delta] current_q [rad] =", format_q(q0, 6))
            print("[mode=delta] target_q  [rad] =", format_q(q_des, 6))
            print("[mode=delta] state_age =", f"{age0:.4f}s")

            if DRY_RUN:
                print("[dry_run] skip move_to_q.")
            else:
                c.move_to_q(
                    q_des,
                    duration_s=float(DURATION_S),
                    hz=float(HZ),
                    max_joint_vel_rad_s=vmax,
                    current_q_max_age_s=float(CURRENT_Q_MAX_AGE_S),
                    fallback_to_sync=True,
                    verbose=True,
                )

            print_state_summary(c, max_age_s=1.0)
            return

        if MODE == "move":
            target_q = get_target_q_from_config()

            if target_q is None:
                raise ValueError(
                    "MODE='move' requires TARGET_Q to be set or INPUT_Q=True"
                )

            target_q = FR3JointClient._validate_q_vec(target_q, name="target_q")

            print("\n[mode=move] smooth move to target_q")
            print("[mode=move] target_q [rad] =", format_q(target_q, 6))

            if DRY_RUN:
                q0, age0 = c.get_joint_state_cached(
                    max_age_s=float(CURRENT_Q_MAX_AGE_S),
                    fallback_to_sync=True,
                )
                q0 = FR3JointClient._validate_q_vec(q0, name="current_q")
                dq = target_q - q0

                if vmax is None:
                    vmax_used = np.deg2rad(np.array([30.0] * 7, dtype=np.float64))
                else:
                    vmax_used = vmax

                duration_min = float(np.max(np.abs(dq) / np.maximum(vmax_used, 1e-6)))
                duration = max(float(DURATION_S), duration_min, 0.2)

                print("[dry_run] current_q [rad] =", format_q(q0, 6))
                print("[dry_run] delta_q   [rad] =", format_q(dq, 6))
                print("[dry_run] duration_s =", f"{duration:.3f}")
                print("[dry_run] skip move_to_q.")
            else:
                c.move_to_q(
                    target_q,
                    duration_s=float(DURATION_S),
                    hz=float(HZ),
                    max_joint_vel_rad_s=vmax,
                    current_q_max_age_s=float(CURRENT_Q_MAX_AGE_S),
                    fallback_to_sync=True,
                    verbose=True,
                )

            print_state_summary(c, max_age_s=1.0)
            return


if __name__ == "__main__":
    main()