import argparse
import json
import socket
import time
from typing import Optional, Tuple


def recv_line(
    sock: socket.socket,
    buf: bytearray,
    timeout_s: float,
) -> Tuple[Optional[str], bytearray]:
    sock.settimeout(timeout_s)

    while True:
        nl = buf.find(b"\n")
        if nl != -1:
            line = bytes(buf[:nl]).decode("utf-8", errors="replace").strip()
            del buf[: nl + 1]

            if not line:
                continue

            return line, buf

        chunk = sock.recv(4096)
        if not chunk:
            return None, buf

        buf.extend(chunk)


def send_json(sock: socket.socket, obj: dict) -> None:
    msg = json.dumps(obj, separators=(",", ":")) + "\n"
    sock.sendall(msg.encode("utf-8"))


class GripperClient:
    def __init__(self, host: str, port: int, timeout_s: float = 5.0):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)

        self.sock: Optional[socket.socket] = None
        self.buf = bytearray()

    def __enter__(self):
        return self.connect()

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def connect(self):
        self.sock = socket.create_connection(
            (self.host, self.port),
            timeout=self.timeout_s,
        )
        return self

    def close(self) -> None:
        try:
            if self.sock is not None:
                self.sock.close()
        except Exception:
            pass

        self.sock = None

    def send_gripper(
        self,
        width: float,
        speed: float = 0.05,
        force: float = 0.1,
    ) -> None:
        assert self.sock is not None

        send_json(
            self.sock,
            {
                "gripper": {
                    "width": float(width),
                    "speed": float(speed),
                    "force": float(force),
                }
            },
        )

    def get_state(self) -> dict:
        assert self.sock is not None

        send_json(self.sock, {"cmd": "get"})

        line, self.buf = recv_line(
            self.sock,
            self.buf,
            timeout_s=2.0,
        )

        if not line:
            raise RuntimeError("get timeout")

        resp = json.loads(line)

        if not resp.get("ok", False):
            raise RuntimeError(f"bad get resp: {resp}")

        return resp

    def ping(self) -> dict:
        assert self.sock is not None

        send_json(self.sock, {"cmd": "ping"})

        line, self.buf = recv_line(
            self.sock,
            self.buf,
            timeout_s=2.0,
        )

        if not line:
            raise RuntimeError("ping timeout")

        return json.loads(line)

    def stop(self) -> dict:
        assert self.sock is not None

        send_json(self.sock, {"cmd": "stop"})

        line, self.buf = recv_line(
            self.sock,
            self.buf,
            timeout_s=2.0,
        )

        if not line:
            raise RuntimeError("stop timeout")

        return json.loads(line)


def print_state(prefix: str, state: dict) -> None:
    print(f"\n===== {prefix} =====")
    print(json.dumps(state, indent=2, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--host", type=str, default="192.168.20.8")
    ap.add_argument("--port", type=int, default=5559)
    ap.add_argument("--timeout", type=float, default=5.0)

    ap.add_argument("--speed", type=float, default=0.05)
    ap.add_argument("--force", type=float, default=0.1)

    ap.add_argument(
        "--sleep",
        type=float,
        default=3.0,
        help="两次动作之间的等待时间，默认 3 秒",
    )

    ap.add_argument(
        "--after_cmd_sleep",
        type=float,
        default=0.3,
        help="每次发送夹爪命令后，读取状态前的等待时间",
    )

    ap.add_argument("--get", action="store_true")
    ap.add_argument("--ping", action="store_true")
    ap.add_argument("--stop", action="store_true")

    motion = ap.add_mutually_exclusive_group()
    motion.add_argument("--open", action="store_true")
    motion.add_argument("--close", action="store_true")
    motion.add_argument("--width", type=float, default=None)

    ap.add_argument(
        "--demo",
        action="store_true",
        help="执行 open -> sleep -> close。默认无参数时也执行该流程。",
    )

    args = ap.parse_args()

    W_OPEN = 0.08
    W_CLOSE = 0.0002

    with GripperClient(args.host, args.port, timeout_s=args.timeout) as g:
        did_something = False

        if args.ping:
            print_state("ping", g.ping())
            did_something = True

        if args.get:
            print_state("get_state", g.get_state())
            did_something = True

        if args.stop:
            print_state("stop", g.stop())
            did_something = True

        if args.open:
            print(f"\n[client] open gripper: width={W_OPEN}")
            g.send_gripper(width=W_OPEN, speed=args.speed, force=args.force)
            time.sleep(args.after_cmd_sleep)
            print_state("after open", g.get_state())
            did_something = True

        elif args.close:
            print(f"\n[client] close gripper: width={W_CLOSE}")
            g.send_gripper(width=W_CLOSE, speed=args.speed, force=args.force)
            time.sleep(args.after_cmd_sleep)
            print_state("after close", g.get_state())
            did_something = True

        elif args.width is not None:
            print(f"\n[client] set gripper width: width={args.width}")
            g.send_gripper(width=args.width, speed=args.speed, force=args.force)
            time.sleep(args.after_cmd_sleep)
            print_state("after set width", g.get_state())
            did_something = True

        if args.demo or not did_something:
            print(f"\n[client] step 1: open gripper, width={W_OPEN}")
            g.send_gripper(width=W_OPEN, speed=args.speed, force=args.force)
            time.sleep(args.after_cmd_sleep)
            print_state("after open", g.get_state())

            print(f"\n[client] sleep {args.sleep} seconds")
            time.sleep(args.sleep)

            print(f"\n[client] step 2: close gripper again, width={W_CLOSE}")
            g.send_gripper(width=W_CLOSE, speed=args.speed, force=args.force)
            time.sleep(args.after_cmd_sleep)
            print_state("after second close", g.get_state())


if __name__ == "__main__":
    main()
