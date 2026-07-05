"""
View franka_exo/robot.urdf poses with Pinocchio + MeshCat.

Examples:
    conda run -n ftservo python test/view_franka_exo_urdf_pose.py --q 0 0 0 0 0 0 0
    conda run -n ftservo python test/view_franka_exo_urdf_pose.py --q-deg 0 0 0 10 0 0 0
    conda run -n ftservo python test/view_franka_exo_urdf_pose.py --joint 4 --delta 0.2
"""

from __future__ import annotations

import argparse
import atexit
import os
import ssl
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import certifi

# Some Windows certificate stores can make tornado/meshcat fail while importing
# ssl.create_default_context. MeshCat here serves localhost visualization, so a
# minimal TLS context is enough for import-time initialization.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
ssl.create_default_context = lambda *args, **kwargs: ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer


DEFAULT_URDF = "franka_exo/robot.urdf"


def install_meshcat_subprocess_patch(patch_dir: Path) -> None:
    """Keep the Windows SSL patch visible to MeshCat's server subprocess."""
    import meshcat.servers.zmqserver as zmqserver
    import meshcat.visualizer as visualizer

    def start_zmq_server_as_subprocess(zmq_url=None, server_args=None):
        if server_args is None:
            server_args = []
        args = [sys.executable, "-u", "-m", "meshcat.servers.zmqserver"]
        if zmq_url is not None:
            args.extend(["--zmq-url", zmq_url])
        if server_args:
            args.extend(server_args)

        env = dict(os.environ)
        meshcat_root = os.path.dirname(os.path.dirname(os.path.dirname(zmqserver.__file__)))
        env["PYTHONPATH"] = str(patch_dir) + os.pathsep + meshcat_root

        server_proc = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )

        line = ""
        while "zmq_url" not in line:
            line = server_proc.stdout.readline().strip().decode("utf-8")
            if server_proc.poll() is not None:
                outs, errs = server_proc.communicate()
                print(outs.decode("utf-8"))
                print(errs.decode("utf-8"))
                raise RuntimeError(
                    "the meshcat server process exited prematurely with exit code "
                    + str(server_proc.poll())
                )

        zmq_url = zmqserver.match_zmq_url(line)
        web_url = zmqserver.match_web_url(server_proc.stdout.readline().strip().decode("utf-8"))

        def cleanup(proc):
            proc.kill()
            proc.wait()

        atexit.register(cleanup, server_proc)
        return server_proc, zmq_url, web_url

    zmqserver.start_zmq_server_as_subprocess = start_zmq_server_as_subprocess
    visualizer.start_zmq_server_as_subprocess = start_zmq_server_as_subprocess


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View a 7-DOF franka_exo URDF pose in MeshCat.")
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument("--q", nargs=7, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    target_group.add_argument("--q-deg", nargs=7, type=float, metavar=("J1", "J2", "J3", "J4", "J5", "J6", "J7"))
    parser.add_argument("--urdf", default=DEFAULT_URDF, help=f"URDF path, default: {DEFAULT_URDF}")
    parser.add_argument("--joint", type=int, choices=range(1, 8), help="Show +delta on one 1-based joint from zero.")
    parser.add_argument("--delta", type=float, default=0.2, help="Joint delta in rad for --joint, default: 0.2")
    parser.add_argument("--show-collision", action="store_true", help="Display collision geometry too.")
    parser.add_argument("--no-browser", action="store_true", help="Start MeshCat without opening the browser.")
    parser.add_argument("--hold-sec", type=float, default=0.0, help="Seconds to hold. 0 means until Ctrl+C.")
    return parser.parse_args()


def target_q(args: argparse.Namespace, model: pin.Model) -> np.ndarray:
    if args.q is not None:
        q = np.asarray(args.q, dtype=float)
    elif args.q_deg is not None:
        q = np.deg2rad(np.asarray(args.q_deg, dtype=float))
    else:
        q = np.zeros(model.nq)

    if args.joint is not None:
        q = np.zeros(model.nq)
        q[args.joint - 1] = args.delta
    return q


def print_model_info(model: pin.Model, q: np.ndarray) -> None:
    print(f"nq={model.nq}, nv={model.nv}")
    print(f"joints={list(model.names)}")
    print("lower=" + np.array2string(model.lowerPositionLimit, precision=5, separator=","))
    print("upper=" + np.array2string(model.upperPositionLimit, precision=5, separator=","))
    print("q_rad=" + np.array2string(q, precision=5, separator=","))
    print("q_deg=" + np.array2string(np.rad2deg(q), precision=2, separator=","))

    below = q < model.lowerPositionLimit
    above = q > model.upperPositionLimit
    if np.any(below | above):
        bad = np.where(below | above)[0] + 1
        print(f"Warning: q is outside URDF joint limits for joints {bad.tolist()}")


def main() -> int:
    args = parse_args()
    urdf_path = Path(args.urdf).resolve()
    package_dir = urdf_path.parent
    patch_dir = Path(__file__).resolve().parent / "meshcat_sitecustomize"
    os.environ["PYTHONPATH"] = str(patch_dir) + os.pathsep + os.environ.get("PYTHONPATH", "")
    install_meshcat_subprocess_patch(patch_dir)

    model, collision_model, visual_model = pin.buildModelsFromUrdf(str(urdf_path), str(package_dir))
    if model.nq != 7:
        raise RuntimeError(f"Expected 7-DOF URDF, got nq={model.nq}")

    q = target_q(args, model)
    print_model_info(model, q)

    viz = MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=not args.no_browser)
    viz.loadViewerModel(rootNodeName="franka_exo")
    viz.displayCollisions(args.show_collision)
    viz.displayVisuals(True)
    viz.display(q)

    print("MeshCat viewer is running.")
    print("Use --q/--q-deg to set a pose, or --joint N --delta 0.2 to inspect a positive joint direction.")
    if args.hold_sec > 0:
        time.sleep(args.hold_sec)
    else:
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Interrupted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
