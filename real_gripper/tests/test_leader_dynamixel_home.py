from __future__ import annotations

import sys
import types
import unittest

import numpy as np


try:
    import dynamixel_sdk  # noqa: F401
except ModuleNotFoundError:
    sdk = types.ModuleType("dynamixel_sdk")
    sdk.GroupSyncRead = object
    sdk.GroupSyncWrite = object
    sdk.PacketHandler = object
    sdk.PortHandler = object
    robotis_def = types.ModuleType("dynamixel_sdk.robotis_def")
    robotis_def.COMM_PORT_BUSY = -1000
    robotis_def.COMM_SUCCESS = 0
    robotis_def.DXL_HIBYTE = lambda value: (value >> 8) & 0xFF
    robotis_def.DXL_LOBYTE = lambda value: value & 0xFF
    sys.modules["dynamixel_sdk"] = sdk
    sys.modules["dynamixel_sdk.robotis_def"] = robotis_def

from real.adapters.leader_dynamixel import (  # noqa: E402
    LeaderDynamixelReader,
    LeaderState,
    to_goal_position_param,
)


def state(position: list[int], velocity: list[int] | None = None) -> LeaderState:
    raw_position = np.asarray(position, dtype=np.float64)
    raw_velocity = np.zeros(7, dtype=np.float64)
    if velocity is not None:
        raw_velocity = np.asarray(velocity, dtype=np.float64)
    return LeaderState(
        timestamp_monotonic=0.0,
        raw_position=raw_position,
        raw_velocity=raw_velocity,
        raw_current=np.zeros(7, dtype=np.float64),
        q_motor=np.zeros(7, dtype=np.float64),
        dq_motor=np.zeros(7, dtype=np.float64),
    )


class FakeLeader:
    joint_ids = (21, 22, 23, 24, 25, 26, 27)

    def __init__(self, states: list[LeaderState | BaseException]):
        self.states = iter(states)
        self.events: list[object] = []

    def read_state(self) -> LeaderState:
        self.events.append("read")
        value = next(self.states)
        if isinstance(value, BaseException):
            raise value
        return value

    def set_position_control_mode(self, velocity: int, acceleration: int) -> None:
        self.events.append(("position_mode", velocity, acceleration))

    def write_goal_position(self, target: np.ndarray) -> None:
        self.events.append(("goal_position", target.tolist()))

    def set_torque(self, enable: bool) -> None:
        self.events.append(("torque", enable))


class LeaderHomeTests(unittest.TestCase):
    def test_goal_position_encoding_is_little_endian(self) -> None:
        self.assertEqual(to_goal_position_param(2048), [0, 8, 0, 0])
        self.assertEqual(to_goal_position_param(-1), [255, 255, 255, 255])

    def test_home_enables_position_torque_then_writes_target_and_disables_after_settle(self) -> None:
        leader = FakeLeader(
            [
                state([2000] * 7),
                state([2048] * 7),
            ]
        )

        result = LeaderDynamixelReader.home_to_raw_position(
            leader,
            [2048] * 7,
            settle_time_s=0.0,
            verbose=False,
        )

        self.assertEqual(result.raw_position.tolist(), [2048.0] * 7)
        self.assertEqual(
            leader.events,
            [
                "read",
                ("position_mode", 20, 5),
                ("torque", True),
                ("goal_position", [2048] * 7),
                "read",
                ("torque", False),
            ],
        )

    def test_home_refuses_excessive_move_before_writing(self) -> None:
        leader = FakeLeader([state([0] * 7)])

        with self.assertRaisesRegex(RuntimeError, "exceeds max_delta_ticks"):
            LeaderDynamixelReader.home_to_raw_position(
                leader,
                [2048] * 7,
                max_delta_ticks=1200,
                verbose=False,
            )

        self.assertEqual(leader.events, ["read"])

    def test_home_disables_torque_if_state_read_fails(self) -> None:
        leader = FakeLeader(
            [
                state([2000] * 7),
                RuntimeError("read failed"),
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "read failed"):
            LeaderDynamixelReader.home_to_raw_position(
                leader,
                [2048] * 7,
                settle_time_s=0.0,
                verbose=False,
            )

        self.assertEqual(leader.events[-1], ("torque", False))


if __name__ == "__main__":
    unittest.main()
