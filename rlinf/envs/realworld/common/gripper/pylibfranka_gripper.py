# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Franka parallel-jaw gripper controlled via pylibfranka.

This module provides a gripper backend that communicates directly with the
Franka gripper through the official pylibfranka Python bindings, without
requiring ROS.  Unlike :class:`FrankyGripper`, which receives a gripper
object from ``franky.Robot``, this class creates its own independent
``pylibfranka.Gripper`` connection to the robot.

The gripper uses a separate network channel from the arm control, so
gripper operations can be performed concurrently with real-time arm
control loops.
"""

from .base_gripper import BaseGripper


class PylibfrankaGripper(BaseGripper):
    """Franka Emika parallel-jaw gripper via pylibfranka.

    Creates an independent ``pylibfranka.Gripper`` connection to the robot.
    This connection is separate from the arm control channel and can be
    used concurrently with an active real-time control loop.

    Args:
        robot_ip: IP address of the Franka robot.
        max_width: Maximum opening width in metres (default 0.08 m).
    """

    def __init__(self, robot_ip: str, max_width: float = 0.08):
        from pylibfranka import Gripper

        self._gripper = Gripper(robot_ip)
        self._max_width = max_width
        self._is_open_flag: bool = True

    def open(self, speed: float = 0.3) -> None:
        self._gripper.move(self._max_width, speed)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        # pylibfranka Gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)
        self._gripper.grasp(0.0, speed, force, 0.08, 0.08)
        self._is_open_flag = False

    def move(self, position: float, speed: float = 0.3) -> None:
        # Convert 0-255 integer range to width in metres.
        # This matches the convention used in FrankaGripper.move().
        width = float(position / (255 * 10))
        self._gripper.move(width, speed)
        self._is_open_flag = width > self._max_width * 0.5

    @property
    def position(self) -> float:
        state = self._gripper.read_once()
        return state.width

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        try:
            self._gripper.read_once()
            return True
        except Exception:
            return False
