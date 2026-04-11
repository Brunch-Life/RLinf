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

"""Franka parallel-jaw gripper controlled via Franky (libfranka).

This module provides a gripper backend that communicates directly with the
Franka gripper through the Franky library, without requiring ROS.
"""

from .base_gripper import BaseGripper


class FrankyGripper(BaseGripper):
    """Franka Emika parallel-jaw gripper controlled via Franky.

    Uses the ``franky.Gripper`` object obtained from a ``franky.Robot``
    instance for direct libfranka communication.

    Args:
        robot_gripper: A ``franky.Gripper`` instance (typically
            ``robot.gripper`` from a ``franky.Robot``).
        max_width: Maximum opening width in metres.
    """

    def __init__(self, robot_gripper, max_width: float = 0.08):
        self._gripper = robot_gripper
        self._max_width = max_width
        self._is_open_flag: bool = True

    def open(self, speed: float = 0.3) -> None:
        self._gripper.move(self._max_width, speed)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        # franky Gripper.grasp(width, speed, force, epsilon_inner, epsilon_outer)
        self._gripper.grasp(0.0, speed, force, 0.08, 0.08)
        self._is_open_flag = False

    def move(self, position: float, speed: float = 0.3) -> None:
        width = float(position / (255 * 10))
        self._gripper.move(width, speed)

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
