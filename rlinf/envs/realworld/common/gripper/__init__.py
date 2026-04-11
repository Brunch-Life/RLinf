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

from typing import Optional

from .base_gripper import BaseGripper

__all__ = [
    "BaseGripper",
    "create_gripper",
]


def create_gripper(
    gripper_type: str = "franka",
    ros=None,
    port: Optional[str] = None,
    robot_gripper=None,
    **kwargs,
) -> BaseGripper:
    """Factory that instantiates the right gripper backend.

    Args:
        gripper_type: ``"franka"`` (ROS-based), ``"robotiq"`` (Modbus RTU),
            ``"franky"`` (libfranka via Franky) or ``"pylibfranka"``
            (libfranka via pylibfranka, legacy fallback).
        ros: :class:`ROSController` instance — required for ``"franka"``.
        port: Serial device path (e.g. ``"/dev/ttyUSB0"``) — required for
            ``"robotiq"``.
        robot_gripper: A ``franky.Gripper`` instance — required for
            ``"franky"``.
        **kwargs: Forwarded to the gripper constructor (e.g. ``max_width``,
            ``baudrate``, ``slave_id``).
    """
    gt = gripper_type.lower()
    if gt == "robotiq":
        if port is None:
            raise ValueError(
                "gripper_connection (serial port) must be specified "
                "for Robotiq grippers."
            )
        from .robotiq_gripper import RobotiqGripper

        return RobotiqGripper(port=port, **kwargs)
    if gt == "franka":
        if ros is None:
            raise ValueError(
                "ROSController instance must be provided for Franka gripper."
            )
        from .franka_gripper import FrankaGripper

        return FrankaGripper(ros=ros, **kwargs)
    if gt == "franky":
        if robot_gripper is None:
            raise ValueError(
                "A franky.Gripper instance (robot.gripper) must be provided "
                "for Franky gripper."
            )
        from .franky_gripper import FrankyGripper

        return FrankyGripper(robot_gripper=robot_gripper, **kwargs)
    if gt == "pylibfranka":
        robot_ip = kwargs.pop("robot_ip", None)
        if robot_ip is None:
            raise ValueError(
                "robot_ip must be specified for pylibfranka gripper."
            )
        from .pylibfranka_gripper import PylibfrankaGripper

        return PylibfrankaGripper(robot_ip=robot_ip, **kwargs)
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        f"Supported types: 'franka', 'robotiq', 'franky', 'pylibfranka'."
    )
