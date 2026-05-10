# Copyright 2025 The RLinf Authors.
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


_FRANKY_HAND_ALIASES = ("franka_hand", "franky_hand", "franky")


def create_gripper(
    gripper_type: str = "franka",
    ros=None,
    port: Optional[str] = None,
    robot_ip: Optional[str] = None,
    **kwargs,
) -> BaseGripper:
    """Factory that instantiates the right gripper backend.

    Args:
        gripper_type: One of:

            * ``"franka"``      — original Franka Hand via ROS (legacy stack).
            * ``"franka_hand"`` / ``"franky_hand"`` / ``"franky"`` — original
              Franka Hand via libfranka (``franky.Gripper``); shares the FCI
              IP with :class:`FrankyController`.
            * ``"robotiq"``     — Robotiq 2F-* over Modbus RTU.
        ros: :class:`ROSController` instance — required for ``"franka"``.
        port: Serial device path (e.g. ``"/dev/ttyUSB0"``) — required for
            ``"robotiq"``.
        robot_ip: FCI IP — required for ``"franka_hand"`` / ``"franky_hand"``.
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
    if gt in _FRANKY_HAND_ALIASES:
        if robot_ip is None:
            raise ValueError(
                f"robot_ip must be specified for gripper_type={gripper_type!r} "
                "(libfranka Franka Hand shares the FCI IP with the arm)."
            )
        from .franky_hand_gripper import FrankyHandGripper

        return FrankyHandGripper(robot_ip=robot_ip, **kwargs)
    if gt == "franka":
        if ros is None:
            raise ValueError(
                "ROSController instance must be provided for the ROS-based "
                "Franka gripper. To drive the original Franka Hand without "
                "ROS, use gripper_type='franka_hand' (FrankyController path)."
            )
        from .franka_gripper import FrankaGripper

        return FrankaGripper(ros=ros, **kwargs)
    raise ValueError(
        f"Unsupported gripper_type={gripper_type!r}. "
        f"Supported types: 'franka', 'franka_hand', 'robotiq'."
    )
