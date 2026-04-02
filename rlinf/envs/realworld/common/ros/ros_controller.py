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

import os
import pathlib
import sys
import time
from typing import Callable, Optional

import psutil
import rospy
from filelock import FileLock

from rlinf.utils.logging import get_logger


class ROSController:
    """Controller for ROS communication. A controller is used for managing one robot."""

    @staticmethod
    def _parse_ros_port() -> int:
        """Return the ROS master port from ``ROS_MASTER_URI`` (default 11311)."""
        uri = os.environ.get("ROS_MASTER_URI", "http://localhost:11311")
        try:
            return int(uri.rsplit(":", 1)[-1].rstrip("/"))
        except (ValueError, IndexError):
            return 11311

    @staticmethod
    def _proc_on_port(proc: "psutil.Process", port: int) -> bool:
        """Check whether a roscore *proc* is bound to *port*."""
        try:
            cmdline = proc.cmdline()
            if "-p" in cmdline:
                idx = cmdline.index("-p")
                if idx + 1 < len(cmdline):
                    return cmdline[idx + 1] == str(port)
            return port == 11311
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            return False

    def __init__(self, ros_version: int = 1):
        """Initialize the ROS controller."""
        self._logger = get_logger()
        self._ros_version = ros_version
        assert self._ros_version == 1, "Currently only ROS 1 is supported."

        # ROS is a global service on the node
        # When there are multiple controllers, concurrency control is needed
        ros_lock_file = "/tmp/.ros.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(ros_lock_file)):
            ros_lock_file = os.path.join(pathlib.Path.home(), ".ros.lock")
        self._ros_lock = FileLock(ros_lock_file)

        self._ros_port = self._parse_ros_port()

        if self._ros_version == 1:
            # roscore is removed in ROS 2
            with self._ros_lock:
                self._ros_core = None
                for proc in psutil.process_iter(["pid", "name"]):
                    if proc.info["name"] == "roscore" and self._proc_on_port(
                        proc, self._ros_port
                    ):
                        self._ros_core = proc

                if self._ros_core is None:
                    cmd = ["roscore"]
                    if self._ros_port != 11311:
                        cmd += ["-p", str(self._ros_port)]
                    self._ros_core = psutil.Popen(
                        cmd, stdout=sys.stdout, stderr=sys.stdout
                    )
                    time.sleep(1)  # Wait for roscore to start

        # Initialize ros node
        rospy.init_node("franka_controller", anonymous=True)

        # ROS channels
        self._output_channels: dict[str, rospy.Publisher] = {}
        self._input_channels: dict[str, rospy.Subscriber] = {}
        self._input_channel_status: dict[str, bool] = {}

    def get_input_channel_status(self, name: str) -> bool:
        """Get the status of a ROS input channel.

        Args:
            name: The name of the ROS input channel.

        Returns:
            bool: The status of the ROS input channel.
        """
        if name not in self._input_channel_status:
            return False
        return self._input_channel_status.get(name, False)

    def create_ros_channel(
        self, name: str, data_class: rospy.Message, queue_size: Optional[int] = None
    ):
        """Create a ROS Publisher channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            queue_size: The size of the queue for the ROS channel. Same as common channel, queue_size 0 means an infinite queue. However, queue_size being None means the channel becomes blocking.
        """
        self._output_channels[name] = rospy.Publisher(
            name, data_class, queue_size=queue_size
        )

    def connect_ros_channel(
        self, name: str, data_class: rospy.Message, callback: Callable
    ):
        """Connect a ROS Subscriber channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            callback: The callback function to handle incoming messages.
        """

        def callback_wrapper(*args, **kwargs):
            # When the callback is called, mark the channel as active
            self._input_channel_status[name] = True
            return callback(*args, **kwargs)

        self._input_channel_status[name] = False
        self._input_channels[name] = rospy.Subscriber(
            name, data_class, callback_wrapper
        )

    def put_channel(self, name: str, data: rospy.Message):
        """Put data into a ROS Publisher channel.

        Args:
            name: The name of the ROS channel.
            data: The data to publish on the ROS channel.
        """
        if name in self._output_channels:
            assert isinstance(data, self._output_channels[name].data_class), (
                f"Invalid data type for ROS channel '{name}'. Expected {self._output_channels[name].data_class}, got {type(data)}."
            )
            self._output_channels[name].publish(data)
        else:
            self._logger.warning(f"ROS channel '{name}' is not created.")
