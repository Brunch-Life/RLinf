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

import time
from typing import Optional

import numpy as np

from .base_camera import BaseCamera, CameraInfo


class RealSenseCamera(BaseCamera):
    """Camera capture for Intel RealSense cameras.

    Adapted from SERL's RSCapture class.
    For RealSense usage, see
    https://github.com/IntelRealSense/librealsense/blob/jupyter/notebooks/quick_start_live.ipynb.
    """

    def __init__(self, camera_info: CameraInfo):
        import pyrealsense2 as rs

        super().__init__(camera_info)

        self._device_info = {}
        for device in rs.context().devices:
            self._device_info[device.get_info(rs.camera_info.serial_number)] = device
        assert camera_info.serial_number in self._device_info.keys(), (
            f"{self._device_info.keys()=}"
        )

        self._serial_number = camera_info.serial_number
        self._device = self._device_info[self._serial_number]
        self._enable_depth = camera_info.enable_depth

        self._rs = rs
        self._resolution = camera_info.resolution
        self._fps = camera_info.fps

        # Aborted prior runs can leave the USB endpoint in VIDIOC_S_FMT=EBUSY.
        # A hardware_reset() + settle sleep clears it; retry once before giving up.
        try:
            self.profile = self._start_pipeline()
        except RuntimeError as exc:
            if "Device or resource busy" not in str(exc):
                raise
            self._device.hardware_reset()
            time.sleep(3.0)
            self._refresh_device()
            self.profile = self._start_pipeline()

        # rs.align allows us to perform alignment of depth frames to color frames
        self._align = rs.align(rs.stream.color)

    def _build_config(self):
        rs = self._rs
        config = rs.config()
        config.enable_device(self._serial_number)
        config.enable_stream(
            rs.stream.color,
            self._resolution[0],
            self._resolution[1],
            rs.format.bgr8,
            self._fps,
        )
        if self._enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self._resolution[0],
                self._resolution[1],
                rs.format.z16,
                self._fps,
            )
        return config

    def _start_pipeline(self):
        self._pipeline = self._rs.pipeline()
        self._config = self._build_config()
        return self._pipeline.start(self._config)

    def _refresh_device(self):
        # hardware_reset drops the old handle; re-enumerate before retrying.
        self._device_info = {}
        for device in self._rs.context().devices:
            self._device_info[device.get_info(self._rs.camera_info.serial_number)] = (
                device
            )
        if self._serial_number in self._device_info:
            self._device = self._device_info[self._serial_number]

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        frames = self._pipeline.wait_for_frames()
        aligned_frames = self._align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        if self._enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if color_frame.is_video_frame():
            frame = np.asarray(color_frame.get_data())
            if self._enable_depth and depth_frame.is_depth_frame():
                depth = np.expand_dims(np.asarray(depth_frame.get_data()), axis=2)
                return True, np.concatenate((frame, depth), axis=-1)
            else:
                return True, frame
        else:
            return False, None

    def _close_device(self) -> None:
        self._pipeline.stop()
        self._config.disable_all_streams()

    @staticmethod
    def get_device_serial_numbers() -> set[str]:
        """Return serial numbers of all connected RealSense cameras."""
        cameras: set[str] = set()
        try:
            import pyrealsense2 as rs
        except ImportError:
            return cameras
        for device in rs.context().devices:
            cameras.add(device.get_info(rs.camera_info.serial_number))
        return cameras
