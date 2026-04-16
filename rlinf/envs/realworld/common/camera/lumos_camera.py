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

"""LUMOS camera capture via OpenCV's V4L2 backend.

LUMOS cameras expose a raw I420 (YU12) stream over V4L2; this class disables
OpenCV's built-in RGB conversion, reshapes the packed YUV buffer, and does
the I420→BGR conversion manually so the output matches the RealSense / ZED
backends (BGR ``uint8``).

Depth is not available from this V4L2 interface.
"""

import glob
import os
from typing import Optional, Union

import numpy as np

from .base_camera import BaseCamera, CameraInfo


class LumosCamera(BaseCamera):
    """Camera capture for LUMOS USB cameras (V4L2, I420 stream).

    ``camera_info.serial_number`` may be:

    * a ``/dev/v4l/by-id/`` filename (preferred — stable across reboots)
    * a ``"videoN"`` shorthand resolved to ``/dev/videoN``
    * a numeric string or int interpreted as a V4L2 device index
    """

    def __init__(self, camera_info: CameraInfo, exposure: Optional[int] = None):
        import cv2

        super().__init__(camera_info)
        self._cv2 = cv2

        if camera_info.enable_depth:
            raise ValueError(
                "LumosCamera does not support depth capture via V4L2."
            )

        self._w, self._h = camera_info.resolution
        dev_path: Union[str, int] = self._resolve_device_path(
            camera_info.serial_number
        )

        self._cap = cv2.VideoCapture(dev_path, cv2.CAP_V4L2)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Failed to open LUMOS camera (serial={camera_info.serial_number}, "
                f"dev_path={dev_path})."
            )

        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YU12"))
        # Keep OpenCV from silently reinterpreting the I420 buffer.
        self._cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._h)
        self._cap.set(cv2.CAP_PROP_FPS, camera_info.fps)
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        if exposure is not None:
            # 1 == manual mode for most V4L2 UVC drivers.
            self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            self._cap.set(cv2.CAP_PROP_EXPOSURE, exposure)

    @staticmethod
    def _resolve_device_path(serial_number: Union[str, int]) -> Union[str, int]:
        if isinstance(serial_number, int):
            return serial_number
        if serial_number.startswith("video"):
            return f"/dev/{serial_number}"
        by_id = f"/dev/v4l/by-id/{serial_number}"
        if os.path.exists(by_id):
            return by_id
        try:
            return int(serial_number)
        except ValueError as exc:
            raise ValueError(
                f"Could not resolve LUMOS serial_number={serial_number!r} to a V4L2 device."
            ) from exc

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        ok, raw = self._cap.read()
        if not ok or raw is None:
            return False, None
        try:
            # I420 packs Y (H×W) + U (H/2×W/2) + V (H/2×W/2) into a (H*3/2)×W buffer.
            yuv = np.ascontiguousarray(raw).reshape(self._h * 3 // 2, self._w)
            bgr = self._cv2.cvtColor(yuv, self._cv2.COLOR_YUV2BGR_I420)
            return True, bgr
        except Exception:
            return False, None

    def _close_device(self) -> None:
        if self._cap is not None:
            self._cap.release()

    @staticmethod
    def get_device_serial_numbers() -> list[str]:
        """Return stable by-id identifiers for connected V4L2 cameras.

        Falls back to ``videoN`` names when ``/dev/v4l/by-id/`` is unavailable.
        """
        devices = glob.glob("/dev/v4l/by-id/*")
        if devices:
            return [os.path.basename(d) for d in devices]
        return [os.path.basename(v) for v in glob.glob("/dev/video*")]
