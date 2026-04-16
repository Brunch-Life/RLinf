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

"""Check LUMOS (V4L2) camera connectivity and frame capture.

Usage::

    python toolkits/realworld_check/test_lumos_camera.py \
        [--serial SERIAL] [--steps 20] [--width 1280] [--height 1280]

``--serial`` accepts any form the LumosCamera backend understands:

* a ``/dev/v4l/by-id/`` filename (preferred, stable across reboots)
* a ``"videoN"`` shorthand resolved to ``/dev/videoN``
* a numeric index (e.g. ``"0"``)

Requires ``opencv-python`` (part of the ``franka`` extra).
"""

import argparse
import time

import numpy as np

from rlinf.envs.realworld.common.camera import CameraInfo, create_camera
from rlinf.envs.realworld.common.camera.lumos_camera import LumosCamera


def main():
    parser = argparse.ArgumentParser(description="LUMOS camera hardware check")
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Device identifier. If omitted, the first enumerated device is used.",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of frames to capture"
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested frame width")
    parser.add_argument(
        "--height", type=int, default=1280, help="Requested frame height"
    )
    parser.add_argument("--fps", type=int, default=30, help="Requested camera FPS")
    parser.add_argument(
        "--exposure",
        type=int,
        default=None,
        help="Manual exposure value (driver-specific). Omit to leave auto.",
    )
    args = parser.parse_args()

    devices = LumosCamera.get_device_serial_numbers()
    if not devices:
        print("[ERROR] No LUMOS / V4L2 cameras detected under /dev/v4l/by-id/.")
        return

    print(f"[INFO] Found {len(devices)} V4L2 device(s):")
    for dev in devices:
        print(f"  serial={dev}")

    serial = args.serial or devices[0]
    print(f"\n[INFO] Testing camera serial={serial}")

    info = CameraInfo(
        name="lumos_test",
        serial_number=serial,
        camera_type="lumos",
        resolution=(args.width, args.height),
        fps=args.fps,
        enable_depth=False,
    )

    camera = create_camera(info)
    assert isinstance(camera, LumosCamera), (
        f"Factory returned unexpected type: {type(camera).__name__}"
    )
    print(f"[INFO] Factory returned {type(camera).__name__} — OK")

    # Optional manual exposure: re-open with exposure if requested. The factory
    # ignores exposure (not part of CameraInfo), so construct directly.
    if args.exposure is not None:
        camera.close() if hasattr(camera, "close") else None
        camera = LumosCamera(info, exposure=args.exposure)
        print(f"[INFO] Applied manual exposure={args.exposure}")

    camera.open()
    print("[INFO] Camera opened. Reading frames...")

    try:
        for step in range(args.steps):
            frame = camera.get_frame(timeout=5)
            if frame is None:
                print(f"  step {step}: no frame")
                continue
            print(
                f"  step {step}: shape={frame.shape}, dtype={frame.dtype}, "
                f"mean={np.mean(frame):.1f}"
            )
            time.sleep(1.0 / args.fps)
    finally:
        camera.close()
    print("[INFO] LUMOS camera check completed.")


if __name__ == "__main__":
    main()
