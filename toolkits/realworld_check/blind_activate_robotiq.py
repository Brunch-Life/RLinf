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

"""Blindly send Robotiq 2F activate frames without waiting for Modbus ACK.

Used when ``probe_rs485_line.py`` shows TX-only (no RX). If the downstream
Modbus RTU transceiver (gripper side) receives our bytes but its reply
can't come back to us (A/B wire half-broken, bad adapter RX, etc.), sending
the activate command blindly will still transition the Robotiq state
machine. Watch the gripper LED:

  red → blue slow-blink → blue solid  = our TX reaches the gripper, only
                                        the return path is broken
  red stays red                        = TX direction also broken, full
                                        cabling/adapter/gripper hw issue

Frames are hard-coded for Robotiq default slave 0x09. CRCs pre-computed
(Modbus RTU CRC-16 little-endian).
"""

import argparse
import time

import serial

# Write 3 output registers starting at 0x03E8. rACT in MSB of first reg.
# Deactivate: data = 00 00 00 00 00 00
_FRAME_DEACTIVATE = bytes.fromhex(
    "09 10 03 E8 00 03 06 00 00 00 00 00 00 73 30".replace(" ", "")
)
# Activate:   data = 01 00 00 00 00 00  (rACT = 1)
_FRAME_ACTIVATE = bytes.fromhex(
    "09 10 03 E8 00 03 06 01 00 00 00 00 00 72 E1".replace(" ", "")
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0",
    )
    p.add_argument("--baudrate", type=int, default=115200)
    args = p.parse_args()

    s = serial.Serial(args.port, args.baudrate, timeout=0.3)
    print(f"[INFO] writing deactivate to {args.port} @ {args.baudrate} baud")
    s.write(_FRAME_DEACTIVATE)
    time.sleep(0.5)

    print("[INFO] writing activate (rACT=1) — watch the gripper LED now")
    s.write(_FRAME_ACTIVATE)

    # Robotiq activation motion takes a few seconds (opens then closes fully)
    deadline = time.monotonic() + 8.0
    rx = bytearray()
    while time.monotonic() < deadline:
        if s.in_waiting:
            rx.extend(s.read(s.in_waiting))
        time.sleep(0.1)

    s.close()

    print(f"[INFO] done. received {len(rx)} bytes during 8 s window")
    if rx:
        print(f"       bytes: {rx.hex(' ')}")
        print("       → Modbus is two-way; software activate should now work.")
    else:
        print("       → no response. Diagnose by gripper LED:")
        print("           red → blue: TX ok, RX half-broken (adapter RX or A/B)")
        print("           red stays: TX broken too (cabling/adapter/gripper hw)")


if __name__ == "__main__":
    main()
