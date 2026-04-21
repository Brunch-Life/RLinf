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

"""Send raw Modbus RTU frames on the RS-485 line and report any bytes read back.

Purpose is to check whether the USB-RS485 adapter's TX LED blinks (link to
adapter OK) and whether anything on the bus answers (downstream device alive).
Useful when ``test_robotiq_gripper.py`` times out with
``ModbusIOException: No response received``.

Usage::

    python toolkits/realworld_check/probe_rs485_line.py \\
        --port /dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0

While the script runs, watch the two LEDs on the USB-RS485 adapter:
  - TX LED should flash at ~5 Hz  → USB→adapter path is fine
  - RX LED only flashes if something on the bus answers → downstream alive
"""

import argparse
import time

import serial

# Modbus RTU: read 3 input registers starting at 0x07D0 from slave 0x09
# (Robotiq 2F default). Pre-computed CRC so we don't need pymodbus here.
_FRAME_READ_ROBOTIQ = bytes.fromhex("09 03 07 D0 00 03 04 0E".replace(" ", ""))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0",
    )
    p.add_argument("--baudrate", type=int, default=115200)
    p.add_argument("--duration", type=float, default=5.0, help="seconds")
    p.add_argument("--period", type=float, default=0.2, help="send interval (s)")
    args = p.parse_args()

    s = serial.Serial(args.port, args.baudrate, timeout=0.2)
    print(
        f"[INFO] probing {args.port} @ {args.baudrate} baud for {args.duration:.1f}s — "
        f"watch the adapter LEDs"
    )

    n_sent = 0
    resp_total = bytearray()
    t_end = time.monotonic() + args.duration
    while time.monotonic() < t_end:
        s.write(_FRAME_READ_ROBOTIQ)
        n_sent += 1
        time.sleep(args.period)
        if s.in_waiting:
            chunk = s.read(s.in_waiting)
            resp_total.extend(chunk)

    s.close()

    print(f"[INFO] sent {n_sent} frames")
    if resp_total:
        print(f"[OK]  received {len(resp_total)} bytes: {resp_total.hex(' ')}")
        print("      → RS-485 bus has an answering device.")
    else:
        print("[FAIL] zero bytes received.")
        print("       TX LED blinking?   yes → USB-RS485 adapter is fine")
        print("                           no  → adapter or USB cable is dead")
        print("       RX LED blinking?   no  → nothing on the bus answered:")
        print("                                 check 24V power, A/B wiring, slave ID")


if __name__ == "__main__":
    main()
