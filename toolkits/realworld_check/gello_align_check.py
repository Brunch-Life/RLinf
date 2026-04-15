"""Live GELLO ↔ Franka joint alignment dashboard.

Reads the robot's current joint positions via :class:`FrankyController`
and the GELLO leader's joint positions via :class:`GelloJointExpert`,
then prints a per-joint delta table that refreshes in place at 5 Hz.

NO motion is issued — this is purely for the human operator to physically
align the GELLO leader arm to the robot before starting data collection.

Usage::

    source .venv/bin/activate
    export RAY_ADDRESS=192.168.120.43:6380
    export PYTHONPATH=$PWD:$PYTHONPATH
    export FRANKA_ROBOT_IP=172.16.0.2
    python toolkits/realworld_check/gello_align_check.py

Press Ctrl-C to exit.

Each joint row shows::

    J<i>  robot=<rad>  gello=<rad>  Δ=<wrapped>  raw=<unwrapped>  hint

* ``Δ`` is wrapped into ``[-π, π]`` so multi-turn Dynamixel readings
  (``raw - 2π`` etc.) report a sane number.  This is the value the
  ``GelloJointIntervention`` stream gate uses internally.
* ``raw`` is the unwrapped delta — useful for diagnosing GELLO calibration
  offsets that are 2π off.
* ``hint`` is a human-readable instruction such as "turn GELLO J3 +0.42 rad
  (≈ +24°)".  Disappears once that joint is within ``align_threshold``.
"""

from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import ray

if not ray.is_initialized():
    ray.init(log_to_driver=False, logging_level="ERROR")

from rlinf.envs.realworld.common.gello.gello_joint_expert import (  # noqa: E402
    GelloJointExpert,
)
from rlinf.envs.realworld.franka.franky_controller import (  # noqa: E402
    FrankyController,
)
from rlinf.envs.realworld.franka.utils import wrap_to_pi  # noqa: E402

ALIGN_THRESHOLD = 0.5  # rad — same default as GelloJointIntervention
REFRESH_HZ = 5.0


def colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def fmt_joint_row(i: int, q_robot: float, q_gello: float) -> str:
    raw_delta = q_gello - q_robot
    wrapped = float(wrap_to_pi(np.array([raw_delta]))[0])
    abs_w = abs(wrapped)

    if abs_w < 0.05:
        c = "32"  # green
        status = "✅"
        hint = ""
    elif abs_w <= ALIGN_THRESHOLD:
        c = "33"  # yellow
        status = "~ "
        hint = ""
    else:
        c = "31"  # red
        status = "❌"
        deg = math.degrees(wrapped)
        # Operator should turn GELLO BY -wrapped to make q_gello match q_robot
        # i.e. if wrapped > 0, GELLO is ahead of robot, turn GELLO BACK
        direction = -wrapped
        direction_deg = math.degrees(direction)
        hint = (
            f"  → turn GELLO J{i + 1} by "
            f"{direction:+.3f} rad ({direction_deg:+.1f}°)"
        )

    delta_str = colour(f"Δ={wrapped:+.3f}", c)
    raw_note = ""
    if abs(raw_delta - wrapped) > 1e-6:
        raw_note = f"  raw={raw_delta:+.3f}"

    return (
        f"  {status} J{i + 1}  "
        f"robot={q_robot:+.3f}  gello={q_gello:+.3f}  "
        f"{delta_str}{raw_note}{hint}"
    )


def main() -> None:
    robot_ip = os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2")
    gripper_port = os.environ.get(
        "FRANKA_GRIPPER_PORT",
        "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0",
    )
    gello_port = os.environ.get(
        "GELLO_PORT",
        "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0",
    )

    print(f"Connecting to Franka at {robot_ip} ...", flush=True)
    controller = FrankyController.launch_controller(
        robot_ip=robot_ip,
        env_idx=0,
        node_rank=0,
        worker_rank=0,
        gripper_type="robotiq",
        gripper_connection=gripper_port,
    )
    for _ in range(60):
        if controller.is_robot_up().wait()[0]:
            break
        time.sleep(0.5)
    else:
        print("ERROR: robot did not come up", file=sys.stderr)
        sys.exit(1)
    print("Franka ready.", flush=True)

    print(f"Connecting to GELLO at {gello_port} ...", flush=True)
    gello = GelloJointExpert(port=gello_port)
    for _ in range(50):
        if gello.ready:
            break
        time.sleep(0.1)
    if not gello.ready:
        print("ERROR: GELLO did not start producing readings", file=sys.stderr)
        sys.exit(1)
    print("GELLO ready.\n", flush=True)

    period = 1.0 / REFRESH_HZ
    n_lines_drawn = 0  # how many lines our last frame occupied

    try:
        while True:
            try:
                q_robot = controller.get_state().wait()[0].arm_joint_position
                q_gello, _grip = gello.get_action()
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"read error: {e}", flush=True)
                time.sleep(period)
                continue

            wrapped = wrap_to_pi(np.asarray(q_gello) - np.asarray(q_robot))
            max_abs = float(np.max(np.abs(wrapped)))
            worst = int(np.argmax(np.abs(wrapped)))

            if max_abs <= ALIGN_THRESHOLD:
                summary = colour(
                    f"ALIGNED  (max Δ={max_abs:.3f} rad on J{worst + 1})", "32;1"
                )
            else:
                summary = colour(
                    f"NOT ALIGNED  (max Δ={wrapped[worst]:+.3f} rad on J{worst + 1}; "
                    f"threshold={ALIGN_THRESHOLD} rad)",
                    "31;1",
                )

            rows = [fmt_joint_row(i, q_robot[i], q_gello[i]) for i in range(7)]

            # Build the full frame
            frame_lines = [summary, *rows]
            new_n = len(frame_lines)

            # Move cursor up to overwrite the previous frame, then clear each
            # remaining line.  Use \r and clear-to-end-of-line for safety.
            if n_lines_drawn:
                sys.stdout.write(f"\033[{n_lines_drawn}F")
            for line in frame_lines:
                sys.stdout.write("\r\033[2K" + line + "\n")
            n_lines_drawn = new_n
            sys.stdout.flush()

            time.sleep(period)
    except KeyboardInterrupt:
        print("\nexiting (no motion was issued).")


if __name__ == "__main__":
    main()
