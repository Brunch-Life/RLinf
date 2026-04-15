"""Move the Franka arm to match the current GELLO joint positions.

Reads the GELLO's current 7-DOF joint positions, shows them to the
operator for confirmation, then moves the robot to that configuration
using FrankyController.reset_joint (slow, synchronous motion at ~4.5%
effective dynamics).

This is meant to be run **before** data collection so that the robot
and GELLO start at the same pose — no fighting during the first reset.

Usage::

    bash examples/embodiment/gello_reset_to_gello.sh

Tweakable knobs: env vars listed in the shell wrapper.
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
    JOINT_LIMITS_LOWER,
    JOINT_LIMITS_UPPER,
)

ALIGN_TOL = 0.08  # rad — considered aligned when all |Δ| < this


def colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def setup_hardware():
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
    return controller, gello


def fmt_joints(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:+.3f}" for v in q) + "]"


def fmt_deg(q: np.ndarray) -> str:
    return "[" + ", ".join(f"{math.degrees(v):+.1f}°" for v in q) + "]"


def check_limits(q: np.ndarray) -> bool:
    """Return True if all joints are within Franka limits."""
    return bool(np.all(q >= JOINT_LIMITS_LOWER) and np.all(q <= JOINT_LIMITS_UPPER))


def main():
    controller, gello = setup_hardware()

    # Average a few GELLO readings for stability.
    print("Reading GELLO position (averaging 20 samples) ...", flush=True)
    samples = []
    for _ in range(20):
        q, _ = gello.get_action()
        samples.append(q.copy())
        time.sleep(0.02)
    target = np.mean(samples, axis=0)

    # Read current robot position.
    state = controller.get_state().wait()[0]
    current = state.arm_joint_position

    delta = target - current
    max_delta = float(np.max(np.abs(delta)))

    print("─" * 64)
    print(f"  GELLO target : {fmt_joints(target)}")
    print(f"                 {fmt_deg(target)}")
    print(f"  Robot current: {fmt_joints(current)}")
    print(f"                 {fmt_deg(current)}")
    print()

    for i in range(7):
        d = delta[i]
        status = colour("OK", "32") if abs(d) < ALIGN_TOL else colour(f"Δ={d:+.3f}", "33")
        print(f"  J{i+1}: {current[i]:+.3f} → {target[i]:+.3f}  ({status})")
    print()
    print(f"  Max |Δ| = {max_delta:.3f} rad ({math.degrees(max_delta):.1f}°)")
    print("─" * 64)

    if max_delta < ALIGN_TOL:
        print(colour("\n  Already aligned! No motion needed.\n", "32"))
        return

    if not check_limits(target):
        print(
            colour("\n  ERROR: GELLO target is outside Franka joint limits!", "31"),
            file=sys.stderr,
        )
        for i in range(7):
            lo, hi = JOINT_LIMITS_LOWER[i], JOINT_LIMITS_UPPER[i]
            if target[i] < lo or target[i] > hi:
                print(
                    f"    J{i+1}: {target[i]:+.3f} not in [{lo:+.3f}, {hi:+.3f}]",
                    file=sys.stderr,
                )
        sys.exit(1)

    if max_delta > 1.5:
        print(
            colour(
                f"\n  WARNING: Max delta is {max_delta:.2f} rad ({math.degrees(max_delta):.0f}°).\n"
                f"  This is a large motion. Make sure the workspace is clear!\n",
                "33",
            )
        )

    ans = input("  Move robot to GELLO position? [y/N]: ").strip().lower()
    if ans != "y":
        print("  Aborted.")
        return

    print("\n  Moving robot (slow, ~4.5% dynamics) ...", flush=True)
    controller.reset_joint(target.tolist()).wait()
    time.sleep(0.5)

    # Verify final position.
    state = controller.get_state().wait()[0]
    final = state.arm_joint_position
    final_delta = target - final
    max_err = float(np.max(np.abs(final_delta)))

    print()
    for i in range(7):
        d = final_delta[i]
        status = colour("✓", "32") if abs(d) < ALIGN_TOL else colour(f"err={d:+.3f}", "31")
        print(f"  J{i+1}: target={target[i]:+.3f}  actual={final[i]:+.3f}  {status}")
    print()

    if max_err < ALIGN_TOL:
        print(colour("  Aligned successfully! Ready for data collection.\n", "32"))
    else:
        print(
            colour(
                f"  Alignment residual {max_err:.3f} rad > {ALIGN_TOL} tol.\n"
                f"  You may need to re-run or check for collisions.\n",
                "33",
            )
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Interrupted.")
        sys.exit(130)
