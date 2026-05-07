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

"""GELLO ↔ Franka two-pose calibration.

Determines BOTH ``joint_signs`` and ``joint_offsets`` for a Franka FR3
GELLO leader by sampling raw Dynamixel motor positions at two known
robot poses.  Per-joint:

* ``sign[i] = +1`` if the raw motor position moved in the same direction
  as the robot joint between the two poses, ``-1`` otherwise.
* ``offset[i]`` is the multiple of π/2 closest to
  ``raw_A[i] - sign[i] * q_A[i]``.

The robot is moved between the two poses via the safe synchronous
``FrankyController.reset_joint`` (with the same ``max_joint_delta``
guard as everywhere else).  Both poses must be reachable from the
current robot configuration without exceeding that guard.

Workflow::

    1. Connect to robot + GELLO (raw Dynamixel; signs and offsets bypassed).
    2. Move robot to POSE A (a known joint config).
    3. Operator physically grabs GELLO and poses it to look exactly
       like the robot.  Press ENTER.
    4. Move robot to POSE B (a different known joint config).
    5. Operator re-poses GELLO to match.  Press ENTER.
    6. Compute signs + offsets, print a ready-to-paste
       DynamixelRobotConfig block.

Run::

    bash examples/embodiment/gello_calibrate.sh
"""

from __future__ import annotations

import math
import os
import sys
import time
from typing import Sequence

import numpy as np
import ray

if not ray.is_initialized():
    ray.init(log_to_driver=False, logging_level="ERROR")

# Use the upstream Dynamixel driver directly so we can read RAW motor
# positions, bypassing all sign / offset calibration.
from gello.dynamixel.driver import DynamixelDriver  # noqa: E402

from rlinf.envs.realworld.franka.franky_controller import (  # noqa: E402
    FrankyController,
)

# ── tunables ────────────────────────────────────────────────────────────
ROBOT_IP = os.environ.get("FRANKA_ROBOT_IP", "172.16.0.2")
GRIPPER_PORT = os.environ.get(
    "FRANKA_GRIPPER_PORT",
    "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0",
)
GELLO_PORT = os.environ.get(
    "GELLO_PORT",
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0",
)
DXL_BAUDRATE = int(os.environ.get("GELLO_BAUDRATE", "1000000"))

# Two reachable robot poses for sign + offset calibration.
#
# POSE_A is the canonical Franka home (gripper down, elbow up).
# POSE_B uses pure π/4 multiples on every joint so the operator can
# eyeball each GELLO joint at 0° / ±45° / ±90° / ±135° instead of
# arbitrary radian values.  Each per-joint delta is exactly π/4
# (≈ 0.785 rad), comfortably under reset_joint's 1.5 rad
# max_joint_delta guard.
#
# Visual description of POSE_B:
#   J1 = -π/4   →  base 45° to operator's right
#   J2 =  0     →  upper arm vertical (straight up)
#   J3 = -π/4   →  shoulder roll −45°
#   J4 = -π/2   →  elbow at 90° (less bent than home)
#   J5 = +π/4   →  forearm roll +45°
#   J6 = +3π/4  →  wrist pitched 135°
#   J7 =  0     →  flange roll 0°
#
# All values are within Franka joint limits.  The robot prints POSE_B
# and asks for confirmation BEFORE issuing any motion — if you don't
# trust this configuration in your workspace, edit this block or
# answer "n" at the prompt.
PI = np.pi
POSE_A = np.array([0.0, -PI / 4, 0.0, -3 * PI / 4, 0.0, PI / 2, PI / 4])
POSE_B = np.array([-PI / 4, 0.0, -PI / 4, -PI / 2, PI / 4, 3 * PI / 4, 0.0])

JOINT_IDS = (1, 2, 3, 4, 5, 6, 7)  # 7 arm joints
GRIPPER_ID = 8  # gripper Dynamixel id (per existing config)
NUM_ARM = len(JOINT_IDS)
TWO_PI = 2.0 * math.pi
# ────────────────────────────────────────────────────────────────────────


def colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def print_banner(title: str) -> None:
    print()
    print(colour("─" * 64, "36"))
    print(colour(f" {title}", "36;1"))
    print(colour("─" * 64, "36"))


def setup_robot() -> FrankyController:
    print(f"Connecting to Franka at {ROBOT_IP} ...", flush=True)
    controller = FrankyController.launch_controller(
        robot_ip=ROBOT_IP,
        env_idx=0,
        node_rank=0,
        worker_rank=0,
        gripper_type="robotiq",
        gripper_connection=GRIPPER_PORT,
    )
    for _ in range(60):
        if controller.is_robot_up().wait()[0]:
            break
        time.sleep(0.5)
    else:
        print("ERROR: robot did not come up", file=sys.stderr)
        sys.exit(1)
    print("Franka ready.", flush=True)
    return controller


def setup_gello_raw() -> DynamixelDriver:
    print(
        f"Opening GELLO Dynamixel chain at {GELLO_PORT} (baud {DXL_BAUDRATE}) ...",
        flush=True,
    )
    driver = DynamixelDriver(
        list(JOINT_IDS) + [GRIPPER_ID],
        port=GELLO_PORT,
        baudrate=DXL_BAUDRATE,
    )
    # Warm up — first few reads may be noisy.
    for _ in range(10):
        driver.get_joints()
        time.sleep(0.01)
    print("GELLO ready.", flush=True)
    return driver


def read_raw_arm(driver: DynamixelDriver, n_samples: int = 30) -> np.ndarray:
    """Average several raw motor reads on the 7 arm joints."""
    samples = []
    for _ in range(n_samples):
        q = driver.get_joints()
        samples.append(np.asarray(q[:NUM_ARM], dtype=np.float64))
        time.sleep(0.01)
    return np.median(np.stack(samples, axis=0), axis=0)


def read_raw_gripper(driver: DynamixelDriver, n_samples: int = 30) -> float:
    samples = []
    for _ in range(n_samples):
        q = driver.get_joints()
        samples.append(float(q[NUM_ARM]))
        time.sleep(0.01)
    return float(np.median(np.asarray(samples)))


def safe_reset_to(controller: FrankyController, q_target: Sequence[float]) -> None:
    """Move robot to ``q_target`` via the slow safe path used everywhere else.

    On any failure (FCI gone, max_joint_delta exceeded, libfranka reflex,
    etc.) we exit with an actionable recovery hint instead of leaving
    the user staring at a stack trace.
    """
    try:
        q_now = controller.get_state().wait()[0].arm_joint_position
        delta = np.asarray(q_target) - np.asarray(q_now)
        max_d = float(np.max(np.abs(delta)))
        worst = int(np.argmax(np.abs(delta)))
        print(
            f"  current: {[round(float(x), 3) for x in q_now]}",
            flush=True,
        )
        print(
            f"  target : {[round(float(x), 3) for x in q_target]}  "
            f"(max Δ={max_d:.3f} rad on J{worst + 1})",
            flush=True,
        )
        if max_d > 1.5:
            print(
                colour(
                    f"\n  ❌ J{worst + 1} would need to move {max_d:.3f} rad — "
                    "exceeds reset_joint safety guard (1.5 rad).",
                    "31;1",
                )
            )
            print(
                "  Manually jog the Franka closer to the target pose using\n"
                "  Desk's hand-guide mode (white button on the arm), then\n"
                "  re-run this script.\n"
            )
            sys.exit(2)
    except Exception:
        # Probe failed — let reset_joint surface the real error below.
        pass

    print("  → calling reset_joint (slow, ~4.5% dynamics) ...", flush=True)
    try:
        controller.reset_joint(list(q_target)).wait()
    except Exception as e:
        msg = str(e)
        print(colour(f"\n  ❌ reset_joint failed: {msg}", "31;1"), file=sys.stderr)
        if "FCI" in msg or "Connection" in msg:
            print(
                "\n  The Franka FCI session is not active.  Open\n"
                "    http://172.16.0.2/desk/\n"
                "  release the User Stop button if pressed, click the\n"
                "  three-dot menu → 'Activate FCI', and unlock the joints\n"
                "  (white LED → blue).  Then re-run this script.",
                file=sys.stderr,
            )
        elif "reflex" in msg.lower() or "discontinuity" in msg.lower():
            print(
                "\n  libfranka reflex tripped during the move — the planned\n"
                "  trajectory likely passed too close to a joint limit or\n"
                "  the workspace boundary.  Try jogging the robot to a more\n"
                "  open configuration first.",
                file=sys.stderr,
            )
        elif "max_joint_delta" in msg:
            print(
                "\n  Manually jog the Franka closer to the calibration\n"
                "  start pose using Desk's hand-guide mode, then re-run.",
                file=sys.stderr,
            )
        sys.exit(2)
    time.sleep(0.5)


def wait_for_enter(prompt: str, driver: DynamixelDriver | None = None) -> None:
    """Block on ENTER while optionally streaming raw GELLO values for visual feedback."""
    print()
    print(prompt)
    if driver is not None:
        print("(raw motor positions stream below — press ENTER when GELLO matches)")
    try:
        if driver is None:
            input("  press ENTER to continue: ")
            return
        # Stream raw values until input is detected on stdin (best-effort).
        import select

        last_print = 0.0
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.05)
            if r:
                sys.stdin.readline()
                return
            now = time.time()
            if now - last_print > 0.2:
                q = driver.get_joints()
                arm = np.asarray(q[:NUM_ARM], dtype=np.float64)
                line = "  raw motors: " + " ".join(
                    f"J{i + 1}={arm[i]:+.3f}" for i in range(NUM_ARM)
                )
                sys.stdout.write("\r\033[2K" + line)
                sys.stdout.flush()
                last_print = now
    except KeyboardInterrupt:
        print()
        print("aborted by user; no calibration was written.", file=sys.stderr)
        sys.exit(130)


def snap_to_half_pi(x: float) -> float:
    """Round x to the nearest k * π/2 in [-8π, 8π]."""
    grid = np.linspace(-8 * np.pi, 8 * np.pi, 8 * 4 + 1)
    return float(grid[int(np.argmin(np.abs(grid - x)))])


def half_pi_label(x: float) -> str:
    """Return a 'k * np.pi/2' style symbolic label for x."""
    k = int(round(x / (np.pi / 2)))
    return f"{k} * np.pi / 2"


def describe_pose(name: str, q: np.ndarray) -> None:
    """Print a pose as both numeric values and human angle units."""
    print(f"  {name}:")
    for i, v in enumerate(q):
        deg = math.degrees(float(v))
        # Closest k * π/4 label, e.g. "−π/4 (≈−45°)".
        k = int(round(float(v) / (PI / 4)))
        if abs(float(v) - k * PI / 4) < 1e-6:
            if k == 0:
                sym = "0"
            elif abs(k) == 4:
                sym = ("−" if k < 0 else "") + "π"
            elif abs(k) == 2:
                sym = ("−" if k < 0 else "") + "π/2"
            else:
                sign = "−" if k < 0 else " "
                num = abs(k)
                sym = f"{sign}{num}π/4" if num != 1 else f"{sign}π/4"
        else:
            sym = f"{v:+.4f} rad"
        print(f"    J{i + 1} = {float(v):+.4f}  ({sym},  {deg:+7.2f}°)")


def confirm_or_exit(prompt: str = "Proceed? [y/N]: ") -> None:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)
    if ans not in ("y", "yes"):
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)


def main() -> None:
    controller = setup_robot()
    driver = setup_gello_raw()

    print_banner("Calibration plan — review before any motion")
    describe_pose("POSE A (Franka home)", POSE_A)
    print()
    describe_pose("POSE B (calibration target)", POSE_B)
    print()
    print(
        "  per-joint Δ (B − A): "
        + ", ".join(f"J{i + 1}={POSE_B[i] - POSE_A[i]:+.3f}" for i in range(NUM_ARM))
    )
    print(
        f"  max |Δ| = {float(np.max(np.abs(POSE_B - POSE_A))):.3f} rad "
        "(safety guard = 1.5 rad)"
    )
    print()
    print(
        "The robot will be moved twice (A → wait → B → wait) using\n"
        "reset_joint at ~4.5% effective dynamics.  Each move is gated by\n"
        "the 1.5 rad max_joint_delta guard."
    )
    print()
    confirm_or_exit("Proceed with calibration? [y/N]: ")

    print_banner("Step 1 — move robot to POSE A and align GELLO")
    safe_reset_to(controller, POSE_A)
    q_robot_A = controller.get_state().wait()[0].arm_joint_position
    print(
        f"  robot now at A: {[round(float(x), 3) for x in q_robot_A]}",
        flush=True,
    )
    wait_for_enter(
        "Physically pose the GELLO leader so it visually matches the Franka.\n"
        "Hold it steady, then press ENTER (with a free finger).",
        driver,
    )
    raw_A = read_raw_arm(driver)
    grip_A = read_raw_gripper(driver)
    print(f"\n  raw motor A: {[round(float(x), 4) for x in raw_A]}")
    print(f"  raw gripper A: {grip_A:.4f}")

    print_banner("Step 2 — move robot to POSE B and re-align GELLO")
    safe_reset_to(controller, POSE_B)
    q_robot_B = controller.get_state().wait()[0].arm_joint_position
    print(
        f"  robot now at B: {[round(float(x), 3) for x in q_robot_B]}",
        flush=True,
    )
    wait_for_enter(
        "Re-pose the GELLO leader so it visually matches the new Franka pose.\n"
        "Press ENTER when ready.",
        driver,
    )
    raw_B = read_raw_arm(driver)
    grip_B = read_raw_gripper(driver)
    print(f"\n  raw motor B: {[round(float(x), 4) for x in raw_B]}")
    print(f"  raw gripper B: {grip_B:.4f}")

    print_banner("Step 3 — solve signs and offsets")
    dq_robot = q_robot_B - q_robot_A
    dq_motor = raw_B - raw_A

    signs = np.where(dq_robot * dq_motor > 0, 1, -1).astype(int)
    print(
        "  Δq_robot:  "
        + " ".join(f"J{i + 1}={dq_robot[i]:+.3f}" for i in range(NUM_ARM))
    )
    print(
        "  Δq_motor:  "
        + " ".join(f"J{i + 1}={dq_motor[i]:+.3f}" for i in range(NUM_ARM))
    )
    print(
        "  signs:     "
        + " ".join(f"J{i + 1}={int(signs[i]):+d}" for i in range(NUM_ARM))
    )

    # Warn about ambiguous joints (very small motor delta → sign unreliable)
    ambiguous = [
        i for i in range(NUM_ARM) if abs(dq_motor[i]) < 0.05 or abs(dq_robot[i]) < 0.1
    ]
    if ambiguous:
        print(
            colour(
                "  WARNING: ambiguous sign on joints "
                + ", ".join(f"J{i + 1}" for i in ambiguous)
                + " (Δ too small).  Re-run with poses that move that joint more.",
                "33;1",
            )
        )

    # offset[i] = raw_A[i] - sign[i] * q_robot_A[i], snapped to k*pi/2.
    offsets_raw = raw_A - signs * q_robot_A
    offsets = np.array([snap_to_half_pi(o) for o in offsets_raw])

    print()
    print("  raw offsets (before snapping):")
    print("   ", [round(float(o), 4) for o in offsets_raw])
    print("  snapped to k * π/2:")
    print("   ", [round(float(o), 4) for o in offsets])

    # Sanity: re-derive q_gello at both poses with the new calibration.
    def calib(raw: np.ndarray) -> np.ndarray:
        return signs * (raw - offsets)

    q_calib_A = calib(raw_A)
    q_calib_B = calib(raw_B)
    res_A = q_calib_A - q_robot_A
    res_B = q_calib_B - q_robot_B
    print()
    print("  residual at A: " + " ".join(f"{r:+.3f}" for r in res_A))
    print("  residual at B: " + " ".join(f"{r:+.3f}" for r in res_B))
    max_res = max(float(np.max(np.abs(res_A))), float(np.max(np.abs(res_B))))
    if max_res > 0.15:
        print(
            colour(
                f"  ⚠ max residual {max_res:.3f} rad — calibration is loose.\n"
                "    Most likely you didn't physically match the robot pose closely;\n"
                "    re-pose the GELLO and re-run the offending step.",
                "33;1",
            )
        )
    else:
        print(colour(f"  ✅ max residual {max_res:.3f} rad — looks clean", "32;1"))

    # Gripper open/close hints — use the upstream gello_get_offset.py heuristic
    grip_open_deg = math.degrees(grip_B) - 0.2
    grip_close_deg = math.degrees(grip_B) - 42.0
    print()
    print(f"  gripper at pose B raw: {grip_B:.4f} rad ({math.degrees(grip_B):.2f}°)")
    print(
        f"  suggested gripper_config:  ({GRIPPER_ID}, "
        f"{int(round(grip_open_deg))}, {int(round(grip_close_deg))})"
    )
    print(
        "  (after pasting, manually open/close the GELLO gripper and tweak "
        "the open/close degree numbers if the binary state is wrong)"
    )

    # Final paste-ready snippet
    print()
    print(colour("─" * 64, "36"))
    print(
        colour(" Paste this into gello/agents/gello_agent.py PORT_CONFIG_MAP:", "36;1")
    )
    print(colour("─" * 64, "36"))
    sign_tuple = ", ".join(str(int(s)) for s in signs)
    print(
        f'    "{GELLO_PORT}": DynamixelRobotConfig(\n'
        f"        joint_ids={tuple(JOINT_IDS)},\n"
        f"        joint_offsets=(\n"
        + "".join(f"            {half_pi_label(o)},\n" for o in offsets)
        + f"        ),\n"
        f"        joint_signs=({sign_tuple}),\n"
        f"        gripper_config=({GRIPPER_ID}, "
        f"{int(round(grip_open_deg))}, {int(round(grip_close_deg))}),\n"
        f"        baudrate=1_000_000,\n"
        f"    ),"
    )
    print()
    print("After pasting, restart any process that imports gello (no need to")
    print("reinstall — it's an editable install).  Then re-run gello_align.sh")
    print("to verify J1..J7 all stay green when you move the GELLO around.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\naborted.", file=sys.stderr)
        sys.exit(130)
