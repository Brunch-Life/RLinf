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

"""Sequential per-joint GELLO ↔ Franka alignment guide.

Walks the operator through aligning **one joint at a time**, J1 → J7.
The current joint's delta is printed in place at 10 Hz.  As soon as it
stays within ``ALIGN_TOL`` for ``STABLE_TICKS`` consecutive frames, the
script confirms it, prints a checkmark, and advances to the next joint.

NO motion is issued — pure read-only alignment dashboard.

Usage::

    source .venv/bin/activate
    export RAY_ADDRESS=192.168.120.43:6380
    export PYTHONPATH=$PWD:$PYTHONPATH
    export FRANKA_ROBOT_IP=172.16.0.2
    python toolkits/realworld_check/gello_align_sequential.py

Tweakable knobs at the top.  Press Ctrl-C to abort at any time — the
GELLO leader and the Franka are not touched.
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

# ── tunables ────────────────────────────────────────────────────────────
ALIGN_TOL = 0.10  # rad — joint considered aligned when |Δ| < this
STABLE_TICKS = 8  # consecutive frames inside tol → advance
REFRESH_HZ = 10.0

# The pose the robot is moved to before alignment starts.  Chosen to be
# visually distinctive and easy for the operator to physically replicate
# on the GELLO leader, NOT the same as the calibration POSE_A — the
# whole point is to verify that the calibration generalises to a
# different configuration.  All joint values are multiples of π/4.
#
# Visual interpretation:
#   J1 = +π/4   →  base rotated 45° (operator-relative direction depends
#                   on Franka mounting; positive J1 in Franka convention
#                   is CCW when viewed from above)
#   J2 =  0     →  upper arm vertical / "straight"
#   J3 =  0     →  no shoulder roll
#   J4 = -π/2   →  elbow at a right angle — forearm perpendicular to
#                   the (vertical) upper arm, i.e. forearm horizontal
#   J5 =  0     →  no forearm roll
#   J6 = +π/2   →  wrist at a right angle from forearm
#   J7 =  0     →  no flange roll
#
# Override via the ALIGN_HOME environment variable as a comma-separated
# list of 7 floats in radians, e.g.::
#
#   ALIGN_HOME="0.0,0.0,0.0,-1.5708,0.0,0.7854,0.7854" \
#       bash examples/embodiment/gello_align.sh
HOME_JOINTS_DEFAULT = [
    math.pi / 4,  # J1
    0.0,  # J2
    0.0,  # J3
    -math.pi / 2,  # J4
    0.0,  # J5
    math.pi / 2,  # J6
    0.0,  # J7
]


def _parse_home_env() -> list[float]:
    raw = os.environ.get("ALIGN_HOME", "").strip()
    if not raw:
        return list(HOME_JOINTS_DEFAULT)
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != 7:
        print(
            f"ERROR: ALIGN_HOME must contain exactly 7 comma-separated "
            f"floats, got {len(parts)}: {raw!r}",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        return [float(p) for p in parts]
    except ValueError as e:
        print(f"ERROR: cannot parse ALIGN_HOME={raw!r}: {e}", file=sys.stderr)
        sys.exit(2)


HOME_JOINTS: list[float] = _parse_home_env()
# ────────────────────────────────────────────────────────────────────────


def colour(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def progress_bar(delta: float, tol: float, width: int = 30) -> str:
    """Return a horizontal bar showing how far the joint is from aligned.

    Centered at zero; left = GELLO ahead of robot, right = GELLO behind.
    Capped at ±π/2 visually so the bar stays readable.
    """
    cap = math.pi / 2
    x = max(-cap, min(cap, delta))
    centre = width // 2
    pos = centre + int(round(x / cap * centre))
    pos = max(0, min(width - 1, pos))

    bar_chars = ["·"] * width
    bar_chars[centre] = "│"
    bar_chars[pos] = "●"
    bar = "".join(bar_chars)

    inside = abs(delta) <= tol
    return colour(f"[{bar}]", "32" if inside else "33")


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


def _format_angle(rad: float) -> str:
    """Pretty-print a radian value with both deg and π/4 multiple form."""
    deg = math.degrees(rad)
    k = int(round(rad / (math.pi / 4)))
    if abs(rad - k * math.pi / 4) < 1e-6:
        if k == 0:
            sym = "0"
        elif abs(k) == 4:
            sym = ("−" if k < 0 else "") + "π"
        elif abs(k) == 2:
            sym = ("−" if k < 0 else "") + "π/2"
        else:
            sym = ("−" if k < 0 else "+") + (f"{abs(k)}π/4" if abs(k) != 1 else "π/4")
    else:
        sym = f"{rad:+.4f} rad"
    return f"{rad:+.4f}  ({sym},  {deg:+7.2f}°)"


def _print_home_pose(home: list[float]) -> None:
    print(colour("Alignment HOME pose:", "36;1"))
    for i, v in enumerate(home):
        print(f"  J{i + 1} = {_format_angle(v)}")
    # Verbal description for operator visual matching
    print()
    print(
        "  Geometric intent: J2 vertical, J4 at right angle (forearm\n"
        "  perpendicular to upper arm), J6 at right angle (wrist 90°\n"
        "  from forearm), and J1 rotated 45° as an asymmetric reference.\n"
        "  Override via the ALIGN_HOME env var if you want a different pose."
    )


def confirm_or_exit(prompt: str = "Proceed with motion to HOME? [y/N]: ") -> None:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)
    if ans not in ("y", "yes"):
        print("aborted by user; no motion was issued.", file=sys.stderr)
        sys.exit(130)


def move_to_home(controller) -> None:
    """Move the robot to ``HOME_JOINTS`` via the safe slow path.

    Bails out with an actionable hint if the robot is too far from home
    (the ``reset_joint`` 1.5 rad max_joint_delta guard refuses big moves)
    or if FCI is offline.
    """
    print()
    _print_home_pose(HOME_JOINTS)
    print()
    confirm_or_exit()
    print(colour("Moving robot to HOME pose before alignment ...", "36;1"))
    try:
        q_now = controller.get_state().wait()[0].arm_joint_position
        delta = np.asarray(HOME_JOINTS) - np.asarray(q_now)
        max_d = float(np.max(np.abs(delta)))
        worst = int(np.argmax(np.abs(delta)))
        print(f"  current: {[round(float(x), 3) for x in q_now]}")
        print(
            f"  home   : {[round(float(x), 3) for x in HOME_JOINTS]}  "
            f"(max Δ={max_d:.3f} rad on J{worst + 1})"
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
                "  Manually jog the Franka closer to the home pose using\n"
                "  Desk's hand-guide mode (white button on the arm), then\n"
                "  re-run this script."
            )
            sys.exit(2)
    except SystemExit:
        raise
    except Exception:
        # If the probe failed, let reset_joint surface the real error.
        pass

    print("  → calling reset_joint (slow, ~4.5% dynamics) ...", flush=True)
    try:
        controller.reset_joint(list(HOME_JOINTS)).wait()
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
        sys.exit(2)
    time.sleep(0.5)
    q_at_home = controller.get_state().wait()[0].arm_joint_position
    print(
        colour(
            f"  ✅ at home: {[round(float(x), 3) for x in q_at_home]}",
            "32;1",
        )
    )


def main() -> None:
    controller, gello = setup_hardware()
    move_to_home(controller)

    period = 1.0 / REFRESH_HZ
    print()
    print("Aligning joints sequentially against the HOME pose.")
    print("Move ONLY the indicated joint until its delta stays within")
    print(f"±{ALIGN_TOL:.3f} rad for {STABLE_TICKS} frames, then it auto-advances.\n")

    try:
        for j in range(7):
            stable = 0
            entered = time.time()
            while True:
                try:
                    q_robot = controller.get_state().wait()[0].arm_joint_position
                    q_gello, _ = gello.get_action()
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(f"\nread error: {e}", flush=True)
                    time.sleep(period)
                    continue

                delta = wrap_to_pi(float(q_gello[j]) - float(q_robot[j]))
                deg = math.degrees(delta)
                hint = "→ turn GELLO BACK" if delta > 0 else "→ turn GELLO FORWARD"
                if abs(delta) <= ALIGN_TOL:
                    hint = colour("INSIDE TOL", "32;1")
                    stable += 1
                else:
                    stable = 0

                bar = progress_bar(delta, ALIGN_TOL)
                line = (
                    f"  J{j + 1}  robot={q_robot[j]:+.3f}  gello={q_gello[j]:+.3f}  "
                    f"Δ={delta:+.3f} rad ({deg:+6.1f}°)  {bar}  "
                    f"stable={stable}/{STABLE_TICKS}  {hint}    "
                )
                # Refresh in place; cap at terminal width to keep one row.
                sys.stdout.write("\r" + line)
                sys.stdout.flush()

                if stable >= STABLE_TICKS:
                    elapsed = time.time() - entered
                    sys.stdout.write(
                        "\r"
                        + colour(
                            f"  J{j + 1} ✅ aligned (Δ={delta:+.3f} rad, "
                            f"took {elapsed:.1f}s)",
                            "32;1",
                        )
                        + " " * 80
                        + "\n"
                    )
                    sys.stdout.flush()
                    break

                time.sleep(period)
    except KeyboardInterrupt:
        print("\n\naborted by user; no motion was issued.", flush=True)
        return

    # Final sanity print: all 7 deltas
    q_robot = controller.get_state().wait()[0].arm_joint_position
    q_gello, _ = gello.get_action()
    deltas = [wrap_to_pi(float(q_gello[i]) - float(q_robot[i])) for i in range(7)]
    max_d = max(abs(d) for d in deltas)
    worst = int(np.argmax(np.abs(np.asarray(deltas))))
    print()
    print(colour("✅ ALL JOINTS ALIGNED", "32;1"))
    print(f"  per-joint Δ (rad): {[f'{d:+.3f}' for d in deltas]}")
    print(
        f"  max |Δ| = {max_d:.3f} rad on J{worst + 1} "
        f"(stream gate threshold = 0.5 rad — well under)"
    )
    print()
    print("You can now Ctrl-C and start collect_data.sh.  Hold the GELLO")
    print("steady so the alignment doesn't drift before the env wrapper")
    print("opens its own stream loop.")
    print()
    print("Holding here so you don't lose the alignment — Ctrl-C to exit.")
    try:
        while True:
            q_robot = controller.get_state().wait()[0].arm_joint_position
            q_gello, _ = gello.get_action()
            deltas = [
                wrap_to_pi(float(q_gello[i]) - float(q_robot[i])) for i in range(7)
            ]
            md = max(abs(d) for d in deltas)
            worst = int(np.argmax(np.abs(np.asarray(deltas))))
            line = f"  hold: max |Δ| = {md:.3f} rad on J{worst + 1}    " + (
                "OK " if md < 0.5 else "DRIFT "
            )
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
            time.sleep(period)
    except KeyboardInterrupt:
        print("\nexiting.", flush=True)


if __name__ == "__main__":
    main()
