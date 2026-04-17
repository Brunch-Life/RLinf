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

"""Live tuner for joint-impedance PD gains (Kq / Kqd).

Attaches to a running collect process's ``FrankyController`` Ray actors
(one per arm) and hot-updates gains via ``reconfigure_compliance_params``.
The tracker exponentially smooths gain changes so taps feel immediate
without jerking the robot.

Run while collect is live::

    python toolkits/realworld_check/tune_impedance.py

Per-arm actor names default to the ones ``DualFrankaJointEnv`` creates
(``FrankyController-0-0`` left, ``FrankyController-0-1000`` right);
override via ``TUNE_LEFT_NAME`` / ``TUNE_RIGHT_NAME`` if your env_idx /
worker_rank differ.

Commands (single key → per-joint taps, piano-row layout):
    q w e r t y u     increase J1..J7 Kq   (by the step factor)
    a s d f g h j     decrease J1..J7 Kq
    Q W E R T Y U     increase J1..J7 Kqd
    A S D F G H J     decrease J1..J7 Kqd
    kq+  / kq-        scale all joints' Kq
    kqd+ / kqd-       scale all joints' Kqd
    arm left|right|both   pick which arm future commands touch (default: both)
    step <factor>     set multiplicative step (default 1.1 ≈ ±10% per tap)
    show              print current Kq/Kqd for the selected arm(s)
    set kq <v1..v7>   overwrite Kq with 7 absolute values
    set kqd <v1..v7>  overwrite Kqd with 7 absolute values
    reset             revert to hand-tuned defaults (see DEFAULT_KQ/KQD)
    help | quit

Safety: per-joint values are clipped to [0.1×, 5×] of ``DEFAULT_KQ`` /
``DEFAULT_KQD``.  The clamp tracks the startup defaults, so re-capture
those after each tuning session to keep headroom on both sides of the
operating point.  Gains stay in effect on the robots after quitting.
"""

from __future__ import annotations

import os
import time

import numpy as np
import ray

from rlinf.envs.realworld.franka.franky_controller import FrankyController
from rlinf.scheduler.worker import WorkerGroup

# Startup values (pushed on attach + target of ``reset``) AND safety-clamp
# anchor: per-joint step commands are clipped to [SCALE_MIN×, SCALE_MAX×]
# of these values.  Captured from a hand-tuned session — re-measure and
# update here when the tuning drifts so the clamp keeps headroom on both
# sides of the current operating point.
DEFAULT_KQ = np.array([103.75, 265.734, 227.273, 221.445, 13.5, 12.818, 5.134])
DEFAULT_KQD = np.array([16.7, 40.263, 25.0, 12.862, 1.5, 2.0, 1.331])

SCALE_MIN, SCALE_MAX = 0.1, 5.0

HELP = """\
commands:
  q w e r t y u     J1..J7 Kq  ↑    (piano-row per-joint tap)
  a s d f g h j     J1..J7 Kq  ↓
  Q W E R T Y U     J1..J7 Kqd ↑
  A S D F G H J     J1..J7 Kqd ↓
  kq+ / kq-         scale all joints' Kq
  kqd+ / kqd-       scale all joints' Kqd
  arm left|right|both   which arm(s) to tune (default: both)
  step <factor>     multiplicative step (default 1.1)
  show              print current gains
  set kq  <v1..v7>  overwrite Kq  (7 absolute values)
  set kqd <v1..v7>  overwrite Kqd (7 absolute values)
  reset             revert to hand-tuned defaults
  help | quit
"""

# Piano-row per-joint bindings (index 0..6 → J1..J7).
KQ_UP_KEYS = "qwertyu"
KQ_DN_KEYS = "asdfghj"
KQD_UP_KEYS = "QWERTYU"
KQD_DN_KEYS = "ASDFGHJ"


class Arm:
    def __init__(self, label: str, group_name: str):
        self.label = label
        self.group_name = group_name
        self.group = WorkerGroup.from_group_name(FrankyController, group_name)
        # ``from_group_name`` sets ``_group_size`` but not ``_world_size``;
        # WorkerGroupFunc.execute() asserts on the latter, so mirror it.
        self.group._world_size = self.group._group_size
        self.kq = DEFAULT_KQ.copy()
        self.kqd = DEFAULT_KQD.copy()

    def push(self) -> None:
        self.group.reconfigure_compliance_params(
            {"Kq": self.kq.tolist(), "Kqd": self.kqd.tolist()}
        ).wait()

    def scale(self, which: str, idx: int | None, factor: float) -> None:
        target = self.kq if which == "Kq" else self.kqd
        ref = DEFAULT_KQ if which == "Kq" else DEFAULT_KQD
        lo, hi = ref * SCALE_MIN, ref * SCALE_MAX
        if idx is None:
            new = target * factor
        else:
            new = target.copy()
            new[idx] = target[idx] * factor
        target[:] = np.clip(new, lo, hi)

    def show(self) -> None:
        print(f"[{self.label}]")
        print(f"  Kq  = {np.round(self.kq, 2).tolist()}")
        print(f"  Kqd = {np.round(self.kqd, 3).tolist()}")


def _attach() -> dict[str, Arm]:
    left_name = os.environ.get("TUNE_LEFT_NAME", "FrankyController-0-0")
    right_name = os.environ.get("TUNE_RIGHT_NAME", "FrankyController-0-1000")
    arms: dict[str, Arm] = {}
    for label, name in (("left", left_name), ("right", right_name)):
        try:
            arms[label] = Arm(label, name)
            print(f"attached {label:>5} ← {name}")
        except (AssertionError, Exception) as e:
            print(f"skip {label} ({name}): {e}")
    if not arms:
        raise RuntimeError(
            "no FrankyController actors found; is collect running and are the "
            "names correct? override via TUNE_LEFT_NAME / TUNE_RIGHT_NAME."
        )
    return arms


def _parse_set_cmd(parts: list[str]) -> tuple[str, np.ndarray] | None:
    if len(parts) != 9 or parts[1] not in ("kq", "kqd"):
        print("usage: set kq|kqd <v1 v2 v3 v4 v5 v6 v7>")
        return None
    try:
        vals = np.array([float(x) for x in parts[2:9]])
    except ValueError:
        print("values must be floats")
        return None
    return ("Kq" if parts[1] == "kq" else "Kqd"), vals


def _parse_scale_cmd(line: str) -> tuple[str, int | None, bool] | None:
    """Return (which, idx_or_None, is_up) for per-joint keys and bulk aliases."""
    if line in ("kq+", "kq-"):
        return "Kq", None, line.endswith("+")
    if line in ("kqd+", "kqd-"):
        return "Kqd", None, line.endswith("+")
    if len(line) != 1:
        return None
    ch = line
    if ch in KQ_UP_KEYS:
        return "Kq", KQ_UP_KEYS.index(ch), True
    if ch in KQ_DN_KEYS:
        return "Kq", KQ_DN_KEYS.index(ch), False
    if ch in KQD_UP_KEYS:
        return "Kqd", KQD_UP_KEYS.index(ch), True
    if ch in KQD_DN_KEYS:
        return "Kqd", KQD_DN_KEYS.index(ch), False
    return None


def main() -> None:
    if not ray.is_initialized():
        try:
            ray.init(address="auto", log_to_driver=False, logging_level="ERROR")
        except Exception as e:
            print(f"ray.init(address='auto') failed: {e}")
            print("start the collect process first so a Ray cluster exists.")
            return

    try:
        arms = _attach()
    except RuntimeError as e:
        print(str(e))
        return

    # Push defaults once so the tuner's tracked state matches the robot.
    for arm in arms.values():
        try:
            arm.push()
        except Exception as e:
            print(f"[{arm.label}] initial push failed: {e}")

    selected: list[str] = list(arms.keys())
    step = 1.1

    print(HELP)
    print(f"selected = {selected}, step = {step}")

    while True:
        try:
            line = input("tune> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        try:
            # `q` now means "J1 Kq ↑", so quit needs an unambiguous keyword.
            if line in ("quit", "exit"):
                break
            if line == "help":
                print(HELP)
                continue
            if line == "show":
                for name in selected:
                    arms[name].show()
                continue
            if line == "reset":
                for name in selected:
                    arms[name].kq[:] = DEFAULT_KQ
                    arms[name].kqd[:] = DEFAULT_KQD
                    arms[name].push()
                print(f"reset {selected} to hand-tuned defaults")
                continue
            if line.startswith("arm "):
                rest = line[4:].strip()
                if rest == "both":
                    selected = list(arms.keys())
                elif rest in arms:
                    selected = [rest]
                else:
                    print(f"unknown arm: {rest!r} (have {list(arms.keys())})")
                    continue
                print(f"selected = {selected}")
                continue
            if line.startswith("step "):
                f = float(line[5:].strip())
                if f <= 1.0:
                    print("step factor must be > 1.0 (multiplicative)")
                    continue
                step = f
                print(f"step = {step}")
                continue
            if line.startswith("set "):
                parsed = _parse_set_cmd(line.split())
                if parsed is None:
                    continue
                which, vals = parsed
                for name in selected:
                    target = arms[name].kq if which == "Kq" else arms[name].kqd
                    target[:] = vals
                    arms[name].push()
                for name in selected:
                    arms[name].show()
                continue

            parsed = _parse_scale_cmd(line)
            if parsed is None:
                print(f"unknown: {line!r}  (type 'help')")
                continue
            which, idx, is_up = parsed
            factor = step if is_up else 1.0 / step
            label = "all" if idx is None else f"J{idx + 1}"
            for name in selected:
                arms[name].scale(which, idx, factor)
                arms[name].push()
            arrow = "↑" if is_up else "↓"
            for name in selected:
                arm = arms[name]
                curr = arm.kq if which == "Kq" else arm.kqd
                shown = np.round(curr if idx is None else curr[idx], 3)
                print(f"[{name}] {which} {label} {arrow}×{factor:.3f} → {shown}")
        except Exception as e:
            print(f"command failed: {e}")

        time.sleep(0.02)

    print("bye (gains remain active on the robots)")


if __name__ == "__main__":
    main()
