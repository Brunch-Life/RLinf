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

"""Diff two ``norm_stats.json`` files produced by ``calculate_norm_stats.py``.

Intended use: after rerunning ``calculate_norm_stats.py`` on the wrap-aware
preprocessed dual-Franka TCP dataset, compare the fresh stats against the
stats shipped with the original (pre-fix) checkpoint to confirm the delta
euler distribution collapsed as expected.

Example::

    python toolkits/dual_franka/compare_norm_stats.py \
        --old /path/to/tcp_v1_14500/Dual-franka-tcp/norm_stats.json \
        --new /path/to/.../Dual-franka-tcp-wrapfix/norm_stats.json

The script prints a per-channel table for the 16-d dual-Franka action layout
``[L_xyz, L_euler, L_pad, L_grip, R_xyz, R_euler, R_pad, R_grip]`` plus a
PASS/FAIL summary that checks the euler channels actually shrunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (index, display name) for the 16-d action layout after the policy's
# _rearrange_state (also matches DeltaActions mask order).
CHANNELS = [
    (0, "L_x"),
    (1, "L_y"),
    (2, "L_z"),
    (3, "L_roll"),
    (4, "L_pitch"),
    (5, "L_yaw"),
    (6, "L_pad"),
    (7, "L_grip_trig"),
    (8, "R_x"),
    (9, "R_y"),
    (10, "R_z"),
    (11, "R_roll"),
    (12, "R_pitch"),
    (13, "R_yaw"),
    (14, "R_pad"),
    (15, "R_grip_trig"),
]
EULER_IDX = [3, 4, 5, 11, 12, 13]
EULER_NAMES = {
    3: "L_roll",
    4: "L_pitch",
    5: "L_yaw",
    11: "R_roll",
    12: "R_pitch",
    13: "R_yaw",
}

# Acceptance thresholds for a successfully wrap-fixed dual-Franka delta
# distribution (values in radians; tuned for 10 Hz GELLO teleop data).
MAX_EULER_Q_RANGE_RAD = 1.0
MAX_EULER_STD_RAD = 0.15


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)["norm_stats"]


def _fmt(x: float) -> str:
    return f"{x:+.4f}"


def _print_section(title: str, key: str, old: dict, new: dict) -> None:
    print(f"\n=== {title} ({key}) ===")
    header = (
        f"{'ch':<16} "
        f"{'OLD q01':>10} {'OLD q99':>10} {'OLD std':>9}  |  "
        f"{'NEW q01':>10} {'NEW q99':>10} {'NEW std':>9}  |  "
        f"{'Δq-range':>10} {'std x':>8}"
    )
    print(header)
    print("-" * len(header))
    for i, name in CHANNELS:
        o_q01, o_q99, o_std = old[key]["q01"][i], old[key]["q99"][i], old[key]["std"][i]
        n_q01, n_q99, n_std = new[key]["q01"][i], new[key]["q99"][i], new[key]["std"][i]
        o_range = o_q99 - o_q01
        n_range = n_q99 - n_q01
        shrink_range = o_range - n_range
        shrink_std = (o_std / n_std) if n_std > 1e-9 else float("inf")
        print(
            f"[{i:2d}] {name:<11} "
            f"{_fmt(o_q01):>10} {_fmt(o_q99):>10} {o_std:>9.4f}  |  "
            f"{_fmt(n_q01):>10} {_fmt(n_q99):>10} {n_std:>9.4f}  |  "
            f"{shrink_range:>+10.4f} {shrink_std:>8.1f}x"
        )


def _verdict(old: dict, new: dict) -> bool:
    print("\n=== verdict (euler action channels) ===")
    all_pass = True
    for i in EULER_IDX:
        name = EULER_NAMES[i]
        o_q01, o_q99 = old["actions"]["q01"][i], old["actions"]["q99"][i]
        n_q01, n_q99 = new["actions"]["q01"][i], new["actions"]["q99"][i]
        o_std, n_std = old["actions"]["std"][i], new["actions"]["std"][i]
        o_range, n_range = o_q99 - o_q01, n_q99 - n_q01

        range_ok = n_range <= MAX_EULER_Q_RANGE_RAD
        std_ok = n_std <= MAX_EULER_STD_RAD
        ok = range_ok and std_ok
        all_pass = all_pass and ok

        flag = "PASS" if ok else "FAIL"
        print(
            f"  [{flag}] {name:<10} "
            f"range {o_range:.3f} -> {n_range:.3f} rad "
            f"(limit {MAX_EULER_Q_RANGE_RAD:.2f}); "
            f"std {o_std:.3f} -> {n_std:.3f} rad "
            f"(limit {MAX_EULER_STD_RAD:.2f})"
        )
    print("\nOverall:", "PASS (wrap fix landed)" if all_pass else "FAIL (see above)")
    return all_pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--old", type=Path, required=True, help="pre-fix norm_stats.json")
    ap.add_argument("--new", type=Path, required=True, help="post-fix norm_stats.json")
    args = ap.parse_args()

    old = _load(args.old)
    new = _load(args.new)

    _print_section("state  (absolute EE layout)", "state", old, new)
    _print_section("action (post-DeltaActions)", "actions", old, new)
    ok = _verdict(old, new)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
