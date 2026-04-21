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

"""Smoke test for the Cartesian-impedance streaming path used by tcp_target
eval — drives one arm's TCP along a sinusoid via ``CartesianImpedanceTracker``,
the same franky session type that ``FrankyController.stream_tcp_impedance``
now uses.

Preconditions on the real robot:
  * Franka is unlocked (press Unlock Joints in Franka Desk).
  * FCI enabled, emergency-stop released.
  * Nothing else is driving the robot (kill any Ray FrankyController first).

Usage::

    /home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \\
        toolkits/realworld_check/test_stream_tcp_impedance.py \\
        --robot_ip 172.16.0.2 --axis z --amplitude_m 0.03 --duration_s 6

Expected: the TCP oscillates ±amplitude along the chosen axis for the
duration window, then the script prints the start/end TCP delta and a
PASS/FAIL hint.
"""

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation as R


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--robot_ip", default="172.16.0.2")
    p.add_argument("--duration_s", type=float, default=6.0)
    p.add_argument("--amplitude_m", type=float, default=0.03)
    p.add_argument(
        "--period_s",
        type=float,
        default=0.1,
        help="How often set_target is called (10 Hz matches env.step).",
    )
    p.add_argument("--freq_hz", type=float, default=0.25, help="Sinusoid frequency")
    p.add_argument("--axis", choices=("x", "y", "z"), default="z")
    p.add_argument("--trans_stiffness", type=float, default=2000.0)
    p.add_argument("--rot_stiffness", type=float, default=200.0)
    args = p.parse_args()

    import franky

    robot = franky.Robot(args.robot_ip)
    robot.recover_from_errors()

    state = robot.state
    O_T_EE = state.O_T_EE
    t0 = np.asarray(O_T_EE.translation, dtype=np.float64)
    q0 = np.asarray(O_T_EE.quaternion, dtype=np.float64)

    axis_idx = {"x": 0, "y": 1, "z": 2}[args.axis]
    print(f"[INFO] connected to {args.robot_ip}")
    print(f"[INFO] start TCP pos={t0.round(4).tolist()}  quat={q0.round(4).tolist()}")
    print(
        f"[INFO] streaming {args.axis}±{args.amplitude_m}m sinusoid @ {args.freq_hz}Hz "
        f"for {args.duration_s}s, set_target every {args.period_s * 1000:.0f}ms"
    )

    n_sent = 0
    with franky.CartesianImpedanceTracker(
        robot,
        translational_stiffness=args.trans_stiffness,
        rotational_stiffness=args.rot_stiffness,
    ) as tracker:
        t_start = time.monotonic()
        while (t := time.monotonic() - t_start) < args.duration_s:
            offset = args.amplitude_m * np.sin(2 * np.pi * args.freq_hz * t)
            target_t = t0.copy()
            target_t[axis_idx] += offset

            T = np.eye(4)
            T[:3, :3] = R.from_quat(q0).as_matrix()
            T[:3, 3] = target_t
            try:
                tracker.set_target(franky.Affine(T))
                n_sent += 1
            except Exception as e:
                print(f"[WARN] t={t:.2f}s set_target failed: {e}")
            time.sleep(args.period_s)

    try:
        robot.join_motion()
    except Exception:
        pass

    final = np.asarray(robot.state.O_T_EE.translation, dtype=np.float64)
    delta = (final - t0).round(4).tolist()
    print(f"[INFO] end   TCP pos={final.round(4).tolist()}")
    print(f"[INFO] delta (end - start) = {delta}  ({n_sent} targets submitted)")
    if np.linalg.norm(np.asarray(delta)) < 1e-3 and n_sent > 5:
        print(
            "[FAIL] TCP did not move despite targets being submitted. "
            "Check: robot unlocked? FCI enabled? impedance gains too soft? "
            "collision triggered (check Franka Desk)?"
        )
    else:
        print("[OK] TCP moved — Cartesian impedance tracking path is live.")


if __name__ == "__main__":
    main()
