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

"""Isolate teleop stutter: GELLO-serial latency vs Franka RPC latency.

Two independent benches. Run each in its own shell; ``collect_data.sh``
must **not** be running (either side would fight for the same ports /
FCI ownership).

Phase A — ``gello`` (no Ray, no Franka needed)::

    python toolkits/realworld_check/bench_teleop_latency.py gello --seconds 5

    Opens both GELLO serial ports, spins ``agent.get_action()`` as fast
    as possible, reports per-arm and paired read latency.  If p95 here
    is > ~2 ms the Dynamixel bus is the bottleneck.

Phase B — ``franka`` (needs Ray up on both nodes, no active collect)::

    bash ray_utils/realworld/start_ray_node0.sh
    bash ray_utils/realworld/start_ray_node1.sh   # on node 1
    python toolkits/realworld_check/bench_teleop_latency.py franka \
        --seconds 5

    Spawns fresh ``FrankyController`` actors (left on node 0, right on
    node 1 — same placement as ``DualFrankaJointEnv``), seeds the joint
    impedance tracker, then fires ``move_joints(current_q)`` targets at
    1 kHz and times each arm's RPC round-trip independently.  If p95
    here is > ~3 ms or max spikes > 20 ms the Ray / network path is the
    bottleneck.

The two numbers together tell you whether the "一卡一卡" comes from the
GELLO-side read thread feeding stale data or from the Franka-side RPC
queue backing up under load.
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import numpy as np

LEFT_GELLO_PORT = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAJEDPC-if00-port0"
)
RIGHT_GELLO_PORT = (
    "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTAM5DHY-if00-port0"
)

LEFT_ROBOT_IP = "172.16.0.2"
RIGHT_ROBOT_IP = "172.16.0.2"
LEFT_GRIPPER_PORT = "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIT8PU-if00-port0"
RIGHT_GRIPPER_PORT = "/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DAAIR00N-if00-port0"


def _report(label: str, samples_ms: list[float]) -> None:
    if not samples_ms:
        print(f"  {label:32s} (no samples)")
        return
    a = np.asarray(samples_ms)
    print(
        f"  {label:32s} n={len(a):6d}  "
        f"mean={a.mean():6.2f}ms  "
        f"p50={np.percentile(a, 50):6.2f}  "
        f"p95={np.percentile(a, 95):6.2f}  "
        f"p99={np.percentile(a, 99):6.2f}  "
        f"max={a.max():6.2f}"
    )


def bench_gello(seconds: float, left_port: str, right_port: str) -> None:
    """Measure Dynamixel read latency on both GELLOs."""
    from gello_teleop.gello_teleop_agent import GelloTeleopAgent

    print(f"[gello] opening left  = {left_port}")
    left = GelloTeleopAgent(port=left_port)
    print(f"[gello] opening right = {right_port}")
    right = GelloTeleopAgent(port=right_port)

    # Warm up the serial link before timing.
    for _ in range(20):
        left.get_action()
        right.get_action()

    dt_l: list[float] = []
    dt_r: list[float] = []
    dt_pair: list[float] = []
    t_end = time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        left.get_action()
        t1 = time.perf_counter()
        right.get_action()
        t2 = time.perf_counter()
        dt_l.append((t1 - t0) * 1000.0)
        dt_r.append((t2 - t1) * 1000.0)
        dt_pair.append((t2 - t0) * 1000.0)

    print(f"\n[gello] {seconds:.1f}s tight-loop get_action() latency (ms):")
    _report("left  get_action", dt_l)
    _report("right get_action", dt_r)
    _report("pair (sequential)", dt_pair)
    if dt_pair:
        rate = 1000.0 / np.mean(dt_pair)
        print(f"  effective pair rate ≈ {rate:.0f} Hz")


def _current_q(ctrl) -> np.ndarray:
    state = ctrl.get_state().wait()[0]
    return np.asarray(state.arm_joint_position, dtype=np.float64)


def bench_franka(
    seconds: float,
    left_ip: str,
    right_ip: str,
    left_grip: str,
    right_grip: str,
    side: str,
) -> None:
    """Measure ``FrankyController.move_joints`` RPC latency per arm."""
    import ray

    from rlinf.envs.realworld.franka.franky_controller import FrankyController

    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level="ERROR")

    left_ctrl = None
    right_ctrl = None

    if side in ("both", "left"):
        print(f"[franka] launching left ctrl on node 0 ({left_ip})")
        left_ctrl = FrankyController.launch_controller(
            robot_ip=left_ip,
            env_idx=8100,
            node_rank=0,
            worker_rank=99,
            gripper_type="robotiq",
            gripper_connection=left_grip,
        )

    if side in ("both", "right"):
        print(f"[franka] launching right ctrl on node 1 ({right_ip})")
        right_ctrl = FrankyController.launch_controller(
            robot_ip=right_ip,
            env_idx=9100,
            node_rank=1,
            worker_rank=99,
            gripper_type="robotiq",
            gripper_connection=right_grip,
        )

    # Wait for both controllers to publish a valid state.
    for name, ctrl in (("left", left_ctrl), ("right", right_ctrl)):
        if ctrl is None:
            continue
        t0 = time.perf_counter()
        while not ctrl.is_robot_up().wait()[0]:
            time.sleep(0.2)
            if time.perf_counter() - t0 > 30:
                raise RuntimeError(f"{name} controller never came up")
        print(f"[franka] {name} ctrl ready")

    q_l: Optional[np.ndarray] = None
    q_r: Optional[np.ndarray] = None
    if left_ctrl is not None:
        q_l = _current_q(left_ctrl).astype(np.float32)
        print(f"[franka] left  seed q = {q_l}")
    if right_ctrl is not None:
        q_r = _current_q(right_ctrl).astype(np.float32)
        print(f"[franka] right seed q = {q_r}")

    # Warm up so the impedance tracker exists and RPC path is primed.
    for _ in range(50):
        if left_ctrl is not None:
            left_ctrl.move_joints(q_l)
        if right_ctrl is not None:
            right_ctrl.move_joints(q_r)
        time.sleep(0.001)

    dt_l: list[float] = []
    dt_r: list[float] = []
    dt_pair: list[float] = []
    t_end = time.perf_counter() + seconds
    period = 0.001
    while time.perf_counter() < t_end:
        loop_start = time.perf_counter()
        lf = left_ctrl.move_joints(q_l) if left_ctrl is not None else None
        rf = right_ctrl.move_joints(q_r) if right_ctrl is not None else None
        t_sent = time.perf_counter()
        if lf is not None:
            lf.wait()
        t_l = time.perf_counter()
        if rf is not None:
            rf.wait()
        t_r = time.perf_counter()
        if lf is not None:
            dt_l.append((t_l - t_sent) * 1000.0)
        if rf is not None:
            dt_r.append((t_r - t_l) * 1000.0)
        dt_pair.append((t_r - loop_start) * 1000.0)

        elapsed = time.perf_counter() - loop_start
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    print(f"\n[franka] {seconds:.1f}s move_joints round-trip latency (ms):")
    _report("left  (node 0 .wait)", dt_l)
    _report("right (node 1 .wait)", dt_r)
    _report("pair (send + both waits)", dt_pair)
    if dt_pair:
        rate = 1000.0 / np.mean(dt_pair)
        print(f"  effective pair rate ≈ {rate:.0f} Hz  (stream period target 1 kHz)")


def bench_rayrpc(seconds: float) -> None:
    """Raw cross-node Ray RPC latency — empty actor method, no franky.

    Isolates Ray/network overhead from everything FrankyController-specific
    (franky init, impedance tracker, libfranka). If this bench is also slow
    on node 1, the root cause is the Ray transport, not franky.
    """
    import ray

    from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker

    if not ray.is_initialized():
        ray.init(address="auto", log_to_driver=False, logging_level="ERROR")

    class _PingActor(Worker):
        def ping(self) -> int:
            return 0

    cluster = Cluster()

    print("[rayrpc] launching ping actor on node 0")
    left = _PingActor.create_group().launch(
        cluster=cluster,
        placement_strategy=NodePlacementStrategy(node_ranks=[0]),
        name="BenchPing-0",
    )
    print("[rayrpc] launching ping actor on node 1")
    right = _PingActor.create_group().launch(
        cluster=cluster,
        placement_strategy=NodePlacementStrategy(node_ranks=[1]),
        name="BenchPing-1",
    )

    # Warm up
    for _ in range(50):
        left.ping().wait()
        right.ping().wait()

    dt_l: list[float] = []
    dt_r: list[float] = []
    t_end = time.perf_counter() + seconds
    period = 0.001
    while time.perf_counter() < t_end:
        t0 = time.perf_counter()
        left.ping().wait()
        t1 = time.perf_counter()
        right.ping().wait()
        t2 = time.perf_counter()
        dt_l.append((t1 - t0) * 1000.0)
        dt_r.append((t2 - t1) * 1000.0)
        elapsed = t2 - t0
        sleep_for = period - elapsed
        if sleep_for > 0:
            time.sleep(sleep_for)

    print(f"\n[rayrpc] {seconds:.1f}s empty-method round-trip latency (ms):")
    _report("left  (node 0 local)", dt_l)
    _report("right (node 1 cross-node)", dt_r)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)

    g = sub.add_parser("gello", help="GELLO serial read latency (no Ray needed)")
    g.add_argument("--seconds", type=float, default=5.0)
    g.add_argument("--left-port", default=LEFT_GELLO_PORT)
    g.add_argument("--right-port", default=RIGHT_GELLO_PORT)

    f = sub.add_parser("franka", help="FrankyController move_joints RPC latency")
    f.add_argument("--seconds", type=float, default=5.0)
    f.add_argument("--left-ip", default=LEFT_ROBOT_IP)
    f.add_argument("--right-ip", default=RIGHT_ROBOT_IP)
    f.add_argument("--left-grip", default=LEFT_GRIPPER_PORT)
    f.add_argument("--right-grip", default=RIGHT_GRIPPER_PORT)
    f.add_argument(
        "--side",
        choices=("both", "left", "right"),
        default="both",
        help="benchmark one arm in isolation to split node-0 vs node-1 cost",
    )

    r = sub.add_parser("rayrpc", help="Raw cross-node Ray RPC latency (no franky)")
    r.add_argument("--seconds", type=float, default=5.0)

    args = ap.parse_args()

    if args.mode == "gello":
        bench_gello(args.seconds, args.left_port, args.right_port)
    elif args.mode == "franka":
        bench_franka(
            args.seconds,
            args.left_ip,
            args.right_ip,
            args.left_grip,
            args.right_grip,
            args.side,
        )
    elif args.mode == "rayrpc":
        bench_rayrpc(args.seconds)


if __name__ == "__main__":
    main()
