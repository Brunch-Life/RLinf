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

"""Decompose env.step() latency for dual-Franka + GELLO collection.

We collect at a wall-clock ~6 Hz even though the YAML asks for 10 Hz
(``step_frequency=10``).  ``DualFrankaJointEnv.step`` pads to the target
period with ``time.sleep`` *before* reading controller state and camera
frames, so the effective period is
``1/step_frequency + state_rpc + camera_grab + outer_wrapper_cost``.

This bench launches the exact same env stack ``collect_real_data.py``
uses (via a ``DataCollector`` subclass) and times each slice over N
steps, then prints percentile statistics so we can tell which cost is
blowing the budget.

Usage::

    bash ray_utils/realworld/start_ray_node0.sh     # head
    bash ray_utils/realworld/start_ray_node1.sh     # worker (on node 1)
    # hydra: "+bench_seconds=N" overrides the 10s default window
    python toolkits/realworld_check/bench_env_step.py +bench_seconds=10

``collect_data.sh`` must NOT be running — we'd fight for FCI ownership
and the camera handles.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import hydra
import numpy as np

from examples.embodiment.collect_real_data import DataCollector
from rlinf.envs.realworld.franka.dual_franka_joint_env import DualFrankaJointEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.scheduler import Cluster, ComponentPlacement


@dataclass
class Timings:
    """Nested accumulators for each layer we care about."""

    iter_total: list[float] = field(default_factory=list)
    outer_step: list[float] = field(default_factory=list)
    record_step: list[float] = field(default_factory=list)
    inner_step: list[float] = field(default_factory=list)
    franka_step_total: list[float] = field(default_factory=list)
    franka_pre_sleep: list[float] = field(default_factory=list)
    franka_sleep: list[float] = field(default_factory=list)
    franka_state_rpc: list[float] = field(default_factory=list)
    franka_get_obs: list[float] = field(default_factory=list)
    franka_cam_grab: list[float] = field(default_factory=list)


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


def _find_wrapped(env, cls):
    """Walk the wrapper chain to find an instance of ``cls``.

    Handles both single-wrapper envs (``.env`` chain) and gym vector envs
    that hold their per-env wrappers in ``.envs[0]``.
    """
    cur = env
    seen = 0
    while cur is not None and seen < 16:
        if isinstance(cur, cls):
            return cur
        next_env = getattr(cur, "env", None)
        if next_env is None:
            envs_list = getattr(cur, "envs", None)
            if envs_list:
                next_env = envs_list[0]
        cur = next_env
        seen += 1
    return None


def _patch_timings(
    outer_env, franka_env: DualFrankaJointEnv, t: Timings
) -> Callable[[], None]:
    """Monkey-patch timing hooks into the env stack.  Returns an undo fn."""

    # Outer CollectEpisode.step: record_step overhead = outer - inner.
    orig_outer_step = outer_env.step
    inner_env = outer_env.env  # NoAutoResetSyncVectorEnv

    def timed_outer_step(action, **kwargs):
        t0 = time.perf_counter()
        obs, reward, term, trunc, info = inner_env.step(action)
        t1 = time.perf_counter()
        outer_env._record_step(action, obs, reward, term, trunc, info)
        outer_env._maybe_flush(term, trunc)
        t2 = time.perf_counter()
        t.inner_step.append((t1 - t0) * 1000.0)
        t.record_step.append((t2 - t1) * 1000.0)
        t.outer_step.append((t2 - t0) * 1000.0)
        return obs, reward, term, trunc, info

    outer_env.step = timed_outer_step

    # DualFrankaJointEnv.step: break out pre-sleep / sleep / state / obs.
    orig_franka_step = franka_env.step
    orig_get_obs = franka_env._get_observation
    orig_get_cam = franka_env._get_camera_frames

    def timed_get_cam():
        t0 = time.perf_counter()
        out = orig_get_cam()
        t.franka_cam_grab.append((time.perf_counter() - t0) * 1000.0)
        return out

    franka_env._get_camera_frames = timed_get_cam

    def timed_get_obs():
        t0 = time.perf_counter()
        out = orig_get_obs()
        t.franka_get_obs.append((time.perf_counter() - t0) * 1000.0)
        return out

    franka_env._get_observation = timed_get_obs

    def timed_franka_step(action: np.ndarray):
        t0 = time.perf_counter()
        # Replicate DualFrankaJointEnv.step so we can time the internal pad
        # sleep separately from the post-sleep state read.  Keep logic
        # aligned with ``dual_franka_joint_env.py::step``.
        from rlinf.envs.realworld.franka.dual_franka_joint_env import (
            ACTION_DIM_PER_ARM,
            JOINT_DIM_PER_ARM,
            NUM_ARMS,
        )

        action = np.clip(
            action, franka_env.action_space.low, franka_env.action_space.high
        )
        actions = action.reshape(NUM_ARMS, ACTION_DIM_PER_ARM)
        is_gripper_effective = [True, True]
        states = [franka_env._left_state, franka_env._right_state]
        ctrls = [franka_env._left_ctrl, franka_env._right_ctrl]

        if not franka_env.config.is_dummy:
            dt = 1.0 / franka_env.config.step_frequency
            target_joints = []
            for arm in range(NUM_ARMS):
                joint_action = actions[arm, :JOINT_DIM_PER_ARM]
                if franka_env.config.joint_action_mode == "absolute":
                    tj = joint_action.copy()
                else:
                    tj = (
                        states[arm].arm_joint_position
                        + joint_action * franka_env.config.joint_action_scale
                    )
                tj = franka_env._clip_joints_to_limits(tj)
                tj = franka_env._clip_joint_velocity(
                    tj, states[arm].arm_joint_position, dt
                )
                target_joints.append(tj)
            for arm in range(NUM_ARMS):
                gripper_val = (
                    actions[arm, JOINT_DIM_PER_ARM] * franka_env.config.action_scale[2]
                )
                is_gripper_effective[arm] = franka_env._gripper_action(
                    ctrls[arm], states[arm], gripper_val
                )
            if not franka_env.config.teleop_direct_stream:
                left_f = ctrls[0].move_joints(target_joints[0].astype(np.float32))
                right_f = ctrls[1].move_joints(target_joints[1].astype(np.float32))
                left_f.wait()
                right_f.wait()

        franka_env._num_steps += 1
        t_pre_sleep = time.perf_counter()

        if not franka_env.config.is_dummy:
            if not franka_env.config.teleop_direct_stream:
                step_time = t_pre_sleep - t0
                sleep_for = max(
                    0.0, (1.0 / franka_env.config.step_frequency) - step_time
                )
                time.sleep(sleep_for)
            t_post_sleep = time.perf_counter()
            left_st_f = ctrls[0].get_state()
            right_st_f = ctrls[1].get_state()
            franka_env._left_state = left_st_f.wait()[0]
            franka_env._right_state = right_st_f.wait()[0]
            t_post_state = time.perf_counter()
        else:
            t_post_sleep = t_pre_sleep
            t_post_state = t_pre_sleep

        observation = franka_env._get_observation()
        reward = franka_env._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            franka_env._success_hold_counter >= franka_env.config.success_hold_steps
        )
        truncated = franka_env._num_steps >= franka_env.config.max_num_steps
        t_end = time.perf_counter()

        t.franka_pre_sleep.append((t_pre_sleep - t0) * 1000.0)
        t.franka_sleep.append((t_post_sleep - t_pre_sleep) * 1000.0)
        t.franka_state_rpc.append((t_post_state - t_post_sleep) * 1000.0)
        t.franka_step_total.append((t_end - t0) * 1000.0)

        return observation, reward, terminated, truncated, {}

    franka_env.step = timed_franka_step

    def undo():
        outer_env.step = orig_outer_step
        franka_env.step = orig_franka_step
        franka_env._get_observation = orig_get_obs
        franka_env._get_camera_frames = orig_get_cam

    return undo


class BenchCollector(DataCollector):
    """Reuse DataCollector's env construction, replace ``run`` with a timing
    loop.  We send zero actions — the 1 kHz GELLO daemon is what actually
    drives the arms in ``teleop_direct_stream`` mode, so the bench sees the
    same RPC / camera load the real collect loop sees.
    """

    def bench(self, seconds: float = 10.0) -> None:
        self.env.reset()
        self.log_info("[bench] warming up for 1s")
        t_warm = time.perf_counter() + 1.0
        while time.perf_counter() < t_warm:
            self.env.step(np.zeros((1, self.action_dim)))

        # Find inner components for patching.
        outer = self.env  # CollectEpisode (or unwrapped)
        franka = _find_wrapped(self.env, DualFrankaJointEnv)
        if franka is None:
            raise RuntimeError("DualFrankaJointEnv not found in env stack")
        is_collect_wrapped = isinstance(self.env, CollectEpisode)

        t = Timings()
        undo = _patch_timings(outer, franka, t) if is_collect_wrapped else None
        if undo is None:
            # Fall back: only time the whole step if CollectEpisode is absent.
            orig = self.env.step

            def timed(action, **kw):
                t0 = time.perf_counter()
                out = orig(action, **kw)
                t.outer_step.append((time.perf_counter() - t0) * 1000.0)
                return out

            self.env.step = timed

        # Match DataCollector.run: outer-loop pad so overall rate ~= target.
        target_period = self._target_step_period
        self.log_info(
            f"[bench] timing {seconds:.1f}s of env.step() "
            f"(outer pad target={target_period})"
        )
        t_end = time.perf_counter() + seconds
        while time.perf_counter() < t_end:
            iter_start = time.perf_counter()
            self.env.step(np.zeros((1, self.action_dim)))
            if target_period is not None:
                elapsed = time.perf_counter() - iter_start
                sleep_for = target_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
            t.iter_total.append((time.perf_counter() - iter_start) * 1000.0)

        print(f"\n[bench] step latency breakdown over {seconds:.1f}s:")
        _report("iter total (with outer pad)", t.iter_total)
        _report("outer env.step (CollectEpisode)", t.outer_step)
        _report("  CollectEpisode._record_step", t.record_step)
        _report("  inner vec env.step", t.inner_step)
        _report("    DualFrankaJointEnv.step", t.franka_step_total)
        _report("      action + gripper calls", t.franka_pre_sleep)
        _report("      time.sleep to period", t.franka_sleep)
        _report("      get_state RPC (both arms)", t.franka_state_rpc)
        _report("      _get_observation total", t.franka_get_obs)
        _report("        _get_camera_frames", t.franka_cam_grab)

        if t.iter_total:
            rate = 1000.0 / np.mean(t.iter_total)
            if target_period:
                print(
                    f"\n  effective step rate ≈ {rate:.2f} Hz "
                    f"(target {1.0 / target_period:.1f} Hz)"
                )
            else:
                print(f"\n  effective step rate ≈ {rate:.2f} Hz (no pad)")


@hydra.main(
    version_base="1.1",
    config_path="../../examples/embodiment/config",
    config_name="realworld_collect_data_gello_joint_dual_franka",
)
def main(cfg):
    seconds = float(cfg.get("bench_seconds", 10.0))

    # collect_data.sh normally injects ``runner.logger.log_path=<ts>`` via CLI;
    # we're bypassing it so stamp a throwaway dir ourselves.  DataCollector's
    # __init__ joins this for the ReplayBuffer path (unused here because the
    # bench never terminates an episode) and the data-collection save_dir.
    import os as _os
    import tempfile as _tempfile

    if cfg.runner.logger.get("log_path") is None:
        bench_dir = _tempfile.mkdtemp(prefix="rlinf_bench_")
        cfg.runner.logger.log_path = bench_dir
        if cfg.env.eval.get("data_collection") is not None:
            cfg.env.eval.data_collection.save_dir = _os.path.join(
                bench_dir, "collected_data"
            )
        print(f"[bench] using throwaway log_path={bench_dir}")

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    collector = BenchCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.bench(seconds).wait()


if __name__ == "__main__":
    main()
