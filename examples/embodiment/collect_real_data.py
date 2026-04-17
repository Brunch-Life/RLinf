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


import os
import threading
import time
from typing import Optional

import hydra
import numpy as np
import ray
import torch

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

STATUS_BUS_NAME = "collect_status_bus"


@ray.remote(num_cpus=0)
class StatusBus:
    """Tiny shared state object: DataCollector pushes, driver renderer polls.

    Exists because Ray's log monitor batches worker stdout at ~500 ms and
    breaks tqdm's ``\\r`` refresh — we can't get a live bar by printing from
    the worker.  The driver has its own TTY, so we ship state to a named
    actor and let the driver render.
    """

    def __init__(self):
        self._state: dict = {}

    def set(self, state: dict) -> None:
        self._state = dict(state)

    def get(self) -> dict:
        return dict(self._state)


class DriverStatusRenderer:
    """Driver-side background thread that polls ``StatusBus`` and paints a
    single refreshing line on ``/dev/tty``.

    ``collect_data.sh`` pipes the driver's stdout through ``tee``, so
    ``sys.stdout`` is not a terminal and tqdm would fall back to per-update
    newlines.  Writing to ``/dev/tty`` bypasses the shell redirect and goes
    straight to the controlling terminal, so ``\\r`` refresh works and the
    ``run_embodiment.log`` file stays clean.
    """

    _BAR_WIDTH = 20
    _LINE_WIDTH = 110

    def __init__(self, bus, target: int, poll_interval_s: float = 0.1):
        self._bus = bus
        self._target = max(1, int(target))
        self._poll_interval_s = poll_interval_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        try:
            self._tty = open("/dev/tty", "w", buffering=1)
        except OSError:
            self._tty = None
        self._t0 = time.time()

    def start(self) -> None:
        if self._tty is None:
            # No controlling terminal (e.g. headless CI); renderer is a no-op.
            return
        self._thread = threading.Thread(
            target=self._loop, name="DriverStatusRenderer", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._tty is not None:
            try:
                self._tty.write("\n")
                self._tty.flush()
                self._tty.close()
            except OSError:
                pass

    def _render(self, state: dict) -> str:
        phase = state.get("phase", "init")
        saved = int(state.get("saved", 0))
        frames = int(state.get("frames", 0))
        ep_t = float(state.get("ep_t", 0.0))
        last_event = state.get("last_event") or ""
        last_event_ts = float(state.get("last_event_ts", 0.0))

        # Hold the last event on screen for 3 s, then blank it out.
        if last_event and (time.time() - last_event_ts) > 3.0:
            last_event = ""

        frac = saved / self._target
        filled = int(round(self._BAR_WIDTH * frac))
        bar = "#" * filled + "-" * (self._BAR_WIDTH - filled)
        total_elapsed = time.time() - self._t0

        return (
            f"[{bar}] {saved:>3}/{self._target} "
            f"phase={phase:<3} frames={frames:>4} ep_t={ep_t:>5.1f}s "
            f"elapsed={total_elapsed:>6.1f}s "
            f"last={last_event:<12}"
        )

    def _loop(self) -> None:
        last_event_ts_seen = 0.0
        while not self._stop.is_set():
            try:
                state = ray.get(self._bus.get.remote())
            except Exception:
                state = {}
            line = self._render(state).ljust(self._LINE_WIDTH)
            try:
                self._tty.write("\r" + line)
                self._tty.flush()
            except OSError:
                break

            # Also surface state transitions as durable log lines on the real
            # stdout (which the shell tee captures to run_embodiment.log).
            ev_ts = float(state.get("last_event_ts", 0.0))
            ev = state.get("last_event") or ""
            if ev and ev_ts and ev_ts != last_event_ts_seen:
                self._tty.write(
                    "\n"
                    + time.strftime("%H:%M:%S")
                    + f" [keyboard] {ev}".ljust(self._LINE_WIDTH)
                    + "\n"
                )
                self._tty.flush()
                last_event_ts_seen = ev_ts

            time.sleep(self._poll_interval_s)


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        if self.cfg.env.eval.get("data_collection", None) and getattr(
            self.cfg.env.eval.data_collection, "enabled", False
        ):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=self.cfg.env.eval.data_collection.save_dir,
                # rank=self._rank,
                # num_envs=1,
                export_format=getattr(
                    self.cfg.env.eval.data_collection, "export_format", "pickle"
                ),
                robot_type=getattr(
                    self.cfg.env.eval.data_collection, "robot_type", "panda"
                ),
                fps=getattr(self.cfg.env.eval.data_collection, "fps", 10),
                only_success=getattr(
                    self.cfg.env.eval.data_collection, "only_success", False
                ),
                finalize_interval=getattr(
                    self.cfg.env.eval.data_collection, "finalize_interval", 100
                ),
            )

        # Resolve the per-env action dim from the wrapped env's action space.
        # SyncVectorEnv exposes a batched space whose trailing dim is the
        # per-env action dim, which already accounts for:
        #   - GripperCloseEnv (single-arm, no_gripper=True) → 6
        #   - single-arm with gripper                       → 7
        #   - DualFrankaEnv (always)                        → 14
        # Using the space directly avoids hardcoding shapes per robot type.
        self.action_dim = int(self.env.action_space.shape[-1])

        # Initialize TrajectoryReplayBuffer
        # Change directory name to 'demos' as requested
        buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

        # Attach to the driver-side status bus (created in main()).  Absent
        # actor (e.g. standalone debug run) is non-fatal — we just skip pushes.
        try:
            self._bus = ray.get_actor(STATUS_BUS_NAME)
        except ValueError:
            self._bus = None

    def _process_obs(self, obs):
        """
        Process observations to match the format expected by EmbodiedRolloutResult.
        """
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)

            val = val.cpu()

            # Map keys: 'images' -> 'main_images', others remain
            if "images" == key:
                ret_obs["main_images"] = val.clone()  # Keep uint8
            else:
                ret_obs[key] = val.clone()

        return ret_obs

    @staticmethod
    def _unwrap_info_scalar(val):
        """Info fields from SyncVectorEnv arrive as len-1 arrays/lists."""
        if val is None:
            return None
        if isinstance(val, (list, tuple)):
            return val[0] if val else None
        if isinstance(val, np.ndarray):
            return val.reshape(-1)[0] if val.size else None
        return val

    def _push_status(self, **state) -> None:
        if self._bus is None:
            return
        # fire-and-forget: ignore the returned ObjectRef so we don't block on
        # RPC latency from the env step loop.
        self._bus.set.remote(state)

    def run(self):
        obs, _ = self.env.reset()
        success_cnt = 0
        self.log_info("[keyboard] ready — press 'a' to start recording")

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
        )

        current_obs_processed = self._process_obs(obs)

        ep_start_time: Optional[float] = None
        ep_step_count: int = 0
        last_event_msg: str = ""
        last_event_ts: float = 0.0
        self._push_status(
            phase="pre",
            saved=0,
            frames=0,
            ep_t=0.0,
            last_event="",
            last_event_ts=0.0,
        )

        while success_cnt < self.num_data_episodes:
            # Zero "no-op" placeholder action; teleop wrappers overwrite this
            # via info["intervene_action"] when the operator is active.
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            kb_event = self._unwrap_info_scalar(info.get("keyboard_event"))
            kb_phase = self._unwrap_info_scalar(info.get("keyboard_phase")) or "pre"
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")
                last_event_msg = kb_event
                last_event_ts = time.time()
                if kb_event in ("start", "restart"):
                    ep_start_time = time.time()
                    ep_step_count = 0

            if kb_phase == "rec":
                ep_step_count += 1
                ep_t = time.time() - ep_start_time if ep_start_time else 0.0
            else:
                ep_t = 0.0

            self._push_status(
                phase=kb_phase,
                saved=success_cnt,
                frames=ep_step_count,
                ep_t=ep_t,
                last_event=last_event_msg,
                last_event_ts=last_event_ts,
            )

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            done = terminated | truncated

            # --- Construct ChunkStepResult ---
            # Prepare action tensor [1, action_dim]
            if isinstance(action, torch.Tensor):
                action_tensor = action.float().cpu()
            else:
                action_tensor = torch.from_numpy(action).float()

            if action_tensor.ndim == 1:
                action_tensor = action_tensor.unsqueeze(0)

            # Reward and Done [1, 1]
            if isinstance(reward, torch.Tensor):
                reward_tensor = reward.float().cpu()
            else:
                reward_tensor = torch.tensor(reward).float()
            if reward_tensor.ndim == 1:
                reward_tensor = reward_tensor.unsqueeze(1)

            if isinstance(done, torch.Tensor):
                done_tensor = done.bool().cpu()
            else:
                done_tensor = torch.tensor(done).bool()
            if done_tensor.ndim == 1:
                done_tensor = done_tensor.unsqueeze(1)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=done_tensor,
                truncations=torch.zeros_like(done_tensor),
                forward_inputs={"action": action_tensor},
            )

            current_rollout.append_step_result(step_result)
            current_rollout.append_transitions(
                curr_obs=current_obs_processed, next_obs=next_obs_processed
            )

            obs = next_obs
            current_obs_processed = next_obs_processed

            if done:
                r_val = (
                    reward[0]
                    if hasattr(reward, "__getitem__") and len(reward) > 0
                    else reward
                )
                if isinstance(r_val, torch.Tensor):
                    r_val = r_val.item()

                self.total_cnt += 1

                if r_val >= 0.5:
                    success_cnt += 1

                    self.log_info(
                        f"Success: {r_val}. Total: {success_cnt}/{self.num_data_episodes}"
                    )

                    # Save Trajectory to the 'demos' directory
                    trajectory = current_rollout.to_trajectory()
                    trajectory.intervene_flags = torch.ones_like(
                        trajectory.intervene_flags
                    )
                    self.buffer.add_trajectories([trajectory])
                else:
                    self.log_info(
                        f"Episode ended (reward={r_val:.2f}). "
                        f"Discarded. Total success: {success_cnt}/{self.num_data_episodes}"
                    )

                # Reset counters before starting next episode.
                ep_start_time = None
                ep_step_count = 0
                self._push_status(
                    phase="pre",
                    saved=success_cnt,
                    frames=0,
                    ep_t=0.0,
                    last_event=last_event_msg,
                    last_event_ts=last_event_ts,
                )

                # Reset for next episode
                obs, _ = self.env.reset()
                current_obs_processed = self._process_obs(obs)
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )

        self.buffer.close()
        self.log_info(
            f"Finished. Demos saved in: {os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
        )
        self.env.close()


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")

    # Named status bus + driver-side refreshing TTY renderer.  Create the bus
    # *before* launching DataCollector so the worker's ``ray.get_actor`` lookup
    # succeeds during __init__.
    bus = StatusBus.options(
        name=STATUS_BUS_NAME, lifetime="detached", namespace=Cluster.NAMESPACE
    ).remote()
    renderer = DriverStatusRenderer(bus, target=cfg.runner.num_data_episodes)
    renderer.start()
    try:
        collector = DataCollector.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )
        collector.run().wait()
    finally:
        renderer.stop()
        try:
            ray.kill(bus)
        except Exception:
            pass


if __name__ == "__main__":
    main()
