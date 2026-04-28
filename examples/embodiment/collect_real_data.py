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
import time
from typing import Optional

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.data.embodied_io_struct import (
    ChunkStepResult,
    EmbodiedRolloutResult,
)
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


def _truthy(val) -> bool:
    """Accept bool, int, or str (``1/true/yes/on``, case-insensitive)."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ("1", "true", "yes", "on")
    return bool(val)


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self.total_cnt = 0
        override_cfg = cfg.env.eval.get("override_cfg", {})
        self.manual_episode_control_only = bool(
            override_cfg.get("manual_episode_control_only", False)
        )
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        # Keep an unwrapped handle for set_task_descriptions — CollectEpisode
        # is a gym.Wrapper and its __getattr__ would forward, but the
        # explicit handle survives even if wrapping rules ever change.
        self._raw_env = self.env

        # Multi-task collection: round-robin episodes through ``tasks``.
        # Each task gets its own save_dir + prompt. When ``tasks`` is
        # missing/empty the wrapper writes a single dataset like before.
        self._tasks: Optional[list[dict]] = None
        self._active_task_idx = 0
        dc_cfg = cfg.env.eval.get("data_collection")
        tasks_cfg = dc_cfg.get("tasks") if dc_cfg is not None else None
        if tasks_cfg:
            base_save_dir = dc_cfg.save_dir
            self._tasks = []
            for t in tasks_cfg:
                name = t["name"]
                self._tasks.append(
                    {
                        "name": name,
                        "save_dir": os.path.join(base_save_dir, name),
                        "prompt": t.get("prompt", ""),
                    }
                )

        if self.cfg.env.eval.get("data_collection", None) and getattr(
            self.cfg.env.eval.data_collection, "enabled", False
        ):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=self.cfg.env.eval.data_collection.save_dir,
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
                resume=_truthy(
                    getattr(self.cfg.env.eval.data_collection, "resume", False)
                ),
                tasks=self._tasks,
            )
            if self._tasks:
                self._switch_to_task(self._active_task_idx)
            self._preexisting_success = int(
                getattr(self.env, "preexisting_episode_count", 0)
            )
            if self._preexisting_success:
                self.log_info(
                    f"[resume] found {self._preexisting_success} pre-existing "
                    f"episodes under {self.cfg.env.eval.data_collection.save_dir}; "
                    f"continuing toward {self.num_data_episodes}"
                )
        else:
            self._preexisting_success = 0

        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])

        buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
        self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")

        self.buffer = TrajectoryReplayBuffer(
            seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
            enable_cache=False,
            auto_save=True,
            auto_save_path=buffer_path,
            trajectory_format="pt",
        )

        # Outer rate limiter for envs (e.g. DualFrankaFrankyEnv direct-stream)
        # that deliberately don't self-pace. Target period comes from
        # data_collection.fps so parquet metadata matches wall-clock rate.
        fps = None
        if self.cfg.env.eval.get("data_collection") is not None:
            fps = self.cfg.env.eval.data_collection.get("fps")
        self._target_step_period = 1.0 / float(fps) if fps else None

    def _switch_to_task(self, idx: int) -> None:
        """Activate task ``self._tasks[idx]`` for the upcoming episode.

        Updates both the CollectEpisode wrapper (selects which dataset
        the next flushed episode lands in) and the underlying
        ``RealWorldEnv`` (so ``obs['task_descriptions']`` carries the
        active task's prompt — that string ends up as lerobot's ``task``
        field per frame).
        """
        if not self._tasks:
            return
        task = self._tasks[idx]
        if hasattr(self.env, "set_active_task"):
            self.env.set_active_task(task["name"])
        self._raw_env.set_task_descriptions([task["prompt"]] * self._raw_env.num_envs)
        self.log_info(
            f"[task] switched to {task['name']!r} "
            f"(idx={idx}, prompt={task['prompt']!r})"
        )

    def _process_obs(self, obs):
        """Reshape env obs into the dict EmbodiedRolloutResult expects."""
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)
        # Str-tuple of extra camera names: debugging metadata from
        # RealWorldEnv; drop so it doesn't reach the Trajectory tensor path.
        obs.pop("extra_view_image_names", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            if not isinstance(val, torch.Tensor):
                continue
            val = val.cpu()
            if key == "images":
                ret_obs["main_images"] = val.clone()
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

    def run(self):
        obs, _ = self.env.reset()
        # Seed from preexisting successful episodes so resume shows the
        # bar at the right starting position and we stop at the right target.
        success_cnt = self._preexisting_success
        if success_cnt >= self.num_data_episodes:
            self.log_info(
                f"[resume] already have {success_cnt} episodes, target "
                f"{self.num_data_episodes} already met — nothing to do."
            )
            self.env.close()
            return
        progress_bar = tqdm(
            total=self.num_data_episodes,
            initial=success_cnt,
            desc="Collecting Data Episodes:",
        )

        current_rollout = EmbodiedRolloutResult(
            max_episode_length=self.cfg.env.eval.max_episode_steps,
        )

        current_obs_processed = self._process_obs(obs)

        while success_cnt < self.num_data_episodes:
            iter_start = time.perf_counter()
            # Teleop wrapper overrides this via info["intervene_action"].
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # Keyboard wrappers (KeyboardStartEndWrapper etc.) surface the
            # operator's current phase via info. ``kb_phase is None`` means
            # no wrapper is attached — preserve upstream "record every step".
            kb_event = self._unwrap_info_scalar(info.get("keyboard_event"))
            kb_phase = self._unwrap_info_scalar(info.get("keyboard_phase"))
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")

            if "intervene_action" in info:
                action = info["intervene_action"]

            next_obs_processed = self._process_obs(next_obs)

            terminated_tensor = terminated.unsqueeze(1)
            truncated_tensor = truncated.unsqueeze(1)
            done_tensor = terminated_tensor | truncated_tensor
            done = bool(done_tensor.any().item())

            action_tensor = torch.as_tensor(action, dtype=torch.float32)
            reward_tensor = reward.float().unsqueeze(1)

            step_result = ChunkStepResult(
                actions=action_tensor,
                rewards=reward_tensor,
                dones=done_tensor,
                terminations=terminated_tensor,
                truncations=truncated_tensor,
                forward_inputs={"action": action_tensor},
            )

            # Rebuild the rollout when a new rec begins (``start``) or the
            # in-progress one is aborted (``abort``). ``restart`` is kept
            # for backward compat with older wrappers that emit it.
            if kb_event in ("start", "restart", "abort"):
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )
            if kb_phase in (None, "rec"):
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

                manual_done = False
                if "manual_done" in info:
                    md = info["manual_done"]
                    if hasattr(md, "__getitem__") and len(md) > 0:
                        manual_done = bool(md[0])
                    else:
                        manual_done = bool(md)

                self.total_cnt += 1
                if self.manual_episode_control_only:
                    save_episode = bool(manual_done)
                else:
                    save_episode = bool(r_val >= 0.5 or manual_done)

                if save_episode:
                    success_cnt += 1

                    self.log_info(
                        f"Success (reward={r_val}, manual_done={manual_done}). "
                        f"Total: {success_cnt}/{self.num_data_episodes}"
                    )

                    trajectory = current_rollout.to_trajectory()
                    trajectory.intervene_flags = torch.ones_like(
                        trajectory.intervene_flags
                    )
                    self.buffer.add_trajectories([trajectory])

                    progress_bar.update(1)

                    # Round-robin to the next task only on successful save.
                    # Aborts (kb_event="abort") never reach here — the
                    # KeyboardStartEndWrapper just clears the buffer and
                    # returns to pre, no done flag, no reset.
                    if self._tasks:
                        self._active_task_idx = (self._active_task_idx + 1) % len(
                            self._tasks
                        )
                        self._switch_to_task(self._active_task_idx)
                else:
                    self.log_info(
                        f"Episode ended (reward={r_val:.2f}). "
                        f"Discarded. Total success: {success_cnt}/{self.num_data_episodes}"
                    )

                reset_options = None
                if success_cnt >= self.num_data_episodes:
                    reset_options = {"skip_wait_for_start": True}
                obs, _ = self.env.reset(options=reset_options)
                current_obs_processed = self._process_obs(obs)
                current_rollout = EmbodiedRolloutResult(
                    max_episode_length=self.cfg.env.eval.max_episode_steps,
                )

            # Pin step + record + wrappers to target period. On ``done``
            # iterations env.reset typically blows past the budget and
            # sleep_for <= 0 — a graceful no-op.
            if self._target_step_period is not None:
                elapsed = time.perf_counter() - iter_start
                sleep_for = self._target_step_period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

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
    collector = DataCollector.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    collector.run().wait()


if __name__ == "__main__":
    main()
