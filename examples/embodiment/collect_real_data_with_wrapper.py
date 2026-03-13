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

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class RealWorldCollectEpisode(CollectEpisode):
    """CollectEpisode variant for interactive data collection with deferred recording.

    Keyboard controls:
        a: Start recording (prior positioning steps are discarded)
        b: End episode as failure (terminated, reward=-1)
        c: End episode as success (terminated, reward=+1)

    Before pressing 'a', the operator can freely position the robot via
    SpaceMouse / GELLO without any data being recorded.  Records the actual
    ``info["intervene_action"]`` instead of the placeholder zero-action.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from rlinf.envs.realworld.common.keyboard.keyboard_listener import (
            KeyboardListener,
        )

        self._listener = KeyboardListener()
        self._recording = False
        self._last_episode_was_recorded = False

    @property
    def last_episode_was_recorded(self) -> bool:
        return self._last_episode_was_recorded

    def reset(self, *, seed=None, options=None):
        self._recording = False
        return super().reset(seed=seed, options=options)

    @staticmethod
    def _inject_success(info: dict, success: bool) -> None:
        """Inject a success flag into info so ``_get_episode_success`` picks it up."""
        flag = torch.tensor(success)
        if isinstance(info, dict):
            ep = info.get("episode")
            if isinstance(ep, dict):
                ep["success_once"] = flag
            else:
                info["success"] = flag

    def step(self, action, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)

        # Discard env's automatic termination (driven by target_ee_pose /
        # success_hold_steps); episode lifecycle is fully keyboard-controlled.
        terminated = torch.zeros_like(terminated, dtype=torch.bool)

        key = self._listener.get_key()

        if key == "a" and not self._recording:
            self._recording = True
            for env_idx in range(self.num_envs):
                self._buffers[env_idx] = self._new_buffer()
                self._buffers[env_idx]["observations"].append(
                    self._slice_copy(obs, env_idx)
                )
            print(
                "\n>>> Recording STARTED "
                "(press 'c' = success, 'b' = fail) <<<"
            )
            return obs, reward, terminated, truncated, info

        if key == "b":
            terminated = torch.ones_like(terminated, dtype=torch.bool)
            reward = torch.full_like(reward, -1.0)
            self._inject_success(info, False)
        elif key == "c":
            terminated = torch.ones_like(terminated, dtype=torch.bool)
            reward = torch.full_like(reward, 1.0)
            self._inject_success(info, True)

        if self._recording:
            recorded_action = action
            if isinstance(info, dict) and "intervene_action" in info:
                recorded_action = info["intervene_action"]
            self._record_step(
                recorded_action, obs, reward, terminated, truncated, info
            )
            self._maybe_flush(terminated, truncated)

        done = any(
            self._scalar_flag(terminated, i) or self._scalar_flag(truncated, i)
            for i in range(self.num_envs)
        )
        if done:
            self._last_episode_was_recorded = self._recording
            self._recording = False

        return obs, reward, terminated, truncated, info


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes

        base_env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )

        log_path = self.cfg.runner.logger.get("log_path") or "logs"
        self.save_dir = os.path.join(log_path, "lerobot_dataset")
        self.log_info(f"Saving LeRobot dataset to: {self.save_dir}")

        runner_cfg = self.cfg.runner
        self.env = RealWorldCollectEpisode(
            env=base_env,
            save_dir=self.save_dir,
            num_envs=1,
            export_format=getattr(runner_cfg, "export_format", "lerobot"),
            robot_type=getattr(runner_cfg, "robot_type", "panda"),
            fps=getattr(runner_cfg, "fps", 10),
            only_success=getattr(runner_cfg, "only_success", False),
            show_goal_site=False,
        )

    def run(self):
        obs, _ = self.env.reset()
        success_cnt = 0
        total_cnt = 0
        progress_bar = tqdm(
            range(self.num_data_episodes), desc="Collecting Data Episodes:"
        )

        action_dim = self.env.action_space.shape[-1]
        self.log_info(
            "Keyboard controls: 'a' = start recording, "
            "'c' = success + end, 'b' = fail + end"
        )

        while success_cnt < self.num_data_episodes:
            action = np.zeros((1, action_dim))
            obs, reward, terminated, truncated, info = self.env.step(action)

            done = bool(terminated.any()) or bool(truncated.any())

            if done:
                if self.env.last_episode_was_recorded:
                    r_val = (
                        reward[0]
                        if hasattr(reward, "__getitem__") and len(reward) > 0
                        else reward
                    )
                    if isinstance(r_val, torch.Tensor):
                        r_val = r_val.item()

                    is_success = float(r_val) > 0
                    success_cnt += int(is_success)
                    total_cnt += 1
                    self.log_info(
                        f"Episode {total_cnt} done. Success: {is_success}. "
                        f"Successes: {success_cnt}/{self.num_data_episodes}"
                    )
                    progress_bar.update(1)
                else:
                    self.log_info(
                        "Episode ended without recording (no 'a' press). "
                        "Resetting..."
                    )

                obs, _ = self.env.reset()

        self.env.close()
        self.log_info(
            f"Finished. {success_cnt} successful episodes collected. "
            f"LeRobot dataset saved to: {self.save_dir}"
        )


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data_wrapper"
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
