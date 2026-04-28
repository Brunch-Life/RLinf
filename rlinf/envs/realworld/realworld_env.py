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

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )

        self._init_env()

        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()
        self._init_replay_override()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        """
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                try:
                    if proc.name() in ros_proc_names:
                        proc.kill()
                        time.sleep(0.5)
                except (
                    psutil.AccessDenied,
                    psutil.NoSuchProcess,
                    psutil.ZombieProcess,
                ):
                    pass

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )
        # Explicit per-env flat-state layout as a tuple of dict keys.
        # When set, ``_wrap_obs`` concatenates ``raw_obs['state'][k]`` in
        # this order instead of falling back to alphabetical sort — load-
        # bearing when a policy hard-codes slice offsets against the flat
        # state (e.g. the dual-Franka rot6d SFT policy's ``state[:20]``).
        # Envs that don't declare it keep the legacy alphabetical path.
        try:
            layouts = self.env.call("get_wrapper_attr", "STATE_LAYOUT")
        except AttributeError:
            self._state_layout = None
        else:
            first = layouts[0] if layouts else None
            self._state_layout = tuple(first) if first else None

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once
        episode_info["intervened_steps"] = self.intervened_steps
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        # Replay override always resumes from frame 0 on reset; the dataset
        # is single-episode so episode boundaries map cleanly to step 0.
        if self._replay_override_cfg is not None:
            self._replay_override_step = 0
        return extracted_obs, infos

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        # Flat-state concat order:
        # - Preferred: inner env declared ``STATE_LAYOUT`` (tuple of keys).
        #   Missing keys are a loud ``KeyError`` so schema drift can't
        #   silently reshuffle bytes. Keys present in the dict but absent
        #   from ``STATE_LAYOUT`` are dropped from the flat tensor — useful
        #   for diagnostic-only fields we don't want to feed the policy.
        # - Legacy: no ``STATE_LAYOUT`` declared → alphabetical sort. Byte
        #   order is implicit; do NOT hard-code slice offsets against it.
        state = raw_obs["state"]
        if self._state_layout is not None:
            missing = [k for k in self._state_layout if k not in state]
            if missing:
                raise KeyError(
                    f"STATE_LAYOUT references missing keys: {missing}. "
                    f"Inner env emits {sorted(state)}."
                )
            full_states = [state[k] for k in self._state_layout]
        else:
            raw_states = OrderedDict(sorted(state.items()))
            full_states = list(raw_states.values())
        full_states = np.concatenate(full_states, axis=-1)
        obs["states"] = full_states

        # Process images: main_image_key picks the primary image; everything
        # else is stacked alphabetically into extra_view_images. For dual-arm
        # envs, set main_image_key to the preferred wrist camera (e.g.
        # "left_wrist_0_rgb") and the remaining views flow into extra.
        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        obs["main_images"] = frames[self.main_image_key]
        raw_images = OrderedDict(sorted(frames.items()))
        raw_images.pop(self.main_image_key)

        extra_view_image_names = None
        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)
            extra_view_image_names = tuple(raw_images.keys())

        obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        # Names travel alongside the image stack so policies can verify
        # index→view; kept as a str-tuple (not tensorized) on purpose.
        if extra_view_image_names is not None:
            obs["extra_view_image_names"] = extra_view_image_names
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        # ``max_episode_steps: null`` in YAML means "no step-count timeout — an
        # external wrapper (e.g. KeyboardRewardDoneWrapper) owns episode end".
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = to_tensor(intervene_action)
        infos["intervene_flag"] = to_tensor(intervene_flag)

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def _init_replay_override(self):
        """Optional dataset-replay override hook.

        When ``cfg.replay_override.enabled`` is true, :py:meth:`chunk_step`
        swaps the incoming model chunk with dataset GT *before* dispatch and
        writes ``(model_chunk, gt_chunk, live_states_per_step,
        dataset_states_per_step)`` per chunk to ``dump_path`` for offline
        analysis. Lets us isolate "is dispatch correct?" from "is the model
        correct?" — replay drives the robot, the model only observes the
        live obs and emits predictions we log without acting on.
        """
        self._replay_override_cfg = None
        self._replay_override_targets = None  # (T, action_dim) lazy-loaded
        self._replay_override_dataset_state = None  # (T, state_dim) lazy
        self._replay_override_step = 0
        self._replay_override_log = None

        cfg_block = self.cfg.get("replay_override", None)
        if cfg_block is None or not bool(cfg_block.get("enabled", False)):
            return
        self._replay_override_cfg = cfg_block
        self._replay_override_log = {
            "frame_indices": [],
            "model_chunks": [],
            "gt_chunks": [],
            "live_states_per_step": [],
            "dataset_states_per_step": [],
        }

    def _lazy_load_replay_override(self, action_dim: int) -> None:
        """Pull the parquet once, cached on first chunk_step. ``action_dim``
        is used to dispatch joint vs rot6d target builders.
        """
        if self._replay_override_targets is not None:
            return
        from rlinf.envs.realworld.common.dataset_replay import (
            load_replay_targets,
        )

        cfg_block = self._replay_override_cfg
        dataset_root = pathlib.Path(cfg_block["dataset_root"])
        episode_index = int(cfg_block.get("episode_index", 0))
        self._replay_override_targets = load_replay_targets(
            dataset_root, episode_index, action_dim
        )

        # Cache the raw state column too so we can compare live vs recorded
        # state per step. State dim is fixed at 68 in the joint-collected
        # dataset; the rot6d-backfill rewrites only the [:20] prefix.
        import pyarrow.parquet as pq

        path = (
            dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
        )
        table = pq.read_table(path, columns=["state"])
        T = table.num_rows
        self._replay_override_dataset_state = np.stack(
            [
                np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
                for i in range(T)
            ]
        )
        print(
            f"[realworld_env/replay_override] loaded ep={episode_index} "
            f"T={T} action_dim={action_dim} from {dataset_root}"
        )

    def _apply_replay_override(self, model_chunk_actions):
        """Capture model chunk and substitute dataset GT for dispatch."""
        if isinstance(model_chunk_actions, torch.Tensor):
            model_np = model_chunk_actions.detach().cpu().numpy()
            return_tensor = True
            device = model_chunk_actions.device
            dtype = model_chunk_actions.dtype
        else:
            model_np = np.asarray(model_chunk_actions)
            return_tensor = False
            device = None
            dtype = None

        num_envs, chunk_size, action_dim = model_np.shape
        self._lazy_load_replay_override(action_dim)

        T_total = self._replay_override_targets.shape[0]
        start = int(self._replay_override_step)
        end = min(start + chunk_size, T_total)
        actual = end - start
        gt = np.zeros_like(model_np)
        if actual > 0:
            slice_targets = self._replay_override_targets[start:end]
            for env_id in range(num_envs):
                gt[env_id, :actual] = slice_targets
        if actual < chunk_size:
            # Past the recorded end — hold the last frame.
            last = self._replay_override_targets[-1]
            for env_id in range(num_envs):
                gt[env_id, actual:] = last

        ds_state = np.zeros(
            (chunk_size, self._replay_override_dataset_state.shape[1]),
            dtype=np.float32,
        )
        if actual > 0:
            ds_state[:actual] = self._replay_override_dataset_state[start:end]
        if actual < chunk_size:
            ds_state[actual:] = self._replay_override_dataset_state[-1]

        self._replay_override_log["frame_indices"].append(start)
        self._replay_override_log["model_chunks"].append(model_np.copy())
        self._replay_override_log["gt_chunks"].append(gt.copy())
        self._replay_override_log["dataset_states_per_step"].append(ds_state)

        self._replay_override_step += chunk_size

        if return_tensor:
            return torch.as_tensor(gt, dtype=dtype, device=device)
        return gt

    def _record_replay_override_live(self, obs_list):
        """Capture live state per step within the just-dispatched chunk."""
        states_per_step = []
        for obs in obs_list:
            s = obs.get("states")
            if torch.is_tensor(s):
                s = s.detach().cpu().numpy()
            else:
                s = np.asarray(s)
            states_per_step.append(s)
        # (num_envs, chunk_size, state_dim)
        live = np.stack(states_per_step, axis=1)
        self._replay_override_log["live_states_per_step"].append(live.copy())

    def _dump_replay_override(self) -> None:
        cfg_block = self._replay_override_cfg
        dump_path = cfg_block.get("dump_path", None)
        if dump_path is None:
            return
        dump_path = pathlib.Path(dump_path)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        log = self._replay_override_log
        # Ragged shapes are fine across episodes once auto_reset kicks in;
        # at chunk-step granularity each entry is regular within itself.
        np.savez(
            dump_path.with_suffix(".tmp.npz"),
            frame_indices=np.asarray(log["frame_indices"], dtype=np.int64),
            model_chunks=np.stack(log["model_chunks"], axis=0)
            if log["model_chunks"]
            else np.zeros((0,), dtype=np.float32),
            gt_chunks=np.stack(log["gt_chunks"], axis=0)
            if log["gt_chunks"]
            else np.zeros((0,), dtype=np.float32),
            live_states_per_step=np.stack(log["live_states_per_step"], axis=0)
            if log["live_states_per_step"]
            else np.zeros((0,), dtype=np.float32),
            dataset_states_per_step=np.stack(log["dataset_states_per_step"], axis=0)
            if log["dataset_states_per_step"]
            else np.zeros((0,), dtype=np.float32),
        )
        # Atomic rename so a crashed run still leaves a valid file.
        os.replace(dump_path.with_suffix(".tmp.npz"), dump_path)

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        if self._replay_override_cfg is not None:
            chunk_actions = self._apply_replay_override(chunk_actions)

        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []

        # Give TCP-pose envs a chance to peek at the whole chunk up-front
        # (currently used for logging / future chunk-level preprocessing).
        # Inner envs that don't implement ``dispatch_chunk`` are unaffected;
        # per-step dispatch still happens in their own ``step``. num_envs==1
        # in realworld; SyncVectorEnv.call broadcasts to the single sub-env.
        if chunk_size > 0:
            try:
                actions_t = chunk_actions[0]
                if isinstance(actions_t, torch.Tensor):
                    actions_np = actions_t.detach().cpu().numpy()
                else:
                    actions_np = np.asarray(actions_t)
                self.env.call("dispatch_chunk", actions_np)
            except AttributeError:
                # Inner env does not implement dispatch_chunk → legacy path.
                pass
            except Exception as e:
                # Non-fatal: fall back to per-step dispatch if the whole-
                # chunk submission errored (bad pose, preempt race, etc.).
                print(
                    f"[realworld_env] dispatch_chunk failed, "
                    f"falling back to per-step: {e}"
                )

        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        if self._replay_override_cfg is not None:
            self._record_replay_override_live(obs_list)
            self._dump_replay_override()

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = torch.stack(
                raw_chunk_intervene_actions, dim=1
            ).reshape(self.num_envs, -1)
            infos_last["intervene_flag"] = torch.stack(raw_chunk_intervene_flag, dim=1)
            infos_list[-1] = infos_last

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )
