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
import time
from collections import deque
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.xsquare.chunk_interpolation import interpolate_segment
from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig

NUM_ARMS = 2
POSE_DIM = 6  # xyz(3) + rpy(3) per arm


@dataclass
class X2RobotS2mDeployConfig(Turtle2RobotConfig):
    """Dual-arm absolute-pose deployment config for openpi s2s inference."""

    use_camera_ids: list = field(default_factory=lambda: [0, 1, 2])
    use_arm: str = "dual"
    state_format: str = "xyz_rpy_gripper"
    enforce_gripper_close: bool = False
    reset_duration: float = 3.0  # seconds for the interpolated abs-pose reset

    def __post_init__(self):
        self.target_ee_pose = np.asarray(self.target_ee_pose, dtype=np.float64).reshape(
            NUM_ARMS, POSE_DIM
        )
        self.reset_ee_pose = np.asarray(self.reset_ee_pose, dtype=np.float64).reshape(
            NUM_ARMS, POSE_DIM
        )
        self.ee_pose_limit_min = np.asarray(
            self.ee_pose_limit_min, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.ee_pose_limit_max = np.asarray(
            self.ee_pose_limit_max, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.reward_threshold = np.asarray(
            self.reward_threshold, dtype=np.float64
        ).reshape(NUM_ARMS, POSE_DIM)
        self.action_scale = np.asarray(self.action_scale, dtype=np.float64)


class X2RobotS2mDeployEnv(Turtle2Env):
    """Turtle2/x2robot env in absolute-pose mode for openpi deployment.

    The controller runs in ``abs`` pose mode (no speed-clamping timer); reset is a
    slow interpolated abs dispatch, episodes go through AbsolutePoseChunkWrapper.
    """

    POSE_MODE = "abs"

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        super().__init__(
            X2RobotS2mDeployConfig(**override_cfg), worker_info, hardware_info, env_idx
        )

    def _reset_target_pose(self) -> np.ndarray:
        grip = self.config.gripper_width_limit_min
        return np.concatenate(
            [
                np.concatenate([self.config.reset_ee_pose[a], [grip]])
                for a in self.config.use_arm_ids
            ]
        )

    def _reset_arms(self) -> None:
        """Interpolate from the current pose to the reset pose via slow abs dispatch.

        The abs-mode controller has no speed-clamping timer, so a safe reset emits
        interpolated waypoints at ``exec_frequency`` over ``reset_duration`` seconds.
        """
        if self.config.is_dummy:
            return
        self._turtle2_state = self._controller.get_state().wait()[0]
        current = self.get_current_pose()
        target = self._reset_target_pose()
        num_points = max(
            1, int(round(self.config.exec_frequency * self.config.reset_duration))
        )
        dt = 1.0 / self.config.exec_frequency
        for point in interpolate_segment(current, target, num_points):
            self.apply_abs_pose(point)
            time.sleep(dt)
        self._turtle2_state = self._controller.get_state().wait()[0]


@dataclass
class X2RobotSm2smDeployConfig(X2RobotS2mDeployConfig):
    """sm2sm deployment config: the model consumes a ``[slave | master]`` state
    window and predicts ``[slave | master]`` actions; only the master half drives
    the followers. Defaults match the fold_towel sm2sm SFT checkpoint."""

    state_history_size: int = 3
    state_future_size: int = 2


class X2RobotSm2smDeployEnv(X2RobotS2mDeployEnv):
    """sm2sm absolute-pose deployment.

    State is a ``(seq_len, 2 * slave_dim)`` window where ``seq_len =
    state_history_size + 1 + state_future_size``: per frame ``[slave_14 |
    master_14]``. Slave frames are the follower history (current appended each
    step, future = last-frame copies); master frames come from ``_master_queue``,
    which is the model's own previously-predicted master (autoregressive),
    initialized to a copy of the current slave. The model output is 28-D
    (slave 14 + master 14); only the master half ``[14:28]`` is dispatched to the
    followers and fed back into the queue.

    NOTE: history frames here are spaced at the inference-frame rate
    (``step_frequency``), i.e. one slave frame captured per env step. If the
    trained ``state_step`` implies a different spacing, refine the slave-history
    sampling accordingly.
    """

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        # Build the sm2sm config and init the base Turtle2Env directly (skip
        # X2RobotS2mDeployEnv.__init__, which would build the s2m config).
        Turtle2Env.__init__(
            self,
            X2RobotSm2smDeployConfig(**override_cfg),
            worker_info,
            hardware_info,
            env_idx,
        )
        self._slave_dim = len(self.config.use_arm_ids) * 7
        self._seq_len = (
            self.config.state_history_size + 1 + self.config.state_future_size
        )
        self._slave_history = None
        self._master_queue = None
        # tcp_pose becomes a (seq_len, 2 * slave_dim) window.
        self.observation_space.spaces["state"].spaces["tcp_pose"] = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(self._seq_len, 2 * self._slave_dim),
            dtype=np.float32,
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def reset(self, *, seed=None, options=None):
        # Clear the autoregressive buffers; they reseed from the reset pose.
        self._slave_history = None
        self._master_queue = None
        return super().reset(seed=seed, options=options)

    def _build_tcp_pose(self) -> np.ndarray:
        cur_slave = self.get_current_pose()  # (slave_dim,)
        if self._slave_history is None:
            self._slave_history = deque(
                [cur_slave.copy() for _ in range(self.config.state_history_size + 1)],
                maxlen=self.config.state_history_size + 1,
            )
            self._master_queue = deque(
                [cur_slave.copy() for _ in range(self._seq_len)], maxlen=self._seq_len
            )
        else:
            self._slave_history.append(cur_slave.copy())
        slave_frames = (
            list(self._slave_history)
            + [self._slave_history[-1]] * self.config.state_future_size
        )
        master_frames = list(self._master_queue)[-self._seq_len :]
        return np.stack(
            [np.concatenate([s, m]) for s, m in zip(slave_frames, master_frames)],
            axis=0,
        ).astype(np.float64)

    def get_abs_action_space(self) -> gym.spaces.Box:
        """28-D action space: [slave_14 | master_14], both bounded by the safety box."""
        base = super().get_abs_action_space()
        return gym.spaces.Box(
            low=np.concatenate([base.low, base.low]),
            high=np.concatenate([base.high, base.high]),
            dtype=np.float32,
        )

    def _step_abs(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Dispatch the master half of a 28-D action and feed it back as the next
        master state (autoregressive)."""
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        master = action[self._slave_dim : 2 * self._slave_dim]
        if self._master_queue is not None:
            self._master_queue.append(master.copy())
        return super()._step_abs(master)
