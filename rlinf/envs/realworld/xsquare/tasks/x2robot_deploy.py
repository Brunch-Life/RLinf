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

import time
from dataclasses import dataclass, field

import numpy as np

from rlinf.envs.realworld.xsquare.chunk_interpolation import interpolate_segment
from rlinf.envs.realworld.xsquare.turtle2_env import Turtle2Env, Turtle2RobotConfig

NUM_ARMS = 2
POSE_DIM = 6  # xyz(3) + rpy(3) per arm


@dataclass
class X2RobotDeployConfig(Turtle2RobotConfig):
    """Dual-arm absolute-pose deployment config for openpi s2m inference."""

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


class X2RobotDeployEnv(Turtle2Env):
    """Turtle2/x2robot env in absolute-pose mode for openpi deployment.

    The controller runs in ``abs`` pose mode (no speed-clamping timer); reset is a
    slow interpolated abs dispatch, episodes go through AbsolutePoseChunkWrapper.
    """

    POSE_MODE = "abs"

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        super().__init__(
            X2RobotDeployConfig(**override_cfg), worker_info, hardware_info, env_idx
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
