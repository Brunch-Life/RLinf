# Copyright 2025 The RLinf Authors.
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

import gymnasium as gym
import numpy as np
from gymnasium import Env, spaces
from scipy.spatial.transform import Rotation as R


class Quat2EulerWrapper(gym.ObservationWrapper):
    """Convert quaternion TCP pose to euler angles.

    Supports single-arm ``(7,)`` -> ``(6,)`` and dual-arm ``(14,)`` -> ``(12,)``.
    """

    def __init__(self, env: Env):
        super().__init__(env)
        pose_shape = self.observation_space["state"]["tcp_pose"].shape[0]
        self._dual = pose_shape == 14
        out_dim = 12 if self._dual else 6
        self.observation_space["state"]["tcp_pose"] = spaces.Box(
            -np.inf, np.inf, shape=(out_dim,)
        )

    def observation(self, observation):
        tcp_pose = observation["state"]["tcp_pose"]
        if self._dual:
            left = tcp_pose[:7]
            right = tcp_pose[7:]
            left_euler = np.concatenate([left[:3], R.from_quat(left[3:].copy()).as_euler("xyz")])
            right_euler = np.concatenate([right[:3], R.from_quat(right[3:].copy()).as_euler("xyz")])
            observation["state"]["tcp_pose"] = np.concatenate([left_euler, right_euler])
        else:
            observation["state"]["tcp_pose"] = np.concatenate(
                (tcp_pose[:3], R.from_quat(tcp_pose[3:].copy()).as_euler("xyz"))
            )
        return observation
