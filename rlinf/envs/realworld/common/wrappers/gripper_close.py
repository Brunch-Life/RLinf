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
from gymnasium.spaces import Box


class GripperCloseEnv(gym.ActionWrapper):
    """
    Use this wrapper to task that requires the gripper to be closed.

    Strips the last action dimension (gripper) and forces it to zero.
    Works with both 7D Cartesian actions (6+1) and 8D joint actions (7+1).
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        assert len(ub.shape) == 1, f"Expected 1D action space, got shape {ub.shape}"
        self._full_action_dim = ub.shape[0]
        self.action_space = Box(ub.low[:-1], ub.high[:-1])

    def action(self, action: np.ndarray) -> np.ndarray:
        new_action = np.zeros((self._full_action_dim,), dtype=np.float32)
        new_action[:-1] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            info["intervene_action"] = info["intervene_action"][:-1]
        return obs, rew, done, truncated, info
