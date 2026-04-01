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
    """Force grippers closed by stripping gripper dimensions from the action.

    Supports single-arm (7-dim -> 6-dim) and dual-arm (14-dim -> 12-dim).
    """

    def __init__(self, env):
        super().__init__(env)
        ub = self.env.action_space
        if ub.shape == (14,):
            self._dual = True
            keep = np.concatenate([np.arange(6), np.arange(7, 13)])
            self.action_space = Box(ub.low[keep], ub.high[keep])
        else:
            assert ub.shape == (7,)
            self._dual = False
            self.action_space = Box(ub.low[:6], ub.high[:6])

    def action(self, action: np.ndarray) -> np.ndarray:
        if self._dual:
            new_action = np.zeros((14,), dtype=np.float32)
            new_action[:6] = action[:6]
            new_action[7:13] = action[6:12]
            return new_action
        new_action = np.zeros((7,), dtype=np.float32)
        new_action[:6] = action.copy()
        return new_action

    def step(self, action):
        new_action = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)
        if "intervene_action" in info:
            ia = info["intervene_action"]
            if self._dual:
                info["intervene_action"] = np.concatenate([ia[:6], ia[7:13]]) if len(ia) == 14 else ia
            else:
                info["intervene_action"] = ia[:6]
        return obs, rew, done, truncated, info
