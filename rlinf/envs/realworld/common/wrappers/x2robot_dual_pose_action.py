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

"""Absolute-pose action wrapper for the Turtle2/x2robot dual-arm env.

The wrapper only selects the dispatch mode and exposes the absolute action space;
the env owns the execution (interpolated absolute dispatch). The delta (RL) path
needs no wrapper — it uses ``Turtle2Env.step``'s default ``dispatch='delta'``.
"""

import gymnasium as gym
import numpy as np


class AbsolutePoseChunkWrapper(gym.Wrapper):
    """Route absolute ee-pose frames to the env's interpolated abs dispatch.

    ``RealWorldEnv.chunk_step`` feeds one absolute pose frame per ``step``; the env
    interpolates from the previous target and dispatches each substep at
    ``exec_frequency``. Frame layout per arm is ``[x, y, z, rx, ry, rz, gripper]``.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.action_space = self.get_wrapper_attr("get_abs_action_space")()

    def step(self, action: np.ndarray):
        return self.env.step(action, dispatch="abs")
