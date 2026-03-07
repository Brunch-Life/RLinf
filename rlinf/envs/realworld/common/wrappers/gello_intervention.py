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



import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_expert import GelloExpert

from scipy.spatial.transform import Rotation as R


class GelloIntervention(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.gripper_enabled = True
        if self.action_space.shape == (6,):
            self.gripper_enabled = False

        port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FTA0OUKN-if00-port0"
        self.expert = GelloExpert(port=port)
        self.last_intervene = 0

    def action(self, action: np.ndarray) -> np.ndarray:
        """
        Input:
        - action: policy action
        Output:
        - action: spacemouse action if nonezero; else, policy action
        """
        target_pos, target_quat, target_gripper = self.expert.get_action()
        r_target = R.from_quat(target_quat.copy())
        tcp_pose = self.get_wrapper_attr('get_tcp_pose')() #7d


        tcp_pos = tcp_pose[:3]
        tcp_quat = tcp_pose[3:]
        r_tcp = R.from_quat(tcp_quat.copy())

        r_delta = r_target * r_tcp.inv()
        delta_rotvec = r_delta.as_rotvec()

        delta_pos = target_pos - tcp_pos


        action_scale = self.get_wrapper_attr('get_action_scale')()
        delta_pos = delta_pos / action_scale[0]
        delta_rotvec = delta_rotvec / action_scale[1]



        expert_a = np.concatenate((delta_pos,delta_rotvec),axis=0)
        expert_a = np.clip(expert_a, -1.0, 1.0)


        if np.linalg.norm(expert_a) > 0.001:
            self.last_intervene = time.time()

        if self.gripper_enabled:
            target_gripper = target_gripper / action_scale[2]
            target_gripper = -(2 * target_gripper - 1.0)
            target_gripper = np.clip(target_gripper, -1.0, 1.0)
            expert_a = np.concatenate((expert_a, target_gripper), axis=0)


        print(f"expert_a: {expert_a}")

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action
        
        return obs, rew, done, truncated, info
