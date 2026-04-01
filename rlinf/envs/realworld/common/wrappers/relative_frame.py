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
from gymnasium import Env
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.franka.utils import (
    construct_adjoint_matrix,
    construct_homogeneous_matrix,
)


class RelativeFrame(gym.Wrapper):
    """Transform observations and actions between base and end-effector frames.

    Supports single-arm (tcp_pose shape ``(7,)``, action ``(6/7,)``)
    and dual-arm (tcp_pose shape ``(14,)``, action ``(12/14,)``).
    """

    def __init__(self, env: Env, include_relative_pose=True):
        super().__init__(env)
        pose_dim = env.observation_space["state"]["tcp_pose"].shape[0]
        self._dual = pose_dim == 14

        self.include_relative_pose = include_relative_pose

        if self._dual:
            self.adjoint_matrices = [np.zeros((6, 6)), np.zeros((6, 6))]
            if self.include_relative_pose:
                self.T_b_r_invs = [np.zeros((4, 4)), np.zeros((4, 4))]
        else:
            self.adjoint_matrix = np.zeros((6, 6))
            if self.include_relative_pose:
                self.T_b_r_inv = np.zeros((4, 4))

    def step(self, action: np.ndarray):
        transformed_action = self.transform_action(action)
        obs, reward, done, truncated, info = self.env.step(transformed_action)

        if "intervene_action" in info:
            info["intervene_action"] = self.transform_action_inv(
                info["intervene_action"]
            )

        self._update_adjoint(obs["state"]["tcp_pose"])
        return self.transform_observation(obs), reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        tcp_pose = obs["state"]["tcp_pose"]
        self._update_adjoint(tcp_pose)

        if self.include_relative_pose:
            if self._dual:
                for arm, pose7 in enumerate([tcp_pose[:7], tcp_pose[7:]]):
                    self.T_b_r_invs[arm] = np.linalg.inv(
                        construct_homogeneous_matrix(pose7)
                    )
            else:
                self.T_b_r_inv = np.linalg.inv(
                    construct_homogeneous_matrix(tcp_pose)
                )

        return self.transform_observation(obs), info

    # ------------------------------------------------------------------ #

    def _update_adjoint(self, tcp_pose: np.ndarray):
        if self._dual:
            self.adjoint_matrices[0] = construct_adjoint_matrix(tcp_pose[:7])
            self.adjoint_matrices[1] = construct_adjoint_matrix(tcp_pose[7:])
        else:
            self.adjoint_matrix = construct_adjoint_matrix(tcp_pose)

    def transform_observation(self, obs):
        if self._dual:
            return self._transform_obs_dual(obs)
        return self._transform_obs_single(obs)

    def _transform_obs_single(self, obs):
        adjoint_inv = np.linalg.inv(self.adjoint_matrix)
        if "tcp_vel" in obs["state"]:
            obs["state"]["tcp_vel"] = adjoint_inv @ obs["state"]["tcp_vel"]
        if self.include_relative_pose:
            T_b_o = construct_homogeneous_matrix(obs["state"]["tcp_pose"])
            T_r_o = self.T_b_r_inv @ T_b_o
            p_r_o = T_r_o[:3, 3]
            quat_r_o = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
            obs["state"]["tcp_pose"] = np.concatenate((p_r_o, quat_r_o))
        return obs

    def _transform_obs_dual(self, obs):
        tcp_vel = obs["state"].get("tcp_vel")
        tcp_pose = obs["state"]["tcp_pose"]

        out_pose_parts = []
        out_vel_parts = []
        for arm in range(2):
            s, e = arm * 7, arm * 7 + 7
            pose7 = tcp_pose[s:e]
            adj_inv = np.linalg.inv(self.adjoint_matrices[arm])

            if tcp_vel is not None:
                vs, ve = arm * 6, arm * 6 + 6
                out_vel_parts.append(adj_inv @ tcp_vel[vs:ve])

            if self.include_relative_pose:
                T_b_o = construct_homogeneous_matrix(pose7)
                T_r_o = self.T_b_r_invs[arm] @ T_b_o
                p = T_r_o[:3, 3]
                q = R.from_matrix(T_r_o[:3, :3].copy()).as_quat()
                out_pose_parts.append(np.concatenate((p, q)))
            else:
                out_pose_parts.append(pose7)

        if self.include_relative_pose:
            obs["state"]["tcp_pose"] = np.concatenate(out_pose_parts)
        if out_vel_parts:
            obs["state"]["tcp_vel"] = np.concatenate(out_vel_parts)
        return obs

    def transform_action(self, action: np.ndarray):
        action = np.array(action)
        if self._dual:
            action[:6] = self.adjoint_matrices[0] @ action[:6]
            action[7:13] = self.adjoint_matrices[1] @ action[7:13]
        else:
            action[:6] = self.adjoint_matrix @ action[:6]
        return action

    def transform_action_inv(self, action: np.ndarray):
        action = np.array(action)
        if self._dual:
            action[:6] = np.linalg.inv(self.adjoint_matrices[0]) @ action[:6]
            action[7:13] = np.linalg.inv(self.adjoint_matrices[1]) @ action[7:13]
        else:
            action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action


class RelativeTargetFrame(RelativeFrame):
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Update adjoint matrix
        self.adjoint_matrix = construct_adjoint_matrix(self.env.target_ee_pose)
        if self.include_relative_pose:
            # Update transformation matrix from the reset pose's relative frame to base frame
            self.T_b_r_inv = np.linalg.inv(
                construct_homogeneous_matrix(self.env.target_ee_pose)
            )

        # Transform observation to spatial frame
        return self.transform_observation(obs), info
