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

"""GELLO intervention wrapper for joint-space control.

This wrapper overrides the policy's joint-space action with joint positions
read directly from the GELLO teleoperation device.
"""

import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert


class GelloJointIntervention(gym.ActionWrapper):
    """Action wrapper that overrides policy action with GELLO joint-space input.

    In **absolute** mode, the GELLO joint positions are used directly as the
    action (with gripper appended).  In **delta** mode, the wrapper computes
    the difference between GELLO joints and the current robot joints and
    normalises it by ``action_scale``.

    Args:
        env: The wrapped environment (must be a :class:`FrankaJointEnv`).
        port: Serial port of the GELLO device.
        gripper_enabled: Whether the gripper channel is present in the action
            space.
        use_delta: If ``True``, produce delta joint actions instead of
            absolute positions.
        action_scale: Scaling factor for delta mode (must match the env's
            ``joint_action_scale``).
    """

    def __init__(
        self,
        env,
        port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        self.expert = GelloJointExpert(port=port)
        self.last_intervene = 0

    def _get_current_joint_positions(self) -> np.ndarray:
        """Read current joint positions from env state without triggering RPC.

        Uses the cached ``_franka_state`` on the env rather than calling
        ``get_joint_positions()`` which would issue a blocking remote call
        and mutate the env's internal state.
        """
        franka_state = self.get_wrapper_attr("_franka_state")
        return franka_state.arm_joint_position.copy()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.expert.ready:
            return action, False

        target_joints, target_gripper = self.expert.get_action()
        current_joints = self._get_current_joint_positions()

        if self.use_delta:
            delta_joints = target_joints - current_joints
            delta_joints = delta_joints / self.action_scale
            expert_a = np.clip(delta_joints, -1.0, 1.0)
        else:
            # Absolute mode: use GELLO joints directly
            expert_a = target_joints.copy()

        # Append gripper action only when gripper is enabled.
        # When gripper is disabled (GripperCloseEnv strips the last dim),
        # the action space is 7D (joints only) so we must NOT append.
        gripper_active = False
        if self.gripper_enabled:
            gripper_action = -(2 * target_gripper - 1.0)
            gripper_action = np.clip(gripper_action, -1.0, 1.0)
            gripper_active = np.abs(gripper_action).item() > 0.5
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # Determine if the GELLO is actively intervening
        movement = np.linalg.norm(target_joints - current_joints)

        if movement > 0.01 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info
