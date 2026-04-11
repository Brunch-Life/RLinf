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

"""Franka environment with joint-space actions via pylibfranka.

This module provides :class:`FrankaJointEnv`, a subclass of :class:`FrankaEnv`
that operates in joint space rather than Cartesian space.  Joint position
targets (absolute or delta) are sent directly to the robot via
:class:`PylibfrankaController`, which streams them at 1 kHz through a
real-time control loop.
"""

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlinf.scheduler import FrankaHWInfo

from .franka_env import FrankaEnv, FrankaRobotConfig
from .pylibfranka_controller import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER


@dataclass
class FrankaJointRobotConfig(FrankaRobotConfig):
    """Configuration for joint-space control of the Franka robot."""

    # Joint-space specific parameters
    joint_position_limits_lower: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_LOWER.copy()
    )
    joint_position_limits_upper: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_UPPER.copy()
    )
    joint_velocity_limit: float = 0.5  # rad/s per joint (conservative)
    joint_action_mode: str = "absolute"  # "absolute" or "delta"
    joint_action_scale: float = 0.1  # scaling for delta mode (rad per unit action)

    def __post_init__(self):
        super().__post_init__()
        self.joint_position_limits_lower = np.array(self.joint_position_limits_lower)
        self.joint_position_limits_upper = np.array(self.joint_position_limits_upper)


class FrankaJointEnv(FrankaEnv):
    """Franka environment with joint-space actions via pylibfranka.

    Action space is 8D: ``[j1, j2, j3, j4, j5, j6, j7, gripper]``.

    In **absolute** mode, the first 7 dimensions are target joint positions
    (radians).  In **delta** mode, they are joint position increments scaled
    by ``joint_action_scale``.
    """

    CONFIG_CLS: type[FrankaJointRobotConfig] = FrankaJointRobotConfig

    def _setup_hardware(self):
        """Use PylibfrankaController instead of FrankaController."""
        from .pylibfranka_controller import PylibfrankaController

        assert self.env_idx >= 0, "env_idx must be set for FrankaJointEnv."
        assert isinstance(self.hardware_info, FrankaHWInfo), (
            f"hardware_info must be FrankaHWInfo, but got {type(self.hardware_info)}."
        )

        if self.config.robot_ip is None:
            self.config.robot_ip = self.hardware_info.config.robot_ip
        if self.config.camera_serials is None:
            self.config.camera_serials = self.hardware_info.config.camera_serials
        if self.config.camera_type is None:
            self.config.camera_type = getattr(
                self.hardware_info.config, "camera_type", "realsense"
            )
        if self.config.gripper_type is None:
            hw_gripper = getattr(
                self.hardware_info.config, "gripper_type", "pylibfranka"
            )
            # FrankaConfig defaults to "franka" (ROS-based), but
            # PylibfrankaController uses "pylibfranka" for the native gripper.
            if hw_gripper == "franka":
                hw_gripper = "pylibfranka"
            self.config.gripper_type = hw_gripper
        if self.config.gripper_connection is None:
            self.config.gripper_connection = getattr(
                self.hardware_info.config, "gripper_connection", None
            )

        controller_node_rank = getattr(
            self.hardware_info.config, "controller_node_rank", None
        )
        if controller_node_rank is None:
            controller_node_rank = self.node_rank

        # Ensure gripper_type is valid for PylibfrankaController
        gripper_type = self.config.gripper_type or "pylibfranka"
        if gripper_type == "franka":
            gripper_type = "pylibfranka"

        self._controller = PylibfrankaController.launch_controller(
            robot_ip=self.config.robot_ip,
            env_idx=self.env_idx,
            node_rank=controller_node_rank,
            worker_rank=self.env_worker_rank,
            gripper_type=gripper_type,
            gripper_connection=self.config.gripper_connection,
        )

    def _init_action_obs_spaces(self):
        """Define joint-space action and observation spaces."""
        assert (
            self.config.camera_serials is not None
            and len(self.config.camera_serials) > 0
        ), "At least one camera serial must be provided for FrankaJointEnv."

        # Initialize Cartesian safety box from base class config.
        # These are used by _clip_position_to_safety_box() which may be
        # called during Cartesian reset motions (_interpolate_move path).
        self._xyz_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[:3],
            high=self.config.ee_pose_limit_max[:3],
            dtype=np.float64,
        )
        self._rpy_safe_space = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[3:],
            high=self.config.ee_pose_limit_max[3:],
            dtype=np.float64,
        )

        # Action space: 8D [j1..j7, gripper]
        if self.config.joint_action_mode == "absolute":
            act_low = np.concatenate(
                [self.config.joint_position_limits_lower, np.array([-1.0])]
            ).astype(np.float32)
            act_high = np.concatenate(
                [self.config.joint_position_limits_upper, np.array([1.0])]
            ).astype(np.float32)
        else:  # delta
            act_low = np.ones(8, dtype=np.float32) * -1.0
            act_high = np.ones(8, dtype=np.float32) * 1.0
        self.action_space = gym.spaces.Box(act_low, act_high, dtype=np.float32)

        # Observation space: includes both Cartesian and joint information
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "tcp_vel": gym.spaces.Box(-np.inf, np.inf, shape=(6,)),
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(7,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(1,)),
                        "tcp_force": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                        "tcp_torque": gym.spaces.Box(-np.inf, np.inf, shape=(3,)),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        f"wrist_{k + 1}": gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for k in range(len(self.config.camera_serials))
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def step(self, action: np.ndarray):
        """Take a step in the environment using joint-space actions.

        Args:
            action: 8D array ``[j1..j7, gripper]``.
                In absolute mode, j1..j7 are target joint positions.
                In delta mode, j1..j7 are scaled joint increments.
        """
        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_action = action[:7]
        gripper_action_value = action[7]

        # Compute target joint positions
        if self.config.joint_action_mode == "absolute":
            target_joints = joint_action
        else:  # delta
            target_joints = (
                self._franka_state.arm_joint_position
                + joint_action * self.config.joint_action_scale
            )

        # Safety: clip to joint limits
        target_joints = self._clip_joints_to_limits(target_joints)

        # Safety: limit velocity (per-step change)
        dt = 1.0 / self.config.step_frequency
        target_joints = self._clip_joint_velocity(
            target_joints, self._franka_state.arm_joint_position, dt
        )

        if not self.config.is_dummy:
            # Send gripper action
            gripper_action_scaled = gripper_action_value * self.config.action_scale[2]
            is_gripper_action_effective = self._gripper_action(gripper_action_scaled)

            # Set joint target (non-blocking — RT loop tracks it at 1 kHz)
            self._controller.move_joints(target_joints.astype(np.float32)).wait()
        else:
            is_gripper_action_effective = True

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._franka_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()

        reward = self._calc_step_reward(observation, is_gripper_action_effective)

        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        reward *= self.config.reward_scale
        return observation, reward, terminated, truncated, {}

    def get_joint_positions(self) -> np.ndarray:
        """Return current 7D joint positions from cached state.

        This reads from the last-updated ``_franka_state`` without
        issuing a new RPC call, making it safe to call from wrappers.
        """
        return self._franka_state.arm_joint_position.copy()

    def _clip_joints_to_limits(self, joint_positions: np.ndarray) -> np.ndarray:
        """Clip joint positions to hardware limits."""
        return np.clip(
            joint_positions,
            self.config.joint_position_limits_lower,
            self.config.joint_position_limits_upper,
        )

    def _clip_joint_velocity(
        self, target: np.ndarray, current: np.ndarray, dt: float
    ) -> np.ndarray:
        """Limit per-step joint change to enforce velocity limits."""
        max_delta = self.config.joint_velocity_limit * dt
        delta = target - current
        delta = np.clip(delta, -max_delta, max_delta)
        return current + delta

    def _get_observation(self) -> dict:
        """Get observation including joint state."""
        if not self.config.is_dummy:
            frames = self._get_camera_frames()
            state = {
                "tcp_pose": self._franka_state.tcp_pose,
                "tcp_vel": self._franka_state.tcp_vel,
                "joint_position": self._franka_state.arm_joint_position,
                "joint_velocity": self._franka_state.arm_joint_velocity,
                "gripper_position": np.array(
                    [self._franka_state.gripper_position]
                ),
                "tcp_force": self._franka_state.tcp_force,
                "tcp_torque": self._franka_state.tcp_torque,
            }
            observation = {
                "state": state,
                "frames": frames,
            }
            return copy.deepcopy(observation)
        else:
            return self._base_observation_space.sample()
