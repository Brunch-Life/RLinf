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

"""Dual-arm Franka environment with joint-space control via Franky."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np

from rlinf.scheduler import DualFrankaHWInfo

from .dual_franka_env import (
    _RIGHT_ARM_ENV_IDX_OFFSET,
    NUM_ARMS,
    TCP_POSE_DIM,
    TCP_VEL_DIM,
    DualFrankaEnv,
    DualFrankaRobotConfig,
)
from .franky_controller import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER

JOINT_DIM_PER_ARM = 7
ACTION_DIM_PER_ARM = 8


@dataclass
class DualFrankaJointRobotConfig(DualFrankaRobotConfig):
    """Configuration for joint-space control of the dual-arm Franka."""

    joint_position_limits_lower: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_LOWER.copy()
    )
    joint_position_limits_upper: np.ndarray = field(
        default_factory=lambda: JOINT_LIMITS_UPPER.copy()
    )
    joint_velocity_limit: float = 0.5
    joint_action_mode: str = "absolute"  # "absolute" or "delta"
    joint_action_scale: float = 0.1  # rad per unit action (delta mode)

    # When True, env.step() does NOT forward joint targets to either
    # controller — an external 1 kHz streaming loop (see
    # DualGelloJointIntervention with direct_stream=True) owns motion
    # commands.  step() still reads state, sleeps to maintain
    # step_frequency, and processes grippers on transitions.
    teleop_direct_stream: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.joint_position_limits_lower = np.array(self.joint_position_limits_lower)
        self.joint_position_limits_upper = np.array(self.joint_position_limits_upper)


class DualFrankaJointEnv(DualFrankaEnv):
    """Dual-arm Franka environment with joint-space actions.

    Action space is 16D: ``[left_j1..j7, left_grip, right_j1..j7, right_grip]``.

    In **absolute** mode each arm's first 7 dims are target joint positions
    (radians); in **delta** mode they are scaled increments relative to the
    current joint position.
    """

    CONFIG_CLS: type[DualFrankaJointRobotConfig] = DualFrankaJointRobotConfig

    def _setup_hardware(self):
        """Launch two FrankyController Ray actors (one per arm)."""
        from .franky_controller import FrankyController

        assert self.env_idx >= 0, "env_idx must be set for DualFrankaJointEnv."

        if self.hardware_info is not None:
            assert isinstance(self.hardware_info, DualFrankaHWInfo), (
                f"hardware_info must be DualFrankaHWInfo, got {type(self.hardware_info)}."
            )
            hw = self.hardware_info.config
            if self.config.left_robot_ip is None:
                self.config.left_robot_ip = hw.left_robot_ip
            if self.config.right_robot_ip is None:
                self.config.right_robot_ip = hw.right_robot_ip
            if self.config.left_camera_serials is None:
                self.config.left_camera_serials = hw.left_camera_serials
            if self.config.right_camera_serials is None:
                self.config.right_camera_serials = hw.right_camera_serials
            if self.config.base_camera_serials is None:
                self.config.base_camera_serials = getattr(
                    hw, "base_camera_serials", None
                )
            if self.config.camera_type is None:
                self.config.camera_type = getattr(hw, "camera_type", "zed")
            if self.config.base_camera_type is None:
                self.config.base_camera_type = getattr(hw, "base_camera_type", None)
            if self.config.left_camera_type is None:
                self.config.left_camera_type = getattr(hw, "left_camera_type", None)
            if self.config.right_camera_type is None:
                self.config.right_camera_type = getattr(hw, "right_camera_type", None)
            if self.config.left_gripper_type is None:
                self.config.left_gripper_type = getattr(
                    hw, "left_gripper_type", "robotiq"
                )
            if self.config.right_gripper_type is None:
                self.config.right_gripper_type = getattr(
                    hw, "right_gripper_type", "robotiq"
                )
            if self.config.left_gripper_connection is None:
                self.config.left_gripper_connection = getattr(
                    hw, "left_gripper_connection", None
                )
            if self.config.right_gripper_connection is None:
                self.config.right_gripper_connection = getattr(
                    hw, "right_gripper_connection", None
                )

        left_node = self.node_rank
        right_node = self.node_rank
        if self.hardware_info is not None:
            hw = self.hardware_info.config
            if hw.left_controller_node_rank is not None:
                left_node = hw.left_controller_node_rank
            if hw.right_controller_node_rank is not None:
                right_node = hw.right_controller_node_rank

        self._left_ctrl = FrankyController.launch_controller(
            robot_ip=self.config.left_robot_ip,
            env_idx=self.env_idx,
            node_rank=left_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.left_gripper_type or "robotiq",
            gripper_connection=self.config.left_gripper_connection,
        )
        self._right_ctrl = FrankyController.launch_controller(
            robot_ip=self.config.right_robot_ip,
            env_idx=self.env_idx + _RIGHT_ARM_ENV_IDX_OFFSET,
            node_rank=right_node,
            worker_rank=self.env_worker_rank,
            gripper_type=self.config.right_gripper_type or "robotiq",
            gripper_connection=self.config.right_gripper_connection,
        )

    def _interpolate_move_both(self, target_poses: np.ndarray, timeout: float = 1.5):
        """Neutralise the parent's Cartesian reset.

        The parent ``DualFrankaEnv.__init__`` calls this right after
        hardware setup to slew each arm to ``reset_ee_pose`` via
        Cartesian impedance.  In joint-space teleop the home pose is
        irrelevant — :class:`DualGelloJointIntervention` does a single
        blocking ``reset_joint`` to GELLO's configuration before the
        1 kHz streamer starts.  Just refresh cached state so the first
        step's obs reads the live pose instead of the default-zero one.
        """
        del target_poses, timeout
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

    def _go_to_rest(self, joint_reset: bool = False):
        """Reset both arms via ``reset_joint`` only (no Cartesian slew).

        The parent implementation first does per-arm ``reset_joint`` when
        ``joint_reset=True`` then Cartesian-interpolates to
        ``reset_ee_pose``.  A joint-space env has no meaningful Cartesian
        target during reset, so we always use ``reset_joint`` and skip
        the interpolation entirely.  ``joint_reset`` is ignored.
        """
        del joint_reset
        ctrls = [self._left_ctrl, self._right_ctrl]
        futures = [
            ctrl.reset_joint(list(self.config.joint_reset_qpos[arm]))
            for arm, ctrl in enumerate(ctrls)
        ]
        for f in futures:
            f.wait()
        time.sleep(0.5)

    def _init_action_obs_spaces(self):
        # Per-arm Cartesian safety boxes — kept for informational clipping
        # even though joint-space step does not consume them.
        self._xyz_safe_spaces = []
        self._rpy_safe_spaces = []
        for arm in range(NUM_ARMS):
            self._xyz_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, :3],
                    high=self.config.ee_pose_limit_max[arm, :3],
                    dtype=np.float64,
                )
            )
            self._rpy_safe_spaces.append(
                gym.spaces.Box(
                    low=self.config.ee_pose_limit_min[arm, 3:],
                    high=self.config.ee_pose_limit_max[arm, 3:],
                    dtype=np.float64,
                )
            )

        # 16D action: 2 x [7 joints + 1 gripper].
        if self.config.joint_action_mode == "absolute":
            arm_low = np.concatenate(
                [self.config.joint_position_limits_lower, np.array([-1.0])]
            )
            arm_high = np.concatenate(
                [self.config.joint_position_limits_upper, np.array([1.0])]
            )
            act_low = np.concatenate([arm_low, arm_low]).astype(np.float32)
            act_high = np.concatenate([arm_high, arm_high]).astype(np.float32)
        else:  # delta
            act_low = np.ones(NUM_ARMS * ACTION_DIM_PER_ARM, dtype=np.float32) * -1.0
            act_high = np.ones(NUM_ARMS * ACTION_DIM_PER_ARM, dtype=np.float32) * 1.0
        self.action_space = gym.spaces.Box(act_low, act_high)

        camera_specs = self._all_camera_specs()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * TCP_POSE_DIM,)
                        ),
                        "tcp_vel": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * TCP_VEL_DIM,)
                        ),
                        "joint_position": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * JOINT_DIM_PER_ARM,)
                        ),
                        "joint_velocity": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * JOINT_DIM_PER_ARM,)
                        ),
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(NUM_ARMS,)),
                        "tcp_force": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * 3,)
                        ),
                        "tcp_torque": gym.spaces.Box(
                            -np.inf, np.inf, shape=(NUM_ARMS * 3,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(224, 224, 3), dtype=np.uint8
                        )
                        for name, _, _ in camera_specs
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def step(self, action: np.ndarray):
        """Take a 16D joint-space step.

        ``action = [left_j1..j7, left_grip, right_j1..j7, right_grip]``.
        """
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(NUM_ARMS, ACTION_DIM_PER_ARM)

        is_gripper_effective = [True, True]
        states = [self._left_state, self._right_state]
        ctrls = [self._left_ctrl, self._right_ctrl]

        if not self.config.is_dummy:
            dt = 1.0 / self.config.step_frequency
            target_joints = []
            for arm in range(NUM_ARMS):
                joint_action = actions[arm, :JOINT_DIM_PER_ARM]
                if self.config.joint_action_mode == "absolute":
                    tj = joint_action.copy()
                else:
                    tj = (
                        states[arm].arm_joint_position
                        + joint_action * self.config.joint_action_scale
                    )
                tj = self._clip_joints_to_limits(tj)
                tj = self._clip_joint_velocity(tj, states[arm].arm_joint_position, dt)
                target_joints.append(tj)

            # Grippers first so they don't contend with a fresh move_joints.
            for arm in range(NUM_ARMS):
                gripper_val = (
                    actions[arm, JOINT_DIM_PER_ARM] * self.config.action_scale[2]
                )
                is_gripper_effective[arm] = self._gripper_action(
                    ctrls[arm], states[arm], gripper_val
                )

            # Only env.step owns motion commands when teleop is NOT
            # direct-streaming.  Direct-stream mode leaves move_joints
            # to the 1 kHz wrapper daemon to avoid two writers racing
            # on franky's motion queue.
            if not self.config.teleop_direct_stream:
                left_f = ctrls[0].move_joints(target_joints[0].astype(np.float32))
                right_f = ctrls[1].move_joints(target_joints[1].astype(np.float32))
                left_f.wait()
                right_f.wait()

        self._num_steps += 1
        if not self.config.is_dummy:
            step_time = time.time() - start_time
            time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))
            left_st_f = ctrls[0].get_state()
            right_st_f = ctrls[1].get_state()
            self._left_state = left_st_f.wait()[0]
            self._right_state = right_st_f.wait()[0]

        observation = self._get_observation()
        reward = self._calc_step_reward(is_gripper_effective)
        terminated = (reward == 1.0) and (
            self._success_hold_counter >= self.config.success_hold_steps
        )
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def _clip_joints_to_limits(self, q: np.ndarray) -> np.ndarray:
        return np.clip(
            q,
            self.config.joint_position_limits_lower,
            self.config.joint_position_limits_upper,
        )

    def _clip_joint_velocity(
        self, target: np.ndarray, current: np.ndarray, dt: float
    ) -> np.ndarray:
        max_delta = self.config.joint_velocity_limit * dt
        delta = np.clip(target - current, -max_delta, max_delta)
        return current + delta

    def get_joint_positions(self) -> np.ndarray:
        """Return stacked ``(2, 7)`` joint positions from cached state.

        Safe to call from wrappers: reads the last ``_left_state`` /
        ``_right_state`` without issuing RPC to the controllers.
        """
        return np.stack(
            [
                self._left_state.arm_joint_position.copy(),
                self._right_state.arm_joint_position.copy(),
            ]
        )

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()
        state = {
            "tcp_pose": np.concatenate(
                [self._left_state.tcp_pose, self._right_state.tcp_pose]
            ),
            "tcp_vel": np.concatenate(
                [self._left_state.tcp_vel, self._right_state.tcp_vel]
            ),
            "joint_position": np.concatenate(
                [
                    self._left_state.arm_joint_position,
                    self._right_state.arm_joint_position,
                ]
            ),
            "joint_velocity": np.concatenate(
                [
                    self._left_state.arm_joint_velocity,
                    self._right_state.arm_joint_velocity,
                ]
            ),
            "gripper_position": np.array(
                [
                    self._left_state.gripper_position,
                    self._right_state.gripper_position,
                ],
                dtype=np.float32,
            ),
            "tcp_force": np.concatenate(
                [self._left_state.tcp_force, self._right_state.tcp_force]
            ),
            "tcp_torque": np.concatenate(
                [self._left_state.tcp_torque, self._right_state.tcp_torque]
            ),
        }
        return copy.deepcopy({"state": state, "frames": frames})
