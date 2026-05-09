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

"""Abstract base for dual-arm Franka envs driven through ``FrankyController``.

Owns the shared lifecycle (controller launch, reset, gripper, pacing);
subclasses define action/obs spaces and per-step arm-motion dispatch.
"""

from __future__ import annotations

import time

import gymnasium as gym
import numpy as np

from rlinf.scheduler import DualFrankaHWInfo

from .dual_franka_env import (
    _RIGHT_ARM_ENV_IDX_OFFSET,
    NUM_ARMS,
    TCP_POSE_DIM,
    TCP_VEL_DIM,
    DualFrankaEnv,
)
from .franky_controller import FrankyController

JOINT_DIM_PER_ARM = 7


class DualFrankaFrankyEnv(DualFrankaEnv):
    """Abstract base. Subclasses set PER_ARM_ACTION_DIM / GRIPPER_IDX_IN_ARM
    and implement ``_init_action_obs_spaces`` + ``_get_observation``."""

    PER_ARM_ACTION_DIM: int = 0
    GRIPPER_IDX_IN_ARM: int = 0

    # ---------------------------------------------------------------- hardware

    def _setup_hardware(self):
        assert self.env_idx >= 0, f"env_idx must be set for {type(self).__name__}."

        if self.hardware_info is not None:
            assert isinstance(self.hardware_info, DualFrankaHWInfo), (
                f"hardware_info must be DualFrankaHWInfo, "
                f"got {type(self.hardware_info)}."
            )
            hw = self.hardware_info.config
            # YAML wins over hardware-info; getattr keeps forward-compat
            # with older hw dataclasses missing newer per-slot fields.
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
                self.config.camera_type = getattr(hw, "camera_type", "realsense")
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
        # Override parent's move_arm-based reset: FrankyController is joint-
        # space only, so we reach home via _go_to_rest and just refresh
        # cached state here.
        del target_poses, timeout
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

    def _go_to_rest(self, joint_reset: bool = False):
        # Open grippers before the home slew so any grasped object is
        # released before the arms travel through home.
        del joint_reset
        try:
            self._left_ctrl.open_gripper()
            self._right_ctrl.open_gripper()
        except Exception as exc:
            self._logger.warning("open_gripper during reset failed: %s", exc)

        left_f = self._left_ctrl.reset_joint(self.config.joint_reset_qpos[0])
        right_f = self._right_ctrl.reset_joint(self.config.joint_reset_qpos[1])
        left_f.wait()
        right_f.wait()
        time.sleep(0.5)
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

    def reset(self, *, seed=None, options=None):
        """Override parent: FrankyController gains are set at construction
        time, so skip the parent's reconfigure_compliance_params call.
        ``options["skip_reset_to_home"]`` lets teleop wrappers keep tracking
        from the episode-end pose instead of bouncing through home."""
        del seed
        skip_reset_to_home = bool((options or {}).get("skip_reset_to_home", False))
        self._num_steps = 0
        self._success_hold_counter = 0

        if self.config.is_dummy:
            return self._get_observation(), {}

        joint_cycle = next(self._joint_reset_cycle)
        joint_reset = joint_cycle == 0
        if joint_reset:
            self._logger.info(
                "Number of resets reached %d, resetting joints.",
                self.config.joint_reset_cycle,
            )

        if skip_reset_to_home:
            self._logger.info(
                "skip_reset_to_home=True: holding arms at episode-end pose "
                "(teleop wrapper will realign to device)."
            )
        else:
            self._go_to_rest(joint_reset)
        self._clear_errors()

        left_st_f = self._left_ctrl.get_state()
        right_st_f = self._right_ctrl.get_state()
        self._left_state = left_st_f.wait()[0]
        self._right_state = right_st_f.wait()[0]
        return self._get_observation(), {}

    # ------------------------------------------------------------------ utils

    def get_joint_positions(self) -> np.ndarray:
        """Stacked ``(2, 7)`` joint positions from cached state (no RPC)."""
        return np.stack(
            [
                self._left_state.arm_joint_position.copy(),
                self._right_state.arm_joint_position.copy(),
            ]
        )

    def _cartesian_safety_boxes(self) -> None:
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

    def _build_observation_space(self, joint_position_dim: int) -> gym.spaces.Dict:
        camera_specs = self._all_camera_specs()
        return gym.spaces.Dict(
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
                            -np.inf, np.inf, shape=(joint_position_dim,)
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

    # --------------------------------------------------------- subclass hooks

    def _init_action_obs_spaces(self):
        raise NotImplementedError(
            f"{type(self).__name__} must implement _init_action_obs_spaces"
        )

    def _get_observation(self) -> dict:
        raise NotImplementedError(
            f"{type(self).__name__} must implement _get_observation"
        )

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        """Override in subclass to issue move_joints / move_tcp_pose."""
        del actions, states, ctrls, dt

    def _pace_between_action_and_state_read(self) -> bool:
        # Joint+direct_stream overrides to False since the 1 kHz daemon
        # owns the rate.
        return True

    # ------------------------------------------------------------------ step

    def step(self, action: np.ndarray):
        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(NUM_ARMS, self.PER_ARM_ACTION_DIM)

        is_gripper_effective = [True, True]
        states = [self._left_state, self._right_state]
        ctrls = [self._left_ctrl, self._right_ctrl]

        if not self.config.is_dummy:
            dt = 1.0 / self.config.step_frequency

            # Grippers first so they don't contend with a fresh motion command.
            for arm in range(NUM_ARMS):
                gripper_val = (
                    actions[arm, self.GRIPPER_IDX_IN_ARM] * self.config.action_scale[2]
                )
                is_gripper_effective[arm] = self._gripper_action(
                    ctrls[arm], states[arm], gripper_val
                )

            self._dispatch_arm_motion(actions, states, ctrls, dt)

        self._num_steps += 1
        if not self.config.is_dummy:
            if self._pace_between_action_and_state_read():
                step_time = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
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
