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

Concrete subclasses live in sibling modules:

* :class:`DualFrankaJointEnv` in :mod:`.dual_franka_joint_env` — joint-space
  control (``absolute`` / ``delta`` actions, 16-D action vector).
* :class:`DualFrankaRot6dEnv` in :mod:`.dual_franka_rot6d_env` — TCP-rot6d
  impedance control (20-D action vector; each step pushes one pose target
  into a franky ``CartesianImpedanceTracker`` per arm).

The base owns the shared lifecycle: Ray-actor controller launch, hardware-info
merge, reset to ``joint_reset_qpos``, cached ``_left_state`` / ``_right_state``,
gripper dispatch, step pacing, reward/terminate bookkeeping. Subclasses
define their own action/observation spaces and per-step arm-motion dispatch.

Never instantiated directly — ``NotImplementedError`` guards the hooks.
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

JOINT_DIM_PER_ARM = 7


class DualFrankaFrankyEnv(DualFrankaEnv):
    """Abstract base for dual-arm Franka envs driven through ``FrankyController``.

    Subclasses must set:

    * ``PER_ARM_ACTION_DIM`` — width of each arm's action block
      (e.g. 8 for joint + gripper, 10 for xyz + rot6d + gripper).
    * ``GRIPPER_IDX_IN_ARM`` — index of the gripper trigger within the
      per-arm slice (e.g. 7 for joint, 9 for rot6d).

    Subclasses must implement:

    * :py:meth:`_init_action_obs_spaces` — build ``action_space`` and
      ``observation_space``.
    * :py:meth:`_get_observation` — return the dict observation.

    Subclasses may override:

    * :py:meth:`_dispatch_arm_motion` — called once per step after grippers;
      default is a no-op (appropriate for pre-dispatched chunk motions).
    * :py:meth:`_pace_between_action_and_state_read` — whether ``step``
      should sleep to maintain ``step_frequency`` before refreshing cached
      state. Default True.
    """

    PER_ARM_ACTION_DIM: int = 0
    GRIPPER_IDX_IN_ARM: int = 0

    # ---------------------------------------------------------------- hardware

    def _setup_hardware(self):
        """Launch two FrankyController Ray actors (one per arm).

        Shared across joint and rot6d subclasses — all FrankyController-based
        envs see the same dual-arm hardware topology.
        """
        from .franky_controller import FrankyController

        assert self.env_idx >= 0, f"env_idx must be set for {type(self).__name__}."

        if self.hardware_info is not None:
            assert isinstance(self.hardware_info, DualFrankaHWInfo), (
                f"hardware_info must be DualFrankaHWInfo, "
                f"got {type(self.hardware_info)}."
            )
            hw = self.hardware_info.config
            # Env-config YAML wins; fall back to hardware-info only when the
            # env YAML didn't specify. ``getattr`` guards forward-compat with
            # older hw dataclasses missing newer per-slot fields.
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
        """Override: skip the parent's Cartesian impedance reset.

        ``DualFrankaEnv.__init__`` calls this right after hardware setup to
        slew each arm to ``reset_ee_pose`` via ``ctrl.move_arm`` — but
        :class:`FrankyController` doesn't expose ``move_arm`` (it's joint-space
        + waypoint only), so the parent's impl would ``AttributeError``.
        FrankyController-based envs reach home via :py:meth:`_go_to_rest`
        (joint-space ``reset_joint``); here we just refresh cached state so
        the first step's obs reads live values.
        """
        del target_poses, timeout
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

    def _go_to_rest(self, joint_reset: bool = False):
        """Drive both arms to ``joint_reset_qpos`` — the primary reset pose.

        This is what eval / autonomous rollout needs so the policy's first
        obs is in-distribution. Teleop wrappers (e.g.
        :class:`DualGelloJointIntervention`) layer their own device-pose
        alignment on top in their :py:meth:`reset`, so collect mode gets
        ``joint_reset_qpos`` → device pose as two sequential motions.

        Grippers are opened first so any object grasped at episode end is
        released before the arms slew through home — avoids carrying the
        workpiece up and dropping it at an unsafe spot.
        """
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
        """Override: skip parent's ``reconfigure_compliance_params`` RPC.

        :class:`FrankyController` sets impedance gains once at construction
        time (``_JOINT_STIFFNESS`` / ``_JOINT_DAMPING`` in
        ``franky_controller.py``); there's no per-episode reconfigure RPC
        on the Ray actor, so the parent's call would ``AttributeError``.
        Body is otherwise identical to ``DualFrankaEnv.reset``, plus one
        per-reset option:

        * ``options["skip_reset_to_home"] = True`` — skip the
          ``reset_joint(joint_reset_qpos)`` slew. GELLO wrappers set this
          so teleop can pick up tracking from the episode-end pose
          instead of bouncing through home.
        """
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

    def _cartesian_safety_boxes(self) -> None:
        """Cache per-arm xyz / rpy safety boxes from ``ee_pose_limit`` config.

        Called from subclass ``_init_action_obs_spaces``. Kept as attributes
        for subclasses that clip Cartesian targets; not consumed by joint-
        space motion dispatch.
        """
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
        """Return the standard state+frames obs space.

        ``joint_position_dim`` lets subclasses set the width of the
        ``joint_position`` slot without duplicating the rest of the schema.
        """
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
        """Submit per-step arm motion. No-op by default.

        Subclasses override to issue ``move_joints`` (joint mode) or similar.
        Rot6d waypoint mode leaves this as a no-op because the whole chunk
        is pre-submitted by :py:meth:`dispatch_chunk` at chunk start.
        """
        del actions, states, ctrls, dt

    def _pace_between_action_and_state_read(self) -> bool:
        """Whether :py:meth:`step` should sleep to maintain ``step_frequency``.

        Returns True by default. Joint mode with ``teleop_direct_stream=True``
        overrides to False — the 1 kHz daemon owns the rate, and the outer
        collection loop paces ``step()`` itself so wrapper/record overhead
        is part of the budget.
        """
        return True

    # ------------------------------------------------------------------ step

    def step(self, action: np.ndarray):
        """Shared dual-arm step: clip, grippers, arm motion, pace, refresh.

        Subclass contract:
        * ``action.reshape(NUM_ARMS, PER_ARM_ACTION_DIM)`` must produce
          per-arm slices where ``slice[GRIPPER_IDX_IN_ARM]`` is the gripper
          trigger.
        * :py:meth:`_dispatch_arm_motion` handles any per-step arm command.
        """
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
