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
from scipy.spatial.transform import Rotation as R

from rlinf.scheduler import DualFrankaHWInfo

from .dual_franka_env import (
    _RIGHT_ARM_ENV_IDX_OFFSET,
    NUM_ARMS,
    TCP_POSE_DIM,
    TCP_VEL_DIM,
    DualFrankaEnv,
    DualFrankaRobotConfig,
)
from rlinf.utils.rot6d import (
    matrix_to_rot6d,
    rot6d_to_quat_xyzw_safe,
)

from .franky_controller import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER

JOINT_DIM_PER_ARM = 7
ACTION_DIM_PER_ARM = 8  # joint mode: 7 joints + 1 gripper trigger
ACTION_DIM_PER_ARM_ROT6D = 10  # rot6d modes: xyz(3) + rot6d(6) + 1 gripper trigger
# Per-arm proprio width written into the overloaded `joint_position` slot
# by `_tcp_rot6d_18d` (xyz(3) + rot6d(6); gripper lives in its own slot).
PROPRIO_DIM_PER_ARM_ROT6D = 9


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
    # "absolute"           = 16-d action [L7joints, L_grip, R7joints, R_grip] (radians)
    # "delta"              = same layout but joints are scaled increments on current q
    # "tcp_rot6d_waypoint" = 20-d action [L_xyz(3), L_rot6d(6), L_grip,
    #                        R_xyz(3), R_rot6d(6), R_grip]; the whole 20-step
    #                        policy chunk is pre-submitted to franky as one
    #                        CartesianWaypointMotion per arm (Ruckig blends
    #                        through waypoints — smoother than per-step).
    # "tcp_rot6d_per_step" = same 20-d layout but every env.step dispatches
    #                        one CartesianMotion per arm to move_arm(); simpler,
    #                        responds instantly to online corrections.
    # In both rot6d modes observation is automatically switched to the matching
    # tcp-rot6d proprio layout (see ``_tcp_rot6d_18d``).
    joint_action_mode: str = "absolute"
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
        """Drive both arms to ``joint_reset_qpos`` — the primary reset pose.

        This is what eval / autonomous rollout needs so the policy's first
        obs is in-distribution. Teleop wrappers (e.g.
        :class:`DualGelloJointIntervention`) layer their own device-pose
        alignment on top in their :py:meth:`reset`, so collect mode gets
        ``joint_reset_qpos`` → device pose as two sequential motions.
        """
        del joint_reset
        left_f = self._left_ctrl.reset_joint(self.config.joint_reset_qpos[0])
        right_f = self._right_ctrl.reset_joint(self.config.joint_reset_qpos[1])
        left_f.wait()
        right_f.wait()
        time.sleep(0.5)
        self._left_state = self._left_ctrl.get_state().wait()[0]
        self._right_state = self._right_ctrl.get_state().wait()[0]

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

        # Action-space construction depends on mode.
        if self.config.joint_action_mode == "absolute":
            arm_low = np.concatenate(
                [self.config.joint_position_limits_lower, np.array([-1.0])]
            )
            arm_high = np.concatenate(
                [self.config.joint_position_limits_upper, np.array([1.0])]
            )
            act_low = np.concatenate([arm_low, arm_low]).astype(np.float32)
            act_high = np.concatenate([arm_high, arm_high]).astype(np.float32)
        elif self.config.joint_action_mode in ("tcp_rot6d_waypoint", "tcp_rot6d_per_step"):
            # Per-arm: [xyz (3, m), rot6d (6, unit), gripper_trigger (1)].
            # xyz bounds come from the safe-box config; rot6d bounds are widened
            # to [-1.5, 1.5] so normalized model outputs have headroom before
            # Gram-Schmidt renormalizes. Gripper trigger is ±1.
            rot6d_low = -1.5 * np.ones(6, dtype=np.float32)
            rot6d_high = 1.5 * np.ones(6, dtype=np.float32)
            left_low = np.concatenate(
                [self.config.ee_pose_limit_min[0, :3], rot6d_low, np.array([-1.0])]
            )
            left_high = np.concatenate(
                [self.config.ee_pose_limit_max[0, :3], rot6d_high, np.array([1.0])]
            )
            right_low = np.concatenate(
                [self.config.ee_pose_limit_min[1, :3], rot6d_low, np.array([-1.0])]
            )
            right_high = np.concatenate(
                [self.config.ee_pose_limit_max[1, :3], rot6d_high, np.array([1.0])]
            )
            act_low = np.concatenate([left_low, right_low]).astype(np.float32)
            act_high = np.concatenate([left_high, right_high]).astype(np.float32)
        else:  # delta joint
            act_low = np.ones(NUM_ARMS * ACTION_DIM_PER_ARM, dtype=np.float32) * -1.0
            act_high = np.ones(NUM_ARMS * ACTION_DIM_PER_ARM, dtype=np.float32) * 1.0
        self.action_space = gym.spaces.Box(act_low, act_high)

        # In rot6d TCP-pose modes, `_get_observation` overloads the
        # `joint_position` slot with `_tcp_rot6d_18d` (18-d
        # [L_xyz, L_rot6d, R_xyz, R_rot6d]) so the alphabetical concat in
        # `realworld_env._wrap_obs` produces state[0:20] in the SFT layout.
        # The declared space must match that real width so `sample()` (dummy
        # mode) and any downstream space-driven tooling agree with runtime.
        if self.config.joint_action_mode in ("tcp_rot6d_waypoint", "tcp_rot6d_per_step"):
            joint_position_dim = NUM_ARMS * PROPRIO_DIM_PER_ARM_ROT6D
        else:
            joint_position_dim = NUM_ARMS * JOINT_DIM_PER_ARM

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
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def dispatch_chunk(self, chunk_actions: np.ndarray) -> None:
        """Submit a whole TCP-pose chunk as one planned motion per arm.

        Called once per 20-step policy chunk in ``tcp_rot6d_waypoint``
        mode, at the start of ``RealWorldEnv.chunk_step``, before the
        per-step ``env.step`` loop begins. Each arm receives one
        ``CartesianWaypointMotion`` built from the full chunk, with
        finite-differenced twists attached to every intermediate
        waypoint so Ruckig blends through them at the policy's intended
        speed instead of decelerating to zero between each 100 ms
        policy sample (the old per-step ``stream_tcp_impedance`` path
        fed the impedance controller 100 ms step jumps, which caused
        J7 chatter + TCP overshoot once the target plateaued).

        Rot6D is decoded to a rotation matrix via Gram-Schmidt, then to
        a quaternion to fit franky's xyz+quat waypoint signature. Adjacent
        quaternions are sign-aligned (``q_i · q_{i-1} >= 0``) so Ruckig
        doesn't plan the long way around when the policy output crosses
        a ±q hemisphere boundary.

        Subsequent chunks preempt the in-flight motion via another
        asynchronous ``robot.move`` call. The last waypoint of each
        chunk is left with zero velocity so that if the stream halts
        (end of eval, termination) the arm comes to a graceful stop
        on the trajectory that's already been planned.

        No-op outside ``tcp_rot6d_waypoint`` mode — joint-space
        ``move_joints`` streaming is already smooth, and
        ``tcp_rot6d_per_step`` dispatches per ``env.step``.
        """
        if self.config.is_dummy:
            return
        if self.config.joint_action_mode != "tcp_rot6d_waypoint":
            return

        chunk_actions = np.asarray(chunk_actions, dtype=np.float64)
        expected_dim = NUM_ARMS * ACTION_DIM_PER_ARM_ROT6D
        assert chunk_actions.ndim == 2 and chunk_actions.shape[1] == expected_dim, (
            f"chunk_actions must be (chunk_size, {expected_dim}); "
            f"got {chunk_actions.shape}"
        )
        chunk_size = chunk_actions.shape[0]
        if chunk_size == 0:
            return

        dt = 1.0 / self.config.step_frequency

        left_poses = np.zeros((chunk_size, 7), dtype=np.float64)
        right_poses = np.zeros((chunk_size, 7), dtype=np.float64)
        for t in range(chunk_size):
            action = np.clip(
                chunk_actions[t],
                self.action_space.low,
                self.action_space.high,
            )
            per_arm = action.reshape(NUM_ARMS, ACTION_DIM_PER_ARM_ROT6D)
            for arm, out in enumerate((left_poses, right_poses)):
                xyz = np.clip(
                    per_arm[arm, 0:3],
                    self.config.ee_pose_limit_min[arm, :3],
                    self.config.ee_pose_limit_max[arm, :3],
                )
                # Hemisphere-align with the previous quat so Ruckig stays on
                # the short geodesic through a chunk; seed t=0 against the
                # live TCP quat read from the robot at chunk start. The
                # ``prev_quat`` also doubles as the fallback for the safe
                # rot6d→quat decode: if the model emits NaN or the two
                # rot6d basis vectors collapse to collinearity, we reuse
                # the previous pose so the chunk still dispatches.
                if t == 0:
                    prev_quat = (
                        self._left_state.tcp_pose[3:]
                        if arm == 0
                        else self._right_state.tcp_pose[3:]
                    )
                else:
                    prev_quat = out[t - 1, 3:]
                rot6d = per_arm[arm, 3:9]
                quat = rot6d_to_quat_xyzw_safe(
                    rot6d, fallback_quat_xyzw=prev_quat
                )  # xyzw
                if np.dot(quat, prev_quat) < 0.0:
                    quat = -quat
                out[t, :3] = xyz
                out[t, 3:] = quat

        left_vels = self._finite_diff_twists(left_poses, dt)
        right_vels = self._finite_diff_twists(right_poses, dt)

        left_f = self._left_ctrl.move_waypoints(left_poses, left_vels)
        right_f = self._right_ctrl.move_waypoints(right_poses, right_vels)
        left_f.wait()
        right_f.wait()

    @staticmethod
    def _finite_diff_twists(poses: np.ndarray, dt: float) -> np.ndarray:
        """Estimate per-waypoint twist ``[v, ω]`` from a pose sequence.

        Central differences on intermediate waypoints, forward diff on
        the first. Last row left as zeros so Ruckig plans a clean stop
        at chunk end — the next chunk preempts this before it fires.

        Angular component is ``(R_{i+1} · R_{i−1}⁻¹).as_rotvec() / Δt``
        (axis * angle-rate, rad/s), matching franky's ``Twist`` convention
        (rotvec, not Euler rates).
        """
        n = poses.shape[0]
        twists = np.zeros((n, 6), dtype=np.float64)
        if n < 2:
            return twists
        for t in range(n - 1):  # last row left zero → graceful stop
            if t == 0:
                lo, hi, h = 0, 1, dt
            else:
                lo, hi, h = t - 1, t + 1, 2.0 * dt
            twists[t, :3] = (poses[hi, :3] - poses[lo, :3]) / h
            drot = R.from_quat(poses[hi, 3:]) * R.from_quat(poses[lo, 3:]).inv()
            twists[t, 3:] = drot.as_rotvec() / h
        return twists

    def step(self, action: np.ndarray):
        """Take one env step in joint-space or rot6d-TCP-pose mode.

        - ``joint_action_mode in ("absolute", "delta")``: 16D action
          ``[L_j1..j7, L_grip, R_j1..j7, R_grip]``.
        - ``joint_action_mode == "tcp_rot6d_waypoint"``: 20D action
          ``[L_xyz(3), L_rot6d(6), L_grip, R_xyz(3), R_rot6d(6), R_grip]``;
          arm motion was already pre-submitted by ``dispatch_chunk`` as one
          ``CartesianWaypointMotion`` per arm; this step just paces 10 Hz
          and dispatches the gripper.
        - ``joint_action_mode == "tcp_rot6d_per_step"``: same 20D layout,
          but each step issues a per-arm ``CartesianMotion`` via
          ``move_arm``.
        """
        rot6d_mode = self.config.joint_action_mode in (
            "tcp_rot6d_waypoint",
            "tcp_rot6d_per_step",
        )
        per_arm_dim = ACTION_DIM_PER_ARM_ROT6D if rot6d_mode else ACTION_DIM_PER_ARM
        grip_idx = 9 if rot6d_mode else JOINT_DIM_PER_ARM  # slot 9 for rot6d, 7 for joint

        start_time = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        actions = action.reshape(NUM_ARMS, per_arm_dim)

        is_gripper_effective = [True, True]
        states = [self._left_state, self._right_state]
        ctrls = [self._left_ctrl, self._right_ctrl]

        if not self.config.is_dummy:
            dt = 1.0 / self.config.step_frequency

            target_joints: list[np.ndarray] = []
            tcp_targets: list[np.ndarray] = []  # per-arm [xyz, quat_xyzw]
            if rot6d_mode:
                for arm in range(NUM_ARMS):
                    xyz = np.clip(
                        actions[arm, 0:3],
                        self.config.ee_pose_limit_min[arm, :3],
                        self.config.ee_pose_limit_max[arm, :3],
                    )
                    # Live TCP quat serves as both the hemisphere-alignment
                    # reference and the decode fallback: if the model emits
                    # NaN or the two rot6d basis vectors are collinear,
                    # reuse the current TCP pose for this step rather than
                    # aborting the rollout.
                    live_quat = states[arm].tcp_pose[3:]
                    rot6d = actions[arm, 3:9]
                    quat = rot6d_to_quat_xyzw_safe(
                        rot6d, fallback_quat_xyzw=live_quat
                    )  # xyzw
                    if np.dot(quat, live_quat) < 0.0:
                        quat = -quat
                    tcp_targets.append(np.concatenate([xyz, quat]))
            else:
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
                    tj = self._clip_joint_velocity(
                        tj, states[arm].arm_joint_position, dt
                    )
                    target_joints.append(tj)

            # Grippers first so they don't contend with a fresh motion command.
            for arm in range(NUM_ARMS):
                gripper_val = (
                    actions[arm, grip_idx] * self.config.action_scale[2]
                )
                is_gripper_effective[arm] = self._gripper_action(
                    ctrls[arm], states[arm], gripper_val
                )

            # Motion dispatch.
            # - tcp_rot6d_waypoint: chunk already in flight (dispatch_chunk).
            # - tcp_rot6d_per_step: issue one CartesianMotion per arm via move_arm.
            # - joint modes: teleop direct-stream mode lets an external 1 kHz
            #   daemon own the motion; otherwise move_joints here.
            if self.config.joint_action_mode == "tcp_rot6d_waypoint":
                del tcp_targets  # not re-sent per-step in chunked waypoint mode
            elif self.config.joint_action_mode == "tcp_rot6d_per_step":
                left_f = ctrls[0].move_arm(tcp_targets[0].astype(np.float32))
                right_f = ctrls[1].move_arm(tcp_targets[1].astype(np.float32))
                left_f.wait()
                right_f.wait()
            elif not self.config.teleop_direct_stream:
                left_f = ctrls[0].move_joints(target_joints[0].astype(np.float32))
                right_f = ctrls[1].move_joints(target_joints[1].astype(np.float32))
                left_f.wait()
                right_f.wait()

        self._num_steps += 1
        if not self.config.is_dummy:
            if not self.config.teleop_direct_stream:
                # Non-direct mode: ``move_joints`` was just issued above and
                # franky needs the full period to track it before we sample
                # state, so sleep *between* action and state read.
                step_time = time.time() - start_time
                time.sleep(max(0.0, (1.0 / self.config.step_frequency) - step_time))
            # Direct-stream mode does not rate-limit here — motion is driven
            # by the 1 kHz daemon independently of ``env.step`` and the outer
            # collection loop owns the step period so that wrapper / record
            # overhead is included in the budget.  Callers MUST pace step()
            # themselves (e.g. ``DataCollector.run``).
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

        joint_position = np.concatenate(
            [
                self._left_state.arm_joint_position,
                self._right_state.arm_joint_position,
            ]
        )
        if self.config.joint_action_mode in ("tcp_rot6d_waypoint", "tcp_rot6d_per_step"):
            # Dataset state[0:20] after alphabetical concat (see
            # RealWorldEnv._wrap_obs) must be
            #   [L_grip, R_grip, L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]
            # so the rot6d SFT policy's _rearrange_state can rebuild
            # [L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]. gripper_position
            # fills [0:2] already; overload joint_position [2:20] with the
            # 18-d rot6d TCP pose encoding.
            joint_position = self._tcp_rot6d_18d()
        state = {
            "tcp_pose": np.concatenate(
                [self._left_state.tcp_pose, self._right_state.tcp_pose]
            ),
            "tcp_vel": np.concatenate(
                [self._left_state.tcp_vel, self._right_state.tcp_vel]
            ),
            "joint_position": joint_position,
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

    def _tcp_rot6d_18d(self) -> np.ndarray:
        """Build the ``[L_xyz, L_rot6d, R_xyz, R_rot6d]`` 18-d vector.

        Live TCP quat is xyzw (franky convention, see
        ``franky_controller.py:302-306``); it is converted to rot6d via
        ``scipy.Rotation.from_quat(...).as_matrix()`` then
        ``matrix_to_rot6d`` (first two columns flattened) — no euler
        on the path, so no wrap artifacts.
        """
        out = np.zeros(18, dtype=np.float32)
        for arm, st in enumerate((self._left_state, self._right_state)):
            base = arm * 9
            out[base : base + 3] = st.tcp_pose[:3]
            mat = R.from_quat(st.tcp_pose[3:]).as_matrix()
            out[base + 3 : base + 9] = matrix_to_rot6d(mat)
        return out
