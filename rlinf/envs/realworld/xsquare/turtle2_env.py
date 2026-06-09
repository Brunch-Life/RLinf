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

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.xsquare.chunk_interpolation import interpolate_segment
from rlinf.envs.realworld.xsquare.turtle2_robot_state import Turtle2RobotState
from rlinf.scheduler import (
    Turtle2HWInfo,
    WorkerInfo,
)
from rlinf.utils.logging import get_logger


@dataclass
class Turtle2RobotConfig:
    use_camera_ids: list[int] = field(default_factory=lambda: [2])  # [0, 1, 2]
    use_arm_ids: list[int] = field(default_factory=lambda: [1])  # [0, 1]

    is_dummy: bool = True
    use_dense_reward: bool = False
    step_frequency: float = 10.0  # Max number of steps per second
    smooth_frequency: int = 50  # Frequency for smooth controller

    # Deployment (absolute-pose) options; defaults keep the existing RL behavior.
    exec_frequency: float = 60.0  # high-frequency dispatch rate for abs mode
    state_format: str = "xyz_quat"  # or "xyz_rpy_gripper"
    camera_names: Optional[list] = None  # view names ordered by use_camera_ids
    task_description: Optional[str] = None
    use_arm: Optional[str] = None  # "dual"/"left"/"right"; None -> use_arm_ids

    # Positions are stored in eular angles (xyz for position, rzryrx for orientation)
    # It will be converted to quaternions internally
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.15, 0.0, 1, 0.0]]
        )
    )
    reset_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array(
            [[0.3, 0, 0.0, 0.2, 0, 0], [0.1, 0, 0.1, 0, 0.8, 0.0]]
        )
    )

    max_num_steps: int = 100
    reward_threshold: np.ndarray = field(default_factory=lambda: np.zeros((2, 6)))
    action_scale: np.ndarray = field(
        default_factory=lambda: np.ones(3)
    )  # [xyz move scale, orientation scale, gripper scale]
    enable_random_reset: bool = False

    random_xy_range: float = 0.05
    random_rz_range: float = np.pi / 10

    # Robot parameters
    # Same as the position arrays: first 3 are position limits, last 3 are orientation limits
    ee_pose_limit_min: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), -np.inf)
    )
    ee_pose_limit_max: np.ndarray = field(
        default_factory=lambda: np.full((2, 6), np.inf)
    )
    gripper_width_limit_min: float = 0.0
    gripper_width_limit_max: float = 5.0
    enforce_gripper_close: bool = True
    enable_gripper_penalty: bool = True
    gripper_penalty: float = 0.1
    save_video_path: Optional[str] = None


class Turtle2Env(gym.Env):
    """Gymnasium environment wrapping the Turtle2 dual-arm robot.

    Supports single- and dual-arm control with optional camera observations,
    dense/sparse rewards, safety-box clipping, and a dummy mode for offline use.
    """

    POSE_MODE: str = "delta"

    def __init__(
        self,
        config: Turtle2RobotConfig,
        worker_info: Optional[WorkerInfo],
        hardware_info: Optional[Turtle2HWInfo],
        env_idx: int,
    ) -> None:
        """Initialize Turtle2Env.

        Args:
            config: Robot and environment configuration.
            worker_info: Scheduler worker info used to resolve node/worker rank.
            hardware_info: Hardware descriptor for the Turtle2 platform.
            env_idx: Index of this environment instance within its worker.
        """
        self._logger = get_logger()
        self.config = config
        if self.config.use_arm is not None:
            self.config.use_arm_ids = {"dual": [0, 1], "left": [0], "right": [1]}[
                self.config.use_arm
            ]
        self.hardware_info = hardware_info
        self.env_idx = env_idx
        self.node_rank = 0
        self.env_worker_rank = 0
        if worker_info is not None:
            self.node_rank = worker_info.cluster_node_rank
            self.env_worker_rank = worker_info.rank

        assert len(self.config.use_arm_ids) > 0 and len(self.config.use_arm_ids) <= 2, (
            "please choose arm IDs from [0, 1]."
        )
        assert (
            len(self.config.use_camera_ids) > 0 and len(self.config.use_camera_ids) <= 3
        ), "please choose camera IDs from [0, 1, 2]."
        self._turtle2_state = Turtle2RobotState()
        self._num_steps = 0
        # Previous absolute target for abs-dispatch interpolation (abs path only).
        self._last_abs_target = None

        if not self.config.is_dummy:
            self._setup_hardware()

        # Init action and observation spaces
        self._init_action_obs_spaces()

        if self.config.is_dummy:
            return

        # Wait for the first frame
        self._reset_arms()
        self._turtle2_state = self._controller.get_state().wait()[0]

        # Init cameras
        self._check_cameras()
        # Video player for displaying camera frames

    def _setup_hardware(self):
        from .turtle2_smooth_controller import Turtle2SmoothController

        assert self.env_idx >= 0, "env_idx must be set for Turtle2Env."

        # Launch Turtle controller
        self._controller = Turtle2SmoothController.launch_controller(
            freq=self.config.smooth_frequency,
            env_idx=self.env_idx,
            node_rank=self.node_rank,
            worker_rank=self.env_worker_rank,
            pose_mode=self.POSE_MODE,
        )

    def _init_action_obs_spaces(self):
        """Initialize action and observation spaces, including arm safety box."""
        self._xyz_safe_space1 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[0, :3].flatten(),
            high=self.config.ee_pose_limit_max[0, :3].flatten(),
            dtype=np.float64,
        )
        self._rpy_safe_space1 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[0, 3:].flatten(),
            high=self.config.ee_pose_limit_max[0, 3:].flatten(),
            dtype=np.float64,
        )
        self._xyz_safe_space2 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[1, :3].flatten(),
            high=self.config.ee_pose_limit_max[1, :3].flatten(),
            dtype=np.float64,
        )
        self._rpy_safe_space2 = gym.spaces.Box(
            low=self.config.ee_pose_limit_min[1, 3:].flatten(),
            high=self.config.ee_pose_limit_max[1, 3:].flatten(),
            dtype=np.float64,
        )
        self.action_space = gym.spaces.Box(
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32) * -1,
            np.ones((len(self.config.use_arm_ids) * 7), dtype=np.float32),
        )

        num_arms = len(self.config.use_arm_ids)
        if self.config.state_format not in ("xyz_rpy_gripper", "xyz_quat"):
            raise ValueError(
                f"Unsupported state_format={self.config.state_format!r}; "
                "expected 'xyz_rpy_gripper' or 'xyz_quat'."
            )
        # tcp_pose is 7 per arm either way: xyz+rpy+gripper, or xyz+quat.
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "tcp_pose": gym.spaces.Box(
                            -np.inf, np.inf, shape=(num_arms * 7,)
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        key: gym.spaces.Box(
                            0, 255, shape=(128, 128, 3), dtype=np.uint8
                        )
                        for key in self._frame_keys()
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    def _frame_keys(self) -> list[str]:
        if not self.config.camera_names:
            raise ValueError(
                "camera_names must list a view name per use_camera_ids entry, e.g. "
                "['face_view', 'left_wrist_view', 'right_wrist_view']."
            )
        if len(self.config.camera_names) != len(self.config.use_camera_ids):
            raise ValueError("camera_names length must match use_camera_ids.")
        return list(self.config.camera_names)

    def get_abs_action_space(self) -> gym.spaces.Box:
        """Absolute ee-pose action space (per arm: xyz, rpy, gripper)."""
        lows, highs = [], []
        for arm_id in self.config.use_arm_ids:
            lows.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_min[arm_id],
                        [self.config.gripper_width_limit_min],
                    ]
                )
            )
            highs.append(
                np.concatenate(
                    [
                        self.config.ee_pose_limit_max[arm_id],
                        [self.config.gripper_width_limit_max],
                    ]
                )
            )
        return gym.spaces.Box(
            low=np.concatenate(lows).astype(np.float32),
            high=np.concatenate(highs).astype(np.float32),
            dtype=np.float32,
        )

    def _build_tcp_pose(self) -> np.ndarray:
        """Per-arm state. ``xyz_rpy_gripper`` concatenates gripper (x2robot ckpt
        layout); ``xyz_quat`` is xyz+quat without gripper (RL / franka style)."""
        poses = {
            0: self._turtle2_state.follow1_pos,
            1: self._turtle2_state.follow2_pos,
        }
        parts = []
        for arm_id in self.config.use_arm_ids:
            p = poses[arm_id]
            if self.config.state_format == "xyz_rpy_gripper":
                parts.append(np.asarray(p[:7], dtype=np.float64))  # xyz, rpy, gripper
            else:  # xyz_quat (validated in _init_action_obs_spaces)
                quat = R.from_euler("xyz", p[3:6]).as_quat()
                parts.append(np.concatenate([p[0:3], quat]).astype(np.float64))
        return np.concatenate(parts, axis=0)

    def get_current_pose(self) -> np.ndarray:
        """Current absolute ee pose (xyz, rpy, gripper per selected arm)."""
        poses = {
            0: self._turtle2_state.follow1_pos,
            1: self._turtle2_state.follow2_pos,
        }
        return np.concatenate(
            [
                np.asarray(poses[a][:7], dtype=np.float64)
                for a in self.config.use_arm_ids
            ]
        )

    def _reset_arms(self) -> None:
        """Move both arms to their reset poses, blocking until they arrive.

        Does nothing in dummy mode.
        """
        if self.config.is_dummy:
            return

        self._logger.info("pre-reset")
        self._controller.move_delta(
            [0.2, 0, 0.1, 0, 0, 0, 0], [0.2, 0, 0.1, 0, 0, 0, 0]
        ).wait()
        time.sleep(2.0)

        if self.config.enable_random_reset:
            random_xy1 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_xy2 = np.random.uniform(
                -self.config.random_xy_range, self.config.random_xy_range, (2,)
            )
            random_euler1 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3,)
            )
            random_euler2 = np.random.uniform(
                -self.config.random_rz_range, self.config.random_rz_range, (3,)
            )
        else:
            random_xy1 = np.zeros(2)
            random_xy2 = np.zeros(2)
            random_euler1 = np.zeros(3)
            random_euler2 = np.zeros(3)

        if 0 in self.config.use_arm_ids:
            left_arm_reset_pose = self.config.reset_ee_pose[0].copy()
            left_arm_reset_pose[:2] += random_xy1
            left_arm_reset_pose[3:6] += random_euler1
            left_arm_reset_pose = left_arm_reset_pose.tolist()
            left_arm_reset_pose.append(0.0)
        else:
            left_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]
        if 1 in self.config.use_arm_ids:
            right_arm_reset_pose = self.config.reset_ee_pose[1].copy()
            right_arm_reset_pose[:2] += random_xy2
            right_arm_reset_pose[3:6] += random_euler2
            right_arm_reset_pose = right_arm_reset_pose.tolist()
            right_arm_reset_pose.append(0.0)
        else:
            right_arm_reset_pose = [0, 0, 0, 0, 0, 0, 0]

        self._logger.info(
            "Going to reset: left=%s, right=%s",
            repr(left_arm_reset_pose),
            repr(right_arm_reset_pose),
        )

        self._controller.move_delta(left_arm_reset_pose, right_arm_reset_pose).wait()

        reach = False
        start_time = time.time()
        while not reach:
            state = self._controller.get_state().wait()[0]
            left_pos = state.follow1_pos
            right_pos = state.follow2_pos
            left_reach = (
                np.linalg.norm(left_pos[:6] - np.array(left_arm_reset_pose)[:6]) < 0.04
                if 0 in self.config.use_arm_ids
                else True
            )
            right_reach = (
                np.linalg.norm(right_pos[:6] - np.array(right_arm_reset_pose)[:6])
                < 0.04
                if 1 in self.config.use_arm_ids
                else True
            )
            reach = left_reach and right_reach
            if time.time() - start_time > 10.0:
                left_err = np.linalg.norm(
                    left_pos[:6] - np.array(left_arm_reset_pose)[:6]
                )
                right_err = np.linalg.norm(
                    right_pos[:6] - np.array(right_arm_reset_pose)[:6]
                )
                raise ValueError(
                    f"Reset arms timeout: left_err={left_err:.6f}, right_err={right_err:.6f}"
                )

            time.sleep(0.1)
        time.sleep(0.5)
        return

    def _check_cameras(self):
        if self.config.is_dummy:
            return

        cam1_ok, cam2_ok, cam3_ok = self._controller.check_cams().wait()[0]
        if 0 in self.config.use_camera_ids and not cam1_ok:
            raise ValueError("Camera 1 not available.")
        if 1 in self.config.use_camera_ids and not cam2_ok:
            raise ValueError("Camera 2 not available.")
        if 2 in self.config.use_camera_ids and not cam3_ok:
            raise ValueError("Camera 3 not available.")

    def reset(self, *, seed=None, options=None):
        if self.config.is_dummy:
            observation = self._get_observation()
            return observation, {}

        # Reset
        self._reset_arms()
        self._num_steps = 0
        self._turtle2_state = self._controller.get_state().wait()[0]
        # Next abs step interpolates from the current (reset) pose.
        self._last_abs_target = None
        observation = self._get_observation()
        # save if debug
        # for key in observation["frames"].keys():
        #     img = Image.fromarray(observation["frames"][key])
        #     img.save(f'{key}.jpg')

        return observation, {}

    def transform_action_ee_to_base(self, action: np.ndarray) -> np.ndarray:
        """Transform action from end-effector frame to base frame.

        Args:
            action: Action array in end-effector coordinates.

        Returns:
            Action array in base frame coordinates.
        """
        action[:6] = np.linalg.inv(self.adjoint_matrix) @ action[:6]
        return action

    def step(
        self, action: np.ndarray, dispatch: str = "delta"
    ) -> tuple[dict, float, bool, bool, dict]:
        """Take a step in the environment.

        Args:
            action: For ``dispatch='delta'`` a delta ee action of shape ``(7,)``
                single arm / ``(14,)`` dual arm; for ``dispatch='abs'`` one absolute
                ee-pose frame of the same shape (xyz, rpy, gripper per arm).
            dispatch: ``'delta'`` (speed-clamped) or ``'abs'`` (interpolated direct
                dispatch from the previous absolute target to this frame).

        Returns:
            Tuple of ``(observation, reward, terminated, truncated, info)``.
        """
        if dispatch == "delta":
            return self._step_delta(action)
        if dispatch == "abs":
            return self._step_abs(action)
        raise ValueError(
            f"Unsupported dispatch={dispatch!r}; expected 'delta' or 'abs'."
        )

    def _step_delta(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        assert action.shape == (len(self.config.use_arm_ids) * 7,), (
            f"Action shape must be {(len(self.config.use_arm_ids) * 7,)}, but got {action.shape}."
        )

        start_time = time.time()

        action = np.clip(action, self.action_space.low, self.action_space.high)

        # deal with dual arms (xyz)
        action = action.reshape(-1, 7)
        xyz_delta = action[:, :3]

        next_position1 = self._turtle2_state.follow1_pos.copy()
        next_position2 = self._turtle2_state.follow2_pos.copy()

        if 0 in self.config.use_arm_ids:
            next_position1[:3] = (
                next_position1[:3] + xyz_delta[0] * self.config.action_scale[0]
            )
        if 1 in self.config.use_arm_ids:
            next_position2[:3] = (
                next_position2[:3] + xyz_delta[-1] * self.config.action_scale[0]
            )

        # deal with dual arms (rpy)
        if 0 in self.config.use_arm_ids:
            next_position1[3:6] = (
                next_position1[3:6] + action[0, 3:6] * self.config.action_scale[1]
            )
        if 1 in self.config.use_arm_ids:
            next_position2[3:6] = (
                next_position2[3:6] + action[-1, 3:6] * self.config.action_scale[1]
            )

        if self.config.enforce_gripper_close:
            next_position1[6] = self.config.gripper_width_limit_min
            next_position2[6] = self.config.gripper_width_limit_min
        else:
            if 0 in self.config.use_arm_ids:
                next_position1[6] = action[0, 6]
            if 1 in self.config.use_arm_ids:
                next_position2[6] = action[-1, 6]

        # clip to safety box
        next_position = self._clip_position_to_safety_box(
            np.stack([next_position1, next_position2])
        )
        next_position1 = next_position[0]
        next_position2 = next_position[1]

        if not self.config.is_dummy:
            self._controller.move_delta(
                next_position1.tolist(), next_position2.tolist()
            ).wait()
        else:
            pass

        self._num_steps += 1
        step_time = time.time() - start_time
        time.sleep(max(0, (1.0 / self.config.step_frequency) - step_time))

        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        else:
            self._turtle2_state = self._turtle2_state
        observation = self._get_observation()
        reward = self._calc_step_reward(observation)
        terminated = reward == 1
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, reward, terminated, truncated, {}

    def _step_abs(
        self, action: np.ndarray
    ) -> tuple[dict, float, bool, bool, dict]:
        """Consume one absolute ee-pose frame.

        Interpolates from the previous absolute target to this frame and dispatches
        each substep at ``exec_frequency`` (direct, unclamped), then returns a single
        observation with reward 0.
        """
        frame = np.asarray(action, dtype=np.float64).reshape(-1)
        if self._last_abs_target is None:
            self._last_abs_target = self.get_current_pose()
        num_substeps = max(
            1, round(self.config.exec_frequency / self.config.step_frequency)
        )
        dt = 1.0 / self.config.exec_frequency
        for point in interpolate_segment(self._last_abs_target, frame, num_substeps):
            t0 = time.time()
            self.apply_abs_pose(point)
            time.sleep(max(0.0, dt - (time.time() - t0)))
        self._last_abs_target = frame
        return self.observe_only()

    def apply_abs_pose(self, pose: np.ndarray) -> None:
        """Dispatch one absolute ee pose, bypassing speed clamping.

        Clips to the safety box and writes to the controller without fetching
        observations; called by ``_step_abs`` for per-substep dispatch.
        Arms not in ``use_arm_ids`` keep their current pose.
        """
        pose = np.asarray(pose, dtype=np.float64).reshape(-1, 7)
        next_position1 = self._turtle2_state.follow1_pos.copy().astype(np.float64)
        next_position2 = self._turtle2_state.follow2_pos.copy().astype(np.float64)
        targets = {0: next_position1, 1: next_position2}
        for row, arm_id in zip(pose, self.config.use_arm_ids, strict=False):
            targets[arm_id][:] = row

        if self.config.enforce_gripper_close:
            next_position1[6] = self.config.gripper_width_limit_min
            next_position2[6] = self.config.gripper_width_limit_min

        next_position = self._clip_position_to_safety_box(
            np.stack([next_position1, next_position2])
        )
        if not self.config.is_dummy:
            self._controller.move_abs(
                next_position[0].tolist(), next_position[1].tolist()
            ).wait()
        else:
            self._turtle2_state.follow1_pos = next_position[0].astype(np.float32)
            self._turtle2_state.follow2_pos = next_position[1].astype(np.float32)

    def observe_only(self) -> tuple[dict, float, bool, bool, dict]:
        """Refresh state and return observation with reward 0 (deployment path)."""
        self._num_steps += 1
        if not self.config.is_dummy:
            self._turtle2_state = self._controller.get_state().wait()[0]
        observation = self._get_observation()
        truncated = self._num_steps >= self.config.max_num_steps
        return observation, 0.0, False, truncated, {}

    @property
    def task_description(self):
        return self.config.task_description or ""

    @property
    def num_steps(self):
        return self._num_steps

    def _calc_step_reward(
        self,
        observation: dict[str, np.ndarray],
    ) -> float:
        """Compute the per-step reward from the current robot state.

        Args:
            observation: Current observation dict (unused directly; reward is
                derived from internal robot state).

        Returns:
            ``1.0`` on success, a dense exponential reward when
            ``use_dense_reward`` is set, or ``0.0`` otherwise.
        """
        if not self.config.is_dummy:
            # Convert orientation to euler angles
            position1 = self._turtle2_state.follow1_pos[0:6]
            position2 = self._turtle2_state.follow2_pos[0:6]
            delta1 = np.abs(position1 - self.config.target_ee_pose[0, 0:6])
            delta2 = np.abs(position2 - self.config.target_ee_pose[1, 0:6])

            success1 = (
                np.all(delta1 <= self.config.reward_threshold)
                if 0 in self.config.use_arm_ids
                else True
            )
            success2 = (
                np.all(delta2 <= self.config.reward_threshold)
                if 1 in self.config.use_arm_ids
                else True
            )
            is_success = success1 and success2

            if is_success:
                reward = 1.0
            else:
                if self.config.use_dense_reward:
                    delta1_sq = (
                        np.sum(np.square(delta1[0:6]))
                        if 0 in self.config.use_arm_ids
                        else 0.0
                    )
                    delta2_sq = (
                        np.sum(np.square(delta2[0:6]))
                        if 1 in self.config.use_arm_ids
                        else 0.0
                    )
                    reward = np.exp(-200 * (delta1_sq + delta2_sq))
                else:
                    reward = 0.0
                self._logger.debug(
                    f"Does not meet success criteria."
                    f"Success threshold: {self.config.reward_threshold}, "
                    f"Current reward={reward}",
                )

            return reward
        else:
            return 0.0

    def _crop_frame(
        self, frame: np.ndarray, reshape_size: tuple[int, int]
    ) -> np.ndarray:
        """Letterbox the frame to ``reshape_size`` (aspect-preserving resize +
        symmetric black padding), replicating openpi ``resize_with_pad``.

        The fold_towel checkpoint was trained on the FULL 640x480 frame fed
        through ``resize_with_pad(224, 224)`` (letterboxed, top/bottom padding) —
        NOT a center crop. The working reference deploy likewise sends the full
        frame and lets the model's ``resize_with_pad`` letterbox it. A center
        crop would change the field of view and drop the letterbox padding the
        policy expects, so we must letterbox here too. (resize_with_pad to 128
        then the model's resize_with_pad to 224 yields the same framing as the
        reference's direct 640->224, only at lower intermediate resolution.)

        Channel order: the turtle2 camera returns RGB (``imgmsg_to_cv2`` with
        ``"rgb8"``) and the reference feeds the model RGB too (its server does
        ``cv2.cvtColor(..., COLOR_BGR2RGB)`` after decoding the bgr8 JPEG), so we
        keep RGB and do NOT swap channels.
        """
        target_h, target_w = reshape_size
        h, w = frame.shape[:2]
        ratio = max(w / target_w, h / target_h)
        resized_w = max(1, int(w / ratio))
        resized_h = max(1, int(h / ratio))
        resized = cv2.resize(
            frame, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR
        )
        pad_h0, rem_h = divmod(target_h - resized_h, 2)
        pad_h1 = pad_h0 + rem_h
        pad_w0, rem_w = divmod(target_w - resized_w, 2)
        pad_w1 = pad_w0 + rem_w
        return cv2.copyMakeBorder(
            resized, pad_h0, pad_h1, pad_w0, pad_w1,
            cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )

    # Robot actions

    def _clip_position_to_safety_box(self, position: np.ndarray) -> np.ndarray:
        """Clip the position array to be within the safety box."""
        position[0, 0:3] = np.clip(
            position[0, 0:3], self._xyz_safe_space1.low, self._xyz_safe_space1.high
        )
        position[0, 3:6] = np.clip(
            position[0, 3:6], self._rpy_safe_space1.low, self._rpy_safe_space1.high
        )
        position[0, 6] = np.clip(
            position[0, 6],
            self.config.gripper_width_limit_min,
            self.config.gripper_width_limit_max,
        )
        position[1, 0:3] = np.clip(
            position[1, 0:3], self._xyz_safe_space2.low, self._xyz_safe_space2.high
        )
        position[1, 3:6] = np.clip(
            position[1, 3:6], self._rpy_safe_space2.low, self._rpy_safe_space2.high
        )
        position[1, 6] = np.clip(
            position[1, 6],
            self.config.gripper_width_limit_min,
            self.config.gripper_width_limit_max,
        )

        position = position.reshape(2, -1)
        return position

    def _get_observation(self) -> dict[str, dict[str, np.ndarray]]:
        """Get current observation from robot state and cameras.

        State layout follows ``state_format`` (single ``tcp_pose`` key); frame keys
        follow ``camera_names``.
        """
        if self.config.is_dummy:
            return self._base_observation_space.sample()

        frames = self._controller.get_cams(self.config.use_camera_ids).wait()[0]
        assert len(frames) == len(self.config.use_camera_ids), "get frames failed."
        frames = [self._crop_frame(f, (128, 128)) for f in frames]
        frames_dict = dict(zip(self._frame_keys(), frames, strict=True))

        observation = {
            "state": {"tcp_pose": self._build_tcp_pose()},
            "frames": frames_dict,
        }
        return copy.deepcopy(observation)
