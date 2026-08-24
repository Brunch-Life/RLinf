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

"""Vertical peg insertion against an independently controlled slot."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import sapien
import torch
from gymnasium import spaces
from gymnasium.vector.utils import batch_space
from mani_skill.agents.robots.panda import PandaWristCam
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.geometry.rotation_conversions import (
    euler_angles_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
)
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import SimConfig


def _tilt_error_to_down(peg_rotation: torch.Tensor) -> torch.Tensor:
    """Return the shortest world-frame rotation aligning local +Z with down."""
    peg_axis = peg_rotation[..., 2]
    downward = torch.zeros_like(peg_axis)
    downward[..., 2] = -1.0
    tilt_cross = torch.linalg.cross(peg_axis, downward, dim=-1)
    tilt_sin = torch.linalg.vector_norm(tilt_cross, dim=-1)
    tilt_cos = torch.sum(peg_axis * downward, dim=-1).clamp(-1.0, 1.0)
    tilt_angle = torch.atan2(tilt_sin, tilt_cos)
    fallback_axis = peg_rotation[..., 0]
    tilt_axis = torch.where(
        (tilt_sin > 1e-6).unsqueeze(-1),
        tilt_cross / tilt_sin.clamp_min(1e-6).unsqueeze(-1),
        fallback_axis,
    )
    return tilt_axis * tilt_angle.unsqueeze(-1)


@register_env("RealAgainstPegSlot-v0", max_episode_steps=200)
class PegSlotEnv(BaseEnv):
    """Insert a grasped vertical peg into an upward-facing movable slot.

    The public action is a normalized flat 9D vector. Its first six values are
    the end-effector delta pose and its last three values are the slot
    ``[dx, dy, dyaw]`` delta. The slot is a kinematic actor, so its height
    remains fixed while it translates in world XY and rotates about world Z.
    The gripper is held closed by the environment.
    """

    SUPPORTED_ROBOTS: ClassVar[list[str]] = ["panda_wristcam"]
    SUPPORTED_REWARD_MODES = ("sparse", "dense", "normalized_dense")
    agent: PandaWristCam

    PEG_HALF_SIZE = (0.009, 0.009, 0.060)
    SLOT_INNER_HALF_WIDTH = 0.012
    SLOT_OUTER_HALF_WIDTH = 0.070
    SLOT_HALF_HEIGHT = 0.012

    SLOT_XY_LIMIT = 0.18
    SLOT_ACTION_SCALE = 0.010
    SLOT_YAW_ACTION_SCALE = np.deg2rad(5.0)
    SLOT_RESET_YAW_RANGE = np.deg2rad(30.0)
    TCP_RESET_ROLL_PITCH_RANGE = np.deg2rad(15.0)
    ARM_POSITION_ACTION_SCALE = 0.25
    ARM_ROTATION_ACTION_SCALE = 0.25

    INSERTION_DEPTH = 0.010
    LATERAL_TOLERANCE = 0.0105
    ORIENTATION_TOLERANCE_RAD = np.deg2rad(12.0)

    _ROBOT_BASE_POSE = sapien.Pose(p=[-0.615, 0.0, 0.0])
    _ROBOT_QPOS = np.array(
        [
            0.0,
            np.pi / 8,
            0.0,
            -5 * np.pi / 8,
            0.0,
            3 * np.pi / 4,
            -np.pi / 4,
            0.008,
            0.008,
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        *args,
        robot_uids: str = "panda_wristcam",
        control_mode: str = "pd_ee_delta_pose",
        **kwargs,
    ):
        if control_mode != "pd_ee_delta_pose":
            raise ValueError("PegSlotEnv requires control_mode='pd_ee_delta_pose'")
        super().__init__(
            *args,
            robot_uids=robot_uids,
            control_mode=control_mode,
            **kwargs,
        )

        self.single_action_space = spaces.Box(-1.0, 1.0, shape=(9,), dtype=np.float32)
        self.action_space = (
            batch_space(self.single_action_space, n=self.num_envs)
            if self.num_envs > 1
            else self.single_action_space
        )

    @property
    def _default_sim_config(self) -> SimConfig:
        return SimConfig(sim_freq=200, control_freq=20)

    @property
    def _default_sensor_configs(self):
        camera_pose = sapien_utils.look_at(
            eye=[0.42, -0.52, 0.62], target=[0.0, 0.0, 0.12]
        )
        return [
            CameraConfig(
                "3rd_view_camera", camera_pose, 256, 256, 1.0, 0.01, 2.0
            )
        ]

    @property
    def _default_human_render_camera_configs(self):
        camera_pose = sapien_utils.look_at(
            eye=[0.72, -0.72, 0.72], target=[0.0, 0.0, 0.14]
        )
        return CameraConfig("render_camera", camera_pose, 512, 512, 1.0, 0.01, 3.0)

    def _load_agent(self, options: dict):
        super()._load_agent(options, self._ROBOT_BASE_POSE)

    def _load_scene(self, options: dict):
        del options
        self.table_scene = TableSceneBuilder(self, robot_init_qpos_noise=0)
        self.table_scene.build()

        peg_builder = self.scene.create_actor_builder()
        peg_builder.add_box_collision(half_size=self.PEG_HALF_SIZE)
        peg_material = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#F26B38"), roughness=0.45
        )
        peg_builder.add_box_visual(half_size=self.PEG_HALF_SIZE, material=peg_material)
        peg_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, 0.4])
        self.peg = peg_builder.build("peg")

        slot_builder = self.scene.create_actor_builder()
        inner = self.SLOT_INNER_HALF_WIDTH
        outer = self.SLOT_OUTER_HALF_WIDTH
        wall_half_width = (outer - inner) / 2
        wall_offset = (outer + inner) / 2
        half_height = self.SLOT_HALF_HEIGHT
        slot_material = sapien.render.RenderMaterial(
            base_color=sapien_utils.hex2rgba("#3274C8"), roughness=0.55
        )
        wall_specs = (
            ([wall_offset, 0.0, 0.0], [wall_half_width, outer, half_height]),
            ([-wall_offset, 0.0, 0.0], [wall_half_width, outer, half_height]),
            ([0.0, wall_offset, 0.0], [inner, wall_half_width, half_height]),
            ([0.0, -wall_offset, 0.0], [inner, wall_half_width, half_height]),
        )
        for position, half_size in wall_specs:
            pose = sapien.Pose(p=position)
            slot_builder.add_box_collision(pose=pose, half_size=half_size)
            slot_builder.add_box_visual(
                pose=pose, half_size=half_size, material=slot_material
            )
        slot_builder.initial_pose = sapien.Pose(p=[0.0, 0.0, self.SLOT_HALF_HEIGHT])
        self.slot = slot_builder.build_kinematic("slot")

        self._slot_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._slot_yaw = torch.zeros(self.num_envs, device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        del options
        with torch.device(self.device):
            env_idx = torch.as_tensor(env_idx, dtype=torch.long, device=self.device)
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)

            qpos = torch.as_tensor(self._ROBOT_QPOS, device=self.device).repeat(
                batch_size, 1
            )
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(torch.zeros_like(qpos))
            self.agent.robot.set_pose(self._ROBOT_BASE_POSE)

            # ``set_qpos`` only stages the articulation state on GPU. Refresh
            # forward kinematics before reading ``tcp.pose``; otherwise resets
            # after the first episode place the peg at the previous episode's
            # terminal TCP pose.
            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            # Randomize the grasped system's initial roll and pitch through the
            # robot IK, rather than rotating the peg independently of the TCP.
            # The target keeps the nominal TCP position and applies a
            # root/world-aligned orientation delta in both directions.
            arm_controller = self.agent.controller.controllers["arm"]
            current_tcp_at_base = arm_controller.ee_pose_at_base[env_idx]
            roll_pitch = (
                torch.rand((batch_size, 2), device=self.device) * 2.0 - 1.0
            ) * self.TCP_RESET_ROLL_PITCH_RANGE
            reset_euler = torch.zeros((batch_size, 3), device=self.device)
            reset_euler[:, :2] = roll_pitch
            delta_quaternion = matrix_to_quaternion(
                euler_angles_to_matrix(reset_euler, "XYZ")
            )
            target_tcp_at_base = Pose.create_from_pq(
                p=current_tcp_at_base.p,
                q=quaternion_multiply(delta_quaternion, current_tcp_at_base.q),
            )

            # GPU IK is a local Jacobian solve, so refine the absolute target a
            # few times in kinematics space before committing articulation qpos.
            reset_qpos = qpos
            ik_iterations = 3 if self.gpu_sim_enabled else 1
            for _ in range(ik_iterations):
                arm_qpos = arm_controller.kinematics.compute_ik(
                    pose=target_tcp_at_base,
                    q0=reset_qpos,
                    is_delta_pose=False,
                    solver_config=arm_controller.config.delta_solver_config,
                )
                if arm_qpos is None:
                    break
                reset_qpos = torch.cat((arm_qpos, qpos[:, 7:]), dim=-1)
            qpos = reset_qpos
            self.agent.robot.set_qpos(qpos)
            self.agent.robot.set_qvel(torch.zeros_like(qpos))

            if self.gpu_sim_enabled:
                self.scene._gpu_apply_all()
                self.scene.px.gpu_update_articulation_kinematics()
                self.scene._gpu_fetch_all()

            # Panda TCP local +Z points down in this reset pose. Keeping the peg's
            # local frame equal to the TCP frame makes its +Z the insertion axis.
            peg_offset = sapien.Pose(p=[0.0, 0.0, self.PEG_HALF_SIZE[2] - 0.006])
            peg_pose = self.agent.tcp.pose[env_idx] * peg_offset
            self.peg.set_pose(peg_pose)
            self.peg.set_linear_velocity(torch.zeros((batch_size, 3)))
            self.peg.set_angular_velocity(torch.zeros((batch_size, 3)))

            # Keep the existing XY range and independently randomize yaw so
            # demonstrations exercise the arm's yaw correction.
            noise = torch.rand((batch_size, 2), device=self.device) * 0.08 - 0.04
            slot_xy = self.agent.tcp.pose.p[env_idx, :2] + noise
            slot_xy.clamp_(-self.SLOT_XY_LIMIT, self.SLOT_XY_LIMIT)
            self._slot_xy[env_idx] = slot_xy
            self._slot_yaw[env_idx] = (
                torch.rand(batch_size, device=self.device) * 2.0 - 1.0
            ) * self.SLOT_RESET_YAW_RANGE
            self._set_slot_pose(env_idx)

    def _set_slot_pose(self, env_idx: torch.Tensor | None = None) -> None:
        if env_idx is None:
            slot_xy = self._slot_xy
            slot_yaw = self._slot_yaw
        else:
            slot_xy = self._slot_xy[env_idx]
            slot_yaw = self._slot_yaw[env_idx]
        slot_position = torch.zeros((len(slot_xy), 3), device=self.device)
        slot_position[:, :2] = slot_xy
        slot_position[:, 2] = self.SLOT_HALF_HEIGHT
        slot_quaternion = torch.zeros((len(slot_xy), 4), device=self.device)
        slot_quaternion[:, 0] = torch.cos(slot_yaw / 2)
        slot_quaternion[:, 3] = torch.sin(slot_yaw / 2)
        self.slot.set_pose(Pose.create_from_pq(p=slot_position, q=slot_quaternion))

    @property
    def peg_tip_pose(self):
        return self.peg.pose * sapien.Pose(p=[0.0, 0.0, self.PEG_HALF_SIZE[2]])

    @property
    def slot_hole_pose(self):
        return self.slot.pose

    def _prepare_action(self, action: Any) -> torch.Tensor:
        """Split the flat policy action and build the Panda controller action."""
        flat_action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if flat_action.ndim == 1:
            flat_action = flat_action.unsqueeze(0)
        expected_shape = (self.num_envs, 9)
        if flat_action.shape != expected_shape:
            raise ValueError(
                f"expected action {expected_shape}, got {tuple(flat_action.shape)}"
            )
        arm_action = flat_action[:, :6]
        slot_action = flat_action[:, 6:]

        slot_action = slot_action.clamp(-1.0, 1.0)
        slot_xy_delta = slot_action[:, :2] * self.SLOT_ACTION_SCALE
        self._slot_xy = (self._slot_xy + slot_xy_delta).clamp(
            -self.SLOT_XY_LIMIT, self.SLOT_XY_LIMIT
        )
        self._slot_yaw += slot_action[:, 2] * self.SLOT_YAW_ACTION_SCALE
        self._slot_yaw = (
            torch.remainder(self._slot_yaw + torch.pi, 2 * torch.pi) - torch.pi
        )
        self._set_slot_pose()

        scaled_arm_action = arm_action.clamp(-1.0, 1.0).clone()
        scaled_arm_action[:, :3] *= self.ARM_POSITION_ACTION_SCALE
        scaled_arm_action[:, 3:] *= self.ARM_ROTATION_ACTION_SCALE
        closed_gripper = -torch.ones((self.num_envs, 1), device=self.device)
        return torch.cat((scaled_arm_action, closed_gripper), dim=-1)

    def reset(self, *args, **kwargs):
        """Reset and expose the standardized RLinf observation in ``info``."""
        raw_obs, info = super().reset(*args, **kwargs)
        info["extracted_obs"] = self._build_rlinf_observation(raw_obs)
        return raw_obs, info

    def step(self, action):
        """Apply the public 9D action and expose the standardized observation."""
        raw_obs, reward, terminated, truncated, info = super().step(
            self._prepare_action(action)
        )
        info["extracted_obs"] = self._build_rlinf_observation(raw_obs)
        return raw_obs, reward, terminated, truncated, info

    def _alignment(self):
        tip_in_slot = self.slot_hole_pose.inv() * self.peg_tip_pose
        lateral_error = torch.linalg.vector_norm(tip_in_slot.p[:, :2], dim=-1)
        insertion_depth = self.SLOT_HALF_HEIGHT - tip_in_slot.p[:, 2]

        peg_axis = self.peg.pose.to_transformation_matrix()[..., :3, 2]
        downward = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        orientation_error = torch.acos(
            torch.sum(peg_axis * downward, dim=-1).clamp(-1.0, 1.0)
        )
        success = (
            (lateral_error <= self.LATERAL_TOLERANCE)
            & (insertion_depth >= self.INSERTION_DEPTH)
            & (insertion_depth <= 2 * self.SLOT_HALF_HEIGHT + 0.010)
            & (orientation_error <= self.ORIENTATION_TOLERANCE_RAD)
        )
        return (
            tip_in_slot,
            lateral_error,
            insertion_depth,
            orientation_error,
            success,
        )

    def evaluate(self):
        tip_in_slot, lateral_error, depth, orientation_error, success = (
            self._alignment()
        )
        return {
            "success": success,
            "peg_tip_in_slot": tip_in_slot.p,
            "lateral_error": lateral_error,
            "insertion_depth": depth,
            "orientation_error": orientation_error,
            "slot_xy": self._slot_xy.clone(),
            "slot_yaw": self._slot_yaw.clone(),
        }

    def _get_obs_extra(self, info: dict):
        observation = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "slot_xy": self._slot_xy,
            "slot_yaw": self._slot_yaw[:, None],
        }
        if self.obs_mode_struct.use_state:
            observation.update(
                peg_pose=self.peg.pose.raw_pose,
                peg_tip_pose=self.peg_tip_pose.raw_pose,
                slot_pose=self.slot.pose.raw_pose,
                peg_tip_in_slot=info["peg_tip_in_slot"],
            )
        return observation

    def _build_rlinf_observation(self, raw_obs: dict) -> dict:
        """Build the standardized observation consumed by RLinf OpenPI.

        This mirrors the expert dataset exactly: task and wrist RGB images plus
        the Panda's nine joint positions. Slot pose is observed through RGB.
        """
        sensor_data = raw_obs["sensor_data"]
        state = raw_obs["agent"]["qpos"][..., :9].to(dtype=torch.float32)
        return {
            "main_images": sensor_data["3rd_view_camera"]["rgb"].to(torch.uint8),
            "wrist_images": sensor_data["hand_camera"]["rgb"].to(torch.uint8),
            "extra_view_images": None,
            "states": state,
            "task_descriptions": self.get_language_instruction(),
        }

    def get_language_instruction(self):
        return [
            "insert the grasped peg downward into the movable slot"
            for _ in range(self.num_envs)
        ]

    def compute_expert_action(self) -> torch.Tensor:
        """Servo position and all three orientation axes toward the hole."""
        position_error = self.slot_hole_pose.p - self.peg_tip_pose.p
        peg_rotation = self.peg.pose.to_transformation_matrix()[..., :3, :3]

        tilt_error = _tilt_error_to_down(peg_rotation)

        peg_tip_in_hole = self.slot_hole_pose.inv() * self.peg_tip_pose
        relative_rotation = peg_tip_in_hole.to_transformation_matrix()[..., :3, :3]
        peg_yaw_in_hole = torch.atan2(
            relative_rotation[:, 1, 0], relative_rotation[:, 0, 0]
        )
        # Peg and hole are square, so rotations separated by 90 degrees are
        # equivalent. Fold the relative yaw onto the nearest aligned pose.
        yaw_error = (
            torch.atan2(
                torch.sin(4.0 * peg_yaw_in_hole),
                torch.cos(4.0 * peg_yaw_in_hole),
            )
            / 4.0
        )

        action = torch.zeros((self.num_envs, 9), device=self.device)
        action[:, :2] = (position_error[:, :2] / 0.025).clamp(-0.7, 0.7)
        action[:, 2] = (position_error[:, 2] / 0.025).clamp(-0.16, 0.0)
        # ``_prepare_action`` and Panda's normalized controller together map a
        # public rotation action to -0.025 rad in the world frame. Negate the
        # desired tilt correction accordingly. Yaw already represents
        # peg-minus-hole, so its sign is positive here. Position, tilt, and yaw
        # are corrected simultaneously; the expert never waits or lifts first.
        action[:, 3:5] = (-tilt_error[:, :2] / 0.025).clamp(-1.0, 1.0)
        action[:, 5] = (yaw_error / 0.025).clamp(-1.0, 1.0)
        return action

    def compute_sparse_reward(self, obs, action, info: dict):
        return info["success"].to(torch.float32)

    def compute_dense_reward(self, obs: Any, action, info: dict):
        lateral_reward = 1.0 - torch.tanh(12.0 * info["lateral_error"])
        orientation_reward = 1.0 - torch.tanh(info["orientation_error"])
        depth_reward = torch.clamp(
            info["insertion_depth"] / self.INSERTION_DEPTH, 0.0, 1.0
        )
        reward = lateral_reward + orientation_reward + 2.0 * depth_reward
        reward[info["success"]] = 5.0
        return reward

    def compute_normalized_dense_reward(self, obs, action, info: dict):
        return self.compute_dense_reward(obs, action, info) / 5.0
