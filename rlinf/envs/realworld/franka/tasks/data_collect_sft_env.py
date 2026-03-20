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

"""Configurable Franka env used for generic SFT-style data collection.

This env reuses the standard :class:`FrankaEnv` reset/step pipeline and keeps
the collection-time runtime parameters in ``override_cfg`` so users do not need
to duplicate a task env just to change prompt text, target pose, reset pose,
workspace limits, or hardware settings.

Task-specific behaviors that need custom reset trajectories or obstacle-aware
rest motions should still be implemented in dedicated task envs.
"""

from dataclasses import dataclass, field

import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class DataCollectSFTConfig(FrankaRobotConfig):
    """Config for the generic real-world SFT data-collection env."""

    task_description: str = ""
    target_ee_pose: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0])
    )
    reward_threshold: np.ndarray = field(
        default_factory=lambda: np.array([0.01, 0.01, 0.01, 0.2, 0.2, 0.2])
    )
    random_xy_range: float = 0.05
    random_z_range_low: float = 0.0
    random_z_range_high: float = 0.1
    random_rz_range: float = np.pi / 6
    enable_random_reset: bool = False
    enable_gripper_penalty: bool = False
    action_scale: np.ndarray = field(
        default_factory=lambda: np.array([0.02, 0.1, 1.0])
    )

    def __post_init__(self):
        if not self.compliance_param:
            self.compliance_param = {
                "translational_stiffness": 2000,
                "translational_damping": 89,
                "rotational_stiffness": 150,
                "rotational_damping": 7,
                "translational_Ki": 0,
                "translational_clip_x": 0.003,
                "translational_clip_y": 0.003,
                "translational_clip_z": 0.01,
                "translational_clip_neg_x": 0.003,
                "translational_clip_neg_y": 0.003,
                "translational_clip_neg_z": 0.01,
                "rotational_clip_x": 0.02,
                "rotational_clip_y": 0.02,
                "rotational_clip_z": 0.02,
                "rotational_clip_neg_x": 0.02,
                "rotational_clip_neg_y": 0.02,
                "rotational_clip_neg_z": 0.02,
                "rotational_Ki": 0,
            }
        if not self.precision_param:
            self.precision_param = {
                "translational_stiffness": 3000,
                "translational_damping": 89,
                "rotational_stiffness": 300,
                "rotational_damping": 9,
                "translational_Ki": 0.1,
                "translational_clip_x": 0.01,
                "translational_clip_y": 0.01,
                "translational_clip_z": 0.01,
                "translational_clip_neg_x": 0.01,
                "translational_clip_neg_y": 0.01,
                "translational_clip_neg_z": 0.01,
                "rotational_clip_x": 0.05,
                "rotational_clip_y": 0.05,
                "rotational_clip_z": 0.05,
                "rotational_clip_neg_x": 0.05,
                "rotational_clip_neg_y": 0.05,
                "rotational_clip_neg_z": 0.05,
                "rotational_Ki": 0.1,
            }

        self.target_ee_pose = np.array(self.target_ee_pose)
        if np.allclose(np.asarray(self.reset_ee_pose), 0):
            self.reset_ee_pose = self.target_ee_pose + np.array(
                [0.0, 0.0, self.random_z_range_high, 0.0, 0.0, 0.0]
            )
        else:
            self.reset_ee_pose = np.array(self.reset_ee_pose)

        self.reward_threshold = np.array(self.reward_threshold)
        self.action_scale = np.array(self.action_scale)

        explicit_min = np.array(self.ee_pose_limit_min)
        explicit_max = np.array(self.ee_pose_limit_max)
        if np.any(explicit_min != 0) or np.any(explicit_max != 0):
            self.ee_pose_limit_min = explicit_min
            self.ee_pose_limit_max = explicit_max
        else:
            # Derive a conservative default workspace from the configured reset
            # jitter ranges. Task-specific collection setups should override the
            # limits explicitly in YAML when they need a larger or asymmetric box.
            self.ee_pose_limit_min = np.array(
                [
                    self.target_ee_pose[0] - self.random_xy_range,
                    self.target_ee_pose[1] - self.random_xy_range,
                    self.target_ee_pose[2] - self.random_z_range_low,
                    self.target_ee_pose[3] - 0.01,
                    self.target_ee_pose[4] - 0.01,
                    self.target_ee_pose[5] - self.random_rz_range,
                ]
            )
            self.ee_pose_limit_max = np.array(
                [
                    self.target_ee_pose[0] + self.random_xy_range,
                    self.target_ee_pose[1] + self.random_xy_range,
                    self.target_ee_pose[2] + self.random_z_range_high,
                    self.target_ee_pose[3] + 0.01,
                    self.target_ee_pose[4] + 0.01,
                    self.target_ee_pose[5] + self.random_rz_range,
                ]
            )


class DataCollectSFTEnv(FrankaEnv):
    """Generic real-world env used by SFT data collection."""

    def __init__(self, override_cfg, worker_info=None, hardware_info=None, env_idx=0):
        config = DataCollectSFTConfig(**override_cfg)
        super().__init__(config, worker_info, hardware_info, env_idx)

    @property
    def task_description(self):
        return self.config.task_description
