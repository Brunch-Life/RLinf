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

import gymnasium as gym
import numpy as np
import pytest


def _import_data_collect_sft():
    pytest.importorskip("cv2")
    pytest.importorskip("scipy")
    from rlinf.envs.realworld.franka.tasks.data_collect_sft_env import (
        DataCollectSFTConfig,
        DataCollectSFTEnv,
    )

    return DataCollectSFTConfig, DataCollectSFTEnv


def test_data_collect_sft_env_is_registered():
    import rlinf.envs.realworld.franka.tasks  # noqa: F401

    spec = gym.spec("DataCollectSFTEnv-v1")
    assert spec.id == "DataCollectSFTEnv-v1"
    assert spec.entry_point == "rlinf.envs.realworld.franka.tasks:DataCollectSFTEnv"


def test_data_collect_sft_config_keeps_explicit_runtime_values():
    DataCollectSFTConfig, _ = _import_data_collect_sft()

    config = DataCollectSFTConfig(
        task_description="collect demonstrations for a generic task",
        target_ee_pose=[0.5, 0.0, 0.2, 3.14, 0.0, 0.0],
        reset_ee_pose=[0.5, 0.0, 0.3, 3.14, 0.0, 0.0],
        ee_pose_limit_min=[0.4, -0.1, 0.1, 2.8, -0.1, -0.2],
        ee_pose_limit_max=[0.6, 0.1, 0.4, 3.2, 0.1, 0.2],
        action_scale=[0.02, 0.1, 1.0],
        camera_type="zed",
        gripper_type="robotiq",
        gripper_connection="/dev/ttyUSB0",
        compliance_param={"translational_stiffness": 123},
        precision_param={"rotational_stiffness": 456},
        enable_gripper_penalty=False,
    )

    assert config.task_description == "collect demonstrations for a generic task"
    assert np.array_equal(
        config.reset_ee_pose,
        np.array([0.5, 0.0, 0.3, 3.14, 0.0, 0.0]),
    )
    assert np.array_equal(
        config.ee_pose_limit_min,
        np.array([0.4, -0.1, 0.1, 2.8, -0.1, -0.2]),
    )
    assert np.array_equal(
        config.ee_pose_limit_max,
        np.array([0.6, 0.1, 0.4, 3.2, 0.1, 0.2]),
    )
    assert np.array_equal(config.action_scale, np.array([0.02, 0.1, 1.0]))
    assert config.camera_type == "zed"
    assert config.gripper_type == "robotiq"
    assert config.gripper_connection == "/dev/ttyUSB0"
    assert config.compliance_param == {"translational_stiffness": 123}
    assert config.precision_param == {"rotational_stiffness": 456}
    assert config.enable_gripper_penalty is False


def test_data_collect_sft_config_uses_safe_defaults():
    DataCollectSFTConfig, _ = _import_data_collect_sft()

    config = DataCollectSFTConfig()
    np.testing.assert_allclose(
        config.target_ee_pose,
        np.array([0.5, 0.0, 0.1, -3.14, 0.0, 0.0]),
    )
    np.testing.assert_allclose(
        config.reset_ee_pose,
        np.array([0.5, 0.0, 0.2, -3.14, 0.0, 0.0]),
    )
    np.testing.assert_allclose(config.action_scale, np.array([0.02, 0.1, 1.0]))
    np.testing.assert_allclose(
        config.ee_pose_limit_min,
        np.array([0.45, -0.05, 0.1, -3.14 - np.pi / 6, -np.pi / 6, -np.pi / 6]),
    )
    np.testing.assert_allclose(
        config.ee_pose_limit_max,
        np.array([0.55, 0.05, 0.2, -3.14 + np.pi / 6, np.pi / 6, np.pi / 6]),
    )


def test_data_collect_sft_config_computes_default_reset_and_workspace():
    DataCollectSFTConfig, _ = _import_data_collect_sft()

    config = DataCollectSFTConfig(
        target_ee_pose=[0.53, -0.07, 0.2, 3.12, 0.19, 0.24],
        clip_x_range=0.05,
        clip_y_range=0.05,
        clip_z_range_low=0.0,
        clip_z_range_high=0.1,
        clip_rz_range=np.pi / 6,
    )

    assert np.array_equal(
        config.reset_ee_pose,
        np.array([0.53, -0.07, 0.3, 3.12, 0.19, 0.24]),
    )
    assert np.array_equal(config.action_scale, np.array([0.02, 0.1, 1.0]))
    np.testing.assert_allclose(
        config.ee_pose_limit_min,
        np.array(
            [
                0.48,
                -0.12,
                0.2,
                3.12 - np.pi / 6,
                0.19 - np.pi / 6,
                0.24 - np.pi / 6,
            ]
        ),
    )
    np.testing.assert_allclose(
        config.ee_pose_limit_max,
        np.array(
            [
                0.58,
                -0.02,
                0.3,
                3.12 + np.pi / 6,
                0.19 + np.pi / 6,
                0.24 + np.pi / 6,
            ]
        ),
    )


def test_data_collect_sft_env_uses_task_description_from_config():
    _, DataCollectSFTEnv = _import_data_collect_sft()

    env = DataCollectSFTEnv(
        override_cfg={
            "is_dummy": True,
            "camera_serials": ["dummy-camera"],
            "task_description": "collect a language-conditioned demonstration",
            "target_ee_pose": [0.5, 0.0, 0.2, 3.14, 0.0, 0.0],
        },
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )

    assert env.task_description == "collect a language-conditioned demonstration"
