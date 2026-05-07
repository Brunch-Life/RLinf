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

"""Unit tests for DualFrankaEnv contracts that would silently break policy
inference or wrappers if violated."""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.dual_euler_obs import DualQuat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.dual_relative_frame import DualRelativeFrame
from rlinf.envs.realworld.franka.dual_franka_env import DualFrankaEnv


@pytest.fixture()
def dummy_env():
    cfg = {
        "is_dummy": True,
        "left_robot_ip": "0.0.0.0",
        "right_robot_ip": "0.0.0.0",
        "left_camera_serials": ["DUMMY_L"],
        "right_camera_serials": ["DUMMY_R"],
        "camera_type": "zed",
        "target_ee_pose": [[0.5, 0, 0.1, -3.14, 0, 0], [0.5, 0, 0.1, -3.14, 0, 0]],
        "reset_ee_pose": [[0.5, 0, 0.3, -3.14, 0, 0], [0.5, 0, 0.3, -3.14, 0, 0]],
        "reward_threshold": [[0.01, 0.01, 0.01, 0.2, 0.2, 0.2]] * 2,
        "action_scale": [1.0, 1.0, 1.0],
        "ee_pose_limit_min": [[-1] * 6, [-1] * 6],
        "ee_pose_limit_max": [[1] * 6, [1] * 6],
        "max_num_steps": 50,
    }
    return DualFrankaEnv(
        override_cfg=cfg, worker_info=None, hardware_info=None, env_idx=0
    )


def test_observation_space_layout(dummy_env):
    """obs schema: tcp_pose 14 (xyz+quat per arm), tcp_vel 12, gripper 2."""
    state_space = dummy_env.observation_space["state"]
    assert state_space["tcp_pose"].shape == (14,)
    assert state_space["tcp_vel"].shape == (12,)
    assert state_space["gripper_position"].shape == (2,)


def test_camera_keys_match_pi05(dummy_env):
    """Cameras are keyed as pi0/pi05 expects, so obs feeds policy with no remap."""
    frames = dummy_env.observation_space["frames"].spaces
    assert "left_wrist_0_rgb" in frames
    assert "right_wrist_0_rgb" in frames


def test_truncation_after_max_steps(dummy_env):
    """Episode truncates at max_num_steps (50)."""
    dummy_env.reset()
    truncated = False
    for _ in range(50):
        _, _, _, truncated, _ = dummy_env.step(dummy_env.action_space.sample())
    assert truncated


def test_quat2euler_wrapper_produces_valid_euler(dummy_env):
    """tcp_pose drops from 14 (xyz+quat) → 12 (xyz+euler) and stays in [-π, π]."""
    wrapped = DualQuat2EulerWrapper(dummy_env)
    obs, _ = wrapped.reset()
    tcp = obs["state"]["tcp_pose"]
    assert tcp.shape == (12,)
    for euler in (tcp[3:6], tcp[9:12]):
        assert np.all(np.abs(euler) <= np.pi + 1e-2)


def test_relative_frame_action_transform_round_trip(dummy_env):
    """transform_action and its inverse should compose to identity."""
    wrapped = DualRelativeFrame(dummy_env)
    wrapped.reset()
    action = np.random.randn(14).astype(np.float32)
    recovered = wrapped.transform_action_inv(wrapped.transform_action(action.copy()))
    np.testing.assert_allclose(recovered, action, atol=1e-6)
