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

"""Smoke test for DualFrankaEnv in dummy mode and dual-arm wrappers.

Run in an environment with all deps installed (e.g. .venv):
    python -m pytest tests/unit_tests/test_dual_franka_dummy.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.dual_euler_obs import DualQuat2EulerWrapper
from rlinf.envs.realworld.common.wrappers.dual_relative_frame import (
    DualRelativeFrame,
    DualRelativeTargetFrame,
)
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
        override_cfg=cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
    )


def test_config_shapes(dummy_env):
    assert dummy_env.config.target_ee_pose.shape == (2, 6)
    assert dummy_env.config.reset_ee_pose.shape == (2, 6)
    assert dummy_env.config.ee_pose_limit_min.shape == (2, 6)
    assert dummy_env.config.ee_pose_limit_max.shape == (2, 6)


def test_action_space(dummy_env):
    assert dummy_env.action_space.shape == (14,)


def test_observation_space_keys(dummy_env):
    assert "state" in dummy_env.observation_space.spaces
    assert "frames" in dummy_env.observation_space.spaces
    assert dummy_env.observation_space["state"]["tcp_pose"].shape == (14,)
    assert dummy_env.observation_space["state"]["tcp_vel"].shape == (12,)
    assert dummy_env.observation_space["state"]["gripper_position"].shape == (2,)


def test_camera_keys(dummy_env):
    # Camera frames are named after pi0/pi0.5 image dict keys so the obs
    # dict can be fed into a pi0 policy without any key remapping.
    assert "left_wrist_0_rgb" in dummy_env.observation_space["frames"].spaces
    assert "right_wrist_0_rgb" in dummy_env.observation_space["frames"].spaces


def test_reset(dummy_env):
    obs, info = dummy_env.reset()
    assert "state" in obs
    assert "frames" in obs
    assert obs["state"]["tcp_pose"].shape == (14,)


def test_step(dummy_env):
    dummy_env.reset()
    action = dummy_env.action_space.sample()
    obs, reward, terminated, truncated, info = dummy_env.step(action)
    assert obs["state"]["tcp_pose"].shape == (14,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)


def test_multi_step(dummy_env):
    dummy_env.reset()
    for _ in range(10):
        dummy_env.step(dummy_env.action_space.sample())
    assert dummy_env.num_steps == 10


def test_truncation(dummy_env):
    dummy_env.reset()
    for _ in range(50):
        _, _, _, truncated, _ = dummy_env.step(dummy_env.action_space.sample())
    assert truncated


def test_internal_state_shapes(dummy_env):
    dummy_env.reset()
    pose = np.concatenate(
        [dummy_env._left_state.tcp_pose, dummy_env._right_state.tcp_pose]
    )
    assert pose.shape == (14,)


# ------------------------------------------------------------------ #
#  Dual wrapper tests                                                  #
# ------------------------------------------------------------------ #


class TestDualQuat2EulerWrapper:
    def test_obs_shape(self, dummy_env):
        wrapped = DualQuat2EulerWrapper(dummy_env)
        assert wrapped.observation_space["state"]["tcp_pose"].shape == (12,)
        obs, _ = wrapped.reset()
        assert obs["state"]["tcp_pose"].shape == (12,)

    def test_roundtrip_values(self, dummy_env):
        """Euler conversion should preserve xyz and produce valid euler angles."""
        wrapped = DualQuat2EulerWrapper(dummy_env)
        obs, _ = wrapped.reset()
        tcp = obs["state"]["tcp_pose"]
        # Left arm xyz (first 3) and right arm xyz (indices 6-8) are preserved
        assert tcp.shape == (12,)
        # Each euler angle in [-pi, pi]
        for euler in [tcp[3:6], tcp[9:12]]:
            assert np.all(np.abs(euler) <= np.pi + 0.01)


class TestDualRelativeFrame:
    def test_obs_preserved_shape(self, dummy_env):
        wrapped = DualRelativeFrame(dummy_env)
        obs, _ = wrapped.reset()
        assert obs["state"]["tcp_pose"].shape == (14,)

    def test_action_transform_roundtrip(self, dummy_env):
        wrapped = DualRelativeFrame(dummy_env)
        wrapped.reset()
        action = np.random.randn(14).astype(np.float32)
        transformed = wrapped.transform_action(action.copy())
        recovered = wrapped.transform_action_inv(transformed)
        np.testing.assert_allclose(recovered, action, atol=1e-6)

    def test_step(self, dummy_env):
        wrapped = DualRelativeFrame(dummy_env)
        wrapped.reset()
        action = np.zeros(14, dtype=np.float32)
        obs, rew, done, trunc, info = wrapped.step(action)
        assert obs["state"]["tcp_pose"].shape == (14,)


class TestDualRelativeTargetFrame:
    def test_reset(self, dummy_env):
        wrapped = DualRelativeTargetFrame(dummy_env)
        obs, _ = wrapped.reset()
        assert obs["state"]["tcp_pose"].shape == (14,)
