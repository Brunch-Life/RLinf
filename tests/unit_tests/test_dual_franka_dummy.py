"""Smoke test for DualFrankaEnv in dummy mode.

Run in an environment with all deps installed (e.g. .venv):
    python -m pytest tests/unit_tests/test_dual_franka_dummy.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from rlinf.envs.realworld.franka.dual_franka_env import (
    DualFrankaEnv,
    DualFrankaRobotConfig,
)


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
    assert "wrist_1" in dummy_env.observation_space["frames"].spaces
    assert "wrist_2" in dummy_env.observation_space["frames"].spaces


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
    pose = np.concatenate([dummy_env._left_state.tcp_pose, dummy_env._right_state.tcp_pose])
    assert pose.shape == (14,)
