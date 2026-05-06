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

"""Unit tests for the dual-arm data collection pipeline.

These tests cover the schema and data flow added to support dual-arm
Franka environments end-to-end:

* ``CollectEpisode._expand_multi_view_images`` fan-out of multi-view
  wrist cameras into per-view frame keys (``wrist_image-0``,
  ``wrist_image-1``).
* ``CollectEpisode._collect_image_keys`` schema detection for the
  ``LeRobotDatasetWriter.create()`` call.
* ``CollectEpisode`` integration that auto-detects wrist images from
  the obs dict and forwards them to the writer as correctly keyed
  per-step frame dicts.
* ``RealWorldEnv._wrap_obs`` that puts the frame named by
  ``main_image_key`` into ``main_images`` and stacks the remaining
  camera frames (alphabetical) into ``extra_view_images`` — the same
  code path for single- and dual-arm envs.

Single-arm regression checks live alongside each test to ensure that the
default path produces frame dicts identical to the legacy single-arm
output.

Run with::

    python -m pytest tests/unit_tests/test_dual_arm_data_collection.py -v
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.wrappers import CollectEpisode

# --------------------------------------------------------------------- #
#  _expand_multi_view_images unit tests                                    #
# --------------------------------------------------------------------- #


def test_expand_multi_view_images_none():
    """None input returns an empty dict."""
    result = CollectEpisode._expand_multi_view_images("wrist_image", None)
    assert result == {}


def test_expand_multi_view_images_single_view_3d():
    """A single [H, W, C] image maps to {base_key: img}."""
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    result = CollectEpisode._expand_multi_view_images("wrist_image", img)
    assert set(result.keys()) == {"wrist_image"}
    assert result["wrist_image"].shape == (64, 64, 3)


def test_expand_multi_view_images_single_view_4d():
    """A single [1, H, W, C] image collapses to {base_key: img[0]}."""
    img = np.zeros((1, 64, 64, 3), dtype=np.uint8)
    result = CollectEpisode._expand_multi_view_images("wrist_image", img)
    assert set(result.keys()) == {"wrist_image"}
    assert result["wrist_image"].shape == (64, 64, 3)


def test_expand_multi_view_images_dual_view():
    """[2, H, W, C] fans out to wrist_image-0, wrist_image-1."""
    left = np.full((64, 64, 3), 100, dtype=np.uint8)
    right = np.full((64, 64, 3), 200, dtype=np.uint8)
    img = np.stack([left, right], axis=0)
    result = CollectEpisode._expand_multi_view_images("wrist_image", img)
    assert set(result.keys()) == {"wrist_image-0", "wrist_image-1"}
    assert result["wrist_image-0"].shape == (64, 64, 3)
    assert result["wrist_image-1"].shape == (64, 64, 3)
    np.testing.assert_array_equal(result["wrist_image-0"], left)
    np.testing.assert_array_equal(result["wrist_image-1"], right)


def test_expand_multi_view_images_triple_view():
    """[3, H, W, C] fans out to three indexed keys."""
    img = np.zeros((3, 64, 64, 3), dtype=np.uint8)
    result = CollectEpisode._expand_multi_view_images("extra_view_image", img)
    assert set(result.keys()) == {
        "extra_view_image-0",
        "extra_view_image-1",
        "extra_view_image-2",
    }


# --------------------------------------------------------------------- #
#  _collect_image_keys unit tests                                          #
# --------------------------------------------------------------------- #


def test_collect_image_keys_single_wrist():
    """Single wrist view produces {wrist_image: shape}."""
    frame = {
        "wrist_image": np.zeros((64, 64, 3), dtype=np.uint8),
        "state": np.zeros(19, dtype=np.float32),
    }
    result = CollectEpisode._collect_image_keys(frame, "wrist_image")
    assert result == {"wrist_image": (64, 64, 3)}


def test_collect_image_keys_dual_wrist():
    """Dual wrist views produce {wrist_image-0: shape, wrist_image-1: shape}."""
    frame = {
        "wrist_image-0": np.zeros((64, 64, 3), dtype=np.uint8),
        "wrist_image-1": np.zeros((64, 64, 3), dtype=np.uint8),
        "state": np.zeros(38, dtype=np.float32),
    }
    result = CollectEpisode._collect_image_keys(frame, "wrist_image")
    assert result == {
        "wrist_image-0": (64, 64, 3),
        "wrist_image-1": (64, 64, 3),
    }


def test_collect_image_keys_no_match():
    """No matching keys returns an empty dict."""
    frame = {
        "image": np.zeros((64, 64, 3), dtype=np.uint8),
        "state": np.zeros(19, dtype=np.float32),
    }
    result = CollectEpisode._collect_image_keys(frame, "wrist_image")
    assert result == {}


def test_collect_image_keys_ignores_non_3d():
    """4-D arrays and non-ndarray values are excluded."""
    frame = {
        "wrist_image-0": np.zeros((64, 64, 3), dtype=np.uint8),
        "wrist_image-1": np.zeros((2, 64, 64, 3), dtype=np.uint8),  # 4-D
        "wrist_image-2": "not an array",
    }
    result = CollectEpisode._collect_image_keys(frame, "wrist_image")
    assert result == {"wrist_image-0": (64, 64, 3)}


# --------------------------------------------------------------------- #
#  Dummy environments                                                      #
# --------------------------------------------------------------------- #


class _SingleArmDummyEnv(gym.Env):
    """Mimics what RealWorldEnv._wrap_obs emits for single-arm Franka."""

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(7,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(-1.0, 1.0, shape=(19,), dtype=np.float32),
                "main_images": gym.spaces.Box(
                    0, 255, shape=(64, 64, 3), dtype=np.uint8
                ),
            }
        )
        self._t = 0

    def _obs(self) -> dict[str, Any]:
        return {
            "states": np.zeros(19, dtype=np.float32),
            "main_images": np.full((64, 64, 3), self._t, dtype=np.uint8),
            "task_descriptions": "single task",
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= 4
        return (
            self._obs(),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"success_once": terminated},
        )


class _DualArmDummyEnv(gym.Env):
    """Mimics a dual-arm obs dict with a stacked ``wrist_images`` tensor.

    This shape is what sim envs (libero, roboverse, …) emit and what the
    LeRobot CollectEpisode pipeline is expected to fan out — it is *not*
    the shape RealWorldEnv produces for a dual-arm Franka: realworld
    stacks secondary views into ``extra_view_images`` instead.
    """

    def __init__(self) -> None:
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(14,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(-1.0, 1.0, shape=(38,), dtype=np.float32),
                "main_images": gym.spaces.Box(
                    0, 255, shape=(64, 64, 3), dtype=np.uint8
                ),
                "wrist_images": gym.spaces.Box(
                    0, 255, shape=(2, 64, 64, 3), dtype=np.uint8
                ),
            }
        )
        self._t = 0

    def _obs(self) -> dict[str, Any]:
        base = np.full((64, 64, 3), self._t + 50, dtype=np.uint8)
        left = np.full((64, 64, 3), self._t + 100, dtype=np.uint8)
        right = np.full((64, 64, 3), self._t + 200, dtype=np.uint8)
        return {
            "states": np.zeros(38, dtype=np.float32),
            "main_images": base,
            "wrist_images": np.stack([left, right], axis=0),
            "task_descriptions": "dual task",
        }

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= 4
        return (
            self._obs(),
            1.0 if terminated else 0.0,
            terminated,
            False,
            {"success_once": terminated},
        )


# --------------------------------------------------------------------- #
#  CollectEpisode frame-format tests (no lerobot dependency)               #
# --------------------------------------------------------------------- #


def test_collect_episode_single_arm_frame_format(tmp_path):
    """Single-arm obs produce frames with ``image`` only (no wrist keys)."""
    captured: list[list[dict]] = []

    env = CollectEpisode(
        _SingleArmDummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
        robot_type="panda",
        fps=10,
    )
    env._write_lerobot_episode = lambda ep: captured.append(ep)

    env.reset()
    for _ in range(4):
        env.step(np.zeros(7, dtype=np.float32))
    env._wait_futures()

    assert len(captured) == 1
    frames = captured[0]
    assert len(frames) == 4

    first = frames[0]
    assert "image" in first
    assert "state" in first
    assert "actions" in first
    assert "task" in first
    assert "done" in first
    assert "is_success" in first
    assert "wrist_image" not in first
    assert "wrist_image-0" not in first
    assert "wrist_image-1" not in first
    assert "left_wrist_image" not in first
    assert "right_wrist_image" not in first
    assert "extra_view_image" not in first

    assert frames[-1]["done"].item() is True
    assert first["state"].shape == (19,)
    assert first["actions"].shape == (7,)

    env.close()


def test_collect_episode_dual_arm_frame_format(tmp_path):
    """Dual-arm obs produce ``wrist_image-0`` and ``wrist_image-1`` frame keys."""
    captured: list[list[dict]] = []

    env = CollectEpisode(
        _DualArmDummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
        robot_type="dual_FR3",
        fps=10,
    )
    env._write_lerobot_episode = lambda ep: captured.append(ep)

    env.reset()
    for _ in range(4):
        env.step(np.zeros(14, dtype=np.float32))
    env._wait_futures()

    assert len(captured) == 1
    frames = captured[0]
    assert len(frames) == 4

    first = frames[0]
    assert "image" in first, "base camera should map to 'image'"
    assert "wrist_image-0" in first, "left wrist should map to 'wrist_image-0'"
    assert "wrist_image-1" in first, "right wrist should map to 'wrist_image-1'"
    assert "wrist_image" not in first, "multi-view should fan out, not use bare key"
    assert "left_wrist_image" not in first
    assert "right_wrist_image" not in first
    assert "extra_view_image" not in first

    assert first["image"].shape == (64, 64, 3)
    assert first["wrist_image-0"].shape == (64, 64, 3)
    assert first["wrist_image-1"].shape == (64, 64, 3)
    assert first["state"].shape == (38,)
    assert first["actions"].shape == (14,)

    wrist_keys = CollectEpisode._collect_image_keys(first, "wrist_image")
    assert set(wrist_keys.keys()) == {"wrist_image-0", "wrist_image-1"}
    for shape in wrist_keys.values():
        assert shape == (64, 64, 3)

    assert frames[-1]["done"].item() is True

    env.close()


def test_collect_episode_dual_arm_image_content(tmp_path):
    """Verify left/right wrist pixel values survive the pipeline."""
    captured: list[list[dict]] = []

    env = CollectEpisode(
        _DualArmDummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
        robot_type="dual_FR3",
        fps=10,
    )
    env._write_lerobot_episode = lambda ep: captured.append(ep)

    env.reset()
    for _ in range(4):
        env.step(np.zeros(14, dtype=np.float32))
    env._wait_futures()

    first = captured[0][0]
    np.testing.assert_array_equal(
        first["image"], np.full((64, 64, 3), 50, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        first["wrist_image-0"], np.full((64, 64, 3), 100, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        first["wrist_image-1"], np.full((64, 64, 3), 200, dtype=np.uint8)
    )

    env.close()


# --------------------------------------------------------------------- #
#  RealWorldEnv._wrap_obs dual branch (uses dummy DualFrankaEnv)           #
# --------------------------------------------------------------------- #


@pytest.fixture()
def dual_realworld_env():
    from omegaconf import OmegaConf

    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    cfg = OmegaConf.create(
        {
            "env_type": "realworld",
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_mode": "raw",
            "wrap_obs_mode": "simple",
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": 100,
            "max_episode_steps": 100,
            "use_spacemouse": False,
            "no_gripper": False,
            "main_image_key": "left_wrist_0_rgb",
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/x",
            },
            "init_params": {"id": "DualFrankaEnv-v1", "num_envs": None},
            "override_cfg": {
                "is_dummy": True,
                "left_robot_ip": "0.0.0.0",
                "right_robot_ip": "0.0.0.0",
                "left_camera_serials": ["DUMMY_L"],
                "right_camera_serials": ["DUMMY_R"],
                "camera_type": "zed",
                "task_description": "dual",
                "target_ee_pose": [
                    [0.5, 0, 0.1, -3.14, 0, 0],
                    [0.5, 0, 0.1, -3.14, 0, 0],
                ],
                "reset_ee_pose": [
                    [0.5, 0, 0.3, -3.14, 0, 0],
                    [0.5, 0, 0.3, -3.14, 0, 0],
                ],
                "reward_threshold": [[0.01] * 6, [0.01] * 6],
                "action_scale": [1.0, 1.0, 1.0],
                "ee_pose_limit_min": [[-1.0] * 6, [-1.0] * 6],
                "ee_pose_limit_max": [[1.0] * 6, [1.0] * 6],
                "max_num_steps": 50,
            },
            "use_relative_frame": False,
        }
    )
    env = RealWorldEnv(
        cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    yield env
    env.close()


def test_realworld_env_dual_branch_emits_semantic_keys(dual_realworld_env):
    obs, _ = dual_realworld_env.reset()
    assert "main_images" in obs
    # Dual-arm uses the same path as single-arm: main_image_key picks the
    # primary frame, remaining frames stack into extra_view_images.
    assert "extra_view_images" in obs
    assert "wrist_images" not in obs
    assert "left_wrist_images" not in obs
    assert "right_wrist_images" not in obs
    # state shape: gripper(2)+force(6)+pose(12 quat)+torque(6)+vel(12) = 38
    # (use_relative_frame=False above, but DualQuat2EulerWrapper still wraps)
    assert obs["states"].shape[-1] == 38


def test_realworld_env_dual_branch_step(dual_realworld_env):
    dual_realworld_env.reset()
    next_obs, reward, term, trunc, info = dual_realworld_env.step(
        np.zeros((1, 14), dtype=np.float32)
    )
    assert "main_images" in next_obs
    assert "extra_view_images" in next_obs
    assert "wrist_images" not in next_obs
    assert "left_wrist_images" not in next_obs
    assert "right_wrist_images" not in next_obs


def test_realworld_env_dual_no_gripper_raises():
    """Dual + no_gripper=True should fail loudly until DualGripperCloseEnv lands."""
    from omegaconf import OmegaConf

    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    cfg = OmegaConf.create(
        {
            "env_type": "realworld",
            "total_num_envs": 1,
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_mode": "raw",
            "wrap_obs_mode": "simple",
            "seed": 0,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": 100,
            "max_episode_steps": 100,
            "no_gripper": True,  # the unsupported combo
            "main_image_key": "left_wrist_0_rgb",
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "/tmp/x",
            },
            "init_params": {"id": "DualFrankaEnv-v1", "num_envs": None},
            "override_cfg": {
                "is_dummy": True,
                "left_robot_ip": "0.0.0.0",
                "right_robot_ip": "0.0.0.0",
                "left_camera_serials": ["DUMMY_L"],
                "right_camera_serials": ["DUMMY_R"],
                "camera_type": "zed",
                "task_description": "dual",
                "target_ee_pose": [
                    [0.5, 0, 0.1, -3.14, 0, 0],
                    [0.5, 0, 0.1, -3.14, 0, 0],
                ],
                "reset_ee_pose": [
                    [0.5, 0, 0.3, -3.14, 0, 0],
                    [0.5, 0, 0.3, -3.14, 0, 0],
                ],
                "reward_threshold": [[0.01] * 6, [0.01] * 6],
                "action_scale": [1.0, 1.0, 1.0],
                "ee_pose_limit_min": [[-1.0] * 6, [-1.0] * 6],
                "ee_pose_limit_max": [[1.0] * 6, [1.0] * 6],
                "max_num_steps": 50,
            },
            "use_relative_frame": False,
        }
    )
    with pytest.raises(NotImplementedError, match="DualGripperCloseEnv"):
        RealWorldEnv(
            cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
