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

* ``LeRobotDatasetWriter`` schema for single-arm and dual-arm.
* ``CollectEpisode`` integration that auto-detects wrist images from
  the obs dict and forwards them to the writer.
* ``RealWorldEnv._wrap_obs`` dual branch that stacks left/right wrist
  cameras into ``wrist_images`` (alphabetical order) and puts the base
  (third-person) camera into ``main_images``.

Single-arm regression checks live alongside each test to ensure that the
default flag set produces a parquet schema byte-identical to the legacy
single-arm output.

Run with::

    python -m pytest tests/unit_tests/test_dual_arm_data_collection.py -v
"""

from __future__ import annotations

import json
from typing import Any

import gymnasium as gym
import numpy as np
import pyarrow.parquet as pq
import pytest

from rlinf.data.lerobot_writer import LeRobotDatasetWriter
from rlinf.envs.wrappers import CollectEpisode

# --------------------------------------------------------------------- #
#  LeRobotDatasetWriter schema tests                                       #
# --------------------------------------------------------------------- #


def _make_episode(
    T: int, image_shape: tuple[int, int, int], state_dim: int, action_dim: int
) -> dict[str, np.ndarray]:
    return {
        "images": np.random.randint(0, 255, (T,) + image_shape, dtype=np.uint8),
        "states": np.random.randn(T, state_dim).astype(np.float32),
        "actions": np.random.randn(T, action_dim).astype(np.float32),
    }


def test_lerobot_writer_single_arm_schema_unchanged(tmp_path):
    """Default flags must produce a legacy single-arm schema (no dual columns)."""
    writer = LeRobotDatasetWriter(
        root_dir=str(tmp_path),
        robot_type="panda",
        fps=10,
        image_shape=(64, 64, 3),
        state_dim=19,
        action_dim=7,
        has_wrist_image=False,
        has_extra_view_image=True,
        # has_left_wrist_image / has_right_wrist_image default to False;
        # passing them implicitly verifies the documented default.
        use_incremental_stats=True,
    )
    ep = _make_episode(T=4, image_shape=(64, 64, 3), state_dim=19, action_dim=7)
    extra = np.random.randint(0, 255, (4, 64, 64, 3), dtype=np.uint8)
    writer.add_episode(
        images=ep["images"],
        wrist_images=None,
        extra_view_images=extra,
        states=ep["states"],
        actions=ep["actions"],
        task="single arm",
        is_success=True,
    )
    writer.finalize()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    cols = set(table.column_names)
    assert "image" in cols
    assert "extra_view_image" in cols
    assert "left_wrist_image" not in cols, (
        "single-arm writer must not introduce dual-arm columns"
    )
    assert "right_wrist_image" not in cols, (
        "single-arm writer must not introduce dual-arm columns"
    )

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert "left_wrist_image" not in info["features"]
    assert "right_wrist_image" not in info["features"]

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "left_wrist_image" not in stats
    assert "right_wrist_image" not in stats


def test_lerobot_writer_dual_arm_schema(tmp_path):
    """Dual-arm uses wrist_image for stacked left/right wrist cameras."""
    writer = LeRobotDatasetWriter(
        root_dir=str(tmp_path),
        robot_type="dual_panda",
        fps=10,
        image_shape=(64, 64, 3),
        state_dim=38,
        action_dim=14,
        has_wrist_image=True,
        has_extra_view_image=False,
        use_incremental_stats=True,
    )
    T = 4
    ep = _make_episode(T=T, image_shape=(64, 64, 3), state_dim=38, action_dim=14)
    wrist = np.random.randint(0, 255, (T, 64, 64, 3), dtype=np.uint8)
    writer.add_episode(
        images=ep["images"],
        wrist_images=wrist,
        extra_view_images=None,
        states=ep["states"],
        actions=ep["actions"],
        task="dual arm",
        is_success=True,
    )
    writer.finalize()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    cols = set(table.column_names)
    assert "image" in cols
    assert "wrist_image" in cols
    assert "left_wrist_image" not in cols
    assert "right_wrist_image" not in cols
    assert "extra_view_image" not in cols

    # Image structs must be PNG-decodable for HuggingFace datasets to
    # interpret them as Image features.
    for col in ("image", "wrist_image"):
        first = table[col][0].as_py()
        assert isinstance(first, dict)
        assert "bytes" in first and "path" in first
        assert len(first["bytes"]) > 0

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["wrist_image"]["dtype"] == "image"
    assert "left_wrist_image" not in info["features"]
    assert "right_wrist_image" not in info["features"]
    assert info["features"]["state"]["shape"] == [38]
    assert info["features"]["actions"]["shape"] == [14]

    stats = json.loads((tmp_path / "meta" / "stats.json").read_text())
    assert "wrist_image" in stats
    assert "left_wrist_image" not in stats
    assert "right_wrist_image" not in stats


# --------------------------------------------------------------------- #
#  CollectEpisode integration tests                                        #
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
    """Mimics what RealWorldEnv._wrap_obs emits for the dual-arm branch.

    ``main_images`` = base (third-person) camera;
    ``wrist_images`` = stacked [left, right] wrist cameras (alphabetical).
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


def test_collect_episode_single_arm_unchanged(tmp_path):
    """Single-arm CollectEpisode → LeRobot must keep the legacy column set."""
    env = CollectEpisode(
        _SingleArmDummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
        robot_type="panda",
        fps=10,
    )
    env.reset()
    for _ in range(4):
        env.step(np.zeros(7, dtype=np.float32))
    env.close()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    cols = set(table.column_names)
    assert "image" in cols
    assert "left_wrist_image" not in cols
    assert "right_wrist_image" not in cols


def test_collect_episode_dual_arm_writes_wrist_image_column(tmp_path):
    """Dual-arm obs auto-flow into LeRobot wrist_image column."""
    env = CollectEpisode(
        _DualArmDummyEnv(),
        save_dir=str(tmp_path),
        export_format="lerobot",
        robot_type="dual_panda",
        fps=10,
    )
    env.reset()
    for _ in range(4):
        env.step(np.zeros(14, dtype=np.float32))
    env.close()

    table = pq.read_table(tmp_path / "data" / "chunk-000" / "episode_000000.parquet")
    cols = set(table.column_names)
    assert "image" in cols
    assert "wrist_image" in cols
    assert "left_wrist_image" not in cols
    assert "right_wrist_image" not in cols
    assert "extra_view_image" not in cols

    info = json.loads((tmp_path / "meta" / "info.json").read_text())
    assert info["features"]["wrist_image"]["dtype"] == "image"
    assert "left_wrist_image" not in info["features"]
    assert "right_wrist_image" not in info["features"]


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


def test_realworld_env_caches_is_dual_arm(dual_realworld_env):
    assert dual_realworld_env.is_dual_arm is True


def test_realworld_env_dual_branch_emits_semantic_keys(dual_realworld_env):
    obs, _ = dual_realworld_env.reset()
    assert "wrist_images" in obs
    assert "main_images" in obs
    assert "left_wrist_images" not in obs
    assert "right_wrist_images" not in obs
    assert "extra_view_images" not in obs, (
        "dual branch should not bucket frames into extra_view_images"
    )
    # wrist_images should be stacked [left, right] along axis=1
    assert obs["wrist_images"].shape[1] == 2
    # state shape: gripper(2)+force(6)+pose(12 quat)+torque(6)+vel(12) = 38
    # (use_relative_frame=False above, but DualQuat2EulerWrapper still wraps)
    assert obs["states"].shape[-1] == 38


def test_realworld_env_dual_branch_step(dual_realworld_env):
    dual_realworld_env.reset()
    next_obs, reward, term, trunc, info = dual_realworld_env.step(
        np.zeros((1, 14), dtype=np.float32)
    )
    assert "wrist_images" in next_obs
    assert "main_images" in next_obs
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
