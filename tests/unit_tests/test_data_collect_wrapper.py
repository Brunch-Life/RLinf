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

"""Unit tests for the real-world data collection wrapper.

These tests exercise the interactive recording logic (key-based start/stop,
success/failure marking) and keyboard reward-done wrappers WITHOUT any real
hardware.  All environment interaction is replaced by a lightweight mock
gym.Env and all keyboard input is injected via mock ``KeyboardListener``.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Lightweight mock environment
# ---------------------------------------------------------------------------

class _MockEnv(gym.Env):
    """Minimal gym.Env that returns deterministic data for testing."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(0, 255, (64, 64, 3), dtype=np.uint8),
                "states": gym.spaces.Box(-1, 1, (7,), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)

    def _obs(self):
        return {
            "main_images": np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            "states": np.zeros(7, dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.0, False, False, {}


class _BatchedMockEnv(gym.Env):
    """Mock env that returns batched (num_envs=1) tensors like RealWorldEnv."""

    metadata = {"render_modes": []}

    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Dict(
            {
                "main_images": gym.spaces.Box(0, 255, (1, 64, 64, 3), dtype=np.uint8),
                "states": gym.spaces.Box(-1, 1, (1, 7), dtype=np.float32),
            }
        )
        self.action_space = gym.spaces.Box(-1, 1, (1, 7), dtype=np.float32)

    def _obs(self):
        return {
            "main_images": np.random.randint(0, 255, (1, 64, 64, 3), dtype=np.uint8),
            "states": np.zeros((1, 7), dtype=np.float32),
        }

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return (
            self._obs(),
            torch.tensor([0.0]),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )


# ---------------------------------------------------------------------------
# Fake KeyboardListener that can be driven by tests
# ---------------------------------------------------------------------------

class _FakeKeyboardListener:
    """Drop-in replacement for KeyboardListener with injectable key state."""

    def __init__(self):
        self._key: str | None = None

    def set_key(self, key: str | None) -> None:
        self._key = key

    def get_key(self) -> str | None:
        return self._key


# ===========================================================================
# Tests: KeyboardRewardDoneWrapper
# ===========================================================================

class TestKeyboardRewardDoneWrapper:
    """Test keyboard-based reward and termination mapping."""

    def _make_wrapper(self, fake_listener: _FakeKeyboardListener):
        from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
            KeyboardRewardDoneWrapper,
        )

        with patch(
            "rlinf.envs.realworld.common.wrappers.reward_done_wrapper.KeyboardListener",
            return_value=fake_listener,
        ):
            wrapper = KeyboardRewardDoneWrapper(_MockEnv())
        return wrapper

    def test_key_a_marks_failure(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key("a")
        intervened, done, reward = wrapper._check_keypress()
        assert intervened is True
        assert done is True
        assert reward == -1

    def test_key_b_neutral(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key("b")
        intervened, done, reward = wrapper._check_keypress()
        assert intervened is True
        assert done is False
        assert reward == 0

    def test_key_c_marks_success(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key("c")
        intervened, done, reward = wrapper._check_keypress()
        assert intervened is True
        assert done is True
        assert reward == 1

    def test_no_key_pressed(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key(None)
        intervened, done, reward = wrapper._check_keypress()
        assert intervened is False
        assert done is False
        assert reward == 0

    def test_step_replaces_reward_when_always_replace(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key("a")
        _, reward, terminated, _, _ = wrapper.step(np.zeros(7))
        assert reward == -1
        assert terminated is True


# ===========================================================================
# Tests: KeyboardRewardDoneMultiStageWrapper
# ===========================================================================

class TestKeyboardRewardDoneMultiStageWrapper:
    """Test multi-stage keyboard reward logic."""

    def _make_wrapper(self, fake_listener: _FakeKeyboardListener):
        from rlinf.envs.realworld.common.wrappers.reward_done_wrapper import (
            KeyboardRewardDoneMultiStageWrapper,
        )

        with patch(
            "rlinf.envs.realworld.common.wrappers.reward_done_wrapper.KeyboardListener",
            return_value=fake_listener,
        ):
            wrapper = KeyboardRewardDoneMultiStageWrapper(_MockEnv())
        wrapper.reward_stage = 0
        return wrapper

    def test_stage_transitions(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)

        listener.set_key("a")
        _, done, reward = wrapper._check_keypress()
        assert wrapper.reward_stage == 0
        assert reward == 0
        assert done is False

        listener.set_key("b")
        _, done, reward = wrapper._check_keypress()
        assert wrapper.reward_stage == 1
        assert reward == pytest.approx(0.1)
        assert done is False

        listener.set_key("c")
        _, done, reward = wrapper._check_keypress()
        assert wrapper.reward_stage == 2
        assert reward == 1
        assert done is True

    def test_key_q_gives_negative_reward(self):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener)
        listener.set_key("q")
        _, done, reward = wrapper._check_keypress()
        assert reward == -1
        assert done is False


# ===========================================================================
# Tests: RealWorldCollectEpisode
# ===========================================================================

class TestRealWorldCollectEpisode:
    """Test interactive recording logic without real hardware."""

    def _make_wrapper(self, fake_listener: _FakeKeyboardListener, tmp_path):
        from examples.embodiment.collect_real_data_with_wrapper import (
            RealWorldCollectEpisode,
        )

        with patch(
            "rlinf.envs.realworld.common.keyboard.keyboard_listener.KeyboardListener",
            return_value=fake_listener,
        ):
            wrapper = RealWorldCollectEpisode(
                env=_BatchedMockEnv(),
                save_dir=str(tmp_path),
                num_envs=1,
                export_format="pickle",
                show_goal_site=False,
            )
        return wrapper

    def test_inject_success_with_episode_dict(self):
        from examples.embodiment.collect_real_data_with_wrapper import (
            RealWorldCollectEpisode,
        )

        info = {"episode": {"some_key": 1}}
        RealWorldCollectEpisode._inject_success(info, True)
        assert "success_once" in info["episode"]
        assert info["episode"]["success_once"].item() is True

    def test_inject_success_without_episode(self):
        from examples.embodiment.collect_real_data_with_wrapper import (
            RealWorldCollectEpisode,
        )

        info = {}
        RealWorldCollectEpisode._inject_success(info, False)
        assert "success" in info
        assert info["success"].item() is False

    def test_inject_success_non_dict_episode(self):
        from examples.embodiment.collect_real_data_with_wrapper import (
            RealWorldCollectEpisode,
        )

        info = {"episode": "not_a_dict"}
        RealWorldCollectEpisode._inject_success(info, True)
        assert "success" in info
        assert info["success"].item() is True

    def test_recording_not_started_until_key_a(self, tmp_path):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))
        wrapper.step(np.zeros((1, 7)))
        for buf in wrapper._buffers:
            assert len(buf["actions"]) == 0

    def test_key_a_starts_recording(self, tmp_path):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key("a")
        wrapper.step(np.zeros((1, 7)))
        assert wrapper._recording is True

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))
        assert len(wrapper._buffers[0]["actions"]) == 1

    def test_key_b_ends_as_failure(self, tmp_path):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key("a")
        wrapper.step(np.zeros((1, 7)))

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))

        listener.set_key("b")
        _, reward, terminated, _, _ = wrapper.step(np.zeros((1, 7)))
        assert bool(terminated.any()) is True
        assert float(reward[0]) == -1.0
        assert wrapper.last_episode_was_recorded is True

    def test_key_c_ends_as_success(self, tmp_path):
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key("a")
        wrapper.step(np.zeros((1, 7)))

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))

        listener.set_key("c")
        _, reward, terminated, _, _ = wrapper.step(np.zeros((1, 7)))
        assert bool(terminated.any()) is True
        assert float(reward[0]) == 1.0
        assert wrapper.last_episode_was_recorded is True

    def test_unrecorded_episode_flag(self, tmp_path):
        """Episode that ends without pressing 'a' is not counted as recorded."""
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key("b")
        wrapper.step(np.zeros((1, 7)))
        assert wrapper.last_episode_was_recorded is False

    def test_intervene_action_recorded(self, tmp_path):
        """When info contains 'intervene_action', it should be recorded instead."""
        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)

        real_action = np.ones((1, 7))
        env_step_orig = wrapper.env.step

        def _step_with_intervene(action):
            obs, reward, term, trunc, info = env_step_orig(action)
            info["intervene_action"] = real_action
            return obs, reward, term, trunc, info

        wrapper.env.step = _step_with_intervene
        wrapper.reset()

        listener.set_key("a")
        wrapper.step(np.zeros((1, 7)))

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))

        recorded = np.asarray(wrapper._buffers[0]["actions"][0])
        np.testing.assert_array_equal(recorded, real_action.squeeze(0))

    def test_pickle_export_on_success(self, tmp_path):
        """A successful recorded episode should produce a .pkl file."""
        import os

        listener = _FakeKeyboardListener()
        wrapper = self._make_wrapper(listener, tmp_path)
        wrapper.reset()

        listener.set_key("a")
        wrapper.step(np.zeros((1, 7)))

        listener.set_key(None)
        wrapper.step(np.zeros((1, 7)))

        listener.set_key("c")
        wrapper.step(np.zeros((1, 7)))

        wrapper.close()

        pkl_files = [f for f in os.listdir(tmp_path) if f.endswith(".pkl")]
        assert len(pkl_files) == 1
        assert "success" in pkl_files[0]
