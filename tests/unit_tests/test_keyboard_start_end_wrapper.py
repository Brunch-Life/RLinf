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

"""Regression tests for KeyboardStartEndWrapper.

Run with::

    python -m pytest tests/unit_tests/test_keyboard_start_end_wrapper.py -v
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
import pytest


class _StubKeyboardListener:
    """Drop-in for KeyboardListener that replays a scripted key sequence."""

    def __init__(self, scripted_keys: list[list[str]] | None = None):
        self._queue: deque[list[str]] = deque(scripted_keys or [])

    def pop_pressed_keys(self) -> list[str]:
        return list(self._queue.popleft()) if self._queue else []


class _StubInnerEnv(gym.Env):
    """Minimal env that yields zero reward and never terminates on its own."""

    observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(2, dtype=np.float32), 0.0, False, False, {}


@pytest.fixture
def make_wrapper(monkeypatch):
    """Build a KeyboardStartEndWrapper whose listener replays scripted keys."""
    from rlinf.envs.realworld.common.wrappers import keyboard_start_end_wrapper as mod

    def _factory(scripted_keys: list[list[str]]):
        listener = _StubKeyboardListener(scripted_keys)
        monkeypatch.setattr(mod, "KeyboardListener", lambda: listener)
        return mod.KeyboardStartEndWrapper(_StubInnerEnv())

    return _factory


def test_press_c_keeps_pre_record_false_so_collectepisode_records_terminating_frame(
    make_wrapper,
):
    """Regression: previously self._recording flipped to False on the 'c'
    press, which made the terminating frame ``pre_record=True``.
    CollectEpisode then skipped that frame in ``_record_step`` (and the
    ``info["episode"]["success_once"]=True`` signal RealWorldEnv emits on
    that step), leaving ``_episode_success`` stuck at False; with
    ``only_success: True`` the entire episode was dropped from the
    LeRobot writer. The fix keeps recording=True for this terminating
    frame; reset() clears it before the next episode.
    """
    env = make_wrapper(
        [
            [],  # consumed by reset()'s drain
            ["a"],  # start a recording episode
            [],  # mid-episode no-op
            ["c"],  # success-end: pedal C
        ]
    )
    env.reset()

    # 1) press 'a' → start
    _, _, term, trunc, info_a = env.step(np.zeros(2, dtype=np.float32))
    assert info_a["keyboard_event"] == "start"
    assert info_a["keyboard_phase"] == "rec"
    assert info_a["pre_record"] is False
    assert term is False and trunc is False

    # 2) plain step inside the recording window
    _, _, _, _, info_mid = env.step(np.zeros(2, dtype=np.float32))
    assert info_mid["keyboard_phase"] == "rec"
    assert info_mid["pre_record"] is False

    # 3) press 'c' → THIS frame is the terminating frame: it must remain
    # recordable so CollectEpisode picks up the reward=1.0 success signal.
    _, reward_c, term_c, trunc_c, info_c = env.step(np.zeros(2, dtype=np.float32))
    assert info_c["keyboard_event"] == "end_success"
    assert reward_c == 1.0
    assert term_c is True and trunc_c is False
    # The actual regression assertions:
    assert info_c["keyboard_phase"] == "rec", (
        "terminating frame must stay phase=rec so CollectEpisode records it"
    )
    assert info_c["pre_record"] is False, (
        "pre_record on the terminating frame would make CollectEpisode "
        "skip the reward=1.0 step and only_success would drop the episode"
    )


def test_reset_clears_recording_after_c_press(make_wrapper):
    """reset() must flip _recording back to False so the next episode
    starts in pre, even though the 'c' branch no longer does it."""
    env = make_wrapper([[], ["a"], ["c"]])  # leading [] for reset drain
    env.reset()
    env.step(np.zeros(2, dtype=np.float32))  # 'a' → rec
    env.step(np.zeros(2, dtype=np.float32))  # 'c' → terminating frame, still rec
    assert env._recording is True
    env.reset()
    assert env._recording is False


def test_press_a_during_rec_aborts_and_resets(make_wrapper):
    """Pressing 'a' mid-rec aborts: phase flips to pre and record_reset=True."""
    env = make_wrapper([[], ["a"], ["a"]])  # leading [] for reset drain
    env.reset()
    env.step(np.zeros(2, dtype=np.float32))  # start
    _, _, term, trunc, info = env.step(np.zeros(2, dtype=np.float32))  # abort
    assert info["keyboard_event"] == "abort"
    assert info["keyboard_phase"] == "pre"
    assert info["pre_record"] is True
    assert info["record_reset"] is True
    assert term is False and trunc is False
