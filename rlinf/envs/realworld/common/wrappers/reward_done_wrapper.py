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

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class BaseKeyboardRewardDoneWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, reward_mode: str = "always_replace"):
        super().__init__(env)
        self.reward_modifier = 0
        self.listener = KeyboardListener()
        self.reward_mode = reward_mode
        assert self.reward_mode in ["always_replace"]

    def _check_keypress(self) -> tuple[bool, bool, float]:
        raise NotImplementedError

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        last_intervened, updated_reward, updated_terminated = self.reward_terminated()
        if last_intervened or self.reward_mode == "always_replace":
            reward = updated_reward
        return observation, reward, updated_terminated, truncated, info

    def reward_terminated(
        self,
    ) -> tuple[float, bool]:
        last_intervened, terminated, keyboard_reward = self._check_keypress()
        return last_intervened, keyboard_reward, terminated


class KeyboardRewardDoneWrapper(BaseKeyboardRewardDoneWrapper):
    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key not in ["a", "b", "c"]:
            return last_intervened, done, reward

        last_intervened = True
        if key == "a":
            reward = -1
            done = True
            last_intervened = True
        elif key == "b":
            reward = 0
            last_intervened = True
        elif key == "c":
            reward = 1
            done = True
            last_intervened = True
        return last_intervened, done, reward


class KeyboardRewardDoneMultiStageWrapper(BaseKeyboardRewardDoneWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.stage_rewards = [0, 0.1, 1]

    def reset(self, *, seed=None, options=None):
        self.reward_stage = 0
        return super().reset(seed=seed, options=options)

    def _check_keypress(self) -> tuple[bool, bool, float]:
        last_intervened = False
        done = False
        reward = 0
        key = self.listener.get_key()
        if key is not None:
            print(f"Key pressed: {key}")
        if key == "a":
            self.reward_stage = 0
        elif key == "b":
            self.reward_stage = 1
        elif key == "c":
            self.reward_stage = 2

        if self.reward_stage == 2:
            done = True

        reward = self.stage_rewards[self.reward_stage]
        if key == "q":
            reward = -1
            done = False
        return last_intervened, done, reward


class KeyboardStartEndWrapper(gym.Wrapper):
    """Gate recording on `a`; end with `b` (fail, -1) or `c` (success, +1).

    Uses the listener's lossless press queue (``pop_pressed_keys``) instead of
    ``get_key``: at 10 Hz step rate a fast tap (<100 ms) often lands entirely
    between two polls and ``get_key`` never sees it — the queue catches every
    initial press in the evdev thread.

    Does not print — stdout inside a Ray actor lags the driver terminal via
    Ray's log monitor.  Events and state are surfaced through ``info`` so the
    driver-side caller can render realtime output.

    info fields added every step:
      - ``keyboard_phase``:  ``"pre"`` | ``"rec"``
      - ``keyboard_event``:  label from the LAST press handled this step —
            ``"start"`` | ``"restart"`` | ``"end_fail"`` | ``"end_success"``
            or ``None`` if no relevant press fired.
      - ``pre_record`` / ``record_reset`` — consumed by CollectEpisode.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._recording = False

    def reset(self, *, seed=None, options=None):
        self._recording = False
        # Drain any presses queued during the reset gap so they can't leak
        # into the next episode.
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        record_reset = False
        event: str | None = None
        for key in self.listener.pop_pressed_keys():
            if key == "a":
                event = "restart" if self._recording else "start"
                self._recording = True
                record_reset = True
            elif key == "b" and self._recording:
                event = "end_fail"
                reward = -1.0
                terminated = True
                break
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                break

        if not isinstance(info, dict):
            info = {}
        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        return obs, reward, terminated, truncated, info
