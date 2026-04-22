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

from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


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

        # Keyboard is the sole terminator in both phases — pre-record must not
        # auto-reset either, or `_align_to_gello` jerks the arms every 30 s.
        terminated = False
        truncated = False

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
