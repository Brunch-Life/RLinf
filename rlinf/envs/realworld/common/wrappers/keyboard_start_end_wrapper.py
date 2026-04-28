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

import math
import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardStartEndWrapper(gym.Wrapper):
    """Foot-pedal-friendly keyboard wrapper for dual-Franka data collection.

    The pedal exposes only ``a`` / ``b`` / ``c`` so the wrapper is rebound
    to the workflow we actually use, not the legacy "b=fail" semantics:

    * ``a`` (pre):  start a fresh rec episode at the current pose.
    * ``a`` (rec):  abort — drop buffer, return to pre. The arm is **not**
                    reset (GELLO mode keeps tracking the operator's pose;
                    other teleop devices that prefer a home-reset on abort
                    should layer that in their own reset logic).
    * ``b`` (rec):  bump ``segment_id`` by 1 for the upcoming frames. Used
                    to mark sub-task boundaries inside one episode.
                    Debounced at 1 s — back-to-back presses inside the
                    window are silently ignored so an accidentally held or
                    bouncing pedal can't shred the segment timeline.
    * ``c`` (rec):  end with success (reward=+1, done=True). Episode is
                    saved by the outer collector.

    info fields added every step:
      * ``keyboard_phase``   — ``"pre"`` | ``"rec"``
      * ``keyboard_event``   — ``"start"`` | ``"abort"`` | ``"segment"`` |
                               ``"end_success"`` | ``None``
      * ``pre_record``       — bool, consumed by CollectEpisode to skip recording
      * ``record_reset``     — bool, consumed by CollectEpisode to clear buffer
      * ``segment_advance``  — bool, consumed by CollectEpisode to bump seg_id
    """

    SEGMENT_DEBOUNCE_S = 1.0

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._recording = False
        self._last_segment_ts = -math.inf

    def reset(self, *, seed=None, options=None):
        self._recording = False
        self._last_segment_ts = -math.inf
        # Drain any presses queued during the reset gap so they can't leak
        # into the next episode.
        self.listener.pop_pressed_keys()
        return self.env.reset(seed=seed, options=options)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Pedal owns episode boundaries — pre/abort must NOT auto-reset.
        terminated = False
        truncated = False

        record_reset = False
        segment_advance = False
        event: str | None = None

        for key in self.listener.pop_pressed_keys():
            if key == "a":
                if self._recording:
                    # Drop the in-progress episode but stay where we are.
                    event = "abort"
                    self._recording = False
                    record_reset = True
                    self._last_segment_ts = -math.inf
                else:
                    # Begin a fresh rec episode at the current pose.
                    event = "start"
                    self._recording = True
                    record_reset = True
                    self._last_segment_ts = -math.inf
            elif key == "b" and self._recording:
                now = time.monotonic()
                if now - self._last_segment_ts >= self.SEGMENT_DEBOUNCE_S:
                    event = "segment"
                    segment_advance = True
                    self._last_segment_ts = now
                # else: silently ignore — keeps mini-segments out of the data.
            elif key == "c" and self._recording:
                event = "end_success"
                reward = 1.0
                terminated = True
                self._recording = False
                break

        if not isinstance(info, dict):
            info = {}
        info["pre_record"] = not self._recording
        info["record_reset"] = record_reset
        info["keyboard_phase"] = "rec" if self._recording else "pre"
        info["keyboard_event"] = event
        info["segment_advance"] = segment_advance
        return obs, reward, terminated, truncated, info
