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
"""Foot-pedal start / success / fail wrapper for autonomous policy eval.

Workflow
--------
After ``reset()`` the wrapper sits idle and the robot holds the reset
pose. Pressing the foot pedal:

* ``a`` (idle):    start rollout — subsequent ``step()`` calls forward
                   to the wrapped env so the policy drives the robot.
* ``c`` (running): episode succeeded. ``terminated=True``, ``reward=1.0``,
                   ``info["eval_result"]="success"``. The wrapper itself
                   calls ``env.reset()`` to drive the robot home before
                   returning, then sits idle waiting for ``a``.
* ``b`` (running): episode failed. ``terminated=True``, ``reward=0.0``,
                   ``info["eval_result"]="failure"``. Same internal reset.

The internal reset on b/c is what makes the pedal feel "live": the eval
env_worker only resets at ``eval_rollout_epoch`` boundaries (auto_reset
is False during eval), so without this the robot would freeze in place
after ``b`` until the rest of the chunk steps in the current epoch ran
out. Driving home immediately on the pedal lets the operator stage the
next workpiece right away.

Pre-running ("idle") ``step()`` deliberately does **not** invoke
``self.env.step()`` — the franky impedance controller keeps holding the
target dispatched by the env's last ``reset()``, so the robot stays
physically still while the human positions the workpiece. The wrapper
returns the most recent observation unchanged so the policy's prediction
loop keeps cycling without committing fresh joint commands. This trades
the "every step calls env.step" gym invariant for human-in-the-loop
episode boundaries that match the operator's pedal.

While running we also force ``terminated`` / ``truncated`` to ``False``
unless the pedal fires — so the policy never gets cut off by the env's
own ``max_episode_steps`` time-out. Set ``max_episode_steps`` large
enough that the pedal is always the boundary owner.
"""

import time
from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, ObsType

from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener


class KeyboardEvalControlWrapper(gym.Wrapper):
    """Foot-pedal-gated start/stop for autonomous policy eval rollouts."""

    # Sleep between idle polls — keeps the wrapper from pinning a CPU core
    # while waiting for the operator to press ``a``.
    IDLE_POLL_S = 0.05

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.listener = KeyboardListener()
        self._running = False
        self._last_obs: Any = None

    def reset(self, *, seed=None, options=None):
        self._running = False
        # Drain queued presses from any human bouncing the pedal during the
        # reset gap so they can't leak into the next episode.
        self.listener.pop_pressed_keys()
        ret = self.env.reset(seed=seed, options=options)
        self._last_obs = ret[0] if isinstance(ret, tuple) else ret
        return ret

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if not self._running:
            # Idle: poll pedal, hold robot via the controller's last target.
            time.sleep(self.IDLE_POLL_S)
            for key in self.listener.pop_pressed_keys():
                if key == "a":
                    self._running = True
                    return self._idle_response(event="start")
            return self._idle_response(event=None)

        # Running: forward to the wrapped env.
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._last_obs = obs

        # Pedal owns episode boundaries — ignore env-side time-out and only
        # honor the human's success / fail signal.
        terminated = False
        truncated = False

        result: str | None = None
        for key in self.listener.pop_pressed_keys():
            if key == "c":
                terminated = True
                reward = 1.0
                result = "success"
                self._running = False
                break
            if key == "b":
                terminated = True
                reward = 0.0
                result = "failure"
                self._running = False
                break

        if result is not None:
            obs = self._reset_after_pedal()

        if not isinstance(info, dict):
            info = {}
        info["eval_phase"] = "rec" if self._running else "pre"
        info["eval_result"] = result
        return obs, reward, terminated, truncated, info

    def _idle_response(self, event: str | None):
        info = {"eval_phase": "pre", "eval_event": event, "eval_result": None}
        return self._last_obs, 0.0, False, False, info

    def _reset_after_pedal(self):
        """Drive the robot back to its reset pose and refresh ``_last_obs``.

        Called when the operator's pedal ends an episode. The eval env_worker
        runs with ``auto_reset=False`` and only resets between rollout epochs,
        so without this the robot would hang at its last commanded TCP target
        until the remaining chunk steps in the current epoch are spent.
        """
        # Drain pedal noise from the bounce/release to keep the next idle
        # poll clean.
        self.listener.pop_pressed_keys()
        ret = self.env.reset()
        new_obs = ret[0] if isinstance(ret, tuple) else ret
        self._last_obs = new_obs
        return new_obs
