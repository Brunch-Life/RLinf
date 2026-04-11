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

"""GELLO intervention wrapper for joint-space control.

This wrapper overrides the policy's joint-space action with joint positions
read directly from the GELLO teleoperation device.

Two streaming modes:

* **step-gated** (default): ``action()`` returns the GELLO joints and the
  env's own ``step()`` forwards them to the controller at
  ``step_frequency`` Hz.  Simple but discards most GELLO samples.

* **direct-stream** (``direct_stream=True``): a daemon thread reads GELLO
  at ~1 kHz and pushes joint targets straight to the controller via
  ``move_joints``, bypassing env.step's rate gate.  Meant to be paired
  with ``FrankaJointRobotConfig.teleop_direct_stream=True`` so env.step
  does NOT also send move_joints — avoids two writers racing on
  franky's motion queue.  Policy buffers still see actions at 10 Hz
  from ``action()``; the stream is purely a hardware control path.
"""

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert


class GelloJointIntervention(gym.ActionWrapper):
    """Action wrapper that overrides policy action with GELLO joint-space input.

    In **absolute** mode, the GELLO joint positions are used directly as the
    action (with gripper appended).  In **delta** mode, the wrapper computes
    the difference between GELLO joints and the current robot joints and
    normalises it by ``action_scale``.

    Args:
        env: The wrapped environment (must be a :class:`FrankaJointEnv`).
        port: Serial port of the GELLO device.
        gripper_enabled: Whether the gripper channel is present in the action
            space.
        use_delta: If ``True``, produce delta joint actions instead of
            absolute positions.
        action_scale: Scaling factor for delta mode (must match the env's
            ``joint_action_scale``).
        direct_stream: If ``True``, start a background thread that streams
            GELLO joints to the controller at ~1 kHz.  Requires
            ``FrankaJointRobotConfig.teleop_direct_stream=True`` on the
            underlying env so env.step does not also forward move_joints.
        stream_period: Stream loop period in seconds (default 0.001 → 1 kHz).
    """

    def __init__(
        self,
        env,
        port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        self.expert = GelloJointExpert(port=port)
        self.last_intervene = 0

        # ── direct-stream daemon ────────────────────────────────────
        self._direct_stream = direct_stream
        self._stream_period = stream_period
        self._stream_thread: threading.Thread | None = None
        self._stream_running = False
        self._stream_last_gripper_open: bool | None = None

        if self._direct_stream:
            self._start_stream_thread()

    # ═══════════════════════════════════════════════════════════════
    #  Direct-stream daemon — bypasses env.step's 10 Hz rate gate
    # ═══════════════════════════════════════════════════════════════

    def _start_stream_thread(self) -> None:
        """Spawn the 1 kHz GELLO → controller pump."""
        controller = self._resolve_controller()
        if controller is None:
            # env still warming up; env.reset will create _controller.
            # Defer until first step when the attribute exists.
            return
        self._stream_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="GelloJointStream",
            daemon=True,
        )
        self._stream_thread.start()

    def _resolve_controller(self):
        """Return the underlying FrankyController handle, or None."""
        try:
            return self.get_wrapper_attr("_controller")
        except (AttributeError, Exception):
            return None

    def _stream_loop(self) -> None:
        """Read GELLO at ~1 kHz, push joint targets to the controller.

        Gripper events are batched on transitions only — gripper
        commands are slow (~100 ms) so streaming them every 1 ms would
        starve the serial/network channel.
        """
        controller = self._resolve_controller()
        if controller is None:
            return

        period = self._stream_period
        while self._stream_running:
            loop_start = time.time()

            if not self.expert.ready:
                time.sleep(period)
                continue

            try:
                q, g = self.expert.get_action()
            except Exception:
                time.sleep(period)
                continue

            try:
                # move_joints is non-blocking — franky preempts the
                # previous motion and re-plans via Ruckig.  .wait() is
                # the Ray actor call ack, not the robot motion.
                controller.move_joints(q.astype(np.float32)).wait()
            except Exception:
                pass  # best-effort streaming, env.step reads state

            if self.gripper_enabled:
                is_open_now = bool(g < 0.5)
                if self._stream_last_gripper_open is None:
                    self._stream_last_gripper_open = is_open_now
                elif is_open_now != self._stream_last_gripper_open:
                    try:
                        if is_open_now:
                            controller.open_gripper().wait()
                        else:
                            controller.close_gripper().wait()
                    except Exception:
                        pass
                    self._stream_last_gripper_open = is_open_now

            elapsed = time.time() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    # ═══════════════════════════════════════════════════════════════
    #  Action override (policy buffer path — runs at env step rate)
    # ═══════════════════════════════════════════════════════════════

    def _get_current_joint_positions(self) -> np.ndarray:
        """Read current joint positions from env state without triggering RPC.

        Uses the cached ``_franka_state`` on the env rather than calling
        ``get_joint_positions()`` which would issue a blocking remote call
        and mutate the env's internal state.
        """
        franka_state = self.get_wrapper_attr("_franka_state")
        return franka_state.arm_joint_position.copy()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not self.expert.ready:
            return action, False

        target_joints, target_gripper = self.expert.get_action()
        current_joints = self._get_current_joint_positions()

        if self.use_delta:
            delta_joints = target_joints - current_joints
            delta_joints = delta_joints / self.action_scale
            expert_a = np.clip(delta_joints, -1.0, 1.0)
        else:
            # Absolute mode: use GELLO joints directly
            expert_a = target_joints.copy()

        # Append gripper action only when gripper is enabled.
        # When gripper is disabled (GripperCloseEnv strips the last dim),
        # the action space is 7D (joints only) so we must NOT append.
        gripper_active = False
        if self.gripper_enabled:
            gripper_action = -(2 * target_gripper - 1.0)
            gripper_action = np.clip(gripper_action, -1.0, 1.0)
            gripper_active = np.abs(gripper_action).item() > 0.5
            expert_a = np.concatenate((expert_a, gripper_action), axis=0)

        # Determine if the GELLO is actively intervening
        movement = np.linalg.norm(target_joints - current_joints)

        if movement > 0.01 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True

        return action, False

    def step(self, action):
        new_action, replaced = self.action(action)

        # Lazy-start the stream thread on first step if the controller
        # wasn't ready at __init__ time.
        if self._direct_stream and self._stream_thread is None:
            self._start_stream_thread()

        obs, rew, done, truncated, info = self.env.step(new_action)
        if replaced:
            info["intervene_action"] = new_action

        return obs, rew, done, truncated, info

    def close(self):
        """Stop the stream thread before tearing down the env."""
        self._stream_running = False
        t = self._stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        return super().close()
