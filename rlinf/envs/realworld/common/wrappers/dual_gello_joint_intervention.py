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

"""Dual-arm GELLO intervention wrapper for joint-space control.

Supports step-gated mode (action forwarded by env.step) and direct-stream
mode (daemon thread pushes joint targets to each controller at ~1 kHz,
bypassing env.step's rate gate — pair with
``DualFrankaJointRobotConfig.teleop_direct_stream=True``).

Three runtime modes selectable via :meth:`set_mode` (default chosen at
construction time, typically wired through the YAML
``gello_default_mode`` field):

* ``"policy"`` — pass the upstream action through unchanged. GELLO is
  ignored. Intended for HG-DAgger rollouts where the policy drives the
  robot until the human clutches in.
* ``"homing"`` — issue a "stay-in-place" target for both arms (current
  joints + neutral grippers in absolute mode, zero vector in delta
  mode). The robot keeps responding every step but does not advance;
  an external GELLO actuator can use this window to slew the leader to
  the Franka's joints before teleop hand-off.
* ``"teleop"`` — overlay the live GELLO joint stream on the action
  (the legacy collection behaviour). This is the default to preserve
  existing pure-teleop YAMLs.
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert

_VALID_MODES = ("policy", "homing", "teleop")


class DualGelloJointIntervention(gym.ActionWrapper):
    """Action wrapper overriding the policy action with two GELLO joint streams.

    Args:
        env: The wrapped dual-arm environment (must be a
            :class:`DualFrankaJointEnv`).
        left_port: Serial port for the left GELLO device.
        right_port: Serial port for the right GELLO device.
        gripper_enabled: Whether gripper channels are present in the
            action space.
        use_delta: If ``True``, produce delta joint actions instead of
            absolute positions.
        action_scale: Scaling factor for delta mode (must match the env's
            ``joint_action_scale``).
        direct_stream: If ``True``, start a background thread that
            streams GELLO joints to both controllers at ~1 kHz.  Requires
            ``DualFrankaJointRobotConfig.teleop_direct_stream=True``.
        stream_period: Stream loop period in seconds (default 0.001 →
            1 kHz).
    """

    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
        default_mode: str = "teleop",
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        self.left_expert = GelloJointExpert(port=left_port)
        self.right_expert = GelloJointExpert(port=right_port)
        self.last_intervene = 0.0

        if default_mode not in _VALID_MODES:
            raise ValueError(
                f"default_mode must be one of {_VALID_MODES}, got {default_mode!r}"
            )
        self._default_mode = default_mode
        self._mode = default_mode
        self._mode_lock = threading.Lock()

        self._direct_stream = direct_stream
        self._stream_period = stream_period
        self._stream_thread: threading.Thread | None = None
        self._stream_running = False
        self._stream_last_gripper_open: list[bool | None] = [None, None]
        self._stream_paused = threading.Event()
        self._stream_paused.set()  # starts unpaused

    # ------------------------------------------------------------------
    # Mode API
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        """Switch the runtime intervention mode (thread-safe)."""
        if mode not in _VALID_MODES:
            raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
        with self._mode_lock:
            self._mode = mode

    @property
    def mode(self) -> str:
        with self._mode_lock:
            return self._mode

    def _freeze_action(self) -> np.ndarray:
        """Stay-in-place action under the current action_mode.

        Absolute: per-arm ``[current_q (7), 0_grip]`` — issuing this as
        the env action makes the franky controller hold the cached
        pose. Delta: zero unit vector. Used in ``homing`` and as a
        safety fall-through in ``teleop`` when GELLOs are not yet
        streaming, so a placeholder upstream action never reaches the
        robot.
        """
        per_arm_dim = 8 if self.gripper_enabled else 7
        if self.use_delta:
            return np.zeros(2 * per_arm_dim, dtype=np.float32)
        current = self._get_current_joint_positions()  # (2, 7)
        if self.gripper_enabled:
            return np.concatenate([current[0], [0.0], current[1], [0.0]]).astype(
                np.float32
            )
        return np.concatenate([current[0], current[1]]).astype(np.float32)

    def _start_stream_thread(self) -> None:
        """Spawn the 1 kHz GELLO → controllers pump."""
        if self._resolve_controllers() == (None, None):
            return
        self._stream_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="DualGelloJointStream",
            daemon=True,
        )
        self._stream_thread.start()

    def _resolve_controllers(self):
        """Return ``(left_ctrl, right_ctrl)`` or ``(None, None)``."""
        try:
            left = self.get_wrapper_attr("_left_ctrl")
            right = self.get_wrapper_attr("_right_ctrl")
            return left, right
        except (AttributeError, Exception):
            return None, None

    def _stream_loop(self) -> None:
        """Read both GELLOs at ~1 kHz, push targets to each controller.

        Per-arm gripper events are edge-triggered only — open/close
        gripper RPCs take ~100 ms and streaming them at 1 kHz would
        starve the serial/network channel.
        """
        left_ctrl, right_ctrl = self._resolve_controllers()
        if left_ctrl is None or right_ctrl is None:
            return

        period = self._stream_period
        ctrls = (left_ctrl, right_ctrl)

        while self._stream_running:
            self._stream_paused.wait()
            if not self._stream_running:
                break

            loop_start = time.time()

            if not (self.left_expert.ready and self.right_expert.ready):
                time.sleep(period)
                continue

            # Read both GELLOs before sending so the two targets are
            # as time-synchronised as possible.
            try:
                left_q, left_g = self.left_expert.get_action()
                right_q, right_g = self.right_expert.get_action()
            except Exception:
                time.sleep(period)
                continue

            # Fire both move_joints in parallel, then wait both RPCs.
            try:
                lf = left_ctrl.move_joints(left_q.astype(np.float32))
                rf = right_ctrl.move_joints(right_q.astype(np.float32))
                lf.wait()
                rf.wait()
            except Exception:
                pass  # best-effort streaming; env.step reads state

            if self.gripper_enabled:
                for arm_idx, (ctrl, grip) in enumerate(zip(ctrls, (left_g, right_g))):
                    is_open_now = grip.item() < 0.5
                    prev = self._stream_last_gripper_open[arm_idx]
                    if prev is None:
                        self._stream_last_gripper_open[arm_idx] = is_open_now
                    elif is_open_now != prev:
                        try:
                            if is_open_now:
                                ctrl.open_gripper()
                            else:
                                ctrl.close_gripper()
                        except Exception:
                            pass
                        self._stream_last_gripper_open[arm_idx] = is_open_now

            elapsed = time.time() - loop_start
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _get_current_joint_positions(self) -> np.ndarray:
        """Return cached ``(2, 7)`` joint positions from the env."""
        return self.get_wrapper_attr("get_joint_positions")()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not (self.left_expert.ready and self.right_expert.ready):
            return action, False

        left_q, left_g = self.left_expert.get_action()
        right_q, right_g = self.right_expert.get_action()
        current = self._get_current_joint_positions()  # (2, 7)

        per_arm = []
        for target_q, current_q in zip((left_q, right_q), (current[0], current[1])):
            if self.use_delta:
                delta_q = (target_q - current_q) / self.action_scale
                arm_a = np.clip(delta_q, -1.0, 1.0)
            else:
                arm_a = target_q.copy()
            per_arm.append(arm_a)

        gripper_active = False
        if self.gripper_enabled:
            grippers = []
            for grip in (left_g, right_g):
                g = -(2 * grip - 1.0)
                g = np.clip(g, -1.0, 1.0)
                grippers.append(g)
                if np.abs(g).item() > 0.5:
                    gripper_active = True
            # Concatenate per-arm [q7, grip1].
            expert_a = np.concatenate(
                [per_arm[0], grippers[0], per_arm[1], grippers[1]], axis=0
            )
        else:
            expert_a = np.concatenate(per_arm, axis=0)

        movement = np.linalg.norm(
            np.concatenate([left_q, right_q]) - np.concatenate([current[0], current[1]])
        )
        if movement > 0.01 or gripper_active:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def _align_to_gello(self) -> bool:
        """Slowly transit both arms to each GELLO's current pose.

        Without this, direct-stream mode would push a large, possibly
        far-away target straight into the impedance tracker on the
        first loop — causing a reference jump and visible shake.

        ``reset_joint`` is a blocking franky ``JointMotion`` with
        internal velocity/acceleration limits, so both arms slew
        smoothly to GELLO.  After it returns the streaming delta is
        ~zero.

        Returns ``True`` on success.
        """
        if not (self.left_expert.ready and self.right_expert.ready):
            return False
        left_ctrl, right_ctrl = self._resolve_controllers()
        if left_ctrl is None or right_ctrl is None:
            return False
        try:
            left_q, _ = self.left_expert.get_action()
            right_q, _ = self.right_expert.get_action()
        except Exception:
            return False
        try:
            lf = left_ctrl.reset_joint(np.asarray(left_q, dtype=np.float64).tolist())
            rf = right_ctrl.reset_joint(np.asarray(right_q, dtype=np.float64).tolist())
            lf.wait()
            rf.wait()
        except Exception:
            return False
        # Refresh cached states so the next step's delta / obs use the
        # post-alignment joint positions instead of the stale home.
        try:
            left_state = left_ctrl.get_state().wait()[0]
            right_state = right_ctrl.get_state().wait()[0]
            self.unwrapped._left_state = left_state
            self.unwrapped._right_state = right_state
        except Exception:
            pass
        return True

    def reset(self, **kwargs):
        """Pause streaming during reset to avoid racing with reset_joint.

        Also tells the inner env to skip its ``reset_joint(home)`` slew:
        we immediately ``_align_to_gello()`` below, and a "home → GELLO"
        double-slew just breaks teleop's "tracking continues" feel.
        """
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options

        self._stream_paused.clear()
        aligned = False
        try:
            result = self.env.reset(**kwargs)
            if self._direct_stream:
                aligned = self._align_to_gello()
        finally:
            self._stream_paused.set()
            if self._direct_stream and aligned and self._stream_thread is None:
                self._start_stream_thread()
        return result

    def step(self, action):
        mode = self.mode

        if mode == "policy":
            effective = action
            replaced = False
        elif mode == "homing":
            effective = self._freeze_action()
            replaced = False
        else:  # "teleop"
            new_action, ok = self.action(action)
            if ok:
                effective = new_action
                replaced = True
            else:
                # GELLO not streaming yet (boot window). Freeze rather
                # than forwarding the upstream action — under
                # ``default_mode='teleop'`` the upstream is typically a
                # placeholder zero vector that would otherwise drive
                # the robot to its joint origin.
                effective = self._freeze_action()
                replaced = False

        # Lazy-start the 1 kHz streamer if controllers weren't ready
        # at __init__ time.
        if self._direct_stream and self._stream_thread is None:
            self._start_stream_thread()

        obs, rew, done, truncated, info = self.env.step(effective)
        if replaced:
            info["intervene_action"] = effective
        return obs, rew, done, truncated, info

    def close(self):
        """Stop the stream thread before tearing down the env."""
        self._stream_running = False
        t = self._stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        return super().close()
