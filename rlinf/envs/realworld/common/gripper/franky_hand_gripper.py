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

"""Original Franka Hand parallel-jaw gripper via the franky / libfranka stack.

This backend talks to the Franka Hand directly through libfranka (no ROS),
mirroring how :class:`FrankyController` talks to the arm.  All move/grasp
commands are dispatched through ``franky.Gripper``'s ``*_async`` API so
they match the fire-and-forget cadence the env step loop expects — a
blocking ``move`` here would stretch a 10 Hz step to ~700 ms and disturb
the 1 kHz joint-impedance tracker.
"""

from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_gripper import BaseGripper

# Franka Hand factory limits (libfranka reports these via state, but we
# also keep static fallbacks for the constructor / homing path).
_FRANKA_HAND_MAX_WIDTH_M = 0.08
_FRANKA_HAND_MAX_SPEED_MPS = 0.1  # Franka Hand caps physical speed at ~0.1 m/s.
_FRANKA_HAND_MIN_SPEED_MPS = 0.01
# epsilon_inner / epsilon_outer — width tolerance around the grasp target.
# Set wide so grasp() reports success for any object width up to max_width.
_GRASP_EPSILON = _FRANKA_HAND_MAX_WIDTH_M


class FrankyHandGripper(BaseGripper):
    """Original Franka Hand controlled through ``franky.Gripper``.

    Args:
        robot_ip: Same FCI IP that drives the arm; the Hand and arm share
            one libfranka session per robot.
        max_width: Physical fully-open width in metres (0.08 for stock Hand).
        do_homing: Run ``homing()`` once on init so subsequent grasp/move
            calls are calibrated.  Skip this only if the Hand has already
            been homed in a sibling process this session.
    """

    def __init__(
        self,
        robot_ip: str,
        max_width: float = _FRANKA_HAND_MAX_WIDTH_M,
        do_homing: bool = True,
    ):
        import franky

        self._franky = franky
        self._logger = get_logger()
        self._robot_ip = robot_ip
        self._max_width = float(max_width)

        self._gripper = franky.Gripper(robot_ip)

        # franky.Gripper.max_width is only valid after homing on some
        # firmware revisions; trust the constructor value first, then
        # refresh post-homing if available.
        self._is_open_flag = True
        self._is_ready_flag = False

        if do_homing:
            try:
                self._gripper.homing()
                self._is_ready_flag = True
            except Exception as exc:
                self._logger.warning(
                    "FrankyHandGripper(%s) homing failed: %s", robot_ip, exc
                )
        else:
            # Best-effort: a successful state read is enough to consider
            # the gripper usable.  The first move/grasp will surface any
            # outstanding fault.
            try:
                _ = self._gripper.width
                self._is_ready_flag = True
            except Exception:
                self._is_ready_flag = False

        if self._is_ready_flag:
            try:
                width_attr = float(getattr(self._gripper, "max_width", max_width))
                if width_attr > 0.0:
                    self._max_width = width_attr
            except Exception:
                pass
            self._logger.info(
                "FrankyHandGripper ready on %s (max_width=%.3f m)",
                robot_ip,
                self._max_width,
            )

    # ── BaseGripper interface ────────────────────────────────────────

    def open(self, speed: float = 0.3) -> None:
        speed_mps = self._normalize_speed(speed)
        try:
            self._gripper.open_async(speed_mps)
        except AttributeError:
            # Older franky versions only expose move_async.
            self._gripper.move_async(self._max_width, speed_mps)
        self._is_open_flag = True

    def close(self, speed: float = 0.3, force: float = 130.0) -> None:
        speed_mps = self._normalize_speed(speed)
        force_n = max(1.0, float(force))
        # grasp() with width=0 + wide epsilon clamps shut on any object.
        self._gripper.grasp_async(
            0.0,
            speed_mps,
            force_n,
            epsilon_inner=_GRASP_EPSILON,
            epsilon_outer=_GRASP_EPSILON,
        )
        self._is_open_flag = False

    def move(self, position: float, speed: float = 0.3) -> None:
        speed_mps = self._normalize_speed(speed)
        width = float(np.clip(position, 0.0, self._max_width))
        self._gripper.move_async(width, speed_mps)

    @property
    def position(self) -> float:
        try:
            return float(self._gripper.width)
        except Exception:
            return 0.0

    @property
    def is_open(self) -> bool:
        return self._is_open_flag

    def is_ready(self) -> bool:
        return self._is_ready_flag

    def cleanup(self) -> None:
        # franky.Gripper releases the libfranka session on GC; no explicit
        # close exists.  Best effort: stop any in-flight motion.
        stop = getattr(self._gripper, "stop_async", None)
        if stop is None:
            return
        try:
            stop()
        except Exception:
            pass

    # ── helpers ──────────────────────────────────────────────────────

    def _normalize_speed(self, speed: float) -> float:
        """Map a [0, 1] speed (BaseGripper convention) to m/s.

        Values already in m/s (e.g. 0.05) pass through clamped because
        the Hand's hard limit is _FRANKA_HAND_MAX_SPEED_MPS.
        """
        s = float(speed)
        if s <= 1.0:
            s = s * _FRANKA_HAND_MAX_SPEED_MPS
        return float(np.clip(s, _FRANKA_HAND_MIN_SPEED_MPS, _FRANKA_HAND_MAX_SPEED_MPS))


def get_default_max_width() -> Optional[float]:
    """Static accessor used by callers that need the catalog value before
    constructing a controller (e.g. for action-space scaling)."""
    return _FRANKA_HAND_MAX_WIDTH_M
