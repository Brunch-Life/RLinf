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

"""Dual-arm Franka env with TCP-rot6d actions.

20-D action vector (10 per arm):
``[L_xyz(3), L_rot6d(6), L_grip(1), R_xyz(3), R_rot6d(6), R_grip(1)]``.

Per-step dispatch: each :py:meth:`step` pushes ``(xyz, quat)`` into a
lazily-started :py:class:`franky.CartesianImpedanceTracker` per arm via
:py:meth:`FrankyController.move_tcp_pose`. No motion generator in the
loop, so per-step target jumps absorb as impedance tracking error (soft)
rather than tripping libfranka's Ruckig trajectory reflexes (hard).
Non-blocking — each set_target returns immediately.

Observation schema is minimal: ``gripper_position`` (2,) +
``tcp_pose_rot6d`` (18,). The concat order is locked in by
``STATE_LAYOUT``, which is consumed by ``RealWorldEnv._wrap_obs`` instead
of its legacy alphabetical sort. The resulting 20-byte flat prefix
``[L_grip, R_grip, L_xyz, L_rot6d, R_xyz, R_rot6d]`` is what the SFT
policy's ``_rearrange_state`` slicer expects.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.utils.rot6d import matrix_to_rot6d, rot6d_to_quat_xyzw_safe

from .dual_franka_env import NUM_ARMS, DualFrankaRobotConfig
from .dual_franka_franky_env import DualFrankaFrankyEnv

ACTION_DIM_PER_ARM = 10  # xyz(3) + rot6d(6) + gripper(1)
# Per-arm width of the ``tcp_pose_rot6d`` obs slot produced by
# :py:meth:`DualFrankaRot6dEnv._tcp_rot6d_18d` (xyz(3) + rot6d(6); gripper
# lives in its own slot).
PROPRIO_DIM_PER_ARM = 9


@dataclass
class DualFrankaRot6dRobotConfig(DualFrankaRobotConfig):
    """Config for :class:`DualFrankaRot6dEnv`.

    Intentionally does NOT inherit joint-action fields
    (``joint_action_mode``, ``joint_action_scale``, ``joint_velocity_limit``,
    ``joint_position_limits_*``, ``teleop_direct_stream``) — rot6d mode
    drives a Cartesian impedance tracker and those knobs have no meaning.

    Cartesian safety bounds are inherited via ``ee_pose_limit_min`` /
    ``ee_pose_limit_max`` from :class:`DualFrankaRobotConfig`.
    """


class DualFrankaRot6dEnv(DualFrankaFrankyEnv):
    """Dual-arm Franka env with 20-D TCP-rot6d waypoint actions."""

    CONFIG_CLS: type[DualFrankaRot6dRobotConfig] = DualFrankaRot6dRobotConfig

    PER_ARM_ACTION_DIM = ACTION_DIM_PER_ARM
    GRIPPER_IDX_IN_ARM = 9  # xyz(3) + rot6d(6) then gripper

    # Explicit flat-state concat order (consumed by
    # ``RealWorldEnv._wrap_obs``). Keeps the 20-byte prefix the SFT policy
    # slicer expects — ``[L_grip, R_grip, L_xyz, L_rot6d, R_xyz, R_rot6d]``
    # — without relying on a key-name alphabetical accident.
    STATE_LAYOUT = ("gripper_position", "tcp_pose_rot6d")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Per-arm previous quat for hemisphere alignment across steps in
        # ``step`` dispatch mode. Seeded on reset from live TCP so the
        # first step doesn't flip sign against the current pose.
        self._prev_step_quat = [None, None]

    def reset(self, *, seed=None, options=None):
        # Clear per-arm step-mode hemisphere seed so the next step after
        # reset aligns against live TCP, not the previous episode's tail.
        self._prev_step_quat = [None, None]
        return super().reset(seed=seed, options=options)

    # ---------------------------------------------------------------- spaces

    def _init_action_obs_spaces(self):
        self._cartesian_safety_boxes()

        # Per-arm action = [xyz (3, m), rot6d (6, unit), gripper_trigger (1)].
        # xyz bounds come from the safe-box config; rot6d bounds are widened
        # to [-1.5, 1.5] so normalized model outputs have headroom before
        # Gram-Schmidt renormalizes. Gripper trigger is ±1.
        rot6d_low = -1.5 * np.ones(6, dtype=np.float32)
        rot6d_high = 1.5 * np.ones(6, dtype=np.float32)
        left_low = np.concatenate(
            [self.config.ee_pose_limit_min[0, :3], rot6d_low, np.array([-1.0])]
        )
        left_high = np.concatenate(
            [self.config.ee_pose_limit_max[0, :3], rot6d_high, np.array([1.0])]
        )
        right_low = np.concatenate(
            [self.config.ee_pose_limit_min[1, :3], rot6d_low, np.array([-1.0])]
        )
        right_high = np.concatenate(
            [self.config.ee_pose_limit_max[1, :3], rot6d_high, np.array([1.0])]
        )
        act_low = np.concatenate([left_low, right_low]).astype(np.float32)
        act_high = np.concatenate([left_high, right_high]).astype(np.float32)
        self.action_space = gym.spaces.Box(act_low, act_high)

        # Minimal obs schema: only what the SFT policy eats. Keys match
        # ``STATE_LAYOUT``. No more smuggling — ``tcp_pose_rot6d`` carries
        # the 18-D per-arm [xyz, rot6d] payload under an honest name.
        camera_specs = self._all_camera_specs()
        self.observation_space = gym.spaces.Dict(
            {
                "state": gym.spaces.Dict(
                    {
                        "gripper_position": gym.spaces.Box(-1, 1, shape=(NUM_ARMS,)),
                        "tcp_pose_rot6d": gym.spaces.Box(
                            -np.inf,
                            np.inf,
                            shape=(NUM_ARMS * PROPRIO_DIM_PER_ARM,),
                        ),
                    }
                ),
                "frames": gym.spaces.Dict(
                    {
                        name: gym.spaces.Box(
                            0, 255, shape=(224, 224, 3), dtype=np.uint8
                        )
                        for name, _, _ in camera_specs
                    }
                ),
            }
        )
        self._base_observation_space = copy.deepcopy(self.observation_space)

    # --------------------------------------------------------- step dispatch

    def _dispatch_arm_motion(
        self,
        actions: np.ndarray,
        states: list,
        ctrls: list,
        dt: float,
    ) -> None:
        """Push one Cartesian target per arm into the impedance tracker.

        Decodes per-arm rot6d → quat with hemisphere alignment against the
        previous step's target, then fires ``move_tcp_pose`` per arm into
        the :class:`franky.CartesianImpedanceTracker` (non-blocking
        ``set_target``).
        """
        del dt

        for arm in range(NUM_ARMS):
            xyz = actions[arm, 0:3]
            rot6d = actions[arm, 3:9]

            prev_quat = self._prev_step_quat[arm]
            if prev_quat is None:
                prev_quat = states[arm].tcp_pose[3:]
            quat = rot6d_to_quat_xyzw_safe(rot6d, fallback_quat_xyzw=prev_quat)
            if float(np.dot(quat, prev_quat)) < 0.0:
                quat = -quat
            self._prev_step_quat[arm] = quat

            ctrls[arm].move_tcp_pose(np.concatenate([xyz, quat]).astype(np.float64))

    # ------------------------------------------------------------ obs + utils

    def _get_observation(self) -> dict:
        if self.config.is_dummy:
            return self._base_observation_space.sample()
        frames = self._get_camera_frames()

        # Keys + order drive the flat state[:20] prefix that the SFT
        # policy's ``_rearrange_state`` slices. Order is locked in by
        # ``STATE_LAYOUT``; the ``tcp_pose_rot6d`` key carries the 18-D
        # per-arm [xyz, rot6d] payload under its real name.
        state = {
            "gripper_position": np.array(
                [
                    self._left_state.gripper_position,
                    self._right_state.gripper_position,
                ],
                dtype=np.float32,
            ),
            "tcp_pose_rot6d": self._tcp_rot6d_18d(),
        }
        return copy.deepcopy({"state": state, "frames": frames})

    def _tcp_rot6d_18d(self) -> np.ndarray:
        """Build the ``[L_xyz, L_rot6d, R_xyz, R_rot6d]`` 18-d vector.

        Live TCP quat is xyzw (franky convention); converted to rot6d via
        ``scipy.Rotation.from_quat(...).as_matrix()`` then
        ``matrix_to_rot6d`` (first two columns flattened) — no euler on
        the path, so no wrap artifacts.
        """
        out = np.zeros(18, dtype=np.float32)
        for arm, st in enumerate((self._left_state, self._right_state)):
            base = arm * PROPRIO_DIM_PER_ARM
            out[base : base + 3] = st.tcp_pose[:3]
            mat = R.from_quat(st.tcp_pose[3:]).as_matrix()
            out[base + 3 : base + 9] = matrix_to_rot6d(mat)
        return out
