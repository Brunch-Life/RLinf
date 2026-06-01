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

"""Absolute end-effector pose interpolation for high-frequency dispatch."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation as R

POSE_DIM = 7  # [x, y, z, rx, ry, rz, gripper] per arm


def interpolate_segment(
    start_pose: np.ndarray, end_pose: np.ndarray, num_substeps: int
) -> np.ndarray:
    """Interpolate from ``start_pose`` to ``end_pose`` into ``num_substeps`` points.

    Position and gripper are linearly interpolated; orientation goes through the
    shortest-path quaternion nlerp (component lerp + normalize). The returned points
    exclude the start and include the end, so consuming them in order drives the arm
    from ``start_pose`` to ``end_pose``.

    Args:
        start_pose: Current target, shape ``(7,)`` single arm or ``(14,)`` dual arm.
        end_pose: Next target, same shape as ``start_pose``.
        num_substeps: Number of high-frequency points to emit (>= 1).

    Returns:
        Array of shape ``(num_substeps, start_pose.size)`` of absolute poses.
    """
    start = np.asarray(start_pose, dtype=np.float64).reshape(-1, POSE_DIM)
    end = np.asarray(end_pose, dtype=np.float64).reshape(-1, POSE_DIM)
    alphas = np.linspace(0.0, 1.0, num_substeps + 1)[1:]

    out = np.empty((num_substeps, start.shape[0], POSE_DIM), dtype=np.float64)
    for arm in range(start.shape[0]):
        s, e = start[arm], end[arm]
        for k in (0, 1, 2, 6):  # xyz + gripper
            out[:, arm, k] = (1.0 - alphas) * s[k] + alphas * e[k]
        q0 = R.from_euler("xyz", s[3:6]).as_quat()
        q1 = R.from_euler("xyz", e[3:6]).as_quat()
        if np.dot(q0, q1) < 0.0:  # shortest path
            q1 = -q1
        q = (1.0 - alphas)[:, None] * q0[None, :] + alphas[:, None] * q1[None, :]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        out[:, arm, 3:6] = R.from_quat(q).as_euler("xyz")
    return out.reshape(num_substeps, -1)
