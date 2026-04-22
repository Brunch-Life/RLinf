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
"""SE(3) body-frame delta / absolute transforms for openpi data pipeline.

Replaces openpi's component-wise ``DeltaActions`` / ``AbsoluteActions``
for the rot6d pose layout. Rotations compose via matrix multiplication,
not scalar subtraction — using the stock openpi transforms on rot6d would
silently produce non-orthogonal garbage.

Layout is described structurally (not via a boolean mask) because a
``pose6d`` block spans 9 channels that need to move together under a
single SE(3) transform:

.. code-block:: python

    DUAL_ARM_ROT6D_LAYOUT = [
        {"kind": "pose6d", "xyz": slice(0, 3), "rot6d": slice(3, 9)},    # L arm
        {"kind": "scalar_abs", "idx": 9},                                 # L grip
        {"kind": "pose6d", "xyz": slice(10, 13), "rot6d": slice(13, 19)}, # R arm
        {"kind": "scalar_abs", "idx": 19},                                # R grip
    ]

State / action shapes match openpi's convention:

- ``state``: shape ``(D,)`` — single chunk-start reference, broadcast
  against every time step of the action chunk (identical to the stock
  ``DeltaActions.__call__``).
- ``actions``: shape ``(H, D)`` — chunk of absolute targets (delta forward)
  or chunk of deltas (delta inverse).

Padding past ``D_real`` (e.g. openpi's ``pad_to_dim(action_dim=32)``
tail) is left untouched so this transform composes with ``pad_to_dim``
without extra plumbing.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
from openpi import transforms as _transforms

from rlinf.utils.rot6d import (
    matrix_to_rot6d,
    pose_to_SE3,
    se3_body_compose,
    se3_body_delta,
)

DUAL_ARM_ROT6D_LAYOUT: tuple[dict[str, Any], ...] = (
    {"kind": "pose6d", "xyz": slice(0, 3), "rot6d": slice(3, 9)},
    {"kind": "scalar_abs", "idx": 9},
    {"kind": "pose6d", "xyz": slice(10, 13), "rot6d": slice(13, 19)},
    {"kind": "scalar_abs", "idx": 19},
)


def _validate_layout(layout: Sequence[dict[str, Any]]) -> None:
    """Fail loud on typos in the structured layout."""
    for entry in layout:
        kind = entry.get("kind")
        if kind == "pose6d":
            assert "xyz" in entry and "rot6d" in entry, entry
            assert isinstance(entry["xyz"], slice) and isinstance(entry["rot6d"], slice)
            assert entry["xyz"].stop - entry["xyz"].start == 3, entry
            assert entry["rot6d"].stop - entry["rot6d"].start == 6, entry
        elif kind == "scalar_abs":
            assert isinstance(entry["idx"], int), entry
        else:
            raise AssertionError(f"unknown layout kind: {entry!r}")


def _state_SE3(state: np.ndarray, xyz_slc: slice, r6_slc: slice) -> np.ndarray:
    """Extract a 4x4 body-frame from a chunk-start state vector."""
    xyz = np.asarray(state[..., xyz_slc], dtype=np.float64)
    r6 = np.asarray(state[..., r6_slc], dtype=np.float64)
    return pose_to_SE3(xyz, r6)


def _action_chunk_SE3(
    actions: np.ndarray, xyz_slc: slice, r6_slc: slice
) -> np.ndarray:
    """Build (H, 4, 4) chunk of absolute-target SE(3) frames."""
    xyz = np.asarray(actions[..., xyz_slc], dtype=np.float64)  # (H, 3)
    r6 = np.asarray(actions[..., r6_slc], dtype=np.float64)  # (H, 6)
    return pose_to_SE3(xyz, r6)


def _write_SE3_back(
    actions: np.ndarray, T_chunk: np.ndarray, xyz_slc: slice, r6_slc: slice
) -> None:
    """Mutate ``actions`` in place with the new pose6d block from ``T_chunk``."""
    actions[..., xyz_slc] = T_chunk[..., :3, 3].astype(actions.dtype)
    actions[..., r6_slc] = matrix_to_rot6d(T_chunk[..., :3, :3]).astype(actions.dtype)


@dataclasses.dataclass(frozen=True)
class RigidBodyDeltaActions(_transforms.DataTransformFn):
    """Convert absolute-target action chunks to body-frame SE(3) deltas.

    For each ``pose6d`` block: ``T_delta[t] = inv(T_state) @ T_abs[t]``,
    where ``T_state`` is built from the chunk-start state (openpi's
    ``state[t0]`` broadcast convention). ``scalar_abs`` slots are left
    untouched (gripper is absolute and not part of the rigid-body group).
    """

    layout: Sequence[dict[str, Any]]

    def __post_init__(self):
        _validate_layout(self.layout)

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        assert actions.ndim >= 2, (
            f"actions must be (H, D); got {actions.shape}"
        )
        # Guard against in-place mutation surprising callers.
        actions = actions.copy()

        for entry in self.layout:
            if entry["kind"] != "pose6d":
                continue
            xyz_slc, r6_slc = entry["xyz"], entry["rot6d"]
            T_state = _state_SE3(state, xyz_slc, r6_slc)
            T_abs = _action_chunk_SE3(actions, xyz_slc, r6_slc)
            T_delta = se3_body_delta(T_state, T_abs)
            _write_SE3_back(actions, T_delta, xyz_slc, r6_slc)

        data["actions"] = actions
        return data


@dataclasses.dataclass(frozen=True)
class RigidBodyAbsoluteActions(_transforms.DataTransformFn):
    """Inverse of :class:`RigidBodyDeltaActions` — recover absolute targets.

    ``T_abs[t] = T_state @ T_delta[t]``. Must mirror the forward transform
    exactly (chunk-start state is the single reference for all H steps).
    """

    layout: Sequence[dict[str, Any]]

    def __post_init__(self):
        _validate_layout(self.layout)

    def __call__(self, data: _transforms.DataDict) -> _transforms.DataDict:
        if "actions" not in data:
            return data
        state = np.asarray(data["state"])
        actions = np.asarray(data["actions"])
        assert actions.ndim >= 2, (
            f"actions must be (H, D); got {actions.shape}"
        )
        actions = actions.copy()

        for entry in self.layout:
            if entry["kind"] != "pose6d":
                continue
            xyz_slc, r6_slc = entry["xyz"], entry["rot6d"]
            T_state = _state_SE3(state, xyz_slc, r6_slc)
            T_delta = _action_chunk_SE3(actions, xyz_slc, r6_slc)
            T_abs = se3_body_compose(T_state, T_delta)
            _write_SE3_back(actions, T_abs, xyz_slc, r6_slc)

        data["actions"] = actions
        return data
