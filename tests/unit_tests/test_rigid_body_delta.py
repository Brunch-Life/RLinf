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
"""Round-trip tests for RigidBodyDeltaActions / RigidBodyAbsoluteActions."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)
from rlinf.utils.rot6d import matrix_to_rot6d

RNG = np.random.default_rng(0)
H = 20  # action chunk length (matches pi05 action_horizon)
D_REAL = 20  # dual-arm real dim before pad
D_PAD = 32  # typical pi05 pad


def _random_dual_state(rng):
    """Build a single chunk-start state vector (D_PAD,) with random poses."""
    state = np.zeros(D_PAD, dtype=np.float32)
    for xyz_slc, r6_slc in (
        (slice(0, 3), slice(3, 9)),
        (slice(10, 13), slice(13, 19)),
    ):
        xyz = rng.normal(size=3) * 0.3
        R_mat = R.random(random_state=rng.integers(1 << 31)).as_matrix()
        r6 = matrix_to_rot6d(R_mat)
        state[xyz_slc] = xyz
        state[r6_slc] = r6
    state[9] = rng.uniform(-1.0, 1.0)  # L grip
    state[19] = rng.uniform(-1.0, 1.0)  # R grip
    return state


def _random_dual_actions(rng):
    """Build an (H, D_PAD) action chunk with random absolute pose6d targets."""
    actions = np.zeros((H, D_PAD), dtype=np.float32)
    for xyz_slc, r6_slc in (
        (slice(0, 3), slice(3, 9)),
        (slice(10, 13), slice(13, 19)),
    ):
        xyz = rng.normal(size=(H, 3)) * 0.3
        mats = R.random(H, random_state=rng.integers(1 << 31)).as_matrix()
        r6 = matrix_to_rot6d(mats)
        actions[:, xyz_slc] = xyz
        actions[:, r6_slc] = r6
    actions[:, 9] = rng.uniform(-1.0, 1.0, size=H)
    actions[:, 19] = rng.uniform(-1.0, 1.0, size=H)
    return actions


def test_delta_then_absolute_round_trip():
    """Random (state, abs chunk) → delta → absolute should recover chunk."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    absolute = RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)

    rng = np.random.default_rng(1)
    for _ in range(50):
        state = _random_dual_state(rng)
        actions_abs = _random_dual_actions(rng)

        data = {"state": state.copy(), "actions": actions_abs.copy()}
        data = delta(data)
        # Sanity: pose6d slots are now small deltas, not raw abs
        assert not np.allclose(data["actions"][:, :3], actions_abs[:, :3])
        data = absolute(data)
        # Recovered chunk matches original absolute targets on real slots
        assert np.allclose(data["actions"][:, :D_REAL], actions_abs[:, :D_REAL], atol=1e-5)


def test_delta_preserves_gripper_absolute():
    """Gripper channels are scalar_abs → untouched by both transforms."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    absolute = RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)

    rng = np.random.default_rng(2)
    state = _random_dual_state(rng)
    actions_abs = _random_dual_actions(rng)

    out = delta({"state": state.copy(), "actions": actions_abs.copy()})
    assert np.allclose(out["actions"][:, 9], actions_abs[:, 9])
    assert np.allclose(out["actions"][:, 19], actions_abs[:, 19])

    out2 = absolute(out)
    assert np.allclose(out2["actions"][:, 9], actions_abs[:, 9])
    assert np.allclose(out2["actions"][:, 19], actions_abs[:, 19])


def test_delta_untouched_pad_tail():
    """Channels past D_REAL (the pad_to_dim tail) must be left alone."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    rng = np.random.default_rng(3)
    state = _random_dual_state(rng)
    actions_abs = _random_dual_actions(rng)
    # Write a non-zero marker in the pad region to catch accidental writes
    actions_abs[:, D_REAL:] = 0.42
    out = delta({"state": state.copy(), "actions": actions_abs.copy()})
    assert np.allclose(out["actions"][:, D_REAL:], 0.42)


def test_delta_identity_when_abs_equals_state():
    """If abs action == state, body-frame delta should be identity."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    rng = np.random.default_rng(4)
    state = _random_dual_state(rng)
    # Broadcast state to a chunk of identical targets (grippers keep their value too)
    actions_abs = np.broadcast_to(state, (H, D_PAD)).copy()
    out = delta({"state": state.copy(), "actions": actions_abs.copy()})
    # Delta xyz → 0
    for xyz_slc in (slice(0, 3), slice(10, 13)):
        assert np.allclose(out["actions"][:, xyz_slc], 0.0, atol=1e-6)
    # Delta rot6d: first column [1,0,0], second [0,1,0] → [1,0,0,0,1,0]
    identity_rot6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    for r6_slc in (slice(3, 9), slice(13, 19)):
        assert np.allclose(out["actions"][:, r6_slc], identity_rot6d, atol=1e-6)


def test_delta_rejects_malformed_layout():
    bad = ({"kind": "scalar_delta", "idx": 5},)  # unsupported kind
    with pytest.raises(AssertionError):
        RigidBodyDeltaActions(bad)


def test_noop_when_actions_missing():
    """Inference time may pass state only (no actions key)."""
    delta = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)
    data = {"state": np.zeros(D_PAD)}
    assert delta(data) is data
