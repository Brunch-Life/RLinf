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
"""Offline round-trip verification of dual-Franka openpi dataset transforms.

Debug step 1: rule out the dataset transform pipeline. Loads one
LeRobot parquet, exercises both SFT data paths end-to-end as numpy:

  * **joint** — pad_to_dim(32) → DualFrankaJointOutputs[:, :16]; should
    be bit-exact (no delta, just pad+slice).
  * **rot6d** — in-memory backfill via toolkits/dual_franka/backfill_rot6d.py
    (the joint-collected dataset is the only thing on disk), then
    rearrange_state → pad_to_dim(32) → RigidBodyDeltaActions →
    RigidBodyAbsoluteActions → DualFrankaRot6dOutputs[:, :20].
    Residuals come from float64↔float32 + Gram-Schmidt rebuild only.

If everything PASSes here, the transform layer is exonerated and the
next step is model inference (step 2).

Run::

    source .venv-openpi/bin/activate
    PYTHONPATH=$(pwd) python toolkits/dual_franka/verify_transform_round_trip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from openpi import transforms as _transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.openpi.policies import (  # noqa: E402
    dual_franka_joint_policy,
    dual_franka_rot6d_policy,
)
from rlinf.models.embodiment.openpi.transforms.rigid_body_delta import (  # noqa: E402
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)
from toolkits.dual_franka.backfill_rot6d import (  # noqa: E402
    build_rot6d_actions,
    build_rot6d_state,
)

PARQUET = (
    REPO_ROOT
    / "logs/merged_dual_franka_cylinder_handover_20260428"
    / "data/chunk-000/episode_000031.parquet"
)
ACTION_DIM = 32  # pi0 default; mirrors training config
HORIZON = 20  # SFT action_horizon
NUM_CHUNKS = 5


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _fsl_to_numpy(col, dim: int) -> np.ndarray:
    flat = np.asarray(col.combine_chunks().values.to_numpy(zero_copy_only=False))
    return flat.reshape(-1, dim).astype(np.float32, copy=False)


def load_episode(path: Path):
    table = pq.read_table(path)
    state = _fsl_to_numpy(table.column("state"), 68)
    actions = _fsl_to_numpy(table.column("actions"), 16)
    return state, actions


# ---------------------------------------------------------------------------
# Joint round-trip
# ---------------------------------------------------------------------------


def joint_round_trip(actions_16: np.ndarray, i: int, H: int):
    """Joint actions: pad → slice. Should be bit-exact."""
    chunk = actions_16[i : i + H].astype(np.float32, copy=True)
    padded = _transforms.pad_to_dim(chunk, ACTION_DIM)  # (H, 32)
    assert padded.shape == (H, ACTION_DIM)
    out = dual_franka_joint_policy.DualFrankaJointOutputs()({"actions": padded})
    recovered = out["actions"]
    assert recovered.shape == (H, 16)
    diff = recovered.astype(np.float64) - chunk.astype(np.float64)
    return chunk, recovered, diff


# ---------------------------------------------------------------------------
# Rot6d round-trip
# ---------------------------------------------------------------------------


# Channel layout in the policy-side 20-d action (after rearrange_state /
# matches actions on disk): [L_xyz(3), L_rot6d(6), L_grip(1),
#                            R_xyz(3), R_rot6d(6), R_grip(1)].
_CHAN_GROUPS = {
    "xyz": np.r_[0:3, 10:13],
    "rot6d": np.r_[3:9, 13:19],
    "grip": np.array([9, 19]),
}


def rot6d_round_trip(
    state_68_rot6d: np.ndarray, actions_20: np.ndarray, i: int, H: int
):
    """Run actions through Inputs (rearrange+pad) → Delta → Absolute → Outputs."""
    state_env = state_68_rot6d[i]  # env-side, 68-d, prefix is rot6d layout
    state_pol = dual_franka_rot6d_policy._rearrange_state(state_env)  # (20,)
    state_padded = _transforms.pad_to_dim(state_pol, ACTION_DIM)  # (32,)

    chunk_20 = actions_20[i : i + H].astype(np.float32, copy=True)
    actions_padded = _transforms.pad_to_dim(chunk_20, ACTION_DIM)  # (H, 32)

    # Forward: absolute targets → body-frame SE(3) deltas.
    forward = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": state_padded.copy(), "actions": actions_padded.copy()}
    )
    deltas = forward["actions"]

    # Inverse: deltas → absolute targets, with the SAME chunk-start state.
    backward = RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)(
        {"state": state_padded.copy(), "actions": deltas.copy()}
    )
    out = dual_franka_rot6d_policy.DualFrankaRot6dOutputs()(backward)
    recovered = out["actions"]
    assert recovered.shape == (H, 20)

    diff = recovered.astype(np.float64) - chunk_20.astype(np.float64)
    return chunk_20, recovered, diff


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def channel_stats(diff: np.ndarray, channel_idx: np.ndarray) -> dict:
    """abs-residual max / mean / p95 over the selected channels."""
    sub = np.abs(diff[..., channel_idx])
    return {
        "max": float(sub.max()) if sub.size else 0.0,
        "mean": float(sub.mean()) if sub.size else 0.0,
        "p95": float(np.percentile(sub, 95)) if sub.size else 0.0,
    }


def _argmax_unravel(arr: np.ndarray) -> tuple[int, int]:
    flat = int(arr.argmax())
    H, D = arr.shape
    return flat // D, flat % D


def _print_chunk_report(
    name: str, i: int, diff: np.ndarray, groups: dict[str, np.ndarray]
):
    parts = [f"{name} i={i:4d}"]
    for ch, idx in groups.items():
        s = channel_stats(diff, idx)
        parts.append(
            f"{ch}: max={s['max']:.3e} mean={s['mean']:.3e} p95={s['p95']:.3e}"
        )
    print("  " + " | ".join(parts))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    print(f"loading {PARQUET}")
    state_68, actions_16 = load_episode(PARQUET)
    T = state_68.shape[0]
    print(f"  T={T}  state[68]  actions[16]")
    if T < HORIZON + 1:
        print(f"episode too short for H={HORIZON}", file=sys.stderr)
        return 1

    # Backfill rot6d in memory (mirrors backfill_rot6d.py output exactly).
    state_68_r6 = build_rot6d_state(state_68)
    actions_20 = build_rot6d_actions(state_68, actions_16)
    print("  backfilled: state[20]+rest, actions[20]")

    # 5 chunk starts spanning episode phases.
    starts = np.linspace(0, T - HORIZON, NUM_CHUNKS, dtype=int).tolist()
    print(f"  chunk starts: {starts}  H={HORIZON}")

    # ---- joint ----
    print("\n[joint] pad → slice round-trip (expect bit-exact)")
    joint_pass = True
    joint_max = 0.0
    for i in starts:
        chunk, recovered, diff = joint_round_trip(actions_16, i, HORIZON)
        max_abs = float(np.abs(diff).max())
        joint_max = max(joint_max, max_abs)
        # Whole 16-d block; report as a single channel.
        s = channel_stats(diff, np.arange(16))
        print(
            f"  joint i={i:4d} | all16: max={s['max']:.3e} mean={s['mean']:.3e} p95={s['p95']:.3e}"
        )
        if max_abs != 0.0:
            joint_pass = False
            t_step, c_idx = _argmax_unravel(np.abs(diff))
            print(f"    ! non-zero residual at H-step={t_step} channel={c_idx}")
            print(
                f"      orig={chunk[t_step, c_idx]!r}  rec={recovered[t_step, c_idx]!r}"
            )
    print(f"[joint] worst max-abs residual: {joint_max:.3e}")

    # ---- rot6d ----
    print("\n[rot6d] rearrange+pad → delta → absolute → slice round-trip")
    print("  budget: xyz<1e-6 m, rot6d<1e-5 (any chan), grip == 0")
    rot6d_pass = True
    rot6d_worst = {"xyz": 0.0, "rot6d": 0.0, "grip": 0.0}
    rot6d_budget = {"xyz": 1e-6, "rot6d": 1e-5, "grip": 0.0}
    for i in starts:
        chunk, recovered, diff = rot6d_round_trip(state_68_r6, actions_20, i, HORIZON)
        _print_chunk_report("rot6d", i, diff, _CHAN_GROUPS)
        for ch, idx in _CHAN_GROUPS.items():
            sub = np.abs(diff[..., idx])
            mx = float(sub.max()) if sub.size else 0.0
            rot6d_worst[ch] = max(rot6d_worst[ch], mx)
            if mx > rot6d_budget[ch]:
                rot6d_pass = False
                # locate the offending step+channel for diagnosis
                rel = int(np.unravel_index(sub.argmax(), sub.shape)[0])
                col_local = int(np.unravel_index(sub.argmax(), sub.shape)[1])
                col = int(idx[col_local])
                print(
                    f"    ! {ch} budget exceeded at chunk i={i} H-step={rel} "
                    f"global-col={col}: |diff|={mx:.3e} > {rot6d_budget[ch]:.0e}"
                )
                print(f"      orig={chunk[rel, col]!r}  rec={recovered[rel, col]!r}")
    print(
        "[rot6d] worst per-channel: "
        + ", ".join(f"{k}={v:.3e}" for k, v in rot6d_worst.items())
    )

    print()
    print(f"joint:  {'PASS' if joint_pass else 'FAIL'}")
    print(f"rot6d:  {'PASS' if rot6d_pass else 'FAIL'}")
    return 0 if (joint_pass and rot6d_pass) else 2


if __name__ == "__main__":
    sys.exit(main())
