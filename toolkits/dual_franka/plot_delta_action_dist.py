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
"""Plot the 20-D body-frame delta-action distribution that feeds Normalize.

The pi0.5 SFT pipeline applies ``RigidBodyDeltaActions`` *before* the
``Normalize`` step, so the values that ``norm_stats`` was computed over
are body-frame deltas, not the absolute targets stored on disk. This
script reproduces those deltas from the merged 80-ep joint-collected
dataset (after :mod:`backfill_rot6d`) and overlays the pinned
``norm_stats`` markers (q01, q99, mean, ±1σ).

Two modes are produced side-by-side:
  * ``chunk[0]`` — only the first delta in each chunk (this is the
    head the model is graded against in offline_eval_rot6d.py).
  * ``chunk[0..H-1]`` — every offset 0..H-1 inside the chunk, which is
    what training actually saw and what norm_stats statistics describe.

Run with the openpi venv (matplotlib lives there)::

    PYTHONPATH=/home/i-yinuo/cynws/RLinf \\
    /home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \\
        toolkits/dual_franka/plot_delta_action_dist.py \\
        --out /tmp/delta_action_dist.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from rlinf.models.embodiment.openpi.transforms import (  # noqa: E402
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyDeltaActions,
)
from toolkits.dual_franka.backfill_rot6d import (  # noqa: E402
    build_rot6d_actions,
    build_rot6d_state,
)

ACTION_DIM = 20
PAD_DIM = 32  # match training pad_to_dim


def _policy_layout_state(state_68: np.ndarray) -> np.ndarray:
    """Same as :func:`_rearrange_state` in the policy: env layout → policy layout."""
    s = state_68[..., :ACTION_DIM]
    return np.concatenate(
        [s[..., 2:11], s[..., 0:1], s[..., 11:20], s[..., 1:2]], axis=-1
    )


def _pad_to_dim(arr: np.ndarray, dim: int) -> np.ndarray:
    pad = dim - arr.shape[-1]
    if pad <= 0:
        return arr
    pad_shape = list(arr.shape)
    pad_shape[-1] = pad
    return np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1)


def _episode_deltas(
    state_pol_32: np.ndarray, actions_pol_32: np.ndarray, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return (chunk0_deltas (T,32), all_chunk_deltas (T*H,32)).

    For each frame ``f`` we form a chunk ``actions[f : f+H]`` (clamped at
    the episode tail by repeating the last frame, matching openpi's
    sequence loader behaviour for terminal frames). The reference state
    is ``state[f]``, broadcast across all H deltas, identical to what
    :class:`RigidBodyDeltaActions` does at training time.
    """
    transform = RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)

    T = state_pol_32.shape[0]
    chunk0_out = np.zeros((T, PAD_DIM), dtype=np.float32)
    all_out = np.zeros((T, horizon, PAD_DIM), dtype=np.float32)

    for f in range(T):
        # Pad the chunk with the final action if we're near the tail —
        # same convention as backfill_rot6d.build_rot6d_actions where the
        # last frame holds the current pose.
        end = min(f + horizon, T)
        chunk = actions_pol_32[f:end]  # (h_real, 32)
        if chunk.shape[0] < horizon:
            tail = np.repeat(chunk[-1:], horizon - chunk.shape[0], axis=0)
            chunk = np.concatenate([chunk, tail], axis=0)
        delta_chunk = transform({"state": state_pol_32[f], "actions": chunk.copy()})[
            "actions"
        ]
        chunk0_out[f] = delta_chunk[0]
        all_out[f] = delta_chunk

    return chunk0_out, all_out.reshape(-1, PAD_DIM)


def _action_axis_labels() -> list[str]:
    parts = []
    for arm in ("L", "R"):
        parts.extend([f"{arm}_x", f"{arm}_y", f"{arm}_z"])
        parts.extend([f"{arm}_r6_{i}" for i in range(6)])
        parts.append(f"{arm}_grip")
    return parts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/home/i-yinuo/cynws/RLinf/logs/"
            "merged_dual_franka_cylinder_handover_20260424"
        ),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=Path(
            "/home/i-yinuo/cynws/RLinf/checkpoints/"
            "dual_franka_cylinder_handover_rot6d_v1/global_step_20000/"
            "YinuoTHU/Dual-franka-cylinder-handover-20260424/norm_stats.json"
        ),
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Action chunk length used during SFT (action_horizon).",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/delta_action_dist.png"))
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Cap episode count for a faster pass; default = all.",
    )
    args = parser.parse_args()

    parquets = sorted(
        (args.dataset_root / "data" / "chunk-000").glob("episode_*.parquet")
    )
    if args.max_episodes is not None:
        parquets = parquets[: args.max_episodes]
    print(f"[load] {len(parquets)} episodes")

    chunk0_chunks: list[np.ndarray] = []
    full_chunks: list[np.ndarray] = []
    for p in parquets:
        table = pq.read_table(p, columns=["state", "actions"])
        state = np.stack(
            [
                np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
                for i in range(table.num_rows)
            ]
        )
        actions = np.stack(
            [
                np.asarray(table.column("actions")[i].as_py(), dtype=np.float32)
                for i in range(table.num_rows)
            ]
        )
        new_state_68 = build_rot6d_state(state)
        new_actions_20 = build_rot6d_actions(state, actions)
        state_pol_20 = _policy_layout_state(new_state_68)
        state_pol_32 = _pad_to_dim(state_pol_20, PAD_DIM)
        actions_pol_32 = _pad_to_dim(new_actions_20, PAD_DIM)

        c0, full = _episode_deltas(state_pol_32, actions_pol_32, args.horizon)
        chunk0_chunks.append(c0)
        full_chunks.append(full)

    chunk0 = np.concatenate(chunk0_chunks, axis=0)[:, :ACTION_DIM]
    fullchunk = np.concatenate(full_chunks, axis=0)[:, :ACTION_DIM]
    print(f"[deltas] chunk[0] N={chunk0.shape[0]}, full chunk N={fullchunk.shape[0]}")

    # ---- norm_stats
    nstats = json.loads(args.norm_stats.read_text())["norm_stats"]["actions"]
    q01 = np.asarray(nstats["q01"])[:ACTION_DIM]
    q99 = np.asarray(nstats["q99"])[:ACTION_DIM]
    mean = np.asarray(nstats["mean"])[:ACTION_DIM]
    std = np.asarray(nstats["std"])[:ACTION_DIM]

    labels = _action_axis_labels()
    fig, axes = plt.subplots(4, 5, figsize=(22, 14), sharey=False)
    axes = axes.ravel()
    for k in range(ACTION_DIM):
        ax = axes[k]
        # X range = union of (chunk[0..H] range) and norm-stat (q01,q99) — so
        # the markers always fit and the chunk[0..H] spread is fully visible.
        x_lo = min(fullchunk[:, k].min(), q01[k]) - 1e-3
        x_hi = max(fullchunk[:, k].max(), q99[k]) + 1e-3
        bin_edges = np.linspace(x_lo, x_hi, args.bins + 1)
        ax.hist(
            fullchunk[:, k],
            bins=bin_edges,
            color="#4878D0",
            alpha=0.55,
            label=f"chunk[0..{args.horizon - 1}]" if k == 0 else None,
        )
        ax.hist(
            chunk0[:, k],
            bins=bin_edges,
            color="#EE854A",
            alpha=0.65,
            label="chunk[0]" if k == 0 else None,
        )
        ax.set_yscale("log")
        ax.set_xlim(x_lo, x_hi)
        # norm_stats markers.
        ax.axvline(
            q01[k],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="q01/q99" if k == 0 else None,
        )
        ax.axvline(q99[k], color="black", linestyle="--", linewidth=1.0)
        ax.axvline(
            mean[k],
            color="red",
            linestyle="-",
            linewidth=1.0,
            label="mean" if k == 0 else None,
        )
        ax.axvline(
            mean[k] - std[k],
            color="red",
            linestyle=":",
            linewidth=0.9,
            label="mean±σ" if k == 0 else None,
        )
        ax.axvline(mean[k] + std[k], color="red", linestyle=":", linewidth=0.9)

        # Coverage stat — what fraction of the full chunk is inside q01..q99.
        inside_full = ((fullchunk[:, k] >= q01[k]) & (fullchunk[:, k] <= q99[k])).mean()
        inside_c0 = ((chunk0[:, k] >= q01[k]) & (chunk0[:, k] <= q99[k])).mean()
        title = (
            f"{labels[k]}\n"
            f"q01={q01[k]:+.3f}  q99={q99[k]:+.3f}\n"
            f"chunk[0..H] inside={inside_full * 100:.1f}%  "
            f"chunk[0] inside={inside_c0 * 100:.1f}%"
        )
        ax.set_title(title, fontsize=9)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        "Body-frame delta-action distribution vs pinned norm_stats\n"
        f"(80-ep merged, horizon={args.horizon}, deltas via RigidBodyDeltaActions)",
        fontsize=12,
    )
    # Single legend in the figure top-right.
    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="upper right", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120)
    print(f"[plot] wrote {args.out}")

    # Print a short tabular summary so the terminal output stands alone.
    print()
    print(
        f"{'axis':>10s}  {'chunk[0] min/max':>26s}  {'chunk[0..H] min/max':>26s}  "
        f"{'q01':>9s}  {'q99':>9s}  {'mean':>9s}  {'std':>9s}"
    )
    for k, lab in enumerate(labels):
        c0min, c0max = chunk0[:, k].min(), chunk0[:, k].max()
        fmin, fmax = fullchunk[:, k].min(), fullchunk[:, k].max()
        print(
            f"{lab:>10s}  {c0min:+10.4f} / {c0max:+10.4f}  "
            f"{fmin:+10.4f} / {fmax:+10.4f}  "
            f"{q01[k]:+9.4f}  {q99[k]:+9.4f}  {mean[k]:+9.4f}  {std[k]:+9.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
