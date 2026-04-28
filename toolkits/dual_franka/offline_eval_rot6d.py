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
"""Offline-replay diagnostic for dual-Franka rot6d SFT.

Loads N frames from a joint-space LeRobot dataset (the 80-ep merged
collection), backfills them into the rot6d_v1 schema on the fly, runs
them through the trained openpi pi0.5 checkpoint, and compares the
predicted absolute action to the dataset-side ground-truth target.

Focus: per-axis (especially XYZ) prediction error, and whether the live
state is in-range against the pinned ``norm_stats.json``.

Run with the openpi venv (the same one ``run_realworld_eval.sh`` uses)::

    PYTHONPATH=/home/i-yinuo/cynws/RLinf \
    /home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \
        toolkits/dual_franka/offline_eval_rot6d.py \
        --num-samples 32

Args control sample count, dataset path, checkpoint, and dump path.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from PIL import Image

# Local utils — run from repo root with PYTHONPATH=$REPO_PATH.
from toolkits.dual_franka.backfill_rot6d import (  # noqa: E402
    build_rot6d_actions,
    build_rot6d_state,
)

ARM_NAMES = ("L", "R")
PER_ARM_SLOTS = (
    ("xyz_x", 0),
    ("xyz_y", 1),
    ("xyz_z", 2),
    ("r6_0", 3),
    ("r6_1", 4),
    ("r6_2", 5),
    ("r6_3", 6),
    ("r6_4", 7),
    ("r6_5", 8),
    ("grip", 9),
)


def _action_axis_labels() -> list[str]:
    out = []
    for arm in ARM_NAMES:
        for name, _ in PER_ARM_SLOTS:
            out.append(f"{arm}_{name}")
    return out


def _state_axis_labels() -> list[str]:
    # env STATE_LAYOUT = (gripper_position, tcp_pose_rot6d) →
    # [L_grip, R_grip, L_xyz(3), L_rot6d(6), R_xyz(3), R_rot6d(6)]
    out = ["L_grip", "R_grip"]
    for axis in ("x", "y", "z"):
        out.append(f"L_xyz_{axis}")
    for i in range(6):
        out.append(f"L_r6_{i}")
    for axis in ("x", "y", "z"):
        out.append(f"R_xyz_{axis}")
    for i in range(6):
        out.append(f"R_r6_{i}")
    return out


def _decode_image(cell) -> np.ndarray:
    """LeRobot stores PIL-encoded bytes under 'image*' columns."""
    if isinstance(cell, dict):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    if isinstance(cell, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(cell)).convert("RGB"))
    return np.asarray(cell)


def _load_episode_samples(parquet_path: Path, frame_indices: list[int]) -> dict:
    table = pq.read_table(parquet_path)
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

    new_state = build_rot6d_state(state)  # (T, 68), prefix rewritten
    new_actions = build_rot6d_actions(state, actions)  # (T, 20) policy layout

    samples = []
    for idx in frame_indices:
        if idx >= table.num_rows:
            continue
        img_main = _decode_image(table.column("image")[idx].as_py())
        img_extra0 = _decode_image(table.column("extra_view_image-0")[idx].as_py())
        img_extra1 = _decode_image(table.column("extra_view_image-1")[idx].as_py())
        samples.append(
            {
                "state": new_state[idx],  # (68,) — policy slices :20
                "action_gt": new_actions[idx],  # (20,) policy layout
                "image": img_main,  # left wrist
                "extra0": img_extra0,  # base
                "extra1": img_extra1,  # right wrist
                "frame_index": idx,
            }
        )
    return samples


def _build_cfg(
    model_path: Path, repo_id: str, action_dim: int = 20, num_action_chunks: int = 20
) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "precision": "bfloat16",
            "num_action_chunks": num_action_chunks,
            "action_dim": action_dim,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_dualfranka_rot6d",
                "train_expert_only": False,
                "detach_critic_input": True,
                "action_chunk": num_action_chunks,
            },
            "openpi_data": {
                "repo_id": repo_id,
            },
        }
    )


def _build_env_obs(samples: list[dict], task: str) -> dict:
    """Match the keys produced by ``RealWorldEnv._wrap_obs`` (then read by
    ``OpenPi0...obs_processor``).
    """
    states = torch.from_numpy(np.stack([s["state"] for s in samples])).float()
    main = torch.from_numpy(np.stack([s["image"] for s in samples]))  # (B, H, W, 3)
    extra = torch.from_numpy(
        np.stack(
            [
                np.stack([s["extra0"], s["extra1"]], axis=0)  # (2, H, W, 3)
                for s in samples
            ]
        )
    )  # (B, 2, H, W, 3)
    return {
        "states": states,
        "main_images": main,
        "wrist_images": None,
        "extra_view_images": extra,
        "extra_view_image_names": [
            ("base_0_rgb", "right_wrist_0_rgb") for _ in samples
        ],
        "task_descriptions": [task] * len(samples),
    }


def _check_state_against_norm_stats(states_used: np.ndarray, norm_stats: dict) -> dict:
    """Per-axis fraction of frames inside [q01, q99]."""
    q01 = np.asarray(norm_stats["state"]["q01"])
    q99 = np.asarray(norm_stats["state"]["q99"])
    mean = np.asarray(norm_stats["state"]["mean"])
    std = np.asarray(norm_stats["state"]["std"])

    # _rearrange_state — same as policy: env [L_grip, R_grip, L_xyz, L_rot6d,
    # R_xyz, R_rot6d] → policy [L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]
    s = states_used[:, :20]
    rearr = np.concatenate([s[:, 2:11], s[:, 0:1], s[:, 11:20], s[:, 1:2]], axis=-1)
    # pad to 32 (norm_stats are 32-D).
    pad = np.zeros((rearr.shape[0], 32 - rearr.shape[1]), dtype=rearr.dtype)
    rearr32 = np.concatenate([rearr, pad], axis=-1)

    inside = (rearr32 >= q01) & (rearr32 <= q99)
    frac = inside.mean(axis=0)

    return {
        "axis_inside_frac": frac.tolist(),
        "state_min": rearr32.min(axis=0).tolist(),
        "state_max": rearr32.max(axis=0).tolist(),
        "norm_q01": q01.tolist(),
        "norm_q99": q99.tolist(),
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
    }


def _fmt_row(label: str, vec: np.ndarray, fmt: str = "{:+.4f}") -> str:
    return f"{label:>12s}: " + " ".join(fmt.format(v) for v in vec)


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
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/i-yinuo/cynws/RLinf/checkpoints/"
            "dual_franka_cylinder_handover_rot6d_v1/global_step_20000/"
        ),
    )
    parser.add_argument(
        "--repo-id", default="YinuoTHU/Dual-franka-cylinder-handover-20260424"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Total frames to evaluate (spread across episodes).",
    )
    parser.add_argument(
        "--episodes",
        type=str,
        default="0,1,2,3",
        help="Comma-separated episode indices to draw from.",
    )
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=8,
        help="Per-episode frame count (overrides --num-samples).",
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Forward batch size.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dump",
        type=Path,
        default=None,
        help="Optional .npz file to dump (gt, pred, state).",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description (override). Defaults to the one in tasks.jsonl.",
    )
    args = parser.parse_args()

    # ---- read task from meta if not overridden
    if args.task is None:
        tasks_jsonl = args.dataset_root / "meta" / "tasks.jsonl"
        with tasks_jsonl.open() as f:
            task = json.loads(f.readline())["task"]
    else:
        task = args.task
    print(f"[task] {task}")

    # ---- gather frames
    ep_indices = [int(x) for x in args.episodes.split(",") if x.strip()]
    samples: list[dict] = []
    for ep in ep_indices:
        path = args.dataset_root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        # Pick frames evenly across this episode.
        n_rows = pq.read_table(path, columns=["state"]).num_rows
        n_take = args.frames_per_episode
        idxs = np.linspace(0, max(0, n_rows - 1), n_take).round().astype(int).tolist()
        ep_samples = _load_episode_samples(path, idxs)
        for s in ep_samples:
            s["episode_index"] = ep
        samples.extend(ep_samples)

    if not samples:
        print("[fatal] no samples loaded", file=sys.stderr)
        return 2
    print(f"[samples] N={len(samples)} across episodes {ep_indices}")

    # ---- build model
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rlinf.models.embodiment.openpi import get_model

    cfg = _build_cfg(args.checkpoint, args.repo_id)
    print(f"[model] loading {args.checkpoint} ...")
    t0 = time.time()
    model = get_model(cfg, torch.bfloat16)
    model.eval()
    model.to(args.device)
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    # Read pinned norm_stats from disk (same file get_model uses).
    nstats_disk_path = args.checkpoint / args.repo_id / "norm_stats.json"
    nstats_disk = json.loads(nstats_disk_path.read_text())["norm_stats"]

    # ---- forward pass in batches
    preds: list[np.ndarray] = []
    gts: list[np.ndarray] = []
    states_seen: list[np.ndarray] = []
    bs = args.batch_size
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        env_obs = _build_env_obs(batch, task)
        # move tensors to device
        for k, v in list(env_obs.items()):
            if torch.is_tensor(v):
                env_obs[k] = v.to(args.device)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(
                env_obs, mode="eval", compute_values=False
            )
        # actions shape (B, action_chunk, 20). The dataset GT is the per-frame
        # next-step target → compare to chunk[0].
        pred = actions[:, 0, :].float().detach().cpu().numpy()
        gt = np.stack([s["action_gt"] for s in batch])  # (B, 20)
        preds.append(pred)
        gts.append(gt)
        states_seen.append(np.stack([s["state"] for s in batch]))

    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    states_seen_arr = np.concatenate(states_seen, axis=0)

    # ---- report
    labels = _action_axis_labels()
    abs_err = np.abs(pred - gt)
    mae = abs_err.mean(axis=0)
    medae = np.median(abs_err, axis=0)
    p95 = np.percentile(abs_err, 95, axis=0)
    bias = (pred - gt).mean(axis=0)
    gt_std = gt.std(axis=0) + 1e-9
    nrmse = np.sqrt(((pred - gt) ** 2).mean(axis=0)) / gt_std

    print()
    print("=" * 96)
    print(
        f"Per-axis prediction error (N={pred.shape[0]} frames, action layout policy-side):"
    )
    print("=" * 96)
    print(
        f"{'axis':>12s}  {'gt_mean':>9s} {'gt_std':>9s} {'pred_mean':>9s} "
        f"{'bias':>9s} {'mae':>9s} {'medae':>9s} {'p95':>9s} {'nrmse':>7s}"
    )
    for k, lab in enumerate(labels):
        print(
            f"{lab:>12s}  {gt[:, k].mean():+9.4f} {gt[:, k].std():9.4f} "
            f"{pred[:, k].mean():+9.4f} {bias[k]:+9.4f} {mae[k]:9.4f} "
            f"{medae[k]:9.4f} {p95[k]:9.4f} {nrmse[k]:7.3f}"
        )

    # XYZ-only summary (the user's primary concern)
    xyz_idx = [0, 1, 2, 10, 11, 12]
    rot_idx = list(range(3, 9)) + list(range(13, 19))
    grip_idx = [9, 19]
    print()
    print(
        f"[summary] XYZ  : MAE mean={mae[xyz_idx].mean():.4f}  per-axis="
        f"{np.array2string(mae[xyz_idx], precision=4, separator=', ')}"
    )
    print(
        f"[summary] rot6d: MAE mean={mae[rot_idx].mean():.4f}  max="
        f"{mae[rot_idx].max():.4f} (axis={labels[rot_idx[int(mae[rot_idx].argmax())]]})"
    )
    print(f"[summary] grip : MAE mean={mae[grip_idx].mean():.4f}")

    # ---- norm_stats range check
    print()
    print("=" * 96)
    print(
        "State vs pinned norm_stats q01..q99 (after _rearrange_state to policy layout):"
    )
    print("=" * 96)
    info = _check_state_against_norm_stats(states_seen_arr, nstats_disk)
    state_labels_pol = (
        [f"L_xyz_{a}" for a in ("x", "y", "z")]
        + [f"L_r6_{i}" for i in range(6)]
        + ["L_grip"]
        + [f"R_xyz_{a}" for a in ("x", "y", "z")]
        + [f"R_r6_{i}" for i in range(6)]
        + ["R_grip"]
        + [f"pad_{i}" for i in range(12)]
    )
    print(
        f"{'axis':>12s}  {'state_min':>10s} {'state_max':>10s} "
        f"{'q01':>10s} {'q99':>10s} {'inside':>7s}"
    )
    for k in range(20):
        flag = " " if info["axis_inside_frac"][k] >= 0.95 else "!"
        print(
            f"{state_labels_pol[k]:>12s}  {info['state_min'][k]:+10.4f} "
            f"{info['state_max'][k]:+10.4f} {info['norm_q01'][k]:+10.4f} "
            f"{info['norm_q99'][k]:+10.4f} {info['axis_inside_frac'][k] * 100:6.1f}% {flag}"
        )

    # ---- norm_stats span warning: if std/q-spread are tiny, normalization
    # blows tiny errors up to dominant magnitudes.
    print()
    print("[norm_stats spread] action span (q99-q01):")
    aspan = (
        np.asarray(nstats_disk["actions"]["q99"])
        - np.asarray(nstats_disk["actions"]["q01"])
    )[:20]
    for k, lab in enumerate(labels):
        print(f"  {lab:>12s}  span={aspan[k]:+.4f}")

    if args.dump is not None:
        args.dump.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            args.dump,
            pred=pred,
            gt=gt,
            state=states_seen_arr,
            labels=np.array(labels),
            norm_stats_path=str(nstats_disk_path),
        )
        print(f"[dump] wrote {args.dump}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
