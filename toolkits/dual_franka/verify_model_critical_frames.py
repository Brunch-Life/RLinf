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
"""Offline check: how well does the SFT model fit handover-critical frames?

Step 2 (offline_eval_rot6d.py) samples each episode uniformly (8 frames out
of ~250) and reports xyz MAE ~4 mm on the 4_28 dataset. That number says
the model is fine *on the average frame* — but handover is a rare event
(~5–10 frames per ep where the two arms meet at midline) and uniform
sampling almost always misses it. A model that reaches the right pose
99 % of the time but blows the handover frames is exactly what we'd see
on the rig: arms move, then meet at the wrong place.

This script picks each episode's handover frame analytically (the index
where ``|L_xyz − R_xyz|`` is minimum in the dataset state) and evaluates
the SFT ckpt on the ±W window around it. If MAE on the critical window
is materially higher than the uniform-sample MAE from step 2, the next
fix is more handover-mode data (or longer SFT), not more code.

Run::

    PYTHONPATH=/home/i-yinuo/cynws/RLinf \\
    /home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \\
        toolkits/dual_franka/verify_model_critical_frames.py
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

from toolkits.dual_franka.backfill_rot6d import (
    L_XYZ_SLICE,
    R_XYZ_SLICE,
    build_rot6d_actions,
    build_rot6d_state,
)


def _decode_image(cell) -> np.ndarray:
    if isinstance(cell, dict):
        return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
    if isinstance(cell, (bytes, bytearray)):
        return np.asarray(Image.open(io.BytesIO(cell)).convert("RGB"))
    return np.asarray(cell)


def _load_episode(path: Path):
    table = pq.read_table(path)
    n = table.num_rows
    state = np.stack(
        [
            np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
            for i in range(n)
        ]
    )
    actions = np.stack(
        [
            np.asarray(table.column("actions")[i].as_py(), dtype=np.float32)
            for i in range(n)
        ]
    )
    return table, state, actions


def _find_handover_frame(state: np.ndarray) -> int:
    """Argmin of two-arm TCP distance in the raw state."""
    L = state[:, L_XYZ_SLICE]  # (T, 3)
    R = state[:, R_XYZ_SLICE]  # (T, 3)
    d = np.linalg.norm(L - R, axis=1)
    return int(d.argmin()), float(d.min())


def _build_critical_samples(parquet_path: Path, window: int) -> list[dict]:
    table, state, actions = _load_episode(parquet_path)
    new_state = build_rot6d_state(state)
    new_actions = build_rot6d_actions(state, actions)
    n = state.shape[0]

    center, min_d = _find_handover_frame(state)
    lo = max(0, center - window)
    hi = min(n - 1, center + window)
    idxs = list(range(lo, hi + 1))

    samples = []
    for idx in idxs:
        samples.append(
            {
                "state": new_state[idx],
                "action_gt": new_actions[idx],
                "image": _decode_image(table.column("image")[idx].as_py()),
                "extra0": _decode_image(
                    table.column("extra_view_image-0")[idx].as_py()
                ),
                "extra1": _decode_image(
                    table.column("extra_view_image-1")[idx].as_py()
                ),
                "frame_index": idx,
                "rel_to_handover": idx - center,
                "two_arm_dist_at_frame": float(
                    np.linalg.norm(state[idx, L_XYZ_SLICE] - state[idx, R_XYZ_SLICE])
                ),
                "handover_center": center,
                "handover_min_dist": min_d,
            }
        )
    return samples


def _build_cfg(model_path: Path, repo_id: str) -> OmegaConf:
    return OmegaConf.create(
        {
            "model_path": str(model_path),
            "precision": "bfloat16",
            "num_action_chunks": 20,
            "action_dim": 20,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_dualfranka_rot6d",
                "train_expert_only": False,
                "detach_critic_input": True,
                "action_chunk": 20,
            },
            "openpi_data": {"repo_id": repo_id},
        }
    )


def _build_env_obs(samples: list[dict], task: str) -> dict:
    states = torch.from_numpy(np.stack([s["state"] for s in samples])).float()
    main = torch.from_numpy(np.stack([s["image"] for s in samples]))
    extra = torch.from_numpy(
        np.stack([np.stack([s["extra0"], s["extra1"]], axis=0) for s in samples])
    )
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


def _action_axis_labels() -> list[str]:
    out = []
    for arm in ("L", "R"):
        for tag in (
            "xyz_x",
            "xyz_y",
            "xyz_z",
            "r6_0",
            "r6_1",
            "r6_2",
            "r6_3",
            "r6_4",
            "r6_5",
            "grip",
        ):
            out.append(f"{arm}_{tag}")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/home/i-yinuo/cynws/RLinf/logs/"
            "merged_dual_franka_cylinder_handover_20260428"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/home/i-yinuo/cynws/RLinf/checkpoints/"
            "dual_franka_cylinder_handover_4_28_rot6d_v1/global_step_18000/"
        ),
    )
    parser.add_argument(
        "--repo-id", default="YinuoTHU/Dual-franka-cylinder-handover-20260428"
    )
    parser.add_argument(
        "--episodes",
        default="0,10,20,31,40,50,60",
        help="Comma-separated episode indices.",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=5,
        help="±W frames around the per-episode handover center.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    if args.task is None:
        with (args.dataset_root / "meta" / "tasks.jsonl").open() as f:
            task = json.loads(f.readline())["task"]
    else:
        task = args.task
    print(f"[task] {task}")

    ep_indices = [int(x) for x in args.episodes.split(",") if x.strip()]
    samples: list[dict] = []
    centers_dump = []
    for ep in ep_indices:
        path = args.dataset_root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        ep_samples = _build_critical_samples(path, args.window)
        for s in ep_samples:
            s["episode_index"] = ep
        samples.extend(ep_samples)
        if ep_samples:
            centers_dump.append(
                (
                    ep,
                    ep_samples[0]["handover_center"],
                    ep_samples[0]["handover_min_dist"],
                )
            )

    if not samples:
        print("[fatal] no samples", file=sys.stderr)
        return 2

    print("[handover centers]")
    for ep, c, d in centers_dump:
        print(f"  ep{ep:3d}: handover_frame={c:3d}  min_two_arm_dist={d * 1000:.1f}mm")
    print(
        f"[samples] N={len(samples)} critical-window frames "
        f"across {len(centers_dump)} episodes (±{args.window} per ep)"
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from rlinf.models.embodiment.openpi import get_model

    cfg = _build_cfg(args.checkpoint, args.repo_id)
    print(f"[model] loading {args.checkpoint} ...")
    t0 = time.time()
    model = get_model(cfg, torch.bfloat16)
    model.eval()
    model.to(args.device)
    print(f"[model] loaded in {time.time() - t0:.1f}s")

    preds, gts, rels = [], [], []
    bs = args.batch_size
    for i in range(0, len(samples), bs):
        batch = samples[i : i + bs]
        env_obs = _build_env_obs(batch, task)
        for k, v in list(env_obs.items()):
            if torch.is_tensor(v):
                env_obs[k] = v.to(args.device)
        with torch.no_grad():
            actions, _ = model.predict_action_batch(
                env_obs, mode="eval", compute_values=False
            )
        preds.append(actions[:, 0, :].float().detach().cpu().numpy())
        gts.append(np.stack([s["action_gt"] for s in batch]))
        rels.extend([s["rel_to_handover"] for s in batch])

    pred = np.concatenate(preds, axis=0)
    gt = np.concatenate(gts, axis=0)
    rels = np.asarray(rels)

    labels = _action_axis_labels()
    err = np.abs(pred - gt)

    print()
    print("=" * 92)
    print(f"Per-axis prediction error on N={pred.shape[0]} HANDOVER-CRITICAL frames")
    print("=" * 92)
    print(f"{'axis':>10s}  {'mae':>9s} {'medae':>9s} {'p95':>9s} {'gt_std':>9s}")
    for k, lab in enumerate(labels):
        print(
            f"{lab:>10s}  {err[:, k].mean():9.4f} {np.median(err[:, k]):9.4f} "
            f"{np.percentile(err[:, k], 95):9.4f} {gt[:, k].std():9.4f}"
        )

    xyz_idx = [0, 1, 2, 10, 11, 12]
    rot_idx = list(range(3, 9)) + list(range(13, 19))
    grip_idx = [9, 19]
    print()
    print(
        f"[critical-window summary] xyz MAE mean={err[:, xyz_idx].mean():.4f}m "
        f"(per-axis={np.array2string(err[:, xyz_idx].mean(0), precision=4, separator=', ')})"
    )
    print(
        f"[critical-window summary] rot6d MAE mean={err[:, rot_idx].mean():.4f} "
        f"max-axis={err[:, rot_idx].mean(0).max():.4f}"
    )
    print(f"[critical-window summary] grip MAE mean={err[:, grip_idx].mean():.4f}")

    # Bin by distance to handover center to see if error grows at the meeting frame
    print()
    print("=" * 92)
    print("Error binned by distance from per-episode handover center")
    print("=" * 92)
    print(
        f"{'rel_to_center':>14s} {'N':>4s} {'xyz_mae_m':>10s} {'rot6d_mae':>10s} {'grip_mae':>10s}"
    )
    for r in sorted({int(x) for x in rels}):
        mask = rels == r
        n = int(mask.sum())
        if n == 0:
            continue
        xyz = err[mask][:, xyz_idx].mean()
        rot = err[mask][:, rot_idx].mean()
        grp = err[mask][:, grip_idx].mean()
        print(f"{r:14d} {n:4d} {xyz:10.4f} {rot:10.4f} {grp:10.4f}")

    print()
    print(
        "Step 2 (uniform sample, 40 frames) reference: xyz MAE 0.0041m, rot6d 0.0093."
    )
    print("If critical-window xyz MAE >> 0.0041m, handover frames are the weak spot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
