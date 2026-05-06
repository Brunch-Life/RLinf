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
"""Replay ``logs/<ts>/demos/trajectory_*.pt`` into a LeRobot dataset.

Recovery path for collect runs that produced ``demos/*.pt`` shards via
``TrajectoryReplayBuffer.auto_save`` but failed to materialize a LeRobot
dataset under ``collected_data/`` (e.g. before the
``KeyboardStartEndWrapper`` ``c``-press regression was fixed: the
terminating frame had ``pre_record=True`` so ``CollectEpisode``'s
``only_success`` gate dropped every episode). The trajectory shards
themselves carry the full ``(state, image, extra_view_image, action,
intervene_flag)`` tuple needed for SFT, so the data is recoverable.

Caveats
-------
* ``rewards / dones / terminations`` in the source ``.pt`` are zero-
  filled — the success-end frame never made it into the buffer (same
  bug). SFT does not use these, so we synthesize ``done`` (last frame)
  and ``is_success`` (always True; the ReplayBuffer only saved
  reward >= 0.5 episodes) instead.
* ``segment_id`` is not preserved in the buffer; every frame is written
  as segment 0. Sub-task boundaries can't be reconstructed offline.

Usage
-----
``rlinf`` is imported below, so run from the repo root::

    export PYTHONPATH=$(pwd)
    python toolkits/dual_franka/pt_demos_to_lerobot.py \\
        --demos-dir logs/20260428-17:07:52/demos \\
        --out-dir   logs/20260428-17:07:52/collected_data_recovered \\
        --task      "pick up the cylinder with the left hand, hand it over to the right hand, and place it on the plate"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

from rlinf.data.lerobot_writer import LeRobotDatasetWriter

_TRAJ_RE = re.compile(r"^trajectory_(\d+)_")


def _discover_pt_shards(demos_dir: Path) -> list[Path]:
    """Return demos/trajectory_<id>_*.pt sorted by integer id."""
    shards: list[tuple[int, Path]] = []
    for p in demos_dir.iterdir():
        if not p.is_file() or p.suffix != ".pt":
            continue
        m = _TRAJ_RE.match(p.name)
        if m is None:
            continue
        shards.append((int(m.group(1)), p))
    shards.sort(key=lambda t: t[0])
    return [p for _, p in shards]


def _squeeze_env(t: torch.Tensor) -> torch.Tensor:
    """Drop the (legacy) num_envs=1 axis at position 1 if present."""
    if t.ndim >= 2 and t.shape[1] == 1:
        return t.squeeze(1)
    return t


def _shard_to_frames(shard: dict, task: str) -> list[dict]:
    actions = _squeeze_env(shard["actions"]).numpy().astype(np.float32)
    intervene = _squeeze_env(shard["intervene_flags"]).numpy().astype(bool)
    obs = shard["curr_obs"]
    state = _squeeze_env(obs["states"]).numpy().astype(np.float32)
    main_img = _squeeze_env(obs["main_images"]).numpy().astype(np.uint8)
    extra = _squeeze_env(obs["extra_view_images"]).numpy().astype(np.uint8)

    T = actions.shape[0]
    if not (state.shape[0] == main_img.shape[0] == extra.shape[0] == T):
        raise ValueError(
            f"length mismatch: actions={T}, state={state.shape[0]}, "
            f"main={main_img.shape[0]}, extra={extra.shape[0]}"
        )

    n_views = extra.shape[1]
    frames: list[dict] = []
    for t in range(T):
        frame: dict = {
            "state": state[t],
            "actions": actions[t],
            "task": task,
            "is_success": np.array([True], dtype=bool),
            "done": np.array([t == T - 1], dtype=bool),
            "intervene_flag": np.array([bool(intervene[t].any())], dtype=bool),
            "image": main_img[t],
        }
        for i in range(n_views):
            frame[f"extra_view_image-{i}"] = extra[t, i]
        frames.append(frame)
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--demos-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--task", type=str, required=True)
    ap.add_argument("--robot-type", type=str, default="dual_FR3")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()

    shards = _discover_pt_shards(args.demos_dir)
    if not shards:
        print(f"No trajectory_*.pt under {args.demos_dir}", file=sys.stderr)
        return 1
    print(f"Found {len(shards)} shards under {args.demos_dir}")

    # LeRobotDataset.create joins repo_id under HF_LEROBOT_HOME when repo_id is
    # not absolute. Force absolute so the dataset lands where the user asked.
    out_dir = args.out_dir.resolve()
    writer = LeRobotDatasetWriter()
    out_dir.mkdir(parents=True, exist_ok=True)

    for ep_idx, shard_path in enumerate(shards):
        shard = torch.load(shard_path, weights_only=False, map_location="cpu")
        frames = _shard_to_frames(shard, task=args.task)
        if writer.dataset is None:
            first = frames[0]
            extra_keys = {
                k: tuple(first[k].shape)
                for k in first
                if k.startswith("extra_view_image-")
            }
            writer.create(
                repo_id=os.path.join(str(out_dir), f"rank_{args.rank}", "id_0"),
                robot_type=args.robot_type,
                fps=args.fps,
                image_shape=tuple(first["image"].shape),
                state_dim=int(first["state"].shape[-1]),
                action_dim=int(first["actions"].shape[-1]),
                has_image=True,
                extra_view_image_keys=extra_keys,
                has_intervene_flag=True,
                has_segment_id=False,
            )
        writer.add_episode(frames)
        print(
            f"  [{ep_idx + 1}/{len(shards)}] {shard_path.name} → {len(frames)} frames"
        )

    writer.finalize()
    print(f"Recovered dataset at {out_dir}/rank_{args.rank}/id_0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
