# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
"""Rewrite the dual-Franka GELLO dataset to train pi05 on TCP-pose targets.

Both state and action are rewritten into an EE-pose layout; the actual
delta subtraction is then performed at training time by the pipeline's
``DeltaActions`` transform (mask ``[True*7, False, True*7, False]``). To enable
it, flip ``extra_delta_transform=True`` on the ``pi05_dualfranka`` TrainConfig.
We intentionally *do not* precompute the delta in preprocessing so the on-disk
actions column stays in the standard openpi LIBERO-style "absolute target"
convention, and the gripper-trigger channels are kept absolute through the
``False`` slots of the delta mask.

State layout after rewrite (``state[:16]``):
    [0]=L_grip_width, [1]=R_grip_width,
    [2:5]=L_xyz, [5:8]=L_euler_xyz, [8]=0,
    [9:12]=R_xyz, [12:15]=R_euler_xyz, [15]=0.
Slots ``[16:68]`` are preserved as-is (unused by pi05).

Action layout after rewrite (16 dims, matches the existing ``_rearrange_state``
output so no policy-side change is required):
    [0:3]=L_xyz_target, [3:6]=L_euler_xyz_target, [6]=0, [7]=L_grip_trigger,
    [8:11]=R_xyz_target, [11:14]=R_euler_xyz_target, [14]=0, [15]=R_grip_trigger.
The target is ``state[t+1]``; last frame copies ``state[t]`` so the chunk-start
``DeltaActions`` subtraction yields zero at the episode boundary.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

# Slot offsets inside the raw 68-d state vector.
LGRIP = slice(0, 1)
RGRIP = slice(1, 2)
TCP_POSE = slice(36, 50)  # L_xyz(3)+L_quat(4)+R_xyz(3)+R_quat(4), quat=xyzw
LEFT_TCP = slice(36, 43)
RIGHT_TCP = slice(43, 50)


def _quat_to_euler_xyz(quat_xyzw: np.ndarray) -> np.ndarray:
    return R.from_quat(quat_xyzw).as_euler("xyz")


def _abs_ee_pose_7d(tcp_pose_7d: np.ndarray) -> np.ndarray:
    """Convert one arm's 7-d [xyz + quat_xyzw] to 7-d [xyz + euler_xyz + 0]."""
    out = np.zeros(7, dtype=np.float32)
    out[0:3] = tcp_pose_7d[0:3]
    out[3:6] = _quat_to_euler_xyz(tcp_pose_7d[3:7])
    # out[6] = 0  # padding
    return out


def _fixed_size_list_array(values: np.ndarray) -> pa.Array:
    """Build a pyarrow FixedSizeListArray[float32, d] from an (n, d) numpy array."""
    assert values.ndim == 2
    n, d = values.shape
    flat = pa.array(values.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, d)


def _rewrite_episode(src_path: Path, dst_path: Path) -> int:
    table = pq.read_table(src_path)

    states = np.asarray(table["state"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32)
    actions = np.asarray(table["actions"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32)
    n = states.shape[0]
    assert states.shape[1] == 68, f"unexpected state dim {states.shape[1]} in {src_path}"
    assert actions.shape[1] == 16, f"unexpected action dim {actions.shape[1]} in {src_path}"

    # --- Build new state (copy first, then overwrite [0:16]).
    new_states = states.copy()
    new_states[:, 0:1] = states[:, LGRIP]
    new_states[:, 1:2] = states[:, RGRIP]
    for t in range(n):
        new_states[t, 2:9] = _abs_ee_pose_7d(states[t, LEFT_TCP])
        new_states[t, 9:16] = _abs_ee_pose_7d(states[t, RIGHT_TCP])

    # --- Build new action: L7 = target L ee pose (from next state), L_grip from
    # original action trigger, same for R. Last frame copies current state.
    new_actions = np.zeros_like(actions)
    # Gripper triggers preserved from original joint-action file.
    new_actions[:, 7] = actions[:, 7]
    new_actions[:, 15] = actions[:, 15]
    for t in range(n):
        nxt = min(t + 1, n - 1)
        new_actions[t, 0:7] = _abs_ee_pose_7d(states[nxt, LEFT_TCP])
        new_actions[t, 8:15] = _abs_ee_pose_7d(states[nxt, RIGHT_TCP])

    # Rebuild the table keeping every other column and its metadata intact. The
    # state/actions columns must stay FixedSizeList<float,dim> so LeRobot's
    # ``hf_transform_to_torch`` can convert them to tensors.
    state_arr = _fixed_size_list_array(new_states)
    actions_arr = _fixed_size_list_array(new_actions)

    fields = []
    columns = []
    for i, name in enumerate(table.schema.names):
        if name == "state":
            fields.append(pa.field("state", state_arr.type))
            columns.append(state_arr)
        elif name == "actions":
            fields.append(pa.field("actions", actions_arr.type))
            columns.append(actions_arr)
        else:
            fields.append(table.schema.field(i))
            columns.append(table.column(i))

    new_schema = pa.schema(fields, metadata=table.schema.metadata)
    out_table = pa.Table.from_arrays(columns, schema=new_schema)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, dst_path)
    return n


def _copy_meta(src_root: Path, dst_root: Path, new_repo_id: str) -> None:
    src_meta = src_root / "meta"
    dst_meta = dst_root / "meta"
    dst_meta.mkdir(parents=True, exist_ok=True)

    # info.json: keep schema (state shape [68], actions shape [16]) but bump
    # version metadata if present so downstream caches don't collide.
    with open(src_meta / "info.json", "r") as f:
        info = json.load(f)
    info["_preprocessed_for"] = "tcp_pose_delta"
    info["_source_repo_id"] = str(src_root)
    with open(dst_meta / "info.json", "w") as f:
        json.dump(info, f, indent=4)

    # episodes.jsonl / tasks.jsonl / episodes_stats.jsonl: copy as-is. We
    # deliberately leave episodes_stats.jsonl alone — the user reruns
    # calculate_norm_stats.py afterwards, which is what actually matters for
    # pi05 normalization.
    for name in ("episodes.jsonl", "tasks.jsonl"):
        shutil.copy2(src_meta / name, dst_meta / name)

    stats_src = src_meta / "episodes_stats.jsonl"
    if stats_src.exists():
        shutil.copy2(stats_src, dst_meta / "episodes_stats.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        type=Path,
        default=Path(
            "/mnt/public1/chenyinuo/RLinf/datasets/collected_data/YinuoTHU/Dual-franka-test"
        ),
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path(
            "/mnt/public1/chenyinuo/RLinf/datasets/collected_data/YinuoTHU/Dual-franka-tcp"
        ),
    )
    parser.add_argument("--new-repo-id", type=str, default="YinuoTHU/Dual-franka-tcp")
    args = parser.parse_args()

    src_root: Path = args.src
    dst_root: Path = args.dst
    assert src_root.exists(), f"source dataset missing: {src_root}"

    # Copy meta first so failures during parquet rewrite leave a coherent partial state.
    _copy_meta(src_root, dst_root, args.new_repo_id)

    # Mirror ``data/chunk-XXX/episode_*.parquet`` under dst.
    data_dir = src_root / "data"
    total_frames = 0
    episode_count = 0
    for chunk_dir in sorted(data_dir.iterdir()):
        if not chunk_dir.is_dir():
            continue
        for ep_file in sorted(chunk_dir.glob("episode_*.parquet")):
            rel = ep_file.relative_to(src_root)
            dst_file = dst_root / rel
            n = _rewrite_episode(ep_file, dst_file)
            total_frames += n
            episode_count += 1
            print(f"[{episode_count:03d}] {rel} -> {dst_file}  ({n} frames)")

    print(
        f"\nDone. {episode_count} episodes, {total_frames} frames written to {dst_root}."
    )
    print(
        "Next: symlink or export HF_LEROBOT_HOME and re-run calculate_norm_stats.py."
    )


if __name__ == "__main__":
    main()
