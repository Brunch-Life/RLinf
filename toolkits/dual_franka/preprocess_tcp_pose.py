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

Angle-wrap handling (IMPORTANT)
--------------------------------
``DeltaActions`` is a raw ``actions -= state`` subtraction with no angle
awareness. If ``state_euler`` frequently lies near ``±π`` (dual-Franka
grippers typically face down, so extrinsic-XYZ roll sits on that boundary),
naive subtraction produces spurious ``±2π`` "deltas" whenever ``scipy`` flips
the representation between adjacent frames — which poisons ``norm_stats.json``
quantiles and the learned delta distribution.

The fix:

- ``state`` stores **canonical** euler (range ``[-π, π]``) so training and
  inference see the same state distribution (env always produces canonical
  euler from the live quat).
- ``action`` stores ``state_euler[t] + wrap_to_pi(euler[t+1] - euler[t])``
  (may fall slightly outside ``[-π, π]`` near the boundary).
- At training: ``action - state = wrap_to_pi(Δeuler)`` — the physical small
  delta, free of 2π jumps.
- At inference: ``AbsoluteActions`` adds the canonical current state; the
  final euler→quat conversion is 2π-periodic so stepping "out of range"
  through the boundary is benign.

State layout after rewrite (``state[:16]``):
    [0]=L_grip_width, [1]=R_grip_width,
    [2:5]=L_xyz, [5:8]=L_euler_xyz (canonical), [8]=0,
    [9:12]=R_xyz, [12:15]=R_euler_xyz (canonical), [15]=0.
Slots ``[16:68]`` are preserved as-is (unused by pi05).

Action layout after rewrite (16 dims, matches the existing ``_rearrange_state``
output so no policy-side change is required):
    [0:3]=L_xyz_target, [3:6]=L_euler_xyz_target (wrap-aware), [6]=0,
    [7]=L_grip_trigger,
    [8:11]=R_xyz_target, [11:14]=R_euler_xyz_target (wrap-aware), [14]=0,
    [15]=R_grip_trigger.
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
    """Batched (n,4) → (n,3) canonical XYZ-extrinsic euler in [-π, π]."""
    return R.from_quat(quat_xyzw).as_euler("xyz")


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians into ``[-π, π]``. Works element-wise on any shape."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _arm_state_and_action_blocks(
    xyz_seq: np.ndarray,  # (n, 3) absolute EE xyz per frame
    quat_seq: np.ndarray,  # (n, 4) EE quat_xyzw per frame
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-arm 7-d state and action blocks for one episode.

    Returns
    -------
    state_block : (n, 7) float32, layout ``[xyz(3), euler_canonical(3), 0]``.
    action_block : (n, 7) float32, layout
        ``[xyz_next(3), euler_target(3), 0]``, where
        ``euler_target[t] = euler_canonical[t] + wrap_to_pi(euler[t+1]-euler[t])``
        so that training's ``DeltaActions`` (``action - state``) yields the
        physical small delta instead of occasional ``±2π`` wrap-artifacts.
    """
    n = xyz_seq.shape[0]
    euler_canon = _quat_to_euler_xyz(quat_seq).astype(np.float32)  # (n, 3)

    # state: xyz + canonical euler + pad
    state_block = np.zeros((n, 7), dtype=np.float32)
    state_block[:, 0:3] = xyz_seq
    state_block[:, 3:6] = euler_canon
    # state_block[:, 6] = 0  # pad

    # action: xyz from next frame; euler = state_euler[t] + wrap_to_pi(Δeuler)
    next_idx = np.minimum(np.arange(n) + 1, n - 1)
    delta_euler = _wrap_to_pi(euler_canon[next_idx] - euler_canon)  # (n, 3)

    action_block = np.zeros((n, 7), dtype=np.float32)
    action_block[:, 0:3] = xyz_seq[next_idx]
    action_block[:, 3:6] = euler_canon + delta_euler  # = "unwrapped next euler"
    # action_block[:, 6] = 0  # pad

    return state_block, action_block


def _fixed_size_list_array(values: np.ndarray) -> pa.Array:
    """Build a pyarrow FixedSizeListArray[float32, d] from an (n, d) numpy array."""
    assert values.ndim == 2
    n, d = values.shape
    flat = pa.array(values.reshape(-1).astype(np.float32), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(flat, d)


def _rewrite_episode(src_path: Path, dst_path: Path) -> int:
    table = pq.read_table(src_path)

    states = np.asarray(
        table["state"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32
    )
    actions = np.asarray(
        table["actions"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32
    )
    n = states.shape[0]
    assert states.shape[1] == 68, (
        f"unexpected state dim {states.shape[1]} in {src_path}"
    )
    assert actions.shape[1] == 16, (
        f"unexpected action dim {actions.shape[1]} in {src_path}"
    )

    # Batch per-arm xyz/quat across the whole episode so we can compute the
    # wrap-aware delta euler in one vectorized pass.
    l_xyz = states[:, LEFT_TCP.start : LEFT_TCP.start + 3]
    l_quat = states[:, LEFT_TCP.start + 3 : LEFT_TCP.stop]
    r_xyz = states[:, RIGHT_TCP.start : RIGHT_TCP.start + 3]
    r_quat = states[:, RIGHT_TCP.start + 3 : RIGHT_TCP.stop]

    l_state_block, l_action_block = _arm_state_and_action_blocks(l_xyz, l_quat)
    r_state_block, r_action_block = _arm_state_and_action_blocks(r_xyz, r_quat)

    # --- Build new state: overwrite [0:16] with the TCP-pose layout, keep tail.
    new_states = states.copy()
    new_states[:, 0:1] = states[:, LGRIP]
    new_states[:, 1:2] = states[:, RGRIP]
    new_states[:, 2:9] = l_state_block
    new_states[:, 9:16] = r_state_block

    # --- Build new action: arm EE targets from _arm_state_and_action_blocks,
    # gripper trigger channels preserved from the original joint-action file.
    new_actions = np.zeros_like(actions)
    new_actions[:, 0:7] = l_action_block
    new_actions[:, 7] = actions[:, 7]
    new_actions[:, 8:15] = r_action_block
    new_actions[:, 15] = actions[:, 15]

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
    print("Next: symlink or export HF_LEROBOT_HOME and re-run calculate_norm_stats.py.")


if __name__ == "__main__":
    main()
