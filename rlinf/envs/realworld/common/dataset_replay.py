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

"""Dataset-replay helpers shared by ``replay_real_data.py`` and the
``RealWorldEnv`` chunk_step override hook.

Builds per-frame replay targets from a single LeRobot ``episode_NNNNNN.parquet``
in the layout that the live env's ``action_space`` consumes:

* **16-d (joint env)** — next-frame measured arm joints from ``state[:, 2:9]`` /
  ``state[:, 9:16]`` plus the dataset's gripper trigger from
  ``actions[:, 7]`` / ``actions[:, 15]``. We don't replay the raw ``actions``
  column because that is the GELLO operator's hand position with a ~0.15 rad
  lead over the arm — fine to record, dangerous to chase.
* **20-d (rot6d env)** — next-frame measured TCP rebuilt via
  :func:`toolkits.dual_franka.backfill_rot6d.build_rot6d_actions`
  (``state[:, 36:43] / [:, 43:50]`` per arm → xyz + rot6d, plus the same
  gripper triggers).

Last-frame holds (``targets[-1] = targets[-1]``) so the replay can run past the
recorded length without diverging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

JOINT_ACTION_DIM = 16
ROT6D_ACTION_DIM = 20


def _episode_parquet(dataset_root: Path, episode_index: int) -> Path:
    path = dataset_root / "data" / "chunk-000" / f"episode_{episode_index:06d}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"replay parquet not found: {path}")
    return path


def peek_first_frame_joints(
    dataset_root: Path, episode_index: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(L_q (7,), R_q (7,))`` from frame 0 of the requested episode.

    Used to override ``env.eval.override_cfg.joint_reset_qpos`` *before* the env
    builds, so reset slews via blocking ``franky.JointMotion`` (Ruckig) to
    dataset[0] — keeps the impedance tracker's first-tick ``K*err`` from
    blowing past the libfranka torque reflex.
    """
    path = _episode_parquet(Path(dataset_root), episode_index)
    table = pq.read_table(path, columns=["state"])
    state0 = np.asarray(table.column("state")[0].as_py(), dtype=np.float32)
    # JointEnv state layout (alphabetical of STATE_LAYOUT, dim 68):
    #   gripper_position(2), joint_position(14), ...
    # → state[2:9] = L joints, state[9:16] = R joints. Same offsets in the
    # rot6d-backfilled dataset (backfill_rot6d only rewrites state[:20]).
    l_q = state0[2:9].astype(np.float64)
    r_q = state0[9:16].astype(np.float64)
    return l_q, r_q


def load_replay_targets(
    dataset_root: Path,
    episode_index: int,
    action_dim: int,
) -> np.ndarray:
    """Return ``(T, action_dim)`` per-frame replay targets.

    Always state-driven (next-frame measured proprio). ``action_dim`` selects
    the layout — 16 for the joint env, 20 for the rot6d env.
    """
    path = _episode_parquet(Path(dataset_root), episode_index)
    table = pq.read_table(path, columns=["state", "actions"])
    T = table.num_rows
    actions = np.stack(
        [
            np.asarray(table.column("actions")[i].as_py(), dtype=np.float32)
            for i in range(T)
        ]
    )
    state = np.stack(
        [
            np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
            for i in range(T)
        ]
    )

    if action_dim == JOINT_ACTION_DIM:
        # JointEnv state layout: gripper_position(2), joint_position(14), ...
        # → state[:, 2:9] = L joints, state[:, 9:16] = R joints. Gripper
        # triggers (±1) come from the dataset's actions column (state's
        # gripper_position is in metres, not the trigger semantic the env
        # consumes).
        l_q = state[:, 2:9]
        r_q = state[:, 9:16]
        l_grip = actions[:, 7:8]
        r_grip = actions[:, 15:16]
        l_q_next = np.empty_like(l_q)
        l_q_next[:-1] = l_q[1:]
        l_q_next[-1] = l_q[-1]
        r_q_next = np.empty_like(r_q)
        r_q_next[:-1] = r_q[1:]
        r_q_next[-1] = r_q[-1]
        targets = np.concatenate([l_q_next, l_grip, r_q_next, r_grip], axis=1).astype(
            np.float32
        )

    elif action_dim == ROT6D_ACTION_DIM:
        # Reuse the SFT pipeline's offline action builder so the replay
        # target series is byte-identical to what training would have
        # consumed: next-frame TCP xyz/quat → rot6d, last-frame holds,
        # gripper from action col.
        from toolkits.dual_franka.backfill_rot6d import build_rot6d_actions

        targets = build_rot6d_actions(state, actions)  # (T, 20)

    else:
        raise ValueError(
            f"unsupported action_dim {action_dim}; expected "
            f"{JOINT_ACTION_DIM} (joint) or {ROT6D_ACTION_DIM} (rot6d)."
        )

    if targets.shape != (T, action_dim):
        raise AssertionError(
            f"target shape mismatch: expected ({T}, {action_dim}), got {targets.shape}"
        )
    return targets
