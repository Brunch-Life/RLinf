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

"""Replay a recorded LeRobot episode through the live realworld env.

Open-loop sanity check: load per-frame targets from a single
``episode_NNNNNN.parquet`` and feed each row to ``env.step`` at the
configured fps. Compares the **dataset path** against the **hardware
trajectory** without any model or transforms in the loop.

Two modes, dispatched by the env's action_dim:

* **Joint env** (``DualFrankaJointEnv``, 16-d): targets are joint
  positions (next-frame state by default), driven through
  :py:meth:`FrankyController.move_joints` →
  ``JointImpedanceTracker.set_target``.
* **Rot6d env** (``DualFrankaRot6dEnv``, 20-d): targets are TCP poses
  built via :func:`toolkits.dual_franka.backfill_rot6d.build_rot6d_actions`
  (next-frame state's xyz/quat → rot6d), driven through
  :py:meth:`FrankyController.move_tcp_pose` →
  ``CartesianImpedanceTracker.set_target``. Used to isolate "is the
  rot6d dispatch path itself capable of reproducing the dataset?" from
  model-output issues.

Run via::

    # joint replay (default)
    bash examples/embodiment/replay_data.sh

    # TCP-pose replay (rot6d env, dataset-driven)
    bash examples/embodiment/replay_data.sh realworld_replay_rot6d_dual_franka

    # override knobs
    bash examples/embodiment/replay_data.sh <config> \\
        replay.episode_index=3
"""

import time
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.dataset_replay import (
    JOINT_ACTION_DIM,
    ROT6D_ACTION_DIM,
    load_replay_targets,
    peek_first_frame_joints,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker


class DatasetReplayer(Worker):
    """Drives ``env.step`` from a parquet ``actions`` column."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        rcfg = cfg.replay
        self.dataset_root = Path(rcfg.dataset_root)
        self.episode_index = int(rcfg.episode_index)
        self.fps = float(rcfg.get("fps", 10.0))
        self.start_frame = int(rcfg.get("start_frame", 0))
        self.max_frames = rcfg.get("max_frames", None)
        if self.max_frames is not None:
            self.max_frames = int(self.max_frames)
        self.log_every = int(rcfg.get("log_every", 20))
        self.prealign_reset = bool(rcfg.get("prealign_reset", True))

        # Pre-align reset before building env — env.reset will then slew via
        # blocking franky.JointMotion to dataset[0], so when the impedance
        # tracker takes over on the first env.step it sees ~zero position
        # error and never trips the 80 Nm torque reflex.
        if self.prealign_reset:
            try:
                l_q, r_q = peek_first_frame_joints(
                    self.dataset_root, self.episode_index
                )
            except FileNotFoundError as e:
                self.log_warning(f"[replay] prealign skipped: {e}")
            else:
                OmegaConf.set_struct(cfg, False)
                cfg.env.eval.override_cfg.joint_reset_qpos = [
                    l_q.tolist(),
                    r_q.tolist(),
                ]
                OmegaConf.set_struct(cfg, True)
                self.log_info(
                    f"[replay] prealign reset to ep{self.episode_index} "
                    f"frame[0] joints: L={np.round(l_q, 3).tolist()}  "
                    f"R={np.round(r_q, 3).tolist()}"
                )

        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
        )
        self.action_dim = int(self.env.action_space.shape[-1])
        # Where the replay target series comes from per frame:
        #   "action" — raw GELLO read at collection time (10 Hz). Subject to
        #              the GELLO-vs-arm 0.15 rad lead; reproduces "follow
        #              the operator's hand" with all the velocity peaks the
        #              tracker has to chase.
        #   "state"  — the arm's actually-measured joint position (next
        #              frame). Per-step deltas bounded by what the tracker
        #              physically achieved during collection (qdot < 2 rad/s),
        #              so far easier to replay faithfully.
        self.target_source = str(rcfg.get("target_source", "state")).lower()
        if self.target_source not in ("action", "state"):
            raise ValueError(
                f"replay.target_source must be 'action' or 'state'; "
                f"got {self.target_source!r}"
            )

    def _verify_rot6d_targets(self, state: np.ndarray, targets: np.ndarray) -> None:
        """Sanity-check rot6d targets: clipping vs ``action_space`` + rot6d round-trip.

        Catches the two failure modes that look like "XYZ wrong but rotation
        right" without surfacing as an exception:

        1. **action_space xyz clipping** — ``env.step`` does
           ``np.clip(action, action_space.low, action_space.high)``. If the
           dataset's measured TCP crosses an arm's per-axis bound (the most
           common case: handover trajectories crossing y=0 with mirrored
           per-arm y bounds), every offending frame's xyz silently snaps to
           the bound while rot6d is untouched.
        2. **rot6d encoding round-trip** — confirm
           ``quat → rot6d → quat`` (with hemisphere alignment) reproduces
           the dataset's measured quat to machine precision. A mismatch
           means the encoder/decoder pair drifted (sign convention, basis
           ordering, gimbal-adjacent quats).
        """
        from rlinf.envs.realworld.franka.dual_franka_rot6d_env import (
            ACTION_DIM_PER_ARM,
        )
        from rlinf.utils.rot6d import (
            quat_xyzw_to_rot6d,
            rot6d_to_quat_xyzw_safe,
        )

        low = self.env.action_space.low
        high = self.env.action_space.high

        # (T, 2, 10) per-arm view: [xyz(3), rot6d(6), grip(1)]
        T = targets.shape[0]
        per_arm = targets.reshape(T, 2, ACTION_DIM_PER_ARM)
        low_arm = low.reshape(2, ACTION_DIM_PER_ARM)
        high_arm = high.reshape(2, ACTION_DIM_PER_ARM)

        for arm, name in enumerate(("L", "R")):
            xyz = per_arm[:, arm, 0:3]
            xyz_min = xyz.min(axis=0)
            xyz_max = xyz.max(axis=0)
            xyz_low = low_arm[arm, 0:3]
            xyz_high = high_arm[arm, 0:3]
            clipped_below = (xyz < xyz_low).sum(axis=0)
            clipped_above = (xyz > xyz_high).sum(axis=0)
            for axis_idx, axis_name in enumerate("xyz"):
                below = int(clipped_below[axis_idx])
                above = int(clipped_above[axis_idx])
                if below or above:
                    self.log_warning(
                        f"[replay/rot6d/verify] {name} {axis_name} clip: "
                        f"{below}/{T} below {xyz_low[axis_idx]:+.3f} "
                        f"({below / T:.0%}), "
                        f"{above}/{T} above {xyz_high[axis_idx]:+.3f} "
                        f"({above / T:.0%}) — dataset range "
                        f"[{xyz_min[axis_idx]:+.3f}, {xyz_max[axis_idx]:+.3f}]. "
                        f"Widen ee_pose_limit in the env yaml."
                    )
                else:
                    self.log_info(
                        f"[replay/rot6d/verify] {name} {axis_name} OK: "
                        f"data [{xyz_min[axis_idx]:+.3f}, {xyz_max[axis_idx]:+.3f}] "
                        f"⊂ bound [{xyz_low[axis_idx]:+.3f}, {xyz_high[axis_idx]:+.3f}]"
                    )

        # rot6d round-trip: quat (dataset) → rot6d (build_rot6d_actions) →
        # quat_decoded (env-side). Done in numpy so we don't depend on the
        # env's per-step _prev hemisphere-seed; we align decoded quat to the
        # dataset quat directly.
        for arm, name, quat_slice in (
            (0, "L", slice(39, 43)),
            (1, "R", slice(46, 50)),
        ):
            quat_ds = state[:, quat_slice]  # (T, 4) xyzw
            # Same as build_rot6d_actions: shift forward, last frame holds.
            quat_target = np.empty_like(quat_ds)
            quat_target[:-1] = quat_ds[1:]
            quat_target[-1] = quat_ds[-1]
            r6 = quat_xyzw_to_rot6d(quat_target)  # (T, 6)
            decoded = np.zeros_like(quat_target)
            for i in range(T):
                q = rot6d_to_quat_xyzw_safe(r6[i], fallback_quat_xyzw=quat_target[i])
                if float(np.dot(q, quat_target[i])) < 0.0:
                    q = -q
                decoded[i] = q
            # Geodesic angle error (rad) per frame: 2 * arccos(|<q1, q2>|).
            dots = np.clip(np.abs(np.einsum("ij,ij->i", decoded, quat_target)), 0, 1)
            ang_err_deg = np.degrees(2.0 * np.arccos(dots))
            self.log_info(
                f"[replay/rot6d/verify] {name} rot6d round-trip: "
                f"max={ang_err_deg.max():.4f}°  mean={ang_err_deg.mean():.4f}° "
                f"(should be ~0; >0.1° means rot6d encoder/decoder drift)"
            )

    def _load_targets(self) -> np.ndarray:
        """Return ``(T, action_dim)`` per-frame replay targets.

        Joint mode honors ``target_source='action'`` (raw GELLO read) for
        diagnostic-only replays — rot6d mode forces ``state`` since the
        dataset has no recorded GELLO TCP. The shared
        :func:`load_replay_targets` covers state-mode for both widths.
        """
        if self.action_dim == JOINT_ACTION_DIM and self.target_source == "action":
            path = (
                self.dataset_root
                / "data"
                / "chunk-000"
                / f"episode_{self.episode_index:06d}.parquet"
            )
            table = pq.read_table(path, columns=["actions"])
            T = table.num_rows
            targets = np.stack(
                [
                    np.asarray(table.column("actions")[i].as_py(), dtype=np.float32)
                    for i in range(T)
                ]
            )
            self.log_info(
                f"[replay/joint] target=action loaded {targets.shape} from {path}"
            )
        else:
            if self.action_dim == ROT6D_ACTION_DIM and self.target_source == "action":
                self.log_warning(
                    "[replay/rot6d] target_source='action' isn't supported "
                    "(no GELLO-TCP recorded). Falling back to state-mode."
                )
            targets = load_replay_targets(
                self.dataset_root, self.episode_index, self.action_dim
            )
            self.log_info(
                f"[replay] target=state(next-frame proprio + action grip) "
                f"action_dim={self.action_dim} shape={targets.shape}"
            )
            if self.action_dim == ROT6D_ACTION_DIM:
                # Re-load raw state once for the rot6d round-trip check.
                path = (
                    self.dataset_root
                    / "data"
                    / "chunk-000"
                    / f"episode_{self.episode_index:06d}.parquet"
                )
                table = pq.read_table(path, columns=["state"])
                T = table.num_rows
                state = np.stack(
                    [
                        np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
                        for i in range(T)
                    ]
                )
                self._verify_rot6d_targets(state, targets)

        if targets.shape[1] != self.action_dim:
            raise ValueError(
                f"target dim mismatch: env wants {self.action_dim}, "
                f"got {targets.shape[1]}."
            )
        return targets

    def run(self):
        targets = self._load_targets()
        T = targets.shape[0]
        end = (
            T if self.max_frames is None else min(T, self.start_frame + self.max_frames)
        )
        if not (0 <= self.start_frame < end):
            raise ValueError(
                f"invalid frame range start={self.start_frame} end={end} (T={T})"
            )

        self.log_info("[replay] resetting env ...")
        obs, _ = self.env.reset()
        self.log_info(
            f"[replay] reset done; replaying frames [{self.start_frame}, {end}) "
            f"of episode {self.episode_index} at {self.fps:.1f} Hz "
            f"(target_source={self.target_source})"
        )
        period = 1.0 / self.fps
        prev_target = None

        for i in range(self.start_frame, end):
            iter_t0 = time.perf_counter()
            a = targets[i : i + 1]  # (num_envs=1, D)
            obs, reward, term, trunc, info = self.env.step(a, auto_reset=False)

            if self.log_every and (i % self.log_every == 0 or i == end - 1):
                if self.action_dim == 16:
                    summary = (
                        f"L_q[:3]={a[0, :3].round(3).tolist()} "
                        f"R_q[:3]={a[0, 8:11].round(3).tolist()} "
                        f"L_grip={a[0, 7]:+.2f} R_grip={a[0, 15]:+.2f}"
                    )
                elif self.action_dim == 20:
                    # rot6d layout: [L_xyz, L_rot6d, L_grip, R_xyz, R_rot6d, R_grip]
                    summary = (
                        f"L_xyz={a[0, :3].round(3).tolist()} "
                        f"R_xyz={a[0, 10:13].round(3).tolist()} "
                        f"L_grip={a[0, 9]:+.2f} R_grip={a[0, 19]:+.2f}"
                    )
                else:
                    summary = f"target[:6]={a[0, :6].round(3).tolist()}"
                step_dt = "n/a"
                if prev_target is not None:
                    step_dt = f"|Δa|={float(np.linalg.norm(a[0] - prev_target)):+.4f}"
                self.log_info(f"[replay] step {i}/{end - 1}  {summary}  {step_dt}")

            prev_target = a[0].copy()

            sleep_for = period - (time.perf_counter() - iter_t0)
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.log_info(f"[replay] done; ran {end - self.start_frame} frames")
        try:
            self.env.close()
        except Exception as e:  # noqa: BLE001
            self.log_warning(f"[replay] env.close raised: {e}")


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_replay_joint_dual_franka",
)
def main(cfg):
    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = ComponentPlacement(cfg, cluster)
    env_placement = component_placement.get_strategy("env")
    replayer = DatasetReplayer.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )
    replayer.run().wait()


if __name__ == "__main__":
    main()
