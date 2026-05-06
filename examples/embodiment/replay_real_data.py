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

import cv2
import hydra
import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.dataset_replay import (
    JOINT_ACTION_DIM,
    ROT6D_ACTION_DIM,
    load_replay_targets,
    peek_first_frame_joints,
)
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker

# Dataset image column names → on-screen panel labels. Order locks the panel
# layout (col 0..2) so DATASET row and LIVE row stay aligned by view.
_DATASET_IMAGE_COLS = ("extra_view_image-0", "image", "extra_view_image-1")
_PANEL_LABELS = ("base", "left_wrist", "right_wrist")

# Dataset state slicing (68-D): used by state-divergence diagnostics.
_GRIP_SLC = slice(0, 2)
_L_JOINT_SLC = slice(2, 9)
_R_JOINT_SLC = slice(9, 16)
_L_XYZ_SLC = slice(36, 39)
_L_QUAT_SLC = slice(39, 43)
_R_XYZ_SLC = slice(43, 46)
_R_QUAT_SLC = slice(46, 50)


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
                    self.dataset_root, self.episode_index, self.start_frame
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
        if self.target_source not in ("action", "state", "model"):
            raise ValueError(
                f"replay.target_source must be 'action' / 'state' / 'model'; "
                f"got {self.target_source!r}"
            )
        # Per-frame model inference is the dataset-obs-driven test: feed
        # dataset[t]'s (image, state) to the SFT ckpt, take chunk[0] as the
        # commanded TCP, dispatch through FrankyController. Isolates "model
        # output bad" from "live cameras / state distribution shift".
        if self.target_source == "model":
            self._model_inference_batch_size = int(
                rcfg.get("model_inference_batch_size", 4)
            )
            self._model_device = str(rcfg.get("model_device", "cuda:0"))

        # Side-by-side compare dump: dataset frames vs live camera frames.
        # PNG bytes are loaded once up front (cheap; ~30 KB/frame/view).
        # Decode + stitch happens per-step at dump time.
        self.dump_compare_dir = rcfg.get("dump_compare_dir", None)
        self.dump_every = int(rcfg.get("dump_every", 1))
        # When True, run() short-circuits: prealign-resets the arms to
        # dataset[start_frame] joints, dumps a single live-vs-dataset
        # comparison panel, and exits. Used to physically diagnose
        # camera/state drift at a specific dataset frame (e.g., the
        # handover_center) without dispatching any motion afterwards.
        self.dump_compare_after_reset_only = bool(
            rcfg.get("dump_compare_after_reset_only", False)
        )
        self._dataset_image_bytes: list[tuple[bytes, ...]] | None = None
        if self.dump_compare_dir:
            self.dump_compare_dir = Path(self.dump_compare_dir)
            self.dump_compare_dir.mkdir(parents=True, exist_ok=True)
            self._load_dataset_image_bytes()

    def _load_dataset_image_bytes(self) -> None:
        path = (
            self.dataset_root
            / "data"
            / "chunk-000"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        table = pq.read_table(path, columns=list(_DATASET_IMAGE_COLS))
        T = table.num_rows
        cols = [table.column(name) for name in _DATASET_IMAGE_COLS]
        rows: list[tuple[bytes, ...]] = []
        for i in range(T):
            row = []
            for col in cols:
                cell = col[i].as_py()
                if isinstance(cell, dict):
                    row.append(cell["bytes"])
                else:
                    row.append(cell)
            rows.append(tuple(row))
        self._dataset_image_bytes = rows
        self.log_info(
            f"[replay/compare] loaded {T} dataset image rows × "
            f"{len(_DATASET_IMAGE_COLS)} views from {path}"
        )

    @staticmethod
    def _decode_png(buf: bytes) -> np.ndarray:
        arr = np.frombuffer(buf, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)
        return img  # BGR for cv2

    @staticmethod
    def _to_bgr(img: np.ndarray) -> np.ndarray:
        if img.ndim == 4:
            img = img[0]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255).astype(np.uint8)
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    @staticmethod
    def _label(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 22), (0, 0, 0), thickness=-1)
        cv2.putText(
            out,
            text,
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return out

    def _live_panel_views(self, obs: dict) -> list[np.ndarray]:
        """Return ``[base, left_wrist, right_wrist]`` BGR uint8 from live obs."""
        main = obs.get("main_images")
        extra = obs.get("extra_view_images")
        if isinstance(main, torch.Tensor):
            main = main.detach().cpu().numpy()
        if isinstance(extra, torch.Tensor):
            extra = extra.detach().cpu().numpy()
        # main: (1,H,W,3) left_wrist; extra: (1,2,H,W,3) [base, right_wrist]
        left_wrist = self._to_bgr(main)
        base = self._to_bgr(extra[:, 0])
        right_wrist = self._to_bgr(extra[:, 1])
        return [base, left_wrist, right_wrist]

    def _dataset_panel_views(self, frame_idx: int) -> list[np.ndarray]:
        if self._dataset_image_bytes is None:
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank, blank, blank]
        if frame_idx >= len(self._dataset_image_bytes):
            blank = np.zeros((224, 224, 3), dtype=np.uint8)
            return [blank, blank, blank]
        return [self._decode_png(b) for b in self._dataset_image_bytes[frame_idx]]

    def _diagnose_reset_alignment(self, obs: dict) -> None:
        """Compare live joints + wrist views against dataset[start_frame].

        Frame 0 is the cleanest sanity check we have: if ``prealign_reset``
        worked, live joints should equal ``state[start_frame, 2:9 / 9:16]``
        to ~1e-3 rad (Ruckig's blocking JointMotion). A larger diff means
        the reset is going somewhere else — typically the YAML's default
        ``joint_reset_qpos`` because the prealign override silently failed
        or the comment-vs-code intent flipped.
        """
        # Live joint positions, walking through any wrapper chain.
        try:
            inner = self.env.env.envs[0]
            while hasattr(inner, "env") and not hasattr(inner, "get_joint_positions"):
                inner = inner.env
            live_q = np.asarray(inner.get_joint_positions())  # (2, 7)
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[replay/diag] couldn't read live joints: {exc}")
            live_q = None

        # Dataset start-frame joints.
        try:
            ds_l, ds_r = peek_first_frame_joints(self.dataset_root, self.episode_index)
        except Exception as exc:  # noqa: BLE001
            self.log_warning(f"[replay/diag] couldn't peek dataset joints: {exc}")
            ds_l = ds_r = None

        if live_q is not None and ds_l is not None:
            diff_l = live_q[0] - ds_l
            diff_r = live_q[1] - ds_r
            self.log_info(
                f"[replay/diag] joint align (live - dataset[{self.start_frame}]): "
                f"L max={np.abs(diff_l).max():.4f} rad mean={np.abs(diff_l).mean():.4f} | "
                f"R max={np.abs(diff_r).max():.4f} rad mean={np.abs(diff_r).mean():.4f}"
            )
            self.log_info(
                f"[replay/diag] live  L={np.round(live_q[0], 3).tolist()}  "
                f"R={np.round(live_q[1], 3).tolist()}"
            )
            self.log_info(
                f"[replay/diag] data  L={np.round(ds_l, 3).tolist()}  "
                f"R={np.round(ds_r, 3).tolist()}"
            )

        # Pre-step camera compare. Saves to ``dump_compare_dir/frame_pre_step.png``
        # so the operator can verify wrist viewpoints at *the same pose* as
        # the dataset, with zero per-step lag.
        if self.dump_compare_dir is not None:
            saved = self.dump_compare_dir
            self.dump_compare_dir = saved  # noqa: F841 - explicit no-op for clarity
            ds_views = self._dataset_panel_views(self.start_frame)
            live_views = self._live_panel_views(obs)
            h, w = ds_views[0].shape[:2]
            live_views = [
                cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA) for v in live_views
            ]
            ds_row = np.concatenate(
                [
                    self._label(v, f"DATASET {lbl}")
                    for v, lbl in zip(ds_views, _PANEL_LABELS)
                ],
                axis=1,
            )
            live_row = np.concatenate(
                [
                    self._label(v, f"LIVE {lbl} (post-reset)")
                    for v, lbl in zip(live_views, _PANEL_LABELS)
                ],
                axis=1,
            )
            panel = np.concatenate([ds_row, live_row], axis=0)
            footer = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
            cv2.putText(
                footer,
                f"PRE-STEP align  ep{self.episode_index}  frame[{self.start_frame}]",
                (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            panel = np.concatenate([panel, footer], axis=0)
            cv2.imwrite(str(self.dump_compare_dir / "frame_pre_step.png"), panel)

    def _dump_compare(self, frame_idx: int, obs: dict) -> None:
        if self.dump_compare_dir is None:
            return
        if self.dump_every > 1 and frame_idx % self.dump_every != 0:
            return
        ds_views = self._dataset_panel_views(frame_idx)
        live_views = self._live_panel_views(obs)
        # Resize all to a common size (use dataset's, typically 224×224).
        h, w = ds_views[0].shape[:2]
        live_views = [
            cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA) for v in live_views
        ]
        ds_row = np.concatenate(
            [
                self._label(v, f"DATASET {lbl}")
                for v, lbl in zip(ds_views, _PANEL_LABELS)
            ],
            axis=1,
        )
        live_row = np.concatenate(
            [
                self._label(v, f"LIVE {lbl}")
                for v, lbl in zip(live_views, _PANEL_LABELS)
            ],
            axis=1,
        )
        panel = np.concatenate([ds_row, live_row], axis=0)
        # Footer with frame index for quick scrubbing.
        footer = np.zeros((22, panel.shape[1], 3), dtype=np.uint8)
        cv2.putText(
            footer,
            f"frame {frame_idx:06d}  ep{self.episode_index}",
            (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        panel = np.concatenate([panel, footer], axis=0)
        out_path = self.dump_compare_dir / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), panel)

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

    def _load_dataset_state(self) -> np.ndarray:
        """Return raw ``(T, 68)`` dataset state for target building + diff."""
        path = (
            self.dataset_root
            / "data"
            / "chunk-000"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        table = pq.read_table(path, columns=["state", "actions"])
        T = table.num_rows
        state = np.stack(
            [
                np.asarray(table.column("state")[i].as_py(), dtype=np.float32)
                for i in range(T)
            ]
        )
        actions = np.stack(
            [
                np.asarray(table.column("actions")[i].as_py(), dtype=np.float32)
                for i in range(T)
            ]
        )
        # Cache for per-step diff and gripper trigger reuse.
        self._dataset_state = state
        self._dataset_actions = actions
        return state

    def _compute_model_targets(self) -> np.ndarray:
        """Per-frame model inference → ``(T, action_dim)`` first-chunk action.

        Dataset's recorded image + state at frame ``t`` is fed to the SFT
        ckpt; the first action of the predicted 20-step chunk becomes the
        commanded target for env.step at frame ``t``. Pre-computed once
        upfront to keep the dispatch loop running at the configured fps.
        """
        import io
        import json

        from PIL import Image

        from rlinf.models.embodiment.openpi import get_model
        from toolkits.dual_franka.backfill_rot6d import build_rot6d_state

        # Read SFT task description from the dataset's tasks.jsonl so we
        # don't depend on the replay yaml's bookkeeping task_description.
        with (self.dataset_root / "meta" / "tasks.jsonl").open() as f:
            task = json.loads(f.readline())["task"]
        self.log_info(f"[replay/model] task: {task}")

        # Decode all frames' images once. Dataset image columns:
        #   image                — left wrist (model's main_images)
        #   extra_view_image-0   — base
        #   extra_view_image-1   — right wrist
        path = (
            self.dataset_root
            / "data"
            / "chunk-000"
            / f"episode_{self.episode_index:06d}.parquet"
        )
        cols = ["image", "extra_view_image-0", "extra_view_image-1"]
        table = pq.read_table(path, columns=cols)
        T = table.num_rows

        def _decode(cell) -> np.ndarray:
            if isinstance(cell, dict):
                return np.asarray(Image.open(io.BytesIO(cell["bytes"])).convert("RGB"))
            if isinstance(cell, (bytes, bytearray)):
                return np.asarray(Image.open(io.BytesIO(cell)).convert("RGB"))
            return np.asarray(cell)

        main_imgs = [_decode(table.column("image")[i].as_py()) for i in range(T)]
        extra0_imgs = [
            _decode(table.column("extra_view_image-0")[i].as_py()) for i in range(T)
        ]
        extra1_imgs = [
            _decode(table.column("extra_view_image-1")[i].as_py()) for i in range(T)
        ]

        rot6d_state = build_rot6d_state(self._dataset_state)  # (T, 26)

        self.log_info(
            f"[replay/model] loading {self.cfg.actor.model.model_path} on "
            f"{self._model_device} ..."
        )
        t0 = time.time()
        model = get_model(self.cfg.actor.model, torch.bfloat16)
        model.eval()
        model.to(self._model_device)
        self.log_info(f"[replay/model] model loaded in {time.time() - t0:.1f}s")

        bs = self._model_inference_batch_size
        out = []
        torch.manual_seed(int(self.cfg.actor.get("seed", 1234)))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.cfg.actor.get("seed", 1234)))
        with torch.no_grad():
            for start in range(0, T, bs):
                end = min(start + bs, T)
                states = (
                    torch.from_numpy(np.stack(rot6d_state[start:end]))
                    .float()
                    .to(self._model_device)
                )
                main = torch.from_numpy(np.stack(main_imgs[start:end])).to(
                    self._model_device
                )
                extra = torch.from_numpy(
                    np.stack(
                        [
                            np.stack([extra0_imgs[i], extra1_imgs[i]], axis=0)
                            for i in range(start, end)
                        ]
                    )
                ).to(self._model_device)
                env_obs = {
                    "states": states,
                    "main_images": main,
                    "wrist_images": None,
                    "extra_view_images": extra,
                    "extra_view_image_names": [("base_0_rgb", "right_wrist_0_rgb")]
                    * (end - start),
                    "task_descriptions": [task] * (end - start),
                }
                actions, _ = model.predict_action_batch(
                    env_obs, mode="eval", compute_values=False
                )
                # actions: (B, chunk_size, action_dim). Take chunk[0] for
                # per-frame test (commands what the model would emit at
                # exactly that frame).
                out.append(actions[:, 0, :].float().cpu().numpy())
                if (start // bs) % 5 == 0:
                    self.log_info(f"[replay/model] inference {end}/{T} frames")

        # Free model: env.step needs the GPU memory back.
        del model
        torch.cuda.empty_cache()
        targets = np.concatenate(out, axis=0).astype(np.float32)
        self.log_info(
            f"[replay/model] computed targets shape={targets.shape} "
            f"first_action={targets[0].round(3).tolist()}"
        )
        return targets

    def _load_targets(self) -> np.ndarray:
        """Return ``(T, action_dim)`` per-frame replay targets.

        Joint mode honors ``target_source='action'`` (raw GELLO read) for
        diagnostic-only replays — rot6d mode forces ``state`` since the
        dataset has no recorded GELLO TCP. The shared
        :func:`load_replay_targets` covers state-mode for both widths.
        ``target_source='model'`` runs SFT inference per dataset frame to
        produce real-robot-executable model actions on dataset obs.
        """
        # Always cache dataset state — used for state-divergence diff regardless.
        self._load_dataset_state()

        if self.target_source == "model":
            return self._compute_model_targets()

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

    def _inner_env(self):
        """Walk through SyncVectorEnv + wrappers to the dual-Franka env."""
        inner = self.env.env.envs[0]
        while hasattr(inner, "env") and not (
            hasattr(inner, "_left_state") and hasattr(inner, "_right_state")
        ):
            inner = inner.env
        return inner

    def _record_state_divergence(self, frame_idx: int) -> None:
        """Compare live arm state vs dataset state[frame_idx + 1].

        Records L/R joint error (rad), TCP xyz error (m) and quat angle
        error (deg). At end of run we summarize max/mean/p95 — gives a
        clean number for "how far does this representation drift from the
        dataset trajectory?".
        """
        if not hasattr(self, "_state_diff_log"):
            self._state_diff_log = []
        target_idx = min(frame_idx + 1, self._dataset_state.shape[0] - 1)
        ds = self._dataset_state[target_idx]
        inner = self._inner_env()
        live_l_q = inner._left_state.arm_joint_position
        live_r_q = inner._right_state.arm_joint_position
        live_l_tcp = inner._left_state.tcp_pose
        live_r_tcp = inner._right_state.tcp_pose

        ds_l_q = ds[_L_JOINT_SLC]
        ds_r_q = ds[_R_JOINT_SLC]
        ds_l_xyz = ds[_L_XYZ_SLC]
        ds_l_quat = ds[_L_QUAT_SLC]
        ds_r_xyz = ds[_R_XYZ_SLC]
        ds_r_quat = ds[_R_QUAT_SLC]

        def _quat_angle_deg(q1: np.ndarray, q2: np.ndarray) -> float:
            d = float(np.clip(abs(np.dot(q1, q2)), 0.0, 1.0))
            return float(np.degrees(2.0 * np.arccos(d)))

        self._state_diff_log.append(
            {
                "frame": frame_idx,
                "L_q_max_rad": float(np.abs(live_l_q - ds_l_q).max()),
                "R_q_max_rad": float(np.abs(live_r_q - ds_r_q).max()),
                "L_xyz_m": float(np.linalg.norm(live_l_tcp[:3] - ds_l_xyz)),
                "R_xyz_m": float(np.linalg.norm(live_r_tcp[:3] - ds_r_xyz)),
                "L_rot_deg": _quat_angle_deg(live_l_tcp[3:], ds_l_quat),
                "R_rot_deg": _quat_angle_deg(live_r_tcp[3:], ds_r_quat),
            }
        )

    def _summarize_state_divergence(self) -> None:
        if not getattr(self, "_state_diff_log", None):
            return
        rows = self._state_diff_log
        keys = (
            "L_q_max_rad",
            "R_q_max_rad",
            "L_xyz_m",
            "R_xyz_m",
            "L_rot_deg",
            "R_rot_deg",
        )
        arr = {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}
        self.log_info(f"[replay/diverge] live vs dataset state — {len(rows)} steps:")
        for k in keys:
            v = arr[k]
            self.log_info(
                f"  {k:>12s}  max={v.max():.4f}  mean={v.mean():.4f}  "
                f"p95={np.percentile(v, 95):.4f}"
            )
        # Optional npz next to the dump dir for offline comparison across runs.
        if self.dump_compare_dir is not None:
            out = self.dump_compare_dir / "state_divergence.npz"
            np.savez(out, frame=np.asarray([r["frame"] for r in rows]), **arr)
            self.log_info(f"[replay/diverge] saved per-step diff to {out}")

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
        self._diagnose_reset_alignment(obs)

        # Reset-only diagnostic: dump live-vs-dataset panel for start_frame
        # (where prealign just put the arms) and exit. Does NOT step the
        # env, so no torque is applied to track any target. Use this to
        # physically inspect "is my live camera looking at the same scene
        # as the dataset at this exact joint config?"
        if self.dump_compare_after_reset_only:
            if self.dump_compare_dir is None:
                self.log_warning(
                    "[replay/compare-only] dump_compare_dir is null; nothing to write."
                )
            else:
                self.log_info(
                    f"[replay/compare-only] arms at dataset[{self.start_frame}] "
                    f"joints; dumping live-vs-dataset panel."
                )
                self._dump_compare(self.start_frame, obs)
            try:
                self.env.close()
            except Exception as e:  # noqa: BLE001
                self.log_warning(f"[replay] env.close raised: {e}")
            return

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
            self._dump_compare(i, obs)
            self._record_state_divergence(i)

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

        self._summarize_state_divergence()
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
