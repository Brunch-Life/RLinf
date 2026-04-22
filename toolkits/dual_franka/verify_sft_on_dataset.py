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

"""Sanity-check that the deployed SFT model + eval-time inference pipeline
reproduce the training dataset's ground-truth actions on its own data.

Why: real-world eval is giving ~82 deg yaw delta. We need to prove whether
that signal comes from (a) genuine OOD real-world state, or (b) a bug in
the inference pipeline (state layout, norm stats, delta/abs conversion).

If this script reports "predicted actions ≈ dataset actions" on a few
episodes, the pipeline is correct and the real-world issue lies in
state/image distribution shift. If predictions diverge, the inference
pipeline has a bug somewhere.

Usage::

    /home/i-yinuo/cynws/RLinf/.venv-openpi/bin/python \\
        toolkits/dual_franka/verify_sft_on_dataset.py \\
        --episodes 0,5,10 --frames-per-ep 8
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf
from PIL import Image
from scipy.spatial.transform import Rotation as R

# Make rlinf importable without running hydra
_REPO = Path("/home/i-yinuo/cynws/RLinf")
sys.path.insert(0, str(_REPO))

DATA_ROOT = _REPO / "logs/20260417-17:57:23/collected_data/rank_0/id_0/data/chunk-000"
MODEL_PATH = "/home/i-yinuo/models/dual_franka/tcp_v1_14500/"
DEFAULT_PROMPT = (
    "pick up the cylinder with the left hand, hand it over to the right hand, "
    "and place it on the plate"
)

# Layout offsets in the raw 68-d state from GELLO collection.
LGRIP, RGRIP = 0, 1
LEFT_TCP_SLICE = slice(36, 43)  # [xyz(3), quat_xyzw(4)]
RIGHT_TCP_SLICE = slice(43, 50)


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    """Wrap radians into [-π, π]. Element-wise on any shape."""
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _rewrite_state_to_tcp(raw_state: np.ndarray) -> np.ndarray:
    """Port of preprocess_tcp_pose._rewrite_episode state rewrite.

    Input: (68,) raw state (GELLO-collect layout).
    Output: (68,) with [0:16] overwritten to TCP-pose layout::

        [L_grip, R_grip, L_xyz(3), L_euler_xyz(3), 0pad, R_xyz(3), R_euler_xyz(3), 0pad]

    ``L_euler_xyz`` / ``R_euler_xyz`` are the canonical scipy XYZ-extrinsic
    euler in ``[-π, π]`` — matching what the env produces from the live quat
    at inference time (see preprocess_tcp_pose docstring).
    """
    s = raw_state.copy()
    s[0] = raw_state[LGRIP]
    s[1] = raw_state[RGRIP]
    for base, sl in ((2, LEFT_TCP_SLICE), (9, RIGHT_TCP_SLICE)):
        tcp = raw_state[sl]
        s[base : base + 3] = tcp[:3]
        s[base + 3 : base + 6] = R.from_quat(tcp[3:7]).as_euler("xyz")
        s[base + 6] = 0.0
    return s


def _rewrite_action_to_tcp(
    raw_curr_state: np.ndarray, raw_next_state: np.ndarray, raw_action: np.ndarray
) -> np.ndarray:
    """Port of preprocess_tcp_pose action rewrite (wrap-aware euler).

    The target is state[t+1]'s EE pose for ``xyz``; for ``euler`` we store
    ``state_euler[t] + wrap_to_pi(euler[t+1] - euler[t])`` so that training's
    ``DeltaActions`` (``action - state``) yields the physical small delta
    instead of ``±2π`` wrap-artifacts. Gripper triggers are preserved from the
    original joint-action file.
    """
    out = np.zeros(16, dtype=np.float32)
    for base, sl in ((0, LEFT_TCP_SLICE), (8, RIGHT_TCP_SLICE)):
        curr = raw_curr_state[sl]
        nxt = raw_next_state[sl]
        euler_curr = R.from_quat(curr[3:7]).as_euler("xyz")
        euler_next = R.from_quat(nxt[3:7]).as_euler("xyz")
        out[base : base + 3] = nxt[:3]
        out[base + 3 : base + 6] = euler_curr + _wrap_to_pi(euler_next - euler_curr)
        # out[base + 6] = 0  # pad
    out[7] = raw_action[7]  # L_grip_trigger
    out[15] = raw_action[15]  # R_grip_trigger
    return out


def _decode_image(binary_struct) -> np.ndarray:
    """Decode a parquet image struct {bytes, path} into (H, W, 3) uint8.

    Training images in this lerobot dataset are stored as JPEG/PNG bytes.
    """
    b = (
        binary_struct.get("bytes")
        if isinstance(binary_struct, dict)
        else binary_struct["bytes"]
    )
    img = Image.open(io.BytesIO(b)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _load_episode(ep_idx: int) -> dict:
    path = DATA_ROOT / f"episode_{ep_idx:06d}.parquet"
    assert path.exists(), f"missing parquet: {path}"
    table = pq.read_table(path)
    n = table.num_rows

    states = np.asarray(
        table["state"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32
    )
    actions = np.asarray(
        table["actions"].to_numpy(zero_copy_only=False).tolist(), dtype=np.float32
    )
    assert states.shape == (n, 68)
    assert actions.shape == (n, 16)

    # Decode all images up-front (fast — 241 frames × 3 images ~ 500 MB RAM).
    imgs_main = [None] * n  # left wrist
    imgs_ev0 = [None] * n  # base
    imgs_ev1 = [None] * n  # right wrist
    for t in range(n):
        imgs_main[t] = _decode_image(table["image"][t].as_py())
        imgs_ev0[t] = _decode_image(table["extra_view_image-0"][t].as_py())
        imgs_ev1[t] = _decode_image(table["extra_view_image-1"][t].as_py())

    tcp_states = np.stack([_rewrite_state_to_tcp(s) for s in states])
    tcp_actions = np.zeros_like(actions)
    for t in range(n):
        nxt = min(t + 1, n - 1)
        tcp_actions[t] = _rewrite_action_to_tcp(states[t], states[nxt], actions[t])

    return {
        "n": n,
        "tcp_states": tcp_states,  # (n, 68), [0:16] = TCP layout
        "tcp_actions": tcp_actions,  # (n, 16)
        "imgs_main": imgs_main,
        "imgs_ev0": imgs_ev0,
        "imgs_ev1": imgs_ev1,
    }


def _build_model(device: str = "cuda") -> torch.nn.Module:
    """Replay rlinf.models.embodiment.openpi.get_model with a synthesized cfg."""
    cfg = OmegaConf.create(
        {
            "model_path": MODEL_PATH,
            "model_type": "openpi",
            "precision": None,
            "num_action_chunks": 20,
            "action_dim": 16,
            "is_lora": False,
            "lora_rank": 32,
            "use_proprio": True,
            "num_steps": 10,
            "add_value_head": False,
            "openpi": {
                "config_name": "pi05_dualfranka",
                "num_images_in_input": 3,
                "noise_level": 0.5,
                "action_chunk": 20,
                "num_steps": 10,
                "train_expert_only": False,
                "action_env_dim": 16,
                "noise_method": "flow_sde",
                "add_value_head": False,
                "value_after_vlm": False,
                "value_vlm_mode": "mean_token",
                "detach_critic_input": True,
                "use_dsrl": False,
                "dsrl_state_dim": 8,
                "dsrl_action_noise_dim": 32,
                "dsrl_num_q_heads": 10,
            },
        }
    )

    from rlinf.models.embodiment.openpi import get_model

    model = get_model(cfg)
    model = model.to(device).eval()
    return model


def _build_env_obs(ep: dict, t: int, device: str) -> dict:
    """Construct ``env_obs`` in the exact layout ``_wrap_obs`` produces for
    dual-franka TCP eval. Batch size = 1.

    Alphabetical state-dict concat order (see ``_wrap_obs``)::

        gripper_position(2) | joint_position(14, tcp_euler) | joint_velocity(14)
        | tcp_force(6) | tcp_pose(14) | tcp_torque(6) | tcp_vel(12)
        = 68 dims, but we only care about [:16] for the model.

    Here we fill the prefix from the dataset's rewritten tcp_state and
    leave the rest as zeros — the model only consumes state[:16].
    """
    state68 = np.zeros(68, dtype=np.float32)
    state68[:16] = ep["tcp_states"][t, :16]

    main_img = ep["imgs_main"][t]  # (H, W, 3) uint8
    ev0 = ep["imgs_ev0"][t]
    ev1 = ep["imgs_ev1"][t]
    # extra_view_images: stack alphabetically → (N, H, W, 3) (single-env batch
    # wraps with an outer dim → (B, N, H, W, 3)).
    extra = np.stack([ev0, ev1], axis=0)  # (2, H, W, 3)

    # Batch dim = 1 for every tensor; torch tensors on `device`.
    env_obs = {
        "states": torch.from_numpy(state68[None]).float().to(device),
        "main_images": torch.from_numpy(main_img[None]).to(device),
        "extra_view_images": torch.from_numpy(extra[None]).to(device),
        "extra_view_image_names": [("base_0_rgb", "right_wrist_0_rgb")],
        "wrist_images": None,
        "task_descriptions": [DEFAULT_PROMPT],
    }
    return env_obs


def _summarize_action_diff(pred: np.ndarray, gt: np.ndarray) -> dict:
    """pred, gt: (20, 16). Compute per-channel + global error summaries."""
    diff = pred - gt
    abs_diff = np.abs(diff)
    return {
        "mae_global": float(abs_diff.mean()),
        "max_abs_global": float(abs_diff.max()),
        "per_step_mae": abs_diff.mean(axis=1).tolist(),
        "per_dim_mae": abs_diff.mean(axis=0).tolist(),
        "per_dim_max": abs_diff.max(axis=0).tolist(),
        "first_wp_delta_lin_mm": [
            float(1000 * np.linalg.norm(diff[0, 0:3])),  # left arm xyz in mm
            float(1000 * np.linalg.norm(diff[0, 8:11])),  # right arm xyz in mm
        ],
        "first_wp_delta_ang_deg": [
            float(np.degrees(np.linalg.norm(diff[0, 3:6]))),  # left euler (rough)
            float(np.degrees(np.linalg.norm(diff[0, 11:14]))),
        ],
        "first_wp_grip_diff": [float(diff[0, 7]), float(diff[0, 15])],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--episodes",
        default="0,5,10,15",
        help="Comma-separated episode indices (0..20 available).",
    )
    p.add_argument(
        "--frames-per-ep",
        type=int,
        default=5,
        help="How many frames per episode to sample.",
    )
    p.add_argument("--device", default="cuda", help='"cuda" or "cpu"')
    args = p.parse_args()

    episodes = [int(x) for x in args.episodes.split(",")]
    print(f"[setup] building model on {args.device}… (loads ~8 GB weights)")
    model = _build_model(args.device)
    chunk_size = model.config.action_horizon
    assert chunk_size == 20, f"expected action_horizon=20, got {chunk_size}"
    print(f"[setup] model ready. action_horizon={chunk_size}")

    all_summaries = []
    for ep_idx in episodes:
        print(f"\n=== episode {ep_idx} ===")
        ep = _load_episode(ep_idx)
        n = ep["n"]
        # Pick sample times evenly across the episode; skip last chunk_size
        # frames so we have full ground-truth to compare against.
        max_t = max(0, n - chunk_size)
        sample_ts = np.linspace(0, max_t, num=args.frames_per_ep, dtype=int).tolist()
        print(f"[ep {ep_idx}] n={n}, sampling t={sample_ts}")

        for t in sample_ts:
            env_obs = _build_env_obs(ep, t, args.device)
            with torch.no_grad():
                pred_actions, _ = model.predict_action_batch(env_obs, mode="eval")
            # pred_actions: (1, 20, 16) tensor
            pred = pred_actions[0].detach().cpu().numpy()
            # Ground truth chunk: dataset actions [t : t+20]
            gt = ep["tcp_actions"][t : t + chunk_size]
            if gt.shape[0] < chunk_size:
                gt = np.concatenate(
                    [gt, np.tile(gt[-1:], (chunk_size - gt.shape[0], 1))], axis=0
                )

            summ = _summarize_action_diff(pred, gt)
            summ["ep"] = ep_idx
            summ["t"] = t
            all_summaries.append(summ)
            print(
                f"  t={t:3d}  mae_global={summ['mae_global']:.4f}  max={summ['max_abs_global']:.4f}  "
                f"wp0 lin(L/R)={summ['first_wp_delta_lin_mm'][0]:6.2f}/"
                f"{summ['first_wp_delta_lin_mm'][1]:6.2f}mm  "
                f"ang(L/R)={summ['first_wp_delta_ang_deg'][0]:5.2f}/"
                f"{summ['first_wp_delta_ang_deg'][1]:5.2f}deg  "
                f"grip(L/R)={summ['first_wp_grip_diff'][0]:+.2f}/"
                f"{summ['first_wp_grip_diff'][1]:+.2f}"
            )
            # Quick dump of per-dim MAE for first sample in first episode
            if ep_idx == episodes[0] and t == sample_ts[0]:
                print("  per-dim MAE (16):")
                for i, v in enumerate(summ["per_dim_mae"]):
                    print(f"    [{i:2d}] {v:.4f}")

    # Aggregate
    if all_summaries:
        mae_global = np.mean([s["mae_global"] for s in all_summaries])
        max_global = np.max([s["max_abs_global"] for s in all_summaries])
        wp0_ang_L = np.mean([s["first_wp_delta_ang_deg"][0] for s in all_summaries])
        wp0_ang_R = np.mean([s["first_wp_delta_ang_deg"][1] for s in all_summaries])
        wp0_lin_L = np.mean([s["first_wp_delta_lin_mm"][0] for s in all_summaries])
        wp0_lin_R = np.mean([s["first_wp_delta_lin_mm"][1] for s in all_summaries])
        print("\n=== aggregate ===")
        print(f"  samples     : {len(all_summaries)}")
        print(f"  mean mae    : {mae_global:.4f}")
        print(f"  max abs err : {max_global:.4f}")
        print(
            f"  wp0 start-jump vs GT: lin L/R = {wp0_lin_L:.2f}/{wp0_lin_R:.2f} mm, "
            f"ang L/R = {wp0_ang_L:.2f}/{wp0_ang_R:.2f} deg"
        )

    print(
        "\n(PASS criterion: wp0 ang error ~ a few deg and lin error ~ few mm. "
        "If you see >20 deg or >50 mm on the dataset itself, pipeline has a bug.)"
    )


if __name__ == "__main__":
    main()
