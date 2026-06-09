#!/usr/bin/env python3
"""Inspect CollectEpisode pickle dumps from the sm2sm rollout.

Renders the exact frames the model received (main face view + the two wrist
views) and dumps the (seq, 28) [slave | master] state window, so we can eyeball
image color/content and state values against the working reference.

The env applies a BGR swap in _crop_frame, so the stored arrays are BGR. We save
each view rendered NATURAL (interpreted as BGR -> converted to RGB for the PNG)
and also report per-channel means so the true channel order is unambiguous.

Usage:
    python _debug/inspect_rollout_dump.py <pickle_or_glob> [--out OUTDIR] [--every N]
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np
from PIL import Image


def _to_np(x):
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def _as_uint8(img):
    img = _to_np(img)
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    return img


def _chw_to_hwc(img):
    # Accept (H,W,3) or (3,H,W).
    if img.ndim == 3 and img.shape[0] == 3 and img.shape[-1] != 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def _save_views(obs, step_idx, out_dir):
    rows = []
    main = obs.get("main_images", obs.get("image"))
    extra = obs.get("extra_view_images", obs.get("wrist_images"))
    views = {}
    if main is not None:
        views["face"] = _chw_to_hwc(_as_uint8(main))
    if extra is not None:
        extra = _as_uint8(extra)
        if extra.ndim == 4:  # (N,H,W,C)
            for i in range(extra.shape[0]):
                views[f"wrist{i}"] = _chw_to_hwc(extra[i])
        elif extra.ndim == 3:
            views["wrist0"] = _chw_to_hwc(extra)

    for name, arr in views.items():
        # Stored array is BGR (env _crop_frame swap). Convert BGR->RGB so the
        # PNG looks natural to a human; that IS what the model consumes as "rgb".
        natural = arr[:, :, ::-1]
        Image.fromarray(natural).save(
            os.path.join(out_dir, f"step{step_idx:03d}_{name}_natural.png")
        )
        r, g, b = arr[..., 0].mean(), arr[..., 1].mean(), arr[..., 2].mean()
        rows.append(
            f"  step{step_idx:03d} {name:7s} shape={arr.shape} dtype={arr.dtype} "
            f"chan_mean(stored c0/c1/c2)={r:.1f}/{g:.1f}/{b:.1f}"
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pickles", help="pickle path or glob")
    ap.add_argument("--out", default="_debug/rollout_view")
    ap.add_argument("--every", type=int, default=5, help="sample every Nth step")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.pickles))
    if not paths:
        raise SystemExit(f"no pickles match {args.pickles!r}")
    os.makedirs(args.out, exist_ok=True)

    for path in paths:
        with open(path, "rb") as f:
            ep = pickle.load(f)
        tag = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(args.out, tag)
        os.makedirs(out_dir, exist_ok=True)

        obs_list = ep.get("observations", [])
        actions = ep.get("actions", [])
        print(f"\n=== {tag} ===")
        print(f"  success={ep.get('success')} n_obs={len(obs_list)} n_act={len(actions)}")
        if obs_list:
            print(f"  obs keys: {list(obs_list[0].keys())}")
            print(f"  task: {obs_list[0].get('task_descriptions')!r}")

        # States: stack the full window per step.
        states = []
        for obs in obs_list:
            st = _to_np(obs.get("states", obs.get("state")))
            if st is not None:
                states.append(st)
        diag = []
        for i in range(0, len(obs_list), max(1, args.every)):
            diag += _save_views(obs_list[i], i, out_dir)
        # always include the last step
        if obs_list and (len(obs_list) - 1) % max(1, args.every) != 0:
            diag += _save_views(obs_list[-1], len(obs_list) - 1, out_dir)
        print("\n".join(diag))

        if states:
            states = np.stack(states, axis=0)  # (T, seq, 28) or (T, 28)
            np.save(os.path.join(out_dir, "states.npy"), states)
            print(f"  states array: shape={states.shape} dtype={states.dtype}")
            np.set_printoptions(precision=4, suppress=True, linewidth=200)
            # Show first step's full window and how slave/master diverge over time.
            print(f"  states[0] (first step, full window):\n{states[0]}")
            if states.ndim == 3 and states.shape[-1] == 28:
                slave = states[:, :, :14]
                master = states[:, :, 14:]
                print(
                    f"  |master-slave| mean over time (should grow as the "
                    f"autoregressive master diverges):\n"
                    f"  {np.abs(master - slave).mean(axis=(1, 2))}"
                )
        # Actions
        if len(actions):
            acts = np.stack([_to_np(a).reshape(-1) for a in actions], axis=0)
            np.save(os.path.join(out_dir, "actions.npy"), acts)
            print(f"  actions array: shape={acts.shape} (dispatched master half is [14:28] of model out)")

    print(f"\nWrote renders + npy under {args.out}/")


if __name__ == "__main__":
    main()
