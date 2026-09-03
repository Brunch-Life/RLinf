#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Vectorized expert collection for RealAgainstPegSlot-v0."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.data.storage.lerobot import add_frame_to_dataset  # noqa: E402
from toolkits.lerobot.collect_realagainst_peg_slot import (  # noqa: E402
    ENV_ID,
    STATE_DIM,
    TASK,
    _create_dataset,
)

LOG = logging.getLogger("collect_realagainst_peg_slot_vectorized")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _collect_batch(
    env: Any,
    *,
    seeds: list[int],
    num_requested: int,
    max_steps: int,
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    """Collect at most one successful expert episode per vector environment."""
    base = env.unwrapped
    num_envs = int(base.num_envs)
    if len(seeds) != num_envs:
        raise ValueError(f"Expected {num_envs} seeds, got {len(seeds)}")
    if not 0 < num_requested <= num_envs:
        raise ValueError(f"num_requested must be in [1, {num_envs}]")

    obs, _ = env.reset(seed=seeds)
    episodes: list[list[dict[str, Any]]] = [[] for _ in range(num_requested)]
    active = torch.arange(num_envs, device=base.device) < num_requested
    succeeded = torch.zeros(num_envs, dtype=torch.bool, device=base.device)

    for _ in range(max_steps):
        if not active.any():
            break

        expert_actions = base.compute_expert_action()
        active_indices = (
            torch.nonzero(active, as_tuple=False).flatten().cpu().tolist()
        )
        main_images = _to_numpy(
            obs["sensor_data"]["3rd_view_camera"]["rgb"]
        ).astype(np.uint8)
        wrist_images = _to_numpy(
            obs["sensor_data"]["hand_camera"]["rgb"]
        ).astype(np.uint8)
        states = _to_numpy(obs["agent"]["qpos"])[..., :STATE_DIM].astype(
            np.float32
        )
        actions = _to_numpy(expert_actions).astype(np.float32)

        for env_idx in active_indices:
            if env_idx >= num_requested:
                continue
            episodes[env_idx].append(
                {
                    "image": main_images[env_idx].copy(),
                    "wrist_image": wrist_images[env_idx].copy(),
                    "state": states[env_idx].copy(),
                    "actions": actions[env_idx].copy(),
                    "task": TASK,
                }
            )

        step_actions = expert_actions.clone()
        step_actions[~active] = 0
        obs, _, terminated, truncated, info = env.step(step_actions)
        success = info["success"].to(torch.bool)
        done = terminated.to(torch.bool) | truncated.to(torch.bool)
        succeeded |= active & success
        active &= ~(success | done)

    successful_episodes = [
        episode if bool(succeeded[env_idx].item()) else []
        for env_idx, episode in enumerate(episodes)
    ]
    return successful_episodes, succeeded[:num_requested].detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=RLINF_ROOT / "data/realagainst_peg_slot_expert_5000",
    )
    parser.add_argument(
        "--repo-id", default="local/realagainst_peg_slot_expert_5000"
    )
    parser.add_argument("--num-episodes", type=int, default=5000)
    parser.add_argument("--max-attempts", type=int, default=6000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s", force=True
    )
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if args.num_episodes <= 0:
        raise ValueError("--num-episodes must be positive")
    if args.max_attempts < args.num_episodes:
        raise ValueError("--max-attempts must be at least --num-episodes")

    root = args.root.expanduser().resolve()
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset exists at {root}; use --overwrite")
        shutil.rmtree(root)

    import datasets
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    import rlinf.envs.maniskill  # noqa: F401

    datasets.disable_progress_bars()
    env = gym.make(
        ENV_ID,
        num_envs=args.num_envs,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=args.max_steps,
    )
    dataset = _create_dataset(
        args.repo_id, root, args.fps, args.image_size
    )

    saved = 0
    attempts = 0
    total_frames = 0
    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            num_requested = min(
                args.num_envs,
                args.num_episodes - saved,
                args.max_attempts - attempts,
            )
            seeds = list(
                range(
                    args.seed + attempts,
                    args.seed + attempts + args.num_envs,
                )
            )
            episodes, batch_success = _collect_batch(
                env,
                seeds=seeds,
                num_requested=num_requested,
                max_steps=args.max_steps,
            )
            attempts += num_requested

            for episode in episodes:
                if not episode:
                    continue
                for frame in episode:
                    add_frame_to_dataset(dataset, frame)
                dataset.save_episode()
                saved += 1
                total_frames += len(episode)

            LOG.info(
                "saved=%d/%d attempts=%d batch_success=%d/%d frames=%d",
                saved,
                args.num_episodes,
                attempts,
                int(batch_success.sum()),
                num_requested,
                total_frames,
            )
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()

    if saved != args.num_episodes:
        raise RuntimeError(
            f"Collected only {saved}/{args.num_episodes} successful episodes "
            f"after {attempts} attempts"
        )
    LOG.info(
        "Dataset ready at %s: episodes=%d attempts=%d frames=%d",
        root,
        saved,
        attempts,
        total_frames,
    )


if __name__ == "__main__":
    main()
