#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Collect one round of scripted-expert DAgger data for RealAgainstPegSlot-v0.

The student acts in GPU-vectorized environments. At every visited state the
simulator expert supplies the supervised action stored in the output dataset.
Both successful and failed student episodes are retained.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.data.storage.lerobot import add_frame_to_dataset  # noqa: E402
from toolkits.lerobot.collect_realagainst_peg_slot import (  # noqa: E402
    ACTION_DIM,
    ENV_ID,
    STATE_DIM,
    TASK,
    _create_dataset,
)

LOG = logging.getLogger("collect_realagainst_peg_slot_dagger")


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _image_to_uint8(image: Any) -> np.ndarray:
    image = _to_numpy(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0 if image.max() <= 1.0 else image, 0, 255)
        image = np.rint(image).astype(np.uint8)
    return image


def _copy_base_dataset(source_root: Path, output_dataset: Any) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = LeRobotDataset(source_root.name, root=source_root)
    previous_episode = None
    copied_episodes = 0
    for sample in source:
        episode = int(_to_numpy(sample["episode_index"]).item())
        if previous_episode is not None and episode != previous_episode:
            output_dataset.save_episode()
            copied_episodes += 1
        add_frame_to_dataset(
            output_dataset,
            {
                "image": _image_to_uint8(sample["image"]),
                "wrist_image": _image_to_uint8(sample["wrist_image"]),
                "state": _to_numpy(sample["state"]).astype(np.float32),
                "actions": _to_numpy(sample["actions"]).astype(np.float32),
                "task": str(sample["task"]),
            },
        )
        previous_episode = episode
    if previous_episode is not None:
        output_dataset.save_episode()
        copied_episodes += 1
    return copied_episodes


def _frame(raw_obs: dict[str, Any], expert_actions: torch.Tensor, env_idx: int):
    return {
        "image": _image_to_uint8(
            raw_obs["sensor_data"]["3rd_view_camera"]["rgb"][env_idx]
        ),
        "wrist_image": _image_to_uint8(
            raw_obs["sensor_data"]["hand_camera"]["rgb"][env_idx]
        ),
        "state": _to_numpy(raw_obs["agent"]["qpos"][env_idx, :STATE_DIM]).astype(
            np.float32
        ),
        "actions": _to_numpy(expert_actions[env_idx]).astype(np.float32),
        "task": TASK,
    }


def _load_student(config_path: Path, checkpoint_path: Path):
    from omegaconf import open_dict

    from rlinf.models import get_model

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.rollout.model
    with open_dict(model_cfg):
        model_cfg.model_path = str(checkpoint_path)
    model = get_model(model_cfg)
    return model.cuda().eval()


def _collect_batch(
    env: Any,
    student: Any,
    seeds: list[int],
    num_requested: int,
    max_steps: int,
) -> tuple[list[list[dict[str, Any]]], np.ndarray]:
    raw_obs, info = env.reset(seed=seeds)
    base = env.unwrapped
    num_envs = base.num_envs
    active = torch.arange(num_envs, device=base.device) < num_requested
    success = torch.zeros(num_envs, dtype=torch.bool, device=base.device)
    episodes: list[list[dict[str, Any]]] = [[] for _ in range(num_requested)]
    policy_chunk = None
    chunk_index = 0

    for _ in range(max_steps):
        if not active.any():
            break
        if policy_chunk is None or chunk_index >= policy_chunk.shape[1]:
            policy_chunk, _ = student.predict_action_batch(
                info["extracted_obs"], mode="eval"
            )
            if policy_chunk.shape[-1] != ACTION_DIM:
                raise ValueError(
                    f"Expected student action dim {ACTION_DIM}, got {policy_chunk.shape}"
                )
            chunk_index = 0

        student_action = policy_chunk[:, chunk_index].clone()
        expert_action = base.compute_expert_action()
        for env_idx in range(num_requested):
            if active[env_idx]:
                episodes[env_idx].append(_frame(raw_obs, expert_action, env_idx))

        student_action[~active] = 0
        raw_obs, _, terminated, truncated, info = env.step(student_action)
        step_success = info["success"].to(dtype=torch.bool)
        success |= active & step_success
        done = terminated.to(dtype=torch.bool) | truncated.to(dtype=torch.bool)
        active &= ~(step_success | done)
        chunk_index += 1

    return episodes, _to_numpy(success[:num_requested])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=RLINF_ROOT
        / "evaluations/maniskill/realagainst_peg_slot_openpi_rlinf_eval.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RLINF_ROOT
        / "logs/archive/realagainst_peg_slot/2026-08-14_pi05_sft_step_10000",
    )
    parser.add_argument(
        "--base-dataset",
        type=Path,
        default=RLINF_ROOT / "data/realagainst_peg_slot_expert",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=RLINF_ROOT / "data/realagainst_peg_slot_dagger_round1",
    )
    parser.add_argument("--repo-id", default="local/realagainst_peg_slot_dagger_round1")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-base-data", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    output_root = args.root.expanduser().resolve()
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset exists at {output_root}; use --overwrite")
        shutil.rmtree(output_root)

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    import rlinf.envs.maniskill  # noqa: F401

    student = _load_student(args.config.resolve(), args.checkpoint.resolve())
    env = gym.make(
        ENV_ID,
        num_envs=args.num_envs,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend="gpu",
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=args.max_steps,
    )
    dataset = _create_dataset(args.repo_id, output_root, args.fps, args.image_size)

    copied = 0
    collected = 0
    successes = 0
    try:
        if not args.skip_base_data:
            copied = _copy_base_dataset(args.base_dataset.resolve(), dataset)
            LOG.info("Copied %d base expert episodes", copied)

        while collected < args.num_episodes:
            requested = min(args.num_envs, args.num_episodes - collected)
            batch_seed = args.seed + collected
            seeds = list(range(batch_seed, batch_seed + args.num_envs))
            episodes, batch_success = _collect_batch(
                env, student, seeds, requested, args.max_steps
            )
            for frames in episodes:
                if not frames:
                    raise RuntimeError("DAgger collector produced an empty episode")
                for frame in frames:
                    add_frame_to_dataset(dataset, frame)
                dataset.save_episode()
            batch_successes = int(batch_success.sum())
            collected += requested
            successes += batch_successes
            LOG.info(
                "Collected %d/%d DAgger episodes; student success %d/%d (%.1f%%)",
                collected,
                args.num_episodes,
                successes,
                collected,
                100.0 * successes / collected,
            )
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()
        del student
        torch.cuda.empty_cache()

    LOG.info(
        "Dataset ready at %s: %d base + %d DAgger episodes; student success %.1f%%",
        output_root,
        copied,
        collected,
        100.0 * successes / collected,
    )


if __name__ == "__main__":
    main()
