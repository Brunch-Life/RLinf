#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Collect scripted expert demonstrations for RealAgainstPegSlot-v0.

The saved policy action is flat and normalized::

    actions[0:6] = Panda end-effector delta pose
    actions[6:9] = movable slot [dx, dy, dyaw]

The state is the Panda qpos (9). The slot pose is observed through RGB. The slot
remains fixed after reset; each episode moves the grasped peg over the slot with
the arm and then lowers the end effector.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

RLINF_ROOT = Path(__file__).resolve().parents[2]
if str(RLINF_ROOT) not in sys.path:
    sys.path.insert(0, str(RLINF_ROOT))

from rlinf.data.storage.lerobot import add_frame_to_dataset  # noqa: E402

LOG = logging.getLogger("collect_realagainst_peg_slot")
ENV_ID = "RealAgainstPegSlot-v0"
TASK = "insert the grasped peg downward into the movable slot"
STATE_DIM = 9
ACTION_DIM = 9


def _to_numpy(value: Any, *, squeeze_batch: bool = True) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if squeeze_batch and array.ndim > 0 and array.shape[0] == 1:
        array = array[0]
    return array


def _scalar_bool(value: Any) -> bool:
    return bool(_to_numpy(value).reshape(-1)[0])


def _rgb(obs: dict[str, Any], camera: str) -> np.ndarray:
    image = _to_numpy(obs["sensor_data"][camera]["rgb"])
    return np.asarray(image, dtype=np.uint8)


def _state(env: Any) -> np.ndarray:
    base = env.unwrapped
    qpos = _to_numpy(base.agent.robot.get_qpos()).astype(np.float32)
    state = qpos[:9]
    if state.shape != (STATE_DIM,):
        raise ValueError(f"Expected state shape {(STATE_DIM,)}, got {state.shape}")
    return state.astype(np.float32)


def _expert_action(env: Any) -> np.ndarray:
    base = env.unwrapped
    xy_error = base._slot_xy[0] - base.peg_tip_pose.p[0, :2]

    action = np.zeros(ACTION_DIM, dtype=np.float32)
    # A full public arm translation action corresponds to an approximately
    # 2.5 cm controller target delta after the environment's action scaling.
    action[:2] = _to_numpy(
        (xy_error / 0.025).clamp(-0.7, 0.7),
        squeeze_batch=False,
    )
    if float(xy_error.norm()) < 0.0025:
        action[2] = -0.16
    return action


def _collect_episode(env: Any, seed: int, max_steps: int) -> list[dict[str, Any]]:
    obs, _ = env.reset(seed=seed)
    frames: list[dict[str, Any]] = []

    for _ in range(max_steps):
        action = _expert_action(env)
        frames.append(
            {
                "image": _rgb(obs, "3rd_view_camera"),
                "wrist_image": _rgb(obs, "hand_camera"),
                "state": _state(env),
                "actions": action,
                "task": TASK,
            }
        )
        obs, _, terminated, truncated, info = env.step(action)
        if _scalar_bool(info["success"]):
            return frames
        if _scalar_bool(terminated) or _scalar_bool(truncated):
            break
    return []


def _create_dataset(repo_id: str, root: Path, fps: int, image_size: int):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    image_shape = (image_size, image_size, 3)
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=root,
        robot_type="panda_peg_slot",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (STATE_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=4,
        image_writer_processes=0,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="local/realagainst_peg_slot")
    parser.add_argument(
        "--root", default=str(RLINF_ROOT / "data/realagainst_peg_slot_expert")
    )
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(args.root).expanduser().resolve()
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Dataset already exists at {root}; use --overwrite")
        shutil.rmtree(root)

    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    import rlinf.envs.maniskill  # noqa: F401

    env = gym.make(
        ENV_ID,
        num_envs=1,
        obs_mode="rgb",
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
        sensor_configs={"width": args.image_size, "height": args.image_size},
        max_episode_steps=args.max_steps,
    )
    dataset = _create_dataset(args.repo_id, root, args.fps, args.image_size)
    saved = 0
    attempts = 0
    try:
        while saved < args.num_episodes and attempts < args.max_attempts:
            seed = args.seed + attempts
            attempts += 1
            frames = _collect_episode(env, seed, args.max_steps)
            if not frames:
                LOG.warning("Seed %d did not succeed", seed)
                continue
            for frame in frames:
                add_frame_to_dataset(dataset, frame)
            dataset.save_episode()
            saved += 1
            LOG.info(
                "Saved episode %d/%d (%d frames)", saved, args.num_episodes, len(frames)
            )
    finally:
        if getattr(dataset, "image_writer", None) is not None:
            dataset.image_writer.wait_until_done()
        env.close()

    if saved != args.num_episodes:
        raise RuntimeError(f"Collected only {saved}/{args.num_episodes} episodes")
    LOG.info("Dataset ready at %s (%d episodes, %d attempts)", root, saved, attempts)


if __name__ == "__main__":
    main()
