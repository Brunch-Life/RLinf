# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""LeRobot SFT data adapter for the ResNet18 + MLP task executor."""

from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader


class ResNetMLPDataset(Dataset):
    """Keep only the current pair of images, qpos, and single-step action."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        return {
            key: sample[key] for key in ("image", "wrist_image", "state", "actions")
        }


def build_resnet_mlp_sft_dataloader(
    cfg: DictConfig,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple[StatefulDataLoader, dict[str, int]]:
    """Read the existing local expert dataset with resumable distributed sampling."""
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if not isinstance(data_paths, (str, Path)):
        if len(data_paths) != 1:
            raise ValueError("ResNet-MLP SFT expects one local LeRobot dataset")
        data_paths = data_paths[0]
    root = Path(data_paths).expanduser().resolve()
    if int(cfg.actor.model.num_action_chunks) != 1:
        raise ValueError("ResNet-MLP SFT requires single-step actions")
    base = LeRobotDataset(
        repo_id=root.name,
        root=root,
        delta_timestamps=None,
        download_videos=False,
    )
    dataset = ResNetMLPDataset(base)
    seed = int(cfg.actor.get("seed", 0))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=not eval_dataset,
        seed=seed,
        drop_last=not eval_dataset,
    )
    workers = int(cfg.data.get("num_workers", 4))
    loader = StatefulDataLoader(
        dataset,
        batch_size=cfg.actor.eval_batch_size
        if eval_dataset
        else cfg.actor.micro_batch_size,
        sampler=sampler,
        drop_last=not eval_dataset,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=int(cfg.data.get("prefetch_factor", 2))
        if workers > 0
        else None,
        multiprocessing_context="spawn" if workers > 0 else None,
        generator=torch.Generator().manual_seed(seed + rank),
    )
    return loader, {"num_samples": len(dataset)}
