# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Factory for the visual ResNet18 + MLP task policy."""

from pathlib import Path

import torch
from omegaconf import DictConfig


def resolve_checkpoint(path: str | Path) -> Path:
    """Accept a weights file or an RLinf checkpoint/actor directory."""
    path = Path(path).expanduser()
    if path.is_file():
        return path
    for relative in (
        "actor/model_state_dict/full_weights.pt",
        "model_state_dict/full_weights.pt",
        "full_weights.pt",
    ):
        candidate = path / relative
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No ResNet-MLP weights found at {path}")


def get_model(cfg: DictConfig, torch_dtype=torch.float32):
    """Build a policy; saved weights include all normalization statistics."""
    from .resnet_mlp_policy import ResNetMLPPolicy

    checkpoint = resolve_checkpoint(cfg.model_path) if cfg.get("model_path") else None
    if checkpoint is None and not cfg.get("norm_stats_path"):
        raise ValueError("norm_stats_path is required for a new ResNet-MLP policy")
    model = ResNetMLPPolicy(
        cfg,
        norm_stats_path=cfg.norm_stats_path if checkpoint is None else None,
        pretrained=checkpoint is None and bool(cfg.get("pretrained_backbone", True)),
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
    return model.to(dtype=torch_dtype)
