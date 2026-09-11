# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Deterministic two-camera behavior cloning for the RealAgainst robot."""

import json
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Normal
from torchvision.models import resnet18

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.q_head import MultiQHead
from rlinf.models.embodiment.modules.utils import make_mlp


class ResNetMLPPolicy(nn.Module, BasePolicy):
    """Encode each view independently, concatenate qpos, and predict one action.

    The actor uses RGB and qpos only. Optional SAC heads add a stochastic
    six-dimensional arm policy and a separate privileged-state twin critic.
    """

    _no_split_modules: list[str] = []

    def __init__(
        self,
        cfg: DictConfig,
        *,
        norm_stats_path: str | Path | None = None,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.action_dim = int(cfg.action_dim)
        self.predicted_action_dim = int(cfg.predicted_action_dim)
        self.state_dim = int(cfg.state_dim)
        self.image_size = int(cfg.image_size)
        if (self.state_dim, self.predicted_action_dim, self.action_dim) != (9, 6, 9):
            raise ValueError(
                "RealAgainst requires state_dim=9 and arm/env action_dim=6/9"
            )
        if self.num_action_chunks != 1:
            raise ValueError("ResNetMLPPolicy predicts one action: num_action_chunks=1")
        if self.image_size < 32:
            raise ValueError("image_size must be >= 32")
        hidden_dims = [int(dim) for dim in cfg.hidden_dims]
        if not hidden_dims or any(dim < 1 for dim in hidden_dims):
            raise ValueError("hidden_dims must contain positive hidden-layer widths")

        self.main_backbone = resnet18(weights=None)
        self.wrist_backbone = resnet18(weights=None)
        if pretrained:
            weights_path = cfg.get("pretrained_resnet_path")
            if not weights_path:
                raise ValueError(
                    "pretrained_resnet_path is required when pretrained_backbone=true"
                )
            state = torch.load(
                Path(weights_path).expanduser(), map_location="cpu", weights_only=True
            )
            self.main_backbone.load_state_dict(state, strict=True)
            self.wrist_backbone.load_state_dict(state, strict=True)
        feature_dim = self.main_backbone.fc.in_features
        self.main_backbone.fc = nn.Identity()
        self.wrist_backbone.fc = nn.Identity()
        self.freeze_resnet = bool(cfg.get("freeze_resnet", False))
        if self.freeze_resnet:
            self.main_backbone.requires_grad_(False)
            self.wrist_backbone.requires_grad_(False)
            self.main_backbone.eval()
            self.wrist_backbone.eval()
        self.mlp = nn.Sequential(
            *make_mlp(
                in_channels=2 * feature_dim + self.state_dim,
                mlp_channels=[*hidden_dims, self.predicted_action_dim],
                act_builder=nn.ReLU,
                last_act=False,
            )
        )

        self.sac_enabled = bool(cfg.get("add_q_head", False))
        if self.sac_enabled:
            self.actor_logstd = nn.Linear(hidden_dims[-1], self.predicted_action_dim)
            nn.init.zeros_(self.actor_logstd.weight)
            nn.init.constant_(
                self.actor_logstd.bias, float(cfg.get("sac_initial_log_std", -3.5))
            )
            self.q_head = MultiQHead(
                hidden_size=int(cfg.get("critic_obs_dim", 37)),
                action_feature_dim=self.predicted_action_dim,
                hidden_dims=list(cfg.get("critic_hidden_dims", [256, 256])),
                num_q_heads=2,
                output_dim=1,
            )

        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )
        stats = None
        if norm_stats_path is not None:
            with Path(norm_stats_path).open() as stream:
                stats = json.load(stream)["norm_stats"]
        for name, source, dim in (
            ("state", "state", self.state_dim),
            ("action", "actions", self.predicted_action_dim),
        ):
            mean = torch.zeros(dim)
            std = torch.ones(dim)
            if stats is not None:
                mean = torch.tensor(stats[source]["mean"][:dim], dtype=torch.float32)
                std = torch.tensor(stats[source]["std"][:dim], dtype=torch.float32)
                if mean.shape != (dim,) or std.shape != (dim,):
                    raise ValueError(f"Wrong {source} statistic dimensions")
                if (
                    not torch.isfinite(mean).all()
                    or not torch.isfinite(std).all()
                    or (std < 0).any()
                ):
                    raise ValueError(f"Invalid {source} statistics")
            self.register_buffer(f"{name}_mean", mean)
            self.register_buffer(f"{name}_std", std.clamp_min(1e-6))

    def train(self, mode: bool = True) -> "ResNetMLPPolicy":
        """Keep frozen encoder BatchNorm statistics fixed during learner updates."""
        super().train(mode)
        if self.freeze_resnet:
            self.main_backbone.eval()
            self.wrist_backbone.eval()
        return self

    @property
    def device(self) -> torch.device:
        """Device hosting the policy."""
        return self.main_backbone.conv1.weight.device

    def _image(self, value: Any) -> torch.Tensor:
        image = torch.as_tensor(value, device=self.device)
        if image.ndim != 4:
            raise ValueError(f"Expected a batched RGB image, got {tuple(image.shape)}")
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if tuple(image.shape[1:]) != (3, self.image_size, self.image_size):
            raise ValueError(
                f"Expected RGB size {self.image_size}, got {tuple(image.shape)}"
            )
        if image.dtype == torch.uint8:
            return image.float() / 255.0
        if not image.is_floating_point():
            raise TypeError("RGB must be uint8 or floating point in [0, 1]")
        return image.float()

    def _observations(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        keys = (
            ("main_images", "wrist_images", "states")
            if "main_images" in batch
            else ("image", "wrist_image", "state")
        )
        state = torch.as_tensor(batch[keys[2]], device=self.device, dtype=torch.float32)
        if state.ndim != 2 or state.shape[1] != self.state_dim:
            raise ValueError(
                f"Expected [batch, {self.state_dim}] state, got {tuple(state.shape)}"
            )
        images = [self._image(batch[key]) for key in keys[:2]]
        if any(image.shape[0] != state.shape[0] for image in images):
            raise ValueError("Images and states must have matching batch sizes")
        return {"image": images[0], "wrist_image": images[1], "state": state}

    def prepare_lerobot_sft_batch(
        self, batch: dict[str, Any]
    ) -> dict[str, torch.Tensor]:
        """Adapt raw LeRobot samples without normalizing inputs twice."""
        result = self._observations(batch)
        actions = torch.as_tensor(
            batch["actions"], device=self.device, dtype=torch.float32
        )
        if actions.ndim == 3 and actions.shape[1] == 1:
            actions = actions.squeeze(1)
        if (
            actions.ndim != 2
            or actions.shape[0] != result["state"].shape[0]
            or actions.shape[-1] not in (6, 9)
        ):
            raise ValueError("Expected single-step [batch, 6 or 9] actions")
        result["actions"] = actions[..., : self.predicted_action_dim]
        return result

    def _encode(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        main_image = (batch["image"] - self.image_mean.float()) / self.image_std.float()
        wrist_image = (
            batch["wrist_image"] - self.image_mean.float()
        ) / self.image_std.float()
        main = self.main_backbone(
            main_image.to(dtype=self.main_backbone.conv1.weight.dtype)
        )
        wrist = self.wrist_backbone(
            wrist_image.to(dtype=self.wrist_backbone.conv1.weight.dtype)
        )
        state = (batch["state"] - self.state_mean.float()) / self.state_std.float()
        features = torch.cat((main, wrist, state.to(dtype=main.dtype)), dim=-1)
        return features

    def _predict_normalized(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self._encode(batch))

    def forward(
        self, forward_type: ForwardType = ForwardType.SFT, **kwargs
    ) -> torch.Tensor | tuple:
        """Dispatch supervised loss or SAC actor/critic forwards."""
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type != ForwardType.SFT:
            return self.default_forward(**kwargs)
        batch = self.prepare_lerobot_sft_batch(kwargs["data"])
        prediction = self._predict_normalized(batch).float()
        target = (batch["actions"] - self.action_mean.float()) / self.action_std.float()
        return (prediction - target).abs().mean()

    def default_forward(self, **kwargs):
        """Require an explicit supported forward type."""
        raise NotImplementedError(
            "Use ForwardType.SFT, SAC or SAC_Q for ResNetMLPPolicy"
        )

    def _sac_distribution(self, obs):
        if not self.sac_enabled:
            raise RuntimeError("SAC requires add_q_head=true")
        # Critic targets and alpha estimation must not update actor BN statistics.
        main_training = self.main_backbone.training
        wrist_training = self.wrist_backbone.training
        if not torch.is_grad_enabled():
            self.main_backbone.eval()
            self.wrist_backbone.eval()
        try:
            features = self._encode(self._observations(obs))
        finally:
            self.main_backbone.train(main_training)
            self.wrist_backbone.train(wrist_training)
        hidden = self.mlp[:-1](features)
        normalized_mean = self.mlp[-1](hidden).float()
        arm_mean = normalized_mean * self.action_std.float() + self.action_mean.float()
        # Inverse tanh preserves SFT deterministic actions up to endpoint epsilon.
        mean = torch.atanh(arm_mean.clamp(-1 + 1e-5, 1 - 1e-5))
        log_std = self.actor_logstd(hidden).float().clamp(-5.0, 1.0)
        return Normal(mean, log_std.exp())

    def sac_forward(self, obs, **kwargs):
        """Sample normalized environment arm actions (not SFT-standardized labels)."""
        distribution = self._sac_distribution(obs)
        raw_action = distribution.rsample()
        action = raw_action.tanh()
        # Stable log Jacobian of tanh, including near-saturated SFT actions.
        log_jacobian = 2 * (
            0.6931471805599453 - raw_action - nn.functional.softplus(-2 * raw_action)
        )
        return action, distribution.log_prob(raw_action) - log_jacobian, None

    def sac_q_forward(self, obs, actions, **kwargs):
        """Evaluate privileged twin Qs; the actor never consumes this state."""
        states = obs[self.cfg.get("critic_observation_key", "critic_states")].to(
            device=self.device, dtype=next(self.q_head.parameters()).dtype
        )
        actions = actions.reshape(states.shape[0], -1)
        if actions.shape[-1] != self.predicted_action_dim:
            raise ValueError(
                "Robot SAC replay actions must contain exactly 6 arm controls"
            )
        return self.q_head(states, actions.to(states.dtype))

    @torch.inference_mode()
    def predict_action_batch(self, env_obs: dict[str, Any], **kwargs):
        """Return one action as [batch, 1, 9]; the three slot controls stay zero."""
        if self.sac_enabled:
            if kwargs.get("mode", "eval") == "train":
                arm, logprobs, _ = self.sac_forward(env_obs)
            else:
                arm = self._sac_distribution(env_obs).mean.tanh()
                logprobs = None
            actions = nn.functional.pad(arm, (0, 3))
            return actions.unsqueeze(1), {
                "prev_logprobs": logprobs,
                "prev_values": None,
                "forward_inputs": {"action": arm, "model_action": arm},
            }
        normalized = self._predict_normalized(self._observations(env_obs)).float()
        arm = (normalized * self.action_std.float() + self.action_mean.float()).clamp(
            -1, 1
        )
        actions = torch.nn.functional.pad(
            arm, (0, self.action_dim - self.predicted_action_dim)
        )
        # The environment worker requires a time axis, even for a single action.
        return actions.unsqueeze(1), {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.flatten(1),
                "model_action": arm.flatten(1),
            },
        }
