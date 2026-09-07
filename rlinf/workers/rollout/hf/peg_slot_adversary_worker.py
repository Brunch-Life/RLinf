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

"""Composite rollout worker for a frozen peg policy and a slot adversary."""

from __future__ import annotations

import copy
from typing import Any, Literal

import torch

from rlinf.models import get_model
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)


def compose_peg_slot_actions(
    robot_actions: torch.Tensor, slot_actions: torch.Tensor
) -> torch.Tensor:
    """Combine frozen robot actions with trainable slot actions.

    The robot owns the first six public action dimensions. The adversary owns
    exactly the final three dimensions, irrespective of what the robot model
    predicted for those dimensions.
    """
    if robot_actions.ndim != 3 or slot_actions.ndim != 3:
        raise ValueError(
            "robot_actions and slot_actions must both have shape "
            "[batch, chunk, action_dim]"
        )
    if robot_actions.shape[:2] != slot_actions.shape[:2]:
        raise ValueError(
            "robot and slot batch/chunk shapes must match, got "
            f"{tuple(robot_actions.shape[:2])} and "
            f"{tuple(slot_actions.shape[:2])}"
        )
    if robot_actions.shape[-1] < 6:
        raise ValueError(
            f"robot action needs at least 6 dimensions, got {robot_actions.shape[-1]}"
        )
    if slot_actions.shape[-1] != 3:
        raise ValueError(
            f"slot adversary action must be 3D, got {slot_actions.shape[-1]}"
        )
    return torch.cat((robot_actions[..., :6], slot_actions), dim=-1)


class PegSlotAdversaryRolloutWorker(AsyncMultiStepRolloutWorker):
    """Roll out a frozen OpenPI robot together with a synced SAC adversary."""

    def init_worker(self):
        """Initialize the synced adversary and the separate frozen robot."""
        super().init_worker()
        robot_model_cfg = copy.deepcopy(self.cfg.rollout.robot_model)
        self.robot_model = get_model(robot_model_cfg)
        if self.robot_model is None:
            raise ValueError(
                "rollout.robot_model did not resolve to a registered model: "
                f"{robot_model_cfg.model_type!r}"
            )
        self.robot_model.eval()
        self.robot_model.requires_grad_(False)

    @torch.inference_mode()
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Predict SAC slot actions and combine them with robot actions."""
        slot_actions, adversary_result = super().predict(env_obs, mode=mode)
        robot_actions, _ = self.robot_model.predict_action_batch(
            env_obs=env_obs, mode="eval"
        )
        if not torch.is_tensor(robot_actions):
            robot_actions = torch.as_tensor(robot_actions)
        robot_actions = robot_actions.to(
            device=slot_actions.device, dtype=slot_actions.dtype
        )
        actions = compose_peg_slot_actions(robot_actions, slot_actions)
        return actions, adversary_result
