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
from rlinf.models.embodiment.resnet_mlp_policy import resolve_checkpoint
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


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


class _PegSlotPolicyPair:
    """Compose a synced learner with a separate, immutable opponent."""

    def init_worker(self):
        super().init_worker()
        self.train_role = self.cfg.rollout.peg_slot_adversary.get(
            "train_role", "adversary"
        )
        if self.train_role not in {"adversary", "robot"}:
            raise ValueError("train_role must be adversary or robot")
        self.deterministic_train_fraction = float(
            self.cfg.rollout.peg_slot_adversary.get("deterministic_train_fraction", 0.0)
        )
        if not 0 <= self.deterministic_train_fraction <= 1:
            raise ValueError("deterministic_train_fraction must be in [0, 1]")
        if self.deterministic_train_fraction and (
            self.train_role != "adversary"
            or self.cfg.actor.model.model_type != "mlp_policy"
            or not self.collect_transitions
        ):
            raise ValueError(
                "Mixed deterministic collection requires a SAC MLP adversary"
            )
        key = "robot_model" if self.train_role == "adversary" else "adversary_model"
        opponent_cfg = copy.deepcopy(self.cfg.rollout.get(key))
        self.opponent_model = None
        if opponent_cfg is not None:
            self.opponent_model = get_model(opponent_cfg)
            if self.opponent_model is None:
                raise ValueError(f"Unknown opponent model: {opponent_cfg.model_type}")
            # The legacy MLP factory does not load model_path.
            if opponent_cfg.model_type == "mlp_policy" and opponent_cfg.get(
                "model_path"
            ):
                self.opponent_model.load_state_dict(
                    torch.load(
                        resolve_checkpoint(opponent_cfg.model_path),
                        map_location="cpu",
                        weights_only=True,
                    ),
                    strict=True,
                )
            self.opponent_model.eval()
            self.opponent_model.requires_grad_(False)
        elif self.train_role == "adversary":
            raise ValueError("Adversary training requires a fixed robot model")

    @torch.inference_mode()
    def predict(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Return the learner's replay action, but execute both policies."""
        learner_actions, result = super().predict(env_obs, mode=mode)
        if mode == "train" and self.deterministic_train_fraction:
            count = int(learner_actions.shape[0] * self.deterministic_train_fraction)
            if count:
                mean_actions, mean_result = self.hf_model.predict_action_batch(
                    env_obs=env_obs, mode="eval", return_obs=False
                )
                learner_actions = learner_actions.clone()
                learner_actions[:count] = mean_actions[:count]
                # SAC recomputes policy log-probabilities for its loss. Keep
                # rollout likelihoods and version shapes aligned with actions.
                for key in ("prev_logprobs", "prev_values"):
                    if result.get(key) is not None:
                        result[key] = result[key].clone()
                        result[key][:count] = mean_result[key][:count]
                replay_actions = learner_actions.reshape(learner_actions.shape[0], -1)
                result["forward_inputs"]["action"] = replay_actions
                result["forward_inputs"]["model_action"] = replay_actions
                deterministic_mask = torch.zeros(
                    (learner_actions.shape[0], 1),
                    dtype=torch.bool,
                    device=learner_actions.device,
                )
                deterministic_mask[:count] = True
                result["forward_inputs"]["deterministic_collection"] = (
                    deterministic_mask
                )
        if self.opponent_model is None:
            opponent_actions = torch.zeros_like(learner_actions[..., :3])
        else:
            opponent_actions, _ = self.opponent_model.predict_action_batch(
                env_obs=env_obs, mode="eval"
            )
            opponent_actions = torch.as_tensor(
                opponent_actions,
                device=learner_actions.device,
                dtype=learner_actions.dtype,
            )
        if self.train_role == "adversary":
            robot_actions, slot_actions = opponent_actions, learner_actions
        else:
            robot_actions, slot_actions = learner_actions, opponent_actions
        return compose_peg_slot_actions(robot_actions, slot_actions), result


class PegSlotAdversaryRolloutWorker(_PegSlotPolicyPair, AsyncMultiStepRolloutWorker):
    """Backward-compatible asynchronous adversary rollout."""


class PegSlotSACRolloutWorker(_PegSlotPolicyPair, MultiStepRolloutWorker):
    """Synchronous SAC rollout used by stage-wise alternating training."""
