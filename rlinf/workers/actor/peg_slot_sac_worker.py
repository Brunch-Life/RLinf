# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Task-specific SAC lifecycle: warm up critics and clear stale gradients."""

import json
from pathlib import Path

from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class PegSlotSACPolicyWorker(EmbodiedSACFSDPPolicy):
    """Reuse SAC losses and replay without mixing data across opponents."""

    def update_one_epoch(self, train_actor=True):
        # Avoid including stale actor gradients in the critic's clipping norm.
        self.optimizer.zero_grad(set_to_none=True)
        warmup = int(self.cfg.actor.get("critic_warmup_updates", 0))
        train_actor = train_actor and self.update_step >= warmup
        result = super().update_one_epoch(train_actor=train_actor)
        if train_actor and self.update_step % self.critic_actor_ratio == 0:
            self._policy_updates = getattr(self, "_policy_updates", 0) + 1
        return result

    def run_training(self, num_updates=None):
        """Reuse SAC's update loop, shortening batches at runner boundaries."""
        configured_updates = self.cfg.algorithm.update_epoch
        updates = configured_updates if num_updates is None else num_updates
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
            raise ValueError("num_updates must be a positive integer")
        previous_step = self.update_step
        try:
            self.cfg.algorithm.update_epoch = updates
            metrics = super().run_training()
        finally:
            self.cfg.algorithm.update_epoch = configured_updates
        if self.update_step - previous_step != updates:
            raise RuntimeError("SAC did not perform the requested optimizer updates")
        metrics.update(
            {
                "sac/critic_updates": self.update_step,
                "sac/actor_updates": getattr(self, "_policy_updates", 0),
                "sac/updates_in_batch": updates,
            }
        )
        return metrics

    def get_update_counts(self):
        return {
            "critic_updates": int(self.update_step),
            "actor_updates": int(getattr(self, "_policy_updates", 0)),
        }

    def save_checkpoint(self, save_base_path, step):
        super().save_checkpoint(save_base_path, step)
        if self._rank == 0:
            (Path(save_base_path) / "peg_slot_update_counts.json").write_text(
                json.dumps(self.get_update_counts(), indent=2) + "\n"
            )

    def load_checkpoint(self, load_base_path):
        initial_entropy_state = None
        if self.cfg.algorithm.entropy_tuning.get("reset_on_resume", False):
            initial_entropy_state = {
                key: value.detach().clone()
                for key, value in self.entropy_temp.state_dict().items()
            }
        counts = json.loads(
            (Path(load_base_path) / "peg_slot_update_counts.json").read_text()
        )
        super().load_checkpoint(load_base_path)
        self.update_step = int(counts["critic_updates"])
        self._policy_updates = int(counts["actor_updates"])
        if initial_entropy_state is not None:
            self.entropy_temp.load_state_dict(initial_entropy_state)
            if self.alpha_optimizer is not None:
                self.alpha_optimizer.state.clear()
            self.log_info(
                f"Reset SAC entropy temperature to {self.entropy_temp.alpha:g} "
                "after checkpoint restore"
            )

    def forward_actor(self, batch):
        self.qf_optimizer.zero_grad(set_to_none=True)
        # Keep FSDP parameter requires_grad flags constant across forwards.
        # Only the actor optimizer steps in this pass; critic weights stay fixed.
        return super().forward_actor(batch)
