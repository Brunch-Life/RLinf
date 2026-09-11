# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Fixed-role SAC with periodic evaluation-driven disturbance budgets."""

import fcntl
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.utils.runner_utils import check_progress

ENVIRONMENT_REVISION = "rigid_peg_v1"


def adjust_budget(budget, success_rate, config):
    """Apply the configured success-rate deadband exactly once."""
    if not math.isfinite(success_rate) or not 0 <= success_rate <= 1:
        raise ValueError("Evaluation success rate must be finite and in [0, 1]")
    if not math.isfinite(budget) or not 0 <= budget <= config.maximum:
        raise ValueError("Budget must be finite and within the configured range")
    if success_rate < config.success_low:
        return (
            0.0 if budget == 0 else max(config.minimum, budget * config.decrease_factor)
        )
    if success_rate > config.success_high:
        return min(config.maximum, max(config.minimum, budget * config.increase_factor))
    return budget


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=lambda item: item.tolist())
        + "\n"
    )
    temporary.replace(path)


class PegSlotSACRunner(EmbodiedRunner):
    """Train exactly one policy; keep optimizer, replay and counters continuous."""

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg=cfg, **kwargs)
        self.budget = float(cfg.budget.initial)
        self.role = str(cfg.rollout.peg_slot_adversary.train_role)
        self.budget_history = []
        key = "robot_model" if self.role == "adversary" else "adversary_model"
        self.opponent_path = str(Path(cfg.rollout[key].model_path).resolve())

    def init_workers(self):
        state = None
        if self.cfg.runner.resume_dir:
            state = json.loads(
                (Path(self.cfg.runner.resume_dir) / "budget_state.json").read_text()
            )
            if state.get("environment_revision") != ENVIRONMENT_REVISION:
                raise ValueError(
                    "Cannot resume replay/optimizers from different peg physics. "
                    "Start a new run and use runner.ckpt_path to transfer policy weights."
                )
        super().init_workers()
        if state is not None:
            if (
                state["role"] != self.role
                or state["opponent_path"] != self.opponent_path
            ):
                raise ValueError(
                    "Resume must use the same train role and fixed opponent"
                )
            if state["global_step"] != self.global_step:
                raise ValueError("Checkpoint budget and learner step do not match")
            self.budget = float(state["next_budget"])
            adjust_budget(self.budget, 0.5, self.cfg.budget)
            self.budget_history = state["history"]
            self.env.reset_training_budget(self.budget).wait()
            # Resume may be between regular weight-sync intervals.
            self.update_rollout_weights()

    def evaluate(self):
        self.env.prepare_budget_evaluation(self.budget).wait()
        metrics = super().evaluate()
        expected = self.cfg.env.eval.total_num_envs * self.cfg.env.eval.rollout_epoch
        if int(metrics["num_trajectories"]) != expected:
            raise RuntimeError("Incomplete evaluation cannot adjust the budget")
        adjust_budget(self.budget, float(metrics["success_once"]), self.cfg.budget)
        return metrics

    def _maybe_eval_and_checkpoint(self, step):
        run_eval, save_model, _ = check_progress(
            self.global_step,
            self.max_steps,
            self.cfg.runner.val_check_interval,
            self.cfg.runner.save_interval,
            1.0,
            run_time_exceeded=False,
        )
        eval_metrics = {}
        if run_eval:
            counts = self.actor.get_update_counts().wait()[0]
            if counts["critic_updates"] != self.global_step:
                raise RuntimeError(f"Runner and actual SAC updates differ: {counts}")
            self.update_rollout_weights()
            metrics = self.evaluate()
            previous = self.budget
            success = float(metrics["success_once"])
            candidate = adjust_budget(previous, success, self.cfg.budget)
            # An extra clean check is only needed when further budget reduction
            # cannot help. Do not restart training or repeat critic warmup.
            if (
                success < self.cfg.budget.success_low
                and previous <= self.cfg.budget.minimum
            ):
                self.env.prepare_budget_evaluation(0.0).wait()
                clean = super().evaluate()
                expected = (
                    self.cfg.env.eval.total_num_envs * self.cfg.env.eval.rollout_epoch
                )
                if int(clean["num_trajectories"]) != expected:
                    raise RuntimeError("Incomplete zero-budget evaluation")
                clean_success = float(clean["success_once"])
                adjust_budget(0.0, clean_success, self.cfg.budget)
                metrics["clean_success_once"] = clean_success
                if clean_success < self.cfg.budget.success_low:
                    self._save_checkpoint()
                    _write_json(
                        Path(self.cfg.runner.logger.log_path) / "paused.json",
                        {
                            "step": self.global_step,
                            "reason": "Low success even at zero budget",
                            "clean_success_rate": clean_success,
                        },
                    )
                    raise RuntimeError(
                        "Low clean success; reducing disturbance cannot fix the learner"
                    )
                candidate = 0.0
            self.budget = candidate
            self.env.set_budget(candidate).wait()
            self.budget_history.append(
                {
                    "step": self.global_step,
                    "evaluated_budget": previous,
                    "success_rate": success,
                    "next_budget": candidate,
                    **counts,
                }
            )
            _write_json(
                Path(self.cfg.runner.logger.log_path) / "budget_history.json",
                self.budget_history,
            )
            eval_metrics = {f"eval/{key}": value for key, value in metrics.items()}
            eval_metrics.update(
                {"budget/evaluated": previous, "budget/next": candidate}
            )
            self.metric_logger.log(data=eval_metrics, step=self.global_step)
            self.logger.info(
                f"Step {self.global_step}: robot success={success:.2%}, "
                f"budget {previous:.6g} -> {candidate:.6g}; train role remains {self.role}"
            )
        if save_model:
            self._save_checkpoint()
        return eval_metrics

    def _save_checkpoint(self):
        super()._save_checkpoint()
        checkpoint = (
            Path(self.cfg.runner.logger.log_path)
            / self.cfg.runner.logger.experiment_name
            / "checkpoints"
            / f"global_step_{self.global_step}"
        )
        _write_json(
            checkpoint / "budget_state.json",
            {
                "environment_revision": ENVIRONMENT_REVISION,
                "role": self.role,
                "opponent_path": self.opponent_path,
                "global_step": self.global_step,
                "next_budget": self.budget,
                "history": self.budget_history,
            },
        )


def validate_peg_slot_sac_config(cfg):
    b = cfg.budget
    if not all(
        math.isfinite(v)
        for v in (
            b.minimum,
            b.initial,
            b.maximum,
            b.decrease_factor,
            b.increase_factor,
        )
    ):
        raise ValueError("Budget settings must be finite")
    if not 0 < b.minimum <= b.initial <= b.maximum:
        raise ValueError("Require 0 < minimum <= initial <= maximum")
    if not 0 < b.decrease_factor < 1 < b.increase_factor:
        raise ValueError("Invalid budget adjustment factors")
    if not 0 <= b.success_low < b.success_high <= 1:
        raise ValueError("Invalid success-rate deadband")
    if cfg.algorithm.update_epoch != 1:
        raise ValueError(
            "update_epoch must be 1: intervals count SAC optimizer updates"
        )
    if cfg.runner.max_steps < 1 or cfg.runner.val_check_interval < 1:
        raise ValueError("Training steps and evaluation interval must be positive")
    if cfg.runner.save_interval == 0 or (
        cfg.runner.save_interval > 0
        and cfg.runner.save_interval % cfg.runner.val_check_interval != 0
    ):
        raise ValueError(
            "save_interval must be negative or a positive multiple of val_check_interval"
        )
    if cfg.runner.get("overlap_env_bootstrap", False):
        raise ValueError("Budget evaluation requires overlap_env_bootstrap=false")
    role = cfg.rollout.peg_slot_adversary.train_role
    if role not in {"robot", "adversary"}:
        raise ValueError("train_role must be robot or adversary")
    if cfg.actor.model.num_action_chunks != 1:
        raise ValueError("This SAC experiment uses single-step actions")
    if role == "robot" and cfg.actor.model.freeze_resnet:
        raise ValueError("The executor experiment requires unfrozen ResNets")
    opponent_key = "robot_model" if role == "adversary" else "adversary_model"
    opponent = cfg.rollout[opponent_key]
    if not opponent.model_path or not Path(opponent.model_path).exists():
        raise ValueError(
            f"Set rollout.{opponent_key}.model_path to a real fixed-opponent checkpoint"
        )
    for split in ("train", "eval"):
        env = cfg.env[split]
        if env.expert_intervention_beta is not None:
            raise ValueError("SAC must not use expert intervention")
        if env.init_params.sac_train_role != role:
            raise ValueError("Environment reward role must match the learner")
        if env.init_params.adversary_budget != b.initial:
            raise ValueError("Use budget.initial for both train/eval initial budgets")


def run_sac_training(cfg):
    """Launch one continuous fixed-role run, without subprocesses or role switching."""
    import ray

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.peg_slot_sac_worker import PegSlotSACPolicyWorker
    from rlinf.workers.env.peg_slot_sac_env_worker import PegSlotSACEnvWorker
    from rlinf.workers.rollout.hf.peg_slot_adversary_worker import (
        PegSlotSACRolloutWorker,
    )

    validate_peg_slot_sac_config(cfg)
    if cfg.runner.only_eval:
        cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    directory = Path(cfg.runner.logger.log_path)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "config.yaml").exists() and not cfg.runner.resume_dir:
            raise RuntimeError(
                "Existing run directory: use a new log_path or explicit resume_dir"
            )
        config_name = "resume_config.yaml" if cfg.runner.resume_dir else "config.yaml"
        (directory / config_name).write_text(OmegaConf.to_yaml(cfg, resolve=True))
        cluster = Cluster(cluster_cfg=cfg.cluster)
        placement = HybridComponentPlacement(cfg, cluster)
        rollout = PegSlotSACRolloutWorker.create_group(cfg).launch(
            cluster,
            name=cfg.rollout.group_name,
            placement_strategy=placement.get_strategy("rollout"),
        )
        env = PegSlotSACEnvWorker.create_group(cfg).launch(
            cluster,
            name=cfg.env.group_name,
            placement_strategy=placement.get_strategy("env"),
        )
        try:
            if cfg.runner.only_eval:
                runner = EmbodiedEvalRunner(cfg=cfg, rollout=rollout, env=env)
                runner.init_workers()
                metrics = runner.evaluate()
                _write_json(
                    directory / "eval_result.json",
                    {**metrics, "environment_revision": ENVIRONMENT_REVISION},
                )
                runner.metric_logger.log(
                    data={f"eval/{k}": v for k, v in metrics.items()}, step=0
                )
                runner.metric_logger.finish()
            else:
                actor = PegSlotSACPolicyWorker.create_group(cfg).launch(
                    cluster,
                    name=cfg.actor.group_name,
                    placement_strategy=placement.get_strategy("actor"),
                )
                runner = PegSlotSACRunner(
                    cfg=cfg, actor=actor, rollout=rollout, env=env, reward=None
                )
                runner.init_workers()
                runner.run()
                counts = actor.get_update_counts().wait()[0]
                if (
                    counts["critic_updates"] != runner.global_step
                    or runner.global_step != cfg.runner.max_steps
                ):
                    raise RuntimeError(
                        f"Training stopped before the requested SAC updates: {counts}"
                    )
                _write_json(
                    directory / "result.json",
                    {
                        "environment_revision": ENVIRONMENT_REVISION,
                        "role": runner.role,
                        "updates": runner.global_step,
                        "next_budget": runner.budget,
                        **counts,
                    },
                )
        finally:
            ray.shutdown()
