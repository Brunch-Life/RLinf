# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Runtime budget control for RealAgainst, without changing generic env workers."""

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.workers.env.env_worker import EnvWorker


class PegSlotSACEnvWorker(EnvWorker):
    @staticmethod
    def _adapter(env):
        while not isinstance(env, ManiskillEnv):
            env = env.env
        return env

    def set_budget(self, budget):
        """Apply new limits on subsequent episode resets only."""
        for env in (*self.env_list, *self.eval_env_list):
            self._adapter(env).env.unwrapped.set_adversary_budget(budget)

    def prepare_budget_evaluation(self, budget):
        """Use the same evaluation seeds at every budget check."""
        for env in self.eval_env_list:
            adapter = self._adapter(env)
            adapter.env.unwrapped.set_adversary_budget(budget)
            adapter._full_reset_count = 0

    def reset_training_budget(self, budget):
        """Only for checkpoint restore, before any rollout has been sent."""
        self.set_budget(budget)
        for stage_id, env in enumerate(self.env_list):
            env.is_start = True
            observation, _ = env.reset()
            self.last_obs_list[stage_id] = observation
            self.last_intervened_info_list[stage_id] = (None, None)
            self.train_prev_done[stage_id].zero_()
