# Copyright 2025 The RLinf Authors.
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

import math

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from rlinf.models.embodiment.mlp_policy.iql_mlp_policy import IQLMLPPolicy
    from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
    from rlinf.models.embodiment.mlp_policy.rlt_mlp_policy import RLTMLPPolicy

    iql_config = cfg.get("iql_config", None)
    if cfg.model_type == "rlt_mlp_policy":
        model = RLTMLPPolicy(
            z_dim=cfg.z_dim,
            proprio_dim=cfg.proprio_dim,
            action_dim=cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            ref_num_action_chunks=cfg.get(
                "ref_num_action_chunks", cfg.num_action_chunks
            ),
            add_q_head=cfg.get("add_q_head", True),
            q_head_type=cfg.get("q_head_type", "default"),
            fixed_std=cfg.get("fixed_std", 0.002),
        )
    elif iql_config is not None:
        model = IQLMLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
        )
        model.configure_iql(iql_config)
    else:
        model = MLPPolicy(
            cfg.obs_dim,
            cfg.action_dim,
            num_action_chunks=cfg.num_action_chunks,
            add_value_head=cfg.add_value_head,
            add_q_head=cfg.get("add_q_head", False),
            q_head_type=cfg.get("q_head_type", "default"),
            value_granularity=cfg.get("value_granularity", "action_level"),
            observation_key=cfg.get("observation_key", "states"),
            hidden_dims=tuple(cfg.get("hidden_dims", (256, 256, 256))),
        )
        initial_std = cfg.get("sac_initial_std", None)
        if initial_std is not None:
            if not model.final_tanh or model.independent_std:
                raise ValueError("sac_initial_std requires an SAC MLP policy")
            initial_std = float(initial_std)
            if not math.isfinite(initial_std) or initial_std <= 0:
                raise ValueError("sac_initial_std must be finite and positive")
            log_std = math.log(initial_std)
            lower, upper = model.logstd_range
            if not lower < log_std < upper:
                raise ValueError("sac_initial_std is outside the model's log-std range")
            # Invert the tanh/range mapping used by the SAC distribution.
            # This controls initialization only; the std head remains trainable.
            raw_log_std = math.atanh(2 * (log_std - lower) / (upper - lower) - 1)
            torch.nn.init.zeros_(model.actor_logstd.weight)
            torch.nn.init.constant_(model.actor_logstd.bias, raw_log_std)

    return model
