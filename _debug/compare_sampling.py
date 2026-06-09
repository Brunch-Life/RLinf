#!/usr/bin/env python3
"""Same-input comparison: RLinf's overridden sample_actions vs openpi's native
PI0Pytorch.sample_actions, on the SAME weights + SAME dumped obs + SAME initial
noise. embed_suffix/embed_prefix are shared, so any action difference isolates
the flow-integration reimplementation.

Run on the head (GPU), RLinf venv. Robot/cluster not needed.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import torch

EMBODIED = "/home/i-yinuo/cynws/RLinf-x2robot/examples/embodiment"
os.environ.setdefault("EMBODIED_PATH", EMBODIED)
os.environ.setdefault("REPO_PATH", "/home/i-yinuo/cynws/RLinf-x2robot")
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
sys.path.insert(0, os.environ["REPO_PATH"])

PKL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sm2sm_dump/rank_0_env_0_episode_0_step_45_fail.pkl"
STEPS = [0, 15, 30]


def build_cfg():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    with initialize_config_dir(version_base="1.1", config_dir=f"{EMBODIED}/config"):
        cfg = compose(config_name="realworld_x2robot_sm2sm_eval")
    model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.actor.model, resolve=True))
    with open_dict(model_cfg):
        model_cfg.precision = OmegaConf.select(cfg, "rollout.model.precision", default="bfloat16")
        model_cfg.model_path = cfg.rollout.model.model_path
    return model_cfg


def make_env_obs(obs, device):
    def t(x):
        return torch.as_tensor(np.asarray(x))[None].to(device)  # add batch dim

    main = t(obs["main_images"])  # (1,128,128,3) uint8
    extra = obs.get("extra_view_images")
    states = torch.as_tensor(np.asarray(obs["states"]), dtype=torch.float32)[None].to(device)
    task = obs.get("task_descriptions", "")
    if isinstance(task, (list, tuple, np.ndarray)):
        task = str(np.asarray(task).reshape(-1)[0])
    return {
        "main_images": main,
        "wrist_images": None,
        "extra_view_images": (t(extra) if extra is not None else None),
        "states": states,
        "task_descriptions": [task],
    }


def main():
    from openpi.models import model as _model
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    from rlinf.models.embodiment.openpi import get_model

    torch.manual_seed(0)
    device = "cuda"
    model_cfg = build_cfg()
    print(f"model_path={model_cfg.model_path}")
    model = get_model(model_cfg).to(device).eval()
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    c = model.config
    print(f"num_steps={c.num_steps} action_horizon={c.action_horizon} action_dim={c.action_dim} "
          f"action_chunk={c.action_chunk} noise_method={c.noise_method} config_name={c.config_name}")

    with open(PKL, "rb") as f:
        ep = pickle.load(f)
    obs_list = ep["observations"]
    print(f"loaded {len(obs_list)} obs; comparing steps {STEPS}")

    for s in STEPS:
        if s >= len(obs_list):
            continue
        env_obs = make_env_obs(obs_list[s], device)

        # Shared preprocessing -> Observation (exactly as predict_action_batch).
        to_proc = model.obs_processor(env_obs)
        proc = model.input_transform(to_proc, transpose=False)
        proc = model.precision_processor(proc)
        observation = _model.Observation.from_dict(proc)

        # Same initial noise for both samplers.
        bsize = observation.state.shape[0]
        ah, ad = model.config.action_horizon, model.config.action_dim
        g = torch.Generator(device=device).manual_seed(1234 + s)
        noise = torch.randn(bsize, ah, ad, generator=g, device=device,
                            dtype=model.action_in_proj.weight.dtype)

        with torch.no_grad():
            # RLinf overridden sampler (eval -> flow_ode, deterministic given noise)
            out_rl = model.sample_actions(observation, noise=noise.clone(), mode="eval",
                                          compute_values=False)
            raw_rl = out_rl["actions"]
            act_rl = model.output_transform({"actions": raw_rl, "state": observation.state})["actions"]

            # Native openpi sampler on the same weights+obs+noise
            raw_ref = PI0Pytorch.sample_actions(
                model, device, observation, noise=noise.clone(),
                num_steps=model.config.num_steps,
            )
            act_ref = model.output_transform({"actions": raw_ref, "state": observation.state})["actions"]

        raw_rl = raw_rl.float().cpu().numpy()[0]
        raw_ref = raw_ref.float().cpu().numpy()[0]
        a_rl = act_rl.float().cpu().numpy()[0]
        a_ref = act_ref.float().cpu().numpy()[0]
        np.set_printoptions(precision=4, suppress=True, linewidth=200)
        print(f"\n===== step {s} =====")
        print(f"raw (pre-unnorm) max|RLinf-native| = {np.abs(raw_rl - raw_ref).max():.6f}")
        print(f"final action shape RLinf={a_rl.shape} native={a_ref.shape}")
        print(f"final action max|RLinf-native| = {np.abs(a_rl - a_ref).max():.6f}")
        print(f"RLinf  action[0] (master half [14:28]):\n{a_rl[0,14:28] if a_rl.ndim==2 else a_rl[14:28]}")
        print(f"native action[0] (master half [14:28]):\n{a_ref[0,14:28] if a_ref.ndim==2 else a_ref[14:28]}")


if __name__ == "__main__":
    main()
