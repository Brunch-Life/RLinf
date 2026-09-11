# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Train either the peg executor or the budgeted slot opponent with SAC."""

import hydra
from omegaconf import OmegaConf

from rlinf.runners.peg_slot_sac_runner import run_sac_training


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realagainst_peg_slot_adversary_sac",
)
def main(cfg):
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    run_sac_training(cfg)


if __name__ == "__main__":
    main()
