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

from __future__ import annotations

from typing import Any, Mapping, Optional

import gymnasium as gym
from gymnasium.envs.registration import register

from rlinf.envs.realworld.common.wrappers import (
    DualGelloIntervention,
    DualQuat2EulerWrapper,
    DualRelativeFrame,
    DualSpacemouseIntervention,
    GelloIntervention,
    GripperCloseEnv,
    KeyboardRewardDoneMultiStageWrapper,
    KeyboardRewardDoneWrapper,
    Quat2EulerWrapper,
    RelativeFrame,
    SpacemouseIntervention,
)
from rlinf.envs.realworld.franka.dual_franka_env import DualFrankaEnv as DualFrankaEnv
from rlinf.envs.realworld.franka.franka_env import FrankaEnv as FrankaEnv
from rlinf.envs.realworld.franka.tasks.bottle import BottleEnv as BottleEnv
from rlinf.envs.realworld.franka.tasks.franka_bin_relocation import (
    FrankaBinRelocationEnv as FrankaBinRelocationEnv,
)
from rlinf.envs.realworld.franka.tasks.peg_insertion_env import (
    PegInsertionEnv as PegInsertionEnv,
)


def _apply_common_wrappers(
    env: gym.Env, env_cfg: Optional[Mapping[str, Any]]
) -> gym.Env:
    """Attach the single/dual franka wrapper stack driven by ``env_cfg``.

    ``env_cfg`` is the env-level config (``self.cfg`` in RealWorldEnv). The same
    flags drive both single- and dual-arm layouts; the wrapper class is picked
    from ``IS_DUAL_ARM`` on the unwrapped env so users do not need a parallel
    ``use_dual_*`` flag.
    """
    cfg: Mapping[str, Any] = env_cfg if env_cfg is not None else {}
    is_dual = getattr(env.unwrapped, "IS_DUAL_ARM", False)

    no_gripper = cfg.get("no_gripper", True)
    if no_gripper:
        if is_dual:
            # DualGripperCloseEnv (the 12D→14D adapter mirroring GripperCloseEnv)
            # is not yet implemented; without this guard, teleop producing a 12D
            # action would surface as an opaque reshape(2, 7) error inside
            # DualFrankaEnv.step.
            raise NotImplementedError(
                "no_gripper=True is not yet supported for dual-arm envs: "
                "DualGripperCloseEnv (the 12D→14D adapter mirroring "
                "GripperCloseEnv) is not implemented in this PR. "
                "Set env.eval.no_gripper=False (or env.train.no_gripper=False)."
            )
        env = GripperCloseEnv(env)

    use_spacemouse = cfg.get("use_spacemouse", True)
    use_gello = cfg.get("use_gello", False)
    if use_spacemouse and use_gello:
        raise ValueError(
            "Only one teleop mode can be active at a time. "
            "Set exactly one of use_spacemouse, use_gello to True."
        )

    gripper_enabled = not no_gripper

    if not env.config.is_dummy and use_spacemouse:
        spacemouse_cls = (
            DualSpacemouseIntervention if is_dual else SpacemouseIntervention
        )
        env = spacemouse_cls(env, gripper_enabled=gripper_enabled)

    if not env.config.is_dummy and use_gello:
        if is_dual:
            left_port = cfg.get("left_gello_port", None)
            right_port = cfg.get("right_gello_port", None)
            if left_port is None or right_port is None:
                raise ValueError(
                    "use_gello=True on a dual-arm env requires both "
                    "left_gello_port and right_gello_port to be set in "
                    "the env config."
                )
            env = DualGelloIntervention(
                env,
                left_port=left_port,
                right_port=right_port,
                gripper_enabled=gripper_enabled,
            )
        else:
            gello_port = cfg.get("gello_port", None)
            if gello_port is None:
                raise ValueError(
                    "use_gello is True but gello_port is not set in the env config. "
                    "Please set env.eval.gello_port (or env.train.gello_port) to the "
                    "serial port of your GELLO device."
                )
            env = GelloIntervention(
                env, port=gello_port, gripper_enabled=gripper_enabled
            )

    keyboard_reward_wrapper = cfg.get("keyboard_reward_wrapper", None)
    if not env.config.is_dummy and keyboard_reward_wrapper:
        if keyboard_reward_wrapper == "multi_stage":
            env = KeyboardRewardDoneMultiStageWrapper(env)
        elif keyboard_reward_wrapper == "single_stage":
            env = KeyboardRewardDoneWrapper(env)

    if cfg.get("use_relative_frame", True):
        env = DualRelativeFrame(env) if is_dual else RelativeFrame(env)
    env = DualQuat2EulerWrapper(env) if is_dual else Quat2EulerWrapper(env)
    return env


def create_franka_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    env = FrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_common_wrappers(env, env_cfg)


def create_dual_franka_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    env = DualFrankaEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_common_wrappers(env, env_cfg)


def create_peg_insertion_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    env = PegInsertionEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_common_wrappers(env, env_cfg)


def create_franka_bin_relocation_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    env = FrankaBinRelocationEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_common_wrappers(env, env_cfg)


def create_bottle_env(
    override_cfg: dict[str, Any],
    worker_info: Any,
    hardware_info: Any,
    env_idx: int,
    env_cfg: Optional[Mapping[str, Any]] = None,
) -> gym.Env:
    env = BottleEnv(
        override_cfg=override_cfg,
        worker_info=worker_info,
        hardware_info=hardware_info,
        env_idx=env_idx,
    )
    return _apply_common_wrappers(env, env_cfg)


register(
    id="FrankaEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:create_franka_env",
)

register(
    id="DualFrankaEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:create_dual_franka_env",
)

register(
    id="PegInsertionEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:create_peg_insertion_env",
)

register(
    id="FrankaBinRelocationEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:create_franka_bin_relocation_env",
)

register(
    id="BottleEnv-v1",
    entry_point="rlinf.envs.realworld.franka.tasks:create_bottle_env",
)
