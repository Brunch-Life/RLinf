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
"""Data configuration for the dual-Franka rot6d_v1 SFT dataset.

Replaces ``dual_franka_dataconfig.py`` (euler + openpi ``DeltaActions``).
Delta / absolute transforms are now SE(3) body-frame rigid-body ops
(see :mod:`rlinf.models.embodiment.openpi.transforms.rigid_body_delta`),
so rotations compose correctly under matrix multiplication rather than
component-wise subtraction.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dual_franka_rot6d_policy
from rlinf.models.embodiment.openpi.transforms import (
    DUAL_ARM_ROT6D_LAYOUT,
    RigidBodyAbsoluteActions,
    RigidBodyDeltaActions,
)


@dataclasses.dataclass(frozen=True)
class DualFrankaRot6dDataConfig(DataConfigFactory):
    """Data configuration for the dual-Franka rot6d_v1 SFT dataset."""

    # Default prompt used when the dataset does not carry an instruction.
    default_prompt: str | None = None

    # If True, dataset actions (absolute targets) are converted to body-frame
    # SE(3) deltas at training time via RigidBodyDeltaActions, and recovered
    # via RigidBodyAbsoluteActions at inference. pi0 / pi05 are trained on
    # deltas so the default is True.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "image",
                        "observation/extra_view_image-0": "extra_view_image-0",
                        "observation/extra_view_image-1": "extra_view_image-1",
                        "observation/state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dual_franka_rot6d_policy.DualFrankaRot6dInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[dual_franka_rot6d_policy.DualFrankaRot6dOutputs()],
        )

        if self.extra_delta_transform:
            # Rotation must compose via SE(3), not scalar subtraction —
            # openpi's DeltaActions is not usable on rot6d slots.
            data_transforms = data_transforms.push(
                inputs=[RigidBodyDeltaActions(DUAL_ARM_ROT6D_LAYOUT)],
                outputs=[RigidBodyAbsoluteActions(DUAL_ARM_ROT6D_LAYOUT)],
            )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("actions",),
        )
