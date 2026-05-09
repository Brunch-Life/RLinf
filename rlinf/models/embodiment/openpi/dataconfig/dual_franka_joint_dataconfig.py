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
"""Data configuration for the dual-Franka joint-space SFT dataset.

Companion to ``dual_franka_rot6d_dataconfig.py``. The two share the same
underlying recording (``DualFrankaFrankyEnv`` with all observation
streams) but differ in which slice of state and which action target the
policy is trained against:

- **rot6d**: train on TCP pose (xyz + 6D rotation) per arm. Requires
  offline backfill of the action stream from joint commands to TCP
  targets.
- **joint** *(this file)*: train directly on the GELLO-side joint
  command stream. No backfill, raw 16-d actions on disk.

Actions are absolute joint-position targets (not deltas), so no
``DeltaActions`` / ``RigidBodyDeltaActions`` is applied at the data
layer — the model predicts absolute joint targets directly.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dual_franka_joint_policy


@dataclasses.dataclass(frozen=True)
class DualFrankaJointDataConfig(DataConfigFactory):
    """Data configuration for the dual-Franka joint-space SFT dataset."""

    # Default prompt used when the dataset does not carry an instruction.
    default_prompt: str | None = None

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
                dual_franka_joint_policy.DualFrankaJointInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[dual_franka_joint_policy.DualFrankaJointOutputs()],
        )

        # Joint commands are absolute targets — no delta transform.

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
