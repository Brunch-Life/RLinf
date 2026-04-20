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
import dataclasses
import pathlib

import numpy as np
import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import dual_franka_policy


@dataclasses.dataclass(frozen=True)
class DualFrankaDataConfig(DataConfigFactory):
    """Data configuration for the RLinf dual-Franka GELLO joint-space dataset."""

    # Default prompt used when the dataset does not carry an instruction.
    default_prompt: str | None = None

    # If True, convert absolute joint actions to delta actions (relative to the
    # current state at the start of the chunk). pi0 / pi05 are trained on deltas.
    extra_delta_transform: bool = True

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Rename raw parquet feature keys into the structure expected by
        # ``DualFrankaInputs``. Camera mapping is taken from
        # DUAL_FRANKA_GELLO_COLLECT.md Section 8.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "cam_base": "extra_view_image-0",
                            "cam_left_wrist": "image",
                            "cam_right_wrist": "extra_view_image-1",
                        },
                        "state": "state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                dual_franka_policy.DualFrankaInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[dual_franka_policy.DualFrankaOutputs()],
        )

        # 16-d action layout: [L7joints, L_grip_trigger, R7joints, R_grip_trigger].
        # Apply delta to the 7+7 joint entries and keep the trigger channels absolute
        # (grippers are already in a small [-1, +1] relative range).
        if self.extra_delta_transform:
            delta_action_mask = np.array(
                [True] * 7 + [False] + [True] * 7 + [False], dtype=bool
            )
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
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
