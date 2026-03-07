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

from rlinf.models.embodiment.openpi.policies import franka_policy


@dataclasses.dataclass(frozen=True)
class _SelectStateDims(_transforms.DataTransformFn):
    """Select specific dimensions from ``observation/state`` during training.

    When loading a dataset whose state vector has more dimensions than the
    policy expects (e.g. 19D full FrankaEnv state vs. 7D tcp_pose+gripper),
    this transform picks the relevant subset.  During inference the client
    already sends the correct 7D state, so the transform is a no-op when
    the input dimension matches the output length.
    """

    indices: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 0)

    def __call__(self, data: dict) -> dict:
        key = "observation/state"
        if key not in data:
            return data
        state = np.asarray(data[key])
        if state.shape[-1] == len(self.indices):
            return data
        data[key] = state[..., list(self.indices)]
        return data


@dataclasses.dataclass(frozen=True)
class CustomDataConfig(DataConfigFactory):
    """
    This config is used to configure transforms that are applied at various parts of the data pipeline.
    For your own dataset, you can copy this class and modify the transforms to match your dataset based on the
    comments below.
    """

    # If provided, will be injected into the input data if the "prompt" key is not present.
    default_prompt: str | None = None
    # Finally we will use delta actions to train, but we can input abs_action(get delta for training via abs_action-state) or delta_action(no other process)
    extra_delta_transform: bool = True  # False for additional process(abs_action - state) to get delta action for training
    # train actions using rotation_6d
    action_train_with_rotation_6d: bool = False
    # Indices into the dataset state vector to select for the 7D policy input.
    # Default (4,5,6,7,8,9,0) picks tcp_pose(6D) + gripper(1D) from the
    # 19D RealWorldEnv state (alphabetical: gripper, tcp_force, tcp_pose, tcp_torque, tcp_vel).
    # Set to None to disable (when the dataset already has 7D state).
    select_state_dims: tuple[int, ...] | None = (4, 5, 6, 7, 8, 9, 0)
    # Parquet column names of extra camera images to load from the LeRobot
    # dataset.  Typically ``("extra_image_0", "extra_image_1")`` for a
    # 3-camera setup.
    extra_image_keys: tuple[str, ...] = ()
    # Mapping of the three Pi0 image slots (base_0_rgb, left_wrist_0_rgb,
    # right_wrist_0_rgb) to observation keys.  Override this to align your
    # physical cameras with the pre-trained semantics:
    #   base_0_rgb        — pre-trained on third-person / global views
    #   left_wrist_0_rgb  — pre-trained on wrist / close-up views
    #   right_wrist_0_rgb — pre-trained on wrist / close-up views
    # For example, if your main ``image`` is a wrist camera and the extras
    # are standing (third-person) cameras, set:
    #   ("observation/extra_image_0", "observation/image", "observation/extra_image_1")
    # Use None to pad a slot with zeros and mask it out.
    pi0_slot_keys: tuple[str | None, str | None, str | None] = (
        "observation/image",
        "observation/extra_image_0",
        "observation/extra_image_1",
    )

    def generate_observations(
        image: np.ndarray, state: np.ndarray, prompt: str
    ) -> dict:
        """Creates an input example for the Franka policy."""
        return {
            "observation/image": image,
            "observation/state": state,
            "prompt": prompt,
        }

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_mapping = {
            "observation/image": "image",
            "observation/state": "state",
            "actions": "actions",
            "prompt": "prompt",
        }
        for i, col in enumerate(self.extra_image_keys):
            repack_mapping[f"observation/extra_image_{i}"] = col

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        input_transforms = []
        if self.select_state_dims is not None:
            input_transforms.append(_SelectStateDims(indices=self.select_state_dims))
        input_transforms.append(
            franka_policy.FrankaEEInputs(
                action_dim=model_config.action_dim,
                model_type=model_config.model_type,
                action_train_with_rotation_6d=self.action_train_with_rotation_6d,
                num_extra_images=len(self.extra_image_keys),
                pi0_slot_keys=self.pi0_slot_keys,
            )
        )

        data_transforms = _transforms.Group(
            inputs=input_transforms,
            outputs=[
                franka_policy.FrankaEEOutputs(
                    action_train_with_rotation_6d=self.action_train_with_rotation_6d
                )
            ],
        )

        if not self.extra_delta_transform:  # for abs_action
            delta_action_mask = _transforms.make_bool_mask(
                9, -1
            )  # [True]x9 + [False]x1, [x,y,z,rotation_6d,gripper] for 10 dim
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
        )
