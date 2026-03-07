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

import einops
import numpy as np
import torch
from openpi import transforms
from openpi.models import model as _model


def make_franka_example() -> dict:
    """Creates a random input example for the Panda policy."""
    return {
        "observation/image": np.random.randint(256, size=(480, 640, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(480, 640, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(7),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FrankaEEOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False

    def __call__(self, data: dict) -> dict:
        return {
            "actions": np.asarray(data["actions"][:, :7])
        }  # use abs actions [x,y,z,rx,ry,rz,gripper] for Franka


@dataclasses.dataclass(frozen=True)
class FrankaEEInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int  # default is defined in the model config(Pi0Config), 32.

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Whether to train actions using rotation_6d or not.
    action_train_with_rotation_6d: bool = False

    # Number of extra camera images available in the dataset (0, 1, or 2).
    # When >0, ``observation/extra_image_0`` (and ``_1``) are read from the
    # data dict and assigned to the Pi0 wrist image slots instead of zeros.
    num_extra_images: int = 0

    # Mapping from Pi0 image slots to observation keys.  The three entries
    # correspond to (base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb).
    # Override this to align your camera semantics with the pre-trained
    # model — e.g. if your main ``image`` column is a wrist camera and the
    # extra columns are third-person views, swap them here so that the
    # third-person view fills ``base_0_rgb`` (pre-trained on global views).
    # ``None`` entries are padded with zeros and masked out.
    pi0_slot_keys: tuple[str | None, str | None, str | None] = (
        "observation/image",
        "observation/extra_image_0",
        "observation/extra_image_1",
    )

    def __call__(self, data: dict) -> dict:
        assert data["observation/state"].shape == (7,), (
            f"Expected state shape (7,), got {data['observation/state'].shape}"
        )
        if isinstance(data["observation/state"], np.ndarray):
            data["observation/state"] = torch.from_numpy(
                data["observation/state"]
            ).float()

        state = data["observation/state"]
        state = transforms.pad_to_dim(state, self.action_dim)

        # Resolve each Pi0 image slot from the configured observation keys.
        slot_images: list[np.ndarray | None] = []
        for key in self.pi0_slot_keys:
            raw = data.get(key) if key is not None else None
            slot_images.append(_parse_image(raw) if raw is not None else None)

        # Build a reference image for zero-padding (use the first non-None).
        ref = next((img for img in slot_images if img is not None), None)
        if ref is None:
            raise ValueError("At least one image must be provided.")

        resolved = tuple(img if img is not None else np.zeros_like(ref) for img in slot_images)
        masks = tuple(np.True_ if img is not None else np.False_ for img in slot_images)

        if self.model_type in (_model.ModelType.PI0, _model.ModelType.PI05):
            names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
            image_masks = masks
        elif self.model_type == _model.ModelType.PI0_FAST:
            names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
            image_masks = (np.True_, np.True_, np.True_)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, resolved, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            assert len(data["actions"].shape) == 2 and data["actions"].shape[-1] == 7, (
                f"Expected actions shape (N, 7), got {data['actions'].shape}"
            )
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs
