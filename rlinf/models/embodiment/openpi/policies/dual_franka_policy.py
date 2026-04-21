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
"""Policy transforms for the RLinf dual-Franka GELLO joint-space dataset.

Dataset layout (see DUAL_FRANKA_GELLO_COLLECT.md):
- state (68 dims): [0:2] gripper widths (L,R), [2:9] left arm 7 joints,
  [9:16] right arm 7 joints, [16:68] velocities/forces/poses (unused by pi0).
- actions (16 dims): [0:7] left 7 joints, [7] left gripper trigger in [-1,+1],
  [8:15] right 7 joints, [15] right gripper trigger.
- images: ``cam_base`` (3rd-person), ``cam_left_wrist``, ``cam_right_wrist``.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# State slice we feed the model: 16 dims laid out to mirror action order.
_STATE_SLICE_DIM = 16

# Stack order ``_wrap_obs`` must produce (alphabetical after removing main =
# left wrist). ``_extract_extra_views`` asserts this so a rig rename fails loud.
_EXPECTED_EXTRA_VIEW_ORDER = ("base_0_rgb", "right_wrist_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _rearrange_state(state: np.ndarray) -> np.ndarray:
    # Raw [0:16] = [L_grip, R_grip, L7joints, R7joints]. Reorder to match the
    # 16-d action layout [L7joints, L_grip, R7joints, R_grip].
    s = np.asarray(state)[..., :_STATE_SLICE_DIM]
    return np.concatenate(
        [s[..., 2:9], s[..., 0:1], s[..., 9:16], s[..., 1:2]], axis=-1
    )


def _extract_extra_views(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(base, right_wrist)`` from stacked (inference) or split (training) layout."""
    stacked = data.get("observation/extra_view_image")
    if stacked is not None:
        names = data.get("observation/extra_view_image_names")
        if names is not None and tuple(names) != _EXPECTED_EXTRA_VIEW_ORDER:
            raise AssertionError(
                f"extra-view camera order drifted: got {tuple(names)}, "
                f"expected {_EXPECTED_EXTRA_VIEW_ORDER}."
            )
        extra = np.asarray(stacked)
        return _parse_image(extra[0]), _parse_image(extra[1])
    return (
        _parse_image(data["observation/extra_view_image-0"]),
        _parse_image(data["observation/extra_view_image-1"]),
    )


@dataclasses.dataclass(frozen=True)
class DualFrankaInputs(transforms.DataTransformFn):
    """Feeds dual-Franka observations into the pi0 / pi05 model."""

    # Pads state and actions up to this dimension. Typically 32 (pi0 default).
    action_dim: int

    # Determines which model will be used. Do not change.
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        state = _rearrange_state(data["observation/state"])
        state = transforms.pad_to_dim(state, self.action_dim)

        # Main view is always the left wrist (main_image_key convention).
        left_wrist_image = _parse_image(data["observation/image"])
        base_image, right_wrist_image = _extract_extra_views(data)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        if "actions" in data:
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        if "prompt" in data:
            prompt = data["prompt"]
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8")
            inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class DualFrankaOutputs(transforms.DataTransformFn):
    """Recovers the 16-d dual-Franka action from the padded model output."""

    output_action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
