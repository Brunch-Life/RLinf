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
"""Policy transforms for the dual-Franka joint-space SFT dataset.

Raw joint-space LeRobot dataset layout (no backfill needed; this is the
collection format that ``DualFrankaJointEnv`` produces directly):

- **state[0:16]** (env ``_wrap_obs`` driven by
  :class:`DualFrankaJointEnv`'s ``STATE_LAYOUT = (gripper_position,
  joint_position, joint_velocity, ...)``):
  ``[L_grip, R_grip, L_jpos(7), R_jpos(7)]``. Slots ``[16:]`` carry the
  legacy joint_velocity / tcp_force / tcp_pose / tcp_torque / tcp_vel
  tail; pi05 ignores them after ``_rearrange_state``.
- **actions[0:16]** (policy-facing, exactly the GELLO-side joint command
  stream the operator was sending):
  ``[L_jpos_action(7), L_grip_trigger(1),
     R_jpos_action(7), R_grip_trigger(1)]``.
- **images**: ``cam_base`` (3rd person), ``cam_left_wrist`` (main view),
  ``cam_right_wrist``.

Unlike ``dual_franka_rot6d_policy``, joint actions are absolute
joint-position targets (commanded, not delta). No SE(3) or
component-wise delta transform is applied at the data layer — the model
learns to predict absolute joint targets directly. This matches how the
joint env's ``joint_action_mode == "absolute"`` consumer operates at
inference time.
"""

import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

# pi05 real state dim after _rearrange_state. Downstream pad_to_dim takes
# this up to model_config.action_dim (typically 32).
_STATE_SLICE_DIM = 16

# Stack order ``_wrap_obs`` produces (alphabetical after removing main =
# left wrist). ``_extract_extra_views`` asserts this so a rig rename
# fails loud instead of silently swapping camera meanings.
_EXPECTED_EXTRA_VIEW_ORDER = ("base_0_rgb", "right_wrist_0_rgb")


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _rearrange_state(state: np.ndarray) -> np.ndarray:
    """Reorder env-side state prefix into the training layout.

    Input  (env side, driven by ``DualFrankaJointEnv.STATE_LAYOUT``):
        state[:16] = [L_grip, R_grip, L_jpos(7), R_jpos(7)]

    Output (policy side, per-arm grouped, matches actions on disk):
        [L_jpos(7), L_grip(1), R_jpos(7), R_grip(1)]
    """
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
class DualFrankaJointInputs(transforms.DataTransformFn):
    """Feeds dual-Franka joint observations into pi0 / pi05."""

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
class DualFrankaJointOutputs(transforms.DataTransformFn):
    """Recovers the 16-d dual-Franka joint action from the padded model output."""

    output_action_dim: int = 16

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, : self.output_action_dim])}
