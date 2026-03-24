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

from pathlib import Path

import numpy as np
import pytest


def _import_realworld_openpi_runtime():
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.openpi.policies.realworld_pnp_policy import (
        RealworldPnPInputs,
        RealworldPnPOutputs,
    )

    return get_openpi_config, RealworldPnPInputs, RealworldPnPOutputs


@pytest.mark.parametrize(
    "extra_view_key",
    ("observation/extra_view_images", "observation/extra_view_image"),
)
def test_realworld_inputs_accept_rollout_extra_view_aliases(extra_view_key: str):
    _, RealworldPnPInputs, _ = _import_realworld_openpi_runtime()

    transform = RealworldPnPInputs(action_dim=7)

    main = np.full((3, 4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((3, 4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((3, 4, 4, 3), 33, dtype=np.uint8)
    extras = np.stack([extra0, extra1], axis=1)
    state = np.arange(57, dtype=np.float32).reshape(3, 19)

    outputs = transform(
        {
            "observation/image": main,
            extra_view_key: extras,
            "observation/state": state,
            "prompt": ["duck-a", "duck-b"],
        }
    )

    expected_state = state[:, [4, 5, 6, 7, 8, 9, 0]]
    np.testing.assert_array_equal(np.asarray(outputs["state"]), expected_state)
    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], extra0)
    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)
    assert outputs["prompt"] == ["duck-a", "duck-b"]
    assert outputs["image_mask"]["base_0_rgb"]
    assert outputs["image_mask"]["left_wrist_0_rgb"]
    assert outputs["image_mask"]["right_wrist_0_rgb"]


def test_realworld_inputs_accept_legacy_unbatched_stacked_extra_view_alias():
    _, RealworldPnPInputs, _ = _import_realworld_openpi_runtime()

    transform = RealworldPnPInputs(action_dim=7)

    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((4, 4, 3), 33, dtype=np.uint8)
    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_view_image": np.stack([extra0, extra1], axis=0),
            "observation/state": np.arange(7, dtype=np.float32),
            "prompt": "duck",
        }
    )

    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], extra0)
    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)


def test_realworld_inputs_support_sparse_slot_masks_and_zero_fill():
    _, RealworldPnPInputs, RealworldPnPOutputs = _import_realworld_openpi_runtime()

    transform = RealworldPnPInputs(
        action_dim=7,
        state_indices=None,
        pi0_slot_keys=(None, "observation/image", None),
    )
    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    outputs = transform(
        {
            "observation/image": main,
            "observation/state": np.arange(7, dtype=np.float32),
            "prompt": b"demo",
        }
    )

    np.testing.assert_array_equal(np.asarray(outputs["state"]), np.arange(7))
    np.testing.assert_array_equal(outputs["image"]["left_wrist_0_rgb"], main)
    np.testing.assert_array_equal(
        outputs["image"]["base_0_rgb"],
        np.zeros_like(main),
    )
    np.testing.assert_array_equal(
        outputs["image"]["right_wrist_0_rgb"],
        np.zeros_like(main),
    )
    assert outputs["prompt"] == "demo"
    assert not outputs["image_mask"]["base_0_rgb"]
    assert outputs["image_mask"]["left_wrist_0_rgb"]
    assert not outputs["image_mask"]["right_wrist_0_rgb"]

    post = RealworldPnPOutputs()
    truncated = post({"actions": np.arange(20, dtype=np.float32).reshape(2, 10)})
    np.testing.assert_array_equal(
        truncated["actions"],
        np.arange(14, dtype=np.float32).reshape(2, 7),
    )


def test_realworld_inputs_accept_direct_extra_image_keys_and_runtime_guards():
    _, RealworldPnPInputs, _ = _import_realworld_openpi_runtime()

    transform = RealworldPnPInputs(action_dim=7)
    main = np.full((4, 4, 3), 11, dtype=np.uint8)
    extra0 = np.full((4, 4, 3), 22, dtype=np.uint8)
    extra1 = np.full((4, 4, 3), 33, dtype=np.uint8)
    outputs = transform(
        {
            "observation/image": main,
            "observation/extra_image_0": extra0,
            "observation/extra_image_1": extra1,
            "observation/state": np.arange(7, dtype=np.float32),
            "prompt": "duck",
        }
    )

    np.testing.assert_array_equal(outputs["image"]["base_0_rgb"], extra0)
    np.testing.assert_array_equal(outputs["image"]["right_wrist_0_rgb"], extra1)

    with pytest.raises(ValueError, match="At least one image must be provided"):
        transform({"observation/state": np.arange(7, dtype=np.float32)})

    with pytest.raises(AssertionError, match="Expected actions shape"):
        transform(
            {
                "observation/image": main,
                "observation/state": np.arange(7, dtype=np.float32),
                "actions": np.zeros((7,), dtype=np.float32),
            }
        )


def test_get_openpi_config_registers_realworld_pnp_and_applies_overrides():
    get_openpi_config, _, _ = _import_realworld_openpi_runtime()

    config = get_openpi_config(
        "pi0_realworld_pnp",
        model_path="/tmp/fake-openpi-checkpoint",
        data_kwargs={"state_indices": None},
    )

    assert config.name == "pi0_realworld_pnp"
    assert config.pytorch_weight_path == "/tmp/fake-openpi-checkpoint"
    assert config.data.extra_delta_transform is True
    assert config.data.state_indices is None
    assert config.data.pi0_slot_keys == (
        "observation/extra_image_0",
        "observation/image",
        "observation/extra_image_1",
    )

    default_config = get_openpi_config("pi0_realworld_pnp")
    assert default_config.data.state_indices == (4, 5, 6, 7, 8, 9, 0)


def test_realworld_duck_place_eval_config_composes(monkeypatch: pytest.MonkeyPatch):
    pytest.importorskip("hydra")

    from hydra import compose, initialize_config_dir

    from rlinf.config import validate_cfg

    repo_root = Path(__file__).resolve().parents[2]
    embodied_dir = repo_root / "examples" / "embodiment"
    monkeypatch.setenv("EMBODIED_PATH", str(embodied_dir))

    with initialize_config_dir(
        version_base="1.1", config_dir=str(embodied_dir / "config")
    ):
        cfg = compose(config_name="realworld_duck_place_pi0_zed_robotiq_eval")

    cfg.runner.only_eval = True
    cfg = validate_cfg(cfg)

    assert cfg.env.eval.init_params.id == "DataCollectSFTEnv-v1"
    assert cfg.actor.model.openpi.config_name == "pi0_realworld_pnp"
    assert cfg.runner.only_eval is True
    assert cfg.rollout.model.model_path != "/path/to/model"
    assert cfg.actor.model.model_path == cfg.rollout.model.model_path
