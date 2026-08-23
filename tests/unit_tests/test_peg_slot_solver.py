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

import math

import pytest
import torch

from rlinf.envs.maniskill.tasks.peg_slot import _tilt_error_to_down


def _downward_rotation() -> torch.Tensor:
    return torch.diag(torch.tensor([1.0, -1.0, -1.0]))


def _x_rotation(angle: float) -> torch.Tensor:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, -sine],
            [0.0, sine, cosine],
        ]
    )


@pytest.mark.parametrize("angle", [math.radians(25.0), math.radians(-25.0)])
def test_tilt_error_corrects_roll_in_world_frame(angle: float):
    peg_rotation = _x_rotation(angle) @ _downward_rotation()

    error = _tilt_error_to_down(peg_rotation.unsqueeze(0))[0]

    expected = torch.tensor([-angle, 0.0, 0.0])
    torch.testing.assert_close(error, expected, atol=1e-6, rtol=1e-6)


def test_tilt_error_is_zero_when_aligned():
    error = _tilt_error_to_down(_downward_rotation().unsqueeze(0))[0]

    torch.testing.assert_close(error, torch.zeros(3), atol=1e-7, rtol=0.0)


def test_tilt_error_handles_antiparallel_axis():
    error = _tilt_error_to_down(torch.eye(3).unsqueeze(0))[0]

    assert torch.isfinite(error).all()
    torch.testing.assert_close(
        error, torch.tensor([math.pi, 0.0, 0.0]), atol=1e-6, rtol=1e-6
    )
