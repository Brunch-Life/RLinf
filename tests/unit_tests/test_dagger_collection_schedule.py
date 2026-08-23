# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from rlinf.data.datasets.dagger.schedule import linear_episode_target


@pytest.mark.parametrize(
    ("step", "expected"),
    [(-1, 16), (0, 16), (2_500, 258), (4_999, 499), (5_000, 500), (9_999, 500)],
)
def test_linear_episode_target_reaches_limit_halfway(step: int, expected: int):
    assert (
        linear_episode_target(
            max_episodes=500,
            initial_episodes=16,
            current_step=step,
            total_steps=10_000,
            end_fraction=0.5,
        )
        == expected
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_episodes": 0},
        {"initial_episodes": 0},
        {"initial_episodes": 501},
        {"total_steps": 0},
        {"end_fraction": 0.0},
        {"end_fraction": 1.1},
    ],
)
def test_linear_episode_target_rejects_invalid_configuration(kwargs: dict):
    valid_kwargs = {
        "max_episodes": 500,
        "initial_episodes": 16,
        "current_step": 0,
        "total_steps": 10_000,
        "end_fraction": 0.5,
    }
    valid_kwargs.update(kwargs)

    with pytest.raises(ValueError):
        linear_episode_target(**valid_kwargs)
