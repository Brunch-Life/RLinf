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

"""Collection schedules for online DAgger datasets."""

from __future__ import annotations

import math


def linear_episode_target(
    *,
    max_episodes: int,
    initial_episodes: int,
    current_step: int,
    total_steps: int,
    end_fraction: float,
) -> int:
    """Return the cumulative episode target for a linear collection ramp."""
    if max_episodes <= 0:
        raise ValueError("max_episodes must be positive for a collection schedule")
    if not 0 < initial_episodes <= max_episodes:
        raise ValueError("initial_episodes must be in [1, max_episodes]")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 < end_fraction <= 1.0:
        raise ValueError("end_fraction must be in (0, 1]")

    end_step = max(1, math.ceil(total_steps * end_fraction))
    scheduled_step = min(max(int(current_step), 0), end_step)
    additional_episodes = (max_episodes - initial_episodes) * scheduled_step // end_step
    return initial_episodes + additional_episodes
