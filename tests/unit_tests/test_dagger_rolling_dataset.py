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

from rlinf.data.datasets.dagger.dataset import RollingLeRobotDataset


def _filtered_dataset(tmp_path, window_size: int) -> RollingLeRobotDataset:
    return RollingLeRobotDataset(
        root_dir=tmp_path,
        require_all_intervene=True,
        window_size=window_size,
        in_memory_mode=True,
    )


def test_filtered_window_does_not_evict_before_capacity(tmp_path):
    dataset = _filtered_dataset(tmp_path, window_size=50_000)
    shard = tmp_path / "rank_0" / "id_0"
    store = object()
    dataset._sub_datasets = [shard]
    dataset._cumulative_lengths = [0, 30]
    dataset._in_memory_shards = {shard: store}

    for valid_indices in ([], [20]):
        dataset._valid_physical_indices = valid_indices
        dataset._update_window_sampling_bounds()
        assert dataset._window_physical_start == 0
        assert dataset._evict_stale_shards() == 0
        assert dataset._in_memory_shards[shard] is store


def test_filtered_window_evicts_only_after_logical_samples_are_trimmed(tmp_path):
    dataset = _filtered_dataset(tmp_path, window_size=2)
    old_shard = tmp_path / "rank_0" / "id_0"
    live_shard = tmp_path / "rank_0" / "id_1"
    dataset._sub_datasets = [old_shard, live_shard]
    dataset._cumulative_lengths = [0, 10, 20]
    dataset._in_memory_shards = {old_shard: object(), live_shard: object()}
    dataset._valid_physical_indices = [2, 12, 15]

    dataset._update_window_sampling_bounds()

    assert dataset._window_valid_slice_lo == 1
    assert dataset._window_physical_start == 12
    assert dataset._evict_stale_shards() == 1
    assert old_shard not in dataset._in_memory_shards
    assert live_shard in dataset._in_memory_shards
