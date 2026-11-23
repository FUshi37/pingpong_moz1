"""Collection archives remain lossless, bounded and immutable on publication."""

from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import artifacts
from world_model.resources import StorageBudget, StorageLimitError


def test_numeric_chunk_roundtrip_preserves_sample_clocks_masks_and_shapes(tmp_path):
    budget = StorageBudget(tmp_path, 1024**2, 0)
    values = dict(ball_position_m=np.arange(120, dtype=np.float32).reshape(5, 8, 3),
                  sample_time_us=np.arange(40, dtype=np.int32).reshape(5, 8)*5000,
                  valid=np.ones((5, 8), dtype=bool), shared_phase=np.array(3, np.int32))
    path = tmp_path/'chunk.npz'
    manifest = artifacts.write_array_archive_unique(path, values, budget)
    restored = artifacts.read_array_archive(path, manifest)
    assert set(restored) == set(values)
    for name in values:
        assert restored[name].dtype == values[name].dtype
        np.testing.assert_array_equal(restored[name], values[name])
    with pytest.raises(FileExistsError):
        artifacts.write_array_archive_unique(path, values, budget)
    assert not list(tmp_path.glob('.wmj-write-*'))


def test_archive_integrity_precedes_loading_and_rejects_pickle(tmp_path):
    budget = StorageBudget(tmp_path, 1024**2, 0)
    path = tmp_path/'chunk.npz'
    manifest = artifacts.write_array_archive_unique(path, {'x': np.ones(3)}, budget)
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='integrity'):
        artifacts.read_array_archive(path, manifest)
    with pytest.raises(ValueError, match='numeric'):
        artifacts.write_array_archive_unique(tmp_path/'object.npz',
                                              {'x': np.array([{}], dtype=object)}, budget)


def test_disk_budget_failure_leaves_existing_data_intact(tmp_path):
    existing = tmp_path/'protected'
    existing.write_bytes(b'prior evidence')
    budget = StorageBudget(tmp_path, 8192, 0)
    with pytest.raises(StorageLimitError):
        artifacts.write_array_archive_unique(tmp_path/'chunk.npz',
                                              {'x': np.ones(10000, np.float32)}, budget)
    assert existing.read_bytes() == b'prior evidence'
    assert not (tmp_path/'chunk.npz').exists()


def test_paths_and_missing_label_values_cannot_escape_archive_contract(tmp_path):
    budget = StorageBudget(tmp_path/'data', 1024**2, 0)
    with pytest.raises(ValueError, match='root'):
        artifacts.write_array_archive_unique(tmp_path/'escape.npz', {'x': np.zeros(1)}, budget)
    with pytest.raises(ValueError, match='finite'):
        artifacts.write_array_archive_unique(budget.root/'bad.npz', {'x': np.array([np.nan])}, budget)
    with pytest.raises(ValueError, match='name'):
        artifacts.write_array_archive_unique(budget.root/'bad.npz', {'../escape': np.ones(1)}, budget)


def test_selected_stream_loading_still_authenticates_the_whole_archive(tmp_path):
    budget = StorageBudget(tmp_path, 1024**2, 0)
    path = tmp_path/'streams.npz'
    manifest = artifacts.write_array_archive_unique(path,
        {'q': np.ones((10, 7), np.float32), 'unused_trace': np.zeros(1000, np.float32)}, budget)
    restored = artifacts.read_array_archive(path, manifest, names={'q'}, release_cache=False)
    assert set(restored) == {'q'}
    with pytest.raises(ValueError, match='stream'):
        artifacts.read_array_archive(path, manifest, names={'missing'})
    with path.open('ab') as stream:
        stream.write(b'bad-unused-field-or-tail')
    with pytest.raises(ValueError, match='integrity'):
        artifacts.read_array_archive(path, manifest, names={'q'})
