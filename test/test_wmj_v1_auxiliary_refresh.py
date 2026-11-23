"""Refresh keeps source families, balanced domains and inherited normalization."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.auxiliary_refresh import family_ids, refresh_indexes, validation_batches


def test_cloned_sources_share_auxiliary_family_and_cannot_cross_partitions():
    groups = [dict(root_seed_family=root, split=split) for root, split in
              [('a', 'train'), ('a', 'train'), ('b', 'validation')]]
    np.testing.assert_array_equal(family_ids(groups), [0, 0, 1])
    groups[1]['split'] = 'validation'
    with pytest.raises(ValueError, match='partition'):
        family_ids(groups)


def test_refresh_batches_are_half_old_half_new_and_exclude_validation():
    old = dict(split=np.array([0, 0, 1]), overlap=np.array([True, False, True]),
               family_id=np.array([0, 1, 2]))
    new = dict(split=np.array([0, 1, 0]), overlap=np.array([False, False, True]),
               family_id=np.array([0, 1, 2]))
    first, second = refresh_indexes(np.random.default_rng(4), old, new, batch_size=512)
    assert len(first) == len(second) == 256
    for rows, indexes in [(old, first), (new, second)]:
        assert np.all(rows['split'][indexes] == 0)
        assert rows['overlap'][indexes].sum() == 128


def test_refresh_cannot_invent_a_missing_contact_bin():
    rows = dict(split=np.array([0]), overlap=np.array([False]), family_id=np.array([0]))
    with pytest.raises(ValueError, match='contact'):
        refresh_indexes(np.random.default_rng(5), rows, rows, batch_size=512)


def test_paired_world_validation_excludes_train_test_and_marks_padding():
    groups = [dict(split=s) for s in ['train','validation','test','validation','validation']]
    batches = validation_batches(groups, width=4)
    assert len(batches) == 1 and batches[0]['useful'] == 3
    np.testing.assert_array_equal(batches[0]['lanes'], [1,3,4,1])
