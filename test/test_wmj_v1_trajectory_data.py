"""Continuous teacher streams reconstruct windows without duplicating history."""

from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import trajectory_data
from world_model.resources import StorageBudget


def arrays(steps=8, lanes=2):
    states = {'q_rad': np.arange((steps+1)*lanes*7, dtype=np.float32).reshape(steps+1, lanes, 7),
              'time_us': np.tile(np.arange(steps+1, dtype=np.int32)[:, None]*5000, (1, lanes)),
              'ball_source_time_us': np.zeros((steps+1, lanes), np.int32)}
    actions = {name: np.full((steps, lanes, 7), value, np.float32)
               for name, value in [('raw_normalized', 1.2), ('clipped_normalized', 1.),
                                    ('projected_normalized', .8), ('accepted_normalized', .8)]}
    substeps = {'q_after': np.zeros((steps, 5, lanes, 7), np.float32)}
    valid = np.ones((steps, lanes), bool)
    groups = [dict(root_seed_family=f'root-{i}', episode='e0', session=f'session-{i}')
              for i in range(lanes)]
    return states, actions, substeps, valid, groups


def create(tmp_path):
    budget = StorageBudget(tmp_path, 8*1024**2, 0)
    return trajectory_data.TrajectoryDataset.create(tmp_path/'dataset', budget,
        identity=dict(config_sha256='a'*64, geometry_sha256='b'*64,
                      control_sha256='c'*64, contact_sha256='d'*64,
                      source_commit='e'*40), split_version='wmj_group_split_v1')


def test_chunk_windows_preserve_endpoint_substep_and_action_alignment(tmp_path):
    dataset = create(tmp_path)
    states, actions, substeps, valid, groups = arrays()
    record = dataset.append_chunk('round000', states=states, actions=actions,
        substeps=substeps, valid=valid, groups=groups,
        seed_ref={'manifest': 'seed_000.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    loaded = trajectory_data.TrajectoryDataset.open(dataset.root, dataset.budget)
    split = record['groups'][0]['split']
    window = loaded.read_window('round000', lane=0, start=2, length=4,
                                split=split, purpose='final_evaluation')
    np.testing.assert_array_equal(window['states']['q_rad'], states['q_rad'][2:7, 0])
    assert window['substeps']['q_after'].shape == (4, 5, 7)
    np.testing.assert_array_equal(window['actions']['raw_normalized'], actions['raw_normalized'][2:6, 0])
    assert window['seed_ref']['manifest'] == 'seed_000.json'
    assert window['replay_control_steps_before_window'] == 2
    assert window['origin'] == 'mjx_teacher'
    assert record['transitions'] == 16
    assert not any('history' in name for name in record['archive']['arrays'])


def test_group_splits_and_test_access_survive_reopen(tmp_path):
    dataset = create(tmp_path)
    states, actions, substeps, valid, groups = arrays(lanes=16)
    first = dataset.append_chunk('round000', states=states, actions=actions,
        substeps=substeps, valid=valid, groups=groups,
        seed_ref={'manifest': 'seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    dataset = trajectory_data.TrajectoryDataset.open(dataset.root, dataset.budget)
    branch_groups = [{**group, 'episode': 'branch', 'session': f'branch-{i}'}
                     for i, group in enumerate(groups)]
    second = dataset.append_chunk('round001', states=states, actions=actions,
        substeps=substeps, valid=valid, groups=branch_groups,
        seed_ref={'manifest': 'branch_seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    assert [g['split'] for g in first['groups']] == [g['split'] for g in second['groups']]
    test_lane = next(i for i, g in enumerate(first['groups']) if g['split'] == 'test')
    for purpose in ('model_fit', 'active_query', 'normalization'):
        with pytest.raises(ValueError, match='test'):
            dataset.read_window('round000', lane=test_lane, start=0, length=4,
                                split='test', purpose=purpose)
    with pytest.raises(ValueError, match='split'):
        dataset.read_window('round000', lane=test_lane, start=0, length=4,
                            split='train', purpose='model_fit')


def test_invalid_padding_is_kept_and_wrong_clock_or_shape_is_rejected(tmp_path):
    dataset = create(tmp_path)
    states, actions, substeps, valid, groups = arrays()
    valid[3:, 0] = False
    record = dataset.append_chunk('padded', states=states, actions=actions,
        substeps=substeps, valid=valid, groups=groups,
        seed_ref={'manifest': 'seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    assert record['valid_transitions'] == int(valid.sum())
    states['time_us'][4, 0] += 1
    with pytest.raises(ValueError, match='5 ms'):
        dataset.append_chunk('bad', states=states, actions=actions, substeps=substeps,
            valid=valid, groups=groups, seed_ref={'manifest': 'seed.json', 'sha256': 'f'*64},
            context_sha256='1'*64)
    assert not (dataset.root/'bad.json').exists()
    assert not (dataset.root/'bad.npz').exists()


def test_train_statistics_ignore_padding_and_refuse_holdout_data():
    normalizer = trajectory_data.TrainNormalizer(2)
    normalizer.update(np.array([[1., 2.], [3., 4.], [999., 999.]]),
                      np.array([True, True, False]), split='train')
    normalizer.update(np.array([[5., 6.]]), np.array([True]), split='train')
    summary = normalizer.to_dict()
    np.testing.assert_allclose(summary['mean'], [3., 4.])
    np.testing.assert_allclose(summary['variance'], [8/3, 8/3])
    assert summary['count'] == 3
    for split in ('test', 'validation'):
        with pytest.raises(ValueError, match='train'):
            normalizer.update(np.array([[1., 2.]]), np.array([True]), split=split)


def test_long_sequence_crosses_storage_chunks_but_never_episode_boundaries(tmp_path):
    dataset = create(tmp_path)
    states, actions, substeps, valid, groups = arrays(steps=8)
    first = dataset.append_chunk('first', states=states, actions=actions,
        substeps=substeps, valid=valid, groups=groups,
        seed_ref={'manifest': 'first_seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    next_states = {k: v.copy() for k, v in states.items()}
    next_states['time_us'] += 40000
    next_states['q_rad'] += states['q_rad'][-1]-states['q_rad'][0]
    dataset.append_chunk('next', states=next_states, actions=actions,
        substeps=substeps, valid=valid, groups=groups,
        seed_ref={'manifest': 'next_seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    split = first['groups'][0]['split']
    window = dataset.read_sequence(['first', 'next'], lane=0, start=6, length=8,
                                   split=split, purpose='final_evaluation')
    np.testing.assert_array_equal(window['states']['time_us'], np.arange(6, 15)*5000)
    assert window['states']['q_rad'].shape == (9, 7)
    assert window['actions']['raw_normalized'].shape == (8, 7)
    assert window['seed_ref']['manifest'] == 'first_seed.json'
    assert window['replay_control_steps_before_window'] == 6
    changed = [{**g, 'episode': 'different'} for g in groups]
    dataset.append_chunk('reset', states=next_states, actions=actions,
        substeps=substeps, valid=valid, groups=changed,
        seed_ref={'manifest': 'reset_seed.json', 'sha256': 'f'*64}, context_sha256='1'*64)
    with pytest.raises(ValueError, match='episode'):
        dataset.read_sequence(['first', 'reset'], lane=0, start=6, length=8,
                               split=split, purpose='final_evaluation')
    with pytest.raises(ValueError, match='continuous'):
        dataset.read_sequence(['next', 'first'], lane=0, start=6, length=8,
                               split=split, purpose='final_evaluation')
