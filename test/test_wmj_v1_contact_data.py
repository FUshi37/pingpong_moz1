"""Actual event windows are balanced by independent impact and stable split."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.contact_data import contact_window_rows, take_contact_windows, check_continuity
from world_model.sampling import sample_model_windows


def example():
    count, lanes = 80, 5
    data = dict(ball_model_valid=np.ones((count, lanes), bool),
        contact_overlap=np.zeros((count, lanes), bool),
        event_impact_id=np.ones((count, lanes), np.int32),
        q_before=np.broadcast_to(np.arange(count)[:, None, None], (count, lanes, 7)).copy())
    data['contact_overlap'][25:35, 0] = True
    groups = [dict(split=s, generator=g, root_seed_family=str(i), region_id=i)
        for i, (s, g) in enumerate((('train', 'directed_contact'), ('train', 'boundaries_failure'),
            ('train', 'robot_identification'), ('validation', 'directed_contact'),
            ('test', 'directed_contact')))]
    return data, dict(steps=16, groups=groups)


def test_four_buckets_use_real_events_and_only_training_split():
    data, record = example()
    rows = contact_window_rows(data, record, controls=8, split='train')
    draw = sample_model_windows(np.random.default_rng(4), rows)
    assert draw.report['counts'] == dict(valid_contact=128, miss_failure=64,
                                        robot_identification=38, free_flight=26)
    batch, refs = take_contact_windows(data, rows, draw.indices, controls=8)
    assert batch['q_before'].shape == (40, 256, 7)
    for index, row in enumerate(refs):
        assert row['lane'] < 3
        np.testing.assert_array_equal(batch['q_before'][:, index, 0],
                                     np.arange(row['start_substep'], row['start_substep']+40))
    with pytest.raises(ValueError, match='Final test'):
        contact_window_rows(data, record, controls=8, split='test')


def test_invalid_physics_is_excluded_without_reclassifying_it_as_task_failure():
    data, record = example()
    data['ball_model_valid'][:, 1] = False
    rows = contact_window_rows(data, record, controls=8, split='train')
    assert not any(row['lane'] == 1 for row in rows)
    assert not any(row['bucket'] == 'miss_failure' for row in rows)


def test_chunk_join_allows_native_quaternion_renormalization_only():
    a = {name+'_after': np.zeros((1, 2, size), np.float32)
         for name, size in (('q', 7), ('dq', 7), ('ball_qpos', 7), ('ball_qvel', 6))}
    a['ball_qpos_after'][..., 3] = 1.
    b = {key.replace('_after', '_before'): value.copy() for key, value in a.items()}
    b['ball_qpos_before'][..., 3] += 1e-7
    check_continuity(a, b)
    b['ball_qpos_before'][..., 0] += .001
    with pytest.raises(ValueError, match='Physical state'):
        check_continuity(a, b)
