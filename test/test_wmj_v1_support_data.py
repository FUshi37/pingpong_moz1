"""Full-chain supervision preserves acquisition splits and native label clocks."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.support_data import (
    POSE_STREAMS, support_chunk, support_statistics_update, sample_support_windows,
)
from world_model.trajectory_data import TrainNormalizer


def fixture():
    steps, lanes = 8, 3
    arm = np.arange(steps*5*lanes*7, dtype=np.float32).reshape(steps, 5, lanes, 7)*1e-5
    support = np.arange(steps*5*lanes*6, dtype=np.float32).reshape(steps, 5, lanes, 6)*1e-6
    arrays = {}
    for name in ('q_before', 'dq_before', 'q_after', 'dq_after'):
        arrays['substep__'+name] = arm.copy()
        arrays['substep__support_'+name] = support.copy()
    for name in ('delayed_q_command', 'delayed_dq_command', 'delayed_ddq_command'):
        arrays['substep__'+name] = arm.copy()
    arrays['substep__robot_model_valid'] = np.ones((steps, 5, lanes), bool)
    for name in POSE_STREAMS:
        shape = (steps, 5, lanes, 3, 3) if 'rotation' in name else (steps, 5, lanes, 3)
        arrays['substep__'+name] = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    record = dict(steps=steps, groups=[dict(split=split, region_id=i, root_seed_family=f'r{i}')
                  for i, split in enumerate(('train', 'validation', 'test'))])
    return arrays, record


def test_support_normalization_excludes_held_out_lanes():
    arrays, record = fixture()
    reference = None
    for corrupt in (False, True):
        if corrupt:
            for name, value in arrays.items():
                if value.dtype.kind == 'f':
                    value[:, :, 1:] = 1e20
        inputs, targets = TrainNormalizer(47), TrainNormalizer(26)
        support_statistics_update(inputs, targets, arrays, record)
        actual = inputs.to_dict(), targets.to_dict()
        assert inputs.count == 40
        if reference is None:
            reference = actual
        else:
            assert actual == reference


def test_sampled_pose_and_support_states_share_exact_substep_and_lane():
    arrays, record = fixture()
    data = support_chunk(arrays)
    batch, refs = sample_support_windows(np.random.default_rng(51), data, record,
        controls=3, batch_size=5, split='validation')
    assert batch['q_before'].shape == (15, 5, 13)
    assert batch['commands']['q'].shape == (15, 5, 7)
    assert np.all(refs['lane'] == 1)
    for i in range(5):
        start = refs['start_control'][i]*5
        np.testing.assert_array_equal(batch['q_before'][:, i, 7:],
            arrays['substep__support_q_before'].reshape(40, 3, 6)[start:start+15, 1])
        for name in POSE_STREAMS:
            np.testing.assert_array_equal(batch[name][:, i], data[name][start:start+15, 1])
    with pytest.raises(ValueError, match='Final test'):
        sample_support_windows(np.random.default_rng(1), data, record,
                               controls=3, batch_size=5, split='test')
