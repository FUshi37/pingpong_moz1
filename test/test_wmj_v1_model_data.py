"""Model batches preserve substep alignment and exclude holdout statistics."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.model_data import robot_chunk, sample_robot_windows, robot_statistics_update
from world_model.trajectory_data import TrainNormalizer


def chunk():
    q = np.arange(4*5*3*7, dtype=np.float32).reshape(4, 5, 3, 7)*.001
    dq = np.ones_like(q)*.1
    values = dict(q_before=q, dq_before=dq, q_after=q+dq*.001, dq_after=dq+.01,
        delayed_q_command=q+.02, delayed_dq_command=dq+.03, delayed_ddq_command=dq+.04,
        robot_model_valid=np.ones((4, 5, 3), bool))
    arrays = {'substep__'+name:value for name,value in values.items()}
    record = dict(steps=4, groups=[dict(split=split, root_seed_family=str(i),
        region_id=i) for i,split in enumerate(('train','validation','test'))])
    return arrays,record


def test_training_statistics_never_mix_validation_or_test_lanes():
    arrays,record = chunk()
    for name,value in arrays.items():
        if value.ndim==4:
            value[:,:,1:]+=10000
    inputs,targets = TrainNormalizer(35),TrainNormalizer(14)
    robot_statistics_update(inputs,targets,arrays,record)
    assert inputs.count==20
    assert inputs.to_dict()['mean'][0] < 1.
    np.testing.assert_allclose(targets.to_dict()['mean'][7:], .01, atol=1e-6)


def test_sequence_batch_uses_only_requested_split_and_five_substeps_per_control():
    arrays,record = chunk()
    batch,refs = sample_robot_windows(np.random.default_rng(8),robot_chunk(arrays),record,
        controls=2,batch_size=8,split='validation')
    assert batch['q_before'].shape==(10,8,7)
    assert batch['commands']['q'].shape==(10,8,7)
    assert np.all(refs['lane']==1)
    for i,(control,lane) in enumerate(zip(refs['start_control'],refs['lane'])):
        expected=arrays['substep__q_before'][control:control+2,:,lane].reshape(10,7)
        np.testing.assert_array_equal(batch['q_before'][:,i],expected)
    with pytest.raises(ValueError,match='Final test'):
        sample_robot_windows(np.random.default_rng(8),robot_chunk(arrays),record,
            controls=2,batch_size=8,split='test')
