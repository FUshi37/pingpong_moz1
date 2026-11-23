"""Roundoff is accepted; changed motion or discrete events are not hidden."""

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'pingpong_controller/tools/rl_sim'))
from world_model.numerical_validation import assess_execution_difference


def test_recorded_cuda_roundoff_does_not_block_execution():
    path = ROOT / 'pingpong_controller/outputs/rl_sim/wmj_v1/gpu_replay_20261106_a01/result.json'
    if not path.exists():
        import pytest
        pytest.skip('Local numerical evidence artifact is not installed')
    original = json.loads(path.read_text())['initial_execution_pair']
    assert original['changed_leaves'] == 66
    assert assess_execution_difference(original)['status'] == 'PASS'
    changed = copy.deepcopy(original)
    next(r for r in changed['changes'] if r['path'] == '[0][0].data.qpos')['max_abs_diff'] = .003
    assert assess_execution_difference(changed)['status'] == 'FAIL'


def test_changed_termination_is_not_a_float_roundoff_exception():
    summary = dict(changed_leaves=1, changes=[dict(path='[0][3]', dtype='bool', max_abs_diff=1.)])
    assert assess_execution_difference(summary)['status'] == 'FAIL'
