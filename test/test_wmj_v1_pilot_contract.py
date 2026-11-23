"""A bounded pilot cannot hide lost counterfactuals or change pulse semantics."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_saved_pulses_use_registered_lanes_and_duration_before_external_clip():
    from world_model.pilot_contract import counterfactual_actions
    source = np.full((128, 5, 7), .999, np.float32)
    cases = [dict(lane=3, joint=1, epsilon=.02, pulse_ms=10),
             dict(lane=0, joint=6, epsilon=.005, pulse_ms=20)]
    result = counterfactual_actions(source, cases)
    np.testing.assert_array_equal(result['base'], source[:, [3, 0]])
    np.testing.assert_allclose(result['plus'][:2, 0, 1], 1.019)
    np.testing.assert_array_equal(result['plus'][2:, 0], result['base'][2:, 0])
    np.testing.assert_allclose(result['minus'][:4, 1, 6], .994)
    with pytest.raises(ValueError):
        counterfactual_actions(source, [dict(cases[0], pulse_ms=15)])


def test_pilot_action_gate_counts_missing_cases_and_requires_sample_size():
    from world_model.pilot_contract import action_gate
    def metric(n=200, rate=.96):
        return dict(effect_exceeds_repeat_cases=n, resolved_direction_agreement_fraction=rate)
    parent = {k: metric() for k in ('outgoing_velocity_m_s', 'actual_apex_m')}
    candidate = {k: metric() for k in parent}
    assert action_gate(parent, candidate)['passed']
    candidate['actual_apex_m'] = metric(99, 1.)
    assert not action_gate(parent, candidate)['passed']
    candidate['actual_apex_m'] = metric(200, .90)
    assert not action_gate(parent, candidate)['passed']  # >5 percentage-point loss
    candidate['actual_apex_m'] = metric(200, None)
    assert not action_gate(parent, candidate)['passed']
