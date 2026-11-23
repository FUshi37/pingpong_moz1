"""Independent collection/model/start/goal distributions retain their contracts."""

from pathlib import Path
import sys

import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import sampling


def test_collection_has_exact_half_global_strata_and_half_weak_regions():
    rng = np.random.default_rng(7)
    draw = sampling.sample_collection_regions(rng, 1024,
        coverage=np.array([0, 10, 100, 1000]),
        normalized_error=np.array([0., 1., 10., 0.]),
        cost_s=np.ones(4), uncertainty=None)
    global_ids = draw.indices[draw.component == 'global']
    np.testing.assert_array_equal(np.bincount(global_ids, minlength=4), np.full(4, 128))
    assert np.count_nonzero(draw.component == 'weak') == 512
    assert np.all(draw.probability > 0)
    assert draw.report['uncertainty_used'] is False
    with pytest.raises(ValueError, match='calibrated'):
        sampling.sample_collection_regions(rng, 1024, coverage=np.zeros(4),
            normalized_error=np.ones(4), cost_s=np.ones(4), uncertainty=np.ones(4))


def test_model_sampling_balances_independent_events_not_overlapping_window_counts():
    rows = []
    for bucket in ('valid_contact', 'miss_failure', 'robot_identification', 'free_flight'):
        for event, copies in (('short', 1), ('long', 100)):
            rows.extend([dict(bucket=bucket, independent_event_id=f'{bucket}/{event}', split='train')]*copies)
    draw = sampling.sample_model_windows(np.random.default_rng(8), rows)
    assert {b: int(np.count_nonzero(draw.component == b)) for b in np.unique(draw.component)} == {
        'valid_contact': 128, 'miss_failure': 64, 'robot_identification': 38, 'free_flight': 26}
    long_count = sum(rows[i]['independent_event_id'].endswith('long') for i in draw.indices)
    assert 95 < long_count < 161  # ~50%, not 100/101 from duplicate windows.
    assert np.all(draw.probability > 0)
    with pytest.raises(sampling.SamplingShortage, match='free_flight'):
        sampling.sample_model_windows(np.random.default_rng(9), rows[:-101])
    rows[0] = {**rows[0], 'split': 'test'}
    with pytest.raises(ValueError, match='train'):
        sampling.sample_model_windows(np.random.default_rng(9), rows)


def test_branch_sampling_requires_lead_time_and_does_not_invent_missing_categories():
    rows = [dict(category=category, time_to_contact_s=.1 if i else .01,
                 seed_id=f'{category}/{i}', split='train')
            for category in ('precontact', 'goal_switch', 'recoverable_disturbance') for i in range(4)]
    draw = sampling.sample_branch_starts(np.random.default_rng(3), rows, minimum_precontact_lead_s=.08)
    assert len(draw.indices) == 256
    assert all(rows[i]['time_to_contact_s'] >= .08 for i in draw.indices
               if rows[i]['category'] == 'precontact')
    with pytest.raises(sampling.SamplingShortage, match='recoverable_disturbance'):
        sampling.sample_branch_starts(np.random.default_rng(3), rows[:8], minimum_precontact_lead_s=.08)


def test_goal_frontier_and_review_fallback_are_explicit_and_do_not_remove_global_support():
    rng = np.random.default_rng(4)
    draw = sampling.sample_goals(rng, 1000, success_rate=np.array([.0, .5, .95, .5]),
                                 improvement=np.array([0., .1, 0., -.1]), observations=np.ones(4))
    assert np.all(draw.indices[draw.component == 'frontier'] == 1)
    assert np.all(draw.indices[draw.component == 'review'] == 2)
    assert np.count_nonzero(draw.component == 'global') == 500
    cold = sampling.sample_goals(rng, 1000, success_rate=np.zeros(4),
                                 improvement=np.zeros(4), observations=np.zeros(4))
    assert np.all(cold.component == 'global')
    assert cold.report['reallocated_to_global'] == {'frontier': 300, 'review': 200}
    np.testing.assert_allclose(cold.probability, .25)
