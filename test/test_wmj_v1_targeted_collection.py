"""Independent supplementary sampling retains regions and physical bounds."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.targeted_collection import acquisition_groups, independent_targets, missing_regions, targeted_draw, warm_fallback
from world_model.datasets import SplitRegistry


def test_replay_comparison_accepts_roundoff_but_keeps_discrete_state_exact():
    from world_model.targeted_recovery import compare_replay_arrays

    first = {'q': np.array([1.], np.float32), 'event': np.array([1], np.int32)}
    second = {'q': np.array([1.+1e-7], np.float32), 'event': np.array([1], np.int32)}
    assert compare_replay_arrays(first, second)['passed']
    second['q'][0] = 1.01
    assert not compare_replay_arrays(first, second)['passed']
    second['q'][0] = 1.
    second['event'][0] = 2
    assert not compare_replay_arrays(first, second)['passed']


def test_nonfinite_replay_diagnostic_remains_serializable_failure():
    import json
    from world_model.targeted_recovery import compare_replay_arrays

    result = compare_replay_arrays({'q': np.array([1.])}, {'q': np.array([np.nan])})
    assert not result['passed']
    json.dumps(result, allow_nan=False)


def test_cloned_warm_sources_stay_in_one_split_with_distinct_episodes():
    groups = acquisition_groups(51, np.array([0, 0, 2, 2]))
    registry = SplitRegistry('test')
    splits = [registry.assign(**group) for group in groups]
    assert splits[0] == splits[1] and splits[2] == splits[3]
    assert groups[0]['root_seed_family'] == groups[1]['root_seed_family']
    assert groups[0]['episode'] != groups[1]['episode']
    assert groups[0]['root_seed_family'] != groups[2]['root_seed_family']


def test_missing_region_report_serializes_numpy_identifiers():
    import json

    missing = missing_regions(np.array([5, 14, 17]), np.array([5, 17]))
    assert json.loads(json.dumps({'missing': missing})) == {'missing': [14]}


def test_independent_candidates_stay_in_each_registered_spatial_stratum():
    lower = np.array([-.38, -.75, 1.]); upper = np.array([.38, -.28, 1.15])
    point = upper.copy()
    a = independent_targets(point, lower, upper, seed=51, region_id=188)
    b = independent_targets(point, lower, upper, seed=51, region_id=188)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (2, 3)
    assert (a >= lower).all() and (a <= upper).all()
    assert (np.abs(a-point) < .02).all()
    assert not np.array_equal(a[0], a[1])


def test_targeted_half_preserves_uniform_coverage_and_declares_probabilities():
    rng = np.random.default_rng(5)
    draw = targeted_draw(rng, 1024, region_ids=np.arange(20), focus_region_ids=[2, 4])
    assert set(draw.indices) == set(range(20))
    assert draw.report['counts'] == {'global': 512, 'focus': 512}
    expected = .5/20 + .5/2
    np.testing.assert_allclose(draw.probability[np.isin(draw.indices, [2, 4])], expected)
    np.testing.assert_allclose(draw.probability[~np.isin(draw.indices, [2, 4])], .5/20)
    with pytest.raises(ValueError):
        targeted_draw(rng, 1024, region_ids=np.arange(20), focus_region_ids=[90])


def test_warm_fallback_never_copies_an_unknown_lane_or_mislabels_its_region():
    rng = np.random.default_rng(9)
    mapping, effective, rows = warm_fallback(rng, [False, True, True, False], [0, 0, 1, 2])
    assert mapping[0] == 1 and mapping[1] == 1 and mapping[2] == 2
    assert mapping[3] in (1, 2)
    np.testing.assert_array_equal(effective, np.array([0, 0, 1, 2])[mapping])
    assert rows[-1]['requested_region_missing']
    with pytest.raises(ValueError, match='admitted'):
        warm_fallback(rng, [False], [0])
