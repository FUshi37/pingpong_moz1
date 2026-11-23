"""Frozen value diagnostics exclude censored returns and preserve source weights."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                      'pingpong_controller/tools/rl_sim'))

from world_model.critic_distribution import (
    failure_horizon, contact_phase, compact_regression, stratified_summary,
)


def test_failure_horizon_does_not_treat_live_window_as_complete_return():
    valid = np.array([[1, 1], [1, 1], [0, 1]], bool)
    failed = np.array([[0, 0], [1, 0], [0, 0]], bool)
    complete, remaining = failure_horizon(valid, failed)
    np.testing.assert_array_equal(complete, [[1, 0], [1, 0], [0, 0]])
    np.testing.assert_array_equal(remaining, [[1, -1], [0, -1], [-1, -1]])


def test_failure_horizon_rejects_post_reset_or_nonprefix_data():
    with pytest.raises(ValueError):
        failure_horizon(np.array([[1], [0], [1]], bool), np.zeros((3, 1), bool))
    with pytest.raises(ValueError):
        failure_horizon(np.ones((3, 1), bool), np.array([[0], [1], [0]], bool))


def test_contact_phase_uses_previous_release_and_current_event():
    contact = np.array([[0], [1], [0], [0], [0]], bool)
    releases = np.array([[0], [0], [1], [0], [0]])
    vz = np.array([[-1.], [-1.], [1.], [1.], [-1.]])
    np.testing.assert_array_equal(contact_phase(contact, releases, vz)[:, 0],
                                  [0, 1, 2, 3, 4])


def test_low_bias_does_not_imply_value_discrimination():
    result = compact_regression(np.ones(4)*-10., np.array([-12., -11., -9., -8.]))
    assert result['bias'] == 0.
    assert result['explained_variance'] == 0.
    assert result['correlation'] is None
    assert result['rmse'] > 1.


def test_source_balancing_and_censoring_are_independent_of_row_count():
    rows = dict(source=np.array([1, 1, 1, 2, 3]), region=np.array([0]*5),
        complete=np.array([1, 1, 1, 1, 0], bool), remaining=np.array([2, 1, 0, 0, -1]),
        phase=np.array([0, 0, 2, 0, 0]), mc=np.array([-10., -10., -10., -10., 999.]),
        ema005_online=np.array([-9., -9., -9., -7., 0.]),
        ema005_target=np.array([-9., -9., -9., -7., 0.]),
        ema005_lambda=np.array([-10., -10., -10., -10., 0.]))
    result = stratified_summary(rows, arms=('ema005',))
    stat = result['overall']['ema005']['online_vs_mc']
    assert stat['count'] == 4
    assert stat['bias'] == 1.5
    assert stat['source_balanced_bias'] == 2.
    assert result['censored_rows'] == 1
    assert result['time_to_failure']['terminal']['ema005']['online_vs_mc']['count'] == 2
