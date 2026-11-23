"""Frozen input diagnostics preserve partitions and avoid self-neighbor leakage."""

import gzip
import json
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'pingpong_controller/tools/rl_sim'))


def test_input_families_cover_every_current_feature_exactly_once():
    from world_model.critic_input_audit import input_groups
    path = ROOT/'pingpong_controller/outputs/rl_sim/wmj_v1/critic_fixed_fit_resume_20261112_a01/result.json.gz'
    layout = json.loads(gzip.decompress(path.read_bytes()))['critic_layout']
    groups = input_groups(layout)
    assert {k:len(v) for k,v in groups.items()} == dict(actor=969, physical=46, task=24, control=1563)
    assert np.array_equal(np.sort(np.concatenate(list(groups.values()))), np.arange(2602))
    layout['fields'][-1]['name'] = 'unknown_field'
    with pytest.raises(ValueError):
        input_groups(layout)


def test_coverage_uses_training_scale_and_separates_constant_drift():
    from world_model.critic_input_audit import distribution_summary, component_stats
    train = np.array([[0, 2], [2, 2]], np.float32)
    query = np.array([[1, 3], [4, 2]], np.float32)
    row = distribution_summary(query, component_stats(train), np.arange(2))
    assert row['active_components'] == 1
    assert row['constant_components'] == 1
    assert row['outside_training_range_fraction'] == .5
    assert row['constant_drift_fraction'] == .5


def test_input_gradient_matches_independent_tanh_chain_rule_and_freezes_weights():
    from world_model.critic_input_audit import input_gradients
    from world_model.neural import init_mlp
    from world_model.state_adapter import numeric_tree_digest
    p = init_mlp(jax.random.PRNGKey(1), (3, 5, 4, 1))
    x = np.array([[.1, .2, -.3], [.4, -.1, .5]], np.float32)
    before = numeric_tree_digest(p)
    h1 = np.tanh(x @ np.asarray(p[0]['w']) + np.asarray(p[0]['b']))
    h2 = np.tanh(h1 @ np.asarray(p[1]['w']) + np.asarray(p[1]['b']))
    expected = (((1-h2*h2)*np.asarray(p[2]['w']).T) @ np.asarray(p[1]['w']).T
                * (1-h1*h1)) @ np.asarray(p[0]['w']).T
    actual = np.asarray(input_gradients(p, jnp.asarray(x)))
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert numeric_tree_digest(p) == before


def test_nearest_neighbor_excludes_same_source_and_wrong_phase():
    from world_model.critic_input_audit import nearest_training_rows
    train = np.array([[0.], [.1], [.2], [2.]], np.float32)
    sources = np.array([1, 1, 2, 3]); phases = np.array([0, 0, 0, 1])
    query = np.array([[0.], [0.]], np.float32)
    distance, index = nearest_training_rows(train, query, sources, np.array([1, 1]),
        phases, np.array([0, 2]), np.array([True]), np.array([1.]), chunk=2)
    assert index.tolist() == [2, -1]
    assert distance[0] == pytest.approx(.2)
    assert np.isinf(distance[1])


def test_constant_inputs_never_create_fake_zero_distance_neighbors():
    from world_model.critic_input_audit import nearest_training_rows
    distance, index = nearest_training_rows(np.ones((3, 2)), np.ones((1, 2)),
        np.arange(3), np.array([5]), np.zeros(3), np.zeros(1),
        np.array([False, False]), np.zeros(2))
    assert index.tolist() == [-1] and np.isinf(distance[0])


def test_sequential_storage_reserves_peak_not_simultaneous_temporaries():
    from world_model.critic_input_audit import sequential_storage_reservation
    arrays = dict(values=np.zeros((30, 2602), np.float32), codes=np.zeros(2602, np.int8))
    result = sequential_storage_reservation(arrays, dict(message='frozen audit'))
    assert result['required_peak_bytes'] == max(result['archive_writer_bytes'], result['report_after_archive_bytes'])+result['log_slack_bytes']
    assert result['required_peak_bytes'] < result['archive_writer_bytes']+result['report_after_archive_bytes']
