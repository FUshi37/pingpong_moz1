"""Frozen reward attribution retains termination, bootstrap and pose semantics."""

from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))


def test_component_returns_reconstruct_total_with_failure_and_padding():
    from world_model.paired_attribution import return_components, TERM_NAMES
    from world_model.actor_critic import ImaginedBatch, actor_return
    terms = {name: jnp.full((3, 2), (i + 1) / 10) for i, name in enumerate(TERM_NAMES)}
    tr = SimpleNamespace(reward=sum(terms.values()), reward_terms=terms,
        valid=jnp.array([[True, True], [True, True], [False, True]]),
        discount=jnp.array([[.9, .9], [0., .9], [0., .9]]),
        trace_continue=jnp.array([[True, True], [False, True], [False, False]]))
    values = jnp.arange(6, dtype=jnp.float32).reshape(3, 2)
    parts = return_components(tr, values)
    expected = actor_return(ImaginedBatch(tr.reward, tr.discount, tr.trace_continue, tr.valid), values)
    np.testing.assert_allclose(parts.sum(-1), expected, rtol=2e-7)
    assert parts[0, -1] == 0.  # The physically failed lane must not bootstrap.
    assert parts[1, -1] > 0.   # A horizon boundary does retain the value term.


def test_hard_failure_cost_has_zero_pathwise_gradient_but_value_path_is_retained():
    from world_model.paired_attribution import return_components, TERM_NAMES
    def objective(x, failure):
        terms = {name: jnp.zeros((1, 1)) for name in TERM_NAMES}
        terms['failure'] = jnp.where(x > 0., -10., 0.).reshape(1, 1)
        tr = SimpleNamespace(reward=sum(terms.values()), reward_terms=terms,
            valid=jnp.ones((1, 1), bool), discount=jnp.full((1, 1), 0. if failure else .9),
            trace_continue=jnp.zeros((1, 1), bool))
        return return_components(tr, (2. * x).reshape(1, 1))[0]
    a = jax.jacrev(lambda x: objective(x, True))(jnp.float32(.2))
    b = jax.jacrev(lambda x: objective(x, False))(jnp.float32(.2))
    assert a[TERM_NAMES.index('failure')] == 0.
    np.testing.assert_allclose(b[-1], 1.8)


def test_orientation_margin_agrees_with_existing_physical_bound():
    from world_model.paired_attribution import orientation_margin_deg
    from world_model.imagined_env import physical_bounds, EpisodeContract
    angles = np.deg2rad([0., 1.99, 2.01, -2.01])
    rot = np.array([[[1., 0., 0.], [0., np.cos(a), -np.sin(a)],
                     [0., np.sin(a), np.cos(a)]] for a in angles], np.float32)
    pose = jnp.asarray(np.concatenate((np.tile([0., -.5, 1.1], (4, 1)), rot[:, :, 0], rot[:, :, 1]), axis=1))
    projection = SimpleNamespace(reference_rotation=jnp.eye(3), q_lower=jnp.full(7, -3.), q_upper=jnp.full(7, 3.))
    control = SimpleNamespace(projection=projection)
    physics = SimpleNamespace(q_rad=jnp.zeros((4, 13)), ball=SimpleNamespace(position_m=jnp.tile(jnp.array([0., -.5, 1.2]), (4, 1))))
    _, bounds = physical_bounds(physics, pose, control, EpisodeContract())
    margin = orientation_margin_deg(pose, projection.reference_rotation, EpisodeContract().roll_pitch_limit_rad)
    np.testing.assert_array_equal(margin >= 0., bounds['orientation'])


def test_gradient_cancellation_and_zero_terms_are_reported_without_nan():
    from world_model.paired_attribution import gradient_statistics
    gradients = [dict(w=jnp.array([1., 0.])), dict(w=jnp.array([-.5, 0.])), dict(w=jnp.zeros(2))]
    report = gradient_statistics(gradients, dict(w=jnp.array([.2, .1])), reward_count=2)
    np.testing.assert_allclose(report['total_reward_gradient_norm'], .5)
    np.testing.assert_allclose(report['dot_actual_update'], [.2, -.1, 0.])
    assert report['cosine_to_total_reward'][1] == -1.
    assert report['cosine_to_total_reward'][2] is None


def test_collision_filter_excludes_racket_and_invalid_contact_slots():
    from world_model.paired_attribution import ball_other_contacts
    pairs = np.array([[9, 3], [2, 9], [9, -1], [9, 7], [4, 8]])
    distances = np.array([-.001, -.002, -.5, .001, -.3])
    assert ball_other_contacts(pairs, distances, ball_id=9, racket_id=3) == [1]


def test_host_observer_preserves_compiled_arguments_and_outputs():
    from world_model.native_entry_collision import observed_jit
    received = []
    def policy_step(state, std):
        return state * 2. + std
    def scoped(state, std):
        return policy_step(state, std)
    wrapped = observed_jit(lambda fn, args, result: received.append((fn, args, result)))(scoped)
    state, std = jnp.arange(3, dtype=jnp.float32), jnp.float32(.03)
    actual = wrapped(state, std)
    np.testing.assert_array_equal(actual, jax.jit(scoped)(state, std))
    assert received[0][0] is policy_step
    assert received[0][1][0] is state
    assert received[0][2] is actual


def test_compact_diagnostic_json_is_lossless_and_protects_existing_evidence(tmp_path):
    import json
    import pytest
    from world_model.native_entry_collision import write_compact_json
    from world_model.resources import StorageBudget
    from world_model.contracts import file_hash
    budget = StorageBudget(tmp_path, 1024 * 1024, 0)
    path = tmp_path / 'result.json'
    payload = dict(values=[1.25, -3., True, None], name='冻结归因')
    digest = write_compact_json(path, payload, budget)
    assert json.loads(path.read_text()) == payload
    assert file_hash(path) == digest
    with pytest.raises(FileExistsError):
        write_compact_json(path, dict(changed=True), budget)
