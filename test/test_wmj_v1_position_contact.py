"""Position-only contact conditioning preserves the earlier physical paths."""

from pathlib import Path
from functools import partial
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.contact_dynamics import ContactState
from world_model.contact_surface import surface_from_vertices
from world_model.hybrid_contact import hybrid_contact_rollout, hybrid_contact_loss
from world_model.neural import init_mlp
from world_model.position_contact import (position_features, position_scale_from_statistics,
    position_contact_step, position_contact_rollout, position_contact_loss)
from world_model.position_contact_data import prepare_position_contact_batch, position_statistics_update
from world_model.soft_contact import SoftContactContract
from world_model.trajectory_data import TrainNormalizer


def fixture():
    surface = surface_from_vertices(np.array([[x, y, z] for x in (-.075, .075)
        for y in (-.075, .075) for z in (-.001, .001)]), normal_axis=2)
    state = ContactState(jnp.array([[0., 0., .019]]), jnp.array([[.1, 0., -.5]]),
        jnp.zeros((1, 3)), jnp.zeros(1, bool), jnp.zeros(1))
    motion = dict(geom_position_m=jnp.zeros((5, 1, 3)),
        geom_rotation_world=jnp.broadcast_to(jnp.eye(3), (5, 1, 3, 3)),
        racket_linear_velocity_m_s=jnp.zeros((5, 1, 3)),
        racket_angular_velocity_rad_s=jnp.zeros((5, 1, 3)))
    extended = dict(motion, joint_q_rad=jnp.full((5, 1, 13), .05),
                    delayed_q_command_rad=jnp.full((5, 1, 7), .06))
    scale = position_scale_from_statistics(dict(source='train', count=10,
        mean=[0.]*20, variance=[1.]*20))
    prior = SoftContactContract(.004, 6.4e-7, .001, (.006836, .228086),
        (.04, .7, .014298, .04, 8.021856), (.1326818, .1182065), 371.599609375, 1., True)
    return surface, state, motion, extended, scale, prior


def test_position_features_and_training_only_scaling():
    q = jnp.arange(26, dtype=jnp.float32).reshape(2, 13)
    command = q[:, :7]+.02
    features = position_features(q, command)
    np.testing.assert_array_equal(features[:, :13], q)
    np.testing.assert_allclose(features[:, 13:], .02, atol=1e-6)
    stats = TrainNormalizer(20)
    arrays = dict(substep__q_before=np.asarray(q[:, :7])[None, None],
        substep__support_q_before=np.asarray(q[:, 7:])[None, None],
        substep__delayed_q_command=np.asarray(command)[None, None],
        substep__ball_model_valid=np.ones((1, 1, 2), bool))
    position_statistics_update(stats, arrays, dict(groups=[{'split':'train'}, {'split':'validation'}]))
    assert stats.count == 1
    np.testing.assert_allclose(stats.mean, features[0], atol=1e-6)
    scale = position_scale_from_statistics(stats.to_dict())
    np.testing.assert_allclose(scale.scale, 1e-4, atol=1e-10)
    with pytest.raises(ValueError):
        position_scale_from_statistics(stats.to_dict() | {'source':'validation'})
    with pytest.raises(ValueError):
        position_features(q, jnp.zeros((2, 13)))


@pytest.mark.parametrize('hybrid', [False, True])
def test_zero_position_weights_preserve_legacy_rollout_and_loss(hybrid):
    surface, state, motion, extended, scale, prior = fixture()
    contract = prior if hybrid else None
    old = init_mlp(jax.random.PRNGKey(9), (19, 16, 8))
    new = (dict(old[0], w=jnp.pad(old[0]['w'], ((0, 20), (0, 0)))), *old[1:])
    original = hybrid_contact_rollout(old, surface, state, motion, ball_radius_m=.02, contract=contract)
    actual = position_contact_rollout(new, surface, state, extended, ball_radius_m=.02,
        position_scale=scale, contract=contract)
    for left, right in zip(jax.tree.leaves(original), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left, right)
    trace = original[1]
    batch = dict(before=trace['before'], after=trace['after'], valid=jnp.ones((5, 1), bool),
        entry=trace['entry'], release=trace['release'], controlled_release=trace['release'])
    expected = hybrid_contact_loss(old, surface, dict(batch, racket=motion), ball_radius_m=.02,
        contract=contract, autoregressive=True)
    result = position_contact_loss(new, surface, dict(batch, racket=extended), ball_radius_m=.02,
        position_scale=scale, contract=contract, autoregressive=True)
    np.testing.assert_allclose(result[0], expected[0], rtol=1e-6, atol=1e-6)


def test_position_target_gradient_and_rejection_of_extra_velocity_input():
    surface, state, _, extended, scale, prior = fixture()
    params = init_mlp(jax.random.PRNGKey(11), (39, 16, 8))
    motion = jax.tree.map(lambda x:x[0], extended)
    def predict(command):
        after, _ = position_contact_step(params, surface, state,
            dict(motion, delayed_q_command_rad=command), ball_radius_m=.02,
            position_scale=scale, contract=prior)
        return after.velocity_m_s[0, 2]
    command = motion['delayed_q_command_rad']
    gradient = jax.jit(jax.grad(predict))(command)
    delta = jnp.zeros_like(command).at[0, 0].set(1e-3)
    finite = (predict(command+delta)-predict(command-delta))/(2e-3)
    np.testing.assert_allclose(gradient[0, 0], finite, rtol=2e-2, atol=1e-4)
    assert np.isfinite(gradient).all() and np.linalg.norm(gradient) > 1e-5
    with pytest.raises(ValueError, match='position-only'):
        position_contact_step(params, surface, state, dict(motion, delayed_dq_command=jnp.ones((1, 7))),
            ball_radius_m=.02, position_scale=scale, contract=prior)


def test_batch_inputs_ignore_joint_velocities_accelerations_and_future_positions():
    shape = (5, 1)
    raw = dict(contact_overlap=jnp.zeros(shape, bool), event_contact_entry=jnp.zeros(shape, bool),
        event_contact_release=jnp.zeros(shape, bool), event_controlled_release=jnp.zeros(shape, bool),
        event_contact_duration_steps=jnp.zeros(shape, jnp.int32), ball_model_valid=jnp.ones(shape, bool),
        racket_geom_position_before_m=jnp.zeros(shape+(3,)),
        racket_geom_rotation_before=jnp.broadcast_to(jnp.eye(3), shape+(3, 3)),
        racket_linear_velocity_before_m_s=jnp.ones(shape+(3,)),
        racket_angular_velocity_before_rad_s=jnp.zeros(shape+(3,)),
        q_before=jnp.zeros(shape+(13,)), delayed_q_command=jnp.ones(shape+(7,)))
    for when in ('before', 'after'):
        raw['ball_qpos_'+when] = jnp.zeros(shape+(7,)).at[..., 3].set(1.)
        raw['ball_qvel_'+when] = jnp.zeros(shape+(6,))
    expected = prepare_position_contact_batch(None, raw)
    altered = dict(raw, dq_before=jnp.full(shape+(13,), 999.), dq_after=jnp.full(shape+(13,), 999.),
        q_after=jnp.full(shape+(13,), 999.), delayed_dq_command=jnp.full(shape+(7,), 999.),
        delayed_ddq_command=jnp.full(shape+(7,), 999.))
    actual = prepare_position_contact_batch(None, altered)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left, right)


def test_auxiliary_fit_preserves_bootstrap_and_legacy_gradients():
    from world_model.contact_fitting import make_contact_objective

    surface, state, _, motion, scale, prior = fixture()
    params = init_mlp(jax.random.PRNGKey(31), (39, 16, 8))
    _, trace = position_contact_rollout(params, surface, state, motion,
        ball_radius_m=.02, position_scale=scale, contract=prior)
    batch = dict(before=trace['before'],
        after=trace['after']._replace(velocity_m_s=trace['after'].velocity_m_s+.01),
        racket=motion, valid=jnp.ones((5, 1), bool), entry=trace['entry'],
        release=trace['release'], controlled_release=trace['release'])
    base = partial(position_contact_loss, position_scale=scale, contract=prior)
    auxiliary = make_contact_objective(base, single_step_weight=256.)
    assert make_contact_objective(base, single_step_weight=0.) is base

    def evaluate(fun, recursive):
        return jax.value_and_grad(lambda p:fun(p, surface, batch,
            ball_radius_m=.02, autoregressive=recursive)[0])(params)

    original = evaluate(base, False)
    bootstrap = evaluate(auxiliary, False)
    for left, right in zip(jax.tree.leaves(original), jax.tree.leaves(bootstrap)):
        np.testing.assert_array_equal(left, right)
    old_recursive = evaluate(base, True)
    mixed = evaluate(auxiliary, True)
    for ar, tf, actual in zip(jax.tree.leaves(old_recursive),
                             jax.tree.leaves(original), jax.tree.leaves(mixed)):
        np.testing.assert_allclose(actual, ar+256.*tf, rtol=2e-5, atol=2e-5)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(mixed))
    assert np.linalg.norm(np.asarray(mixed[1][-1]['w'])) > 0
    for invalid in (-1., float('inf'), float('nan')):
        with pytest.raises(ValueError):
            make_contact_objective(base, single_step_weight=invalid)


def test_auxiliary_resume_cannot_reinterpret_executed_recursive_updates():
    from world_model.contact_fitting import validate_contact_resume_recipe

    old = dict(bootstrap_updates=2048, controls=8, learning_rate=3e-4)
    new = dict(old, single_step_aux_weight=256.)
    validate_contact_resume_recipe(old, new, completed=2048)
    validate_contact_resume_recipe(new, new, completed=4096)
    validate_contact_resume_recipe(old, dict(old, single_step_aux_weight=0.), completed=4096)
    with pytest.raises(ValueError, match='executed'):
        validate_contact_resume_recipe(old, new, completed=2049)
    with pytest.raises(ValueError, match='executed'):
        validate_contact_resume_recipe(new, old, completed=2049)
    with pytest.raises(ValueError, match='recipe'):
        validate_contact_resume_recipe(old, dict(new, learning_rate=1e-4), completed=2048)
