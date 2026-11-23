"""Frozen robot recursion supplies the exact coupled contact conditioning."""

import jax
import jax.numpy as jnp
import numpy as np

from test_wmj_v1_world_physics import setup_physics
from world_model.position_contact import position_contact_rollout
from world_model.world_physics import physics_rollout


def test_fitting_motion_matches_coupled_world_and_ignores_future_joint_labels():
    from world_model.physics_fitting import predicted_racket_motion
    robot, contact, state, options = setup_physics()
    state = state._replace(dq_rad_s=state.dq_rad_s.at[:, 7:].set(.03))
    commands = {name:jnp.zeros((10, 1, 7)) for name in ('q', 'dq', 'ddq')}
    raw = dict(q_before=jnp.broadcast_to(state.q_rad, (10, 1, 13)),
               dq_before=jnp.broadcast_to(state.dq_rad_s, (10, 1, 13)),
               **{'delayed_'+key+'_command':value for key, value in commands.items()})
    motion = predicted_racket_motion(robot, raw, scale=options['robot_scale'], chain=options['chain'])
    corrupted = dict(raw, q_before=raw['q_before'].at[1:].set(123.),
                     dq_before=raw['dq_before'].at[1:].set(-456.))
    other = predicted_racket_motion(robot, corrupted, scale=options['robot_scale'], chain=options['chain'])
    for left, right in zip(jax.tree.leaves(motion), jax.tree.leaves(other)):
        np.testing.assert_array_equal(left, right)
    after, _ = physics_rollout(robot, contact, state, commands, **options)
    expected, _ = position_contact_rollout(contact, options['surface'], state.ball, motion,
        ball_radius_m=.02, position_scale=options['position_scale'])
    for left, right in zip(jax.tree.leaves(after.ball), jax.tree.leaves(expected)):
        np.testing.assert_allclose(left, right, rtol=2e-5, atol=1e-7)


def test_physics_checkpoint_rejects_unregistered_conditioning():
    from world_model.physics_components import validate_contact_conditioning
    validate_contact_conditioning('wmj_contact_component_checkpoint_v1',
                                  'recorded_native_pose_twist_and_joint_positions')
    validate_contact_conditioning('wmj_contact_coupled_checkpoint_v1',
                                  'own_predicted_full_chain_pose_twist_and_joint_positions')
    import pytest
    with pytest.raises(ValueError):
        validate_contact_conditioning('wmj_contact_coupled_checkpoint_v1', 'extra_joint_velocity')
    with pytest.raises(ValueError):
        validate_contact_conditioning('wmj_contact_component_checkpoint_v1',
                                      'own_predicted_full_chain_pose_twist_and_joint_positions')
