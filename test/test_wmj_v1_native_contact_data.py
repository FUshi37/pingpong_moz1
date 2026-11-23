"""Contact supervision consumes native same-time pose/twist without frozen FK."""

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.native_contact_data import prepare_native_contact_batch


def test_native_contact_conditioning_is_exact_and_keeps_ball_future_as_labels():
    shape = (5, 2)
    data = dict(contact_overlap=jnp.zeros(shape, bool),
        event_contact_entry=jnp.zeros(shape, bool), event_contact_release=jnp.zeros(shape, bool),
        event_controlled_release=jnp.zeros(shape, bool), event_contact_duration_steps=jnp.zeros(shape, jnp.int32),
        ball_model_valid=jnp.ones(shape, bool),
        racket_geom_position_before_m=jnp.full(shape+(3,), .3),
        racket_geom_rotation_before=jnp.broadcast_to(jnp.eye(3), shape+(3, 3)),
        racket_linear_velocity_before_m_s=jnp.full(shape+(3,), .2),
        racket_angular_velocity_before_rad_s=jnp.full(shape+(3,), .1))
    for when in ('before', 'after'):
        data['ball_qpos_'+when] = jnp.zeros(shape+(7,)).at[..., 3].set(1.)
        data['ball_qvel_'+when] = jnp.zeros(shape+(6,))
    batch = prepare_native_contact_batch(None, data)
    np.testing.assert_array_equal(batch['racket']['geom_position_m'], data['racket_geom_position_before_m'])
    np.testing.assert_array_equal(batch['racket']['racket_linear_velocity_m_s'], data['racket_linear_velocity_before_m_s'])
    changed = prepare_native_contact_batch(None, {**data, 'ball_qpos_after': data['ball_qpos_after'].at[..., 0].set(99.)})
    np.testing.assert_array_equal(batch['before'].position_m, changed['before'].position_m)
    np.testing.assert_array_equal(changed['after'].position_m[..., 0], 99.)
