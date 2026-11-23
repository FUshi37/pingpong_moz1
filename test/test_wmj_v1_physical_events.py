"""Finite-contact edges and observed apexes remain distinct from reward hits."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model import physical_events


def trace(overlap, force, vertical_before=None, vertical_after=None):
    length = len(overlap)
    p0 = np.zeros((length, 1, 7), np.float32)
    p0[:, :, 2] = 1.1
    p0[:, :, 3] = 1.
    before = np.zeros((length, 1, 6), np.float32)
    after = before.copy()
    before[:, 0, 2] = np.ones(length) if vertical_before is None else vertical_before
    after[:, 0, 2] = before[:, 0, 2] if vertical_after is None else vertical_after
    return dict(ball_qpos_before=jnp.asarray(p0), ball_qpos_after=jnp.asarray(p0),
                ball_qvel_before=jnp.asarray(before), ball_qvel_after=jnp.asarray(after),
                contact_overlap=jnp.asarray(overlap)[:, None],
                contact_normal_force_n=jnp.asarray(force, jnp.float32)[:, None])


def run(state, values, setup=None):
    mask = jnp.ones_like(values['contact_overlap'], bool)
    return physical_events.scan_physical_events(state, values, valid=mask,
        initialization_active=jnp.zeros_like(mask) if setup is None else setup)


def test_contact_across_control_chunks_produces_one_release_and_integrated_impulse():
    state = physical_events.init_physical_events(1)
    state, first = jax.jit(run)(state, trace([False, False, True, True, True], [0, 0, 1, 2, 3]))
    state, second = jax.jit(run)(state, trace([True, False, False, False, False], [4, 0, 0, 0, 0]))
    assert int(first['contact_entry'].sum()) == 1
    assert int(first['contact_release'].sum()) == 0
    assert int(second['contact_release'].sum()) == 1
    assert int(second['controlled_release'].sum()) == 1
    assert int(second['impact_id'][1, 0]) == 1
    assert int(second['contact_duration_steps'][1, 0]) == 4
    np.testing.assert_allclose(second['release_normal_impulse_n_s'][1, 0], .010, atol=1e-8)


def test_held_launch_and_zero_force_touch_are_not_controlled_hits():
    state = physical_events.init_physical_events(1)
    held = trace([True, True, False], [1., 1., 0.])
    state, events = run(state, held, jnp.array([[True], [True], [False]]))
    assert int(events['contact_release'].sum()) == 1
    assert not bool(events['controlled_release'].any())
    state, events = run(state, trace([True, True, False], [0., 0., 0.]))
    assert int(events['contact_release'].sum()) == 1
    assert not bool(events['controlled_release'].any())
    assert int(state.impact_id[0]) == 2


def test_actual_apex_interpolates_only_the_observed_velocity_crossing():
    state = physical_events.init_physical_events(1)
    values = trace([False]*3, [0.]*3, [.02, .01, -.01], [.01, -.01, -.02])
    values['ball_qpos_before'] = values['ball_qpos_before'].at[1, 0, :3].set(jnp.array([.1, -.4, 1.4]))
    values['ball_qpos_after'] = values['ball_qpos_after'].at[1, 0, :3].set(jnp.array([.102, -.398, 1.400004]))
    _, events = run(state, values)
    np.testing.assert_array_equal(events['actual_apex'][:, 0], [False, True, False])
    np.testing.assert_allclose(events['apex_position_m'][1, 0], [.101, -.399, 1.400002], atol=1e-7)
    assert events['apex_velocity_m_s'][1, 0, 2] == 0.
    _, suppressed = run(state, values, jnp.ones((3, 1), bool))
    assert not bool(suppressed['actual_apex'].any())


def test_invalid_padding_does_not_advance_contact_state_or_create_events():
    state = physical_events.init_physical_events(1)
    values = trace([True, False], [1., 0.])
    after, events = physical_events.scan_physical_events(state, values,
        valid=jnp.zeros((2, 1), bool), initialization_active=jnp.zeros((2, 1), bool))
    for expected, actual in zip(jax.tree.leaves(state), jax.tree.leaves(after)):
        np.testing.assert_array_equal(actual, expected)
    assert not bool(events['contact_entry'].any())
    assert not bool(events['contact_release'].any())


def test_small_hop_apex_can_occur_in_the_first_free_interval_after_release():
    state, _ = run(physical_events.init_physical_events(1), trace([True], [1.]))
    _, events = run(state, trace([False], [0.], [.003], [-.00681]))
    assert bool(events['contact_release'][0, 0])
    assert bool(events['actual_apex'][0, 0])
