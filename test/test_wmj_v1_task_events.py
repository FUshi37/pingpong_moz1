"""Goal accounting from physical releases and actual apexes, without resets."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.task import (
    init_task, record_apex, record_release, record_failure, switch_completed_goals,
)


def release(state, index, **kwargs):
    return record_release(state, impact_id=jnp.array([index]),
                          physical_release=jnp.array([True]),
                          controlled_hit=jnp.array([True]), **kwargs)


def apex(state, good=True):
    point = state.goal_xyz + jnp.array([[0.0 if good else 0.04, 0., 0.]])
    return record_apex(state, point, jnp.array([0.01]), jnp.array([True]))


def test_initial_launch_and_duplicate_release_do_not_count():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    state, event = release(state, 0, initialization_launch=jnp.array([True]))
    assert not bool(event[0])
    state, event = apex(state)
    assert int(state.apex_count[0]) == 0
    state, accepted = release(state, 1)
    state, duplicate = release(state, 1)
    assert bool(accepted[0]) and not bool(duplicate[0])
    assert int(state.hit_count[0]) == 1


def test_arrival_by_third_hit_then_ten_consecutive_apexes_and_full_survival():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    for i in range(1, 17):
        state, _ = release(state, i)
        state, result = apex(state, good=i >= 3)
        assert bool(result.completed[0]) == (i == 16)
        if i < 16:
            assert not bool(result.success[0])
    assert bool(result.success[0])
    assert int(state.arrival_hit[0]) == 3
    assert int(state.stability_start_hit[0]) == 3
    assert int(state.max_stable_streak[0]) == 14


def test_late_arrival_is_failure_even_with_ten_later_good_apexes():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    for i in range(1, 17):
        state, _ = release(state, i)
        state, result = apex(state, good=i >= 4)
    assert bool(result.completed[0]) and not bool(result.success[0])
    assert int(state.arrival_hit[0]) == 4


def test_interrupted_stability_run_cannot_be_mistaken_for_ten_consecutive():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    for i in range(1, 17):
        state, _ = release(state, i)
        state, result = apex(state, good=i != 8)
    assert not bool(result.success[0])


def test_drop_after_ten_good_contacts_still_fails_goal():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    for i in range(1, 12):
        state, _ = release(state, i)
        state, _ = apex(state)
    state, result = record_failure(state, jnp.array([True]))
    assert bool(result.completed[0]) and not bool(result.success[0])


def test_goal_switch_waits_for_last_apex_and_preserves_impact_identity():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    new_goal = jnp.array([[.1, -.6, 1.4]])
    for i in range(1, 17):
        state, _ = release(state, i)
        if i == 16:
            unchanged, switched = switch_completed_goals(state, new_goal)
            assert not bool(switched[0])
            np.testing.assert_array_equal(unchanged.goal_xyz, state.goal_xyz)
        state, _ = apex(state)
    last_id = state.last_impact_id
    state, switched = jax.jit(switch_completed_goals)(state, new_goal)
    assert bool(switched[0]) and int(state.goal_id[0]) == 1
    np.testing.assert_array_equal(state.last_impact_id, last_id)
    np.testing.assert_array_equal(state.goal_xyz, new_goal)
    assert int(state.hit_count[0]) == 0
    state, accepted = release(state, 16)
    assert not bool(accepted[0])


def test_batched_jit_matches_eager_event_replay():
    state = init_task(jnp.array([[0., -.5, 1.425], [.1, -.6, 1.35]]))
    kwargs = dict(impact_id=jnp.array([1, 1]), physical_release=jnp.array([True, True]),
                  controlled_hit=jnp.array([True, False]))
    eager = record_release(state, **kwargs)
    compiled = jax.jit(record_release)(state, **kwargs)
    for a, b in zip(jax.tree.leaves(eager), jax.tree.leaves(compiled)):
        np.testing.assert_array_equal(a, b)


def test_uncontrolled_touch_cannot_reassign_a_later_apex_to_an_older_hit():
    state = init_task(jnp.array([[0., -.5, 1.425]]))
    state, _ = release(state, 1)
    state, accepted = record_release(state, impact_id=jnp.array([2]),
        physical_release=jnp.array([True]), controlled_hit=jnp.array([False]))
    state, result = apex(state)
    assert not bool(accepted[0])
    assert not bool(result.valid_apex[0])
    assert bool(state.failed[0])
    assert int(state.apex_count[0]) == 0
