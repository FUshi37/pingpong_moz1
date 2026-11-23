"""A new ball trial may reset perception, but must retain warmed controls."""

from collections import namedtuple
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.teacher_initialization import carry_warmed_controller, camera_cache_fields
from world_model.observation_engine import init_camera
from world_model.teacher_trace import contact_pair_masks


def test_new_trial_retains_controller_memory_and_episode_parameters():
    State = namedtuple('State', ('arm_cmd_q', 'rmp_state', 'rmp_output_q_delay_buffer',
        'prev_action', 'prev_arm_qvel', 'prev_racket_pos', 'prev_racket_vel',
        'action_buffer', 'execution_last_command_q', 'command_buffer', 'dr_ball_mass',
        'rng', 'step_count', 'cached_ball_obs_pos', 'hit_count', 'prev_contact'))
    warm = State(*(jnp.asarray(i+10.) for i in range(len(State._fields))))
    fresh = State(*(jnp.asarray(0.) for _ in State._fields))
    result = carry_warmed_controller(fresh, warm)
    for name in State._fields[:12]:
        np.testing.assert_array_equal(getattr(result, name), getattr(warm, name))
    np.testing.assert_array_equal(result.rng, warm.rng)
    assert result.step_count == result.hit_count == 0
    assert result.cached_ball_obs_pos == result.prev_contact == 0


def test_contact_labels_separate_other_body_collisions_and_inactive_pairs():
    geoms = jnp.array([[[1, 2], [3, 1], [2, 4]], [[2, 1], [1, 3], [1, 4]]])
    distance = jnp.array([[-.001, -.002, -.1], [.002, .001, -.001]])
    pair, other = contact_pair_masks(geoms, distance, ball_geom_id=1, racket_geom_id=2)
    np.testing.assert_array_equal(pair, [[True, False, False], [True, False, False]])
    np.testing.assert_array_equal(other, [[False, True, False], [False, False, True]])


def test_native_last_valid_sample_fields_are_vectors_not_boolean_flags():
    p = jnp.arange(12, dtype=jnp.float32).reshape(4, 3)
    v = p*.01
    camera = init_camera(p, jnp.zeros((4, 2), jnp.uint32), ar1_enabled=False)
    fields = camera_cache_fields(camera, v)
    np.testing.assert_array_equal(fields['ball_obs_valid_pos'], p)
    np.testing.assert_array_equal(fields['ball_obs_valid_vel'], v)
    assert fields['fixed_contact_ball_observed'].shape == (4,)
    assert fields['fixed_contact_ball_observed'].dtype == bool
