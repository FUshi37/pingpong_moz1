"""Complete numerical seed restoration and phase-safe batched selection."""

from pathlib import Path
import sys
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import pytest

SIM = Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'
sys.path.insert(0, str(SIM))

from world_model.state_adapter import pack_numeric_state, restore_numeric_state, concatenate_phase_group, numeric_tree_digest


class Seed(NamedTuple):
    position: object
    rng: object
    queue: object
    shared_index: object


def seed(index=3):
    return Seed(jnp.array([[.1, .2, .3]]), jnp.array([[4, 5]], jnp.uint32),
                jnp.arange(35, dtype=jnp.float32).reshape(1, 5, 7), jnp.array(index))


def test_seed_roundtrip_keeps_history_rng_and_scalar_phase():
    before = seed()
    arrays, manifest = pack_numeric_state(before, context_sha256='a' * 64)
    zero_template = Seed(jnp.zeros((1, 3)), jnp.zeros((1, 2), jnp.uint32),
                         jnp.zeros((1, 5, 7)), jnp.array(0))
    restored = restore_numeric_state(zero_template, arrays, manifest, context_sha256='a' * 64)
    for left, right in zip(before, restored):
        np.testing.assert_array_equal(left, right)


def test_seed_restore_rejects_wrong_context_or_corrupted_leaf():
    arrays, manifest = pack_numeric_state(seed(), context_sha256='a' * 64)
    with pytest.raises(ValueError, match='context'):
        restore_numeric_state(seed(), arrays, manifest, context_sha256='b' * 64)
    arrays['leaf_000000'] = arrays['leaf_000000'] + 1
    with pytest.raises(ValueError, match='integrity'):
        restore_numeric_state(seed(), arrays, manifest, context_sha256='a' * 64)


def test_unrelated_scalar_ring_phases_cannot_be_silently_concatenated():
    joined = concatenate_phase_group([seed(3), seed(3)])
    assert joined.position.shape == (2, 3)
    assert joined.shared_index.shape == ()
    with pytest.raises(ValueError, match='phase'):
        concatenate_phase_group([seed(3), seed(4)])


def test_episode_context_identity_cannot_ignore_sampled_mass_or_controller_phase():
    original = {'mass': jnp.array([.004]), 'controller': seed(3)}
    changed_mass = dict(original, mass=jnp.array([.0041]))
    changed_phase = dict(original, controller=seed(4))
    assert len(numeric_tree_digest(original)) == 64
    assert numeric_tree_digest(original) != numeric_tree_digest(changed_mass)
    assert numeric_tree_digest(original) != numeric_tree_digest(changed_phase)
