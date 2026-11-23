"""Physical diagnostics retain failed predictions and use native event times."""

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))

from world_model.contact_dynamics import ContactState


def test_release_error_keeps_missed_and_nonfinite_predictions():
    from world_model.physics_metrics import ball_metrics
    position = jnp.zeros((3, 2, 3))
    velocity = position.at[..., 2].set(1.)
    state = ContactState(position, velocity, position, jnp.zeros((3, 2), bool), jnp.zeros((3, 2)))
    release = jnp.zeros((3, 2), bool).at[1].set(True)
    target = dict(before=state, after=state, controlled_release=release,
                  valid=jnp.ones((3, 2), bool), entry=release, release=release)
    predicted = state._replace(velocity_m_s=velocity.at[1, 0, 0].set(.2).at[1, 1, 0].set(jnp.nan))
    trace = dict(before=predicted, after=predicted, entry=jnp.zeros((3, 2), bool),
                 release=jnp.zeros((3, 2), bool))
    metrics, events = ball_metrics(trace, target)
    values, mask = metrics['outgoing_velocity_m_s']
    np.testing.assert_array_equal(mask, release)
    np.testing.assert_allclose(values[1, 0], .2)
    assert np.isnan(values[1, 1])
    assert int(events['release_missed']) == 2
    assert int(events['true_releases']) == 2


def test_precontact_mask_uses_only_observed_future_entry_within_25ms():
    from world_model.physics_metrics import precontact_mask
    entries = jnp.zeros((32, 2), bool).at[27, 0].set(True).at[0, 1].set(True)
    mask = np.asarray(precontact_mask(entries, jnp.ones_like(entries)))
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 0]), np.arange(2, 28))
    np.testing.assert_array_equal(np.flatnonzero(mask[:, 1]), [0])
