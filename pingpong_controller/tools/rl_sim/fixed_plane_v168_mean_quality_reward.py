"""Fixed-reference physical event-quality rewards, with exact metric populations."""
import math
from typing import Mapping
import jax
import jax.numpy as jnp

FIELDS = (
    ('fixed_plane_v168_center_quality_reward', 'hit_contact_center_dist_sum',
     'hit_event_count', .003),
    ('fixed_plane_v168_xy_quality_reward', 'rapid_steady_steady_racket_vxy_sq_sum',
     'rapid_steady_steady_events', .053 ** 2),
    ('fixed_plane_v168_omega_quality_reward', 'rapid_steady_steady_racket_omega_sq_sum',
     'rapid_steady_steady_events', .25 ** 2),
    ('fixed_plane_v168_adjacent_quality_reward', 'v80_late_pair_jump_sum_m',
     'v80_late_pair_event', .007),
    ('fixed_plane_v168_ball_xy_quality_reward', 'hit_vxy_sq_sum',
     'hit_event_count', .025 ** 2),
)


def mean_quality_rewards(statistics: Mapping[str, jax.Array], *,
                         weights: tuple[float, ...],
                         references: tuple[float, ...]) -> dict[str, jax.Array]:
    """Score -(numerator-reference*events)/scale without editing event accounting.

    References are fixed measured source means. Each numerator retains a strict
    improvement slope below and above its reference; there is no reward deadband.
    The event offset prevents a quality cost from rewarding fewer measured events.
    Geometry, event masks and denominators come from the existing physical ledger.
    """
    if (len(weights) != len(FIELDS) or len(references) != len(FIELDS)
            or not all(math.isfinite(float(w)) and w >= 0. for w in weights)
            or not all(math.isfinite(float(r)) and r > 0. for r in references)):
        raise ValueError('Five finite nonnegative weights and positive references required')
    result = {}
    shape = None
    for (key, numerator, denominator, scale), weight, reference in zip(
            FIELDS, weights, references):
        value = jnp.asarray(statistics[numerator], dtype=jnp.float32)
        events = jnp.asarray(statistics[denominator], dtype=jnp.float32)
        if value.shape != events.shape or (shape is not None and value.shape != shape):
            raise ValueError('Physical quality statistics must share shape (N,)')
        shape = value.shape
        result[key] = jnp.where(events > 0.,
            -float(weight) * (value - float(reference) * events) / scale, 0.)
    return result
