"""Pure, default-off small-recovery DR operations without privileged inputs."""

import math
from typing import Mapping, Sequence

import jax.numpy as jnp


def episode_weighted_summaries(rows: Sequence[Mapping]) -> dict[str, float]:
    """Weight completed-episode summaries by their actual population counts."""
    counts = [float(row.get("episodes", 0)) for row in rows]
    if any(not math.isfinite(n) or n < 0 for n in counts):
        return dict(episodes=float("nan"), mean_hits=float("nan"),
                    mean_len=float("nan"), mean_return=float("nan"))
    total = sum(counts)
    result = {"episodes": total}
    for key in ("mean_hits", "mean_len", "mean_return"):
        values = [(n, float(row.get(key, float("nan"))))
                  for n, row in zip(counts, rows, strict=True) if n > 0]
        result[key] = (sum(n * value for n, value in values) / total
                       if total > 0 and all(math.isfinite(v) for _, v in values)
                       else float("nan"))
    return result


def flight_impulse_velocity(velocity_m_s, delta_xy_m_s, *, step, hit_count,
                            in_contact, clearance_m, scheduled_step: int,
                            min_hits: int, min_clearance_m: float):
    """Apply one scheduled XY impulse; position and vertical velocity stay exact."""
    applied = ((step == scheduled_step) & (hit_count >= min_hits)
               & (~in_contact) & (clearance_m >= min_clearance_m))
    delta = jnp.concatenate((delta_xy_m_s, jnp.zeros_like(delta_xy_m_s[:, :1])), axis=-1)
    return velocity_m_s + jnp.where(applied[:, None], delta, 0.), applied


def contact_pair_values(hit_count, *, sampled_mu, sampled_time, sampled_damping,
                        nominal_friction, nominal_solref, min_hits: int):
    """Keep launch nominal and coherently scale both tangent directions."""
    dynamic = hit_count >= min_hits
    factor = jnp.where(dynamic, sampled_mu/nominal_friction[0], 1.)
    friction = factor[:,None]*jnp.asarray(nominal_friction)
    return (friction, jnp.where(dynamic,sampled_time,nominal_solref[0]),
            jnp.where(dynamic,sampled_damping,nominal_solref[1]))


def camera_dropout_transition(*, refresh, remaining, missing, start, duration):
    """Consume missing camera frames only at refresh; require reacquisition."""
    started = refresh & (remaining == 0) & (~missing) & start
    blocked = refresh & ((remaining > 0) | started)
    left = jnp.where(refresh,
        jnp.where(remaining > 0, remaining - 1,
                  jnp.where(started, duration - 1, 0)), remaining)
    return blocked, left.astype(jnp.int32), started


def fractional_camera_refresh(step, last_sample_step, ratio: float, *, independent: bool):
    """Keep the camera clock independent of successful-measurement timestamps."""
    previous=step-1 if independent else last_sample_step
    return (jnp.floor(step.astype(jnp.float32)*ratio)
            > jnp.floor(previous.astype(jnp.float32)*ratio))


def short_gap_prediction(position_m, velocity_m_s, source_age_s, *, missing,
                         max_gap_s: float, gravity_z: float):
    """Bounded current-time ballistic estimate from the last accepted observation.

    Source age is separate from represented-state age. No simulator truth or
    contact oracle is accepted; a gap beyond the explicit horizon is invalid.
    """
    predicted = missing & (source_age_s <= max_gap_s) & (source_age_s >= 0.)
    valid = (~missing) | predicted
    t = source_age_s[:, None]
    gravity = jnp.array([0., 0., gravity_z], dtype=velocity_m_s.dtype)
    p = position_m + velocity_m_s * t + .5 * gravity * t * t
    v = velocity_m_s + gravity * t
    return (jnp.where(predicted[:, None], p,
                     jnp.where(valid[:, None], position_m, 0.)),
            jnp.where(predicted[:, None], v,
                     jnp.where(valid[:, None], velocity_m_s, 0.)),
            predicted, valid)
