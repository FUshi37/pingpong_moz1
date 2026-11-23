"""Default-identity absolute-height reweighting and bilateral margin costs."""

import math
import jax
import jax.numpy as jnp
from fixed_plane_v92 import SignedHeightReward


def reweight_absolute_height(reward: SignedHeightReward, *,
                             multiplier: float) -> SignedHeightReward:
    """Scale only the absolute score; keep adjacent and unresolved components."""
    if not math.isfinite(float(multiplier)) or multiplier <= 0.:
        raise ValueError("Absolute height multiplier must be finite and positive")
    if multiplier == 1.:
        return reward
    absolute = reward.absolute * float(multiplier)
    return reward._replace(absolute=absolute,
        total=absolute + reward.adjacent_penalty + reward.unresolved_penalty)


def symmetric_height_margin_penalty(*, eligible: jax.Array,
                                   apex_z_m: jax.Array, weight: float) -> jax.Array:
    """Cost only resolved late impacts, zero within1.415--1.435m.

    The quadratic cost reaches weight at either registered band edge and
    saturates at four times weight. Existing non-saturating height costs remain.
    """
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError("Height margin weight must be finite and nonnegative")
    mask=jnp.asarray(eligible,dtype=bool)
    apex=jnp.asarray(apex_z_m,dtype=jnp.float32)
    if mask.ndim != 1 or mask.shape != apex.shape:
        raise ValueError("Height-margin inputs must have identical (N,) shapes")
    excess=jnp.maximum(jnp.abs(apex-1.425)-.010,0.)/.015
    return jnp.where(mask,-float(weight)*jnp.minimum(excess*excess,4.),0.)


def damping_focus_margin_penalty(*, eligible: jax.Array, apex_z_m: jax.Array,
                                damping_multiplier: jax.Array,
                                damping_range: tuple[float, float],
                                weight: float) -> jax.Array:
    """Focus late margin cost on low-damping overshoot and high-damping undershoot.

    Privileged damping affects reward only; actor inputs and physical sampling
    remain unchanged. Zero cost at nominal damping and inside the margin band.
    """
    lo, hi = damping_range
    if not (math.isfinite(lo) and math.isfinite(hi) and 0. < lo < 1. < hi):
        raise ValueError('Damping focus requires a finite range around unity')
    kd = jnp.asarray(damping_multiplier, dtype=jnp.float32)
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    if kd.shape != apex.shape:
        raise ValueError('Damping/apex must have identical (N,) shapes')
    delta = kd - 1.
    normalized = jnp.where(delta < 0., delta / (1. - lo), delta / (hi - 1.))
    focus = jnp.clip(-jnp.sign(apex - 1.425) * normalized, 0., 1.)
    return symmetric_height_margin_penalty(
        eligible=eligible, apex_z_m=apex, weight=weight) * focus


def late_vxy_excess_penalty(*, eligible: jax.Array, racket_vxy_m_s: jax.Array,
                           weight: float) -> jax.Array:
    """Penalize late impact XY speed above0.040m/s without a saturation tail."""
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError('Late XY weight must be finite and nonnegative')
    mask = jnp.asarray(eligible, dtype=bool)
    speed = jnp.asarray(racket_vxy_m_s, dtype=jnp.float32)
    if mask.ndim != 1 or speed.shape != mask.shape:
        raise ValueError('Late XY inputs must have identical (N,) shapes')
    return jnp.where(mask, -float(weight) * jnp.maximum(speed - .040, 0.) ** 2, 0.)


def late_action_variation_penalty(*, action_delta: jax.Array,
                                  confirmed_hits: jax.Array, weight: float) -> jax.Array:
    """Negative cost per policy tick after three confirmed launch contacts.

    Inputs are normalized seven-joint action differences (N,7) and counts (N,).
    This changes reward only, without filtering actions or changing sampling.
    """
    if not math.isfinite(float(weight)) or weight < 0.:
        raise ValueError('Action variation weight must be finite and nonnegative')
    delta = jnp.asarray(action_delta, dtype=jnp.float32)
    counts = jnp.asarray(confirmed_hits)
    if delta.ndim != 2 or delta.shape[-1] != 7 or counts.shape != delta.shape[:1]:
        raise ValueError('Action variation requires (N,7) actions and (N,) counts')
    return jnp.where(counts >= 3, -float(weight) * jnp.sum(delta ** 2, axis=-1), 0.)
