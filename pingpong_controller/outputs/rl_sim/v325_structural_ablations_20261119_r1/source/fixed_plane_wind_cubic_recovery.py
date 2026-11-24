"""Default-off periodic wind recovery reward; no actions or physical state writes."""
from dataclasses import field, make_dataclass
from functools import lru_cache
import math
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE, verify_source,
)

MODE = 'clean10_dynamics_reference_center_negy_v20'
WARMUP_STEPS = 8192
INCOMPATIBLE_FLAGS = (
    'wind_clean_group_priority', 'wind_normalized_group_priority',
    'wind_clean_apex_priority', 'wind_precontact_center', 'wind_precontact_balance',
    'wind_release_precision', 'wind_cycle_release_precision', 'wind_measured_precision',
    'wind_steady_braking', 'wind_confirmed_launch_braking', 'wind_center_tail',
    'wind_calm_release', 'wind_early_recovery', 'wind_mature_precision_reward',
    'wind_calm_return_height', 'wind_impact_normal_precision',
)


def cubic_recovery_contract() -> dict:
    return dict(base=BASE_MODE, population='wind only',
        revision='safe_periodic_return_velocity_v1', preserved_height_reward=True,
        target='base-frame normal-stroke cubic velocity; same midpoint and positive contact velocity',
        restitution_reference=.592, reference_height_reserve_m=.015,
        safety=dict(preview_s=.080, old_target_margin_m=.025, full_blend_margin_m=.040),
        return_loss='unchanged4/s,0.10m/s sigma,squared cap4,native phase and onset gain',
        warmup_policy_steps=WARMUP_STEPS, original_reward_clock='unchanged at birth',
        own_resume='preserve original precision and correction phases; no physical clock edits',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE, true_step=SOURCE_STEP,
                    Adam_t=SOURCE_ADAM_T, stage_index=15),
        preserved='clean rewards,height costs,braking,terminal,physics,observations,PPO,sampling,gates;BC0')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCubicRecoveryConfig', [
        ('wind_cubic_recovery', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_cubic_recovery_config(cfg: Any, enabled: bool) -> Any:
    if not enabled:
        return cfg
    if any(getattr(cfg, flag, False) for flag in INCOMPATIBLE_FLAGS):
        raise ValueError('Cubic recovery cannot stack a different reward recipe')
    return _config_type(type(cfg))(**vars(cfg))


def cubic_reward_phase(payload: dict, sha256: str) -> int:
    verify_source(payload, sha256, mode=MODE, config_flag='wind_cubic_recovery')
    if any(payload.get('env_cfg', {}).get(flag, False) for flag in INCOMPATIBLE_FLAGS):
        raise ValueError('Cubic recovery cannot stack a different reward recipe')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)


def periodic_recovery_reference(
    incoming_m_s: Any, outgoing_m_s: Any, *, period_s: Any, elapsed_s: Any,
    bounds_m: tuple[float, float], xp: Any,
) -> dict:
    """Return [...]-shaped base-frame Z targets for a closed periodic stroke.

    The existing .592 nominal collision reference supplies the advisory
    normal-stroke speed. A cubic Hermite curve returns to the midpoint with
    the same positive contact velocity, with a slower downward return in
    between. This is a reward target, not an exact inverse of contact or a
    certified robot trajectory. Only the explicit reward recipe uses it.
    """
    lower_m, upper_m = bounds_m
    if not (math.isfinite(lower_m) and math.isfinite(upper_m)
            and upper_m - lower_m > .030):
        raise ValueError('Periodic reference requires finite bounds with15mm reserve')
    incoming, outgoing = xp.asarray(incoming_m_s), xp.asarray(outgoing_m_s)
    period, elapsed = xp.asarray(period_s), xp.asarray(elapsed_s)
    finite_velocities = xp.all(xp.isfinite(incoming) & xp.isfinite(outgoing), axis=-1)
    incoming = xp.where(finite_velocities[..., None], incoming, 0.)
    outgoing = xp.where(finite_velocities[..., None], outgoing, 0.)
    finite_clock = xp.isfinite(period) & xp.isfinite(elapsed)
    safe_period = xp.where(xp.isfinite(period) & (period > 0.), period, 1.)
    safe_elapsed = xp.where(xp.isfinite(elapsed), elapsed, -1.)
    phase = safe_elapsed / safe_period
    s = xp.clip(phase, 0., 1.)
    delta = outgoing - incoming
    magnitude = xp.linalg.norm(delta, axis=-1)
    normal = delta / xp.maximum(magnitude[..., None], 1.e-9)
    normal_speed = xp.sum((outgoing + .592 * incoming) * normal, axis=-1) / 1.592
    vertical_speed = normal_speed * normal[..., 2]
    amplitude = xp.abs(vertical_speed * safe_period) / (6. * math.sqrt(3.))
    acceleration = 6. * xp.abs(vertical_speed) / safe_period
    valid = (finite_velocities & finite_clock & (period > 0.)
             & (phase >= 0.) & (phase <= 1.) & (magnitude > 1.e-9)
             & (normal[..., 2] > 0.) & (vertical_speed >= 0.)
             & (amplitude <= .5 * (upper_m - lower_m) - .015))
    midpoint = .5 * (lower_m + upper_m)
    z_m = midpoint + vertical_speed * safe_period * (s - 3. * s**2 + 2. * s**3)
    vz_m_s = vertical_speed * (1. - 6. * s + 6. * s**2)
    return dict(z_m=xp.where(valid, z_m, midpoint),
                vz_m_s=xp.where(valid, vz_m_s, 0.), valid=valid,
                amplitude_m=amplitude, max_abs_acceleration_m_s2=acceleration)


def safe_recovery_velocity(
    racket_z_m: Any, racket_vz_m_s: Any, periodic: dict, *,
    bounds_m: tuple[float, float], xp: Any,
) -> dict:
    """Keep the original velocity target near the current or previewed edge.

    The25mm reserve and80ms constant-velocity preview reuse the existing
    braking reward's quantities. They are reward blending, not new gates or
    a certified stopping-distance calculation.
    """
    lower_m, upper_m = bounds_m
    z, vz = xp.asarray(racket_z_m), xp.asarray(racket_vz_m_s)
    midpoint = .5 * (lower_m + upper_m)
    original = xp.clip(-4. * (z - midpoint), -.15, .15)
    preview = z + .080 * vz
    margin = xp.minimum(xp.minimum(z - lower_m, upper_m - z),
                        xp.minimum(preview - lower_m, upper_m - preview))
    progress = xp.clip((margin - .025) / .015, 0., 1.)
    valid = periodic['valid'] & xp.isfinite(periodic['vz_m_s'])
    blend = xp.where(valid, progress**2 * (3. - 2. * progress), 0.)
    desired = xp.where(valid, periodic['vz_m_s'], original)
    return dict(vz_m_s=xp.where(blend > 0., original + blend * (desired - original), original),
                blend=blend)


def cubic_return_correction(
    metrics: dict, incoming_m_s: Any, outgoing_m_s: Any, *, period_s: Any,
    elapsed_s: Any, windy: Any, gain: Any, reference_valid: Any,
    reward_steps: Any, dt_s: float, bounds_m: tuple[float, float], xp: Any,
) -> dict:
    """Replace only the existing wind return component; keep protected outputs exact."""
    periodic = periodic_recovery_reference(incoming_m_s, outgoing_m_s,
        period_s=period_s, elapsed_s=elapsed_s, bounds_m=bounds_m, xp=xp)
    safe = safe_recovery_velocity(metrics['racket_z'], metrics['dynamics/racket_vz_m_s'],
        periodic, bounds_m=bounds_m, xp=xp)
    active = (windy & reference_valid & periodic['valid'] & (safe['blend'] > 0.)
        & (metrics['dynamics/recovery_phase'] > .5)
        & (metrics['dynamics/confirmed_hits'] >= 5)
        & (metrics['dynamics/recent'] > .5) & (metrics['terminated'] < .5))
    onset = 1. + 2. * xp.exp(-xp.maximum(metrics['dynamics/confirmed_hits'] - 5., 0.) / 3.)
    error = (metrics['dynamics/racket_vz_m_s'] - safe['vz_m_s']) / .10
    desired = -gain * 4. * dt_s * onset * xp.minimum(error**2, 4.)
    original = metrics['reward/dynamics_posthit_return']
    progress = xp.clip(reward_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)
    active = active & (warmup > 0.)
    replacement = xp.where(active, original + warmup * (desired - original), original)
    delta = xp.where(active, replacement - original, 0.)
    return {'reward/cubic_return_delta': delta,
            'cubic_recovery/replacement_return_reward': replacement,
            'cubic_recovery/original_return_reward': original,
            'cubic_recovery/target_vz_m_s': safe['vz_m_s'],
            'cubic_recovery/target_z_m': periodic['z_m'],
            'cubic_recovery/safety_blend': safe['blend'],
            'cubic_recovery/active': active.astype(xp.float32),
            'cubic_recovery/warmup': warmup}
