"""Default-off release-outcome precision costs with native wind dynamics."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

import numpy as np

from fixed_plane_wind_balanced_precontact import BIRTH_MODE, BIRTH_SHA256
from fixed_plane_wind_dynamics_guidance import DRAG_FACTOR_KG_M

MODE = 'clean10_dynamics_reference_center_negy_v8'


def release_contract() -> dict:
    """Describe scalar rewards separately from unchanged physical gate metrics."""
    return dict(revision='model_release_outcome_precision_v1', apex_target_m=1.425,
        return_sigma_m={'325': .0055, '326': .0065}, apex_sigma_m=.008,
        cost_per_release_per_component=4., huber_cap_per_component=8.,
        timing='existing upward detached physical release, recent/live, original shelter/ramp',
        forecast='existing native incremental-wind integrator, causal wind samples, actual release state/mass/gravity; existing reference period/episode anchor',
        terms='independent nonpositive Huber forecast-return-XY and forecast-apex costs; no positive release income',
        warmup='same reward-only 8192-step preview/balance clock; base reward mature',
        preserved='all gate observations/definitions/thresholds/hold, physics, sampling, policy and learner')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindReleasePrecisionConfig', [
        ('wind_release_precision', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_release_config(cfg: Any, enabled: bool) -> Any:
    """Add exactly one reward flag; leave every previous recipe untouched."""
    if not enabled:
        return cfg
    if not getattr(cfg, 'wind_precontact_balance', False):
        raise ValueError('Release precision requires its reviewed balanced preview recipe')
    return _config_type(type(cfg))(**vars(cfg))


def forecast_release(start_m: Any, velocity_m_s: Any, duration_s: Any,
                     wind_m_s: Any, mass_kg: Any, gravity_m_s2: Any,
                     return_m: Any, *, xp: Any, drag_factor_kg_m: float = DRAG_FACTOR_KG_M,
                     force_limit_n: float = .020) -> dict:
    """Predict [N] outcome errors using the unchanged per-lane wind model.

    Positions/velocities have shape [N,3], wind [N,32,3], and physical parameters
    [N]. Gravity is evaluated separately per lane, including measured DR.
    The returned forecast is advisory reward information, never gate telemetry.
    """
    from fixed_plane_wind_dynamics_guidance import integrate_flight
    n = start_m.shape[0]
    duration_s = xp.broadcast_to(xp.asarray(duration_s), (n,))
    mass_kg = xp.broadcast_to(xp.asarray(mass_kg), (n,))
    gravity_m_s2 = xp.broadcast_to(xp.asarray(gravity_m_s2), (n,))

    def one(start, velocity, duration, wind, mass, gravity):
        return integrate_flight(start, velocity, duration, wind, mass, xp=xp,
            drag_factor_kg_m=drag_factor_kg_m, force_limit_n=force_limit_n,
            gravity_m_s2=gravity)

    if xp is np:
        rows = [one(*values) for values in zip(start_m, velocity_m_s, duration_s,
                                               wind_m_s, mass_kg, gravity_m_s2)]
        result = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
    else:
        import jax
        result = jax.vmap(one)(start_m, velocity_m_s, duration_s, wind_m_s,
                              mass_kg, gravity_m_s2)
    error = xp.linalg.norm(result['endpoint_m'][..., :2] - return_m[..., :2], axis=-1)
    apex_error = result['apex_m'] - 1.425
    valid = ((duration_s > 0.) & (mass_kg > 0.) & (gravity_m_s2 > 0.)
             & xp.isfinite(error) & xp.isfinite(apex_error))
    return dict(valid=valid, return_error_m=xp.where(valid, error, 0.),
                apex_error_m=xp.where(valid, apex_error, 0.))


def release_precision_terms(metrics: dict, prediction: dict, *, gain: Any,
                            reward_steps: Any, version: int, xp: Any) -> dict:
    """Return independent, capped nonpositive [N] physical-release costs."""
    if version not in (325, 326):
        raise ValueError('Release precision is scoped to V325/V326')
    dtype = metrics['dynamics/release_event'].dtype
    progress = xp.clip(xp.asarray(reward_steps, dtype=dtype)/8192., 0., 1.)
    warmup = progress**2*(3.-2.*progress)
    active = (metrics['dynamics/release_event'] * (metrics['dynamics/recent'] > .5)
              * (metrics['terminated'] < .5) * gain)
    valid = prediction['valid'] & (metrics['dynamics/reference_valid'] > .5)

    def loss(error, sigma):
        normalized = xp.abs(xp.nan_to_num(error, nan=0., posinf=0., neginf=0.))/sigma
        huber = .5*xp.minimum(normalized, 1.)**2 + xp.maximum(normalized-1., 0.)
        return xp.where(valid, xp.minimum(huber, 8.), 8.)

    xy = loss(prediction['return_error_m'], .0055 if version == 325 else .0065)
    apex = loss(prediction['apex_error_m'], .008)
    return {
        'reward/release_return_precision_cost': -4.*active*warmup*xy,
        'reward/release_apex_precision_cost': -4.*active*warmup*apex,
        'release_precision/warmup': xp.ones_like(active)*warmup,
        'release_precision/active': active,
        'release_precision/valid': active*valid,
        'release_precision/invalid': active*~valid,
        'release_precision/return_error_m_sum': active*xp.where(valid, xp.nan_to_num(prediction['return_error_m']), 0.),
        'release_precision/apex_error_m_sum': active*xp.where(valid, xp.nan_to_num(prediction['apex_error_m']), 0.),
    }


def environment_terms(env: Any, core: Any, metrics: dict, *, gain: Any,
                      reward_steps: Any) -> dict:
    """Read native state and return scalar rewards without changing any state."""
    import jax.numpy as jnp
    from fixed_plane_wind_dynamics_guidance import expected_wind_samples
    duration = jnp.clip(jnp.nan_to_num(metrics['dynamics/period_s'], nan=.5), .08, 1.2)
    wind = expected_wind_samples(core, env.cfg.wind_config, core.step_count*env.dt,
                                 duration, xp=jnp)
    reference = core.racket_anchor.at[:, 2].set(metrics['dynamics/target_return_z_m'])
    prediction = forecast_release(
        core.data.qpos[:, env.ball_qadr:env.ball_qadr+3],
        core.data.qvel[:, env.ball_vadr:env.ball_vadr+3], duration, wind,
        core.model.body_mass[:, env.ball_body_id], jnp.abs(core.dr_gravity_z), reference,
        xp=jnp, drag_factor_kg_m=env.cfg.wind_config.drag_factor_kg_m,
        force_limit_n=env.cfg.wind_config.force_limit_n)
    return release_precision_terms(metrics, prediction, gain=gain,
        reward_steps=reward_steps, version=env.cfg.wind_center_precision_version, xp=jnp)


def source_resume_context(payload: dict, sha256: str) -> dict:
    """Keep the pinned common parent and exact own-recipe reward phase."""
    from fixed_plane_wind_balanced_precontact import source_resume_context as balanced_context
    from fixed_plane_wind_balanced_precontact import MODE as balanced_mode
    mode = payload.get('args', {}).get('wind_training_mix')
    if mode == BIRTH_MODE:
        return balanced_context(payload, sha256)
    if mode != MODE or payload.get('env_cfg', {}).get('wind_release_precision') is not True:
        raise ValueError('Only the pinned parent or this exact release reward may resume')
    # Reuse the immutable common-counter/phase validator after authenticating
    # this recipe explicitly; the serialized source is never modified.
    checked = {**payload, 'args': {**payload['args'], 'wind_training_mix': balanced_mode}}
    return balanced_context(checked, sha256)
