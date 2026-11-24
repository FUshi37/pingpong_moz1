"""Opt-in 4 m/s flight budgets and quality-qualified pose expansion."""
from __future__ import annotations

from dataclasses import asdict,dataclass,field,make_dataclass
from functools import lru_cache
import math

import numpy as np

REVISION = 'wind4_full20_time_exposure_20261114'
FORCE_LIMIT_N = .020
REFERENCE_LIFT_M = .380
REFERENCE_MASS_KG = .004
REFERENCE_SAMPLES = 24


@lru_cache(maxsize=4)
def _wind_config_type(parent_type):
    return make_dataclass('StrongWindConfig',[('peak_wind_fraction',float,field(default=.25))],
                          bases=(parent_type,),frozen=True)


def strong_wind_config(parent, peak_fraction):
    if peak_fraction == 0.:
        return parent
    if not 0. <= peak_fraction <= 1.:
        raise ValueError('Invalid maximum-wind episode fraction')
    return _wind_config_type(type(parent))(**asdict(parent),peak_wind_fraction=peak_fraction)


def flight_reference(wind_samples, *, xp, lift_m=REFERENCE_LIFT_M):
    """Midpoint/Picard same-position flight; input (...,24,3), output (...,2)."""
    count = wind_samples.shape[-2]
    u = (xp.arange(count, dtype=wind_samples.dtype) + .5) / count
    vz0 = xp.sqrt(2. * 9.81 * xp.asarray(lift_m))
    flight_s = 2. * vz0 / 9.81
    vertical = xp.broadcast_to(vz0[...,None] - 9.81 * flight_s[...,None] * u,
                               wind_samples.shape[:-1])
    horizontal = xp.zeros_like(wind_samples[..., :2])
    factor = .5 * 1.2 * .47 * math.pi * .020**2
    def refine(_, carry):
        horizontal = carry[0]
        velocity = xp.concatenate((horizontal, vertical[..., None]), axis=-1)
        relative = velocity - wind_samples
        raw = factor * (xp.linalg.norm(velocity, axis=-1, keepdims=True) * velocity
                        - xp.linalg.norm(relative, axis=-1, keepdims=True) * relative)
        force = raw * xp.minimum(1., FORCE_LIMIT_N / xp.maximum(
            xp.linalg.norm(raw, axis=-1, keepdims=True), 1e-12))
        acceleration = force[..., :2] / REFERENCE_MASS_KG
        target = -flight_s[...,None] * xp.mean((1. - u)[..., None] * acceleration, axis=-2)
        horizontal = target[..., None, :] + flight_s[...,None,None] / count * (
            xp.cumsum(acceleration, axis=-2) - .5 * acceleration)
        delta = flight_s[...,None] * xp.mean(acceleration, axis=-2)
        peak = xp.max(xp.linalg.norm(raw, axis=-1), axis=-1)
        return horizontal,target,delta,peak
    zero_xy=xp.zeros_like(wind_samples[...,0,:2])
    carry=(horizontal,zero_xy,zero_xy,xp.zeros_like(wind_samples[...,0,0]))
    if xp is np:
        for index in range(5):
            carry=refine(index,carry)
    else:
        import jax
        carry=jax.lax.fori_loop(0,5,refine,carry)
    _,target,delta,peak=carry
    tilt = xp.arctan2(xp.linalg.norm(delta, axis=-1), 2. * vz0) * (180. / math.pi)
    return target, tilt, peak


@lru_cache(maxsize=32)
def constant_reference(speed_m_s: float) -> dict:
    wind = np.zeros((1, REFERENCE_SAMPLES, 3), dtype=np.float64)
    wind[..., 0] = speed_m_s
    target, tilt, peak = flight_reference(wind, xp=np)
    return dict(outgoing_speed_m_s=float(np.linalg.norm(target[0])),
                tilt_deg=float(tilt[0]), raw_force_peak_n=float(peak[0]))


def outgoing_reference(env, state):
    """Expected future wind only, including OU mean reversion and gust phase."""
    import jax.numpy as jnp
    from fixed_plane_v325_wind import startup_gain, wind_clock_time
    cfg = env.cfg.wind_config
    u = (jnp.arange(REFERENCE_SAMPLES, dtype=jnp.float32) + .5) / REFERENCE_SAMPLES
    rotation=state.data.site_xmat[:,env.racket_site_id].reshape((-1,3,3))
    normal_z=jnp.abs(rotation[:,2,2])
    surface_z=(state.data.geom_xpos[:,env.racket_geom_id,2]+normal_z*state.model.geom_size[
        :,env.racket_geom_id,int(env.cfg.fixed_contact_racket_surface_size_axis)])
    lift=jnp.clip(1.425-surface_z-.020,.20,.50)
    flight_s=2.*jnp.sqrt(2.*9.81*lift)/9.81
    offset=flight_s[:,None]*u[None,:]
    now = state.step_count * env.dt
    clock = wind_clock_time(state, now, cfg)
    phase = (clock[:, None] + offset - state.wind_gust_start_s[:, None]) / state.wind_gust_duration_s[:, None]
    active_now=(clock>=state.wind_gust_start_s)&(clock<state.wind_gust_start_s+state.wind_gust_duration_s)
    gust = jnp.where(active_now[:,None] & (phase > 0.) & (phase < 1.),
                     jnp.sin(jnp.pi * phase)**2, 0.)
    wind = (state.wind_background_m_s[:, None, :]
            + state.wind_ou_m_s[:, None, :] * jnp.exp(-offset / cfg.ou_tau_s)[..., None]
            + gust[..., None] * state.wind_gust_vector_m_s[:, None, :])
    wind *= jnp.minimum(1., cfg.max_speed_m_s / jnp.maximum(
        jnp.linalg.norm(wind, axis=-1, keepdims=True), 1e-12))
    wind = jnp.where((state.wind_mode > 0)[:, None, None], wind, 0.)
    # Startup is causal: no pre-emptive compensation before the fifth hit.
    gain = startup_gain(state, now, cfg)
    wind *= gain[:, None, None]
    return flight_reference(wind, xp=jnp,lift_m=lift)


def wind_quality_limits(version: int, speed_m_s: float) -> dict:
    from fixed_plane_wind_objectives import quality_limits
    limits = quality_limits(version)
    allowance = constant_reference(float(speed_m_s))['outgoing_speed_m_s']
    limits['ball_vxy_m_s'] += allowance
    limits['ball_vxy_rms_m_s'] += allowance
    return limits


def quality_gate(version: int, lesson, native_gate):
    """Retain physical metrics; only required ball XY motion gets a budget."""
    if not getattr(lesson, 'strong_wind_reward', False):
        return native_gate

    def build(speed):
        limits = wind_quality_limits(version, float(speed))

        def passed(q):
            return (q.get('steady_events', 0.) >= 100
                    and 1.40 <= q.get('apex_m', 0.) <= 1.45
                    and q.get('in_band', 0.) >= limits['in_band']
                    and all(math.isfinite(q.get(k, math.inf)) and q[k] <= value
                            for k, value in limits.items() if k != 'in_band'))
        return passed

    result = build(lesson.max_speed_m_s)
    result.for_wind = lambda name: build(0. if name == 'clean' else lesson.max_speed_m_s)
    return result


@dataclass
class ExpansionGate:
    """Do not widen a pose envelope while sampled-training quality is failing."""
    streak: int = 0
    required_hold: int = 16

    def add(self, *, quality_ok: bool, window_size: int,
            physical_failure: bool) -> bool:
        good = quality_ok and window_size == 32 and not physical_failure
        self.streak = self.streak + 1 if good else 0
        return self.streak >= self.required_hold


def apex_target_reward(metrics, *, xp):
    event = metrics['rapid_steady_steady_events']
    apex = metrics['rapid_steady_steady_apex_sum_z_m'] / xp.maximum(event, 1.)
    return xp.where(event > 0., 4. * xp.exp(-.5 * ((apex - 1.425) / .012)**2), 0.)


def strong_contract() -> dict:
    from fixed_plane_wind_progression import progression_contract
    return dict(revision=REVISION, max_wind_m_s=4., force_limit_n=FORCE_LIMIT_N,
                relative_roll_pitch_deg=20., remaining_lessons=23,peak_wind_episode_fraction=.25,
                pose_growth='128 actual updates to full20deg; strict quality/hold16 gate only final-angle graduation',
                course_order='pose; all constant to4m/s; slow to4m/s; mixed to4m/s',
                progression=progression_contract(),
                reference='24-midpoint five-Picard causal expected-wind flight; no unseen future gust timing',
                reference_lift_m=REFERENCE_LIFT_M, reference_mass_kg=REFERENCE_MASS_KG,
                ball_metrics='physical XY retained; wind budget only, clean gate unchanged',
                native_velocity_reward='wind-compensated velocity, physical dynamics unchanged',
                apex_target_m=1.425, apex_sigma_m=.012, apex_reward_max=4.,
                band_extra_cost=32., force_clipped_fraction_gate=.0001,
                actor_critic_action=[114, 453, 7])
