"""Opt-in clean reward routing and independent windy height/motion feedback."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any


def precision_contract() -> dict:
    """Keep physical gates fixed; put useful reward slopes inside those gates."""
    return dict(revision='wind_gate_precision_negy_v1',
        clean_recovery_route='exclude clean mode',
        clean_reward='retain upstream original reward plus existing clean guards',
        observed_apex=dict(target_m=1.425, sigma_m=.010, extra_huber_weight=8.,
                           extra_band_miss_cost=16., height='actual measured flight peak'),
        outgoing_z=dict(extra_weight=2., sigma_m_s=.025, half_squared_cap=8.),
        racket_xy=dict(target_m_s=.045, excess_sigma_m_s=.020,
                       cost_per_s=8., huber_cap=6., interval_cap_s=1.2),
        scope='wind after five confirmed hits and original one-second ramp',
        preserved='survival credit; theoretical normal/3D output; clean10; BC0; PPO; all gates',
        unchanged='physics,control,observations,resets,32 stages,4m/s,Y<=0,20deg')


def positive_contract() -> dict:
    """Bound reward units and add precision income instead of extra failure incentives."""
    return dict(revision='wind_gate_positive_precision_negy_v2',
        clean_recovery_route='exclude clean mode', clean_native_scale=.01,
        clean_native_cap_per_step=.10, physical_failures='unscaled native penalties',
        clean_survival_rate_per_s=50., clean_quality_rate_per_s=20.,
        observed_apex=dict(target_m=1.425,sigma_m=.008,rate_per_s=40.,
            elapsed='twice actual release-to-apex age, capped1.2s; total credit <=80/s'),
        outgoing_z=dict(sigma_m_s=.025,rate_per_s=10.,release_interval_cap_s=.8),
        racket_xy=dict(target_m_s=.045,excess_sigma_m_s=.020,rate_per_s=20.,interval_cap_s=1.2),
        native_objective='retain every original clean component with common scaling; clip only extreme live tails',
        scope='same five-hit shelter and one-second ramp; no new negative precision costs',
        preserved='old direct velocity/apex/center/motion/safety costs,wind survival50/s,theoretical references,PPO,gates')


@lru_cache(maxsize=8)
def _config_type(parent: type) -> type:
    return make_dataclass('WindGatePrecisionConfig',
        [('wind_gate_precision', bool, field(default=True)),
         ('wind_gate_precision_version', int, field(default=325))],
        bases=(parent,), frozen=True)


@lru_cache(maxsize=8)
def _positive_config_type(parent: type) -> type:
    return make_dataclass('WindPositivePrecisionConfig',
        [('wind_gate_precision_positive',bool,field(default=True))], bases=(parent,),frozen=True)


def with_precision_config(cfg: Any, enabled: bool, version: int, *, positive: bool = False):
    """Add no fields to historical modes or to the unchanged early curriculum."""
    if not enabled:
        return cfg
    if version not in (325, 326):
        raise ValueError('Gate precision is scoped to V325/V326')
    result = _config_type(type(cfg))(**vars(cfg), wind_gate_precision=True,
                                    wind_gate_precision_version=version)
    return _positive_config_type(type(result))(**vars(result)) if positive else result


def recovery_gain(gain: Any, wind_mode: Any, *, enabled: bool):
    """The earlier bounded recovery layer must not cancel clean native reward."""
    return gain * (wind_mode > 0) if enabled else gain


def normalized_clean_reward(native_reward: Any, terminated: Any, *, clean_gain: Any, xp: Any):
    """Keep clean task-component order; bound live scales, retain real failure costs."""
    normalized = xp.where(terminated > .5,native_reward,
                          xp.clip(.01*native_reward,-.10,.10))
    return xp.where(clean_gain <= 0.,native_reward,xp.where(clean_gain >= 1.,normalized,
        (1.-clean_gain)*native_reward+clean_gain*normalized))


def group_reward_metrics(reward: Any, windy: Any, metrics: dict, *, xp: Any) -> dict:
    """Log both populations even when M13 has no optional recovery recipe."""
    cancellation = metrics.get('reward/recovery_native_replaced',xp.zeros_like(reward))
    return {
        'objective_group/clean_sample':(~windy).astype(xp.float32),
        'objective_group/wind_sample':windy.astype(xp.float32),
        'objective_group/clean_reward_sum':xp.where(windy,0.,reward),
        'objective_group/wind_reward_sum':xp.where(windy,reward,0.),
        'objective_group/clean_native_cancellation':xp.where(windy,0.,cancellation),
    }


def reward_terms(metrics: dict, height: dict, *, wind_gain: Any, height_gain: Any,
                 version: int, xp: Any, positive: bool = False,
                 release_elapsed_s: Any = None, apex_elapsed_s: Any = None,
                 clean_gain: Any = None, dt_s: float = .005) -> dict:
    """Return [N] event losses; no new positive rewards or ball-XY penalty.

    Height uses the actual flight tracker and its saved release gain. The
    release loss uses physical outgoing Z velocity. Racket motion is charged
    at confirmed impacts, weighted by elapsed hit time instead of hit count.
    """
    if version not in (325, 326):
        raise ValueError('Gate precision is scoped to V325/V326')
    valid = metrics['terminated'] < .5
    hit = metrics['hit_event_count']
    speed = xp.sqrt(xp.maximum(metrics['hit_racket_vxy_sq_sum'], 0.) /
                    xp.maximum(hit, 1.))
    excess = xp.maximum(speed - .045, 0.) / .020
    huber = lambda x: .5 * xp.minimum(xp.abs(x), 1.) ** 2 + xp.maximum(xp.abs(x) - 1., 0.)
    interval = xp.clip(metrics['dynamics/hit_interval_s'], 0., 1.2)
    recent = valid & (metrics['dynamics/recent'] > .5)
    if positive:
        if release_elapsed_s is None or apex_elapsed_s is None or clean_gain is None:
            raise ValueError('Positive precision requires physical clocks and clean population gain')
        from fixed_plane_wind_objectives import quality_limits
        limits = quality_limits(version)
        margin = xp.clip(metrics['dynamics/z_margin_m']/.025,0.,1.)
        motion_score = xp.exp(-.5*excess**2)
        z_score = xp.exp(-.5*xp.square(metrics['dynamics/outgoing_error_vz_m_s']/.025))
        apex_score = xp.exp(-.5*xp.square((height['apex_m']-1.425)/.008))
        released = metrics['dynamics/release_event']*(metrics['dynamics/reference_valid']>.5)
        steady = xp.maximum(metrics['rapid_steady_steady_events'],1.)
        native_apex = metrics['rapid_steady_steady_apex_sum_z_m']/steady
        center = metrics['hit_contact_center_dist_sum']/xp.maximum(hit,1.)
        omega_sq = metrics['rapid_steady_steady_racket_omega_sq_sum']/steady
        clean_quality = xp.minimum(motion_score,xp.minimum(
            xp.exp(-.5*xp.square(center/limits['center_m'])),xp.minimum(
                xp.exp(-.5*xp.square((native_apex-1.425)/.008)),
                xp.exp(-.5*omega_sq/limits['racket_omega_rms_rad_s']**2))))
        return {
            'reward/gate_observed_apex':height_gain*xp.where(valid,
                40.*xp.clip(2.*apex_elapsed_s,0.,1.2)*height['apex_event']*apex_score,0.),
            'reward/gate_outgoing_z':wind_gain*xp.where(valid,
                10.*xp.clip(release_elapsed_s,0.,.8)*released*z_score,0.),
            'reward/gate_racket_xy':wind_gain*xp.where(recent,
                20.*interval*hit*motion_score*margin,0.),
            'reward/gate_clean_survival':clean_gain*xp.where(recent,50.*dt_s*margin,0.),
            'reward/gate_clean_quality':clean_gain*xp.where(recent,
                20.*interval*hit*clean_quality*margin,0.),
        }
    motion = -8. * interval * hit * xp.minimum(huber(excess), 6.)
    apex_loss = 8. * height['apex_event'] * xp.minimum(
        huber((height['apex_m'] - 1.425) / .010), 8.)
    band_loss = 16. * xp.maximum(height['band_event'] - height['in_band'], 0.)
    released = metrics['dynamics/release_event'] * (metrics['dynamics/reference_valid'] > .5)
    z_loss = 2. * released * xp.minimum(.5 * xp.square(
        metrics['dynamics/outgoing_error_vz_m_s'] / .025), 8.)
    return {
        'reward/gate_observed_apex': -height_gain * xp.where(valid, apex_loss + band_loss, 0.),
        'reward/gate_outgoing_z': -wind_gain * xp.where(valid, z_loss, 0.),
        'reward/gate_racket_xy': wind_gain * xp.where(recent, motion, 0.),
    }
