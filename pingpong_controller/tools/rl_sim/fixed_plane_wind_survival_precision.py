"""Opt-in time-normalized survival and independent contact/output precision credit."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any


def survival_precision_contract() -> dict:
    return dict(revision='wind_survival_precision_v1', survival_rate_per_s=50.,
        previous_survival_rate_per_s=2., confirmed_quality_rate_per_s=16.,
        previous_confirmed_quality_rate_per_s=4., outgoing_quality_rate_per_s=20.,
        release_interval_cap_s=.8, confirmed_interval_cap_s=1.2,
        output_quality='minimum Cauchy score of independent XY and Z theoretical velocity errors',
        output_sigma_m_s=[.035,.035,.030],
        contact_cost_per_hit=dict(clean=1.,wind=1.), previous_contact_cost=dict(clean=.5,wind=.25),
        precision='retain all existing velocity, measured-apex, center, motion and safety costs',
        timing='positive release credit times actual elapsed physical-release time; no per-edge bonus',
        scope='wind after five-hit shelter/ramp; clean gets only extra confirmed-contact precision loss',
        preserved='source-A optimizer; full learner; observations, control, physics, reset,32 courses and every gate',
        evidence='reward algebra and finite first training updates; no monotone-learning guarantee')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindSurvivalPrecisionConfig',
        [('wind_survival_precision',bool,field(default=True))],bases=(parent,),frozen=True)


def with_survival_precision_config(cfg: Any, enabled: bool):
    return _config_type(type(cfg))(**vars(cfg),wind_survival_precision=True) if enabled else cfg


def reward_terms(metrics: dict, *, wind_gain: Any, clean_gain: Any, dt_s: float,
                 release_elapsed_s: Any, xp: Any) -> dict:
    """Supplement existing feedback; no state/action changes or relaxed quality limits.

    All inputs and outputs are [N]. Release intervals come from the physical
    release clock, independently of confirmation delay. Credits are bounded
    per unit time, so increasing contact frequency cannot farm a fixed bonus.
    """
    valid=(metrics['terminated']<.5)&(metrics['dynamics/recent']>.5)
    hit=metrics['hit_event_count']
    count=xp.maximum(hit,1.)
    center=metrics['hit_contact_center_dist_sum']/count/.0055
    huber=.5*xp.minimum(center,1.)**2+xp.maximum(center-1.,0.)
    center_loss=xp.minimum(huber,4.)
    racket_xy_sq=metrics['hit_racket_vxy_sq_sum']/count
    steady=xp.maximum(metrics['rapid_steady_steady_events'],1.)
    omega_sq=metrics['rapid_steady_steady_racket_omega_sq_sum']/steady
    contact_quality=xp.exp(-.5*(center**2+racket_xy_sq/.055**2+omega_sq/.28**2))
    margin=xp.clip(metrics['dynamics/z_margin_m']/.025,0.,1.)
    interval=xp.clip(metrics['dynamics/hit_interval_s'],0.,1.2)
    ex=metrics['dynamics/outgoing_error_vx_m_s']/.035
    ey=metrics['dynamics/outgoing_error_vy_m_s']/.035
    ez=metrics['dynamics/outgoing_error_vz_m_s']/.030
    output_quality=xp.minimum(1./(1.+ex**2+ey**2),1./(1.+ez**2))
    released=metrics['dynamics/release_event']*(metrics['dynamics/reference_valid']>.5)
    release_time=xp.clip(release_elapsed_s,0.,.8)
    return {
        'reward/continuation_survival':wind_gain*xp.where(valid,48.*dt_s*margin,0.),
        'reward/continuation_hit_quality':wind_gain*xp.where(valid,
            12.*interval*hit*contact_quality*margin,0.),
        'reward/continuation_outgoing_quality':wind_gain*xp.where(valid,
            20.*release_time*released*output_quality,0.),
        'reward/continuation_contact_precision':-(.75*wind_gain+.5*clean_gain)*xp.where(
            valid,hit*center_loss,0.),
    }
