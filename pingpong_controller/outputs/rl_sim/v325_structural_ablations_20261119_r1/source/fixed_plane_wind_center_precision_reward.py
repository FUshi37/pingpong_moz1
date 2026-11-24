"""Default-off positive centering credit; no change to any graduation gate."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any


def center_reward_contract() -> dict:
    """Keep mean and second-moment feedback explicit and time normalized."""
    return dict(
        revision='center_mean_rms_positive_v1',
        rate_per_s=dict(clean=80., wind=40.),
        score='0.5*(max(1-d/(2*mean_limit),0)+max(1-(d/(2*rms_limit))**2,0))',
        mean_limit_m={'325': .0055, '326': .0065},
        rms_limit_m={'325': .007, '326': .008},
        timing='confirmed physical hits; actual interval capped1.2s; recent and live',
        scope='original five-hit shelter and one-second ramp; no early-launch change',
        warmup=dict(actual_updates=64, policy_steps=8192,
                    clock='persistent total_env_steps', shape='smoothstep'),
        bounded_positive=True,
        preserved='all previous rewards, gates, observations, control, physics, PPO and sampling',
    )


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCenterPrecisionRewardConfig', [
        ('wind_center_precision_reward', bool, field(default=True)),
        ('wind_center_precision_version', int, field(default=325)),
    ], bases=(parent,), frozen=True)


def with_center_reward_config(cfg: Any, enabled: bool, version: int) -> Any:
    """Add only reward flags; historical modes retain their exact config type."""
    if not enabled:
        return cfg
    if version not in (325, 326):
        raise ValueError('Centering reward is scoped to V325/V326')
    return _config_type(type(cfg))(**vars(cfg), wind_center_precision_reward=True,
                                  wind_center_precision_version=version)


def center_reward_terms(metrics: dict, *, clean: Any, gain: Any,
                        total_env_steps: Any, version: int, xp: Any) -> dict:
    """Return [N] credit and diagnostics using native confirmed-hit quantities.

    Below the two score cutoffs, expected credit prices E[d] and E[d**2]
    separately. There is no new negative tail, fixed per-hit income, policy
    input, random draw, physics state, or gate reinterpretation.
    """
    if version not in (325, 326):
        raise ValueError('Centering reward is scoped to V325/V326')
    mean_limit_m = .0055 if version == 325 else .0065
    rms_limit_m = .007 if version == 325 else .008
    hit = metrics['hit_event_count']
    center_m = metrics['hit_contact_center_dist_sum'] / xp.maximum(hit, 1.)
    linear = xp.maximum(1. - center_m / (2. * mean_limit_m), 0.)
    quadratic = xp.maximum(1. - (center_m / (2. * rms_limit_m)) ** 2, 0.)
    progress = xp.clip(total_env_steps / 8192., 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    live = (metrics['terminated'] < .5) & (metrics['dynamics/recent'] > .5)
    events = gain * xp.where(live, hit, 0.)
    elapsed_s = xp.clip(metrics['dynamics/hit_interval_s'], 0., 1.2)
    rate_per_s = xp.where(clean, 80., 40.)
    credit = events * elapsed_s * rate_per_s * warmup * .5 * (linear + quadratic)
    return {
        'reward/center_mean_rms_credit': credit,
        'center_precision/warmup': warmup,
        'center_precision/confirmed_events': events,
        'center_precision/linear_score_sum': events * linear,
        'center_precision/square_score_sum': events * quadratic,
    }


def clean_apex_contract() -> dict:
    return dict(revision='clean_independent_apex_positive_v1', rate_per_s=40.,
        target_m=1.425, sigma_m=.008, interval_cap_s=1.2,
        events='existing clean native apex events used by graduation',
        scope='clean only; original shelter/ramp and same64-update reward warmup',
        independence='credit does not disappear when center/motion is the joint-score bottleneck',
        preserved='wind objective and every old reward/gate')


@lru_cache(maxsize=16)
def _clean_apex_config_type(parent: type) -> type:
    return make_dataclass('WindCleanApexRewardConfig', [
        ('wind_clean_apex_preservation', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_clean_apex_config(cfg: Any, enabled: bool) -> Any:
    return _clean_apex_config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def clean_apex_reward_terms(metrics: dict, *, clean: Any, gain: Any,
                            total_env_steps: Any, xp: Any,
                            rate_per_s: float = 40.) -> dict:
    """Pay the clean gate's own apex estimate independently of other scores."""
    if rate_per_s not in (40., 120.):
        raise ValueError('Use the registered clean apex reward rates')
    event = metrics['wind_height/apex_events']
    apex_m = metrics['wind_height/apex_sum_m'] / xp.maximum(event, 1.)
    score = xp.exp(-.5 * ((apex_m - 1.425) / .008) ** 2)
    progress = xp.clip(total_env_steps / 8192., 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    live = (metrics['terminated'] < .5) & (metrics['dynamics/recent'] > .5)
    elapsed_s = xp.clip(metrics['dynamics/hit_interval_s'], 0., 1.2)
    credit = clean * gain * warmup * xp.where(live, rate_per_s * elapsed_s * event * score, 0.)
    return {'reward/clean_apex_preservation_credit': credit}


def clean_apex_priority_contract() -> dict:
    """Override only the independent clean-height reward amplitude."""
    return {**clean_apex_contract(), 'revision': 'clean_independent_apex_positive_v2',
            'rate_per_s': 120., 'previous_rate_per_s': 40.,
            'preserved': 'all other rewards and every nonreward field/gate'}


@lru_cache(maxsize=16)
def _clean_apex_priority_type(parent: type) -> type:
    return make_dataclass('WindCleanApexPriorityConfig', [
        ('wind_clean_apex_priority', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_clean_apex_priority_config(cfg: Any, enabled: bool) -> Any:
    return _clean_apex_priority_type(type(cfg))(**vars(cfg)) if enabled else cfg
