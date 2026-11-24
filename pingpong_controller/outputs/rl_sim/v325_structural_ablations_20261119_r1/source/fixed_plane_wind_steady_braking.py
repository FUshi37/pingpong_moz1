"""Reward feedback for late racket-Z braking during steady recovery."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source as verify_s16_source,
)


MODE = 'clean10_dynamics_reference_center_negy_v11'


def steady_braking_contract() -> dict:
    return dict(
        base=BASE_MODE, rate_per_s=dict(clean=2., wind=4.),
        excess_deadband=.25, braking_cost_cap=4., confirmed_hits_min=12,
        scope='live detached recovery outside the existing imminent-contact phase',
        source_metric='dynamics/braking_cost, existing two-sided 80 ms preview',
        warmup_policy_steps=8192,
        resume_phase='reuse authenticated elapsed policy steps, capped at 8192',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
    )


@lru_cache(maxsize=16)
def _reward_config_type(parent: type) -> type:
    return make_dataclass('WindSteadyBrakingConfig', [
        ('wind_steady_braking', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_steady_braking_config(cfg: Any, enabled: bool) -> Any:
    """Add a reward-only flag to the selected versioned recipe."""
    return _reward_config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def active_recovery(*, confirmed_hits, recent, contact, precontact, terminated):
    return (confirmed_hits >= 12) & recent & ~contact & ~precontact & ~terminated


def steady_braking_cost(braking_cost, *, windy, active, gain, reward_steps,
                        dt_s: float, xp: Any):
    """Price excessive outward speed per unit time in either population."""
    progress = xp.clip(reward_steps / 8192., 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)
    excess = xp.clip(braking_cost, .25, 4.) - .25
    rate = xp.where(windy, 4., 2.)
    return xp.where(active, -dt_s * rate * excess * gain * warmup, 0.)


def verify_source(payload: dict, sha256: str) -> int:
    """Authenticate the full learner and recover its reward phase on resume."""
    verify_s16_source(payload, sha256, mode=MODE, config_flag='wind_steady_braking')
    return min(8192, (payload['global_update'] - SOURCE_UPDATE) * 128)
