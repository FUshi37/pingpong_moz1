"""Default-off clean joint-quality price over the original v2 reward."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE, verify_source,
)

MODE = 'clean10_dynamics_reference_center_negy_v21'
ORIGINAL_RATE_PER_S = 20.
CANDIDATE_RATE_PER_S = 100.
WARMUP_POLICY_STEPS = 8192
INCOMPATIBLE_FLAGS = (
    'wind_clean_group_priority', 'wind_normalized_group_priority',
    'wind_clean_apex_priority', 'wind_precontact_center', 'wind_precontact_balance',
    'wind_release_precision', 'wind_cycle_release_precision', 'wind_measured_precision',
    'wind_steady_braking', 'wind_confirmed_launch_braking', 'wind_center_tail',
    'wind_calm_release', 'wind_early_recovery', 'wind_mature_precision_reward',
    'wind_calm_return_height', 'wind_impact_normal_precision', 'wind_cubic_recovery',
)


def clean_joint_contract() -> dict:
    """Change only the price of the existing minimum-quality clean credit."""
    return dict(base=BASE_MODE, population='clean only',
        old_rate_per_s=ORIGINAL_RATE_PER_S, new_rate_per_s=CANDIDATE_RATE_PER_S,
        score='unchanged native min(center,apex,motion,angular) with existing margin',
        events='unchanged confirmed hits, interval cap1.2s, recent/live and shelter',
        warmup_policy_steps=WARMUP_POLICY_STEPS,
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE, true_step=SOURCE_STEP,
                    Adam_t=SOURCE_ADAM_T, stage_index=15),
        own_resume='preserve original precision and new credit maturity; physical clock unchanged',
        preserved='all wind rewards, independent center/apex, survival/failure, physics, '
                  'observations, control, PPO, sampling and every original gate;BC0')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCleanJointQualityConfig', [
        ('wind_clean_joint_quality_priority', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_joint_quality_config(cfg: Any, enabled: bool) -> Any:
    """Retain exact historical config identity unless the selector is enabled."""
    if not enabled:
        return cfg
    if any(getattr(cfg, flag, False) for flag in INCOMPATIBLE_FLAGS):
        raise ValueError('Clean joint-quality priority cannot stack a different recipe')
    return _config_type(type(cfg))(**vars(cfg))


def joint_reward_phase(payload: dict, sha256: str) -> int:
    """Authenticate the original full learner or a same-reward continuation."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_clean_joint_quality_priority')
    if any(payload.get('env_cfg', {}).get(flag, False) for flag in INCOMPATIBLE_FLAGS):
        raise ValueError('Clean joint-quality priority cannot stack a different recipe')
    return min(WARMUP_POLICY_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)


def extra_clean_joint_quality(original_credit: Any, *, clean: Any,
                             reward_steps: Any, xp: Any) -> dict:
    """Add only to native nonnegative joint credits [N], retaining their timing."""
    original_credit = xp.asarray(original_credit)
    progress = xp.clip(xp.asarray(reward_steps, dtype=original_credit.dtype)
                       / WARMUP_POLICY_STEPS, 0., 1.)
    warmup = progress**2 * (3. - 2. * progress)
    increase = CANDIDATE_RATE_PER_S / ORIGINAL_RATE_PER_S - 1.
    extra = xp.where(clean, increase * warmup * original_credit, 0.)
    return dict(delta=extra, warmup=warmup)


def add_credit(original_reward: Any, delta: Any, *, xp: Any) -> Any:
    """Preserve every protected reward bit when its extra component is zero."""
    return xp.where(delta != 0., original_reward + delta, original_reward)
