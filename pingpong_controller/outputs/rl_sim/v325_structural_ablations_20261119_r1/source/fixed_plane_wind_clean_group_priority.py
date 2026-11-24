"""Optional reward importance for rare clean episodes; no sampling change."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any


def group_priority_contract() -> dict:
    """Describe one conditional reward multiplier, not an optimizer change."""
    return dict(revision='clean_reward_importance_v1', clean_multiplier=9.,
        wind_multiplier=1., nominal_reset_probability=dict(clean=.1, wind=.9),
        interpretation='relative importance9 compensates nominal rare-clean prior; '
                       'actual step fractions are logged; equal policy-gradient influence is not assumed',
        scope='all clean reward including launch, survival, precision, motion and failure',
        warmup=dict(updates=64, policy_steps=8192,
                    formula='1+8*smoothstep(total_env_steps/8192)'),
        within_clean='all existing reward component ratios and signs preserved',
        preserved='wind reward, sampling, physics, control, observations, PPO and all gates')


@lru_cache(maxsize=16)
def _priority_config_type(parent: type) -> type:
    return make_dataclass('WindCleanGroupPriorityConfig', [
        ('wind_clean_group_priority', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_group_priority_config(cfg: Any, enabled: bool) -> Any:
    """Keep the exact historical object and config type when disabled."""
    return _priority_config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def priority_reward(reward: Any, *, clean: Any, total_env_steps: Any,
                    xp: Any) -> tuple[Any, dict]:
    """Scale whole clean rewards [N], preserving wind values bit for bit.

    The native persistent clock survives episode resets. No new state,
    observation, random draw, event timing, live mask or hit bonus is added.
    Negative failure rewards retain their sign and relative component price.
    """
    reward = xp.asarray(reward)
    progress = xp.clip(xp.asarray(total_env_steps, dtype=reward.dtype) / 8192., 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    multiplier = xp.where(clean, 1. + 8. * warmup, xp.ones_like(reward))
    weighted = xp.where(clean, reward * multiplier, reward)
    return weighted, {
        'reward/clean_group_priority_delta': weighted - reward,
        'clean_priority/multiplier': multiplier,
        'clean_priority/warmup': warmup,
    }
