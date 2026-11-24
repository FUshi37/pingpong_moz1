"""Opt-in4.5 relative task balance for the existing causal preview reward."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

MODE = 'clean10_dynamics_reference_center_negy_v7'
BIRTH_MODE = 'clean10_dynamics_reference_center_negy_v2'
BIRTH_SHA256 = '64b6aed4c5177a0fb1d97d8bf946a88c258399aca96708d01b902e77337012d9'
BIRTH_UPDATE = 14500
BIRTH_STEP = 3510239232
BIRTH_ADAM_T = 966575
PRIORITY = 4.5


def balance_contract() -> dict:
    """Keep physical thresholds separate from conditional reward importance."""
    return dict(revision='balanced_causal_precontact_v1', relative_clean_priority=PRIORITY,
        clean_episode_fraction=.1, center_rates_per_s=dict(clean=80., wind=40.),
        nominal_center_importance='.1*80 == .9*40/4.5 == 8',
        clean_multiplier=1., wind_multiplier='1/(1+3.5*smoothstep(reward_steps/8192))',
        scope='whole reward after preview, including negative failures; component ratios/signs retained',
        warmup_policy_steps=8192, base_reward_maturity_policy_steps=8192,
        source=dict(sha256=BIRTH_SHA256, update=BIRTH_UPDATE, true_step=BIRTH_STEP,
                    Adam_t=BIRTH_ADAM_T, stage_index=15),
        preserved='sampling, all gate definitions/numbers/windows/hold, PPO, observations, control, physics')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindBalancedPrecontactConfig', [
        ('wind_precontact_balance', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_balance_config(cfg: Any, enabled: bool) -> Any:
    """Add only one reward flag to the explicitly selected preview recipe."""
    if not enabled:
        return cfg
    if (not getattr(cfg, 'wind_precontact_center', False)
            or getattr(cfg, 'wind_normalized_group_priority', False)
            or getattr(cfg, 'wind_clean_group_priority', False)):
        raise ValueError('Balanced preview requires its distinct reviewed reward recipe')
    return _config_type(type(cfg))(**vars(cfg))


def balanced_reward(reward: Any, *, clean: Any, reward_steps: Any,
                    xp: Any) -> tuple[Any, dict]:
    """Scale complete [N] windy rewards; clean values remain bitwise exact."""
    reward = xp.asarray(reward)
    progress = xp.clip(xp.asarray(reward_steps, dtype=reward.dtype)/8192., 0., 1.)
    warmup = progress**2*(3.-2.*progress)
    relative = 1.+(PRIORITY-1.)*warmup
    weighted = xp.where(clean, reward, reward/relative)
    return weighted, {
        'reward/precontact_balance_delta': weighted-reward,
        'precontact_balance/multiplier': xp.where(clean, xp.ones_like(reward), 1./relative),
        'precontact_balance/relative_clean_priority': relative,
        'precontact_balance/warmup': warmup,
    }


def source_resume_context(payload: dict, sha256: str) -> dict:
    """Admit the reviewed S16 learner or its exact reward continuation."""
    args = payload.get('args', {})
    update, step, stage = (payload.get(k) for k in ('global_update', 'step', 'stage_index'))
    opt = payload.get('opt')
    try:
        adam = int(opt['t'] if isinstance(opt, dict) else opt.t)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing balanced-preview optimizer accounting') from error
    if (args.get('curriculum_profile') != 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v325'
            or type(update) is not int or update < BIRTH_UPDATE
            or type(step) is not int or type(stage) is not int or not 15 <= stage < 32):
        raise ValueError('Balanced preview requires its reviewed V325 S16 continuation')
    elapsed = update-BIRTH_UPDATE
    if step != BIRTH_STEP+elapsed*131072 or adam != BIRTH_ADAM_T+elapsed*32:
        raise ValueError('Balanced-preview update/step/Adam counters disagree')
    if args.get('wind_training_mix') == BIRTH_MODE:
        if (elapsed or sha256 != BIRTH_SHA256 or stage != 15
                or bool(payload.get('stage_converged', False))):
            raise ValueError('Unreviewed balanced-preview birth source')
        return dict(birth_update=BIRTH_UPDATE, policy_steps=0)
    cfg = payload.get('env_cfg', {})
    if (args.get('wind_training_mix') != MODE
            or cfg.get('wind_precontact_center') is not True
            or cfg.get('wind_precontact_balance') is not True
            or payload.get('precontact_reward_birth_update', BIRTH_UPDATE) != BIRTH_UPDATE):
        raise ValueError('Only this exact balanced-preview recipe can continue its phase')
    return dict(birth_update=BIRTH_UPDATE, policy_steps=min(elapsed*128, 8192))
