"""Default-off normalized task priority and reward-only continuation clocks."""
from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

MODE = 'clean10_dynamics_reference_center_negy_v5'
BIRTH_MODE = 'clean10_dynamics_reference_center_negy_v3'
BIRTH_SHA256 = '5ae8d4393375ab7398f0ff4996e16a9eac4b88117520e093d11da1eb28ada6a3'
BIRTH_UPDATE = 13328
BIRTH_STEP = 3356622848
BIRTH_ADAM_T = 929071
WARMUP_POLICY_STEPS = 8192


def normalized_priority_contract() -> dict:
    """Retain9:1 relative importance without multiplying clean value targets."""
    return dict(revision='normalized_clean_priority_v1', relative_clean_priority=9.,
        clean_multiplier=1., wind_multiplier='1/(1+8*smoothstep(reward_steps/8192))',
        interpretation='raw9:1 weighting divided by one positive common factor',
        scope='whole reward, including negative failures; component ratios/signs preserved',
        warmup_policy_steps=WARMUP_POLICY_STEPS,
        source=dict(sha256=BIRTH_SHA256, update=BIRTH_UPDATE, true_step=BIRTH_STEP,
                    Adam_t=BIRTH_ADAM_T),
        resume='own-recipe source progress offsets reward clocks only; physical state untouched',
        preserved='sampling, observations, physics, control, PPO, all curriculum gates')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindNormalizedPriorityConfig', [
        ('wind_normalized_group_priority', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_normalized_config(cfg: Any, enabled: bool) -> Any:
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def normalized_reward(reward: Any, *, clean: Any, total_env_steps: Any,
                      xp: Any) -> tuple[Any, dict]:
    """Return weighted rewards [N]; clean values remain bitwise unchanged."""
    reward = xp.asarray(reward)
    progress = xp.clip(xp.asarray(total_env_steps, dtype=reward.dtype) / 8192., 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    relative = 1. + 8. * warmup
    multiplier = xp.where(clean, xp.ones_like(reward), 1. / relative)
    weighted = xp.where(clean, reward, reward / relative)
    return weighted, {
        'reward/normalized_priority_delta': weighted - reward,
        'normalized_priority/multiplier': multiplier,
        'normalized_priority/relative_clean_priority': relative,
        'normalized_priority/warmup': warmup,
    }


def source_resume_steps(payload: dict, sha256: str) -> int:
    """Authenticate the reward phase without changing any learner leaf."""
    args = payload.get('args', {})
    mode = args.get('wind_training_mix')
    update, step = payload.get('global_update'), payload.get('step')
    stage = payload.get('stage_index')
    opt = payload.get('opt')
    try:
        adam_t = int(opt['t'] if isinstance(opt, dict) else opt.t)
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError('Missing reward source optimizer accounting') from error
    if (args.get('curriculum_profile') != 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v325'
            or type(update) is not int or update < BIRTH_UPDATE
            or type(step) is not int or type(stage) is not int or not 14 <= stage < 32):
        raise ValueError('Normalized reward requires its reviewed V325 continuation')
    elapsed = update - BIRTH_UPDATE
    if step != BIRTH_STEP + elapsed * 131072 or adam_t != BIRTH_ADAM_T + elapsed * 32:
        raise ValueError('Reward source update/step/Adam counters disagree')
    if mode == BIRTH_MODE:
        if sha256 != BIRTH_SHA256 or elapsed:
            raise ValueError('Unreviewed normalized-reward birth source')
        return 0
    if (mode != MODE
            or payload.get('env_cfg', {}).get('wind_normalized_group_priority') is not True):
        raise ValueError('Reward progress can resume only from this exact recipe')
    return min(elapsed * 128, WARMUP_POLICY_STEPS)


def bind_reward_resume(factory: Any, policy_steps: int) -> Any:
    """Supply static reward context to every later domain constructor."""
    if not 0 <= policy_steps <= WARMUP_POLICY_STEPS:
        raise ValueError('Invalid reward continuation clock')
    if not policy_steps:
        return factory

    def build(*args, **kwargs):
        return factory(*args, **kwargs, reward_resume_policy_steps=policy_steps)

    return build
