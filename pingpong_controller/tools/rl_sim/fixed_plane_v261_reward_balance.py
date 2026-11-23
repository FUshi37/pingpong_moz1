"""V261: sharpen steady height and align motion/center reward with S7 gates."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from fixed_plane_v258_gaussian_force import (
    GaussianBoundedForceConfig, STAGE_NAMES as PARENT_STAGES,
    validate_config as validate_parent,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v261_reward_balance'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v258'
STAGE_NAMES = tuple(s.replace('v258_', 'v261_', 1) for s in PARENT_STAGES)
START_STAGE = STAGE_NAMES[6]
SOURCE = dict(update=824, step=1612972032, adam_t=402724,
    sha256='c834edad973fee7d5ebdb7c40097da2161540ecee7741157726d5c36cc4b9ea3')
CHANGES = {
    'fixed_plane_v191_absolute_height_multiplier': 2.,
    'fixed_plane_v191_margin_weight': 12.,
    'fixed_plane_v158_late_squared_vxy_weight': 4.5,
    'fixed_plane_v159_mean_center_penalty_weight': 2.25,
}
PARENT_VALUES = dict(zip(CHANGES, (1., 6., 3., 1.5)))


@dataclass(frozen=True)
class RewardBalanceConfig(GaussianBoundedForceConfig):
    """Version marker and the existing optional absolute-height coefficient."""
    v261_enabled: bool = True
    fixed_plane_v191_absolute_height_multiplier: float = 1.


def validate_config(cfg: RewardBalanceConfig) -> None:
    """Retain every parent contract; accept one complete reward setting only."""
    if not cfg.v261_enabled:
        raise ValueError('V261 reward schema requires its explicit version marker')
    current = {key:getattr(cfg, key) for key in CHANGES}
    if current not in (PARENT_VALUES, CHANGES):
        raise ValueError('V261 rejects partial or mixed reward coefficients')
    validate_parent(replace(cfg, **PARENT_VALUES))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all11 lessons and every gate; apply reward repair at S7 onward."""
    if tuple(s.name for s in source) != PARENT_STAGES:
        raise ValueError('V261 requires the complete V258 course')
    result = []
    for index, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        cfg = RewardBalanceConfig(**vars(stage.cfg))
        if index >= 6:
            cfg = replace(cfg, **CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V261 reward-only repair from stopped V258S7u824. Absolute signed '
            'height multiplier2, bilateral margin12, counted late XY-square4.5, '
            'mean contact-center2.25. Band event30 and adjacent multiplier3 '
            'retain V258 values. No new near-Y cost. Preserve all11 domains, '
            'gates,32-pass hold, independent/retention validation,72/398/7 '
            'actor/critic/action and full actor/critic/Adam.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *,
        requested_start_stage: str, resume_curriculum_state: bool,
        reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin the stopped full source or require exact own-stage causal history."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))
                != (72,398,7)):
        raise ValueError('V261 requires full72/398/7 actor/critic/Adam without reset')
    parent = payload.get('args', {}).get('curriculum_profile')
    if parent == PARENT_PROFILE:
        if (requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_index') != 7
                or payload.get('stage_name') != PARENT_STAGES[6]
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('step') != SOURCE['step']
                or int(payload['opt'].t) != SOURCE['adam_t']):
            raise ValueError('V261 pins stopped V258S7u824 with fresh reward history')
    elif parent == PROFILE:
        index = payload.get('stage_index', 0)
        if not 1 <= index <= 11 or payload.get('stage_name') != STAGE_NAMES[index-1]:
            raise ValueError('V261 own stage identity differs')
        if resume_curriculum_state:
            if requested_start_stage != payload['stage_name']:
                raise ValueError('V261 continuation requires exact causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V261 rejects another source profile')


def validate_training_contract(settings: Any, source_args: Mapping[str, Any]) -> None:
    """Keep all numerical PPO/BC/batch and graduation settings unchanged."""
    keys = ('n_envs','n_steps','minibatch_size','update_epochs','learning_rate',
        'gamma','gae_lambda','time_limit_bootstrap','clip_range','target_kl',
        'ent_coef','min_log_std','vf_coef','max_grad_norm','hidden_dim',
        'asymmetric_critic','critic_command_history_steps',
        'counterfactual_supervision_coef','counterfactual_vxy_weight_mode',
        'conditional_reset_done','right_arm_pd_profile','convergence_window',
        'min_stage_updates','stage_metric_warmup_updates','advance_eval_n_envs',
        'advance_eval_steps','advance_eval_retry_updates')
    changed = [k for k in keys if getattr(settings,k,None) != source_args.get(k)]
    if changed:
        raise ValueError('V261 changed non-reward training settings: '+','.join(changed))
    if (settings.n_envs != 1024 or settings.n_steps != 128
            or settings.max_stages != 11 or settings.allow_unconverged_advance
            or settings.max_stage_updates not in (-1,64,128)):
        raise ValueError('V261 retains1024x128, all11 stages and genuine graduation')
