"""Opt-in backoff of ineffective reward amplification, with robust XY tails."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
from fixed_plane_v298_practical_dr_completion import PracticalDRConfig
PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v299_safe_reward_backoff'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v298_practical_dr_completion'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v299_reward_backoff_enabled'})

@dataclass(frozen=True)
class RewardBackoffConfig(PracticalDRConfig):
    v299_reward_backoff_enabled: bool = False

def validate_config(cfg: RewardBackoffConfig) -> None:
    from fixed_plane_v297_feasible_reward_gain import validate_config as original_prices
    original_prices(cfg)
    if cfg.v299_reward_backoff_enabled is not True or cfg.v297_feasible_reward_multiplier != 1.0:
        raise ValueError('V299 explicitly removes only the failed16x reward gain')
    if cfg.v296_action_prepare_weight_per_s > 0.0:
        if cfg.v296_action_prepare_weight_per_s != 512.0 or cfg.v298_early_xy_huber_transition_m_s != 0.05:
            raise ValueError('V299 preserves512/s and the0.05m/s robust XY tail')
    elif cfg.v298_early_xy_huber_transition_m_s != 0.0:
        raise ValueError('V299 nominal lessons retain zero reward-tail mode')

def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V299 requires all17 V298 lessons')
    out = []
    for s in source:
        cfg = RewardBackoffConfig(**vars(s.cfg) | {'v297_feasible_reward_multiplier': 1.0}, v299_reward_backoff_enabled=True)
        validate_config(cfg)
        out.append(replace(s, cfg=cfg, notes='V299 reward-only backoff: remove ineffective16x amplification; retain original512/s startup-label and8 early-XY price,including V2980.05m/s Huber tail. No model/critic/Adam/history rollback or reset. All17 V298 gates/domains,strict nominal,72/393/7,control/physics/DR and all1024x128 numeric PPO/BC/KL/loss guards unchanged. Native rolling32/hold16,independent288x1200 and nominal validation;all32 minibatches,no update cap or forced advancement.'))
    return out

def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v298_practical_dr_completion import validate_training_contract as previous
    previous(settings)

def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str, resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool, allow_obs_dim_migration: bool) -> None:
    if reset_optimizer or reset_critic or allow_obs_dim_migration or (payload.get('opt') is None) or (not {'pi', 'v'}.issubset(payload.get('params', {}))) or (tuple((payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))) != (72, 393, 7)) or (payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V299 requires full uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        total = int(payload.get('global_update', 0)) - 3101
        tail = payload.get('stage_history_tail')
        if not resume_curriculum_state or requested_start_stage != START_STAGE or payload.get('stage_name') != START_STAGE or (payload.get('stage_index') != 12) or (total < 1) or (payload.get('step') != 3579314176 + total * 131072) or (int(payload['opt'].t) != 883326 + total * 32) or (not isinstance(tail, list)) or (not tail) or (int(tail[-1]['stage_update']) != int(payload['stage_update'])) or (tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V299 requires latest full V298 S12 state and exact history/Adam')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V299 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V299 rejects another source or from-scratch launch')
