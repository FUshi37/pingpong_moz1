"""V231 transfers passed nominal lessons and changes only recovery rewards."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v226_stability_recovery import StabilityRecoveryConfig, STABILITY_LIMITS
from fixed_plane_v229_noninferiority import PARENT_PROFILE, PARENT_SHA256, REPLAY_SHA256, validate_anchor_contract

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v231'
START_STAGE = 'fplc_v231_s06_near_y_005mm_bridge'
RETENTION_STAGE = 'fplc_v225_s01_workspace_alignment'


@dataclass(frozen=True)
class RecoveryOnlyRewardConfig(StabilityRecoveryConfig):
    """Default-off training-only cohort selection; no state/observation extension."""
    v231_nominal_reward_parity: bool = False


def recovery_only_penalty(penalty, group):
    """Keep the extra ball-motion loss only for near/mirror recovery lanes [N]."""
    return jnp.where(jnp.asarray(group) > 0, jnp.asarray(penalty), 0.)


def build_stages(original: Sequence[Any], bridged: Sequence[Any]) -> list[Any]:
    """Carry the exact passed five-stage prefix and all12 bridged recovery tasks."""
    if len(original) != 12 or len(bridged) != 17:
        raise ValueError('V231 requires the complete V225 prefix/V226 recovery course')
    base = original[4]
    result = list(original[:5])
    for index, stage in enumerate(bridged[5:], 6):
        cfg = RecoveryOnlyRewardConfig(**vars(stage.cfg), v231_nominal_reward_parity=True)
        cfg = replace(cfg,
            fixed_plane_v158_late_squared_vxy_weight=base.cfg.fixed_plane_v158_late_squared_vxy_weight,
            fixed_plane_v158_late_squared_omega_weight=base.cfg.fixed_plane_v158_late_squared_omega_weight)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v226_', 'fplc_v231_', 1),
            max_recent_mean_hit_racket_vxy=base.max_recent_mean_hit_racket_vxy,
            max_recent_rapid_steady_racket_vxy_rms_m_s=base.max_recent_rapid_steady_racket_vxy_rms_m_s,
            max_recent_rapid_steady_racket_omega_rms_rad_s=base.max_recent_rapid_steady_racket_omega_rms_rad_s,
            notes='V231 transfers the exact passed V225 S1--S5 and starts its '
            '12 recovery lessons at S6. Nominal rewards retain parent values; '
            'extra physical-ball-XY cost applies only to near/mirror cohorts. '
            'Only non-nominal early anchor cost uses the V226 reachable allowance. '
            'Retain72/393/7, control/physics/DR/PPO, successfulS5 replay KL and '
            'all original task/height/view/recovery gates, stricter ball caps '
            'and mandatory same-RNG frozenS5 retention. No nominal reoptimization '
            'with V228/V230 motion weights. No forced advancement.'))
    return result


def retention_stage(stage):
    """Apply protected ball caps to validation without reinterpreting old courses."""
    return replace(stage, max_recent_mean_hit_vxy=STABILITY_LIMITS['ball_mean_m_s'],
                   max_recent_rms_hit_vxy=STABILITY_LIMITS['ball_rms_m_s'])


def validate_paired_stability_contract(settings) -> None:
    if (settings.advance_mode != 'converged' or settings.advance_validation_mode != 'block'
            or settings.advance_retention_stage_name != RETENTION_STAGE
            or settings.advance_retention_final_stage_name not in (None, '')
            or settings.advance_eval_deterministic or settings.advance_eval_n_envs != 288
            or settings.advance_eval_steps != 1200):
        raise ValueError('V231 requires blocking288x1200 original-domain S5 retention')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool) -> None:
    source = payload.get('args', {}).get('curriculum_profile')
    full = (not reset_optimizer and not reset_critic and not allow_obs_dim_migration
            and payload.get('opt') is not None
            and {'pi', 'v'}.issubset(payload.get('params', {}))
            and tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) == (72, 393, 7))
    if source == PARENT_PROFILE:
        row = payload.get('last_row', {})
        if (not full or requested_start_stage != START_STAGE or resume_curriculum_state
                or payload.get('stage_name') != 'fplc_v225_s05_placed_xy_08mm'
                or payload.get('stage_index') != 5 or payload.get('stage_update') != 129
                or payload.get('step') != 1443758080 or int(payload['opt'].t) != 362007
                or any(float(row.get(x, 0.)) != 1. for x in (
                    'convergence/stage_converged', 'advance_eval/passed', 'retention_eval/passed'))):
            raise ValueError('V231 requires passed S5 and explicit fresh S6 recovery history')
        return
    if not full or source != PROFILE or payload.get('args', {}).get('max_stage_updates') != -1:
        raise ValueError('V231 rejects failed/capped/predecessor/incompatible continuation')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V231 requires exact causal stage history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
