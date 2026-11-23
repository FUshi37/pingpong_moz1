"""Default-off gain on the same bounded late physical-ball tail loss."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v314_late_ball_tail import LateBallTailConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v315_late_ball_tail_gain'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v314_late_ball_tail'
START_STAGE = 'fplc_v281_s16_inward_momentum_080mms'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v315_late_ball_xy_tail_gain_enabled'})


@dataclass(frozen=True)
class LateBallTailGainConfig(LateBallTailConfig):
    """Reward-only opt-in; retain the parent's physical and learner contracts."""
    v315_late_ball_xy_tail_gain_enabled: bool = False


def late_ball_tail_gain_correction(*, parent_penalty: Any, ball_vxy_m_s: Any,
                                  previous_hits: Any, confirmed_event: Any,
                                  terminated: Any, enabled: bool
                                  ) -> tuple[Any, dict[str, Any]]:
    """Replace only the late (N,) event loss by -min(16*x*x,64).

    Here x=max(native5ms_ball_vxy-.030,0)/.030. The parent remains logged
    unchanged; return only target-minus-parent. First3 contacts, failures
    and unconfirmed events retain their exact existing rewards. The bound
    is part of the reward, never a sample, gradient, action or update cap.
    """
    if type(enabled) is not bool:
        raise ValueError('V315 opt-in must be Boolean')
    old, velocity, counts, event, failure = map(jnp.asarray,
        (parent_penalty, ball_vxy_m_s, previous_hits, confirmed_event, terminated))
    if old.ndim != 1 or any(v.shape != old.shape for v in (velocity, counts, event, failure)):
        raise ValueError('V315 requires identical (N,) arrays')
    active = enabled & (counts >= 3) & event.astype(bool) & ~failure.astype(bool)
    excess = jnp.maximum(velocity-.030, 0.)
    target = -jnp.minimum(16.*(excess/.030)**2, 64.)
    extra = jnp.where(active, target-old, 0.)
    return extra, dict(active=(active & (excess > 0.)).astype(jnp.float32),
        target_penalty=jnp.where(active, target, 0.),
        original_penalty=jnp.where(active, old, 0.), extra_penalty=extra)


def validate_config(cfg: LateBallTailGainConfig) -> None:
    from fixed_plane_v314_late_ball_tail import validate_config as previous
    previous(cfg)
    if type(cfg.v315_late_ball_xy_tail_gain_enabled) is not bool:
        raise ValueError('V315 reward opt-in must be Boolean')
    if cfg.v315_late_ball_xy_tail_gain_enabled and not cfg.v314_late_ball_xy_tail_enabled:
        raise ValueError('V315 requires the unchanged V314 event ledger')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[15].name != START_STAGE:
        raise ValueError('V315 requires all 17 unchanged V314 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = LateBallTailGainConfig(**vars(stage.cfg),
                                    v315_late_ball_xy_tail_gain_enabled=i >= 15)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 15 else
            'V315 reward-only gain on the same native late physical-ball '
            'XY tail. Only nonfailure confirmed previouscount>=3 replaces '
            '-min(2*x*x,16) by -min(16*x*x,64), x=max(true5ms_vxy-.030,0)/.030. '
            'Log the parent unchanged; add target-minus-parent only. Keep '
            'first3 rewards,first15 lessons,all17 domains/gates,72/393/7, '
            'control,physics,DR,reset and numeric1024x128 PPO/BC/KL exact. '
            'Latest full actor/critic/Adam/raw96history;all32,no rejection, '
            'rollback,rewind,reset,update cap or forced advancement.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v314_late_ball_tail import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
                            requested_start_stage: str,
                            resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool,
                            allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in (
                'obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V315 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 16 or update < 466
                or payload.get('global_update') != update
                or payload.get('step') != 3958898688+update*131072
                or int(payload['opt'].t) != 975998+update*32
                or not isinstance(tail, list) or len(tail) != 96
                or [int(r['stage_update']) for r in tail] != list(range(update-95, update+1))
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V315 requires latest full V314 S16 after 128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V315 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V315 rejects another source or from-scratch launch')
