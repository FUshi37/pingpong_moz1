"""Default-off late physical-ball reward margin; all original gates exact."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v316_late_ball_margin import LateBallMarginConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v317_late_ball_margin_gain'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v316_late_ball_margin'
START_STAGE = 'fplc_v281_s16_inward_momentum_080mms'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v317_late_ball_xy_margin_gain_enabled'})


@dataclass(frozen=True)
class LateBallMarginGainConfig(LateBallMarginConfig):
    """Reward-only opt-in; retain the parent's physical and learner contracts."""
    v317_late_ball_xy_margin_gain_enabled: bool = False


def late_ball_margin_gain_correction(*, parent_penalty: Any, ball_vxy_m_s: Any,
                                  previous_hits: Any, confirmed_event: Any,
                                  terminated: Any, enabled: bool
                                  ) -> tuple[Any, dict[str, Any]]:
    """Replace only the late (N,) event loss by -min(256*x*x,64).

    Here x=max(native5ms_ball_vxy-.025,0)/.030. The parent remains logged
    unchanged; return only target-minus-parent. First3 contacts, failures
    and unconfirmed events retain their exact existing rewards. The bound
    is part of the reward, never a sample, gradient, action or update cap.
    """
    if type(enabled) is not bool:
        raise ValueError('V317 opt-in must be Boolean')
    old, velocity, counts, event, failure = map(jnp.asarray,
        (parent_penalty, ball_vxy_m_s, previous_hits, confirmed_event, terminated))
    if old.ndim != 1 or any(v.shape != old.shape for v in (velocity, counts, event, failure)):
        raise ValueError('V317 requires identical (N,) arrays')
    active = enabled & (counts >= 3) & event.astype(bool) & ~failure.astype(bool)
    excess = jnp.maximum(velocity-.025, 0.)
    target = -jnp.minimum(256.*(excess/.030)**2, 64.)
    extra = jnp.where(active, target-old, 0.)
    return extra, dict(active=(active & (excess > 0.)).astype(jnp.float32),
        target_penalty=jnp.where(active, target, 0.),
        original_penalty=jnp.where(active, old, 0.), extra_penalty=extra)


def validate_config(cfg: LateBallMarginGainConfig) -> None:
    from fixed_plane_v316_late_ball_margin import validate_config as previous
    previous(cfg)
    if type(cfg.v317_late_ball_xy_margin_gain_enabled) is not bool:
        raise ValueError('V317 reward opt-in must be Boolean')
    if cfg.v317_late_ball_xy_margin_gain_enabled and not cfg.v316_late_ball_xy_margin_enabled:
        raise ValueError('V317 requires the unchanged V316 event ledger')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[15].name != START_STAGE:
        raise ValueError('V317 requires all 17 unchanged V316 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = LateBallMarginGainConfig(**vars(stage.cfg),
                                    v317_late_ball_xy_margin_gain_enabled=i >= 15)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 15 else
            'V317 changes only coefficient16->256 on the same native late '
            'physical-ball .025m/s margin loss. Nonfailure confirmed '
            'previouscount>=3 only: total=-min(256*(max(vxy-.025,0)/.030)^2,64). '
            'The64 event bound remains exact; add target-minus-parent only. '
            'First3,first15,all17 domains/gates,72/393/7,control,physics,DR,reset '
            'and numeric1024x128 PPO/BC/KL exact. Full actor/critic/Adam/raw96 '
            'history;all32;no rejection,rollback,rewind,reset,cap or forced advancement.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v316_late_ball_margin import validate_training_contract as previous
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
        raise ValueError('V317 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 16 or update < 786
                or payload.get('global_update') != update
                or payload.get('step') != 3958898688+update*131072
                or int(payload['opt'].t) != 975998+update*32
                or not isinstance(tail, list) or len(tail) != 96
                or [int(r['stage_update']) for r in tail] != list(range(update-95, update+1))
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V317 requires latest full V316 S16 after 128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V317 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V317 rejects another source or from-scratch launch')
