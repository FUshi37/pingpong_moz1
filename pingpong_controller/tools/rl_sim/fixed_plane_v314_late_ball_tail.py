"""Default-off bounded late physical-ball XY tail reward."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v311_clean_preimpact import CleanPreimpactConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v314_late_ball_tail'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v313_stage16_preimpact'
START_STAGE = 'fplc_v281_s16_inward_momentum_080mms'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v314_late_ball_xy_tail_enabled'})


@dataclass(frozen=True)
class LateBallTailConfig(CleanPreimpactConfig):
    """Reward-only switch; no new state, observation or control parameter."""
    v314_late_ball_xy_tail_enabled: bool = False


def late_ball_tail_penalty(*, ball_vxy_m_s: Any, previous_hits: Any,
                          confirmed_event: Any, terminated: Any,
                          enabled: bool) -> tuple[Any, dict[str, Any]]:
    """Price native late confirmed XY speed above .030 m/s, at most 16/event.

    Inputs are identical (N,) arrays. Velocity is the original true 5 ms
    position-difference speed. The first three contacts, unconfirmed events
    and failures retain the parent reward exactly. This bounds an event
    reward term, never an action, gradient, sample or minibatch acceptance.
    """
    if type(enabled) is not bool:
        raise ValueError('V314 opt-in must be Boolean')
    velocity, counts, event, failure = map(jnp.asarray,
        (ball_vxy_m_s, previous_hits, confirmed_event, terminated))
    if velocity.ndim != 1 or any(v.shape != velocity.shape for v in (counts, event, failure)):
        raise ValueError('V314 requires identical (N,) arrays')
    active = enabled & (counts >= 3) & event.astype(bool) & ~failure.astype(bool)
    excess = jnp.maximum(velocity-.030, 0.)
    penalty = jnp.where(active, -jnp.minimum(2.*(excess/.030)**2, 16.), 0.)
    return penalty, dict(active=(active & (excess > 0.)).astype(jnp.float32),
        physical_ball_vxy_m_s=jnp.where(active, velocity, 0.),
        excess_m_s=jnp.where(active, excess, 0.), penalty=penalty)


def validate_config(cfg: LateBallTailConfig) -> None:
    from fixed_plane_v311_clean_preimpact import validate_config as previous
    previous(cfg)
    if type(cfg.v314_late_ball_xy_tail_enabled) is not bool:
        raise ValueError('V314 reward opt-in must be Boolean')
    if cfg.v314_late_ball_xy_tail_enabled and (
            not cfg.v311_clean_preimpact_enabled
            or getattr(cfg, 'fixed_plane_v167_global_reward_multiplier', 1.) != 1.):
        raise ValueError('V314 requires unchanged V313 startup and unit reward scale')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[15].name != START_STAGE:
        raise ValueError('V314 requires all 17 unchanged V313 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = LateBallTailConfig(**vars(stage.cfg),
                                v314_late_ball_xy_tail_enabled=i >= 15)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 15 else
            'V314 reward-only late physical-ball XY tail loss. Only native '
            'nonterminal confirmed events at previouscount>=3 add '
            '-min(2*(max(true5ms_ball_vxy-.030,0)/.030)^2,16). Keep first3 '
            'rewards, all V313 startup preparation, first15 lessons, all17 '
            'domains/gates,72/393/7,control,physics,DR,reset and numeric '
            '1024x128 PPO/BC/KL exact. Latest full S16 actor/critic/Adam/'
            'raw96history;all32,no rejection,rollback,rewind,reset,cap or '
            'forced advancement. Event loss cap is reward shaping only.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v313_stage16_preimpact import validate_training_contract as previous
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
        raise ValueError('V314 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 16 or update < 319
                or payload.get('global_update') != update
                or payload.get('step') != 3958898688+update*131072
                or int(payload['opt'].t) != 975998+update*32
                or not isinstance(tail, list) or len(tail) != 96
                or [int(r['stage_update']) for r in tail] != list(range(update-95, update+1))
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V314 requires latest full V313 S16 after 128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V314 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V314 rejects another source or from-scratch launch')
