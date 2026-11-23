"""Opt-in dense physical first-push reward;retain every legacy transition."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v307_launch_gradient import LaunchGradientConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v308_first_push'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v307_launch_gradient'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v308_first_push_weight_per_s'})


@dataclass(frozen=True)
class FirstPushConfig(LaunchGradientConfig):
    """Default-off reward weight;no observation or state extension."""
    v308_first_push_weight_per_s: float = 0.


def first_push_penalty(*, racket_vz_m_s, previous_hits, in_contact,
        episode_time_s, initial_y_m, group, target_vz_m_s: float,
        window_s: float, weight_per_s: float, dt_s: float):
    """Price only an under-speed physical first push using(N,) SI arrays.

    Read native contact and previous count0;pending first contact is allowed.
    Stop immediately on separation,confirmation or the original launch window.
    The one-sided pseudo-Huber deficit uses0.25m/s scale and bounded slope.
    This creates no event credit,counter,controller command or fake survival.
    """
    if (isinstance(weight_per_s, bool) or not math.isfinite(weight_per_s)
            or not 0. <= weight_per_s <= 400.):
        raise ValueError('V308 first-push reward rate must be finite0--400/s')
    if (not math.isfinite(target_vz_m_s) or not 0. < target_vz_m_s <= 1.2
            or not math.isfinite(window_s) or not 0. < window_s <= 1.
            or not math.isfinite(dt_s) or not 0. < dt_s < .1):
        raise ValueError('Invalid V308 SI launch target,window or timestep')
    velocity = jnp.asarray(racket_vz_m_s)
    values = [jnp.asarray(v) for v in (previous_hits, in_contact,
                                     episode_time_s, initial_y_m, group)]
    if velocity.ndim != 1 or any(v.shape != velocity.shape for v in values):
        raise ValueError('V308 requires identical(N,) physical reward arrays')
    counts, contact, time_s, initial, groups = values
    fraction = jnp.clip((initial-.035)/.010, 0., 1.)
    support = jnp.where(groups == 1, fraction*fraction*(3.-2.*fraction), 0.)
    active = ((counts == 0) & contact.astype(bool) & (time_s <= window_s)
              & (support > 0.) & (weight_per_s > 0.))
    deficit = jnp.maximum(target_vz_m_s-velocity, 0.)/.25
    cost = -weight_per_s*dt_s*support*(jnp.sqrt(1.+deficit*deficit)-1.)
    penalty = jnp.where(active, cost, 0.)
    return penalty, dict(active=active.astype(jnp.float32),
        physical_vz_m_s=jnp.where(active, velocity, 0.),
        deficit_m_s=jnp.where(active, jnp.maximum(target_vz_m_s-velocity, 0.), 0.))


def validate_config(cfg: FirstPushConfig) -> None:
    from fixed_plane_v307_launch_gradient import validate_config as previous
    previous(cfg)
    weight = cfg.v308_first_push_weight_per_s
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0. <= weight <= 400.:
        raise ValueError('Invalid V308 first-push reward rate')
    if weight > 0. and (not cfg.v225_enabled
            or cfg.fixed_contact_pre_hit_launch_vz_target_m_s != .8
            or cfg.fixed_contact_pre_hit_launch_vz_window_s != .35):
        raise ValueError('V308 preserves the original0.8m/s/0.35s launch target')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V308 requires all17 unchanged V307 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = FirstPushConfig(**vars(stage.cfg),
            v308_first_push_weight_per_s=400. if index >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if index < 11 else
            'V308 reward-only dense physical first push. Native previouscount0, '
            'actual contact and original0.35s window;group1 smooth35--45mm. '
            'One-sided deficit below original0.8m/s,0.25m/s pseudoHuber scale, '
            '400/s cost;pending firstcontact allowed. Zero after separation, '
            'confirmation,window or on nominal/mirror. No fake event,command '
            'or observation. All other rewards,first11 lessons,all17 IDs/'
            'domains/gates,72/393/7,control,physics,DR,reset and numeric '
            '1024x128 PPO/BC/KL exact. Latest full actor/critic/Adam/raw '
            'history;all32,no rejection,rollback,rewind,reset,cap or forcedadvance.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v307_launch_gradient import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V308 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0)); total = update-4869
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update or update < 4997
                or payload.get('step') != 3811049472+total*131072
                or int(payload['opt'].t) != 939902+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V308 requires latest V307 S12 full history after>=128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V308 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V308 rejects another source or from-scratch launch')
