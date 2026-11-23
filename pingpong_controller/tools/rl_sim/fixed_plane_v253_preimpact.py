"""V253 shapes third-impact preparation and sustained horizontal quality."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v252_energy_reserve import EnergyReserveConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v253'
START_STAGE = 'fplc_v253_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v253_s01_workspace_alignment'
REWARD_CHANGES = {
    'v241_physical_apex_huber_weight': 10000.,
    'v241_late_ball_xy_weight': 8.,
    'v247_startup_physical_weak_penalty_weight': 80.,
    'v247_late_physical_band_penalty_weight': 80.,
    'v253_preimpact_energy_weight_per_s': 100.,
    'v253_preimpact_energy_lead_s': .080,
    'v253_flight_xy_reward_weight_per_s': 52.,
}


@dataclass(frozen=True)
class PreparationRewardConfig(EnergyReserveConfig):
    """Default-off reward terms; state and actor/critic layouts are unchanged."""

    v253_preimpact_energy_weight_per_s: float = 0.
    v253_preimpact_energy_lead_s: float = .080
    v253_flight_xy_reward_weight_per_s: float = 0.


def validate_config(cfg: PreparationRewardConfig) -> None:
    from fixed_plane_v252_energy_reserve import validate_config as previous
    previous(cfg)
    weights = (cfg.v253_preimpact_energy_weight_per_s,
               cfg.v253_flight_xy_reward_weight_per_s)
    if any(not math.isfinite(value) or not 0. <= value <= 200. for value in weights):
        raise ValueError('V253 reward rates must be finite in [0,200] per second')
    if not math.isfinite(cfg.v253_preimpact_energy_lead_s) or not 0. < cfg.v253_preimpact_energy_lead_s <= .12:
        raise ValueError('V253 preparation lead must be finite in (0,.12] seconds')
    if any(weights) and not cfg.v225_enabled:
        raise ValueError('V253 requires the preserved near-side juggling domain')


def preimpact_energy_penalty(
    *, previous_hit_count, pending_hit, in_contact, ball_z_m, racket_z_m,
    ball_vz_m_s, racket_vz_m_s, target_apex_z_m, gravity_m_s2,
    ball_radius_m: float, effective_restitution: float,
    max_racket_vz_m_s: float, lead_s: float, velocity_scale_m_s: float,
    weight_per_s: float, dt_s: float, prepare_previous_hits: int = 2,
):
    """Price missing upward preparation; legacy calls select the third impact.

    Inputs are true simulation quantities of shape (N,), used only by reward.
    A constant-velocity racket and ballistic ball predict the contact time and
    height. The required racket velocity follows the normal impact equation
    for the existing reward apex and nominal restitution; it changes no plant,
    policy target, collision or action. The one-sided pseudo-Huber cost remains
    graded for weak strokes, and the existing physical outcome costs bound
    excessive upward energy. Static support and pending impacts are excluded.
    """
    scalars = (ball_radius_m,effective_restitution,max_racket_vz_m_s,lead_s,
               velocity_scale_m_s,weight_per_s,dt_s)
    if prepare_previous_hits not in (1, 2):
        raise ValueError('Preparation must select the second or third impact')
    if (not all(math.isfinite(value) for value in scalars)
            or any(value <= 0. for value in scalars[:5]) or dt_s <= 0.
            or weight_per_s < 0.):
        raise ValueError('Invalid V253 preparation reward parameters')
    gravity = jnp.maximum(jnp.asarray(gravity_m_s2),1e-6)
    ball_vz = jnp.asarray(ball_vz_m_s)
    racket_vz = jnp.asarray(racket_vz_m_s)
    gap = jnp.asarray(ball_z_m)-jnp.asarray(racket_z_m)-ball_radius_m
    relative_vz = ball_vz-racket_vz
    discriminant = jnp.maximum(relative_vz**2+2.*gravity*jnp.maximum(gap,0.),0.)
    ttc = (relative_vz+jnp.sqrt(discriminant))/gravity
    contact_z = jnp.asarray(racket_z_m)+ball_radius_m+racket_vz*ttc
    incoming_vz = ball_vz-gravity*ttc
    outgoing_vz = jnp.sqrt(2.*gravity*jnp.maximum(jnp.asarray(target_apex_z_m)-contact_z,0.))
    required_vz = jnp.clip((outgoing_vz+effective_restitution*incoming_vz)
                           /(1.+effective_restitution),0.,max_racket_vz_m_s)
    active = ((jnp.asarray(previous_hit_count)==prepare_previous_hits) & (~jnp.asarray(pending_hit,dtype=bool))
              & (~jnp.asarray(in_contact,dtype=bool)) & (ball_vz < 0.)
              & (gap > 0.) & (ttc > 0.) & (ttc <= lead_s))
    error = jnp.maximum(required_vz-racket_vz,0.)/velocity_scale_m_s
    cost = -weight_per_s*dt_s*(jnp.sqrt(1.+error**2)-1.)
    return jnp.where(active,cost,0.),active,required_vz,ttc


def flight_horizontal_quality_reward(
    *, previous_hit_count, in_contact, relative_z_m, speed_m_s,
    weight_per_s: float, dt_s: float,
):
    """Bounded (N,) flight credit after hit three, with 18 mm/s headroom.

    No credit is paid during support/contact or early launch. Credit is
    bounded per second, so more weak contacts cannot farm extra event credit.
    The physical hit/survival/view/motion gates remain the acceptance test.
    """
    if not math.isfinite(weight_per_s) or weight_per_s < 0. or not math.isfinite(dt_s) or dt_s <= 0.:
        raise ValueError('Invalid V253 horizontal flight reward parameters')
    active = ((jnp.asarray(previous_hit_count)>=3) & (~jnp.asarray(in_contact,dtype=bool))
              & (jnp.asarray(relative_z_m)>.030))
    excess = jnp.maximum(jnp.asarray(speed_m_s)-.018,0.)/.010
    score = jnp.exp(-.5*excess**2)
    return jnp.where(active,weight_per_s*dt_s*score,0.)


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source)!=17 or source[8].name!='fplc_v252_s09_near_y_035mm_bridge':
        raise ValueError('V253 requires the complete17-lesson V252 course')
    result=[]
    for stage in source:
        cfg=replace(PreparationRewardConfig(**vars(stage.cfg)),**REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,
            name=stage.name.replace('fplc_v252_','fplc_v253_',1),
            notes='V253 rewards physical third-impact upward preparation within80ms '
            'of moving-racket ballistic contact,100/s pseudo-Huber shortfall; '
            'bounded late free-flight horizontal quality52/s. Reduce rare '
            'outcome Huber40000->10000,weak600->80,lateband240->80,lateXY28->8 '
            'instead of multiplying late fines. Existing first energy band, '
            'physical goals,17 domains/gates,72/393/7,physics/reset/control/DR '
            'and original uncapped1024x128 PPO remain. No sample rejection, '
            'rollback, optimizer/critic reset or historical-actor veto.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v252_energy_reserve import validate_training_contract as previous
    if settings.advance_retention_stage_name!=RETENTION_STAGE:
        raise ValueError('V253 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v252_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str,Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V253 requires full uncapped72/393/7 actor/critic/Adam')
    source=payload.get('args',{}).get('curriculum_profile')
    if source in ('fixed_plane_limit_cycle_qvel_rmp_v235','fixed_plane_limit_cycle_qvel_rmp_v252'):
        old_prefix='fplc_'+source.rsplit('_',1)[1]+'_'
        expected=str(payload.get('stage_name','')).replace(old_prefix,'fplc_v253_',1)
        if resume_curriculum_state or requested_start_stage!=expected or int(payload.get('stage_index',0))<9:
            raise ValueError('V253 starts fresh reward statistics only in the current lesson')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V253 continuation requires exact causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V253 rejects another source or a from-scratch launch')
