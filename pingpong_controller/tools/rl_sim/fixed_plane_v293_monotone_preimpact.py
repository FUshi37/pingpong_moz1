"""Opt-in velocity-monotone preparation reward; original physics stay exact."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v292_critical_startup import CriticalStartupConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v293_monotone_preimpact'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v292_critical_startup'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v293_monotone_preimpact_enabled'})


@dataclass(frozen=True)
class MonotonePreparationConfig(CriticalStartupConfig):
    """Default-off reward calculation mode; no additional physical state."""
    v293_monotone_preimpact_enabled: bool = False


def preimpact_energy_penalty(
    *, previous_hit_count, pending_hit, in_contact, ball_z_m, racket_z_m,
    ball_vz_m_s, racket_vz_m_s, target_apex_z_m, gravity_m_s2,
    ball_radius_m: float, effective_restitution: float,
    max_racket_vz_m_s: float, lead_s: float, velocity_scale_m_s: float,
    weight_per_s: float, dt_s: float, prepare_previous_hits: int = 2,
):
    """Use a velocity-independent window and required velocity for (N,) inputs.

    Ballistic TTC and contact height use the current racket position held
    fixed. Actual racket velocity enters only the one-sided shortfall. At a
    fixed ball/racket position, increasing upward velocity therefore cannot
    worsen this reward or switch on its penalty. Smooth onset over the last
    min(30ms,lead/2) of the original lead removes a discontinuous entry cost.
    This is a reward proxy only, never a changed collision or control target.
    """
    scalars = (ball_radius_m, effective_restitution, max_racket_vz_m_s,
               lead_s, velocity_scale_m_s, weight_per_s, dt_s)
    if prepare_previous_hits not in (1, 2):
        raise ValueError('Preparation must select the second or third impact')
    if (not all(math.isfinite(value) for value in scalars)
            or any(value <= 0. for value in scalars[:5]) or dt_s <= 0.
            or weight_per_s < 0.):
        raise ValueError('Invalid V293 preparation reward parameters')
    gravity = jnp.maximum(jnp.asarray(gravity_m_s2), 1e-6)
    ball_vz = jnp.asarray(ball_vz_m_s)
    racket_vz = jnp.asarray(racket_vz_m_s)
    gap = jnp.asarray(ball_z_m) - jnp.asarray(racket_z_m) - ball_radius_m
    discriminant = jnp.maximum(ball_vz ** 2 + 2. * gravity * jnp.maximum(gap, 0.), 0.)
    ttc = (ball_vz + jnp.sqrt(discriminant)) / gravity
    contact_z = jnp.asarray(racket_z_m) + ball_radius_m
    incoming_vz = ball_vz - gravity * ttc
    outgoing_vz = jnp.sqrt(2. * gravity * jnp.maximum(
        jnp.asarray(target_apex_z_m) - contact_z, 0.))
    required_vz = jnp.clip((outgoing_vz + effective_restitution * incoming_vz)
                           / (1. + effective_restitution), 0., max_racket_vz_m_s)
    active = ((jnp.asarray(previous_hit_count) == prepare_previous_hits)
              & ~jnp.asarray(pending_hit, dtype=bool)
              & ~jnp.asarray(in_contact, dtype=bool)
              & (ball_vz < 0.) & (gap > 0.) & (ttc > 0.) & (ttc < lead_s))
    fraction = jnp.clip((lead_s - ttc) / min(.030, .5 * lead_s), 0., 1.)
    urgency = fraction * fraction * (3. - 2. * fraction)
    error = jnp.maximum(required_vz - racket_vz, 0.) / velocity_scale_m_s
    cost = -weight_per_s * dt_s * urgency * (jnp.sqrt(1. + error ** 2) - 1.)
    return jnp.where(active, cost, 0.), active, required_vz, ttc


def second_preparation_penalty(*, initial_y_m, group, target_apex_z_m, **kwargs):
    """Keep the original second-impact near gain and reserve exactly."""
    from fixed_plane_v276_second_launch import (
        NEAR_Y_START_M, NEAR_Y_FULL_M, SECOND_APEX_RESERVE_M)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - NEAR_Y_START_M)
                        / (NEAR_Y_FULL_M - NEAR_Y_START_M), 0., 1.)
    smooth = fraction * fraction * (3. - 2. * fraction)
    scale = jnp.where(jnp.asarray(group) == 1, 8. * smooth, 0.)
    cost, active, required_vz, ttc = preimpact_energy_penalty(
        **kwargs, target_apex_z_m=jnp.maximum(
            jnp.asarray(target_apex_z_m), SECOND_APEX_RESERVE_M),
        prepare_previous_hits=1)
    active = active & (scale > 0.)
    return jnp.where(active, cost * scale, 0.), active, required_vz, ttc


def validate_config(cfg: MonotonePreparationConfig) -> None:
    from fixed_plane_v292_critical_startup import validate_config as previous
    previous(cfg)
    if type(cfg.v293_monotone_preimpact_enabled) is not bool:
        raise ValueError('V293 reward mode must be an explicit boolean')
    if cfg.v293_monotone_preimpact_enabled and (
            cfg.v253_preimpact_energy_weight_per_s <= 0.
            or cfg.v276_second_prepare_weight_per_s <= 0.):
        raise ValueError('V293 requires both original preparation prices')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V293 requires all17 original V292 physical lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = MonotonePreparationConfig(**vars(stage.cfg),
            v293_monotone_preimpact_enabled=i >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V293 reward-only monotone preparation: V253 third and V276 second '
            'use ball-only ballistic TTC/current fixed surface height with30ms '
            'smooth onset inside the unchanged lead. Upward actual racket '
            'velocity cannot switch on/worsen the loss at fixed ball/racket '
            'positions. Original goals,restitution,VZ cap,weights,near gains '
            'and pending/contact semantics exact; every other reward,17 V281 '
            'domains/gates,72/393/7,control/physics/reset/DR and all numeric '
            '1024x128 PPO/BC/KL unchanged. Latest full state/history; all32 '
            'minibatches, no cap/reset/rewind/rejection/rollback or forced advance.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v292_critical_startup import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V293 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0)); tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2340
                or payload.get('step') != 3462791168 + (update - 2212) * 131072
                or int(payload['opt'].t) != 854878 + (update - 2212) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V293 requires latest full V292 S12 after128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V293 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V293 rejects another source or from-scratch launch')
