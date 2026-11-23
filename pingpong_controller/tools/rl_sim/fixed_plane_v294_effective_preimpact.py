"""Default-off critical preparation term appended after legacy cancellation."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v292_critical_startup import CriticalStartupConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v294_effective_preimpact'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v292_critical_startup'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v294_critical_preparation_scale'})


@dataclass(frozen=True)
class EffectivePreparationConfig(CriticalStartupConfig):
    """One reward scalar; inherit V292 directly, not the rejected V293 mode."""
    v294_critical_preparation_scale: float = 0.


def effective_preparation_penalty(
    *, previous_hit_count, pending_hit, in_contact, ball_z_m, racket_z_m,
    ball_vz_m_s, racket_vz_m_s, base_apex_goal_m, third_apex_goal_m,
    initial_y_m, group, gravity_m_s2, ball_radius_m: float,
    effective_restitution: float, max_racket_vz_m_s: float,
    second_lead_s: float, third_lead_s: float, second_weight_per_s: float,
    third_weight_per_s: float, near_preparation_gain: float,
    weight_scale: float, dt_s: float,
):
    """Return a distinct effective (N,) loss; caller appends it at step end.

    Pure V293 preparation helpers are reused without activating its profile.
    The original V253/V276/V279 composition remains intact. This NEW term is
    never passed as old_preparation_cost to V279. Group1 initialY75--85mm
    smooth support and scalar0.25 limit the additional signal to hard starts.
    Inputs must use the actual reward caller's finite-difference ball velocity
    and contact-point racket velocity. No counter, target or control changes.
    """
    if not math.isfinite(weight_scale) or not 0. <= weight_scale <= 1.:
        raise ValueError('V294 weight scale must be finite in[0,1]')
    from fixed_plane_v293_monotone_preimpact import (
        preimpact_energy_penalty, second_preparation_penalty)
    from fixed_plane_v255_corner import near_preparation_multiplier
    common = dict(previous_hit_count=previous_hit_count, pending_hit=pending_hit,
        in_contact=in_contact, ball_z_m=ball_z_m, racket_z_m=racket_z_m,
        ball_vz_m_s=ball_vz_m_s, racket_vz_m_s=racket_vz_m_s,
        gravity_m_s2=gravity_m_s2, ball_radius_m=ball_radius_m,
        effective_restitution=effective_restitution,
        max_racket_vz_m_s=max_racket_vz_m_s, velocity_scale_m_s=.25, dt_s=dt_s)
    second, second_active, second_target, second_ttc = second_preparation_penalty(
        **common, initial_y_m=initial_y_m, group=group,
        target_apex_z_m=base_apex_goal_m, lead_s=second_lead_s,
        weight_per_s=second_weight_per_s)
    third, third_active, third_target, third_ttc = preimpact_energy_penalty(
        **common, target_apex_z_m=third_apex_goal_m, lead_s=third_lead_s,
        weight_per_s=third_weight_per_s)
    if near_preparation_gain > 0.:
        third = third * near_preparation_multiplier(initial_y_m=initial_y_m,
            group=group, gain=near_preparation_gain)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .075) / .010, 0., 1.)
    support = jnp.where(jnp.asarray(group) == 1,
        fraction * fraction * (3. - 2. * fraction), 0.)
    scale = weight_scale * support
    second_effective, third_effective = scale * second, scale * third
    active = (scale > 0.) & (second_active | third_active)
    target = jnp.where(second_active, second_target, third_target)
    ttc = jnp.where(second_active, second_ttc, third_ttc)
    metrics = dict(critical_support=support, active=active.astype(jnp.float32),
        second_penalty=second_effective, third_penalty=third_effective,
        target_vz=jnp.where(active, target, 0.), ttc=jnp.where(active, ttc, 0.),
        racket_vz=jnp.where(active, jnp.asarray(racket_vz_m_s), 0.))
    return second_effective + third_effective, metrics


def validate_config(cfg: EffectivePreparationConfig) -> None:
    from fixed_plane_v292_critical_startup import validate_config as previous
    previous(cfg)
    scale = cfg.v294_critical_preparation_scale
    if not math.isfinite(scale) or not 0. <= scale <= 1.:
        raise ValueError('V294 preparation scale must be finite in[0,1]')
    if scale > 0. and (cfg.v279_energy_progress_weight <= 0.
            or cfg.v253_preimpact_energy_weight_per_s <= 0.
            or cfg.v276_second_prepare_weight_per_s <= 0.):
        raise ValueError('V294 requires the original cancellation and preparation context')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V294 requires all17 original V292 physical lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = EffectivePreparationConfig(**vars(stage.cfg),
            v294_critical_preparation_scale=.25 if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V294 reward-only EFFECTIVE critical preparation: keep V292 '
            'and original V253/V276/V279 composition intact. Append a NEW '
            'named monotone preimpact loss after all old cancellation, only '
            'group1 initialY75--85mm, scale0.25. Ball-only TTC/fixed surface '
            'target and30ms smooth onset, original target/weights/near gains. '
            'Returned PPO reward delta must equal this term. No V293 mode, '
            'new event/state/obs/control/physics. All17 V281 domains/gates, '
            '72/393/7 and numerical1024x128 PPO/BC/KL exact. Latest full '
            'state/history, all32 applied, no cap/reset/rewind/rejection/'
            'rollback/interpolation or forced advancement.')))
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
        raise ValueError('V294 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0)); tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2414
                or payload.get('step') != 3462791168 + (update - 2212) * 131072
                or int(payload['opt'].t) != 854878 + (update - 2212) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V294 requires latest full V292 S12 state and exact history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V294 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V294 rejects another source or from-scratch launch')
