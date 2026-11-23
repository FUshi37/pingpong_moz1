"""Default-off mid-near bridge for the effective preparation reward."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v294_effective_preimpact import EffectivePreparationConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v295_mid_near_preimpact'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v294_effective_preimpact'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v295_mid_near_preparation_scale'})


@dataclass(frozen=True)
class MidNearPreparationConfig(EffectivePreparationConfig):
    """One reward scalar; inherit V292 directly, not the rejected V293 mode."""
    v295_mid_near_preparation_scale: float = 0.


def bridge_support(initial_y_m, group):
    """Complement the old critical support without increasing its full tail."""
    y = jnp.asarray(initial_y_m)
    broad_x = jnp.clip((y - .035) / .010, 0., 1.)
    old_x = jnp.clip((y - .075) / .010, 0., 1.)
    broad = broad_x * broad_x * (3. - 2. * broad_x)
    old = old_x * old_x * (3. - 2. * old_x)
    return jnp.where(jnp.asarray(group) == 1, jnp.maximum(broad - old, 0.), 0.)


def mid_near_preparation_penalty(
    *, previous_hit_count, pending_hit, in_contact, ball_z_m, racket_z_m,
    ball_vz_m_s, racket_vz_m_s, base_apex_goal_m, third_apex_goal_m,
    initial_y_m, group, gravity_m_s2, ball_radius_m: float,
    effective_restitution: float, max_racket_vz_m_s: float,
    second_lead_s: float, third_lead_s: float, second_weight_per_s: float,
    third_weight_per_s: float, near_preparation_gain: float,
    weight_scale: float, dt_s: float,
):
    """Return the missing mid-near part of the effective preparation loss.

    Original V294 critical preparation and all earlier terms remain exact.
    Bridge support is smooth35--45mm minus smooth75--85mm, group1 only.
    Equal0.25 weights make the two terms one smooth broad support, without
    adding cost at or beyond85mm. Helpers retain original actual-Y near gains,
    ball-only TTC, fixed-surface target,30ms onset and original reward inputs.
    """
    if not math.isfinite(weight_scale) or not 0. <= weight_scale <= 1.:
        raise ValueError('V295 weight scale must be finite in[0,1]')
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
    support = bridge_support(initial_y_m, group)
    scale = weight_scale * support
    second_effective, third_effective = scale * second, scale * third
    active = (scale > 0.) & (second_active | third_active)
    target = jnp.where(second_active, second_target, third_target)
    ttc = jnp.where(second_active, second_ttc, third_ttc)
    metrics = dict(bridge_support=support, active=active.astype(jnp.float32),
        second_penalty=second_effective, third_penalty=third_effective,
        target_vz=jnp.where(active, target, 0.), ttc=jnp.where(active, ttc, 0.),
        racket_vz=jnp.where(active, jnp.asarray(racket_vz_m_s), 0.))
    return second_effective + third_effective, metrics


def validate_config(cfg: MidNearPreparationConfig) -> None:
    from fixed_plane_v294_effective_preimpact import validate_config as previous
    previous(cfg)
    scale = cfg.v295_mid_near_preparation_scale
    if not math.isfinite(scale) or not 0. <= scale <= 1.:
        raise ValueError('V295 preparation scale must be finite in[0,1]')
    if scale > 0. and scale != cfg.v294_critical_preparation_scale:
        raise ValueError('V295 bridge must match the preserved V294 preparation scale')
    if scale > 0. and (cfg.v279_energy_progress_weight <= 0.
            or cfg.v253_preimpact_energy_weight_per_s <= 0.
            or cfg.v276_second_prepare_weight_per_s <= 0.):
        raise ValueError('V295 requires the original cancellation and preparation context')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V295 requires all17 original V294 physical lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = MidNearPreparationConfig(**vars(stage.cfg),
            v295_mid_near_preparation_scale=.25 if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V295 reward-only missing-support bridge. Preserve every V294 term, '
            'especially its0.25 critical preparation. Append0.25 times '
            '(smooth35--45mm minus smooth75--85mm) group1 preparation loss, '
            'after all old shaping. Original second/third helper targets, '
            'near gains,120ms lead and30ms onset unchanged. Covers failed '
            '47/60/73mm starts without reweighting85mm+ or nominal/mirror. '
            'No events/state/obs/control/physics/gates/PPO changes. All17 '
            'V281 identities and numeric1024x128 settings, latest full '
            'actor/critic/Adam/history, all32 applied, no cap/reset/rewind/'
            'rejection/rollback/interpolation/forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v294_effective_preimpact import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V295 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0)); tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2660
                or payload.get('step') != 3504734208 + (update - 2532) * 131072
                or int(payload['opt'].t) != 865118 + (update - 2532) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V295 requires latest full V294 S12 state after128 actual updates and exact history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V295 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V295 rejects another source or from-scratch launch')
