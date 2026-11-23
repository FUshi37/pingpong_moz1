"""Opt-in observable reference and physical-velocity preparation envelopes."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v303_preimpact_timing import PreimpactTimingConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v304_timing_envelope'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v303_preimpact_timing'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v304_timing_envelope_reward'})


@dataclass(frozen=True)
class TimingEnvelopeConfig(PreimpactTimingConfig):
    """Default-off reward replacement; no new controller or observation state."""
    v304_timing_envelope_reward: bool = False


def validate_config(cfg: TimingEnvelopeConfig) -> None:
    from fixed_plane_v303_preimpact_timing import validate_config as previous
    previous(cfg)
    if type(cfg.v304_timing_envelope_reward) is not bool:
        raise ValueError('V304 reward switch must be an explicit boolean')
    if cfg.v304_timing_envelope_reward and not cfg.v303_late_action_label_fade:
        raise ValueError('V304 requires the exact V303 reward ledger')


def _smooth(value):
    x = jnp.clip(value, 0., 1.)
    return x*x*(3.-2.*x)


def preparation_reward_correction(*, old_label_cost, old_preparation_cost,
                                  new_reward):
    """Replace the parent's net preparation cost using identical (N,) arrays."""
    label, physical, desired = map(jnp.asarray,
        (old_label_cost, old_preparation_cost, new_reward))
    if label.ndim != 1 or label.shape != physical.shape or label.shape != desired.shape:
        raise ValueError('V304 reward ledger requires identical (N,) vectors')
    return desired-label-physical


def preparation_velocity_floor(*, ttc_s, required_vz_m_s):
    ttc, required = jnp.asarray(ttc_s), jnp.asarray(required_vz_m_s)
    if ttc.ndim != 1 or ttc.shape != required.shape:
        raise ValueError('V304 floor requires identical(N,) vectors')
    returning = -.4*_smooth((ttc-.100)/.080)
    approaching = required*_smooth((.100-ttc)/.100)
    return jnp.where(ttc >= .100, returning, approaching)


def reference_preparation_cost(*, ttc_s, reference_z_lead_m, dt_s: float):
    """Prepare a0--35mm reference lead early;allow late reference braking."""
    ttc, lead = jnp.asarray(ttc_s), jnp.asarray(reference_z_lead_m)
    if ttc.ndim != 1 or ttc.shape != lead.shape or not 0. < dt_s < .1:
        raise ValueError('V304 reference reward requires SI(N,) vectors')
    target = .035*_smooth((.180-ttc)/.090)
    retained = _smooth((ttc-.060)/.030)
    error = (jnp.maximum(target-lead, 0.)+jnp.maximum(lead-.060, 0.))/.020
    cost = -200.*dt_s*retained*(jnp.sqrt(1.+error*error)-1.)
    return cost, target, retained


def envelope_reward(*, previous_hits, pending, in_contact, ball_z_m,
        racket_z_m, ball_vz_m_s, racket_vz_m_s, gravity_m_s2,
        ball_radius_m: float, base_apex_goal_m, third_apex_goal_m,
        reference_z_lead_m, initial_y_m, group, effective_restitution: float,
        max_racket_vz_m_s: float, second_weight_per_s: float,
        third_weight_per_s: float, near_preparation_gain: float, dt_s: float):
    """Negative(N,) preparation loss;never prescribe or apply joint actions.

    Keep original ball-only TTC,impact-energy target and actual-Y phase
    weights. Permit returning velocity early,ramp its floor to zero at100ms,
    then to required impact velocity at contact. The reference lead prepares
    ahead of the plant response;its cost fades over90--60ms. This is a reward
    proxy,not a calibrated plant delay or a controller reference command.
    """
    from fixed_plane_v293_monotone_preimpact import preimpact_energy_penalty
    from fixed_plane_v276_second_launch import (
        NEAR_Y_START_M, NEAR_Y_FULL_M, SECOND_APEX_RESERVE_M)
    from fixed_plane_v255_corner import near_preparation_multiplier
    scalars = (ball_radius_m, effective_restitution, max_racket_vz_m_s,
               second_weight_per_s, third_weight_per_s, dt_s)
    if not all(math.isfinite(x) and x > 0. for x in scalars):
        raise ValueError('Invalid V304 positive SI reward parameters')
    counts = jnp.asarray(previous_hits)
    values = [jnp.asarray(x) for x in (pending, in_contact, ball_z_m, racket_z_m,
        ball_vz_m_s, racket_vz_m_s, gravity_m_s2, base_apex_goal_m,
        third_apex_goal_m, reference_z_lead_m, initial_y_m, group)]
    if counts.ndim != 1 or any(x.shape != counts.shape for x in values):
        raise ValueError('V304 preparation inputs require identical(N,) shapes')
    common = dict(previous_hit_count=counts, pending_hit=pending,
        in_contact=in_contact, ball_z_m=ball_z_m, racket_z_m=racket_z_m,
        ball_vz_m_s=ball_vz_m_s, racket_vz_m_s=racket_vz_m_s,
        gravity_m_s2=jnp.abs(jnp.asarray(gravity_m_s2)), ball_radius_m=ball_radius_m,
        effective_restitution=effective_restitution,
        max_racket_vz_m_s=max_racket_vz_m_s, lead_s=.180,
        velocity_scale_m_s=.25, weight_per_s=0., dt_s=dt_s)
    _, second_active, required_second, ttc = preimpact_energy_penalty(**common,
        target_apex_z_m=jnp.maximum(jnp.asarray(base_apex_goal_m), SECOND_APEX_RESERVE_M),
        prepare_previous_hits=1)
    _, third_active, required_third, _ = preimpact_energy_penalty(**common,
        target_apex_z_m=third_apex_goal_m, prepare_previous_hits=2)
    y, groups = jnp.asarray(initial_y_m), jnp.asarray(group)
    support = jnp.where(groups == 1, _smooth((y-.035)/.010), 0.)
    second_gain = jnp.where(groups == 1,
        8.*_smooth((y-NEAR_Y_START_M)/(NEAR_Y_FULL_M-NEAR_Y_START_M)), 0.)
    third_gain = near_preparation_multiplier(initial_y_m=y, group=groups,
                                           gain=near_preparation_gain)
    required = jnp.where(counts == 1, required_second, required_third)
    floor = preparation_velocity_floor(ttc_s=ttc, required_vz_m_s=required)
    error = jnp.maximum(floor-jnp.asarray(racket_vz_m_s), 0.)/.25
    shape = jnp.sqrt(1.+error*error)-1.
    second_cost = jnp.where(second_active, -second_weight_per_s*second_gain*dt_s*shape, 0.)
    third_cost = jnp.where(third_active, -third_weight_per_s*third_gain*dt_s*shape, 0.)
    reference, reference_target, reference_retained = reference_preparation_cost(
        ttc_s=ttc, reference_z_lead_m=reference_z_lead_m, dt_s=dt_s)
    active = (second_active | third_active) & (support > 0.)
    reference = jnp.where(active, reference, 0.)
    scale = support*_smooth((.180-ttc)/.030)
    physical = scale*(second_cost+third_cost)
    reference = scale*reference
    return physical+reference, dict(active=active.astype(jnp.float32),
        ttc=jnp.where(active, ttc, 0.), required_vz=jnp.where(active, required, 0.),
        velocity_floor=jnp.where(active, floor, 0.),
        reference_target_m=jnp.where(active, reference_target, 0.),
        reference_lead_m=jnp.where(active, jnp.asarray(reference_z_lead_m), 0.),
        reference_retained=jnp.where(active, reference_retained, 0.),
        physical_cost=physical, reference_cost=reference)


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V304 requires all17 unchanged V303 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = TimingEnvelopeConfig(**vars(stage.cfg), v304_timing_envelope_reward=index >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(stage.notes if index < 11 else
            'V304 reward-only preparation envelope: remove the net unqualified '
            'joint-action label and reprice the two physical preparation costs. '
            'Early reference lead0--35mm,upper60mm,200/s;fade90--60ms. '
            'Physical velocity floor-.4m/s at180ms->0 at100ms->required at0ms, '
            'smooth transitions and30ms onset. Original impact-energy goals, '
            'actual-Y phase weights and broad35--45mm support retained. No '
            'joint-action target,new observation,state or controller. All17 '
            'IDs/domains,all physical gates,72/393/7,control,physics,DR,reset '
            'and numerical1024x128 PPO/BC/KL unchanged. Latest full actor/'
            'critic/Adam/raw history,all32,no rejection,rollback,rewind,reset, '
            'update cap or forced advancement. Native graduation required.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v303_preimpact_timing import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V304 requires full uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0)); total = update-4000
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update or update < 4263
                or payload.get('step') != 3697147904+total*131072
                or int(payload['opt'].t) != 912094+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V304 requires latest V303 S12 full history after>=128 actual r2 updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V304 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V304 rejects another source or from-scratch launch')
