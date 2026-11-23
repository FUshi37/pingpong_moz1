"""Default-off reward for upward intent consistent with the hard pose set."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v287_startup_center import StartupCenterConfig


PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v288_safe_stroke'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v287_startup_center'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v288_raw_stroke_weight_per_s', 'v288_feasible_stroke_weight_per_s'})


@dataclass(frozen=True)
class SafeStrokeConfig(StartupCenterConfig):
    """Additional reward fields only; no observation, plant or control state."""
    v288_raw_stroke_weight_per_s: float = 0.
    v288_feasible_stroke_weight_per_s: float = 0.


def validate_config(cfg: SafeStrokeConfig) -> None:
    from fixed_plane_v287_startup_center import validate_config as previous
    previous(cfg)
    raw = cfg.v288_raw_stroke_weight_per_s
    feasible = cfg.v288_feasible_stroke_weight_per_s
    if (not math.isfinite(raw) or not 0. <= raw <= 1000.
            or not math.isfinite(feasible) or not 0. <= feasible <= 2000.):
        raise ValueError('V288 stroke reward rates must be finite and bounded')
    if (raw > 0.) != (feasible > 0.):
        raise ValueError('V288 enables raw-intent and feasibility costs together')
    if raw > 0. and (cfg.v286_reference_velocity_weight_per_s <= 0.
                    or not cfg.cartesian_pose_projection_enabled
                    or not cfg.arm_action_limiter
                    or cfg.action_command_mode != 'velocity'
                    or not cfg.recovered_rmp_direct_qvel_target
                    or not cfg.recovered_rmp_bounded_qvel_reference):
        raise ValueError('V288 requires the original executable-stroke and pose contracts')


def safe_stroke_penalties(*, raw_command_vz_m_s,
        projection_correction_norm_rad_s, reference_z_lead_m, active,
        weak_fraction, initial_y_m, raw_weight_per_s, feasible_weight_per_s,
        dt_s):
    """Price raw downward intent and unnecessary projection; arrays are (N,).

    Reuse the original weak-flight mask and15--30mm ready-reference fade.
    The raw rate has the same one-sided0.35m/s floor and0.20m/s scale as the
    executable-rate cue. Projection correction is a pseudo-Huber cost of
    its seven-joint RMS, normalized by1rad/s. Both prices are nonpositive;
    no physical hit, survival or convergence credit is generated.
    """
    from fixed_plane_v286_vertical_intent import velocity_preparation_penalty
    raw, previous = velocity_preparation_penalty(
        command_vz_m_s=raw_command_vz_m_s,
        reference_z_lead_m=reference_z_lead_m, active=active,
        weak_fraction=weak_fraction, initial_y_m=initial_y_m,
        weight_per_s=raw_weight_per_s, dt_s=dt_s)
    enabled = previous['active']
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    scale = fraction * fraction * (3. - 2. * fraction)
    rms = jnp.asarray(projection_correction_norm_rad_s) / jnp.sqrt(7.)
    feasible = (-feasible_weight_per_s * dt_s * scale
                * jnp.clip(jnp.asarray(weak_fraction), 0., 1.)
                * previous['lead_fade'] * (jnp.sqrt(1. + rms * rms) - 1.))
    feasible = jnp.where(enabled, feasible, 0.)
    return raw, feasible, dict(
        active=enabled,
        raw_reference_vz_m_s=previous['command_vz_m_s'],
        correction_norm_rad_s=jnp.where(
            enabled, jnp.asarray(projection_correction_norm_rad_s), 0.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V288 requires the complete17 physical V287 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = SafeStrokeConfig(**vars(stage.cfg),
            v288_raw_stroke_weight_per_s=100. if i >= 11 else 0.,
            v288_feasible_stroke_weight_per_s=500. if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V288 S12+ reward-only upward intent consistent with the existing pose set. '
            'Within the original weak near pre-third mask and15--30mm ready-lead fade, '
            'price joint-limit-clipped PRE-pose-projection reference VZ below0.35m/s '
            'with100/s pseudo-Huber/0.20m/s scale; price existing projection-correction '
            'seven-joint RMS with500/s pseudo-Huber/1rad/s scale. Preserve the existing '
            'POSTprojection cue and every prior reward. Both new prices are nonpositive, '
            'with no physical credit or control override. All17 domains/V281 IDs, gates, '
            '72/393/7,DR/reset/physics/control and numeric1024x128 PPO remain exact. '
            'Latest full actor/critic/Adam/history, all minibatches, no cap/reset/rewind/'
            'rejection or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v287_startup_center import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V288 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or payload.get('stage_update') != 494 or payload.get('step') != 3237609472
                or int(payload['opt'].t) != 799902
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != 494 or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V288 pins latest V287 S12u494 and exact physical history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V288 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V288 rejects another source or from-scratch launch')
