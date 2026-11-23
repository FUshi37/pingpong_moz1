"""Default-off complete raw-twist reward for weak startup coordination."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v288_safe_stroke import SafeStrokeConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v289_coordinated_stroke'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v288_safe_stroke'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v289_coordinated_stroke_weight_per_s'})


@dataclass(frozen=True)
class CoordinatedStrokeConfig(SafeStrokeConfig):
    """A reward field only; no new control or observation state."""
    v289_coordinated_stroke_weight_per_s: float = 0.


def validate_config(cfg: CoordinatedStrokeConfig) -> None:
    from fixed_plane_v288_safe_stroke import validate_config as previous
    previous(cfg)
    weight = cfg.v289_coordinated_stroke_weight_per_s
    if not math.isfinite(weight) or not 0. <= weight <= 1000.:
        raise ValueError('V289 coordinated reward rate must be finite in[0,1000]/s')
    if weight > 0. and cfg.v288_raw_stroke_weight_per_s <= 0.:
        raise ValueError('V289 requires the original raw and feasible stroke contract')


def coordinated_stroke_penalty(*, command_twist, reference_z_lead_m,
        active, weak_fraction, initial_y_m, weight_per_s, dt_s):
    """Price raw world-frame [vx,vy,vz,wx,wy,wz] of shape(N,6).

    The reward target alone is[0,0,>=0.5m/s,0,0,0rad/s], with0.20m/s linear
    and0.50rad/s angular scales. Reuse the exact V286 original weak-near
    eligibility and15--30mm ready-lead fade. No reference, action or plant is
    changed. This cost is nonpositive and creates no physical event credit.
    """
    from fixed_plane_v286_vertical_intent import velocity_preparation_penalty
    twist = jnp.asarray(command_twist)
    _, previous = velocity_preparation_penalty(
        command_vz_m_s=twist[..., 2], reference_z_lead_m=reference_z_lead_m,
        active=active, weak_fraction=weak_fraction, initial_y_m=initial_y_m,
        weight_per_s=weight_per_s, dt_s=dt_s)
    target = jnp.asarray([0., 0., .5, 0., 0., 0.], dtype=twist.dtype)
    scales = jnp.asarray([.20, .20, .20, .50, .50, .50], dtype=twist.dtype)
    error = (twist - target) / scales
    error = error.at[..., 2].set(jnp.minimum(error[..., 2], 0.))
    squared_rms = jnp.mean(jnp.square(error), axis=-1)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    support = fraction * fraction * (3. - 2. * fraction)
    cost = (-weight_per_s * dt_s * support
            * jnp.clip(jnp.asarray(weak_fraction), 0., 1.)
            * previous['lead_fade'] * (jnp.sqrt(1. + squared_rms) - 1.))
    enabled = previous['active']
    return jnp.where(enabled, cost, 0.), dict(active=enabled,
        raw_xy_m_s=jnp.where(enabled, jnp.linalg.norm(twist[..., :2], axis=-1), 0.),
        raw_vz_m_s=jnp.where(enabled, twist[..., 2], 0.),
        raw_omega_rad_s=jnp.where(enabled, jnp.linalg.norm(twist[..., 3:], axis=-1), 0.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V289 requires the complete17 physical V288 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = CoordinatedStrokeConfig(**vars(stage.cfg),
            v289_coordinated_stroke_weight_per_s=150. if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V289 reward-only weak startup coordinated raw twist target. Inside '
            'the original weak-near mask/ready-lead fade, price raw world twist '
            'distance to[0,0,>=.5,0,0,0] at150/s with linear.20m/s/angular.50rad/s '
            'scales and six-component RMS pseudo-Huber. Analytic target is only '
            'a reward; never a controller, action override or physical hit credit. '
            'All prior rewards,17 V281 physical domains/gates,72/393/7,control/'
            'physics/DR/reset and numeric1024x128 PPO remain exact. No cap, '
            'reset, rewind, rejection, rollback or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v288_safe_stroke import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V289 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 622
                or payload.get('step') != 3237609472 + (update - 494) * 131072
                or int(payload['opt'].t) != 799902 + (update - 494) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V289 requires latest complete V288 S12 and at least128 actual new updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V289 requires exact own physical history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V289 rejects another source or from-scratch launch')
