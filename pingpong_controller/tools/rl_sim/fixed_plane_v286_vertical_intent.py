"""Shape early executable upward intent when weak-start preparation is lacking."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v284_reference_prepare import ReferencePreparationConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v286_vertical_intent'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v285_survival_objective'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v286_reference_velocity_weight_per_s'})


@dataclass(frozen=True)
class ReferenceVelocityConfig(ReferencePreparationConfig):
    """Default-off reward field; no observation, state or control extension."""
    v286_reference_velocity_weight_per_s: float = 0.


def validate_config(cfg: ReferenceVelocityConfig) -> None:
    from fixed_plane_v284_reference_prepare import validate_config as previous
    previous(cfg)
    weight = cfg.v286_reference_velocity_weight_per_s
    if not math.isfinite(weight) or not 0. <= weight <= 1000.:
        raise ValueError('V286 velocity reward must be finite in [0,1000]/s')
    if weight > 0. and cfg.v284_reference_prepare_weight_per_s <= 0.:
        raise ValueError('V286 requires the preserved weak-start preparation mask')


def velocity_preparation_penalty(*, command_vz_m_s, reference_z_lead_m,
        active, weak_fraction, initial_y_m, weight_per_s, dt_s):
    """Price insufficient executable reference rate; all inputs have shape (N,).

    The existing physical weak-flight mask supplies activation. The new cost
    fades over 15--30 mm reference lead and is zero once the reference is
    prepared. A one-sided 0.35 m/s floor has no extra credit above the floor.
    This is a command preparation cost, never physical hit/survival credit.
    """
    command = jnp.asarray(command_vz_m_s)
    fade = jnp.clip((.030 - jnp.asarray(reference_z_lead_m)) / .015, 0., 1.)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    scale = fraction * fraction * (3. - 2. * fraction)
    weak = jnp.clip(jnp.asarray(weak_fraction), 0., 1.)
    enabled = (jnp.asarray(active, dtype=bool) & (fade > 0.)
               & (scale > 0.) & (weak > 0.) & (weight_per_s > 0.))
    deficit = jnp.maximum(.35 - command, 0.) / .20
    cost = (-weight_per_s * dt_s * scale * weak * fade
            * (jnp.sqrt(1. + deficit * deficit) - 1.))
    return jnp.where(enabled, cost, 0.), dict(
        active=enabled, command_vz_m_s=jnp.where(enabled, command, 0.),
        lead_fade=jnp.where(enabled, fade, 0.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V286 requires the complete17 physical V285 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = ReferenceVelocityConfig(**vars(stage.cfg),
            v286_reference_velocity_weight_per_s=400. if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V286 S12+ reward-only weak-start executable upward reference intent. '
            'Current-pose Jacobian times postprojection dq_ref; one-sided0.35m/s '
            'floor,0.20m/s scale,400/s pseudo-Huber cost within original weak '
            'falling50--180ms TTC/pre-third mask. Fade15--30mm reference-Z lead; '
            'prepared states, first bootstrap, nominal/mirror and late states exact. '
            'No new state, observation, command, physical credit or success gate. '
            'All17 physical lessons/V281 IDs, all previous rewards/gates/DR/control '
            'and numerical1024x128 PPO exact; latest full state/history, all '
            'minibatches, no cap/reset/rewind/rejection or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v285_survival_objective import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V286 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or payload.get('stage_update') != 197 or payload.get('step') != 3198681088
                or int(payload['opt'].t) != 790398
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != 197 or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V286 pins latest V285 S12u197 and exact physical history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V286 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V286 rejects another source or from-scratch launch')
