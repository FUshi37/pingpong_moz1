"""Allow bounded startup contact offsets without weakening actual quality gates."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v286_vertical_intent import ReferenceVelocityConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v287_startup_center'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v286_vertical_intent'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v287_startup_contact_center_radius_m'})


@dataclass(frozen=True)
class StartupCenterConfig(ReferenceVelocityConfig):
    """Default-off local startup reward deadband; no control/state extension."""
    v287_startup_contact_center_radius_m: float = 0.


def validate_config(cfg: StartupCenterConfig) -> None:
    from fixed_plane_v286_vertical_intent import validate_config as previous
    previous(cfg)
    radius = cfg.v287_startup_contact_center_radius_m
    if not math.isfinite(radius) or radius < 0.:
        raise ValueError('V287 center deadband must be finite and nonnegative')
    if radius > 0.:
        if not cfg.hit_contact_center_excess_radius_m <= radius <= .020:
            raise ValueError('V287 startup center radius must be in[legacy radius,20mm]')
        if (not cfg.v225_enabled or not cfg.fixed_contact_center_penalty_all_physical_outcomes
                or not cfg.fixed_contact_include_hit_contact_center_excess_penalty
                or not cfg.fixed_contact_include_task_aligned_auxiliary_reward
                or not cfg.fixed_contact_hit_center_penalty_outside_budget
                or cfg.hit_contact_center_excess_penalty_weight <= 0.
                or cfg.hit_contact_center_excess_sigma_m <= 0.):
            raise ValueError('V287 requires the original fully charged physical center penalty')


def startup_center_correction(*, old_center_penalty, contact_distance_m,
        physical_center_event, previous_hits, group, initial_y_m,
        new_radius_m, weight, sigma_m):
    """Reprice the same real physical event; all input arrays have shape(N,).

    The combined old penalty plus this correction remains nonpositive. Only
    post-first/pre-third near-side acquisition receives the12mm deadband.
    Physical counts, measured distances and the7mm aggregate quality gate
    stay unchanged. The old penalty remains as a separate auditable ledger.
    """
    old = jnp.asarray(old_center_penalty)
    distance = jnp.asarray(contact_distance_m)
    f = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    scale = f * f * (3. - 2. * f)
    active = (jnp.asarray(physical_center_event, dtype=bool)
              & (jnp.asarray(previous_hits) >= 1) & (jnp.asarray(previous_hits) <= 2)
              & (jnp.asarray(group) == 1) & (scale > 0.) & (new_radius_m > 0.))
    new_penalty = -weight * jnp.square(jnp.maximum(distance - new_radius_m, 0.) / sigma_m)
    correction = jnp.where(active, scale * (new_penalty - old), 0.)
    return correction, dict(active=active,
        distance_m=jnp.where(active, distance, 0.),
        effective_penalty=jnp.where(active, old + correction, 0.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V287 requires the complete17 physical V286 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = StartupCenterConfig(**vars(stage.cfg),
            v287_startup_contact_center_radius_m=.012 if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V287 S12+ reward-only local startup contact-center deadband12mm. '
            'Reprice original fully charged physical center event only for near-side '
            'post-first/pre-third acquisition, Y50--65mm smooth activation. The '
            'combined center penalty stays nonpositive and remains quadratic outside '
            '12mm. First bootstrap, nominal/mirror and all post-third rewards exact. '
            'All prior energy/preparation/actual survival rewards,17 physical lessons/'
            'V281 IDs,7mm aggregate center/other quality/stability/nominal gates, '
            '72/393/7,DR/control/physics and numeric1024x128 PPO remain unchanged. '
            'Latest full state/history, all minibatches, no cap/reset/rewind/rejection '
            'or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v286_vertical_intent import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V287 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or payload.get('stage_update') != 293 or payload.get('step') != 3211264000
                or int(payload['opt'].t) != 793470
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != 293 or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V287 pins latest V286 S12u293 and exact physical history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V287 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V287 rejects another source or from-scratch launch')
