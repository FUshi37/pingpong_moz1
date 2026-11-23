"""V237: bounded weak-third-release cost without rewarding overpowered hits."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import jax.numpy as jnp

from fixed_plane_v236_launch_first import LaunchFirstConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v237'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v236'
RETENTION_STAGE = 'fplc_v237_s01_workspace_alignment'


@dataclass(frozen=True)
class ReleaseMarginConfig(LaunchFirstConfig):
    """One additional reward coefficient; no physical/observation fields."""

    v237_release_shortfall_weight: float = 0.


def release_shortfall(*, active, score, group, weight: float):
    """Charge [N] non-nominal true release events for bounded V122 shortfall."""
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('Release shortfall weight must be finite and nonnegative')
    eligible = jnp.asarray(active, dtype=bool) & (jnp.asarray(group) > 0)
    shortfall = 1. - jnp.clip(jnp.asarray(score), 0., 1.)
    return jnp.where(eligible, -float(weight) * shortfall, 0.)


def build_stages(source):
    """Keep V236 phase shaping and every gate; price weak third launches."""
    if len(source) != 17 or not source[7].name.startswith('fplc_v236_s08_'):
        raise ValueError('V237 requires the complete V236 curriculum')
    result = []
    for stage in source:
        weight = 30. if stage.cfg.v225_near_y_max_m > 0. else 0.
        if weight and stage.cfg.fixed_plane_v122_third_release_vz_weight <= 0.:
            raise ValueError('V237 requires the preserved V122 release score')
        cfg = ReleaseMarginConfig(**vars(stage.cfg), v237_release_shortfall_weight=weight)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v236_', 'fplc_v237_', 1),
            notes='V237 retains V236 launch-first anchoring. At the existing '
                  'first true third-contact release, non-nominal episodes '
                  'receive -30*(1-clipped_v122_score). The existing2.5m/s '
                  'release target,15-weight credit, hit confirmation, height '
                  'and motion gates remain unchanged. The added loss is zero '
                  'at/above target, zero for nominal episodes and never '
                  'counts or rejects a sample. Full actor/critic/Adam continuation; '
                  'unchanged1024x128 PPO and no update cap.'))
    return result


def validate_training_contract(settings):
    from fixed_plane_v236_launch_first import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V237 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v236_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic, allow_obs_dim_migration):
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or payload.get('args',{}).get('max_stage_updates') != -1):
        raise ValueError('V237 requires full uncapped72/393/7 continuation')
    source = payload.get('args',{}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name','')).replace('fplc_v236_','fplc_v237_',1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index',0)) < 8):
            raise ValueError('V237 starts fresh reward statistics in the current recovery lesson')
    elif source == PROFILE:
        if not resume_curriculum_state or requested_start_stage != payload.get('stage_name'):
            raise ValueError('V237 own continuation must restore exact history')
    else:
        raise ValueError('V237 rejects another source')
