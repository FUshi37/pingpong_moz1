"""V239 prices every physical release until the third startup hit confirms."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import jax.numpy as jnp

from fixed_plane_v237_release_margin import ReleaseMarginConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v239'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v238'
START_STAGE = 'fplc_v239_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v239_s01_workspace_alignment'


@dataclass(frozen=True)
class ReleaseApexConfig(ReleaseMarginConfig):
    """Reward coefficients only; no physical state or observation additions."""

    v239_release_apex_weight: float = 0.
    v239_release_apex_scale_m: float = .05


def validate_config(cfg):
    weight = float(cfg.v239_release_apex_weight)
    scale = float(cfg.v239_release_apex_scale_m)
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('V239 release-apex weight must be finite and nonnegative')
    if not math.isfinite(scale) or scale <= 0.:
        raise ValueError('V239 release-apex scale must be finite and positive')
    if weight > 0. and not cfg.v225_enabled:
        raise ValueError('V239 requires the preserved recovery population')


def release_apex_penalty(*, release_event, previous_hit_count, group,
                         predicted_apex_z_m, target_apex_z_m,
                         weight: float, scale_m: float):
    """Price every real [N] startup release, including repeated weak attempts.

    The existing predicted apex includes the current ball height and gravity.
    No pending-hit acceptance, one-time claim or successful confirmation is
    required. The smooth bounded cost is zero at/above the existing apex goal.
    """
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('Release-apex weight must be finite and nonnegative')
    if not math.isfinite(scale_m) or scale_m <= 0.:
        raise ValueError('Release-apex scale must be finite and positive')
    active = (jnp.asarray(release_event, dtype=bool)
              & (jnp.asarray(previous_hit_count) == 2)
              & (jnp.asarray(group) > 0))
    shortfall = jnp.maximum(jnp.asarray(target_apex_z_m)
                            - jnp.asarray(predicted_apex_z_m), 0.)
    squared = shortfall * shortfall
    cost = -float(weight) * squared / (squared + float(scale_m) ** 2)
    return jnp.where(active, cost, 0.), active, jnp.where(active, shortfall, 0.)


def build_stages(source):
    """Preserve all V238 domains, gates and rewards; add one outcome cost."""
    if len(source) != 17 or source[7].name != 'fplc_v238_s08_near_y_020mm_bridge':
        raise ValueError('V239 requires the complete V238 curriculum')
    result = []
    for stage in source:
        cfg = ReleaseApexConfig(**vars(stage.cfg),
            v239_release_apex_weight=80. if stage.cfg.v225_near_y_max_m > 0. else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v238_', 'fplc_v239_', 1),
            notes='V239 adds only a bounded physical-release apex-shortfall loss '
                  'until the third startup hit confirms, in non-nominal lanes. '
                  'Every physical release is eligible, including weak retries '
                  'after a one-time credit has already been claimed. Use the '
                  'existing phase apex goal and predicted apex, weight80 and '
                  'scale0.05m. No extra reward above target. All V238 rewards, '
                  '17 domains and absolute gates, control, physics, DR, '
                  '72/393/7 and uncapped1024x128 PPO remain unchanged.'))
    return result


def validate_training_contract(settings):
    from fixed_plane_v238_release_motion import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V239 retains the unchanged nominal absolute domain')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v238_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic, allow_obs_dim_migration):
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V239 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v238_', 'fplc_v239_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V239 starts fresh reward statistics in the current recovery lesson')
    elif source == PROFILE:
        if not resume_curriculum_state or requested_start_stage != payload.get('stage_name'):
            raise ValueError('V239 own continuation must restore exact history')
    else:
        raise ValueError('V239 rejects another source')
