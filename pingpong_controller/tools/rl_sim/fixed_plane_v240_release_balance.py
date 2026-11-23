"""V240 balances physical startup apex shaping with recovery motion quality."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import jax.numpy as jnp

from fixed_plane_v239_release_apex import (
    ReleaseApexConfig,
    validate_resume_payload as previous_resume,
    validate_training_contract as previous_training,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v240'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v238'
START_STAGE = 'fplc_v240_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v240_s01_workspace_alignment'


@dataclass(frozen=True)
class ReleaseBalanceConfig(ReleaseApexConfig):
    """One additional recovery-only weight on the existing XY event cost."""

    v240_extra_ball_xy_weight: float = 0.


def validate_config(cfg):
    weight = float(cfg.v240_extra_ball_xy_weight)
    if not math.isfinite(weight) or weight < 0.:
        raise ValueError('V240 extra XY weight must be finite and nonnegative')
    if weight > 0. and (not cfg.v225_enabled
            or not math.isfinite(cfg.fixed_plane_v166_physical_ball_xy_weight)
            or cfg.fixed_plane_v166_physical_ball_xy_weight <= 0.):
        raise ValueError('V240 requires the unchanged recovery cohort and base XY cost')


def extra_xy_penalty(*, existing_cost, group, base_weight: float, extra_weight: float):
    """Reuse the same [N] event, speed and normalization; nominal cost is zero."""
    if (not math.isfinite(base_weight) or base_weight <= 0.
            or not math.isfinite(extra_weight) or extra_weight < 0.):
        raise ValueError('V240 XY weights must be finite with a positive base')
    return jnp.where(jnp.asarray(group) > 0,
                     jnp.asarray(existing_cost) * (extra_weight / base_weight), 0.)


def build_stages(source):
    """Keep V239 apex shaping and every gate; price recovery XY more strongly."""
    if len(source) != 17 or source[7].name != 'fplc_v239_s08_near_y_020mm_bridge':
        raise ValueError('V240 requires the complete V239 candidate')
    result = []
    for stage in source:
        cfg = ReleaseBalanceConfig(**vars(stage.cfg),
            v240_extra_ball_xy_weight=2. if stage.cfg.v225_near_y_max_m > 0. else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v239_', 'fplc_v240_', 1),
            notes='V240 retains V239 every-physical-release startup apex loss '
                  'and all V238 rewards. Only non-nominal cohorts receive an '
                  'extra2 units of the existing normalized ball-XY event cost '
                  '(effective4->6); nominal rewards remain V238-exact. Every '
                  'absolute motion/survival gate, all17 domains,72/393/7, '
                  'control/physics/DR and uncapped1024x128 PPO are preserved. '
                  'Continue latest full V238 S8u70, never roll back an actor.'))
    return result


def validate_training_contract(settings):
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V240 retains nominal absolute validation')
    previous_training(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v239_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage, **kwargs):
    """Reuse unchanged full-state guards through a metadata-only proxy."""
    proxy = dict(payload)
    proxy['args'] = dict(payload.get('args', {}))
    if proxy['args'].get('curriculum_profile') == PROFILE:
        proxy['args']['curriculum_profile'] = 'fixed_plane_limit_cycle_qvel_rmp_v239'
        proxy['stage_name'] = str(proxy.get('stage_name', '')).replace('fplc_v240_', 'fplc_v239_', 1)
    previous_resume(proxy, requested_start_stage=requested_start_stage.replace(
        'fplc_v240_', 'fplc_v239_', 1), **kwargs)
