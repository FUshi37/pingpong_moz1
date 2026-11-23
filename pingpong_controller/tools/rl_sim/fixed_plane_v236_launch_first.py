"""V236: launch-first anchor shaping; dynamics and absolute gates stay V235."""

from dataclasses import dataclass, replace
from typing import Any

import jax.numpy as jnp

from fixed_plane_v234_reward_only import RewardOnlyConfig
from fixed_plane_v235_absolute_gates import relaxed_hit_admission, retention_stage

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v236'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v235'
START_STAGE = 'fplc_v236_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v236_s01_workspace_alignment'


@dataclass(frozen=True)
class LaunchFirstConfig(RewardOnlyConfig):
    """Reward-only fields; no additional observation or physical state."""

    v236_launch_first: bool = False
    v236_return_start_hits: int = 3
    v236_return_full_hits: int = 6


def build_stages(source):
    if len(source) != 17 or source[7].name != 'fplc_v235_s08_near_y_020mm_bridge':
        raise ValueError('V236 requires the complete V235 course')
    return [replace(stage,
        name=stage.name.replace('fplc_v235_', 'fplc_v236_', 1),
        cfg=replace(LaunchFirstConfig(**vars(stage.cfg)), v236_launch_first=True),
        notes='V236 preserves all V235 domains, gates, motion costs, control, '
              'physics, observation and PPO. Non-nominal startup anchor cost '
              'permits the initial offset without a reset-time return deadline, '
              'but penalizes further outward motion. After three confirmed '
              'hits the late center allowance shrinks with smoothstep and '
              'reaches the exact old late reward at six hits. Nominal reward '
              'is numerically unchanged. No hit reinterpretation or filtering.')
            for stage in source]


def anchor_rewards(*, original_early, original_late, distance_m, initial_y_m,
                   group, hit_count, cfg: Any, dt_s: float):
    """Allow non-nominal [N] launch offsets, then restore legacy centering."""
    counts = jnp.asarray(hit_count)
    non_nominal = jnp.asarray(group) > 0
    offset = jnp.abs(jnp.asarray(initial_y_m))
    distance = jnp.asarray(distance_m)
    weight = jnp.where(counts == 2,
                      float(cfg.early_racket_xy_anchor_transition_weight),
                      float(cfg.early_racket_xy_anchor_penalty_weight))
    excess = jnp.maximum(distance - offset
        - float(cfg.early_racket_xy_anchor_deadband_m), 0.) / float(cfg.early_racket_xy_anchor_scale_m)
    shape = jnp.where(excess > 2., 4. + 4. * (excess - 2.), excess ** 2)
    early = jnp.where(counts <= int(cfg.early_racket_xy_anchor_hit_count),
                      -weight * shape * dt_s, 0.)
    start, full = int(cfg.v236_return_start_hits), int(cfg.v236_return_full_hits)
    if not 3 <= start < full:
        raise ValueError('V236 requires a finite return ramp after three hits')
    fraction = jnp.clip((counts - start) / float(full - start), 0., 1.)
    smooth = fraction * fraction * (3. - 2. * fraction)
    allowance = offset * (1. - smooth)
    violation = jnp.maximum(distance - allowance - .005, 0.) / .020
    late = jnp.where(counts >= 3,
        -float(cfg.fixed_plane_v134_late_xy_anchor_penalty_weight) * dt_s
        * (jnp.sqrt(1. + violation ** 2) - 1.), 0.)
    # Select the old tensors at the completed ramp to preserve exact parity.
    late = jnp.where(counts >= full, original_late, late)
    return (jnp.where(non_nominal, early, original_early),
            jnp.where(non_nominal, late, original_late))


def validate_training_contract(settings):
    from fixed_plane_v235_absolute_gates import validate_training_contract as old
    from types import SimpleNamespace
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V236 retains the nominal absolute-gate domain')
    old(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v235_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic, allow_obs_dim_migration):
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V236 requires full uncapped 72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != 'fplc_v235_s08_near_y_020mm_bridge'
                or requested_start_stage != START_STAGE or resume_curriculum_state):
            raise ValueError('V236 changes reward at current S8 with fresh convergence history')
    elif source == PROFILE:
        if not resume_curriculum_state or requested_start_stage != payload.get('stage_name'):
            raise ValueError('V236 own continuation restores exact stage/history')
    else:
        raise ValueError('V236 rejects a different parent or control stack')
