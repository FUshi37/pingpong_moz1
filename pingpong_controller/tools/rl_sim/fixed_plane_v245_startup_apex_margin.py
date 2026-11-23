"""V245 adds startup reward headroom while keeping physical goals fixed."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from fixed_plane_v242_startup_objectives import StartupObjectivesConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v245'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v244'
START_STAGE = 'fplc_v245_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v245_s01_workspace_alignment'


@dataclass(frozen=True)
class StartupApexMarginConfig(StartupObjectivesConfig):
    """Default-off reward fields; never used for confirmation or observation."""

    v245_second_apex_reward_margin_m: float = 0.
    v245_third_apex_reward_margin_m: float = 0.


def _validate_margins(second_margin_m: float, third_margin_m: float) -> None:
    if any(not math.isfinite(value) or not 0. <= value <= .025
           for value in (second_margin_m, third_margin_m)):
        raise ValueError('Startup reward margins must be finite in[0,.025]m')


def validate_config(cfg: StartupApexMarginConfig) -> None:
    second = float(cfg.v245_second_apex_reward_margin_m)
    third = float(cfg.v245_third_apex_reward_margin_m)
    _validate_margins(second, third)
    if second or third:
        schedule = cfg.fixed_plane_limit_cycle_apex_schedule_z_m
        if (not cfg.v225_enabled or cfg.v241_physical_apex_huber_weight <= 0.
                or not cfg.v242_startup_vz_outcome_only or len(schedule) != 4
                or schedule[2] + third > schedule[3] + 1e-9
                or schedule[1] + second >= schedule[2] + third):
            raise ValueError('Startup reward margin needs preserved outcome task and safe schedule')


def reward_apex_target(
    base_goal_m: jax.Array, previous_hit_count: jax.Array, *,
    second_margin_m: float, third_margin_m: float,
) -> jax.Array:
    """Offset only physical-release phases1/2 in the reward calculation."""
    _validate_margins(second_margin_m, third_margin_m)
    if second_margin_m == third_margin_m == 0.:
        return base_goal_m
    count = jnp.asarray(previous_hit_count)
    base = jnp.asarray(base_goal_m)
    return jnp.where(count == 1, base + second_margin_m,
                     jnp.where(count == 2, base + third_margin_m, base))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Add reward headroom; preserve all existing configuration values."""
    if (len(source) != 17
            or source[7].name != 'fplc_v244_s08_near_y_020mm_bridge'):
        raise ValueError('V245 requires the complete V244 curriculum')
    result = []
    for stage in source:
        cfg = replace(StartupApexMarginConfig(**vars(stage.cfg)),
            v245_second_apex_reward_margin_m=.015,
            v245_third_apex_reward_margin_m=.025)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v244_', 'fplc_v245_', 1),
            notes='V245 adds only physical-release reward margins: +15mm '
                  'at phase1 and +25mm at phase2, giving1.264/1.425m reward '
                  'goals. Static support and all late reward goals stay '
                  'exact. Keep physical goals1.14/1.249/1.4/1.425 and every '
                  'confirmation/absolute gate, all other rewards,17 domains, '
                  '72/393/7,physics/control/DR/reset and every numeric '
                  'uncapped1024x128 PPO setting. Full latest actor/critic/Adam; '
                  'no rollback or rejected batches.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v244_startup_tracking import (
        validate_training_contract as previous,
    )
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V245 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v244_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    dimensions = tuple(payload.get(k) for k in (
        'obs_dim', 'critic_obs_dim', 'act_dim'))
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or dimensions != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V245 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace(
            'fplc_v244_', 'fplc_v245_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V245 starts new reward statistics in current lesson')
    elif source == PROFILE:
        if (not resume_curriculum_state
                or requested_start_stage != payload.get('stage_name')):
            raise ValueError('V245 own continuation must restore exact history')
    else:
        raise ValueError('V245 rejects another source')
