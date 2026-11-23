"""V251 allows startup energy reserve instead of exact intermediate peaks."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from fixed_plane_v249_bootstrap_energy import BootstrapEnergyConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v251'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v250'
START_STAGE = 'fplc_v251_s09_near_y_035mm_bridge'
RETENTION_STAGE = 'fplc_v251_s01_workspace_alignment'
STARTUP_APEX_CEILING_M = 1.45


@dataclass(frozen=True)
class EnergyBandConfig(BootstrapEnergyConfig):
    """Default-off reward mode; no state, observation or physical fields."""

    v251_startup_energy_band_enabled: bool = False


def validate_config(cfg: EnergyBandConfig) -> None:
    from fixed_plane_v249_bootstrap_energy import validate_config as previous
    previous(cfg)
    if cfg.v251_startup_energy_band_enabled and (
            not cfg.v225_enabled or cfg.v241_physical_apex_huber_weight <= 0.
            or cfg.v245_second_apex_reward_margin_m != 0.
            or cfg.v245_third_apex_reward_margin_m != 0.
            or tuple(cfg.fixed_plane_limit_cycle_apex_schedule_z_m) != (1.14, 1.249, 1.4, 1.425)):
        raise ValueError('V251 requires original physical goals and no startup point margins')


def first_energy_band_credit(
    *, counted_hit: jax.Array, previous_hit_count: jax.Array,
    apex_z_m: jax.Array, lower_z_m: jax.Array, upper_z_m: float,
    weight: float, sigma_m: float,
) -> jax.Array:
    """Credit a confirmed first hit throughout [lower,upper], excluding support."""
    if (not math.isfinite(weight) or weight < 0. or not math.isfinite(sigma_m)
            or sigma_m <= 0. or not math.isfinite(upper_z_m)):
        raise ValueError('Invalid first energy-band reward parameters')
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    error = (jnp.maximum(jnp.asarray(lower_z_m)-apex, 0.)
             + jnp.maximum(apex-upper_z_m, 0.))
    active = jnp.asarray(counted_hit, dtype=bool) & (jnp.asarray(previous_hit_count) == 0)
    return jnp.where(active, weight*jnp.exp(-.5*(error/sigma_m)**2), 0.)


def startup_energy_band_cost(
    *, existing_cost: jax.Array, existing_error: jax.Array,
    release_event: jax.Array, goal_hit_count: jax.Array,
    apex_z_m: jax.Array, base_goal_z_m: jax.Array,
    upper_z_m: float, weight_per_m2: float, delta_m: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Replace phase1/2 release costs; preserve all other costs exactly."""
    if (not math.isfinite(weight_per_m2) or weight_per_m2 < 0.
            or not math.isfinite(delta_m) or delta_m <= 0.
            or not math.isfinite(upper_z_m)):
        raise ValueError('Invalid startup energy-band reward parameters')
    phase = jnp.asarray(goal_hit_count)
    active = jnp.asarray(release_event, dtype=bool) & (phase >= 1) & (phase <= 2)
    apex = jnp.asarray(apex_z_m, dtype=jnp.float32)
    error = (jnp.maximum(jnp.asarray(base_goal_z_m)-apex, 0.)
             + jnp.maximum(apex-upper_z_m, 0.))
    shape = jnp.where(error <= delta_m, error**2, 2.*delta_m*error-delta_m**2)
    return (jnp.where(active, -weight_per_m2*shape, existing_cost),
            jnp.where(active, error, existing_error), active)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep original startup floors and allow safe extra bounce energy."""
    if len(source) != 17 or source[8].name != 'fplc_v250_s09_near_y_035mm_bridge':
        raise ValueError('V251 requires all17 V250 lessons')
    result = []
    for stage in source:
        cfg = replace(EnergyBandConfig(**vars(stage.cfg)),
            v251_startup_energy_band_enabled=True,
            v245_second_apex_reward_margin_m=0.,
            v245_third_apex_reward_margin_m=0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v250_', 'fplc_v251_', 1),
            notes='V251 startup reward uses safe energy bands: confirmedfirst '
                  'credit1.16..1.45m; actual phase1/2 release Huber distance '
                  'to1.249..1.45/1.4..1.45m, including weak retries. Remove '
                  '15/25mm startup point margins. Keep physical goals and '
                  'all hit accounting/gates, fourth-plus symmetric1.425m '
                  'Huber40000,weak600,lateband240,lateXY28,72/393/7, '
                  'all17 domains/control/physics/DR/resets and original '
                  'uncapped1024x128 PPO. Full latest250actor/critic/Adam; '
                  'no sample rejection, rollback, new action or timer.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v250_bootstrap_credit import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V251 retains original absolute nominal validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v250_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V251 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v250_', 'fplc_v251_', 1)
        if resume_curriculum_state or requested_start_stage != expected or int(payload.get('stage_index', 0)) < 9:
            raise ValueError('V251 starts new reward statistics in the current lesson')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V251 own continuation restores exact history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V251 rejects another source')
