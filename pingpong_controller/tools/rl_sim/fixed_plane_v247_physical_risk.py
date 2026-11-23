"""V247 prices startup release weakness and both resolved late apex tails."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from fixed_plane_v246_reward_consolidation import RewardConsolidationConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v247'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v246'
START_STAGE = 'fplc_v247_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v247_s01_workspace_alignment'


@dataclass(frozen=True)
class PhysicalRiskConfig(RewardConsolidationConfig):
    """Default-zero outcome costs; no observed or physical state additions."""

    v247_startup_physical_weak_penalty_weight: float = 0.
    v247_late_physical_band_penalty_weight: float = 0.


def _validate_weights(startup_weight: float, late_weight: float) -> None:
    if any(not math.isfinite(w) or not 0. <= w <= 1000.
           for w in (startup_weight, late_weight)):
        raise ValueError('V247 physical risk weights must be finite in [0,1000]')


def validate_config(cfg: PhysicalRiskConfig) -> None:
    from fixed_plane_v245_startup_apex_margin import validate_config as previous
    previous(cfg)
    weights = (float(cfg.v247_startup_physical_weak_penalty_weight),
               float(cfg.v247_late_physical_band_penalty_weight))
    _validate_weights(*weights)
    if any(w > 0. for w in weights) and (
            not cfg.v225_enabled or not cfg.fixed_plane_v92_signed_height_enabled
            or cfg.v241_physical_apex_huber_weight <= 0.
            or cfg.fixed_plane_v211_late_weak_penalty_weight != 0.):
        raise ValueError('V247 requires physical release supervision and no duplicate V211 cost')


def physical_outcome_risk_penalties(
    *, release_event: jax.Array, release_goal_hit_count: jax.Array,
    physical_apex_z_m: jax.Array, late_resolved_event: jax.Array,
    quality_apex_floor_z_m: jax.Array, startup_weight: float, late_weight: float,
) -> tuple[jax.Array, jax.Array]:
    """Return two [N] costs on real release outcomes, including weak retries.

    Phase zero is excluded from startup release costs. The separately supplied
    late resolved mask must be the existing fourth-plus gate-event mask;
    late costs use its inclusive 1.40--1.45 m band. Release and resolution are
    different clocks. Neither hit classification nor gate semantics change.
    """
    _validate_weights(float(startup_weight), float(late_weight))
    event = jnp.asarray(release_event, dtype=bool)
    phase = jnp.asarray(release_goal_hit_count)
    apex = jnp.asarray(physical_apex_z_m)
    floor = jnp.asarray(quality_apex_floor_z_m)
    late_event = jnp.asarray(late_resolved_event, dtype=bool)
    if event.ndim != 1 or any(x.shape != event.shape for x in (phase, apex, floor, late_event)):
        raise ValueError('V247 release events, phases, apex and floors require matching [N] shapes')
    startup = event & (phase >= 1) & (phase <= 2) & (apex < floor)
    late = late_event & ((apex < 1.4) | (apex > 1.45))
    return (jnp.where(startup, -float(startup_weight), 0.),
            jnp.where(late, -float(late_weight), 0.))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Replace the late recognition cost with physical outcome category costs."""
    if len(source) != 17 or source[7].name != 'fplc_v246_s08_near_y_020mm_bridge':
        raise ValueError('V247 requires the complete V246 curriculum')
    result = []
    for stage in source:
        cfg = replace(PhysicalRiskConfig(**vars(stage.cfg)),
            fixed_plane_v211_late_weak_penalty_weight=0.,
            v241_late_ball_xy_weight=20.,
            v247_startup_physical_weak_penalty_weight=200.,
            v247_late_physical_band_penalty_weight=80.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v246_', 'fplc_v247_', 1),
            notes='V247 changes only outcome reward: native late recognition '
                  'cost40->0; actual release phase1/2 weak-entry cost200 '
                  'below the unchanged scheduled quality floor, and symmetric '
                  'late resolved gate-event outside1.4..1.45m cost80. Static phase0 '
                  'excluded; no hit/gate/sample reinterpretation. Increase '
                  'existing late ball-XY extra12->20 only after hit3. Keep all '
                  'other V246 rewards,17 domains/gates,conv_len1,72/393/7, '
                  'control/physics/DR/reset and original uncapped1024x128 PPO. '
                  'Latest full actor/critic/Adam; fresh current reward window.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v246_reward_consolidation import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V247 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v246_s01_workspace_alignment'}))


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V247 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v246_', 'fplc_v247_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V247 starts new reward statistics in the current lesson')
    elif source == PROFILE:
        if not resume_curriculum_state or requested_start_stage != payload.get('stage_name'):
            raise ValueError('V247 own continuation must restore exact history')
    else:
        raise ValueError('V247 rejects another source')
