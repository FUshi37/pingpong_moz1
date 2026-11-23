"""V242 removes conflicting startup VZ credits and strengthens late XY cost."""

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp

from fixed_plane_v241_physical_outcome import PhysicalOutcomeConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v242'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v241'
START_STAGE = 'fplc_v242_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v242_s01_workspace_alignment'


@dataclass(frozen=True)
class StartupObjectivesConfig(PhysicalOutcomeConfig):
    """Default-off reward correction; no new physical or observed state."""

    v242_startup_vz_outcome_only: bool = False


def validate_config(cfg: StartupObjectivesConfig) -> None:
    """Require physical outcome supervision before removing velocity credits."""
    if not isinstance(cfg.v242_startup_vz_outcome_only, bool):
        raise ValueError('V242 startup reward switch must be boolean')
    if cfg.v242_startup_vz_outcome_only and (
            not cfg.v225_enabled or cfg.v241_physical_apex_huber_weight <= 0.
            or cfg.fixed_plane_v124_edge_vz_reward_weight != 0.):
        raise ValueError(
            'V242 requires physical outcome loss and no constant VZ credit')


def startup_stroke_correction(
    *, existing_credit: jax.Array, previous_hit_count: jax.Array, enabled: bool,
) -> jax.Array:
    """Cancel only the existing [N] velocity-reference credit before hit three."""
    credit = jnp.asarray(existing_credit)
    if not enabled:
        return jnp.zeros_like(credit)
    count = jnp.asarray(previous_hit_count)
    return jnp.where((count >= 1) & (count <= 2), -credit, 0.)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve every domain/gate and all late vertical shaping exactly."""
    if (len(source) != 17
            or source[7].name != 'fplc_v241_s08_near_y_020mm_bridge'):
        raise ValueError('V242 requires the complete V241 curriculum')
    result = []
    for stage in source:
        cfg = replace(StartupObjectivesConfig(**vars(stage.cfg)),
            v242_startup_vz_outcome_only=True,
            fixed_plane_v124_edge_vz_reward_weight=0.,
            v241_late_ball_xy_weight=4.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v241_', 'fplc_v242_', 1),
            notes='V242 cancels only the existing smooth-stroke VZ credit at '
                  'previous confirmed counts1/2 and disables the once-only '
                  'constant0.8m/s third-edge credit. V241 physical phase-apex '
                  'loss, V122 release credit, contact-Z cost and all late '
                  'vertical rewards remain. Existing late XY extra2->4, '
                  'giving weights8/10 after hit3. All17 domains, absolute '
                  'gates,72/393/7,physics/control/DR and uncapped1024x128 PPO '
                  'are unchanged. Continue latest full V241 state.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v241_physical_outcome import (
        validate_training_contract as previous,
    )
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V242 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v241_s01_workspace_alignment'}))


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
        raise ValueError('V242 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace(
            'fplc_v241_', 'fplc_v242_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError(
                'V242 starts fresh reward statistics in the current lesson')
    elif source == PROFILE:
        if (not resume_curriculum_state
                or requested_start_stage != payload.get('stage_name')):
            raise ValueError('V242 own continuation must restore exact history')
    else:
        raise ValueError('V242 rejects another source')
