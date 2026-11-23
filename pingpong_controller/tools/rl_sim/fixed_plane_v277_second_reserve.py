"""Bounded actual second-impact reserve credit; preserve all physical lessons."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp
from fixed_plane_v276_second_launch import SecondPreparationConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v277_second_reserve'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v276_second_launch'
START_STAGE = 'fplc_v277_s11_near_y_080mm'
RETENTION_STAGE = 'fplc_v277_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v277_second_reserve_credit_weight'})


@dataclass(frozen=True)
class SecondReserveConfig(SecondPreparationConfig):
    """Reward-only field, default disabled for every historical profile."""
    v277_second_reserve_credit_weight: float = 0.


def validate_config(cfg: SecondReserveConfig) -> None:
    from fixed_plane_v276_second_launch import validate_config as previous
    previous(cfg)
    weight = cfg.v277_second_reserve_credit_weight
    if not math.isfinite(weight) or not 0. <= weight <= 320.:
        raise ValueError('Invalid V277 bounded second reserve credit')


def second_reserve_credit(*, counted_hit, previous_hit_count, initial_y_m,
                          group, apex_z_m, weight):
    """Credit the actual confirmed second launch once, within a safe height band.

    Inputs have shape (N,). True reset offset is privileged reward input only.
    A flat 1.30--1.40 m reserve band avoids a precise early apex deadline;
    Gaussian tails decay outside it. Unconfirmed retries receive no credit.
    """
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    smooth = fraction * fraction * (3. - 2. * fraction)
    error = jnp.maximum(1.30 - jnp.asarray(apex_z_m), 0.) + jnp.maximum(jnp.asarray(apex_z_m) - 1.40, 0.)
    eligible = jnp.asarray(counted_hit) & (jnp.asarray(previous_hit_count) == 1) & (jnp.asarray(group) == 1)
    return jnp.where(eligible, weight * smooth * jnp.exp(-.5 * jnp.square(error / .030)), 0.)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Retain all17 domains/gates, changing only S11+ startup rewards."""
    if len(source) != 17 or source[10].name != 'fplc_v276_s11_near_y_080mm':
        raise ValueError('V277 requires the complete V276 course')
    result = []
    for index, stage in enumerate(source):
        cfg = SecondReserveConfig(**vars(stage.cfg))
        if index >= 10:
            cfg = replace(cfg, v277_second_reserve_credit_weight=320., v255_near_preparation_gain=3.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, name=stage.name.replace('fplc_v276_', 'fplc_v277_', 1),
            notes=('V277 reward-only repair: bounded confirmed second-impact 1.30--1.40m '
                   'reserve credit320 in hard near-Y50--65mm lanes; third preparation '
                   'near gain7->3. S1--S10 unchanged. All17 physical lessons/DR, gates, '
                   '72/393/7 and numerical PPO/BC/KL unchanged. Full latest actor/critic/Adam; '
                   'no sample rejection, rollback, forced advancement or update cap.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v276_second_launch import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V277 requires unchanged nominal retention')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v276_s01_workspace_alignment'}))


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V277 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 11
                or payload.get('stage_name') != 'fplc_v276_s11_near_y_080mm'
                or payload.get('stage_update') != 102 or payload.get('step') != 3086614528
                or int(payload['opt'].t) != 763065):
            raise ValueError('V277 pins the full V276 S11u102 and fresh reward history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V277 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V277 rejects another source or from-scratch launch')
