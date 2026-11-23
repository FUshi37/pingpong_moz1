"""Reward-only second-to-third startup repair in the complete V275 course."""
from dataclasses import dataclass, replace
import math
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v255_corner import CornerRewardConfig
from fixed_plane_v253_preimpact import preimpact_energy_penalty

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v276_second_launch'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v275_stable_completion'
START_STAGE = 'fplc_v276_s11_near_y_080mm'
RETENTION_STAGE = 'fplc_v276_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v276_second_prepare_weight_per_s', 'v276_second_prepare_lead_s'})
SECOND_APEX_RESERVE_M = 1.30
NEAR_Y_START_M = .050
NEAR_Y_FULL_M = .065


@dataclass(frozen=True)
class SecondPreparationConfig(CornerRewardConfig):
    """Default-zero reward fields; no observation or physical state extension."""
    v276_second_prepare_weight_per_s: float = 0.
    v276_second_prepare_lead_s: float = .120


def validate_config(cfg: SecondPreparationConfig) -> None:
    from fixed_plane_v255_corner import validate_config as previous
    previous(cfg)
    weight = cfg.v276_second_prepare_weight_per_s
    lead = cfg.v276_second_prepare_lead_s
    if (not math.isfinite(weight) or not 0. <= weight <= 200.
            or not math.isfinite(lead) or not 0. < lead <= .120
            or (weight > 0. and not cfg.v225_enabled)):
        raise ValueError('Invalid V276 second-launch reward envelope')


def second_preparation_penalty(*, initial_y_m, group, target_apex_z_m, **kwargs):
    """Prepare the second impact in hard near-side lanes, before weak retries.

    All arrays are (N,). The true reset coordinate is reward-only. Smoothly
    enable the existing eightfold corner gain over 50--65 mm and use a soft
    1.30 m second-launch reserve. Existing outcome/ceiling costs remain active.
    Contact detection, hit counts, physical dynamics and sample weights do not
    depend on this calculation.
    """
    fraction = jnp.clip((jnp.asarray(initial_y_m) - NEAR_Y_START_M)
                        / (NEAR_Y_FULL_M - NEAR_Y_START_M), 0., 1.)
    smooth = fraction * fraction * (3. - 2. * fraction)
    scale = jnp.where(jnp.asarray(group) == 1, 8. * smooth, 0.)
    cost, active, required_vz, ttc = preimpact_energy_penalty(
        **kwargs, target_apex_z_m=jnp.maximum(jnp.asarray(target_apex_z_m), SECOND_APEX_RESERVE_M),
        prepare_previous_hits=1)
    active = active & (scale > 0.)
    return jnp.where(active, cost * scale, 0.), active, required_vz, ttc


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all17 domains and all V275 gates; alter only S11+ preparation."""
    if len(source) != 17 or source[10].name != 'fplc_v275_s11_near_y_080mm':
        raise ValueError('V276 requires the complete17-stage V275 course')
    result = []
    for index, stage in enumerate(source):
        cfg = SecondPreparationConfig(**vars(stage.cfg))
        if index >= 10:
            cfg = replace(cfg, v276_second_prepare_weight_per_s=100.,
                          v253_preimpact_energy_lead_s=.120)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v275_', 'fplc_v276_', 1),
            notes=('V276 reward-only startup repair: S11+ second-impact '
                'preparation100/s with0..8 near-side50..65mm gain,120ms lead '
                'and soft1.30m reserve; existing third lead80->120ms. '
                'S1--S10/nominal retention rewards, all17 domains/DR, '
                '72/393/7, numerical PPO/BC/KL, contact accounting and '
                'all V275 rolling/block/retention gates remain unchanged. '
                'Latest full actor/critic/Adam; no sample rejection, rollback, '
                'forced advancement or update cap.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v275_stable_completion import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V276 requires unchanged nominal retention')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name':'fplc_v275_s01_workspace_alignment'}))


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V276 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 11
                or payload.get('stage_name') != 'fplc_v275_s11_near_y_080mm'
                or payload.get('stage_update') != 1855 or payload.get('step') != 3073245184
                or int(payload['opt'].t) != 759801):
            raise ValueError('V276 pins the full V275 S11u1855 and fresh reward history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V276 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V276 rejects another source or from-scratch launch')
