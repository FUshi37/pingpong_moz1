"""Restore the native physical preimpact loss removed by clean startup."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import jax.numpy as jnp
from fixed_plane_v310_qualified_acquisition import QualifiedAcquisitionConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v311_clean_preimpact'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v310_qualified_acquisition'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v311_clean_preimpact_enabled'})


@dataclass(frozen=True)
class CleanPreimpactConfig(QualifiedAcquisitionConfig):
    """Default-off reward restoration without new state or observation."""
    v311_clean_preimpact_enabled: bool = False


def preparation_correction(*, physical_cost: Any, clean_support: Any,
                           previous_hits: Any, enabled: bool
                           ) -> tuple[Any, dict[str, Any]]:
    """Restore exactly the removed share of the native (N,) V304 loss.

    The parent retains (1-support)*cost in its partial clean bridge. Adding
    support*cost restores one original physical cost, without reintroducing
    the removed joint-action label, reference loss or other old rewards.
    The native loss already contains its TTC envelope, actual-Y support,
    velocity floor and original SI weights. No action is prescribed.
    """
    if type(enabled) is not bool:
        raise ValueError('V311 opt-in must be Boolean')
    cost, support, counts = map(jnp.asarray,
        (physical_cost, clean_support, previous_hits))
    if cost.ndim != 1 or support.shape != cost.shape or counts.shape != cost.shape:
        raise ValueError('V311 requires identical (N,) arrays')
    active = enabled & (counts >= 1) & (counts <= 2) & (support > 0.)
    restored = jnp.where(active, support*cost, 0.)
    return restored, dict(active=(active & (cost < 0.)).astype(jnp.float32),
        original_physical_cost=jnp.where(active, cost, 0.),
        restored_penalty=restored)


def validate_config(cfg: CleanPreimpactConfig) -> None:
    from fixed_plane_v310_qualified_acquisition import validate_config as previous
    previous(cfg)
    if type(cfg.v311_clean_preimpact_enabled) is not bool:
        raise ValueError('V311 reward opt-in must be Boolean')
    if cfg.v311_clean_preimpact_enabled and (
            not cfg.v310_qualified_acquisition_enabled
            or not cfg.v304_timing_envelope_reward):
        raise ValueError('V311 requires the unchanged V310 clean timing ledger')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V311 requires all 17 unchanged V310 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = CleanPreimpactConfig(**vars(stage.cfg),
                                   v311_clean_preimpact_enabled=i >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 11 else
            'V311 reward-only physical preimpact restoration in clean startup. '
            'Only native previouscount1/2 restores clean_support times the '
            'already computed V304 physical loss. Retain its original TTC/Y '
            'envelope and SI weights; no action label, reference loss or '
            'controller target. Preserve V310 qualified acquisition budget '
            'and every other reward, first11 lessons, all17 domains/gates, '
            '72/393/7, control, physics, DR, reset and numeric1024x128 '
            'PPO/BC/KL. Latest full state/rawhistory; all32, no rejection, '
            'rollback, rewind, reset, cap or forced advancement.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v310_qualified_acquisition import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
                            requested_start_stage: str,
                            resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool,
                            allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in (
                'obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V311 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 5695
                or payload.get('step') != 3885760512+(update-5439)*131072
                or int(payload['opt'].t) != 958142+(update-5439)*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V311 requires latest full V310 S12 after 256 real updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V311 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V311 rejects another source or from-scratch launch')
