"""Restore the registered hard-near acquisition budget with quality scoring."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence
import jax.numpy as jnp
from fixed_plane_v309_clean_startup import CleanStartupConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v310_qualified_acquisition'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v309_clean_startup'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v310_qualified_acquisition_enabled'})


@dataclass(frozen=True)
class QualifiedAcquisitionConfig(CleanStartupConfig):
    """Default-off event reward; no observation, state or control extension."""
    v310_qualified_acquisition_enabled: bool = False


def qualified_acquisition_correction(*, old_clean_credit: Any,
        legacy_acquisition_reward: Any, legacy_critical_acquisition_extra: Any,
        support: Any, previous_hits: Any, confirmed_event: Any,
        terminated: Any, enabled: bool) -> tuple[Any, dict[str, Any]]:
    """Restore only a true nonterminal third event's original (N,) budget.

    V309 supplies its physical apex/XY quality score through credit/512.
    V291/V292 supply the actual original event budget, including their
    50--65 mm and 75--85 mm ramps. No event exceeds that original budget or
    loses its V309 floor. Original later-failure clawbacks remain unchanged.
    """
    if type(enabled) is not bool:
        raise ValueError('V310 opt-in must be Boolean')
    credit = jnp.asarray(old_clean_credit)
    values = [jnp.asarray(v) for v in (
        legacy_acquisition_reward, legacy_critical_acquisition_extra,
        support, previous_hits, confirmed_event, terminated)]
    if credit.ndim != 1 or any(v.shape != credit.shape for v in values):
        raise ValueError('V310 requires identical (N,) reward arrays')
    legacy, critical, bridge, counts, event, failure = values
    active = (enabled & (counts == 2) & event.astype(bool)
              & ~failure.astype(bool) & (bridge > 0.))
    budget = jnp.maximum(legacy + critical, 0.)
    quality = credit / 512.
    extra = jnp.where(active,
        quality * jnp.maximum(budget - 512. * bridge, 0.), 0.)
    return extra, dict(active=active.astype(jnp.float32),
        legacy_budget=jnp.where(active, budget, 0.),
        effective_third_credit=jnp.where(active, credit * bridge + extra, 0.),
        extra_credit=extra)


def validate_config(cfg: QualifiedAcquisitionConfig) -> None:
    from fixed_plane_v309_clean_startup import validate_config as previous
    previous(cfg)
    if type(cfg.v310_qualified_acquisition_enabled) is not bool:
        raise ValueError('V310 reward opt-in must be Boolean')
    if cfg.v310_qualified_acquisition_enabled and (
            not cfg.v309_clean_startup_reward_enabled
            or cfg.v291_near_acquisition_credit != 1024.
            or cfg.v292_critical_startup_extra_scale != 3.):
        raise ValueError('V310 requires the original registered acquisition budget')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V310 requires all 17 unchanged V309 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = QualifiedAcquisitionConfig(**vars(stage.cfg),
                                        v310_qualified_acquisition_enabled=i >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if i < 11 else
            'V310 reward-only qualified acquisition budget restoration. '
            'Only actual nonterminal native previouscount2 third confirmation '
            'can gain extra credit. Keep the V309 apex/XY quality score and '
            '512 floor;restore only the inherited V291/V292 state-dependent '
            'budget up to4096. Preserve original late-failure clawbacks. '
            'All other V309 rewards,first11 lessons,all17 domains/gates, '
            '72/393/7,control,physics,DR,reset and every numeric1024x128 '
            'PPO/BC/KL unchanged. Full latest state/rawhistory;all32,no '
            'rejection,rollback,rewind,reset,cap or forced advancement.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v309_clean_startup import validate_training_contract as previous
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
        raise ValueError('V310 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 5379
                or payload.get('step') != 3861118976 + (update-5251)*131072
                or int(payload['opt'].t) != 952126 + (update-5251)*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V310 requires latest complete V309 S12 history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V310 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V310 rejects another source or from-scratch launch')
