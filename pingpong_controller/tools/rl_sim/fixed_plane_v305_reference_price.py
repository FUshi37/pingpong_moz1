"""Opt-in repricing of the existing early reference preparation reward."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v304_timing_envelope import TimingEnvelopeConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v305_reference_price'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v304_timing_envelope'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v305_reference_preparation_scale'})


@dataclass(frozen=True)
class ReferencePriceConfig(TimingEnvelopeConfig):
    """Identity default; change no state, observation or physical target."""
    v305_reference_preparation_scale: float = 1.


def validate_scale(scale: float) -> None:
    if isinstance(scale, bool) or not math.isfinite(scale) or not 1. <= scale <= 5.:
        raise ValueError('V305 reference reward price must be finite in [1,5]')


def reference_cost_correction(*, old_cost, scale: float):
    """Multiply only the existing negative (N,) reference preparation cost."""
    validate_scale(scale)
    cost = jnp.asarray(old_cost)
    if cost.ndim != 1:
        raise ValueError('V305 requires an (N,) reference cost vector')
    return (scale-1.)*cost, scale*cost


def validate_config(cfg: ReferencePriceConfig) -> None:
    from fixed_plane_v304_timing_envelope import validate_config as previous
    previous(cfg)
    validate_scale(cfg.v305_reference_preparation_scale)
    if cfg.v305_reference_preparation_scale != 1. and not cfg.v304_timing_envelope_reward:
        raise ValueError('V305 requires the exact V304 reference reward ledger')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V305 requires all 17 unchanged V304 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = ReferencePriceConfig(**vars(stage.cfg),
            v305_reference_preparation_scale=5. if index >= 11 else 1.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(stage.notes if index < 11 else
            'V305 reward-only reference price: multiply the existing V304 '
            'early reference-height cost by five (200->1000/s). Its exact '
            'support,180--90ms height schedule,90--60ms fade,upper band and '
            'pseudo-Huber shape remain unchanged. Physical velocity floor, '
            'impact goals and every other reward remain exact. All 17 '
            'IDs/domains/gates,72/393/7,control,physics,reset,DR and numerical '
            '1024x128 PPO/BC/KL unchanged. Latest full actor/critic/Adam and '
            'unmodified history;all32,no rejection,rollback,rewind,reset, '
            'update cap or forced advancement. Native graduation required.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v304_timing_envelope import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V305 requires full uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        total = update-4284
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update or update < 4412
                or payload.get('step') != 3734372352+total*131072
                or int(payload['opt'].t) != 921182+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V305 requires latest V304 S12 history after >=128 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V305 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V305 rejects another source or from-scratch launch')
