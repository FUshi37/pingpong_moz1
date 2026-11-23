"""Default-off extra startup reward weight for the measured critical Y tail."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v291_acquisition_success import AcquisitionSuccessConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v292_critical_startup'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v291_acquisition_success'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v292_critical_startup_extra_scale'})


@dataclass(frozen=True)
class CriticalStartupConfig(AcquisitionSuccessConfig):
    """One reward-only scalar; no environment state or actor input extension."""
    v292_critical_startup_extra_scale: float = 0.


def critical_reward_correction(*, potential_reward, acquisition_reward,
        group, initial_y_m, extra_scale):
    """Reweight existing terms using episode-constant group/Y support.

    Arrays have shape(N,). At extra_scale3, the two original terms receive
    fourfold total weight at Y>=85mm, with a smooth75--85mm bridge. Their
    original true-termination, timeout, counter and clawback semantics remain
    intact. Below75mm and outside group1 the correction is exactly zero.
    """
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .075) / .010, 0., 1.)
    support = fraction * fraction * (3. - 2. * fraction)
    support = jnp.where(jnp.asarray(group) == 1, support, 0.)
    scale = extra_scale * support
    potential_extra = scale * jnp.asarray(potential_reward)
    acquisition_extra = scale * jnp.asarray(acquisition_reward)
    return potential_extra + acquisition_extra, dict(
        critical_support=support, extra_scale=scale,
        potential_extra=potential_extra, acquisition_extra=acquisition_extra)


def validate_config(cfg: CriticalStartupConfig) -> None:
    from fixed_plane_v291_acquisition_success import validate_config as previous
    previous(cfg)
    scale = cfg.v292_critical_startup_extra_scale
    if not math.isfinite(scale) or not 0. <= scale <= 7.:
        raise ValueError('V292 extra scale must be finite in[0,7]')
    if scale > 0. and (cfg.v291_near_acquisition_credit != 1024.
            or cfg.v290_startup_potential_weight != 96. or not cfg.v225_enabled):
        raise ValueError('V292 requires the exact original acquisition/potential rewards')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V292 requires all17 original V291 physical lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = CriticalStartupConfig(**vars(stage.cfg),
            v292_critical_startup_extra_scale=3. if i >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V292 reward-only critical near-start emphasis: original group1 '
            'initialY75--85mm smooth ramp adds up to3x existing V290 potential '
            'and V291 real-third/clawback rewards, giving4x total at85mm+. '
            'All other rewards and all17 V281 domains/gates unchanged. No new '
            'event/count/state/observation;72/393/7,control/physics/reset/DR and '
            'all1024x128 numerical PPO/BC/KL settings exact. Latest full state '
            'and exact history; all32 minibatches, no cap/reset/rewind/rejection/'
            'rollback, interpolation or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v291_acquisition_success import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V292 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2201
                or payload.get('step') != 3444572160 + (update - 2073) * 131072
                or int(payload['opt'].t) != 850430 + (update - 2073) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V292 requires latest full stalled V291 S12 after128 real updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V292 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V292 rejects another source or from-scratch launch')
