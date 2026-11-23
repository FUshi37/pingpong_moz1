"""Opt-in near-side first-launch energy reserve; no physical-event changes."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v305_reference_price import ReferencePriceConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v306_first_launch_reserve'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v305_reference_price'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v306_near_first_launch_credit'})
FIRST_APEX_LOWER_M = 1.20
FIRST_APEX_UPPER_M = 1.35
FIRST_APEX_SIGMA_M = .025


@dataclass(frozen=True)
class FirstLaunchReserveConfig(ReferencePriceConfig):
    """Default-off reward weight; retain the original actor-visible schedule."""
    v306_near_first_launch_credit: float = 0.


def first_launch_correction(*, old_credit, counted_hit, previous_hits,
        apex_z_m, terminated, initial_y_m, group, weight: float):
    """Reprice one true confirmed first hit using identically shaped(N,) arrays.

    Near-side support ramps over35--45mm. Net credit is bounded0--512,
    with a flat1.20--1.35m reserve band and25mm Gaussian tails. A simultaneous
    true termination receives zero new credit. This never changes the native
    hit event, confirmation, physical apex, observation or actor target.
    """
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0. <= weight <= 512.:
        raise ValueError('V306 first-launch credit must be finite in[0,512]')
    old = jnp.asarray(old_credit)
    values = [jnp.asarray(v) for v in (counted_hit, previous_hits, apex_z_m,
                                     terminated, initial_y_m, group)]
    if old.ndim != 1 or any(v.shape != old.shape for v in values):
        raise ValueError('V306 first-launch reward requires identical(N,) vectors')
    counted, previous, apex, terminal, initial, groups = values
    fraction = jnp.clip((initial-.035)/.010, 0., 1.)
    support = jnp.where(groups == 1, fraction*fraction*(3.-2.*fraction), 0.)
    event = counted.astype(bool) & (previous == 0) & (support > 0.) & (weight > 0.)
    error = jnp.maximum(FIRST_APEX_LOWER_M-apex, 0.) + jnp.maximum(apex-FIRST_APEX_UPPER_M, 0.)
    desired = weight*jnp.exp(-.5*jnp.square(error/FIRST_APEX_SIGMA_M))
    desired = jnp.where(terminal.astype(bool), 0., desired)
    correction = jnp.where(event, support*(desired-old), 0.)
    return correction, dict(active=event.astype(jnp.float32),
        first_apex_m=jnp.where(event, apex, 0.), support=jnp.where(event, support, 0.),
        effective_credit=old+correction)


def validate_config(cfg: FirstLaunchReserveConfig) -> None:
    from fixed_plane_v305_reference_price import validate_config as previous
    previous(cfg)
    weight = cfg.v306_near_first_launch_credit
    if isinstance(weight, bool) or not math.isfinite(weight) or not 0. <= weight <= 512.:
        raise ValueError('Invalid V306 first-launch reward weight')
    if weight > 0. and (not cfg.v225_enabled or not cfg.v251_startup_energy_band_enabled
            or cfg.fixed_plane_v94_first_hit_apex_reward_weight != 64.):
        raise ValueError('V306 requires the unchanged native first-launch reward ledger')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V306 requires all17 unchanged V305 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = FirstLaunchReserveConfig(**vars(stage.cfg),
            v306_near_first_launch_credit=512. if index >= 11 else 0.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=stage.notes if index < 11 else
            'V306 reward-only near-side first-launch reserve. On the actual '
            'confirmed first event only,group1 smooth35--45mm support reprices '
            'the existing first-apex credit with bounded512,1.20--1.35m flat '
            'band,25mm Gaussian tails;zero new credit on true termination. '
            'Keep the actor-visible apex schedule,native counters and every '
            'other reward. First11 lessons,all17 IDs/domains/gates,72/393/7, '
            'bounded q_ref,RMP/PD,physics,DR,reset and numerical1024x128 '
            'PPO/BC/KL exact. Restore latest complete actor/critic/Adam and '
            'unmodified history;all32,no rejection,rollback,rewind,reset, '
            'update cap or forced advancement. Require native graduation.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v305_reference_price import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or payload.get('args',{}).get('max_stage_updates') != -1):
        raise ValueError('V306 requires complete uncapped72/393/7 state')
    source = payload.get('args',{}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update',0)); total = update-4417
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or int(payload.get('stage_update',0)) != update or update < 4673
                or payload.get('step') != 3751804928+total*131072
                or int(payload['opt'].t) != 925438+total*32
                or not isinstance(tail,list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V306 requires latest V305 S12 full history after>=256 actual updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V306 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V306 rejects another source or from-scratch launch')
