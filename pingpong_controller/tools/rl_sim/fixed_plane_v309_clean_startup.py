"""Opt-in near-side startup reward with genuine progress and survival credit."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v308_first_push import FirstPushConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v309_clean_startup'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v308_first_push'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v309_clean_startup_reward_enabled'})
ALIVE_REWARD_PER_S = 50.
FAILURE_PENALTY = 2000.


@dataclass(frozen=True)
class CleanStartupConfig(FirstPushConfig):
    """Default-off reward switch; all inherited settings remain unchanged."""

    v309_clean_startup_reward_enabled: bool = False


def clean_startup_reward(*, old_reward: Any, previous_hits: Any,
                         confirmed_event: Any, apex_abs_m: Any,
                         ball_vxy_m_s: Any, terminated: Any,
                         initial_y_m: Any, group: Any, enabled: bool,
                         dt_s: float) -> tuple[Any, dict[str, Any]]:
    """Replace only near-side previous-count 0--2 reward using (N,) arrays.

    Native hit events supply bounded 20/40/512 credit. Absolute apex targets
    are 1.15/1.25/1.39 m, with 25/35/40 mm sigmas. Horizontal speed sigmas
    are 0.05/0.05/0.03 m/s. Failure gives no event credit and costs 2000.
    Positive survival credit is 50/s; steady, nominal, mirror and small-near
    transitions keep their exact original reward. Nothing changes samples,
    counts, state, observations, termination or control.
    """
    if (type(enabled) is not bool or not math.isfinite(dt_s)
            or not 0. < dt_s < .1):
        raise ValueError('V309 requires a Boolean opt-in and SI timestep')
    old = jnp.asarray(old_reward)
    values = [jnp.asarray(x) for x in (
        previous_hits, confirmed_event, apex_abs_m, ball_vxy_m_s,
        terminated, initial_y_m, group)]
    if old.ndim != 1 or any(x.shape != old.shape for x in values):
        raise ValueError('V309 requires identical (N,) reward arrays')
    counts, event, apex, vxy, terminal, initial, groups = values
    fraction = jnp.clip((initial - .035) / .010, 0., 1.)
    support = jnp.where(groups == 1,
                        fraction * fraction * (3. - 2. * fraction), 0.)
    active = enabled & (counts >= 0) & (counts < 3) & (support > 0.)
    phase = jnp.clip(counts.astype(jnp.int32), 0, 2)
    target = jnp.asarray([1.15, 1.25, 1.39], dtype=old.dtype)[phase]
    apex_sigma = jnp.asarray([.025, .035, .040], dtype=old.dtype)[phase]
    xy_sigma = jnp.asarray([.05, .05, .03], dtype=old.dtype)[phase]
    weight = jnp.asarray([20., 40., 512.], dtype=old.dtype)[phase]
    score = jnp.exp(-.5 * ((apex - target) / apex_sigma) ** 2
                    - .5 * (vxy / xy_sigma) ** 2)
    credit = jnp.where(event.astype(bool) & ~terminal.astype(bool),
                       weight * score, 0.)
    clean = (ALIVE_REWARD_PER_S * dt_s + credit
             - FAILURE_PENALTY * terminal.astype(old.dtype))
    # Direct replacement avoids cancellation of large old penalties at full
    # support. Fractional support keeps the inherited smooth 35--45 mm bridge.
    blended = jnp.where(support >= 1., clean,
                         old + support * (clean - old))
    result = jnp.where(active, blended, old)
    return result, dict(
        active=active.astype(jnp.float32),
        support=jnp.where(active, support, 0.),
        old_reward=jnp.where(active, old, 0.),
        new_reward=jnp.where(active, clean, 0.),
        event_credit=jnp.where(active, credit, 0.),
        failure_penalty=jnp.where(active,
                                  -FAILURE_PENALTY * terminal, 0.),
        correction=jnp.where(active, result - old, 0.))


def validate_config(cfg: CleanStartupConfig) -> None:
    from fixed_plane_v308_first_push import validate_config as previous
    previous(cfg)
    if type(cfg.v309_clean_startup_reward_enabled) is not bool:
        raise ValueError('V309 reward opt-in must be Boolean')
    if cfg.v309_clean_startup_reward_enabled and (
            not cfg.v225_enabled
            or getattr(cfg, 'fixed_plane_v167_global_reward_multiplier', 1.) != 1.):
        raise ValueError('V309 requires the original near-side reward units')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V309 requires all 17 unchanged V308 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = CleanStartupConfig(**vars(stage.cfg),
                                 v309_clean_startup_reward_enabled=index >= 11)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            notes=stage.notes if index < 11 else
            'V309 reward-only near startup cleanup: group1 smoothY35--45mm, '
            'native previouscount0--2. Replace the entire old returned startup '
            'reward by50/s survival,real confirmed20/40/512 credit with '
            '1.15/1.25/1.39m apex and50/50/30mm/s XY scores,and2000 failure '
            'cost with zero failure credit. No forced early recenter or large '
            'coupled action penalties. All later rewards,nominal/mirror/small '
            'near,first11 lessons,all17 IDs/domains/gates,72/393/7,control, '
            'physics,DR,reset and every numeric1024x128 PPO/BC/KL stay exact. '
            'Restore latest full state/raw history;all32,no rejection, '
            'rollback,rewind,reset,updatecap or forced advancement.'))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v308_first_push import validate_training_contract as previous
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
        raise ValueError('V309 requires complete uncapped 72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 5166
                or payload.get('step') != 3828350976 + (update-5001)*131072
                or int(payload['opt'].t) != 944126 + (update-5001)*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V309 requires latest complete V308 S12 history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V309 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V309 rejects another source or from-scratch launch')
