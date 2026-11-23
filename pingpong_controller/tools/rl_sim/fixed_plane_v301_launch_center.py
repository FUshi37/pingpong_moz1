"""Opt-in matched ball/racket launch allowances with unchanged real gates."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v299_safe_reward_backoff import RewardBackoffConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v301_launch_center'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v299_safe_reward_backoff'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS = frozenset({'v301_center_return_full_hits'})


@dataclass(frozen=True)
class LaunchCenterConfig(RewardBackoffConfig):
    """Reward schedule only; zero retains the exact parent reward path."""
    v301_center_return_full_hits: int = 0


def validate_config(cfg: LaunchCenterConfig) -> None:
    from fixed_plane_v299_safe_reward_backoff import validate_config as previous
    previous(cfg)
    if cfg.v301_center_return_full_hits not in (0, 10):
        raise ValueError('V301 uses only the opt-in ten-hit centering schedule')
    if cfg.v301_center_return_full_hits and (
            not cfg.v236_launch_first or cfg.v236_return_start_hits != 3
            or cfg.v236_return_full_hits != 6
            or cfg.fixed_plane_v146_ball_anchor_xy_penalty_weight != 30.
            or cfg.fixed_plane_v134_late_xy_anchor_penalty_weight != 100.):
        raise ValueError('V301 requires the unchanged parent centering ledger')


def center_reward_correction(*, old_ball_penalty, old_racket_penalty,
        ball_offset_xy_m, racket_offset_xy_m, initial_y_m, group, hit_count,
        return_full_hits: int, ball_weight: float, racket_weight: float,
        dt_s: float):
    """Reprice existing penalties, leaving physical counts and metrics intact.

    Center offsets are(N,2); other arrays are(N,). Near-side support is smooth35--45mm and zero on
    nominal/mirror controls. Preserve the initial displacement through hit3,
    then shrink both positive-Y allowances together until hit10. X and
    opposite-side displacement retain a centering cost; further outward
    displacement still costs reward. At hit10 restore the exact original tensors.
    The combined old penalties and correction are always nonpositive.
    """
    if return_full_hits != 10 or any(
            not math.isfinite(value) or value <= 0.
            for value in (ball_weight, racket_weight, dt_s)):
        raise ValueError('Invalid V301 centering reward schedule or SI units')
    old_ball = jnp.asarray(old_ball_penalty)
    old_racket = jnp.asarray(old_racket_penalty)
    ball_offset, racket_offset = map(jnp.asarray, (ball_offset_xy_m, racket_offset_xy_m))
    arrays = [jnp.asarray(value) for value in (initial_y_m, group, hit_count)]
    if old_ball.ndim != 1 or old_racket.shape != old_ball.shape or any(
            value.shape != old_ball.shape for value in arrays) or (
            ball_offset.shape != (old_ball.shape[0], 2)
            or racket_offset.shape != ball_offset.shape):
        raise ValueError('V301 centering inputs require identical(N,) shapes')
    initial, groups, counts = arrays
    support = jnp.clip((initial-.035)/.010, 0., 1.)
    support = jnp.where(groups == 1, support*support*(3.-2.*support), 0.)
    fraction = jnp.clip((counts-3)/float(return_full_hits-3), 0., 1.)
    allowance = jnp.abs(initial)*(1.-fraction*fraction*(3.-2.*fraction))

    def penalty(offset_xy, weight):
        allowed_y = jnp.clip(offset_xy[:, 1], 0., allowance)
        residual = offset_xy.at[:, 1].add(-allowed_y)
        distance = jnp.linalg.norm(residual, axis=-1)
        excess = jnp.maximum(distance-.005, 0.)/.020
        return -weight*dt_s*(jnp.sqrt(1.+excess*excess)-1.)

    new_ball = penalty(ball_offset, ball_weight)
    new_racket = jnp.where(counts >= 3,
        penalty(racket_offset, racket_weight), old_racket)
    complete = counts >= return_full_hits
    ball_correction = jnp.where(complete, 0., support*(new_ball-old_ball))
    racket_correction = jnp.where(complete, 0., support*(new_racket-old_racket))
    return ball_correction+racket_correction, dict(
        ball_correction=ball_correction, racket_correction=racket_correction,
        effective_ball_penalty=old_ball+ball_correction,
        effective_racket_penalty=old_racket+racket_correction,
        allowance_m=jnp.where(complete, 0., support*allowance),
        active=((support > 0.) & ~complete).astype(jnp.float32))


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V301 requires all17 unchanged V299 lessons')
    result = []
    for index, stage in enumerate(source):
        cfg = LaunchCenterConfig(**vars(stage.cfg),
            v301_center_return_full_hits=10 if index >= 11 else 0)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V301 reward-only matched ball/racket launch allowances. Parent '
            'ball-to-world-anchor30/s remained fully active during launch '
            'although racket centering had an allowance. Group1 smooth35--45mm '
            'now gives both existing penalties a positive-Y initial-offset allowance through3 '
            'confirmed hits,then smooth return through10 instead of6. Outward '
            'excess and X/opposite-side displacement still cost reward;combined penalties stay nonpositive. '
            'At10 restore exact old tensors. Nominal/mirror andS1--S11 exact. '
            'Every other reward,all17 domains/IDs and all .94/.91/11.5/.86, '
            'apex1.37--1.41,XY.026/.030,center/racket/view/safety/coverage and '
            'strict nominal gates unchanged.72/393/7,control/physics/DR/reset '
            'and all numerical1024x128 PPO/BC/KL unchanged. Restore latest '
            'full actor/critic/Adam/raw history;all32,no rejection/reset/'
            'rewind/rollback/update cap/forced advancement. Require real '
            'rolling32/hold16,independent288x1200 and nominal graduation.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v299_safe_reward_backoff import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str, Any], *,
        requested_start_stage: str, resume_curriculum_state: bool,
        reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in
                     ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V301 requires full uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('global_update', 0))
        total = update-3215
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE
                or payload.get('stage_index') != 12
                or int(payload.get('stage_update', 0)) != update
                or update < 3562
                or payload.get('step') != 3594256384+total*131072
                or int(payload['opt'].t) != 886974+total*32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update
                or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V301 requires latest V299 S12 full causal state '
                             'after at least128 actual r2 updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V301 requires exact own causal history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V301 rejects another source or from-scratch launch')
