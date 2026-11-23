"""Opt-in S7 reward-only control (V259) and near-side candidate (V260)."""
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v258_gaussian_force import (
    GaussianBoundedForceConfig, REWARD_CHANGES as PARENT_REWARDS,
    validate_config as validate_parent,
)

PROFILES = ('fixed_plane_limit_cycle_qvel_rmp_v259_reward_control',
            'fixed_plane_limit_cycle_qvel_rmp_v260_reward_near_y')
VERSIONS = dict(zip(PROFILES, (259, 260)))
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v258'
PARENT_STAGE = 'fplc_v258_s07_rebound_cap_1pct'
SOURCE_SHA256 = '7d2d0aa84654e6c0307ffc87de5a744445f7269f55c11422d06a71bcee123c23'
SOURCE = dict(update=410, step=1558708224, adam_t=389615)
REWARD_CHANGES = {PROFILES[0]: {}, PROFILES[1]: {
    'fixed_plane_v165_steady_band_violation_weight': 60.,
    'fixed_plane_v191_margin_weight': 12.,
    'fixed_plane_v158_late_squared_vxy_weight': 6.,
}}
EXTRA_FIELDS = frozenset({'v259_enabled', 'v259_version',
    'v259_near_y_position_weight_per_s', 'v259_toward_y_weight_per_s',
    'v259_safe_y_m', 'v259_y_width_m'})


@dataclass(frozen=True)
class RewardTuningConfig(GaussianBoundedForceConfig):
    """Reward metadata only; actor72/critic398/action7 stay unchanged."""
    v259_enabled: bool = True
    v259_version: int = 259
    v259_near_y_position_weight_per_s: float = 0.
    v259_toward_y_weight_per_s: float = 0.
    v259_safe_y_m: float = -.475
    v259_y_width_m: float = .04


def near_y_rewards(ball_m: jax.Array, racket_m: jax.Array,
        velocity_m_s: jax.Array, *, position_weight: float,
        velocity_weight: float, safe_y_m: float, width_m: float,
        dt_s: float) -> tuple[jax.Array, jax.Array]:
    """Return two (N,) costs; increasing world Y approaches the robot.

    Keep centerward recovery free of directional cost in the far half.
    No cost pushes the ball toward the far workspace edge. Native centering,
    workspace termination and all physical state updates remain unchanged.
    """
    ball_y = jnp.asarray(ball_m)[:, 1]
    racket_y = jnp.asarray(racket_m)[:, 1]
    ball_excess = jnp.maximum((ball_y-safe_y_m)/width_m, 0.)
    racket_excess = jnp.maximum((racket_y-safe_y_m)/width_m, 0.)
    position = -position_weight*dt_s*jnp.minimum(
        ball_excess**2+.5*racket_excess**2, 4.)
    near = jnp.clip((ball_y-safe_y_m+.5*width_m)/(.5*width_m), 0., 1.)
    toward = -velocity_weight*dt_s*near*jnp.minimum(
        (jnp.maximum(jnp.asarray(velocity_m_s)[:, 1], 0.)/.1)**2, 4.)
    return position, toward


def validate_config(cfg: RewardTuningConfig) -> None:
    """Review one explicit reward envelope without weakening V258 checks."""
    validate_parent(replace(cfg, **PARENT_REWARDS))
    weights = np.asarray([cfg.v259_near_y_position_weight_per_s,
                          cfg.v259_toward_y_weight_per_s])
    if (not cfg.v259_enabled or cfg.v259_version not in VERSIONS.values()
            or cfg.v259_safe_y_m != -.475 or cfg.v259_y_width_m != .04
            or not np.isfinite(weights).all() or np.any(weights < 0.)
            or np.any(weights > [20., 10.])):
        raise ValueError('Invalid reward-only near-Y envelope')
    allowed = (PARENT_REWARDS if cfg.v259_version == 259 else
               PARENT_REWARDS | REWARD_CHANGES[PROFILES[1]])
    # Earlier lessons/retention retain the native quality coefficients.
    actual = {key:getattr(cfg, key) for key in PARENT_REWARDS}
    if actual != PARENT_REWARDS and actual != allowed:
        raise ValueError('Mixed reward tuning coefficients')
    if cfg.v259_version == 259 and np.any(weights != 0.):
        raise ValueError('V259 control must keep near-Y reward off')


def build_stages(source: Sequence[Any], *, profile: str) -> list[Any]:
    """Keep all11 domains/gates; resume S7 with fresh reward statistics."""
    if profile not in PROFILES or len(source) != 11 or source[6].name != PARENT_STAGE:
        raise ValueError('Reward trials require the complete V258 course')
    result = []
    for i, stage in enumerate(source):
        cfg = RewardTuningConfig(**vars(stage.cfg), v259_version=VERSIONS[profile])
        if i >= 6:
            cfg = replace(cfg, **REWARD_CHANGES[profile])
        if profile == PROFILES[1] and i >= 1:
            scale = .5 if i == 1 else 1.
            cfg = replace(cfg, v259_near_y_position_weight_per_s=20.*scale,
                v259_toward_y_weight_per_s=10.*scale)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v258_', f'fplc_v{VERSIONS[profile]}_', 1),
            notes=('Reward-only GPU0 S7 experiment: unchanged72/398/7, full '
                'actor/critic/Adam,1024x128, physics/forces/reset/observations '
                'and every gate. V259 control; V260 band60/margin12/XY6 plus '
                'one-sided +Y position20/s and toward-robot velocity10/s.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, profile: str,
        requested_start_stage: str, resume_curriculum_state: bool,
        reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Pin full source; never import old reward history into a new trial."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(key) for key in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)):
        raise ValueError('Reward trials require full72/398/7 state without resets')
    parent = payload.get('args', {}).get('curriculum_profile')
    expected = f'fplc_v{VERSIONS[profile]}_s07_rebound_cap_1pct'
    if requested_start_stage != expected or payload.get('stage_index') != 7:
        raise ValueError('Reward trials stay in the same S7 domain')
    if parent == PARENT_PROFILE:
        if (resume_curriculum_state or payload.get('stage_name') != PARENT_STAGE
                or payload.get('stage_update') != SOURCE['update']
                or payload.get('step') != SOURCE['step']
                or int(payload['opt'].t) != SOURCE['adam_t']
                or payload['args'].get('max_stage_updates') != -1):
            raise ValueError('Pin V258 S7u410 and start fresh reward history')
    elif parent != profile or not resume_curriculum_state or payload.get('stage_name') != expected:
        raise ValueError('Own trial continuation requires exact causal S7 history')


def validate_training_contract(settings: Any, source_args: Mapping[str, Any]) -> None:
    """Change reward terms only; pin the source's numerical PPO/BC settings."""
    keys = ('n_envs', 'n_steps', 'minibatch_size', 'update_epochs',
        'learning_rate', 'gamma', 'gae_lambda', 'time_limit_bootstrap',
        'clip_range', 'target_kl', 'ent_coef', 'min_log_std', 'vf_coef',
        'max_grad_norm', 'hidden_dim', 'asymmetric_critic',
        'critic_command_history_steps', 'counterfactual_supervision_coef',
        'counterfactual_vxy_weight_mode', 'conditional_reset_done',
        'right_arm_pd_profile', 'convergence_window', 'min_stage_updates',
        'stage_metric_warmup_updates', 'advance_eval_n_envs',
        'advance_eval_steps', 'advance_eval_retry_updates')
    changed = [key for key in keys if getattr(settings, key, None) != source_args.get(key)]
    if changed:
        raise ValueError('Reward-only trial changed training settings: '+','.join(changed))
    if (settings.n_envs != 1024 or settings.n_steps != 128
            or settings.max_stage_updates not in (32, 64, 96, 128)
            or settings.max_stages != 7 or settings.allow_unconverged_advance):
        raise ValueError('Reward trials require bounded unchanged1024x128 S7 training')
