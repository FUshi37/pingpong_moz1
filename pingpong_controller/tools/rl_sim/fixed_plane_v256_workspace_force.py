"""V256: strict workspace, small physical forces, then measured camera noise."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v254_camera_noise import (
    CameraPositionNoiseConfig, PARENT_PROFILE, PARENT_SHA256, PARENT_STAGE,
    validate_noise_config,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v256'
RACKET_LOW_M = (-.145, -.595, 1.)
RACKET_HIGH_M = (.145, -.355, 1.15)
BALL_LOW_M = (-.145, -.595, .95)
BALL_HIGH_M = (.145, -.355, 1.5)
LABELS = ('workspace', 'center_bridge', 'center_commit', 'flight_04mn',
          'flight_08mn', 'flight_12mn', 'rebound_10mms', 'rebound_20mms',
          'rebound_30mms', 'camera_ar1_1x', 'camera_ar1_1x_consolidation')
STAGE_NAMES = tuple(f'fplc_v256_s{i:02d}_{name}' for i, name in enumerate(LABELS, 1))
START_STAGE = STAGE_NAMES[0]
EXTRA_STATE_FIELDS = frozenset({'v256_previous_contact', 'v256_rebound_cooldown_ticks'})
CRITIC_STATE_FIELDS = EXTRA_STATE_FIELDS


@dataclass(frozen=True)
class WorkspaceForceConfig(CameraPositionNoiseConfig):
    """Versioned runtime schema; physics tick1ms, policy tick5ms, world XYZ."""

    v256_enabled: bool = True
    ball_position_noise_ar1_enabled: bool = False
    v256_ball_workspace_lower_m: tuple[float, float, float] = BALL_LOW_M
    v256_ball_workspace_upper_m: tuple[float, float, float] = BALL_HIGH_M
    v256_flight_force_std_n: tuple[float, float, float] = (0., 0., 0.)
    v256_rebound_force_std_n: float = 0.
    v256_center_reward_per_s: float = 0.
    v256_outward_penalty_per_s: float = 0.


def workspace_outside(position_m: jax.Array, low_m: Sequence[float],
                      high_m: Sequence[float]) -> jax.Array:
    """Check (N,3) true world positions against closed bounds, including NaNs."""
    p = jnp.asarray(position_m)
    return jnp.any((p < jnp.asarray(low_m)) | (p > jnp.asarray(high_m))
                   | ~jnp.isfinite(p), axis=-1)


def spatial_rewards(ball_m: jax.Array, racket_m: jax.Array, velocity_m_s: jax.Array,
                    *, center_weight: float, outward_weight: float,
                    dt_s: float) -> tuple[jax.Array, jax.Array]:
    """Reward horizontal co-centering; penalize each outward axis separately."""
    low, high = jnp.asarray(RACKET_LOW_M[:2]), jnp.asarray(RACKET_HIGH_M[:2])
    center, radius = (low + high) * .5, (high - low) * .5
    ball = (jnp.asarray(ball_m)[:, :2] - center) / radius
    racket = (jnp.asarray(racket_m)[:, :2] - center) / radius
    score = jnp.exp(-2. * jnp.sum(ball**2 + racket**2, axis=-1))
    outward_rate = jnp.sum(jnp.maximum(ball * jnp.asarray(velocity_m_s)[:, :2] / radius, 0.), axis=-1)
    return center_weight * dt_s * score, -outward_weight * dt_s * outward_rate


def sample_forces(
    keys: jax.Array, *, contact: jax.Array, previous_contact: jax.Array,
    other_contact: jax.Array, vz_m_s: jax.Array, cooldown_ticks: jax.Array,
    flight_std_n: Sequence[float], rebound_std_n: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample independent world-force innovations for one1ms physics interval.

    Rebound force lasts exactly one tick, on the first rising separation edge,
    with220 physics ticks of debounce. Its Z component is identically zero.
    No force is applied while touching the racket or another object.
    """
    normals = jax.vmap(lambda key: jax.random.normal(key, (6,), dtype=jnp.float32))(keys)
    flying = ~contact & ~other_contact
    rebound = (flying & previous_contact & (vz_m_s > 0.) & (cooldown_ticks == 0)
               & (rebound_std_n > 0.))
    force = normals[:, :3] * jnp.asarray(flight_std_n) * flying[:, None]
    impulse = normals[:, 3:] * jnp.asarray([rebound_std_n, rebound_std_n, 0.])
    force += jnp.where(rebound[:, None], impulse, 0.)
    cooldown = jnp.where(rebound, 220, jnp.maximum(cooldown_ticks - 1, 0))
    return force, rebound, cooldown


def validate_config(cfg: WorkspaceForceConfig) -> None:
    """Pin world bounds and prohibit the removed randomization families."""
    validate_noise_config(cfg)
    if (tuple(cfg.cartesian_pose_position_lower_m) != RACKET_LOW_M
            or tuple(cfg.cartesian_pose_position_upper_m) != RACKET_HIGH_M
            or tuple(cfg.v256_ball_workspace_lower_m) != BALL_LOW_M
            or tuple(cfg.v256_ball_workspace_upper_m) != BALL_HIGH_M
            or not cfg.fixed_contact_workspace_termination_enabled
            or cfg.dr_randomize_pd or cfg.dr_randomize_recovered_rmp
            or tuple(cfg.dr_ball_mass_range) != (.004, .004)
            or cfg.dr_rmp_target_update_hold_probability != 0.
            or cfg.dr_rmp_target_update_hold_tail_probability != 0.
            or cfg.dr_execution_command_hold_probability != 0.
            or cfg.racket_launch_hold_time_s != 0.
            or cfg.stationary_ball_training):
        raise ValueError('V256 pins strict workspace and nominal RMP/PD/mass/publications')
    std = np.asarray(cfg.v256_flight_force_std_n)
    weights = (cfg.v256_rebound_force_std_n, cfg.v256_center_reward_per_s,
               cfg.v256_outward_penalty_per_s)
    if (std.shape != (3,) or not np.isfinite(std).all() or np.any(std < 0.)
            or np.any(std > [.0012, .0012, .0006])
            or not np.isfinite(weights).all() or min(weights) < 0.
            or cfg.v256_rebound_force_std_n > .12
            or cfg.v256_center_reward_per_s > 1. or cfg.v256_outward_penalty_per_s > 2.):
        raise ValueError('V256 forces and spatial rewards exceed preregistered small candidates')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Resume submitted V172 into11 new lessons; no historical DR tail."""
    if len(source) != 17 or source[9].name != PARENT_STAGE:
        raise ValueError('V256 requires the preserved complete V172 source course')
    base = source[9]
    cfg = WorkspaceForceConfig(**vars(base.cfg))
    cfg = replace(cfg, cartesian_pose_position_lower_m=RACKET_LOW_M,
        cartesian_pose_position_upper_m=RACKET_HIGH_M,
        fixed_contact_racket_x_bounds_m=(RACKET_LOW_M[0], RACKET_HIGH_M[0]),
        fixed_contact_racket_y_bounds_m=(RACKET_LOW_M[1], RACKET_HIGH_M[1]),
        fixed_contact_racket_z_bounds_m=(RACKET_LOW_M[2], RACKET_HIGH_M[2]),
        fixed_contact_ball_z_bounds_m=(BALL_LOW_M[2], BALL_HIGH_M[2]),
        ball_low_termination_z_m=BALL_LOW_M[2], ball_high_termination_z_m=BALL_HIGH_M[2],
        ball_view_x_bounds_m=(BALL_LOW_M[0], BALL_HIGH_M[0]),
        ball_view_y_bounds_m=(BALL_LOW_M[1], BALL_HIGH_M[1]),
        ball_view_z_bounds_m=(BALL_LOW_M[2], BALL_HIGH_M[2]),
        fixed_contact_workspace_termination_enabled=True,
        fixed_contact_workspace_termination_penalty_enabled=True)
    changes = ({}, {'v256_center_reward_per_s': .5, 'v256_outward_penalty_per_s': 1.},
        {'v256_center_reward_per_s': 1., 'v256_outward_penalty_per_s': 2.},
        *({'v256_flight_force_std_n': (s, s, s / 2.)} for s in (.0004, .0008, .0012)),
        *({'v256_rebound_force_std_n': s} for s in (.04, .08, .12)),
        {'ball_position_noise_ar1_enabled': True}, {})
    result = []
    for name, change in zip(STAGE_NAMES, changes, strict=True):
        cfg = replace(cfg, **change)
        validate_config(cfg)
        result.append(replace(base, name=name, cfg=cfg, notes=(
            'V256 revision of V254: V172 clean source; strict world workspace at every1ms '
            'tick; centered horizontal juggling; small independent flight forces, '
            'then one-tick rising-release forces, then exact1x camera AR(1). '
            'No RMP/PD, ball-mass or publication DR. Actor72, critic398; no training yet.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Authenticate the submitted parent or an exact own-stage full resume."""
    from fixed_plane_v254_camera_noise import validate_resume_payload as validate_parent
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if requested_start_stage != START_STAGE:
            raise ValueError('V256 requires the first workspace lesson')
        from fixed_plane_v254_camera_noise import START_STAGE as OLD_START
        validate_parent(payload, requested_start_stage=OLD_START,
            resume_curriculum_state=resume_curriculum_state, reset_optimizer=reset_optimizer,
            reset_critic=reset_critic, allow_obs_dim_migration=allow_obs_dim_migration)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V256 requires an uncapped full72/398/7 state without resets/migration')
    index = payload.get('stage_index', 0)
    if not 1 <= index <= len(STAGE_NAMES) or payload.get('stage_name') != STAGE_NAMES[index - 1]:
        raise ValueError('V256 source stage differs from the registered course')
    if resume_curriculum_state:
        if requested_start_stage != payload['stage_name']:
            raise ValueError('V256 own resume requires exact stage and causal history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
