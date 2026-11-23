"""V258: Gaussian force sigmas with bounded, small rebound perturbations."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v256_workspace_force import (
    WorkspaceForceConfig, PARENT_PROFILE, PARENT_SHA256, PARENT_STAGE,
    RACKET_LOW_M, RACKET_HIGH_M, BALL_LOW_M, BALL_HIGH_M,
    validate_config as validate_workspace_config,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v258'
LABELS = ('workspace', 'center_bridge', 'center_commit', 'flight_04mn',
          'flight_08mn', 'flight_1mn', 'rebound_cap_1pct',
          'rebound_cap_5pct', 'rebound_cap_10pct',
          'camera_ar1_1x', 'camera_ar1_1x_consolidation')
STAGE_NAMES = tuple(f'fplc_v258_s{i:02d}_{name}' for i, name in enumerate(LABELS, 1))
START_STAGE = STAGE_NAMES[0]
FIELDS = frozenset({'v258_enabled', 'v258_rebound_force_limit_n',
    'v258_rebound_angle_limit_deg', 'v258_rebound_speed_relative_limit',
    'v258_rebound_reference_speed_m_s'})
REBOUND_SIGMAS_N = (.058, .072, .082)
REFERENCE_SPEED_M_S = 2.53
REWARD_CHANGES = {
    'fixed_plane_v165_steady_band_violation_weight': 30.,
    'fixed_plane_v191_margin_weight': 6.,
    'fixed_plane_v158_late_squared_vxy_weight': 3.,
}
LEGACY_REWARD_VALUES = dict(zip(REWARD_CHANGES, (0., 0., 2.5)))
REWARD_SOURCE = dict(
    update=720, step=1429209088, adam_t=358438,
    sha256='e4e925deceaac0d2b72c22011f9a0ca1c6c90ba0394872c5a419fcc3fc61efa1')


@dataclass(frozen=True)
class GaussianBoundedForceConfig(WorkspaceForceConfig):
    """Opt-in72/398/7 schema; rebound sigmas refer to2.53m/s, per axis."""

    v258_enabled: bool = True
    v258_rebound_force_limit_n: float = .20
    v258_rebound_angle_limit_deg: float = 1.
    v258_rebound_speed_relative_limit: float = .01
    v258_rebound_reference_speed_m_s: float = REFERENCE_SPEED_M_S
    # Default-off replay of pre-repair V258 checkpoints remains available.
    fixed_plane_v165_steady_band_violation_weight: float = 0.
    fixed_plane_v191_margin_weight: float = 0.


def reward_contract(cfg: Any) -> str:
    """Identify either complete reward contract; reject mixed revisions."""
    saved = cfg if isinstance(cfg, Mapping) else vars(cfg)
    values = {key: saved.get(key, 0.) for key in REWARD_CHANGES}
    if values == REWARD_CHANGES:
        return 'height_tail_v1'
    if values == LEGACY_REWARD_VALUES:
        return 'legacy_v172'
    raise ValueError('V258 requires one complete registered reward revision')


def is_reward_transition(payload: Mapping[str, Any]) -> bool:
    """Recognize only the user's safely stopped pre-repair S1 full state."""
    return bool(
        payload.get('args', {}).get('curriculum_profile') == PROFILE
        and payload.get('stage_name') == START_STAGE
        and payload.get('stage_index') == 1
        and payload.get('stage_update') == REWARD_SOURCE['update']
        and payload.get('step') == REWARD_SOURCE['step']
        and getattr(payload.get('opt'), 't', None) == REWARD_SOURCE['adam_t']
        and reward_contract(payload.get('env_cfg') or {}) == 'legacy_v172')


def sample_velocity_scaled_forces(
    keys: jax.Array, *, contact: jax.Array, previous_contact: jax.Array,
    other_contact: jax.Array, vz_m_s: jax.Array, cooldown_ticks: jax.Array,
    flight_std_n: Sequence[float], rebound_std_n: float,
    free_velocity_m_s: jax.Array, reference_speed_m_s: float,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Sample (N,3) world forces; scale only the one-tick rebound XY sigma.

    Use the paired unforced end-of-tick speed, including gravity. Below the
    reference speed, sigma scales linearly with speed; above it, sigma stays
    at its configured value. Flight XYZ draws, release detection and the220ms
    cooldown retain the V256 sampling contract. Added rebound Z is zero.
    """
    normals = jax.vmap(lambda key: jax.random.normal(key, (6,), dtype=jnp.float32))(keys)
    speed = jnp.linalg.norm(free_velocity_m_s, axis=-1)
    speed_scale = jnp.where(jnp.isfinite(speed),
        jnp.clip(speed / reference_speed_m_s, 0., 1.), 0.)
    sigma = rebound_std_n * speed_scale
    flying = ~contact & ~other_contact
    rebound = (flying & previous_contact & (vz_m_s > 0.) & (cooldown_ticks == 0)
               & (rebound_std_n > 0.))
    force = normals[:, :3] * jnp.asarray(flight_std_n) * flying[:, None]
    impulse = normals[:, 3:] * jnp.stack((sigma, sigma, jnp.zeros_like(sigma)), axis=-1)
    force += jnp.where(rebound[:, None], impulse, 0.)
    cooldown = jnp.where(rebound, 220, jnp.maximum(cooldown_ticks - 1, 0))
    return force, rebound, cooldown


def limit_rebound_force(
    force_n: jax.Array, free_velocity_m_s: jax.Array, *, mass_kg: jax.Array,
    dt_s: float, force_limit_n: float, angle_limit_deg: float,
    speed_relative_limit: float,
) -> tuple[jax.Array, jax.Array]:
    """Scale an (N,3) force along its sampled direction; never overwrite velocity.

    ``mass_kg`` has shape(N,). ``free_velocity_m_s`` is the paired unforced
    end-of-tick velocity, including gravity. Intersect force norm,3-D angle and speed-shell constraints along
    the sampled ray. Preserve small samples exactly; zero-speed lanes get no
    impulse. The speed lower bound uses the first shell crossing.
    """
    force = jnp.asarray(force_n)
    velocity = jnp.asarray(free_velocity_m_s)
    delta = force * (dt_s / jnp.maximum(jnp.asarray(mass_kg), 1e-12))[:, None]
    speed_squared = jnp.sum(velocity**2, axis=-1)
    delta_squared = jnp.sum(delta**2, axis=-1)
    dot = jnp.sum(velocity * delta, axis=-1)
    tiny = 1e-30
    tangent = jnp.tan(jnp.deg2rad(angle_limit_deg))
    cross = jnp.linalg.norm(jnp.cross(velocity, delta), axis=-1)
    angle_coefficient = cross - tangent * dot
    angle_scale = jnp.where(angle_coefficient > 0.,
        tangent * speed_squared / jnp.maximum(angle_coefficient, tiny), 1.)
    force_scale = force_limit_n / jnp.maximum(jnp.linalg.norm(force, axis=-1), tiny)
    r = speed_relative_limit
    upper = speed_squared * (2. * r + r * r)
    upper_root = jnp.sqrt(jnp.maximum(dot**2 + delta_squared * upper, 0.))
    upper_scale = jnp.where(dot >= 0., upper / jnp.maximum(upper_root + dot, tiny),
        (-dot + upper_root) / jnp.maximum(delta_squared, tiny))
    lower = speed_squared * (2. * r - r * r)
    discriminant = dot**2 - delta_squared * lower
    lower_scale = jnp.where((dot < 0.) & (discriminant > 0.),
        lower / jnp.maximum(-dot + jnp.sqrt(jnp.maximum(discriminant, 0.)), tiny), 1.)
    scale = jnp.clip(jnp.minimum(jnp.minimum(force_scale, angle_scale),
                                jnp.minimum(upper_scale, lower_scale)), 0., 1.)
    # Leave ordinary Gaussian draws byte-identical and keep capped draws inside
    # the analytic surface despite float32 integration/round-off.
    scale = jnp.where(scale < 1., scale * (1. - 2e-6), scale)
    valid = ((speed_squared > 1e-16) & jnp.all(jnp.isfinite(velocity), axis=-1)
             & jnp.all(jnp.isfinite(force), axis=-1))
    scale = jnp.where(valid, scale, 0.)
    limited = jnp.where(valid[:, None], force * scale[:, None], 0.)
    return limited, scale


def validate_config(cfg: GaussianBoundedForceConfig) -> None:
    """Keep V256 geometry/control contracts; use the new force-only envelope."""
    # Reuse every workspace/DR/noise check; replace only the legacy flight
    # envelope during that check because V258's XYZ sigma is isotropic.
    validate_workspace_config(replace(cfg, v256_flight_force_std_n=(0., 0., 0.)))
    reward_contract(cfg)
    if cfg.hit_apex_lift_penalty_weight != 500.:
        raise ValueError('V258 retains the native diagnostic-only lift coefficient')
    std = np.asarray(cfg.v256_flight_force_std_n)
    limits = (cfg.v258_rebound_force_limit_n, cfg.v258_rebound_angle_limit_deg,
              cfg.v258_rebound_speed_relative_limit, cfg.v258_rebound_reference_speed_m_s)
    if (not cfg.v258_enabled or not cfg.v256_enabled
            or std.shape != (3,) or not np.isfinite(std).all()
            or np.any(std < 0.) or np.any(std > .001)
            or cfg.v256_rebound_force_std_n > REBOUND_SIGMAS_N[-1]
            or not np.isfinite(limits).all() or min(limits) <= 0.
            or cfg.v258_rebound_force_limit_n > .20
            or cfg.v258_rebound_angle_limit_deg > 1.
            or cfg.v258_rebound_speed_relative_limit > .01
            or cfg.v258_rebound_reference_speed_m_s != REFERENCE_SPEED_M_S
            or cfg.ball_flight_drag_coefficient_m_inv != 0.):
        raise ValueError('V258 requires <=1mN flight, <=82mN rebound sigma at2.53m/s, '
                         '200mN/1deg/1% caps')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep all11 lessons and quality gates; use the32-pass hold without a wait."""
    from fixed_plane_v256_workspace_force import STAGE_NAMES as PREVIOUS_STAGES
    if tuple(s.name for s in source) != PREVIOUS_STAGES:
        raise ValueError('V258 requires the complete preserved V256 course')
    result = []
    for i, (stage, name) in enumerate(zip(source, STAGE_NAMES, strict=True)):
        flight = 0. if i < 3 else (.0004, .0008, .001)[min(i - 3, 2)]
        rebound = 0. if i < 6 else REBOUND_SIGMAS_N[min(i - 6, 2)]
        cfg = GaussianBoundedForceConfig(**vars(stage.cfg))
        cfg = replace(cfg, v256_flight_force_std_n=(flight,) * 3,
            v256_rebound_force_std_n=rebound, **REWARD_CHANGES)
        validate_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, min_updates=0, notes=(
            'V258 user-requested in-place revision: flight XYZ0.4/0.8/1mN; '
            'rebound XY58/72/82mN times min(free outgoing speed/2.53m/s,1), '
            'about1/5/10% at the1deg cap for near-vertical bounces at/below2.53m/s. '
            'One1ms rebound tick; combined force limited to0.20N,1deg and1% speed change. '
            'Keep strict workspace,quality gates and1x camera. '
            'No minimum-update wait; retain32 consecutive gate passes and independent validation. '
            'Reward height_tail_v1: late band cost30, bilateral margin6, '
            'late XY squared cost3; retain native diagnostic lift coefficient500, '
            'launch, signed-height, adjacent, center and angular rewards. '
            'Actor72,critic398.')))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Accept the pinned V172 source or exact own-state continuation only."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if requested_start_stage != START_STAGE:
            raise ValueError('V258 requires the first workspace lesson')
        from fixed_plane_v254_camera_noise import (
            START_STAGE as PARENT_START, validate_resume_payload as validate_parent)
        validate_parent(payload, requested_start_stage=PARENT_START,
            resume_curriculum_state=resume_curriculum_state, reset_optimizer=reset_optimizer,
            reset_critic=reset_critic, allow_obs_dim_migration=allow_obs_dim_migration)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 398, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V258 requires an uncapped full72/398/7 own state without resets')
    index = payload.get('stage_index', 0)
    if not 1 <= index <= len(STAGE_NAMES) or payload.get('stage_name') != STAGE_NAMES[index - 1]:
        raise ValueError('V258 source stage differs from the registered course')
    saved = payload.get('env_cfg') or {}
    rebound = 0. if index < 7 else REBOUND_SIGMAS_N[min(index - 7, 2)]
    if (not FIELDS.issubset(saved)
            or saved['v258_rebound_reference_speed_m_s'] != REFERENCE_SPEED_M_S
            or saved['v258_rebound_force_limit_n'] != .20
            or saved.get('v256_rebound_force_std_n') != rebound):
        raise ValueError('V258 resume requires the updated velocity-scaled force contract')
    if reward_contract(saved) == 'legacy_v172':
        if (not is_reward_transition(payload) or resume_curriculum_state
                or requested_start_stage != START_STAGE):
            raise ValueError('V258 reward repair pins stopped S1u720 and fresh history')
        return
    if resume_curriculum_state:
        if requested_start_stage != payload['stage_name']:
            raise ValueError('V258 own resume requires exact stage and causal history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
