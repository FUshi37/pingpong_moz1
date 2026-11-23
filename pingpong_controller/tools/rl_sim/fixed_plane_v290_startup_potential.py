"""Reward-only analytic startup potential, with exact transition accounting."""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import jax.numpy as jnp

from fixed_plane_v289_coordinated_stroke import CoordinatedStrokeConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v290_startup_potential'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v289_coordinated_stroke'
START_STAGE = 'fplc_v281_s12_near_y_100mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
DISCOUNT = .9995
EXTRA_FIELDS = frozenset({'v290_startup_potential_weight'})
REPLACED_REWARD_FIELDS = (
    'v284_reference_prepare_weight_per_s', 'v286_reference_velocity_weight_per_s',
    'v288_raw_stroke_weight_per_s', 'v288_feasible_stroke_weight_per_s',
    'v289_coordinated_stroke_weight_per_s',
)


@dataclass(frozen=True)
class StartupPotentialConfig(CoordinatedStrokeConfig):
    """One reward field; no additional observation, optimizer or state leaf."""
    v290_startup_potential_weight: float = 0.


def startup_potential(*, ball_z_m, ball_vz_m_s, racket_z_m, ball_radius_m,
        command_vz_m_s, gravity_m_s2, phase, group, initial_y_m, weight):
    """Return bounded (N,) energy/readiness potential from an exact state.

    Hard-near support is the existing smooth50--65mm Y support. Phase0 is
    zero. Phases1/2 mix80% true vertical ball-energy progress1.14--1.40m and
    20% published-reference upward readiness0--0.5m/s after15--30mm ball
    clearance. From phase3 carry a constant until actual termination, so
    completing launch never removes previously acquired potential credit.
    """
    gravity = jnp.maximum(jnp.abs(jnp.asarray(gravity_m_s2)), 1e-6)
    apex = jnp.asarray(ball_z_m) + jnp.square(jnp.asarray(ball_vz_m_s)) / (2. * gravity)
    energy = jnp.clip((apex - 1.14) / .26, 0., 1.)
    clearance = jnp.asarray(ball_z_m) - jnp.asarray(racket_z_m) - jnp.asarray(ball_radius_m)
    clear = jnp.clip((clearance - .015) / .015, 0., 1.)
    clear = clear * clear * (3. - 2. * clear)
    ready = jnp.clip(jnp.asarray(command_vz_m_s) / .5, 0., 1.) * clear
    score = jnp.where(jnp.asarray(phase) >= 3, 1., .8 * energy + .2 * ready)
    score = jnp.where(jnp.asarray(phase) > 0, score, 0.)
    fraction = jnp.clip((jnp.asarray(initial_y_m) - .050) / .015, 0., 1.)
    support = fraction * fraction * (3. - 2. * fraction)
    support = jnp.where(jnp.asarray(group) == 1, support, 0.)
    return weight * support * score


def potential_reward(previous, following, terminated):
    """Shape gamma*Phi(s')-Phi(s); only true MDP termination zeroes Phi(s')."""
    following = jnp.where(jnp.asarray(terminated, dtype=bool), 0., following)
    return DISCOUNT * following - previous, following


def state_potential(env, state, weight):
    """Use the same exact EnvState mapping on both sides of each transition."""
    position = state.data.site_xpos[:, env.racket_site_id]
    jacobian_z = jnp.cross(state.data.xaxis[:, env.arm_jids, :],
        position[:, None, :] - state.data.xanchor[:, env.arm_jids, :])[:, :, 2]
    velocity = jnp.sum(jacobian_z * state.arm_cmd_qvel, axis=-1)
    return startup_potential(
        ball_z_m=state.data.xpos[:, env.ball_body_id, 2],
        ball_vz_m_s=state.data.qvel[:, env.ball_vadr + 2],
        racket_z_m=position[:, 2],
        ball_radius_m=state.model.geom_size[:, env.ball_geom_id, 0],
        command_vz_m_s=velocity, gravity_m_s2=state.dr_gravity_z,
        phase=state.hit_count, group=state.v225_group,
        initial_y_m=state.v225_reset_y_offset_m, weight=weight)


def validate_config(cfg: StartupPotentialConfig) -> None:
    from fixed_plane_v289_coordinated_stroke import validate_config as previous
    previous(cfg)
    weight = cfg.v290_startup_potential_weight
    if not math.isfinite(weight) or not 0. <= weight <= 256.:
        raise ValueError('V290 startup potential must be finite in[0,256]')
    if weight > 0. and (not cfg.v225_enabled or cfg.v279_energy_progress_weight <= 0.
            or any(getattr(cfg, name) != 0. for name in REPLACED_REWARD_FIELDS)
            or getattr(cfg, 'fixed_plane_v167_global_reward_multiplier', 1.) != 1.):
        raise ValueError('V290 replaces only the five experimental preparation prices')


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source) != 17 or source[11].name != START_STAGE:
        raise ValueError('V290 requires all17 physical V289 lessons')
    result = []
    for i, stage in enumerate(source):
        cfg = StartupPotentialConfig(**vars(stage.cfg))
        if i >= 11:
            cfg = replace(cfg, v290_startup_potential_weight=96.,
                **{name:0. for name in REPLACED_REWARD_FIELDS})
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg, notes=(
            'V290 reward-only analytic startup potential replaces five experimental '
            'preparation prices. Hard-near Phi is bounded0--96: phase0 zero; '
            'phase1/2 true ball energy plus clearance-qualified published-reference '
            'upward readiness; phase3+ constant carry. Reward is exactly '
            '0.9995*Phi(next)-Phi(previous); true termination zeroes next Phi, '
            'time-limit truncation retains it for unchanged bootstrap. No new '
            'state/history/observation, event count, plant or control. All other '
            'rewards,17 V281 domains/gates and numerical PPO/BC/KL remain exact. '
            'Latest full actor/critic/Adam/history, all32 minibatches, no cap, '
            'reset, rewind, rejection, rollback or forced advancement.')))
    return result


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v289_coordinated_stroke import validate_training_contract as previous
    previous(settings)
    if settings.gamma != DISCOUNT:
        raise ValueError('V290 potential discount must equal unchanged PPO gamma')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V290 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0))
        tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 752
                or payload.get('step') != 3254648832 + (update - 624) * 131072
                or int(payload['opt'].t) != 804062 + (update - 624) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V290 requires latest complete V289 S12 after128 actual new updates')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V290 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V290 rejects another source or from-scratch launch')
