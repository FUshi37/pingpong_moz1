"""V241: phase-apex outcome tracking and late horizontal-motion rewards."""

from dataclasses import dataclass, replace
import math
from types import SimpleNamespace

import jax.numpy as jnp

from fixed_plane_v240_release_balance import ReleaseBalanceConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v241'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v240'
START_STAGE = 'fplc_v241_s08_near_y_020mm_bridge'
RETENTION_STAGE = 'fplc_v241_s01_workspace_alignment'


@dataclass(frozen=True)
class PhysicalOutcomeConfig(ReleaseBalanceConfig):
    """Default-off reward fields; no physical state or observation additions."""

    v241_physical_apex_huber_weight: float = 0.
    v241_physical_apex_huber_delta_m: float = .05
    v241_late_ball_xy_weight: float = 0.


def validate_config(cfg):
    weights = (float(cfg.v241_physical_apex_huber_weight), float(cfg.v241_late_ball_xy_weight))
    if any(not math.isfinite(w) or w < 0. for w in weights):
        raise ValueError('V241 reward weights must be finite and nonnegative')
    if (not math.isfinite(cfg.v241_physical_apex_huber_delta_m)
            or cfg.v241_physical_apex_huber_delta_m <= 0.):
        raise ValueError('V241 Huber delta must be finite and positive')
    if any(w > 0. for w in weights) and not cfg.v225_enabled:
        raise ValueError('V241 requires the preserved fixed-plane population')
    if weights[1] > 0. and (cfg.fixed_plane_v166_physical_ball_xy_weight <= 0.
            or bool(getattr(cfg, 'v231_nominal_reward_parity', False))):
        raise ValueError('V241 requires the existing all-cohort base XY cost')


def release_goal_hit_count(*, confirmed_hit_count, contact_time_s, last_counted_hit_time_s):
    """Keep the contacted phase if confirmation already occurred before release."""
    count = jnp.asarray(confirmed_hit_count)
    contact_time = jnp.asarray(contact_time_s)
    already_counted = ((count > 0) & (contact_time >= 0.)
                       & (jnp.asarray(last_counted_hit_time_s) >= contact_time))
    return jnp.maximum(count - already_counted.astype(count.dtype), 0)


def physical_apex_huber_penalty(*, release_event, previous_hit_count,
                                predicted_apex_z_m, target_apex_z_m,
                                weight_per_m2: float, delta_m: float):
    """Track each real [N] post-first phase apex, including unconfirmed retries.

    Count zero is excluded so initial static-support separation is never
    labeled a dynamic juggling outcome. Both high and low errors are priced;
    the linear Huber tail keeps deep weak outcomes distinct instead of flat.
    """
    if not math.isfinite(weight_per_m2) or weight_per_m2 < 0.:
        raise ValueError('Outcome weight must be finite and nonnegative')
    if not math.isfinite(delta_m) or delta_m <= 0.:
        raise ValueError('Huber delta must be finite and positive')
    active = jnp.asarray(release_event, dtype=bool) & (jnp.asarray(previous_hit_count) >= 1)
    error = jnp.abs(jnp.asarray(predicted_apex_z_m) - jnp.asarray(target_apex_z_m))
    shape = jnp.where(error <= delta_m, error * error, 2. * delta_m * error - delta_m ** 2)
    return (jnp.where(active, -float(weight_per_m2) * shape, 0.), active,
            jnp.where(active, error, 0.))


def late_ball_xy_penalty(*, existing_cost, previous_hit_count,
                         base_weight: float, extra_weight: float):
    """Price existing confirmed-hit XY events from hit four in every cohort."""
    if (not math.isfinite(base_weight) or base_weight <= 0.
            or not math.isfinite(extra_weight) or extra_weight < 0.):
        raise ValueError('Late XY weights must be finite with a positive base')
    return jnp.where(jnp.asarray(previous_hit_count) >= 3,
                     jnp.asarray(existing_cost) * (extra_weight / base_weight), 0.)


def build_stages(source):
    """Replace two partial startup costs; preserve all domains and gates."""
    if len(source) != 17 or source[7].name != 'fplc_v240_s08_near_y_020mm_bridge':
        raise ValueError('V241 requires the complete V240 curriculum')
    result = []
    for stage in source:
        cfg = replace(PhysicalOutcomeConfig(**vars(stage.cfg)),
            v237_release_shortfall_weight=0., v239_release_apex_weight=0.,
            v241_physical_apex_huber_weight=30000.,
            v241_physical_apex_huber_delta_m=.05, v241_late_ball_xy_weight=2.)
        validate_config(cfg)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace('fplc_v240_', 'fplc_v241_', 1),
            notes='V241 replaces V237/V239 partial startup losses with a '
                  'symmetric physical phase-apex Huber loss on every release '
                  'after one confirmed hit, including weak retries and every '
                  'cohort. Initial static support is excluded. Existing phase '
                  'goals remain unchanged; weight30000/m2, delta0.05m. Existing '
                  'ball-XY cost receives extra2 from hit4 in every cohort; '
                  'earlier V240 cohort costs remain. All17 domains, absolute '
                  'gates,72/393/7,physics/control/DR and uncapped1024x128 PPO '
                  'remain unchanged. Restore latest full V240 actor/critic/Adam.'))
    return result


def validate_training_contract(settings):
    from fixed_plane_v240_release_balance import validate_training_contract as previous
    if settings.advance_retention_stage_name != RETENTION_STAGE:
        raise ValueError('V241 retains nominal absolute validation')
    previous(SimpleNamespace(**{**vars(settings),
        'advance_retention_stage_name': 'fplc_v240_s01_workspace_alignment'}))


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic, allow_obs_dim_migration):
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V241 requires full uncapped72/393/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        expected = str(payload.get('stage_name', '')).replace('fplc_v240_', 'fplc_v241_', 1)
        if (resume_curriculum_state or requested_start_stage != expected
                or int(payload.get('stage_index', 0)) < 8):
            raise ValueError('V241 starts fresh reward statistics in the current recovery lesson')
    elif source == PROFILE:
        if not resume_curriculum_state or requested_start_stage != payload.get('stage_name'):
            raise ValueError('V241 own continuation must restore exact history')
    else:
        raise ValueError('V241 rejects another source')
