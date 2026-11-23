"""V225: unchanged 72-D V172 with coherent near-side recovery populations."""
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
import numpy as np
import jax
import jax.numpy as jnp
from mjx_juggle_env import MjxFixedPlaneV159RewardConfig

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v225'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v172'
PARENT_SHA256 = '96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a'
START_STAGE = 'fplc_v225_s01_workspace_alignment'
RETENTION_STAGE = START_STAGE
TEMPLATE_PATH = Path(__file__).resolve().parent/'references/v225_mocap_near_side_templates.json'
TEMPLATE_SHA256 = 'ce0489bd200ce45ee95a6310769a7396086bffc72e333a19eafc5d7e61569bc3'
V225_RANDOMIZED_STATE_FIELDS = frozenset({
    'v225_group', 'v225_reset_y_offset_m', 'v225_target_vy_m_s',
    'v225_scheduled_step', 'v225_momentum_applied',
})


def low_disk_stop_reason(free_bytes: int) -> str | None:
    """Keep the existing reserve while identifying the owning experiment."""
    if free_bytes < int(1.5*1024**3):
        return f'V225 disk reserve reached: free={free_bytes/1024**3:.3f} GiB'
    return None


@dataclass(frozen=True)
class NearSideRecoveryConfig(MjxFixedPlaneV159RewardConfig):
    """Opt-in population/diagnostic fields never appended to actor or critic."""
    v225_enabled: bool = True
    v225_episode_weighted_convergence: bool = True
    v225_near_y_max_m: float = 0.
    v225_nominal_fraction: float = 1.
    v225_mirror_fraction: float = 0.
    v225_momentum_delta_limit_m_s: float = 0.
    v225_momentum_target_cap_m_s: float = .11
    v225_min_group_episodes: int = 8
    v225_min_hit3_rate: float = .95
    v225_min_hit12_rate: float = .90
    v225_contraction_ratio: float = .8
    v225_min_contraction_rate: float = .75


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Begin at passed V172 S10; preserve all source rewards and hard gates."""
    if len(source) != 17 or source[9].name != 'fplc_v159_gpu1_s10_clean_commit':
        raise ValueError('V225 requires the immutable complete V172 course')
    base = source[9]
    cfg = NearSideRecoveryConfig(**vars(base.cfg))
    from fixed_plane_v224_history_recovery import intersect_workspace
    low, high = intersect_workspace(cfg.cartesian_pose_position_lower_m,
        cfg.cartesian_pose_position_upper_m, (-.15, -.60, .95), (.15, -.35, 1.20), inset_m=.005)
    cfg = replace(cfg, cartesian_pose_position_lower_m=low,
        cartesian_pose_position_upper_m=high,
        fixed_contact_racket_x_bounds_m=(low[0], high[0]),
        fixed_contact_racket_y_bounds_m=(low[1], high[1]),
        fixed_contact_racket_z_bounds_m=(low[2], high[2]))
    lessons = [('workspace_alignment', {})]
    lessons += [(f'placed_xy_{int(v*1000):02d}mm', {'racket_launch_xy_jitter': v})
                for v in (.001, .002, .004, .008)]
    lessons += [(f'near_y_{int(v*1000):03d}mm', dict(v225_near_y_max_m=v,
        episode_target_y_range_m=(-.02, v), v225_nominal_fraction=nominal,
        v225_mirror_fraction=.1, v225_contraction_ratio=ratio))
        for v, nominal, ratio in [(.05, .7, .98), (.08, .6, .95), (.10, .5, .90)]]
    lessons += [(f'inward_momentum_{int(delta*1000):03d}mms',
        dict(v225_momentum_delta_limit_m_s=delta, v225_momentum_target_cap_m_s=cap,
             v225_contraction_ratio=ratio))
        for delta, cap, ratio in [(.02, .035, .90), (.04, .065, .85), (.08, .11, .80)]]
    lessons += [('unchanged_consolidation', {})]
    result = []
    for index, (label, values) in enumerate(lessons, 1):
        cfg = replace(cfg, **values)
        result.append(replace(base, name=f'fplc_v225_s{index:02d}_{label}', cfg=cfg,
            min_updates=120 if index == 12 else 60,
            notes='V225 restores submitted clean V17272/393/7 with unchanged control, '
                  'physics, reward and 2degree attitude. Intersected safe workspace; '
                  'coherent complete racket/ball reset via existing IK and native RMP '
                  'initialization, while actor/reward center remains(0,-.49). '
                  'Paired mocap Y-position/incoming-momentum templates; nominal and '
                  'opposite-side controls. At most one Y velocity perturbation in '
                  'free flight after>=3hits, never ball teleport/contact/launch support. '
                  'Worst-group and finite-impact recovery admission plus central retention. '
                  'No attitude/history/contact/persistent-acceleration DR. Simulation only.'))
    return result


def sample_episode(keys: jax.Array, cfg: NearSideRecoveryConfig):
    """Sample group(0 nominal/1 near/2 mirror), physical reset offset and vY."""
    if cfg.v225_near_y_max_m <= 0.:
        count = keys.shape[0]
        return jnp.zeros(count, jnp.int32), jnp.zeros(count), jnp.zeros(count)
    blob = TEMPLATE_PATH.read_bytes()
    if hashlib.sha256(blob).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('V225 real joint-template reference changed')
    tuples = [(x['ey_m'], x['incoming_vy_m_s']) for x in json.loads(blob)['templates']
              if x['ey_m'] <= cfg.v225_near_y_max_m]
    if not tuples:
        raise ValueError('V225 has no joint templates in the declared support')
    table = jnp.asarray(tuples, dtype=jnp.float32)
    uniform = jax.vmap(lambda k: jax.random.uniform(jax.random.fold_in(k, 22501)))(keys)
    indices = jax.vmap(lambda k: jax.random.randint(jax.random.fold_in(k, 22502), (), 0, len(tuples)))(keys)
    pair = table[indices]
    group = jnp.where(uniform < cfg.v225_nominal_fraction, 0,
        jnp.where(uniform >= 1.-cfg.v225_mirror_fraction, 2, 1)).astype(jnp.int32)
    offset = jnp.where(group == 1, pair[:, 0], jnp.where(group == 2, -jnp.minimum(pair[:, 0], .02), 0.))
    target_velocity = jnp.clip(pair[:, 1], -.02, float(cfg.v225_momentum_target_cap_m_s))
    target_velocity = jnp.where(group == 1, target_velocity,
        jnp.where(group == 2, -jnp.maximum(jnp.abs(target_velocity), .005), 0.))
    return group, offset, target_velocity


def select_intervention_rows(original, candidate, event):
    """Keep complete solver/derived state exact in every unperturbed lane."""
    return jax.tree.map(lambda old, new: jnp.where(
        event.reshape((event.shape[0],)+(1,)*(old.ndim-1)), new, old), original, candidate)


def momentum_intervention(velocity, target_vy, *, group, already_applied,
        step, scheduled_step, hit_count, in_contact, clearance_m,
        maximum_delta_m_s: float):
    """One bounded Y-only stress impulse; all histories advance normally."""
    event = ((group > 0) & ~already_applied & (step >= scheduled_step)
             & (hit_count >= 3) & ~in_contact & (clearance_m >= .05))
    delta = jnp.where(event, jnp.clip(target_vy-velocity[:, 1],
        -maximum_delta_m_s, maximum_delta_m_s), 0.)
    return velocity.at[:, 1].add(delta), event, delta


def recovery_gate(metrics: Mapping[str, Any], cfg: NearSideRecoveryConfig) -> dict[str, float]:
    """Gate every initial cohort, including failures before any valid impact."""
    if cfg.v225_near_y_max_m <= 0.:
        return {'required': 0., 'passed': 1.}
    arrays = {name: np.asarray(value) for name, value in metrics.items()}
    done = (arrays['terminated']+arrays['truncated']) > .5
    active = (np.cumsum(done, axis=0)-done) == 0
    groups = arrays['v225/group'][0].astype(int)
    hits = (arrays['hit_event_count'] > .5) & active
    impulses = (arrays['v225/momentum_event'] > .5) & active
    ey = arrays['v225/ball_ey_m']
    result = {'required': 1.}
    passed = True
    lane_contraction = np.zeros(len(groups), dtype=bool)
    lane_postkick_hit3 = np.zeros(len(groups), dtype=bool)
    for group in [0, 1, 2]:
        lanes = np.flatnonzero(groups == group)
        count = len(lanes)
        hit_counts = hits[:, lanes].sum(axis=0)
        p3 = float(np.mean(hit_counts >= 3)) if count else 0.
        p12 = float(np.mean(hit_counts >= 12)) if count else 0.
        result.update({f'group{group}_episodes': float(count),
                       f'group{group}_hit3_rate': p3, f'group{group}_hit12_rate': p12})
        ok = count >= cfg.v225_min_group_episodes and p3 >= cfg.v225_min_hit3_rate and p12 >= cfg.v225_min_hit12_rate
        if group > 0:
            successes = 0
            exposed = 0
            for lane in lanes:
                if cfg.v225_momentum_delta_limit_m_s > 0.:
                    events = np.flatnonzero(impulses[:, lane])
                    if not len(events):
                        continue
                    first = events[0]
                    exposed += 1
                else:
                    first = -1
                    exposed += 1
                following = np.flatnonzero(hits[:, lane] & (np.arange(len(hits)) > first))
                if len(following) >= 3:
                    lane_postkick_hit3[lane] = True
                    e1, e3 = abs(ey[following[0], lane]), abs(ey[following[2], lane])
                    lane_contraction[lane] = e3 <= max(.015, cfg.v225_contraction_ratio*e1)
                    successes += int(lane_contraction[lane])
            contraction = successes/count if count else 0.
            result[f'group{group}_exposure_rate'] = exposed/count if count else 0.
            result[f'group{group}_three_bounce_contraction_rate'] = contraction
            ok = ok and contraction >= cfg.v225_min_contraction_rate and exposed >= .9*count
            if cfg.v225_momentum_delta_limit_m_s > 0.:
                post3 = float(np.mean(lane_postkick_hit3[lanes])) if count else 0.
                result[f'group{group}_postkick_hit3_rate'] = post3
                ok = ok and post3 >= cfg.v225_min_hit3_rate
        passed = passed and ok
    # Realized starting physical Y support must span every introduced severity bin.
    offsets = arrays['v225/reset_y_offset_m'][0]
    for index, (low, high) in enumerate([(.02, .05), (.05, .08), (.08, .10001)], 1):
        if cfg.v225_near_y_max_m <= low:
            continue
        lanes = np.flatnonzero((groups == 1) & (offsets >= low) & (offsets < high))
        count = len(lanes)
        p3 = float(np.mean(hits[:, lanes].sum(axis=0) >= 3)) if count else 0.
        p12 = float(np.mean(hits[:, lanes].sum(axis=0) >= 12)) if count else 0.
        contraction = float(np.mean(lane_contraction[lanes])) if count else 0.
        result[f'near_bin{index}_episodes'] = float(count)
        result[f'near_bin{index}_hit3_rate'] = p3
        result[f'near_bin{index}_hit12_rate'] = p12
        result[f'near_bin{index}_contraction_rate'] = contraction
        passed = (passed and count >= cfg.v225_min_group_episodes
                  and p3 >= cfg.v225_min_hit3_rate and p12 >= cfg.v225_min_hit12_rate
                  and contraction >= cfg.v225_min_contraction_rate)
        if cfg.v225_momentum_delta_limit_m_s > 0.:
            post3 = float(np.mean(lane_postkick_hit3[lanes])) if count else 0.
            result[f'near_bin{index}_postkick_hit3_rate'] = post3
            passed = passed and post3 >= cfg.v225_min_hit3_rate
    result['passed'] = float(passed)
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    """Only exact full V172 parent or causal, uncapped V225 continuation."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V225 requires uncapped full72/393/7 without resets/migration')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_index') != 10 or payload.get('stage_update') != 314
                or payload.get('stage_name') != 'fplc_v159_gpu1_s10_clean_commit'
                or payload.get('step') != 1334837248 or int(payload['opt'].t) != 335771):
            raise ValueError('V225 pins submitted V172S10u314 with fresh course history')
        for key in ['convergence/stage_converged', 'advance_eval/passed', 'retention_eval/passed']:
            if float(payload.get('last_row', {}).get(key, 0.)) != 1.:
                raise ValueError('V225 parent did not pass its graduation')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V225 own resume requires exact stage/history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError('V225 rejects another source profile')
