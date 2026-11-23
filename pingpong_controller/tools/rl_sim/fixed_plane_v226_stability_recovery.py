"""V226 recovery primitives with unchanged 72-D actor and nominal behavior."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import jax.numpy as jnp
from fixed_plane_v225_near_side import NearSideRecoveryConfig, sample_episode
from fixed_contact_plane_training import early_racket_xy_anchor_weight

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v226'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v225'
PARENT_SHA256='b102831e018585b7127668fcc40a8de258325e2a7a58551f02ddc3b6ddce2608'
START_STAGE='fplc_v226_s01_workspace_alignment'
STABILITY_LIMITS={
    'ball_mean_m_s':.022427797306882536,
    'ball_rms_m_s':.02598830132295352,
    'racket_mean_m_s':.03981705965351012,
    'racket_late_rms_m_s':.048369445546174505,
    'racket_late_omega_rms_rad_s':.22821480921795842,
}


@lru_cache(maxsize=1)
def reference_parameters():
    """Authenticate the immutable passed-S5 actor for paired retention checks."""
    from validate_juggle_mjx_ppo import load_checkpoint
    import jax
    path = Path('/home/yangzhe/Project/pingpong_controller/pingpong_controller/'
                'outputs/rl_sim/fplc_v225_gpu1_near_side_1024x128_seed20266510_'
                '20261016_r1/05_fplc_v225_s05_placed_xy_08mm.pkl')
    if hashlib.sha256(path.read_bytes()).hexdigest() != PARENT_SHA256:
        raise ValueError('Frozen V226 stability reference changed')
    return jax.tree.map(jnp.asarray, load_checkpoint(path)['params'])


@dataclass(frozen=True)
class StabilityRecoveryConfig(NearSideRecoveryConfig):
    """Default-off reward-only successor; no new actor/critic/runtime fields."""
    v226_enabled: bool = False
    v226_anchor_return_speed_m_s: float = .02
    fixed_plane_v166_physical_ball_xy_weight: float = 0.


def build_stages(source):
    """Retain all12 V225 domains and insert feasible small recovery bridges."""
    if len(source)!=12 or source[5].name!='fplc_v225_s06_near_y_050mm':
        raise ValueError('V226 requires the original complete V225 course')
    items=[(stage,stage.name.split('_',3)[3],{}) for stage in source[:5]]
    items += [(source[5],f'near_y_{int(cap*1000):03d}mm_bridge',dict(
        v225_near_y_max_m=cap,v225_nominal_fraction=nominal,
        episode_target_y_range_m=(-.02,cap)))
        for cap,nominal in [(.005,.8),(.010,.75),(.020,.7),(.035,.7)]]
    items += [(stage,stage.name.split('_',3)[3],{}) for stage in source[5:8]]
    items += [(source[8],'inward_momentum_010mms_bridge',dict(
        v225_momentum_delta_limit_m_s=.01,v225_momentum_target_cap_m_s=.020))]
    items += [(stage,stage.name.split('_',3)[3],{}) for stage in source[8:]]
    result=[]
    for index,(stage,label,changes) in enumerate(items,1):
        cfg=StabilityRecoveryConfig(**vars(stage.cfg),v226_enabled=True,
            fixed_plane_v166_physical_ball_xy_weight=2.)
        cfg=replace(cfg,**changes)
        result.append(replace(stage,name=f'fplc_v226_s{index:02d}_{label}',cfg=cfg,
            min_updates=120 if index==17 else 60,
            max_recent_mean_hit_vxy=STABILITY_LIMITS['ball_mean_m_s'],
            max_recent_rms_hit_vxy=STABILITY_LIMITS['ball_rms_m_s'],
            max_recent_mean_hit_racket_vxy=STABILITY_LIMITS['racket_mean_m_s'],
            max_recent_rapid_steady_racket_vxy_rms_m_s=STABILITY_LIMITS['racket_late_rms_m_s'],
            max_recent_rapid_steady_racket_omega_rms_rad_s=STABILITY_LIMITS['racket_late_omega_rms_rad_s'],
            notes='V226 keeps72/393/7, original dynamics/control/observation and all '
                  'V225 target domains, strengthens stability gates to the passedS5 '
                  'observed envelope. Interpolated5/10/20/35mm bridges and one10mm/s '
                  'momentum bridge. The original3000-weight early center penalty '
                  'uses a shrinking0.02m/s reachable allowance only for non-nominal '
                  'resets; nominal cost is exact. Existing physical-ball-XY squared '
                  'event cost weight2 targets RMS without action filtering. No '
                  'relaxed height/hit/survival/return gates or forced advancement.'))
    return result


def validate_return_speed(speed_m_s: float) -> None:
    """Keep the reward reference inside the existing 0.03m/s motion deadband."""
    if not math.isfinite(float(speed_m_s)) or not 0. < speed_m_s <= .03:
        raise ValueError('V226 return-reference speed must be in (0,.03] m/s')


def sample_recovery_episode(keys, cfg):
    """Use paired real templates; interpolate bridges below measured20--50mm."""
    cap=float(cfg.v225_near_y_max_m)
    if cap<=0. or cap>=.05:
        return sample_episode(keys,cfg)
    group,offset,velocity=sample_episode(keys,replace(cfg,v225_near_y_max_m=.05))
    fraction=cap/.05
    return group,offset*fraction,velocity*fraction


def reachable_anchor_penalty(original, *, distance_m, initial_y_m,
        elapsed_s, group, hit_count, cfg, dt_s: float):
    """Replace only non-nominal early cost with a shrinking reachable envelope.

    The center goal never moves. A0.02m/s default envelope acknowledges the
    displaced initial state and shrinks to zero in finite time; it cannot
    grant a permanent free region or reward standing still. No sensor,
    command, action, reference integration, physical state or gate is changed.
    Inputs/outputs are (N,); the penalty is already integrated over dt_s.
    """
    speed=float(cfg.v226_anchor_return_speed_m_s)
    validate_return_speed(speed)
    counts=jnp.asarray(hit_count)
    non_nominal=jnp.asarray(group)>0
    active=non_nominal & (counts<=int(cfg.early_racket_xy_anchor_hit_count))
    allowance=jnp.where(active,jnp.maximum(jnp.abs(initial_y_m)
        -speed*jnp.maximum(jnp.asarray(elapsed_s),0.),0.),0.)
    error=jnp.maximum(jnp.asarray(distance_m)-allowance,0.)
    excess=(jnp.maximum(error-float(cfg.early_racket_xy_anchor_deadband_m),0.)
            /float(cfg.early_racket_xy_anchor_scale_m))
    shape=jnp.where(excess>2.,4.+4.*(excess-2.),excess**2)
    weight=early_racket_xy_anchor_weight(counts,
        weight=cfg.early_racket_xy_anchor_penalty_weight,
        transition_weight=cfg.early_racket_xy_anchor_transition_weight)
    changed=jnp.where(active,-weight*shape*float(dt_s),0.)
    return jnp.where(non_nominal,changed,jnp.asarray(original)),allowance


def validate_resume_payload(payload, *, requested_start_stage,
        resume_curriculum_state, reset_optimizer, reset_critic,
        allow_obs_dim_migration, allow_bounded_probe_start=False):
    """Restore passedS5 full state; never promote the regressedS6 or a probe."""
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ['obs_dim','critic_obs_dim','act_dim'])!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V226 requires unreset uncapped full72/393/7')
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        allowed={START_STAGE}
        if allow_bounded_probe_start:allowed.add('fplc_v226_s10_near_y_050mm')
        if (resume_curriculum_state or requested_start_stage not in allowed
                or payload.get('stage_name')!='fplc_v225_s05_placed_xy_08mm'
                or payload.get('stage_index')!=5 or payload.get('stage_update')!=129
                or payload.get('step')!=1443758080):
            raise ValueError('V226 pins passedV225S5 with fresh reward history')
        if any(float(payload.get('last_row',{}).get(k,0.))!=1. for k in
               ['convergence/stage_converged','advance_eval/passed','retention_eval/passed']):
            raise ValueError('V226 parent did not pass all original graduation checks')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V226 own resume requires exact stage/history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
    else:
        raise ValueError('V226 rejects another source profile')
