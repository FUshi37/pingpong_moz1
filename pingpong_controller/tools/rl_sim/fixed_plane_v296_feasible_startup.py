"""Default-off feasible startup labels and matched early physical XY pricing.

The physical proposal is never applied to control. It is a privileged reward
label copied from a recorded diagnostic; policy observations/actions stay72/7.
"""
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence
import jax.numpy as jnp
from fixed_plane_v295_mid_near_preimpact import MidNearPreparationConfig

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v296_feasible_startup'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v295_mid_near_preimpact'
START_STAGE='fplc_v281_s12_near_y_100mm'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'
EXTRA_FIELDS=frozenset({'v296_action_prepare_weight_per_s','v296_early_xy_extra_weight'})

@dataclass(frozen=True)
class FeasibleStartupConfig(MidNearPreparationConfig):
    """Two reward prices; no state, observation or control extension."""
    v296_action_prepare_weight_per_s: float=0.
    v296_early_xy_extra_weight: float=0.


def near_support(initial_y_m, group):
    fraction=jnp.clip((jnp.asarray(initial_y_m)-.035)/.010,0.,1.)
    return jnp.where(jnp.asarray(group)==1,fraction*fraction*(3.-2.*fraction),0.)


def action_label_penalty(*,raw_action,target_action,requested_vz,feasible_vz,
        blend,weight_per_s:float,dt_s:float):
    """Negative pseudo-Huber label error plus unresolved upward shortfall.

    All N transitions remain ordinary PPO samples. Linear tails avoid a dead
    Gaussian score far from the label. Labels use the same safe projection as
    the diagnostic, but never replace the policy's actual physical command.
    """
    if (not math.isfinite(weight_per_s) or weight_per_s<0.
            or not math.isfinite(dt_s) or dt_s<=0.):
        raise ValueError('Invalid V296 action-label reward units')
    raw,target=jnp.asarray(raw_action),jnp.asarray(target_action)
    if raw.ndim!=2 or raw.shape[-1]!=7 or raw.shape!=target.shape:
        raise ValueError('V296 labels require identical(N,7) actions')
    if any(jnp.asarray(x).shape!=raw.shape[:1] for x in (requested_vz,feasible_vz,blend)):
        raise ValueError('V296 label scalars require(N,)')
    error=(raw-target)/.25
    joint_cost=jnp.mean(jnp.sqrt(1.+error*error)-1.,axis=-1)
    shortfall=jnp.maximum(jnp.asarray(requested_vz)-jnp.asarray(feasible_vz),0.)/.25
    feasibility_cost=jnp.sqrt(1.+shortfall*shortfall)-1.
    return -weight_per_s*dt_s*jnp.asarray(blend)*(joint_cost+feasibility_cost)


def early_xy_penalty(*,existing_cost,previous_hits,initial_y_m,group,
        base_weight:float,extra_weight:float):
    """Reuse the exact confirmed-hit event/speed/normalization, phases1/2."""
    if (not math.isfinite(base_weight) or base_weight<=0.
            or not math.isfinite(extra_weight) or extra_weight<0.):
        raise ValueError('Invalid V296 physical XY weights')
    phase=(jnp.asarray(previous_hits)==1)|(jnp.asarray(previous_hits)==2)
    return (jnp.asarray(existing_cost)*(extra_weight/base_weight)
        *near_support(initial_y_m,group)*phase)


def validate_config(cfg:FeasibleStartupConfig)->None:
    from fixed_plane_v295_mid_near_preimpact import validate_config as previous
    previous(cfg)
    action,xy=cfg.v296_action_prepare_weight_per_s,cfg.v296_early_xy_extra_weight
    if (not math.isfinite(action) or not 0.<=action<=2048.
            or not math.isfinite(xy) or not 0.<=xy<=16.):
        raise ValueError('Invalid V296 reward prices')
    if (action>0.)!=(xy>0.):
        raise ValueError('V296 requires coupled launch and physical XY rewards')
    if action>0. and (cfg.action_command_mode!='velocity'
            or not cfg.cartesian_pose_projection_enabled
            or cfg.fixed_plane_v166_physical_ball_xy_weight<=0.
            or xy!=cfg.v241_late_ball_xy_weight):
        raise ValueError('V296 requires original safe QVEL and matching late XY price')


def build_stages(source:Sequence[Any])->list[Any]:
    if len(source)!=17 or source[11].name!=START_STAGE:
        raise ValueError('V296 requires all17 original V295 lessons')
    result=[]
    for index,stage in enumerate(source):
        cfg=FeasibleStartupConfig(**vars(stage.cfg),
            v296_action_prepare_weight_per_s=512. if index>=11 else 0.,
            v296_early_xy_extra_weight=stage.cfg.v241_late_ball_xy_weight if index>=11 else 0.)
        validate_config(cfg)
        result.append(replace(stage,cfg=cfg,notes=(
            'V296 reward-only feasible startup label512/s and early confirmed-hit '
            'XY extra matching the existing late price. Preserve all V295 terms. '
            'Diagnostic target uses original q_ref FK, current-policy safe '
            'projection and bounded vertical increment; it is NEVER applied '
            'as an action. Label pseudo-Huber cost includes unresolved planned '
            'upward shortfall. Group1 smooth35--45mm,phase1/2,180ms TTC/30ms '
            'onset,1.39m apex,original0.665 restitution/0.9m/s cap. XY cost '
            'reuses exact V166 confirmed events and measured outgoing speed. '
            'All17 V281 domains/gates,72/393/7,control/physics/DR and all '
            'numeric1024x128 PPO/BC/KL unchanged. Latest full state/history, '
            'all32 accepted,no reset/rewind/rejection/rollback/cap or forced advance.')))
    return result


def validate_training_contract(settings:Any)->None:
    from fixed_plane_v295_mid_near_preimpact import validate_training_contract as previous
    previous(settings)

def bounded_vertical_increment(jacobian, base_velocity, velocity_limit, requested_vz):
    """Change only vertical Cartesian velocity, with bounded seven-joint QVEL.

    Arrays are (N,6,7), (N,7), (7,), (N,). Translation is world XYZ;
    angular rows use world XYZ radians/s. Arm ordering is the existing seven
    joints. Damping1e-6 bounds singular solves. This is a proposal before the
    unchanged environment's reference projection and RMP/PD.
    """
    limit = jnp.asarray(velocity_limit)
    base = jnp.clip(base_velocity, -limit, limit)
    scaled = jacobian * limit[None, None, :]
    gram = scaled @ jnp.swapaxes(scaled, -1, -2)
    unit = jnp.zeros((jacobian.shape[0], 6)).at[:, 2].set(1.)
    solve = jnp.linalg.solve(gram + 1e-6 * jnp.eye(6)[None], unit[..., None])[..., 0]
    direction = jnp.einsum('nij,ni->nj', scaled, solve) * limit[None, :]
    safe_direction = jnp.where(jnp.abs(direction) > 1e-8, direction, 1.)
    a = (-limit - base) / safe_direction
    b = (limit - base) / safe_direction
    lower = jnp.max(jnp.where(jnp.abs(direction) > 1e-8, jnp.minimum(a, b), -jnp.inf), axis=-1)
    upper = jnp.min(jnp.where(jnp.abs(direction) > 1e-8, jnp.maximum(a, b), jnp.inf), axis=-1)
    before = jnp.einsum('nj,nj->n', jacobian[:, 2, :], base)
    delta = jnp.clip(requested_vz - before, lower, upper)
    proposed = jnp.clip(base + direction * delta[:, None], -limit, limit)
    achieved = jnp.einsum('nij,nj->ni', jacobian, proposed)
    return proposed, achieved, direction


def feasible_preparation_target(env, state, raw):
    cfg = env.cfg
    # Match the existing reference projection's FK/Jacobian convention exactly.
    reference_data = state.data.replace(
        qpos=state.data.qpos.at[:,env.arm_qadr].set(state.arm_cmd_q),
        qvel=state.data.qvel.at[:,env.arm_vadr].set(0.))
    reference_data = env.batched_forward(state.model, reference_data)
    reference_position = reference_data.site_xpos[:,env.racket_site_id]
    axes = reference_data.xaxis[:,env.arm_jids,:]
    offsets = reference_position[:,None,:] - reference_data.xanchor[:,env.arm_jids,:]
    jac = jnp.concatenate((jnp.swapaxes(jnp.cross(axes,offsets),1,2),
                           jnp.swapaxes(axes,1,2)),axis=1)
    ball_z = state.data.xpos[:,env.ball_body_id,2]
    vz = state.data.qvel[:,env.ball_vadr+2]
    surface = state.data.site_xpos[:,env.racket_site_id,2] + state.model.geom_size[:,env.ball_geom_id,0]
    gravity = jnp.maximum(jnp.abs(state.dr_gravity_z), 1e-6)
    gap = ball_z-surface
    ttc = (vz+jnp.sqrt(vz*vz+2.*gravity*jnp.maximum(gap,0.)))/gravity
    incoming = vz-gravity*ttc
    outgoing = jnp.sqrt(2.*gravity*jnp.maximum(1.39-surface,0.))
    restitution = float(cfg.fixed_plane_limit_cycle_effective_restitution)
    requested = jnp.clip((outgoing+restitution*incoming)/(1.+restitution),0.,.9)
    # Use the CURRENT policy's safe XY/angular feedback, not a persistent
    # previous command. This is a read-only action proposal calculation;
    # actual env.step retains its original single projection and RMP/PD.
    denominator = (env.arm_vel_limit_rad_s[None,:]*float(cfg.action_velocity_scale)
                   *state.action_scale_mult[:,None])
    current_nominal = jnp.clip(raw*denominator,-env.arm_vel_limit_rad_s,env.arm_vel_limit_rad_s)
    current_projection = env._project_cartesian_pose_reference(
        state,state.data.qpos[:,env.arm_qadr],current_nominal)
    base_velocity = current_projection[1]
    proposed, achieved, _ = bounded_vertical_increment(
        jac,base_velocity,env.arm_vel_limit_rad_s,requested)
    denominator = (env.arm_vel_limit_rad_s[None,:]*float(cfg.action_velocity_scale)
                   *state.action_scale_mult[:,None])
    action = jnp.clip(proposed/jnp.maximum(denominator,1e-6),-1.,1.)
    x = jnp.clip((state.v225_reset_y_offset_m-.035)/.010,0.,1.)
    support = jnp.where(state.v225_group==1,x*x*(3.-2.*x),0.)
    onset = jnp.clip((.180-ttc)/.030,0.,1.)
    eligible = ((state.hit_count>=1)&(state.hit_count<=2)&~state.pending_hit
                &(vz<0.)&(gap>.001)&(ttc>0.)&(ttc<.180))
    blend = support*onset*onset*(3.-2.*onset)*eligible
    selected = raw+blend[:,None]*(action-raw)
    diag = dict(proposal_blend=blend,proposal_ttc=ttc,
        proposal_requested_vz=requested,proposal_linearized_vz=achieved[:,2],
        proposal_action=action)
    return selected,diag

def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V296 requires complete uncapped72/393/7 state')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        update = int(payload.get('stage_update', 0)); tail = payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get('stage_name') != START_STAGE or payload.get('stage_index') != 12
                or update < 2816
                or payload.get('step') != 3525181440 + (update - 2688) * 131072
                or int(payload['opt'].t) != 870110 + (update - 2688) * 32
                or not isinstance(tail, list) or not tail
                or int(tail[-1]['stage_update']) != update or tail[-1]['stage_name'] != START_STAGE):
            raise ValueError('V296 requires latest full V295 S12 state after128 actual updates and exact history')
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get('stage_name'):
                raise ValueError('V296 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload), next_stage=requested_start_stage)
    else:
        raise ValueError('V296 rejects another source or from-scratch launch')
