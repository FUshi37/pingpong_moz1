"""Frozen module derivatives on independent training seeds;1MiB output scope."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .attribution_common import source_report
from .audit_full_prior_world import AuditReader, release_cpu_heap
from .auxiliary_refresh import restore_auxiliary
from .build_training_seed_bank import load_training_bank
from .contact_geometry import surface_features
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .control_projection import project_reference
from .course_control import control_step
from .full_prior_world import full_prior_world_step, world_actor_observation
from .imagined_env import EpisodeContract, finite_physics, physical_bounds, select_lanes
from .kinematics import forward_kinematics
from .migration import actor_mean
from .module_gradient_tools import cast_floats, private_clone, probe_direction, distribution
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .pilot_checkpoint import load_pilot_actor
from .resources import StorageBudget
from .robot_support_model import support_features
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .support_losses import racket_kinematics
from .world_control import ControlContract


def phase_plan(cfg: dict, guard: RunGuard) -> dict:
    plan=json.loads(authenticate(repository_root(),cfg['module_gradient_diagnostic']['plan']).read_text())
    if (not plan['authorized'] or plan['parameter_updates'] or plan['physics_changes']
            or plan['reward_changes'] or plan['constraint_changes'] or plan['max_new_output_bytes']!=2**20):
        raise ValueError('This diagnostic requires the bounded no-update authorization')
    guard.budget=StorageBudget(guard.budget.root,min(guard.budget.max_bytes,
        plan['phase_start_used_bytes']+plan['max_new_output_bytes']),guard.budget.min_free_bytes)
    guard.budget.check(additional_bytes=128*1024)
    return plan


def checkpoint(guard: RunGuard) -> None:
    guard.check()
    guard.budget.check(additional_bytes=128*1024)


def load_runtime(cfg: dict, guard: RunGuard, plan: dict) -> tuple:
    root=repository_root()/cfg['resources']['output_root']
    reader=AuditReader(root/'support_replay_full_a01',guard.budget,max_bytes=0)
    params,options,refs,parent,ar=load_components(cfg,reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    native=build_native_environment(cfg,n_envs=1)
    control=ControlContract.from_native(native.environment)
    bank,meta=load_training_bank(cfg,plan['training_bank'],reader=reader,options=options,
        control_contract=control,model_references=dict(refs,auxiliary=ar))
    ids=np.asarray(plan['source_indices'])
    if [meta['source_rows'][i] for i in ids]!=plan['source_rows']:
        raise ValueError('Independent source plan changed')
    state,_=bank.gather(ids)
    if any(meta['source_rows'][i]['split']!='train' for i in ids):
        raise ValueError('Only training sources are authorized')
    restored,_=restore_auxiliary(cfg,plan['auxiliary'])
    auxiliary=dict(parameters=restored['parameters'],scale=restored['scale'])
    actor,_=load_pilot_actor(cfg,plan['actor'])
    return state,actor,params['robot'],auxiliary,options,control,dict(refs,auxiliary=plan['auxiliary'])


def capture_states(state: Any, actor: Any, robot: Any, auxiliary: Any,
                   options: dict, control: Any, episode: Any, guard: RunGuard) -> tuple:
    """Keep first own-model hold and contact states;never read future teacher labels."""
    n=len(state.episode_steps);context=state.context;goal=state.task.goal_xyz
    @jax.jit
    def step(world):
        action=actor_mean(actor,world_actor_observation(world,goal))
        after,trace=full_prior_world_step(robot,auxiliary,world,action,context=context,
            control_contract=control,task_goal_xyz=goal,ar1_enabled=False,**options)
        within,_=physical_bounds(after.physics,after.cached_racket_pose,control,episode)
        return after,trace,within&finite_physics(after.physics),action
    world=state.world;control_states=world;control_raw=actor_mean(actor,world_actor_observation(world,goal))
    found_hold=np.zeros(n,bool);found_contact=np.zeros(n,bool);active=np.ones(n,bool)
    hold_time=np.full(n,-1,int);contact_time=np.full(n,-1,int);captured=None
    for tick in range(64):
        checkpoint(guard)
        after,trace,usable,raw=step(world)
        projection=trace['control']['projection']
        hold=np.asarray(projection.fallback_to_hold|projection.nonlinear_backstop)&active&~found_hold
        control_states=select_lanes(jnp.asarray(hold),world,control_states)
        control_raw=jnp.where(jnp.asarray(hold)[:,None],raw,control_raw)
        hold_time[hold]=tick;found_hold|=hold
        overlap=np.asarray(trace['physics']['contact']['overlap'])&active[None]
        first=overlap.argmax(axis=0);exists=overlap.any(axis=0)
        pick=lambda x:x[jnp.asarray(first),jnp.arange(n)]
        candidate=jax.tree.map(pick,dict(q=trace['physics']['q_before_rad'],dq=trace['physics']['dq_before_rad_s'],
            ball=trace['physics']['contact']['before'],commands=trace['control']['delayed_commands'],
            racket=trace['physics']['racket_before'],solver=trace['physics']['solver_terms']))
        if captured is None:captured=candidate
        use=exists&~found_contact
        captured=select_lanes(jnp.asarray(use),candidate,captured)
        contact_time[use]=tick*5+first[use];found_contact|=use
        world=select_lanes(jnp.asarray(active),after,world)
        active &= np.asarray(usable)
    evidence=dict(source_count=n,found_projection_hold=int(found_hold.sum()),found_contact=int(found_contact.sum()),
        contact_missing_indices=np.flatnonzero(~found_contact).tolist(),hold_control_ticks=hold_time.tolist(),
        first_contact_substeps=contact_time.tolist(),capture_horizon_controls=64,
        origin='own-model rollout from existing independent native training seeds;not new native labels')
    step.clear_cache();release_cpu_heap()
    return control_states,control_raw,captured,np.flatnonzero(found_contact),evidence


def geometry_modes(surface: Any, ball: Any, position: Any, rotation: Any, radius: float) -> Any:
    """Mirror the native sphere/convex branch decisions without changing its output."""
    from mujoco.mjx._src import collision_convex, math
    def one(p,x,r):
        local=r.T@(p-x);faces=surface.convex.face;normals=surface.convex.face_normal
        support=jnp.sum((local[None]-normals*radius-faces[:,0])*normals,axis=-1)
        face_id=jnp.argmax(support);face=faces[face_id];normal=normals[face_id]
        pt=collision_convex._project_pt_onto_plane(local,face[0],normal)
        p0=jnp.roll(face,1,axis=0);side=jnp.cross(face-p0,normal)
        distances=jnp.sum((pt-p0)*side,axis=-1);on_face=jnp.all(distances<=0.)
        score=jnp.where(jnp.all(side==0,axis=-1)|(distances<0),1e12,distances)
        edge_id=jnp.argmin(score)
        closest=math.closest_segment_point(p0[edge_id],face[edge_id],pt)
        chosen=jnp.where(on_face,pt,closest)
        vector,length=math.normalize_with_norm(chosen-local)
        return jnp.stack((face_id,jnp.where(on_face,-1,edge_id),jnp.any(support>=0).astype(jnp.int32),
            on_face.astype(jnp.int32),(jnp.dot(chosen,vector)>0).astype(jnp.int32),(length<1e-6).astype(jnp.int32)))
    return jax.vmap(one)(ball,position,rotation)


def normalized_directions(seed: int, shape: tuple) -> np.ndarray:
    values=np.random.default_rng(seed).normal(size=shape)
    return (values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)).astype(np.float32)


def component_functions(data: dict, *, high_precision: bool, seed: int) -> dict:
    """Build private functions;FP32 modules retain the actual production arithmetic."""
    from . import neural,robot_support_model,solver_auxiliary,soft_contact,hybrid_contact,tangent_contact
    dtype=jnp.float64 if high_precision else jnp.float32
    d=cast_floats(data,dtype);world=d['control_world'];raw=d['control_raw'];context=d['context'];control=d['control']
    opt=d['options'];chain=opt['chain'];surface=opt['surface'];contract=opt['contact_contract'];radius=opt['ball_radius_m']
    cp=d['contact'];q,dq,commands=cp['q'],cp['dq'],cp['commands'];ball=cp['ball'];racket=cp['racket'];solver=cp['solver']
    n=len(raw);m=len(q)
    a_dir=jnp.asarray(normalized_directions(seed,(n,7)),dtype)
    q_dir=jnp.asarray(normalized_directions(seed+1,(n,13)),dtype)*.01
    net_dir=jnp.asarray(normalized_directions(seed+2,(m,47)),dtype)
    ball_dir=jnp.asarray(normalized_directions(seed+3,(m,3)),dtype)*.001
    force_dir=jnp.asarray(normalized_directions(seed+4,(m,12)),dtype)*jnp.asarray([.1]*3+[1.]*3+[10.]*6,dtype)
    mlp=private_clone(neural.mlp,float64=True) if high_precision else neural.mlp
    robot_fn=private_clone(robot_support_model.support_robot_step,overrides={'mlp':mlp}) if high_precision else robot_support_model.support_robot_step
    aux_fn=private_clone(solver_auxiliary.predict,overrides={'mlp':mlp}) if high_precision else solver_auxiliary.predict
    prior_fn=private_clone(tangent_contact.tangent_prior_increment,float64=True) if high_precision else tangent_contact.tangent_prior_increment
    if high_precision:
        soft_fn=private_clone(soft_contact.soft_contact_increment,float64=True)
        hybrid_fn=private_clone(hybrid_contact.hybrid_contact_step,float64=True,overrides={'soft_contact_increment':soft_fn})
        contact_fn=private_clone(tangent_contact.tangent_contact_step,overrides={
            'hybrid_contact_step':hybrid_fn,'soft_contact_increment':soft_fn,'tangent_prior_increment':prior_fn})
    else:contact_fn=tangent_contact.tangent_contact_step
    off=lambda count:jnp.zeros((count,1),jnp.int32)
    def fingerprint(p):
        return jnp.stack((p.fallback_to_hold,p.current_safe,p.nonlinear_backstop,p.intervention),axis=-1)
    def fk(alpha):
        with jax.default_matmul_precision('highest'):
            value=jax.vmap(lambda x:forward_kinematics(chain,x))(world.physics.q_rad+alpha*q_dir)
        return jnp.concatenate((value.site_position/.01,value.site_rotation.reshape(n,9)),axis=-1),off(n)
    def projection(alpha):
        denominator=(control.projection.velocity_limit[None]*control.action_velocity_scale
            *context.action_scale_mult[:,None])
        requested=jnp.clip(jnp.clip(raw+alpha*a_dir,-1,1)*denominator,
            -control.projection.velocity_limit[None],control.projection.velocity_limit[None])
        p=project_reference(chain,control.projection,world.physics.q_rad,world.control.reference_q_rad,
            requested)
        y=jnp.concatenate(((p.reference_q-world.control.reference_q_rad)/.01,
            p.accepted_qvel/control.projection.velocity_limit),axis=-1)
        return y,fingerprint(p)
    def control8(alpha):
        def tick(current,_):
            after,trace=control_step(current,raw+alpha*a_dir,world.physics.q_rad,context,chain=chain,contract=control)
            return after,(trace['delayed_commands'],fingerprint(trace['projection']))
        final,(cmd,modes)=jax.lax.scan(tick,world.control,None,length=8)
        y=jnp.concatenate(((cmd['q'][-1,-1]-world.physics.q_rad[:,:7])/.01,cmd['dq'][-1,-1],
            cmd['ddq'][-1,-1]/100.,(final.reference_q_rad-world.control.reference_q_rad)/.01),axis=-1)
        return y,jnp.swapaxes(modes,0,1)
    features=support_features(q,dq,commands)
    def unpack(f):
        q1=jnp.concatenate((f[:,:7],f[:,35:41]),axis=-1)
        dq1=jnp.concatenate((f[:,7:14],f[:,41:47]),axis=-1)
        cmd=dict(q=f[:,:7]+f[:,14:21],dq=f[:,21:28],ddq=f[:,28:35])
        return q1,dq1,cmd
    def robot_net(alpha):
        z=(features-opt['robot_scale'].input_mean)/opt['robot_scale'].input_scale+alpha*net_dir
        with jax.default_matmul_precision('highest'):value=mlp(d['robot'],z)
        return value,off(m)
    def robot_transition(alpha):
        f=features+alpha*net_dir*opt['robot_scale'].input_scale
        q1,dq1,cmd=unpack(f)
        with jax.default_matmul_precision('highest'):
            nextq,nextdq=robot_fn(d['robot'],opt['robot_scale'],q1,dq1,cmd)
        return jnp.concatenate(((nextq-q)/.01,nextdq),axis=-1),off(m)
    def auxiliary(alpha):
        scale=d['auxiliary']['scale'];f=features+alpha*net_dir*scale.input_scale
        value=aux_fn(d['auxiliary']['parameters'],scale,f)/scale.residual_scale
        return value,off(m)
    def geo(alpha):
        p=ball.position_m+alpha*ball_dir
        with jax.default_matmul_precision('highest'):
            g=surface_features(surface,p,racket['geom_position'],racket['rotation'],radius)
            modes=geometry_modes(surface,p,racket['geom_position'],racket['rotation'],radius)
        y=jnp.concatenate((g['sphere_gap_m'][:,None]/.001,
            (g['contact_position_world_m']-ball.position_m)/.001,g['contact_normal_world']),axis=-1)
        return y,modes
    geometry=surface_features(surface,ball.position_m,racket['geom_position'],racket['rotation'],radius)
    physical=dict(ball_position_m=ball.position_m,ball_velocity_m_s=ball.velocity_m_s,
        ball_omega_world_rad_s=ball.omega_world_rad_s,geom_position_m=racket['geom_position'],
        racket_linear_velocity_m_s=racket['velocity'],racket_angular_velocity_rad_s=racket['omega'])
    def force_modes(g,pred):
        f=pred.force_contact_n
        saturated=jnp.linalg.norm(f[:,1:]/jnp.asarray(contract.friction,dtype),axis=-1)>=f[:,0]*(1-1e-6)
        return jnp.stack((g['sphere_gap_m']<0.,f[:,0]>0.,saturated),axis=-1)
    def fixed_prior(alpha):
        v=ball.velocity_m_s+alpha*force_dir[:,:3];omega=ball.omega_world_rad_s+alpha*force_dir[:,3:6]
        with jax.default_matmul_precision('highest'):
            pred=prior_fn(contract,geometry,solver+alpha*force_dir[:,6:],**dict(physical,
                ball_velocity_m_s=v,ball_omega_world_rad_s=omega))
        return jnp.concatenate((pred.delta_velocity_world_m_s/.08,pred.delta_omega_world_rad_s/30),axis=-1),force_modes(geometry,pred)
    packet=dict(geom_position_m=racket['geom_position'],geom_rotation_world=racket['rotation'],
        racket_linear_velocity_m_s=racket['velocity'],racket_angular_velocity_rad_s=racket['omega'],
        joint_q_rad=q,delayed_q_command_rad=commands['q'])
    def coupled(alpha):
        b=ball._replace(position_m=ball.position_m+alpha*ball_dir,
            velocity_m_s=ball.velocity_m_s+alpha*force_dir[:,:3],omega_world_rad_s=ball.omega_world_rad_s+alpha*force_dir[:,3:6])
        with jax.default_matmul_precision('highest'):
            after,trace=contact_fn(None,surface,b,packet,solver+alpha*force_dir[:,6:],
                ball_radius_m=radius,position_scale=opt['position_scale'],contract=contract,learned=False)
            modes=geometry_modes(surface,b.position_m,racket['geom_position'],racket['rotation'],radius)
        y=jnp.concatenate(((after.position_m-ball.position_m)/.003,
            after.velocity_m_s/.08,after.omega_world_rad_s/30),axis=-1)
        return y,jnp.concatenate((modes,trace['overlap'][:,None]),axis=-1)
    return dict(FK=fk,projection=projection,control8steps_frozen_encoder=control8,
        execution_network=robot_net,execution_transition=robot_transition,
        auxiliary_network=auxiliary,contact_geometry=geo,soft_prior_fixed_geometry=fixed_prior,
        contact_geometry_plus_prior=coupled)


def audit(cfg: dict, args: Any, guard: RunGuard, report: dict) -> None:
    plan=phase_plan(cfg,guard)
    state,actor,robot,auxiliary,options,control,refs=load_runtime(cfg,guard,plan)
    original=numeric_tree_digest((actor,robot,auxiliary))
    world,raw,contact,selected,capture=capture_states(state,actor,robot,auxiliary,options,
        control,EpisodeContract.from_config(cfg),guard)
    report.update(plan=cfg['module_gradient_diagnostic']['plan'],model_references=refs,
        training_bank=plan['training_bank'],source_indices=plan['source_indices'],capture=capture,
        source_initial_state_sha256=numeric_tree_digest(state),parameters_before_sha256=original,
        modules={},precision_scope='diagnostic copies only;production checkpoint and simulation sources unchanged')
    if len(selected)==0:raise ValueError('No contact state captured;cannot diagnose contact modules')
    contact=jax.tree.map(lambda x:x[selected],contact)
    data=dict(control_world=world,control_raw=raw,context=state.context,control=control,
        options=options,contact=contact,robot=robot,auxiliary=auxiliary)
    report['contact_probe_bank_rows']=np.asarray(plan['source_indices'])[selected].tolist()
    report['feature_scales']=dict(action='normalized QVEL units;independent unit direction per source',
        FK='0.01rad times epsilon',network='training input standard deviation times epsilon',
        ball_position='0.001m times epsilon',ball_velocity='0.1m/s times epsilon',
        ball_spin='1rad/s times epsilon',solver_terms='10m/s2 or rad/s2 times epsilon',
        control8='8steps with measured encoder state held at captured value;module isolation,not a task rollout',
        branch_scope='projection endpoint flags;native mesh face/edge/inside/separation;force saturation indicator;not every internal numerical branch')
    if args.detail_from is not None:
        from .audit_module_gradient_details import run_details
        run_details(data,plan,args.detail_from,report,guard)
        report['parameters_after_sha256']=numeric_tree_digest((actor,robot,auxiliary))
        if report['parameters_after_sha256']!=original:raise ValueError('Frozen parameters changed')
        report.update(parameters_unchanged=True,no_network_update=True,
            task_qualified=False,production_precision_unchanged=True)
        return
    for high in (False,True):
        name='FP64_diagnostic' if high else 'FP32_production'
        with jax.enable_x64(high):
            functions=component_functions(data,high_precision=high,seed=plan['seed'])
            values={}
            for module,function in functions.items():
                checkpoint(guard)
                values[module]=probe_direction(function,plan['epsilons'],dtype=jnp.float64 if high else jnp.float32,
                    check=lambda:checkpoint(guard))
                best=min((r['unchanged_branch_relative_error'].get('p95',float('inf')) for r in values[module]['rows']),default=float('inf'))
                print(f'[MODULE] {name}/{module} best_stable_p95={best:.6g} primal_max={values[module]["forward_vs_ad_primal_abs"].get("maximum")}',flush=True)
                report['modules'][name]=values
                gc.collect();release_cpu_heap()
        del functions
        jax.clear_caches();release_cpu_heap()
    report['parameters_after_sha256']=numeric_tree_digest((actor,robot,auxiliary))
    if report['parameters_after_sha256']!=original:raise ValueError('Frozen parameters changed')
    report.update(status='FROZEN_MODULE_GRADIENT_DIAGNOSIS_COMPLETE',parameters_unchanged=True,
        no_network_update=True,task_qualified=False,production_precision_unchanged=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--detail-from',type=Path)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,900.,startup_reserve_bytes=2**20) as guard:
        phase_plan(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False);report=source_report(cfg,args)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in ['audit_module_gradients.py','module_gradient_tools.py','audit_module_gradient_details.py']}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
