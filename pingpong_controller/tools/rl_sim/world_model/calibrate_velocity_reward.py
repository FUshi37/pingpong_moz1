"""Frozen replacement velocity-cost calibration; no optimizer or parameter update."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .audit_full_prior_world import release_cpu_heap
from .audit_module_gradients import load_runtime
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_features
from .course_goals import build_goal_table
from .course_rollout import course_policy_window
from .free_imagination import make_runtime as free_runtime
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract, release_reward
from .mixed_precision import make_runtime as precision_runtime
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .paired_attribution import flat_tree, target_from_checkpoint
from .paired_precision import load_actor, random32_clone
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget
from .velocity_reward import (VelocityRewardContract, candidate_course_step, clone_globals,
                              diagnostic_course_step, velocity_costs)


OBJECTIVES=('base_return','old_xy','old_z','velocity_xy','velocity_z')


def objective_function(runtime, initial, actor_keys, table, episode, reward, std):
    observed=diagnostic_course_step(runtime)
    window_fn=random32_clone(course_policy_window,dict(course_step=observed,
        course_critic_features=course_critic_features))
    options=dict(stage_id='S01',control_contract=runtime.control,
        episode_contract=episode,reward_contract=reward,**runtime.options)
    def function(actor, critic):
        window=window_fn(actor,runtime.robot,runtime.auxiliary,initial,actor_keys,table,
            horizon=64,exploration_std=std,**options)
        tr=window.transitions
        base=sum(value for key,value in tr.reward_terms.items() if key not in ('velocity_xy','velocity_z'))
        batch=ImaginedBatch(base,tr.discount,tr.trace_continue,tr.valid)
        values=frozen_value(critic,tr.next_critic_features)
        zero=jnp.zeros_like(values)
        result=jnp.stack([actor_return(batch,values)]+[
            actor_return(batch._replace(reward=tr.reward_terms[name]),zero)
            for name in ('apex_xy','apex_z','velocity_xy','velocity_z')],axis=-1)
        modes=jnp.stack((tr.valid,tr.terminated,tr.model_truncated,
            tr.task_events['valid_release']>0,tr.task_events['valid_apex']>0),axis=-1)
        return result,dict(modes=modes,valid=tr.valid,reward=base,
            releases=tr.task_events['valid_release'].sum(axis=0),terminated=window.environment.terminated,
            model_truncated=window.environment.model_truncated)
    return function


def coefficient_grid(gram, means, weights, criteria):
    gram=np.asarray(gram,float);means=np.asarray(means,float)
    base=np.array([1.,0.,0.,0.,0.]);base_norm=np.sqrt(max(gram[0,0],0.))
    baseline_z=gram[2,0]/max(base_norm,1e-30)
    baseline_xy=gram[1,0]/max(base_norm,1e-30)
    rows=[]
    for xy in weights:
        for z in weights:
            vector=np.array([1.,-1.,-1.,xy,z])
            norm=np.sqrt(max(vector@gram@vector,0.))
            cosine=float(vector@gram@base/max(norm*base_norm,1e-30))
            height=float(gram[2]@vector/max(norm,1e-30));horizontal=float(gram[1]@vector/max(norm,1e-30))
            cost_xy=abs(xy*means[3])/max(abs(means[1]),1e-30)
            cost_z=abs(z*means[4])/max(abs(means[2]),1e-30)
            norm_ratio=float(norm/max(base_norm,1e-30))
            checks=dict(finite=bool(np.isfinite([cosine,height,horizontal,cost_xy,cost_z,norm_ratio]).all()),
                cost_xy=criteria['cost_ratio_min']<=cost_xy<=criteria['cost_ratio_max'],
                cost_z=criteria['cost_ratio_min']<=cost_z<=criteria['cost_ratio_max'],
                total_gradient=criteria['total_gradient_ratio_min']<=norm_ratio<=criteria['total_gradient_ratio_max'],
                direction=cosine>=criteria['baseline_cosine_min'],
                height=height>=criteria['height_direction_retention_min']*baseline_z if baseline_z>0 else height>=0,
                xy=horizontal>=baseline_xy-1e-8)
            checks={key:bool(value) for key,value in checks.items()}
            rows.append(dict(xy_weight=xy,z_weight=z,eligible=all(checks.values()),checks=checks,
                mean_candidate_return=float(means@vector),cost_xy_ratio=float(cost_xy),cost_z_ratio=float(cost_z),
                total_gradient_ratio=norm_ratio,baseline_gradient_cosine=cosine,
                old_height_directional_derivative=height,old_xy_directional_derivative=horizontal,
                baseline_height_directional_derivative=float(baseline_z),baseline_xy_directional_derivative=float(baseline_xy)))
    return rows


def local_native_calibration(plan, reward):
    result=json.loads(authenticate(repository_root(),plan['native_audit']).read_text())
    seen={}
    starts=np.asarray(result['windows']['starts']).ravel()
    for h in (8,1,32,64,128):
        for row in result['release_alignment'][str(h)]['pairs']:
            stamp=int(starts[row['window']]*5+row['native_release_substep'])
            key=(row['source_index'],stamp)
            seen.setdefault(key,row)
    rows=list(seen.values())
    p=jnp.asarray([r['native_position_m'] for r in rows]);v=jnp.asarray([r['native_velocity_m_s'] for r in rows])
    g=jnp.asarray([r['goal_xyz_m'] for r in rows]);accepted=jnp.ones(len(rows),bool)
    old=release_reward(p,v,g,accepted,contract=reward)
    new,info=velocity_costs(p,v,g,accepted,contract=reward)
    def old_f(vel):
        t=release_reward(p,vel,g,accepted,contract=reward)
        return (t['apex_xy']+t['apex_z']).sum()
    def new_f(vel):
        t,_=velocity_costs(p,vel,g,accepted,contract=reward)
        return sum(t.values()).sum()
    old_grad,new_grad=jax.grad(old_f)(v),jax.grad(new_f)(v)
    ideal=np.asarray(info['target_velocity_m_s']-v)
    # Test the existing continuous inverse against actual1ms semi-implicit flight.
    targets=np.asarray(info['target_velocity_m_s']);positions=np.asarray(p).copy();vel=targets.copy()
    apices=np.full_like(positions,np.nan);done=np.zeros(len(rows),bool)
    for tick in range(1000):
        nextv=vel+np.array([0.,0.,-.00981]);nextp=positions+nextv*.001
        crossing=(vel[:,2]>0)&(nextv[:,2]<=0)&~done
        fraction=np.clip(vel[:,2]/.00981,0.,1.)
        apices[crossing]=(positions+fraction[:,None]*(nextp-positions))[crossing]
        done|=crossing;positions,vel=nextp,nextv
        if done.all():break
    return dict(unique_releases=len(rows),source_count=len({r['source_index'] for r in rows}),
        above_2deg_releases=sum(max(abs(x) for x in r['native_roll_pitch_deg'])>2 for r in rows),
        invalid_inverse_count=int((~np.asarray(info['valid'])).sum()),
        target_velocity_xyz_quantiles=np.quantile(targets,[0,.5,.95,1.],axis=0).tolist(),
        scale_xyz_quantiles=np.quantile(np.asarray(info['scale_xyz_m_s']),[0,.5,.95,1.],axis=0).tolist(),
        old_cost_xy=distribution(old['apex_xy']),old_cost_z=distribution(old['apex_z']),
        unit_velocity_cost_xy=distribution(new['velocity_xy']),unit_velocity_cost_z=distribution(new['velocity_z']),
        old_velocity_gradient_norm=distribution(np.linalg.norm(old_grad,axis=-1)),
        new_velocity_gradient_norm=distribution(np.linalg.norm(new_grad,axis=-1)),
        old_gradient_towards_target_dot=distribution(np.sum(np.asarray(old_grad)*ideal,axis=-1)),
        new_gradient_towards_target_dot=distribution(np.sum(np.asarray(new_grad)*ideal,axis=-1)),
        discrete1ms_apex_error_m=distribution(np.linalg.norm(apices-np.asarray(g),axis=-1)),
        source_pairs=[dict(source_index=r['source_index'],release_substep=k[1]) for k,r in seen.items()],
        scope='native states used only for reward calibration; no native returns enter a learner')


def audit(cfg,args,guard,report,plan):
    population=json.loads(authenticate(repository_root(),plan['population_plan']).read_text())
    population=dict(population,actor=plan['actor'])
    loader=clone_globals(load_runtime,load_pilot_actor=load_actor)
    initial,actor,robot,auxiliary,options,control,refs=loader(cfg,guard,population)
    critic=target_from_checkpoint(plan['learner'])
    frozen=numeric_tree_digest((actor,critic,robot,auxiliary))
    reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    episode=EpisodeContract.from_config(cfg)
    goals=np.unique(np.asarray(initial.task.goal_xyz),axis=0)
    table=build_goal_table(goals,static_admitted=np.ones(len(goals),bool))
    keys=jax.random.split(jax.random.PRNGKey(plan['seed']+2),len(plan['source_indices']))
    report.update(model_references=refs,parameters_before_sha256=frozen,initial_state_sha256=numeric_tree_digest(initial),
        objective_names=OBJECTIVES,local_native=local_native_calibration(plan,reward),runs={})
    for mode in ('bounded_2deg','free_orientation'):
        runtime=(free_runtime(robot,auxiliary,options,control) if mode=='free_orientation'
                 else precision_runtime('fp32',robot,auxiliary,options,control))
        for label,std in [('deterministic',0.),('sampled',.03)]:
            guard.check();name=mode+'_'+label
            fn=objective_function(runtime,initial,keys,table,episode,reward,std)
            def scalar(params,weights):
                value,trace=fn(params,critic)
                return jnp.sum(value.mean(axis=0)*weights),(value,trace)
            derivative=jax.jit(jax.value_and_grad(scalar,has_aux=True))
            forward=jax.jit(fn)
            center,trace=jax.device_get(forward(actor,critic))
            gradients=[];deltas=[];mode_changes=[];started=time.monotonic()
            for index,objective in enumerate(OBJECTIVES):
                guard.check()
                (_, (value,tr)),gradient=jax.device_get(derivative(actor,jnp.eye(5,dtype=jnp.float32)[index]))
                flat=flat_tree(gradient)
                if not np.isfinite(flat).all() or not np.isfinite(value).all():
                    raise FloatingPointError('Nonfinite calibration gradient or value')
                gradients.append(flat)
                deltas.append(float(np.max(np.abs(value-center))))
                mode_changes.append(int(np.any(tr['modes']!=trace['modes'],axis=(0,2)).sum()))
                print(f'[VELOCITY_GRAD] {name}/{objective} norm={np.linalg.norm(flat):.7g}',flush=True)
            matrix=np.stack(gradients);gram=matrix@matrix.T;means=center.mean(axis=0)
            grid=coefficient_grid(gram,means,plan['weights'],plan['criteria'])
            report['runs'][name]=dict(mean_objectives=means.tolist(),per_source_objectives=center.tolist(),
                gradient_gram=gram.tolist(),gradient_norms=np.linalg.norm(matrix,axis=-1).tolist(),
                parameter_count=matrix.shape[1],grid=grid,forward_vs_ad_primal_max=deltas,
                forward_vs_ad_mode_changed_lanes=mode_changes,valid_controls=int(trace['valid'].sum()),
                releases=np.asarray(trace['releases']).tolist(),terminated=int(trace['terminated'].sum()),
                model_truncated=int(trace['model_truncated'].sum()),wall_s=time.monotonic()-started)
            print('[VELOCITY_GRID] '+name+' eligible='+str([(r['xy_weight'],r['z_weight']) for r in grid if r['eligible']]),flush=True)
            derivative.clear_cache();forward.clear_cache();del derivative,forward,matrix,gradients
            jax.clear_caches();release_cpu_heap()
    first=next(iter(report['runs'].values()))['grid']
    eligible=[(row['xy_weight'],row['z_weight']) for i,row in enumerate(first)
              if all(run['grid'][i]['eligible'] for run in report['runs'].values())]
    selected=min(eligible,key=lambda x:(abs(np.log2(x[0]))+abs(np.log2(x[1])),x)) if eligible else None
    report['jointly_eligible_coefficients']=eligible;report['selected_coefficients']=selected
    # A no-update smoke checks the actual replacement path on the same initial
    # seed and deterministic action. No synthetic velocity is applied to physics.
    runtime=free_runtime(robot,auxiliary,options,control)
    candidate=VelocityRewardContract(enabled=True,xy_weight=selected[0] if selected else 1.,z_weight=selected[1] if selected else 1.)
    from .full_prior_world import world_actor_observation
    from .migration import actor_mean
    raw=actor_mean(actor,world_actor_observation(initial.world,initial.task.goal_xyz))
    opts=dict(stage_id='S01',control_contract=control,episode_contract=episode,reward_contract=reward,**options)
    original=jax.jit(lambda s,a:runtime.course_step(robot,auxiliary,s,a,s.task.goal_xyz,**opts))
    replacement=candidate_course_step(runtime,candidate)
    changed=jax.jit(lambda s,a:replacement(robot,auxiliary,s,a,s.task.goal_xyz,**opts))
    base,tr=original(initial,raw);new,ntr=changed(initial,raw);jax.block_until_ready(new)
    report['replacement_smoke']=dict(state_digest_equal=numeric_tree_digest(base)==numeric_tree_digest(new),
        max_state_abs=max(float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float))))
                          for a,b in zip(jax.tree.leaves(base),jax.tree.leaves(new)) if np.size(a)),
        terminated_equal=bool(np.array_equal(tr.terminated,ntr.terminated)),
        reward_fields=list(ntr.reward_terms),scope='first control often precedes release; event replacement covered by analytic tests')
    after=numeric_tree_digest((actor,critic,robot,auxiliary))
    if after!=frozen:raise ValueError('Frozen parameters changed')
    report.update(status='VELOCITY_REWARD_FROZEN_CALIBRATION_COMPLETE',parameters_after_sha256=after,
        parameters_unchanged=True,default_reward_changed=False,optimizer_calls=0,closed_loop_improvement_verified=False,
        limitations=['Existing free-attitude control/model-reference gaps remain',
                     'Frozen local/pathwise gradients do not establish training gain',
                     'Continuous ballistic inverse retains existing1ms discretization bias',
                     'A selected coefficient is not training or default-promotion approval'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config();setting=cfg['outgoing_velocity_reward_candidate']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20:
        raise ValueError('Frozen reward calibration is not authorized')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients'])!=plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_velocity_reward_calibration',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_sha256={p.name:file_hash(p) for p in paths},actor_updates=0,critic_updates=0,world_updates=0)
        write_compact_json(args.output_dir/'implementation.json',dict(report=report,sources={n:(Path(__file__).parent/n).read_text()
            for n in ('velocity_reward.py','calibrate_velocity_reward.py')}),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
