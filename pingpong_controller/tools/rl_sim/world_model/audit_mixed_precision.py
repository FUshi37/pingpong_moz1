"""Bounded frozen H64 probes for explicit WMJ precision runtimes."""
from __future__ import annotations
import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any
import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import ImaginedBatch, actor_return, frozen_value
from .artifacts import write_json_unique, write_array_archive_unique
from .attribution_common import source_report
from .audit_attribution_world import target_critic
from .audit_full_prior_world import release_cpu_heap
from .audit_module_gradients import load_runtime
from .contracts import authenticate, load_config, repository_root
from .full_prior_world import world_actor_observation
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .migration import actor_mean
from .mixed_precision import make_runtime
from .module_gradient_tools import distribution
from .resources import StorageBudget
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def phase_plan(cfg: dict, guard: Any) -> dict:
    plan=json.loads(authenticate(repository_root(),cfg['mixed_precision_candidate']['plan']).read_text())
    if (not plan['authorized'] or plan['parameter_updates'] or plan['default_replacement']
            or plan['max_new_output_bytes']!=2*2**20):
        raise ValueError('Frozen numerical-candidate authorization changed')
    guard.budget=StorageBudget(guard.budget.root,min(guard.budget.max_bytes,
        plan['phase_start_used_bytes']+plan['max_new_output_bytes']),guard.budget.min_free_bytes)
    check(guard)
    return plan


def check(guard: Any) -> None:
    guard.check()
    guard.budget.check(additional_bytes=192*1024)


def rollout_function(runtime: Any, initial: Any, actor: Any, critic: Any,
                     episode: Any, reward: Any) -> Any:
    """Keep the original reward/bootstrap semantics and expose only frozen traces."""
    initial=runtime.prepare_course(initial)
    options=dict(stage_id='S01',control_contract=runtime.control,
        episode_contract=episode,reward_contract=reward,**runtime.options)
    def components(bias):
        policy=dict(actor,out=dict(actor['out'],b=actor['out']['b']+bias.astype(jnp.float32)))
        def tick(state,_):
            raw=actor_mean(policy,world_actor_observation(state.world,state.task.goal_xyz))
            after,tr=runtime.course_step(runtime.robot,runtime.auxiliary,state,raw,
                state.task.goal_xyz,**options)
            observed,ct=runtime.controller(state.world.control,raw,state.world.physics.q_rad,
                state.context,chain=runtime.options['chain'],contract=runtime.control)
            p=ct['projection']
            modes=jnp.stack((p.fallback_to_hold,p.current_safe,p.nonlinear_backstop,
                tr.terminated,tr.model_truncated,tr.task_events['valid_release']>0,
                tr.task_events['valid_apex']>0),axis=-1)
            return after,(tr,modes,raw,after.world.physics.ball.position_m,
                after.world.cached_racket_pose[:,:3],
                jnp.where(tr.valid & ~tr.model_truncated,
                    jnp.max(jnp.abs(observed.reference_q_rad-after.world.control.reference_q_rad),axis=-1),0.))
        final,(tr,modes,raw,ball,pose,control_delta)=jax.lax.scan(jax.checkpoint(tick),initial,None,length=64)
        values=frozen_value(critic,tr.next_critic_features).astype(tr.reward.dtype)
        batch=ImaginedBatch(tr.reward,tr.discount,tr.trace_continue,tr.valid)
        direct=actor_return(batch,jnp.zeros_like(values))
        bootstrap=actor_return(batch._replace(reward=jnp.zeros_like(tr.reward)),values)
        output=jnp.stack((direct,bootstrap),axis=-1)
        return output,dict(modes=modes,raw_action=raw,ball_position=ball,racket_position=pose,
            control_observer_delta=control_delta,valid=tr.valid,
            reward=tr.reward,final_q=final.world.physics.q_rad,final_dq=final.world.physics.dq_rad_s,
            hits=final.events.impact_id,task_hits=final.task.hit_count,apexes=final.task.apex_count,
            terminated=final.terminated,model_truncated=final.model_truncated)
    return components


def difference_row(plus: tuple, minus: tuple, center: dict, jacobian: Any,
                   direction: Any, epsilon: float) -> dict:
    vp,ap=plus;vm,am=minus
    central=(np.asarray(vp,float)-np.asarray(vm,float))/(2*epsilon)
    automatic=np.einsum('nkj,j->nk',jacobian,direction)
    changed=np.any((np.asarray(ap['modes'])!=center['modes'])|
                   (np.asarray(am['modes'])!=center['modes']),axis=(0,2))
    direct_a, direct_f=automatic[:,0],central[:,0]
    magnitude=np.maximum(np.abs(direct_a),np.abs(direct_f))
    informative=magnitude>1e-4;common=~changed & informative
    relative=np.abs(direct_a-direct_f)/np.maximum(magnitude,1e-8)
    return dict(epsilon=epsilon,automatic_components=automatic.tolist(),central_components=central.tolist(),
        summed_automatic=automatic.sum(axis=0).tolist(),summed_central=central.sum(axis=0).tolist(),
        center_endpoint_changed_indices=np.flatnonzero(changed).tolist(),
        informative_direct=int(informative.sum()),common_mode_informative_direct=int(common.sum()),
        direct_relative_error=distribution(relative[informative]),
        common_mode_direct_relative_error=distribution(relative[common]),
        direct_direction_agreement=float(np.mean(direct_a[informative]*direct_f[informative]>0)) if informative.any() else None,
        common_mode_direction_agreement=float(np.mean(direct_a[common]*direct_f[common]>0)) if common.any() else None)


def audit(cfg: dict, args: Any, guard: Any, report: dict, plan: dict) -> None:
    initial,actor,robot,auxiliary,options,control,refs=load_runtime(cfg,guard,plan)
    critic=target_critic(cfg,'u20');frozen=numeric_tree_digest((actor,critic,robot,auxiliary))
    episode=EpisodeContract.from_config(cfg)
    reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    with jax.enable_x64(args.mode!='fp32'):
        runtime=make_runtime(args.mode,robot,auxiliary,options,control)
        fn=rollout_function(runtime,initial,actor,critic,episode,reward)
        zero=jnp.zeros(7,jnp.float32)
        # Trace before expensive compilation so carry/interface errors fail early.
        shape=jax.eval_shape(fn,zero)
        state=runtime.prepare_course(initial)
        report.update(initial_state_sha256=numeric_tree_digest(initial),model_references=refs,
            dtype_contract=dict(physical_q=str(state.world.physics.q_rad.dtype),
                control_reference=str(state.world.control.reference_q_rad.dtype),
                observation_camera=str(state.world.observation.camera.held_position_m.dtype),
                actor_weight=str(actor['out']['w'].dtype),world_weight=str(runtime.robot[0]['w'].dtype)),
            parameters_before_sha256=frozen)
        forward=jax.jit(fn)
        jvp=jax.jit(lambda direction:jax.jvp(fn,(zero,),(direction,),has_aux=True))
        check(guard);started=time.monotonic();center=jax.device_get(forward(zero));cold=time.monotonic()-started
        report['cold_forward_compile_and_first_s']=cold
        gradients=[];jvp_times=[];primal_deltas=[];ad_mode_changes=[]
        for joint in range(7):
            check(guard);started=time.monotonic()
            primal,tangent,aux=jax.device_get(jvp(jnp.eye(7,dtype=jnp.float32)[joint]))
            jvp_times.append(time.monotonic()-started);gradients.append(tangent)
            primal_deltas.append(float(np.max(np.abs(primal-center[0]))))
            ad_mode_changes.append(int(np.any(aux['modes']!=center[1]['modes'],axis=(0,2)).sum()))
            print(f'[MIXED_GRAD] {args.mode} basis={joint} time={jvp_times[-1]:.3f}s',flush=True)
        jacobian=np.stack(gradients,axis=-1)
        if not np.isfinite(jacobian).all():raise FloatingPointError('Nonfinite frozen H64 derivatives')
        direction=np.random.default_rng(plan['seed']).normal(size=7);direction/=np.linalg.norm(direction)
        own=jacobian.sum(axis=(0,1));own/=max(np.linalg.norm(own),1e-12)
        report.update(center_components=np.asarray(center[0]).tolist(),output_bias_jacobian=jacobian.tolist(),
            forward_vs_jvp_primal_max=primal_deltas,jvp_primal_mode_changed_lanes=ad_mode_changes,
            jvp_times_s=jvp_times,finite_differences={})
        for name,d in [('shared',direction),('own_gradient',own)]:
            rows=[]
            for epsilon in plan['epsilons']:
                check(guard)
                plus=jax.device_get(forward(jnp.asarray(epsilon*d,jnp.float32)))
                minus=jax.device_get(forward(jnp.asarray(-epsilon*d,jnp.float32)))
                row=difference_row(plus,minus,center[1],jacobian,d,epsilon);rows.append(row)
                print(f'[MIXED_FD] {args.mode}/{name} eps={epsilon} auto={row["summed_automatic"][0]:.6g} fd={row["summed_central"][0]:.6g}',flush=True)
            report['finite_differences'][name]=dict(direction=d.tolist(),rows=rows)
        times=[]
        for repeat in range(3):
            check(guard);started=time.monotonic();jax.block_until_ready(forward(zero));times.append(time.monotonic()-started)
        report['warm_forward_times_s']=times
        arrays=dict(center[1],return_components=np.asarray(center[0]))
        report['center_trace']=write_array_archive_unique(args.output_dir/'center_trace.npz',arrays,guard.budget)
        report['center_summary']=dict(valid_steps=int(np.asarray(center[1]['valid']).sum()),
            task_hits=np.asarray(center[1]['task_hits']).tolist(),apexes=np.asarray(center[1]['apexes']).tolist(),
            terminated=np.asarray(center[1]['terminated']).tolist(),
            model_truncated=np.asarray(center[1]['model_truncated']).tolist(),
            control_observer_delta=distribution(center[1]['control_observer_delta']))
        forward.clear_cache();jvp.clear_cache()
    report['parameters_after_sha256']=numeric_tree_digest((actor,critic,robot,auxiliary))
    if report['parameters_after_sha256']!=frozen:raise ValueError('Frozen parameters changed')
    report.update(status='FROZEN_MIXED_H64_COMPLETE',parameters_unchanged=True,task_qualified=False)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--mode',choices=('fp32','mixed','fp64_reference'),required=True)
    parser.add_argument('--task',choices=('gradients','actions'),default='gradients')
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.,startup_reserve_bytes=2**20) as guard:
        plan=phase_plan(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False);report=source_report(cfg,args)
        report.update(schema='wmj_v1_frozen_mixed_precision',mode=args.mode,plan=cfg['mixed_precision_candidate']['plan'])
        sources=['mixed_precision.py','audit_mixed_precision.py']
        if args.task=='actions':sources.extend(['audit_mixed_actions.py','mixed_action_seed.py'])
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in sources}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            if args.task=='gradients':audit(cfg,args,guard,report,plan)
            else:
                from .audit_mixed_actions import audit_actions
                audit_actions(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)

if __name__=='__main__':main()
