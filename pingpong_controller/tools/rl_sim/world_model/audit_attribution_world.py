"""Frozen native-action world replay and policy-return derivative attribution."""

import argparse
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .actor_critic import ImaginedBatch, actor_return, frozen_value, init_critic
from .artifacts import read_array_archive, write_json_unique
from .attribution_common import population, actors, frozen_models, plan_and_budget, worker_check, source_report, initial_interface
from .attribution_checks import compare_initial_interfaces, world_event_trace
from .audit_action_effects import initial_world
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, file_hash, load_config, repository_root
from .course_env import course_step, init_course
from .evaluate_frozen import json_numeric
from .frozen_task import init_evaluation, consume_events, fail_evaluation
from .full_prior_world import full_prior_world_step, world_actor_observation
from .imagined_env import EpisodeContract, init_imagined, finite_physics, physical_bounds, select_lanes
from .imagined_rewards import RewardContract, release_reward, control_reward, approach_reward
from .migration import actor_mean
from .model_numerics import precision_scope
from .physical_events import physical_event_substep
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


TERM_NAMES=('hit','apex_xy','apex_z','precision','steady_horizontal','survival',
            'failure','action_smooth','projection','approach')


def target_critic(cfg, actor_name):
    """Read only authenticated target-critic leaves; no learner/pool restoration."""
    if actor_name=='baseline':
        return init_critic(jax.random.PRNGKey(cfg['bounded_imagination_pilot']['seed']+1),2602)
    root=repository_root()/cfg['resources']['output_root']
    refs={'u10':('pilot_train_20261110_a01/learner_u010.json','0b14f41587fc1d27c49c21a321cd00e389a8df6b39ef67d672efe69bb6d17244'),
          'u20':('pilot_train_20261110_a02/learner_u020.json','1b3ace1937e05399f534a9722c69ad33c0b33e5d3dd334b66f0a6efc29ae182c')}
    relative,digest=refs[actor_name];path=authenticate(repository_root(),dict(path=str(root/relative),sha256=digest))
    meta=json.loads(path.read_text()); specs=meta['state']['leaves']
    selected={f'leaf_{i:06d}':s for i,s in enumerate(specs) if "['learner'].target_critic" in s['path']}
    arrays=read_array_archive(path.parent/meta['archive_file'],meta['archive'],names=selected)
    result=[]
    for layer in range(3):
        values={}
        for key in ('b','w'):
            expected=f"['learner'].target_critic[{layer}]['{key}']"
            name=next(k for k,s in selected.items() if s['path']==expected);value=arrays[name];spec=selected[name]
            if (hashlib.sha256(value.tobytes()).hexdigest()!=spec['sha256'] or list(value.shape)!=spec['shape']):
                raise ValueError('Target critic leaf changed')
            values[key]=jnp.asarray(value)
        result.append(values)
    return tuple(result)


def audit(cfg,args,guard,report):
    plan=plan_and_budget(cfg,guard);p=population(cfg);lanes=np.asarray(plan['selected_lane_ids'])
    report['native_initial_digest_before_components']=numeric_tree_digest(p['initial'])
    # Build native environment before importing/loading mutable MJX component paths.
    robot,auxiliary,physics,model_refs=frozen_models(cfg,guard.budget)
    report['native_initial_digest_after_components']=numeric_tree_digest(p['initial'])
    world,context,events=initial_world(dict(native=p['initial'],events=p['events']),p['env'],physics,lanes)
    goals=p['goals'][lanes];n=len(lanes)
    initial_task=init_evaluation(goals,jax.random.PRNGKey(plan['seed']+1),jnp.asarray(lanes))
    control=p['control'];episode=EpisodeContract.from_config(cfg)
    reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    parameters=actors(cfg);frozen=numeric_tree_digest((parameters,robot,auxiliary))
    native_report=json.loads((args.native_dir/'result.json').read_text())
    if (native_report['status']!='NATIVE_FROZEN_ATTRIBUTION_COMPLETE'
            or native_report['plan']!=cfg['frozen_attribution_diagnostic']['plan']):
        raise ValueError('Frozen native initial population or completion changed')
    collision_path=args.collision_dir/'result.json'
    collision=json.loads(collision_path.read_text())
    if collision['status']!='ORIGINAL_GRAPH_COLLISION_ATTRIBUTION_COMPLETE':
        raise ValueError('Original native graph replay is incomplete')
    report['initial_comparison']=compare_initial_interfaces(initial_interface(p,lanes),collision['initial_interface'])
    report['initial_comparison'].update(reference=dict(path=str(collision_path),sha256=file_hash(collision_path)),
        native_all_arrays_hash_equal=native_report['initial_numeric_sha256']==numeric_tree_digest(p['initial']),
        entire_native_tree_hash_required=False)
    if not report['initial_comparison']['passed']:
        raise ValueError('World initial physical/control/history interface differs')
    report.update(native_result=dict(path=str(args.native_dir/'result.json'),sha256=file_hash(args.native_dir/'result.json')),
        selected_case_ids=plan['selected_case_ids'],model_references=model_refs,
        initial_world_sha256=numeric_tree_digest((world,context,events)),runs={},gradients={})

    def step(state,raw,native_active):
        before,task,event=state
        active=~task.done & native_active
        task=task._replace(done=task.done|~native_active)
        after,trace=full_prior_world_step(robot,auxiliary,before,raw,context=context,
            control_contract=control,task_goal_xyz=goals[:,0],ar1_enabled=False,**physics)
        with precision_scope(physics['matmul_precision']):
            physical=model_event_trace(trace['physics'],surface=physics['surface'],
                ball_radius_m=physics['ball_radius_m'],contact_contract=physics['contact_contract'])
        def substep(carry,inputs):
            current,labels_state=carry
            usable=~current.done
            labels_state,labels=physical_event_substep(labels_state,inputs,valid=usable,
                initialization_active=jnp.zeros(n,bool))
            current,scoring=consume_events(current,labels,goals,perturb=False)
            terms=release_reward(labels['release_position_m'],labels['release_velocity_m_s'],
                goals[:,0],scoring['valid_release'],contract=reward)
            output=world_event_trace(current,scoring,labels)
            return (current,labels_state),(output,terms)
        (task,event),(labels,releases)=jax.lax.scan(substep,(task,event),physical)
        finite=finite_physics(after.physics);within,bounds=physical_bounds(after.physics,after.cached_racket_pose,control,episode)
        code=jnp.where(~finite,1,jnp.where(~bounds['ball'],3,jnp.where(~bounds['racket'],4,
            jnp.where(~bounds['orientation'],5,jnp.where(~bounds['joints'],6,0)))))
        failure=active&(code!=0)
        task=task._replace(done=task.done&~failure,success=task.success&~failure)
        task=fail_evaluation(task,jnp.where(failure,code,0))
        terms={k:v.sum(axis=0) for k,v in releases.items() if k!='total'}
        dense=control_reward(after.control.previous_accepted_action,before.control.previous_accepted_action,
            trace['control']['clipped_action'],active=active,physical_failure=failure,dt_s=.005,contract=reward)
        terms.update({k:v for k,v in dense.items() if k!='total'})
        terms['approach']=approach_reward(before.physics.ball.position_m,before.physics.ball.velocity_m_s,
            before.cached_racket_pose[:,:3],active=active,dt_s=.005,contract=reward)
        after=select_lanes(active&finite,after,before)
        return (after,task,event),(labels,terms,after.physics.ball.position_m,after.cached_racket_pose[:,:3],active)

    forward=jax.jit(step)
    for name in parameters:
        for mode in ('deterministic','sampled'):
            key=f'{name}_{mode}'
            reference=native_report['runs'][key]
            native=json.loads(authenticate(repository_root(),reference).read_text())
            arrays=read_array_archive(args.native_dir/native['archive_file'],native['archive'])
            state=(world,initial_task,events);records=[];sums={k:np.zeros(n) for k in TERM_NAMES}
            dsums={k:np.zeros(n) for k in TERM_NAMES};position_errors=[];pose_errors=[];matched_masks=[]
            for index,(raw,active) in enumerate(zip(arrays['raw_action'],arrays['active'])):
                worker_check(guard)
                state,out=forward(state,jnp.asarray(raw),jnp.asarray(active))
                labels,terms,ball,pose,used=jax.device_get(out)
                if not all(np.isfinite(v).all() for v in terms.values()):raise FloatingPointError('Nonfinite world reward')
                for k,v in terms.items():sums[k]+=v;dsums[k]+=episode.gamma**index*v
                position_errors.append(np.linalg.norm(ball-arrays['ball_position_m'][index],axis=-1))
                pose_errors.append(np.linalg.norm(pose-arrays['racket_position_m'][index],axis=-1));matched_masks.append(used)
                for sub,lane in zip(*np.nonzero(labels['valid_release']|labels['valid_apex'])):
                    r={k:json_numeric(v[sub,lane]) for k,v in labels.items()}
                    r.update(case_id=plan['selected_case_ids'][lane],control_tick=index,substep=int(sub));records.append(r)
            task=jax.device_get(state[1]);errors=np.asarray(position_errors);pose_errors=np.asarray(pose_errors);mask=np.asarray(matched_masks)
            outcomes=[dict(case_id=case,hits=int(task.hits[i]),apexes=int(task.apexes[i]),
                failure_code=int(task.failure_code[i]),elapsed_ms=int(task.elapsed_ms[i]),
                action_source_ended_without_model_failure=bool(task.done[i] and not task.failure_code[i]))
                for i,case in enumerate(plan['selected_case_ids'])]
            rows=[]
            for case in plan['selected_case_ids']:
                for kind,vector in [('valid_release','release_velocity_m_s'),('valid_apex','apex_position_m')]:
                    a=[r for r in native['events'] if r['case_id']==case and r[kind]]
                    b=[r for r in records if r['case_id']==case and r[kind]]
                    for i,(left,right) in enumerate(zip(a,b)):
                        delta=np.asarray(right[vector])-left[vector]
                        rows.append(dict(case_id=case,kind=kind,ordinal=i+1,delta=delta.tolist(),norm=float(np.linalg.norm(delta)),
                            native_time_s=left['elapsed_ms']*.001,world_time_s=right['elapsed_ms']*.001,
                            native=left[vector],world=right[vector]))
            details=dict(actor=name,mode=mode,native_reference=reference,events=records,outcomes=outcomes,
                ordinal_pairs=rows,reward_sums={k:v.tolist() for k,v in sums.items()},
                discounted_reward_sums={k:v.tolist() for k,v in dsums.items()},
                same_clock_errors=dict(valid_control_pairs=int(mask.sum()),
                    ball_p95_m=float(np.quantile(errors[mask],.95)) if mask.any() else None,
                    racket_p95_m=float(np.quantile(pose_errors[mask],.95)) if mask.any() else None),
                semantics='Own robot/ball/control/camera recursion; identical native actions until either path ends. Ordinal events can differ in time; unmatched events remain in each outcome. No future-state corrections.')
            sha=write_json_unique(args.output_dir/(key+'.json'),details,guard.budget)
            report['runs'][key]=dict(path=str(args.output_dir/(key+'.json')),sha256=sha)
            print(f'[WORLD] {key} releases={int(task.hits.sum())} pairs={len(rows)}',flush=True)
    forward.clear_cache();jax.clear_caches();release_cpu_heap()
    # Closed-loop64-step derivative decomposition. Frozen critic weights retain
    # state derivatives. Only seven output-bias directions are probed; no update.
    initial_course=init_course(init_imagined(world,context,goals[:,0],events=events),jax.random.PRNGKey(plan['seed']),jnp.asarray(lanes))
    options=dict(stage_id='S01',control_contract=control,episode_contract=episode,reward_contract=reward,**physics)
    def components(bias,actor,critic):
        actor=dict(actor,out=dict(actor['out'],b=actor['out']['b']+bias))
        def tick(current,_):
            raw=actor_mean(actor,world_actor_observation(current.world,current.task.goal_xyz))
            following,tr=course_step(robot,auxiliary,current,raw,current.task.goal_xyz,**options)
            return following,tr
        _,transitions=jax.lax.scan(jax.checkpoint(tick),initial_course,None,length=64)
        zeros=jnp.zeros_like(transitions.reward)
        batch=ImaginedBatch(transitions.reward,transitions.discount,transitions.trace_continue,transitions.valid)
        columns=[actor_return(batch._replace(reward=transitions.reward_terms[k]),zeros) for k in TERM_NAMES]
        columns.append(actor_return(batch._replace(reward=zeros),frozen_value(critic,transitions.next_critic_features)))
        return jnp.stack(columns,axis=-1)
    # JVPs avoid differentiating with respect to any stored parameter tree.
    derivative=jax.jit(lambda actor,critic,direction:jax.jvp(lambda bias:components(bias,actor,critic),
        (jnp.zeros(7,jnp.float32),),(direction,)))
    value=jax.jit(components)
    for name,actor in parameters.items():
        critic=target_critic(cfg,name);before=numeric_tree_digest(critic);columns=[];values=None
        for joint in range(7):
            worker_check(guard)
            values,tangent=derivative(actor,critic,jnp.eye(7,dtype=jnp.float32)[joint])
            jax.block_until_ready(tangent);columns.append(np.asarray(tangent))
        jacobian=np.stack(columns,axis=-1);values=np.asarray(values)
        if not np.isfinite(jacobian).all():raise FloatingPointError('Nonfinite frozen derivative')
        direction=jacobian.sum(axis=(0,1));norm=float(np.linalg.norm(direction));direction=direction/max(norm,1e-12)
        probes=[]
        for eps in (.0003,.001):
            plus=np.asarray(value(jnp.asarray(eps*direction,dtype=jnp.float32),actor,critic))
            minus=np.asarray(value(jnp.asarray(-eps*direction,dtype=jnp.float32),actor,critic))
            automatic=float((jacobian*direction).sum());central=float((plus-minus).sum()/(2*eps))
            probes.append(dict(epsilon=eps,automatic=automatic,central=central,
                relative_error=abs(automatic-central)/max(abs(automatic),abs(central),1e-8)))
        if numeric_tree_digest(critic)!=before:raise ValueError('Frozen critic changed')
        report['gradients'][name]=dict(term_names=list(TERM_NAMES)+['target_bootstrap'],
            per_case_discounted_return_terms=values.tolist(),output_bias_jacobian=jacobian.tolist(),
            finite_difference=probes,critic_sha256=before,horizon_controls=64,exploration_std=0.,
            meaning='Diagnostic seven-output-bias derivatives at existing stationary-release cases;not the full training-pool gradient;finite differences may cross contact/task modes.')
        print(f'[GRADIENT] {name} norm={norm:.6g} finite_difference={probes}',flush=True)
    report.update(status='WORLD_FROZEN_ATTRIBUTION_COMPLETE',parameters_unchanged=numeric_tree_digest((parameters,robot,auxiliary))==frozen)
    if not report['parameters_unchanged']:raise ValueError('Frozen models changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--native-dir',type=Path,required=True)
    parser.add_argument('--collision-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        plan_and_budget(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False);report=source_report(cfg,args)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in ['audit_attribution_world.py','attribution_common.py']}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('CUDA required')
            audit(cfg,args,guard,report)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
