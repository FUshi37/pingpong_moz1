"""Same-batch native action forks from the32 corrected-control reset sources."""
from __future__ import annotations
import time

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import compare_effects, model_event_trace, summarize_outcomes
from .audit_free_imagination import take_native, endpoint_bounds
from .audit_new_pose_world import finite_intervals
from .audit_full_prior_world import release_cpu_heap
from .corrected_candidate import branch_actions, resolved_effects
from .imagined_env import finite_physics
from .native_pair import assess_native_pair
from .physical_events import scan_physical_events
from .state_adapter import numeric_tree_digest


def first_failure(alive):
    """Record every original lane, including immediate failures and censoring."""
    alive=np.asarray(alive,bool)
    return [int(np.flatnonzero(~alive[:,i])[0]+1) if (~alive[:,i]).any() else None for i in range(alive.shape[1])]


def run_actions(data,models,cfg,args,guard,report,plan):
    from mjx_topology_execution import make_collect
    env=data['environment'];adapter=data['adapter'];rt=data['runtime'];control=data['control'];episode=data['episode']
    frozen=numeric_tree_digest((data['actor'],data['robot'],models))

    def native_tick(carry,action):
        native,history,event,alive=carry
        after,_,_,_,metrics=env.step(native,action)
        trace=metrics['wmj_trace'];finite=finite_intervals(trace)
        history,_=adapter.advance(after,history)
        within,_=endpoint_bounds(env,after,adapter,control,episode)
        valid=alive[None]&within[None]&(jnp.cumsum((~finite|trace['other_ball_collision']).astype(jnp.int32),axis=0)==0)
        event,labels=scan_physical_events(event,trace,valid=valid,initialization_active=jnp.zeros_like(valid))
        alive=alive&within&jnp.all(finite&~trace['other_ball_collision'],axis=0)
        return (after,history,event,alive),(labels,after.prev_action,alive)

    scoped,execution=make_collect(native_tick,mode='project');native_step=jax.jit(scoped)
    @jax.jit
    def world_run(auxiliary,initial,context,event,actions,goals):
        def tick(carry,action):
            world,event,alive=carry
            after,tr=rt.world_step(data['robot'],auxiliary,world,action,context=context,
                control_contract=control,task_goal_xyz=goals,ar1_enabled=False,**data['options'])
            with jax.default_matmul_precision('highest'):
                trace=model_event_trace(tr['physics'],surface=data['options']['surface'],
                    ball_radius_m=data['options']['ball_radius_m'],contact_contract=data['options']['contact_contract'])
            finite=jnp.ones((5,32),bool)
            for key in ('ball_qpos_after','ball_qvel_after'):
                finite &= jnp.all(jnp.isfinite(trace[key]),axis=-1)
            finite &= jnp.all(jnp.isfinite(tr['physics']['q_after_rad']),axis=-1)
            finite &= jnp.all(jnp.isfinite(tr['physics']['dq_after_rad_s']),axis=-1)
            within,_=rt.bounds(after.physics,after.cached_racket_pose,control,episode)
            valid=alive[None]&within[None]&(jnp.cumsum((~finite).astype(jnp.int32),axis=0)==0)
            event,labels=scan_physical_events(event,trace,valid=valid,initialization_active=jnp.zeros_like(valid))
            alive=alive&within&finite_physics(after.physics)&jnp.all(finite,axis=0)
            return (after,event,alive),(labels,tr['control']['accepted_action'],alive,jnp.all(finite,axis=0))
        return jax.lax.scan(tick,(initial,event,jnp.ones(32,bool)),actions)[1]

    all_native={k:[] for k in ('base','plus','minus')};all_repeat=[];all_cases=[]
    all_world={model:{k:[] for k in all_native} for model in models}
    groups=[];started=time.monotonic()
    for group in range(4):
        guard.check();lanes=np.arange(group*8,(group+1)*8)
        mapping,actions,cases=branch_actions(data['actions'],lanes,plan['pulse_assignments'])
        native=take_native(data['native_seed'],mapping,32);history=take_native(data['history_seed'],mapping,32)
        events=jax.tree.map(lambda v:jnp.asarray(v[0,mapping]),data['events_bank'])
        initial=jax.tree.map(lambda v:v[mapping],data['initial']);context=jax.tree.map(lambda v:v[mapping],data['context'])
        goal=data['goals'][mapping];carry=(native,history,events,jnp.ones(32,bool))
        a=numeric_tree_digest(take_native(carry,np.arange(8),32));b=numeric_tree_digest(take_native(carry,np.arange(24,32),32))
        if a!=b:raise ValueError('Baseline and repeat complete seeds differ')
        labels=[];accepted=[];alive=[];repeat_errors=[];group_started=time.monotonic()
        for t in range(128):
            before=carry;carry,(lab,act,live)=native_step(carry,jnp.asarray(actions[t]))
            if group==0 and t==0:
                jax.block_until_ready(carry);instrumented=env._step_impl;env._step_impl=data['native_unobserved_step']
                plain_scoped,_=make_collect(lambda s,a:env.step(s,a)[0],mode='project');plain=jax.jit(plain_scoped)
                expected=plain(before[0],jnp.asarray(actions[t]));jax.block_until_ready(expected)
                pair=assess_native_pair(expected,carry[0],np.ones(32,bool),env,
                    limits=cfg['frozen_evaluation_numerics']['absolute_tolerances'])
                env._step_impl=instrumented;plain.clear_cache();report['action_native_pair']=pair
                if not pair['pass_']:raise ValueError('Action-fork instrumentation failed native numeric pairing')
            lab,act,live=jax.device_get((lab,act,live));labels.append(lab);accepted.append(act);alive.append(live)
            fields=dict(qpos=carry[0].data.qpos,qvel=carry[0].data.qvel,accepted=carry[0].prev_action)
            repeat_errors.append({k:float(np.max(np.abs(np.asarray(v[:8])-np.asarray(v[24:])))) for k,v in fields.items()})
            if t%32==31:
                guard.check();print(f'[CORRECTED_ACTION] group={group+1}/4 controls={t+1}/128',flush=True)
        joined=jax.tree.map(lambda *v:np.concatenate(v),*labels);accepted=np.stack(accepted);alive=np.stack(alive)
        branches={name:summarize_outcomes({k:v[:,i*8:(i+1)*8] for k,v in joined.items()})
                  for i,name in enumerate(('base','plus','minus','repeat'))}
        for branch in all_native:all_native[branch].extend(branches[branch])
        all_repeat.extend(branches['repeat'])
        for i,row in enumerate(cases):
            row.update(source_index=plan['source_indices'][row['lane']],region_id=plan['source_rows'][row['lane']]['region_id'],
                native_accepted_central_pulse_norm=float(np.linalg.norm(accepted[:,8+i]-accepted[:,16+i])))
        all_cases.extend(cases);predictions={};model_times={};model_failures={};nonfinite={}
        for name,auxiliary in models.items():
            guard.check();start=time.monotonic()
            lab,taken,live,finite=jax.device_get(world_run(auxiliary,initial,context,events,jnp.asarray(actions),goal))
            model_times[name]=time.monotonic()-start;lab=jax.tree.map(lambda v:v.reshape((-1,)+v.shape[2:]),lab)
            outcomes={branch:summarize_outcomes({k:v[:,i*8:(i+1)*8] for k,v in lab.items()})
                      for i,branch in enumerate(('base','plus','minus','repeat'))}
            for branch in all_native:all_world[name][branch].extend(outcomes[branch])
            predictions[name]=outcomes;model_failures[name]=first_failure(live);nonfinite[name]=int((~finite).sum())
            for i,row in enumerate(cases):
                row[name+'_accepted_central_pulse_norm']=float(np.linalg.norm(taken[:,8+i]-taken[:,16+i]))
        groups.append(dict(group=group,source_indices=[plan['source_indices'][i] for i in lanes],cases=cases,
            complete_base_repeat_initial_sha256=a,native=branches,world=predictions,
            native_first_failure=first_failure(alive),world_first_failure=model_failures,world_nonfinite_controls=nonfinite,
            native_repeat_endpoint_max={k:max(row[k] for row in repeat_errors) for k in repeat_errors[0]},
            wall_s=time.monotonic()-group_started,world_wall_s=model_times))
        del carry,labels,joined,accepted,lab;release_cpu_heap()
    comparisons={name:dict(all_pairs=compare_effects(all_native,world),
        repeat_resolved=resolved_effects(all_native,world,all_repeat,all_cases)) for name,world in all_world.items()}
    report['action_effects']=dict(groups=groups,comparisons=comparisons,wall_s=time.monotonic()-started,
        cases=all_cases,source_count=32,native_counterfactual_controls=16384,native_parallelism=32,execution=execution,
        modes=['base','plus','minus','repeat'],continuation='same frozen baseline raw actions;only initial10/20ms pulse differs',
        validity='per-branch first physical/task failure is terminal for all outcome labels; rollout continues only diagnostically',
        reference='analytic apex separately named;actual apex requires observed event',full_world_qualification=False)
    if frozen!=numeric_tree_digest((data['actor'],data['robot'],models)):raise ValueError('Frozen action audit changed parameters')
    native_step.clear_cache();world_run.clear_cache();jax.clear_caches();release_cpu_heap()
