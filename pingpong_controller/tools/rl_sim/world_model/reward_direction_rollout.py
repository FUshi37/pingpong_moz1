"""Same-state closed-loop native/model probes, with realized reward accounting."""
from __future__ import annotations
import time

import jax
import jax.numpy as jnp
import numpy as np

from .audit_free_imagination import endpoint_bounds
from .audit_new_pose_world import finite_intervals
from .course_env import course_critic_features
from .corrected_reward import observed_step
from .full_prior_world import world_actor_observation
from .imagined_rewards import control_reward,approach_reward
from .migration import actor_mean
from .native_pair import assess_native_pair
from .physical_events import scan_physical_events
from .reward_direction import METRIC_KEYS,metric_task,observed_course,summarize_stream
from .task import record_failure
from .teacher_trace import instrument_environment
from .state_adapter import numeric_tree_digest
from .velocity_reward import VelocityRewardContract,candidate_course_step
from .actor_critic import frozen_value


def failure_code(bounds,nonfinite,other):
    return (nonfinite.astype(jnp.int32)+2*other.astype(jnp.int32)
            +4*(~bounds['ball']).astype(jnp.int32)+8*(~bounds['racket']).astype(jnp.int32)
            +16*(~bounds['joints']).astype(jnp.int32))


def make_native_step(data):
    env=data['environment'];adapter=data['adapter'];reward=data['reward'];rt=data['runtime']
    task_step=metric_task(observed_step(rt).__globals__['task_substep'])
    def step(carry,actor,noise):
        native,history,event,task,alive=carry
        raw=actor_mean(actor,adapter.observation(native,history,task.goal_xyz))+noise
        after,_,_,_,metrics=env.step(native,raw);trace=metrics['wmj_trace']
        finite=finite_intervals(trace);bad=~finite|trace['other_ball_collision']
        valid=alive[None]&(jnp.cumsum(bad.astype(jnp.int32),axis=0)==0)
        event,labels=scan_physical_events(event,trace,valid=valid,initialization_active=jnp.zeros_like(valid))
        def consume(task,label):
            new,terms,events=task_step(task,label,task.goal_xyz,contract=reward)
            return new,(terms,events)
        newtask,(release,events)=jax.lax.scan(consume,task,labels)
        history,_=adapter.advance(after,history)
        within,bounds=endpoint_bounds(env,after,adapter,data['control'],data['episode'])
        numeric=~jnp.all(finite,axis=0);other=jnp.any(trace['other_ball_collision'],axis=0)
        failed=alive&(~within|numeric|other);code=jnp.where(failed,failure_code(bounds,numeric,other),0)
        newtask,_=record_failure(newtask,failed)
        terms={k:v.sum(axis=0) for k,v in release.items() if k!='total'}
        dense=control_reward(after.prev_action,native.prev_action,jnp.clip(raw,-1.,1.),active=alive,
            physical_failure=failed,dt_s=.005,contract=reward)
        terms.update({k:v for k,v in dense.items() if k!='total'})
        terms['approach']=approach_reward(native.data.qpos[:,env.ball_qadr:env.ball_qadr+3],
            native.data.qvel[:,env.ball_vadr:env.ball_vadr+3],native.data.site_xpos[:,env.racket_site_id],
            active=alive,dt_s=.005,contract=reward)
        terms={k:jnp.where(alive,v,0.) for k,v in terms.items()}
        counts={k:jnp.sum(v,axis=0) for k,v in events.items()}
        return (after,history,event,newtask,alive&~failed),dict(terms=terms,events=counts,valid=alive,
            failure=failed,failure_code=code,raw=raw,accepted=after.prev_action)
    return step


def make_world_step(data):
    rt=data['runtime'];step=observed_course(rt)
    options=dict(stage_id='S01',control_contract=data['control'],episode_contract=data['episode'],
                 reward_contract=data['reward'],**data['options'])
    def world(state,actor,noise):
        raw=actor_mean(actor,world_actor_observation(state.world,state.task.goal_xyz))+noise
        after,tr=step(data['robot'],data['auxiliary'],state,raw,state.task.goal_xyz,**options)
        bounds={k:tr.task_events['bound_'+k] for k in ('ball','racket','joints')}
        fail=tr.terminated|tr.model_truncated
        code=jnp.where(fail,failure_code(bounds,tr.model_truncated,jnp.zeros_like(fail)),0)
        keys=('valid_release','valid_apex',*METRIC_KEYS)
        return after,dict(terms=tr.reward_terms,events={k:tr.task_events[k] for k in keys},valid=tr.valid,
            failure=fail,failure_code=code,raw=raw,accepted=after.world.control.previous_accepted_action)
    return world


def collect_branches(data,plan,guard,report,directions,noise,critic):
    from mjx_topology_execution import make_collect
    rt=data['runtime'];env=data['environment'];original=env._step_impl
    report['native_observer']=instrument_environment(env,data['bundle'].stack,support_trace=True)
    tick=make_native_step(data);scoped,execution=make_collect(tick,mode='project');native_step=jax.jit(scoped)
    world_step=jax.jit(make_world_step(data));report['execution']=execution
    outputs={};frozen=numeric_tree_digest((data['actor'],data['robot'],data['auxiliary'],critic))
    for mode,std in plan['policy_modes'].items():
        direction=directions[mode];mode_runs={}
        branches=[('base',0.)]+[(sign+'_'+str(e),value*e) for e in plan['epsilons'] for sign,value in [('plus',1),('minus',-1)]]+[('repeat',0.)]
        for branch,offset in branches:
            guard.check();started=time.monotonic()
            actor=jax.tree.map(lambda p,g:p+jnp.asarray(offset,jnp.float32)*g,data['actor'],direction)
            native=(data['native'],data['history'],data['events'],data['initial'].task,jnp.ones(32,bool))
            world=data['initial'];streams={'native':[],'world':[]};boot={};state_deltas=[]
            for t in range(128):
                before=native
                native,nt=native_step(native,actor,jnp.asarray(std*noise[t]))
                world,wt=world_step(world,actor,jnp.asarray(std*noise[t]))
                if mode=='deterministic' and branch=='base' and t==0:
                    jax.block_until_ready(native);instrumented=env._step_impl;env._step_impl=original
                    plain_scoped,_=make_collect(lambda s,a:env.step(s,a)[0],mode='project');plain=jax.jit(plain_scoped)
                    expected=plain(before[0],nt['raw']);jax.block_until_ready(expected)
                    pair=assess_native_pair(expected,native[0],np.ones(32,bool),env,limits=data['numeric_tolerances'])
                    env._step_impl=instrumented;plain.clear_cache();report['native_pair']=pair
                    if not pair['pass_']:raise ValueError('Native reward observation failed numerical pairing')
                    candidate=VelocityRewardContract(enabled=True,xy_weight=plan['candidate_weights']['xy'],z_weight=plan['candidate_weights']['z'])
                    ordinary=candidate_course_step(rt,candidate)
                    reference=jax.jit(lambda s,a:ordinary(data['robot'],data['auxiliary'],s,a,s.task.goal_xyz,
                        stage_id='S01',control_contract=data['control'],episode_contract=data['episode'],reward_contract=data['reward'],**data['options']))
                    expected_world,tr=reference(data['initial'],wt['raw']);jax.block_until_ready(expected_world)
                    delta=max(float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float))))
                        for a,b in zip(jax.tree.leaves(expected_world),jax.tree.leaves(world)) if np.size(a))
                    from .reward_direction import score_terms
                    _,score=score_terms(wt['terms'],plan['candidate_weights'])
                    report['world_observer_pair']=dict(max_state_abs=delta,max_reward_abs=float(np.max(np.abs(np.asarray(score-tr.reward)))),
                        terminated_equal=bool(np.array_equal(tr.terminated,wt['failure'])))
                    if delta>1e-5 or not report['world_observer_pair']['terminated_equal']:raise ValueError('World observer changed state or failure')
                    reference.clear_cache()
                streams['native'].append(jax.device_get(nt));streams['world'].append(jax.device_get(wt))
                if t in (63,127):
                    value=frozen_value(critic,course_critic_features(world))
                    boot[str(t+1)]=np.asarray(jnp.where(~(world.terminated|world.model_truncated),data['episode'].gamma**(t+1)*value,0.)).tolist()
                if t%32==31:guard.check()
            joined={kind:jax.tree.map(lambda *v:np.stack(v),*rows) for kind,rows in streams.items()}
            records={kind:summarize_stream(stream,plan['source_indices'],data['goals'],data['episode'].gamma,plan['candidate_weights']) for kind,stream in joined.items()}
            records['world_target_bootstrap']=boot
            records['temporary_actor_sha256']=numeric_tree_digest(actor)
            records['wall_s']=time.monotonic()-started
            records['active_raw_action_rms']={k:float(np.sqrt(np.mean(v['raw'][v['valid']]**2))) for k,v in joined.items()}
            mode_runs[branch]=records
            print(f'[REWARD_PAIR] {mode}/{branch} nativeH64={np.mean(records["native"]["64"]["candidate"]):.7g} worldH64={np.mean(records["world"]["64"]["candidate"]):.7g}',flush=True)
        outputs[mode]=mode_runs
    if numeric_tree_digest((data['actor'],data['robot'],data['auxiliary'],critic))!=frozen:raise ValueError('Frozen weights changed')
    native_step.clear_cache();world_step.clear_cache();jax.clear_caches()
    return outputs
