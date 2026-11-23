"""Replay baseline/u10/u20 without updates; identify native contact and rewards."""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from .artifacts import write_json_unique, write_array_archive_unique
from .attribution_common import population, actors, plan_and_budget, worker_check, source_report, apex_replay_deltas
from .contracts import authenticate, load_config, repository_root, file_hash
from .evaluate_frozen import endpoint_code, json_numeric, FAILURES
from .frozen_task import fail_evaluation
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract, release_reward, control_reward, approach_reward
from .migration import actor_mean
from .native_evaluation import build_evaluation_step
from .native_pair import assess_native_pair
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .teacher_trace import contact_pair_masks


def contact_observer(env, before, after, labels):
    """Report actual already-computed ball contacts; no new geometry query."""
    _, other = contact_pair_masks(after.contact.geom, after.contact.dist,
        ball_geom_id=env.ball_geom_id, racket_geom_id=env.racket_contact_geom_id)
    depth = jnp.where(other, after.contact.dist, jnp.inf)
    index = jnp.argmin(depth, axis=-1)
    geom = jnp.take_along_axis(after.contact.geom, index[:,None,None], axis=1)[:,0]
    present = jnp.any(other, axis=-1)
    return dict(release_position_m=labels['release_position_m'],
        other_geom_pair=jnp.where(present[:,None],geom,-1),
        other_contact_dist_m=jnp.where(present,jnp.min(depth,axis=-1),0.))


def audit(cfg, args, guard, report):
    from mjx_topology_execution import make_collect
    plan = plan_and_budget(cfg, guard)
    p = population(cfg); env=p['env']; adapter=p['adapter']; goals=p['goals']
    lanes=np.asarray(plan['selected_lane_ids']); n=128
    parameters=actors(cfg); digest=numeric_tree_digest(parameters)
    reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    episode=EpisodeContract.from_config(cfg)
    native_step, hook=build_evaluation_step(env,p['bundle'].stack,perturb=False,trace_observer=contact_observer)

    def tick(native, history, task, events, keys, actor, std):
        observation=adapter.observation(native,history,task.task.goal_xyz)
        split=jax.vmap(lambda key:jax.random.split(key,2))(keys)
        noise=jax.vmap(lambda key:jax.random.normal(key,(7,)))(split[:,1])
        action=actor_mean(actor,observation)+std*noise
        result=native_step(native,action,wmj_payload=(task,events),wmj_goals=goals)
        after=result[0]; new_task,new_events=result[-1]['wmj_payload'];trace=result[-1]['wmj_trace']
        new_history,_=adapter.advance(after,history)
        code=endpoint_code(env,after,adapter,p['control'],episode)
        failed=(code!=0)&~task.done
        new_task=new_task._replace(done=new_task.done&~failed,success=new_task.success&~failed)
        new_task=fail_evaluation(new_task,jnp.where(failed,code,0))
        terms=release_reward(trace['release_position_m'],trace['release_velocity_m_s'],
            task.task.goal_xyz[None],trace['valid_release'],contract=reward)
        terms={k:jnp.sum(v,axis=0) for k,v in terms.items() if k!='total'}
        dense=control_reward(after.prev_action,native.prev_action,jnp.clip(action,-1.,1.),
            active=~task.done,physical_failure=(new_task.failure_code!=0)&~task.done,dt_s=.005,contract=reward)
        terms.update({k:v for k,v in dense.items() if k!='total'})
        terms['approach']=approach_reward(native.data.qpos[:,env.ball_qadr:env.ball_qadr+3],
            native.data.qvel[:,env.ball_vadr:env.ball_vadr+3],native.data.site_xpos[:,env.racket_site_id],
            active=~task.done,dt_s=.005,contract=reward)
        compact=dict(raw_action=action[lanes],active=(~task.done)[lanes],
            ball_position_m=after.data.qpos[lanes,env.ball_qadr:env.ball_qadr+3],
            ball_velocity_m_s=after.data.qvel[lanes,env.ball_vadr:env.ball_vadr+3],
            racket_position_m=after.data.site_xpos[lanes,env.racket_site_id],
            accepted_action=after.prev_action[lanes])
        return after,new_history,new_task,new_events,split[:,0],trace,terms,compact,action

    scoped,execution=make_collect(tick,mode='project');step=jax.jit(scoped)
    plain,_=make_collect(lambda s,a:env.step(s,a)[0],mode='project');plain=jax.jit(plain)
    keys=jax.random.split(jax.random.PRNGKey(plan['seed']+2),n)
    check=step(p['initial'],p['history'],p['task'],p['events'],keys,parameters['baseline'],jnp.float32(0))
    paired=plain(p['initial'],check[-1]);jax.block_until_ready(paired)
    pair=assess_native_pair(paired,check[0],p['known'],env,
        limits=cfg['frozen_evaluation_numerics']['absolute_tolerances'])
    if not pair['pass_']:raise ValueError('Observer changed physical zero-force results')
    geom_names={str(i):dict(geom=mujoco.mj_id2name(env.mj_model,mujoco.mjtObj.mjOBJ_GEOM,i),
        body=mujoco.mj_id2name(env.mj_model,mujoco.mjtObj.mjOBJ_BODY,int(env.mj_model.geom_bodyid[i])))
        for i in range(env.mj_model.ngeom)}
    report.update(plan=cfg['frozen_attribution_diagnostic']['plan'],native_pair=pair,hook=hook,execution=execution,
        selected_case_ids=plan['selected_case_ids'],initial_numeric_sha256=numeric_tree_digest(p['initial']),
        xml=dict(path=str(p['bundle'].stack.xml),sha256=file_hash(p['bundle'].stack.xml)),
        ball_geom_id=env.ball_geom_id,racket_geom_id=env.racket_contact_geom_id,geom_names=geom_names,runs={})
    for name,actor in parameters.items():
        for mode,std in [('deterministic',0.),('sampled',cfg['imagination']['exploration_std'])]:
            state=(p['initial'],p['history'],p['task'],p['events'],keys)
            streams=[];records=[];reward_sums=None;discounted=None
            for index in range(12000):
                if np.asarray(state[2].done).all():break
                worker_check(guard)
                result=step(*state,actor,jnp.float32(std));state=result[:5]
                trace,terms,compact=jax.device_get(result[5:8]);streams.append(compact)
                if reward_sums is None:
                    reward_sums={k:np.zeros(n) for k in terms};discounted={k:np.zeros(n) for k in terms}
                for k,v in terms.items():reward_sums[k]+=v;discounted[k]+=episode.gamma**index*v
                selected=trace['valid_release']|trace['valid_apex']|(trace['failure_code']!=0)
                for sub,lane in zip(*np.nonzero(selected)):
                    r={k:json_numeric(v[sub,lane]) for k,v in trace.items()}
                    r.update(case_id=p['cases'][lane]['case_id'],control_tick=index,substep=int(sub))
                    records.append(r)
            old=json.loads(authenticate(repository_root(),plan['evidence'][name][mode]).read_text())
            task=jax.device_get(state[2])
            outcomes=[dict(case_id=c['case_id'],hits=int(task.hits[i]),elapsed_ms=int(task.elapsed_ms[i]),
                failure=FAILURES[int(task.failure_code[i])]) for i,c in enumerate(p['cases'])]
            changed=[dict(current=r,previous=o) for r,o in zip(outcomes,old['cases'])
                if (r['hits']!=o['hits'] or r['failure']!=o['physical_failure']
                    or abs(r['elapsed_ms']*.001-o['elapsed_s'])>.006)]
            apex_delta=apex_replay_deltas(records,old['events'])
            arrays={k:np.stack([x[k] for x in streams]) for k in streams[0]}
            archive=write_array_archive_unique(args.output_dir/f'{name}_{mode}.npz',arrays,guard.budget)
            details=dict(actor=name,mode=mode,
                events=[r for r in records if r['case_id'] in plan['selected_case_ids'] or r['failure_code']==2],
                outcomes=outcomes,
                reward_sums={k:v.tolist() for k,v in reward_sums.items()},
                discounted_reward_sums={k:v.tolist() for k,v in discounted.items()},
                replay_outcome_differences=changed,matched_apex_max_abs_delta_m=float(max(apex_delta,default=0.)),
                archive=archive,archive_file=f'{name}_{mode}.npz',original=plan['evidence'][name][mode])
            sha=write_json_unique(args.output_dir/f'{name}_{mode}.json',details,guard.budget)
            report['runs'][f'{name}_{mode}']=dict(path=str(args.output_dir/f'{name}_{mode}.json'),sha256=sha)
            print(f'[ATTRIBUTION] {name}/{mode} ticks={len(streams)} outcome_differences={len(changed)} apex_max_delta={max(apex_delta,default=0.):.6g}',flush=True)
    report.update(status='NATIVE_FROZEN_ATTRIBUTION_COMPLETE',parameters_unchanged=numeric_tree_digest(parameters)==digest)
    if not report['parameters_unchanged']:raise ValueError('Frozen actors changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        plan_and_budget(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=source_report(cfg,args)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,config=cfg,
            sources={n:(Path(__file__).parent/n).read_text() for n in
                ['attribution_common.py','audit_attribution_native.py','native_evaluation.py']}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('CUDA required')
            audit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
