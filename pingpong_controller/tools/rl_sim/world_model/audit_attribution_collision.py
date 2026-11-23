"""Original evaluator graph, host-only collision inspection; no new trace outputs."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import mujoco

from . import evaluate_frozen
from .artifacts import write_json_unique
from .attribution_common import population,actors,plan_and_budget,worker_check,source_report,apex_replay_deltas,initial_interface
from .contracts import authenticate,file_hash,load_config,repository_root
from .imagined_env import EpisodeContract
from .native_evaluation import build_evaluation_step
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def original_policy_step(p,actor,cfg):
    """Compile the verbatim original nested function with the same closure values."""
    module=ast.parse(textwrap.dedent(inspect.getsource(evaluate_frozen.audit)))
    node=next(n for n in ast.walk(module) if isinstance(n,ast.FunctionDef) and n.name=='policy_step')
    source=ast.unparse(node)
    factory='def factory(env,adapter,control,episode,task_step,goals,actor,n):\n'+textwrap.indent(source,'    ')+'\n    return policy_step\n'
    namespace=dict(evaluate_frozen.__dict__)
    exec(compile(factory,'<original-frozen-policy-step>','exec'),namespace)
    task_step,hook=build_evaluation_step(p['env'],p['bundle'].stack,perturb=False)
    fn=namespace['factory'](p['env'],p['adapter'],p['control'],EpisodeContract.from_config(cfg),task_step,p['goals'],actor,p['goals'].shape[0])
    return fn,dict(nested_source_sha256=hashlib.sha256(source.encode()).hexdigest(),hook=hook,
        additional_jit_inputs=False,additional_jit_outputs=False,collision_inspection='host_after_original_step_only')


def audit(cfg,args,guard,report):
    from mjx_topology_execution import make_collect
    plan=plan_and_budget(cfg,guard);p=population(cfg);actor=actors(cfg)['u20'];frozen=numeric_tree_digest(actor)
    report['initial_numeric_sha256']=numeric_tree_digest(p['initial'])
    report['initial_interface']=initial_interface(p,np.asarray(plan['selected_lane_ids']))
    fn,identity=original_policy_step(p,actor,cfg);scoped,execution=make_collect(fn,mode='project');step=jax.jit(scoped)
    report.update(original_function=identity,execution=execution,runs={},actor_reference=cfg['bounded_imagination_pilot']['actor_exports']['20'])
    env=p['env'];model=env.mj_model
    for mode,std in [('deterministic',0.),('sampled',cfg['imagination']['exploration_std'])]:
        keys=jax.random.split(jax.random.PRNGKey(plan['seed']+2),128)
        state=(p['initial'],p['history'],p['task'],p['events'],keys);records=[];collisions=[]
        for tick in range(12000):
            if np.asarray(state[2].done).all():break
            worker_check(guard)
            out=step(*state,jnp.asarray(std,jnp.float32));state=out[:5];trace=jax.device_get(out[-1])
            for sub,lane in zip(*np.nonzero(trace['valid_release']|trace['valid_apex']|(trace['failure_code']!=0))):
                r={k:evaluate_frozen.json_numeric(v[sub,lane]) for k,v in trace.items()}
                r.update(case_id=p['cases'][lane]['case_id'],control_tick=tick,substep=int(sub));records.append(r)
            failed=np.flatnonzero(np.any(trace['failure_code']==2,axis=0))
            if len(failed):
                geoms=np.asarray(out[0].data.contact.geom);dist=np.asarray(out[0].data.contact.dist)
                for lane in failed:
                    for pair,distance in zip(geoms[lane],dist[lane]):
                        if env.ball_geom_id not in pair or env.racket_contact_geom_id in pair or distance>0:continue
                        names=[dict(id=int(i),geom=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(i)),
                            body=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_BODY,int(model.geom_bodyid[i]))) for i in pair]
                        collisions.append(dict(case_id=p['cases'][lane]['case_id'],control_tick=tick,
                            elapsed_ms=int(out[2].elapsed_ms[lane]),geoms=names,distance_m=float(distance),
                            ball_position_m=np.asarray(out[0].data.qpos[lane,env.ball_qadr:env.ball_qadr+3]).tolist()))
        old=json.loads(authenticate(repository_root(),plan['evidence']['u20'][mode]).read_text());task=jax.device_get(state[2])
        outcomes=[dict(case_id=c['case_id'],hits=int(task.hits[i]),failure=evaluate_frozen.FAILURES[int(task.failure_code[i])],elapsed_ms=int(task.elapsed_ms[i])) for i,c in enumerate(p['cases'])]
        changed=[dict(current=r,previous=o) for r,o in zip(outcomes,old['cases'])
            if r['hits']!=o['hits'] or r['failure']!=o['physical_failure'] or abs(r['elapsed_ms']*.001-o['elapsed_s'])>1e-6]
        deltas=apex_replay_deltas(records,old['events'])
        report['runs'][mode]=dict(outcomes=outcomes,collisions=collisions,replay_outcome_differences=changed,
            matched_apex_count=len(deltas),matched_apex_max_abs_delta_m=max(deltas,default=0.),
            nominal_23_events=[r for r in records if r['case_id']=='nominal_23'],original=plan['evidence']['u20'][mode])
        print(f'[ORIGINAL] {mode} changed_outcomes={len(changed)} apex_delta={max(deltas,default=0.):.6g} collisions={collisions}',flush=True)
    report.update(status='ORIGINAL_GRAPH_COLLISION_ATTRIBUTION_COMPLETE',parameters_unchanged=numeric_tree_digest(actor)==frozen)
    if not report['parameters_unchanged']:raise ValueError('Frozen actor changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        plan_and_budget(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False);report=source_report(cfg,args)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,source=Path(__file__).read_text()),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('CUDA required')
            audit(cfg,args,guard,report)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
