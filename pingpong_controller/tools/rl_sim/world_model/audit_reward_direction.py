"""Frozen candidate-gradient directions evaluated in native MJX and imagination."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate,config_hash,file_hash,load_config,repository_root
from .corrected_reward import objective_function
from .course_goals import build_goal_table
from .native_entry_collision import write_compact_json
from .paired_attribution import flat_tree,target_from_checkpoint
from .reward_direction import make_prepare,compare_returns
from .reward_direction_rollout import collect_branches
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget


def make_noise(keys):
    """Match the registered200Hz policy RNG splitting and FP32 normal sampling."""
    def tick(keys,_):
        split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
        value=jax.vmap(lambda k:jax.random.normal(k,(7,),dtype=jnp.float32))(split[:,1])
        return split[:,0],value
    return jax.lax.scan(tick,keys,None,length=128)[1]


def audit(cfg,args,guard,report,plan):
    started=time.monotonic();data=make_prepare()(cfg,args,guard,report,plan)
    data['numeric_tolerances']=cfg['frozen_evaluation_numerics']['absolute_tolerances']
    critic=target_from_checkpoint(plan['learner']);actor=data['actor']
    frozen=numeric_tree_digest((actor,critic,data['robot'],data['auxiliary']))
    goals=np.unique(np.asarray(data['goals']),axis=0)
    table=build_goal_table(goals,static_admitted=np.ones(len(goals),bool))
    keys=jax.random.split(jax.random.PRNGKey(plan['seed']+2),32)
    noise=np.asarray(jax.jit(make_noise)(keys));weights=jnp.array([1.,-1.,-1.,plan['candidate_weights']['xy'],plan['candidate_weights']['z']],jnp.float32)
    report.update(preparation_wall_s=time.monotonic()-started,parameters_before_sha256=frozen,
        common_initial_course_sha256=numeric_tree_digest(data['initial']),exploration_noise_sha256=numeric_tree_digest(noise),
        noise_shape=list(noise.shape),gradient_directions={})
    directions={}
    for mode,std in plan['policy_modes'].items():
        guard.check();start=time.monotonic()
        function=objective_function(data['runtime'],data['initial'],keys,table,data['episode'],data['reward'],std)
        def objective(params):
            values,trace=function(params,critic)
            return jnp.mean(values@weights),(values,{k:trace[k] for k in ('terms','valid','releases','terminated')})
        derivative=jax.jit(jax.value_and_grad(objective,has_aux=True))
        (value,(values,trace)),gradient=jax.device_get(derivative(actor));flat=flat_tree(gradient);norm=float(np.linalg.norm(flat))
        if not np.isfinite(flat).all() or norm<1e-10:raise ValueError('Nonfinite or zero reward direction')
        directions[mode]=jax.tree.map(lambda g:jnp.asarray(g/norm,jnp.float32),gradient)
        report['gradient_directions'][mode]=dict(norm=norm,parameter_count=len(flat),
            normalized_direction_sha256=numeric_tree_digest(directions[mode]),objective=float(value),
            per_source_objectives=values.tolist(),per_source_bootstrap=np.asarray(trace['terms']['target_bootstrap']).tolist(),
            direct_candidate_return=np.asarray(values@np.asarray(weights)-trace['terms']['target_bootstrap']).tolist(),
            valid_controls=int(trace['valid'].sum()),release_count=int(trace['releases'].sum()),
            physical_failures=int(trace['terminated'].sum()),wall_s=time.monotonic()-start)
        print(f'[REWARD_DIRECTION] {mode} norm={norm:.7g} objective={float(value):.7g}',flush=True)
        derivative.clear_cache();del derivative; jax.clear_caches();release_cpu_heap()
    branches=collect_branches(data,plan,guard,report,directions,noise,critic)
    report['comparisons']={}
    for mode,runs in branches.items():
        path=args.output_dir/(mode+'.json')
        digest=write_compact_json(path,dict(source_indices=plan['source_indices'],runs=runs),guard.budget)
        comparisons={}
        for horizon in ('64','128'):
            comparisons[horizon]={}
            for epsilon in plan['epsilons']:
                mapped={'base':'base','repeat':'repeat','plus':'plus_'+str(epsilon),'minus':'minus_'+str(epsilon)}
                engines={engine:{k:runs[name][engine][horizon] for k,name in mapped.items()} for engine in ('native','world')}
                comparisons[horizon][str(epsilon)]=compare_returns(engines['native'],engines['world'],plan['acceptance'])
        report['comparisons'][mode]=dict(details=dict(path=str(path),sha256=digest),horizons=comparisons)
        # The derivative's baseline and its separately instrumented forward path
        # can differ numerically. Report it instead of assuming bitwise replay.
        baseline=np.asarray(runs['base']['world']['64']['candidate'])
        expected=np.asarray(report['gradient_directions'][mode]['direct_candidate_return'])
        report['comparisons'][mode]['gradient_vs_probe_baseline_return_max_abs']=float(np.max(np.abs(baseline-expected)))
    after=numeric_tree_digest((actor,critic,data['robot'],data['auxiliary']))
    if after!=frozen:raise ValueError('Frozen parameter tree changed')
    for name,digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Preserved source changed:'+name)
    report.update(status='FROZEN_MJX_REWARD_DIRECTION_COMPLETE',parameters_after_sha256=after,parameters_unchanged=True,
        actor_updates=0,critic_updates=0,world_updates=0,optimizer_calls=0,default_replaced=False,native_controls=49152,
        world_forward_controls=49152,full_task_qualified=False,
        scope='Realized returns only in native/model comparisons;H64target value solely defines the frozen direction and is separately reported. No learner uses native data.')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args();cfg=load_config();setting=cfg['reward_direction_audit']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20
        or any(plan[k] for k in ('actor_updates','critic_updates','world_updates','default_replacement'))):
        raise ValueError('Only the authorized frozen pairing may run')
    authenticate(repository_root(),plan['prerequisite'])
    for key in ('physics','control'):
        if config_hash(cfg[key])!=plan[key+'_sha256']:raise ValueError('Preserved contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients'])!=plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_reward_direction_audit',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={f.name:file_hash(f) for f in sorted(Path(__file__).parent.glob('*.py'))})
        names=['reward_direction.py','reward_direction_rollout.py','audit_reward_direction.py']
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text() for n in names},
            protocol=authenticate(repository_root(),plan['protocol']).read_text()),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA GPU required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
