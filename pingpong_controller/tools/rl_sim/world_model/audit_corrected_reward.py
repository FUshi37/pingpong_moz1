"""Frozen u12288 inverse-speed reward scale and H64 derivative recheck."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from .audit_full_prior_world import release_cpu_heap
from .audit_module_gradients import load_runtime
from .calibrate_velocity_reward import coefficient_grid, OBJECTIVES
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .corrected_reward import (BASE, REPLACEMENT, directional_summary, native_reward_calibration,
                               objective_function)
from .course_goals import build_goal_table
from .expanded_auxiliary import restore_candidate
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .paired_attribution import flat_tree, target_from_checkpoint
from .paired_precision import load_actor
from .projection_precision import make_runtime
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget
from .velocity_reward import clone_globals


def run_mode(runtime, initial, actor, critic, keys, table, episode, reward, std, plan, guard):
    started = time.monotonic()
    fn = objective_function(runtime, initial, keys, table, episode, reward, std)
    forward = jax.jit(lambda params:fn(params, critic))
    center, trace = jax.device_get(forward(actor))
    repeat, repeat_trace = jax.device_get(forward(actor))
    def scalar(params, weights):
        values, diagnostic = fn(params, critic)
        return jnp.sum(values*weights), (values, diagnostic['modes'])
    derivative = jax.jit(jax.value_and_grad(scalar, has_aux=True))
    gradients = []; primal = []; changed = []
    n = len(center); lane117 = plan['source_indices'].index(117)
    for i in range(6):
        guard.check()
        weights = np.broadcast_to(np.eye(5, dtype=np.float32)[i]/n, (n, 5)).copy() if i < 5 else np.zeros((n,5),np.float32)
        if i == 5:weights[lane117] = REPLACEMENT
        (_, (value, modes)), gradient = jax.device_get(derivative(actor, jnp.asarray(weights)))
        flat = flat_tree(gradient)
        if not np.isfinite(flat).all() or not np.isfinite(value).all():
            raise FloatingPointError('Nonfinite frozen derivative')
        gradients.append(flat);primal.append(float(np.max(np.abs(value-center))))
        changed.append(np.flatnonzero(np.any(modes!=trace['modes'],axis=(0,2))).tolist())
        print(f'[REWARD_GRAD] std={std} component={i} norm={np.linalg.norm(flat):.7g}',flush=True)
    matrix = np.stack(gradients[:5]);gram = matrix@matrix.T
    grid = coefficient_grid(gram, center.mean(axis=0), plan['weights'], plan['criteria'])
    result = dict(mean_objectives=center.mean(axis=0).tolist(), per_source_objectives=center.tolist(),
        discounted_terms={k:np.asarray(v).tolist() for k,v in trace['terms'].items()},
        gradient_gram=gram.tolist(), gradient_norms=np.linalg.norm(matrix,axis=-1).tolist(),
        parameter_count=matrix.shape[1], grid=grid, forward_vs_ad_primal_max=primal,
        forward_vs_ad_mode_changed_indices=changed, forward_repeat_max=float(np.max(np.abs(center-repeat))),
        repeat_mode_changed_indices=np.flatnonzero(np.any(trace['modes']!=repeat_trace['modes'],axis=(0,2))).tolist(),
        valid_controls=int(trace['valid'].sum()), per_source_valid_controls=trace['valid'].sum(axis=0).tolist(),
        releases=trace['releases'].tolist(),terminated=trace['terminated'].tolist(),
        model_truncated=trace['model_truncated'].tolist(), probes={}, source117_lane=lane117,
        source117_unit_replacement_gradient_norm=float(np.linalg.norm(gradients[5])))
    vectors = [gradients[0], REPLACEMENT@matrix, gradients[5]]
    directions = np.stack([v/np.linalg.norm(v) for v in vectors])
    # A JVP can be nonfinite even while the VJP and actual forward values are
    # finite. Preserve both results; independent per-source reverse contractions
    # must not be replaced with a zero or with an invented finite JVP.
    reverse_tangents = np.empty((3,n,5),np.float64)
    for lane in range(n):
        guard.check()
        for component in range(5):
            weights = np.zeros((n,5),np.float32);weights[lane,component]=1.
            (_, _), gradient = jax.device_get(derivative(actor,jnp.asarray(weights)))
            flat = flat_tree(gradient)
            if not np.isfinite(flat).all():raise FloatingPointError('Nonfinite per-source VJP')
            reverse_tangents[:,lane,component] = directions@flat
        if lane%8==7:print(f'[REWARD_VJP] std={std} sources={lane+1}/{n}',flush=True)
    derivative.clear_cache();del derivative;release_cpu_heap()
    flat_actor, unravel = ravel_pytree(actor)
    automatic = jax.jit(lambda params, direction:jax.jvp(lambda x:fn(x, critic), (params,), (direction,), has_aux=True))
    for direction_index,(name, vector) in enumerate(zip(plan['fd']['directions'], vectors)):
        guard.check();norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm <= 1e-12:raise ValueError('Uninformative derivative direction:'+name)
        direction = unravel(jnp.asarray(vector/norm, jnp.float32))
        primal_value, jvp_tangent, ad_trace = jax.device_get(automatic(actor, direction))
        tangent = reverse_tangents[direction_index]
        probe = dict(direction_norm_before_normalization=float(norm),
            derivative_per_source_objectives=tangent.tolist(),
            derivative_method='per-source reverse-mode contraction; raw JVP reported independently',
            jvp_nonfinite_coordinates=np.argwhere(~np.isfinite(jvp_tangent)).tolist(),
            jvp_derivative_per_source_objectives=[[float(x) if np.isfinite(x) else None for x in row] for row in jvp_tangent],
            forward_vs_jvp_primal_max=float(np.max(np.abs(primal_value-center))),
            forward_vs_jvp_mode_changed_indices=np.flatnonzero(np.any(ad_trace['modes']!=trace['modes'],axis=(0,2))).tolist(),
            aggregate_vs_per_source_reverse_abs=np.abs(matrix@(vector/norm)-tangent.mean(axis=0)).tolist(),
            reverse_vs_finite_jvp_abs=distribution(np.abs(tangent-jvp_tangent)),rows=[])
        for epsilon in plan['fd']['epsilons']:
            guard.check()
            plus_actor = unravel(flat_actor+jnp.asarray(epsilon*vector/norm,jnp.float32))
            minus_actor = unravel(flat_actor-jnp.asarray(epsilon*vector/norm,jnp.float32))
            plus, pt = jax.device_get(forward(plus_actor));minus, mt = jax.device_get(forward(minus_actor))
            if not np.isfinite(plus).all() or not np.isfinite(minus).all():raise FloatingPointError('Nonfinite symmetric probe')
            summaries = {label:directional_summary(center,repeat,tangent,plus,minus,trace['modes'],
                pt['modes'],mt['modes'],epsilon,weight,plan['fd'])
                for label,weight in [('baseline',BASE),('unit_replacement',REPLACEMENT)]}
            raw_delta = np.asarray(pt['raw']-mt['raw'])
            accepted_delta = np.asarray(pt['accepted']-mt['accepted'])
            row = dict(epsilon=epsilon, plus_objectives=plus.tolist(),minus_objectives=minus.tolist(),
                objective_derivative=( (plus-minus)/(2*epsilon) ).tolist(),summaries=summaries,
                raw_action_central_norm_per_source=np.linalg.norm(raw_delta,axis=(0,2)).tolist(),
                accepted_action_central_norm_per_source=np.linalg.norm(accepted_delta,axis=(0,2)).tolist(),
                note='Action differences include diagnostic padding; masks/mode changes are separately retained')
            probe['rows'].append(row)
            print(f'[REWARD_FD] std={std} {name} eps={epsilon} resolved='+str({k:v['resolved_count'] for k,v in summaries.items()}),flush=True)
        result['probes'][name] = probe
    result['wall_s'] = time.monotonic()-started
    forward.clear_cache();automatic.clear_cache();jax.clear_caches();release_cpu_heap()
    return result


def audit(cfg,args,guard,report,plan):
    loader = clone_globals(load_runtime,load_pilot_actor=load_actor,restore_auxiliary=restore_candidate)
    initial,actor,robot,auxiliary,options,control,refs = loader(cfg,guard,plan)
    critic = target_from_checkpoint(plan['learner'])
    frozen = numeric_tree_digest((actor,critic,robot,auxiliary))
    runtime = make_runtime(robot,auxiliary,options,control)
    reward = RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    episode = EpisodeContract.from_config(cfg)
    goals = np.unique(np.asarray(initial.task.goal_xyz),axis=0)
    table = build_goal_table(goals,static_admitted=np.ones(len(goals),bool))
    keys = jax.random.split(jax.random.PRNGKey(plan['seed']+2),len(plan['source_indices']))
    report.update(model_references=refs,parameters_before_sha256=frozen,
        initial_state_sha256=numeric_tree_digest(initial),exploration_rng_sha256=numeric_tree_digest(keys),
        objective_names=OBJECTIVES,local_native=native_reward_calibration(plan,reward),runs={})
    print('[REWARD_SETUP] frozen u12288 and corrected projection restored; 32 original sources',flush=True)
    for label,std in [('deterministic',0.),('sampled',.03)]:
        report['runs'][label] = run_mode(runtime,initial,actor,critic,keys,table,episode,reward,std,plan,guard)
        write_compact_json(args.output_dir/(label+'.json'),report['runs'][label],guard.budget)
    grids = [row['grid'] for row in report['runs'].values()]
    eligible = [(row['xy_weight'],row['z_weight']) for i,row in enumerate(grids[0]) if all(g[i]['eligible'] for g in grids)]
    after = numeric_tree_digest((actor,critic,robot,auxiliary))
    if frozen!=after:raise ValueError('Frozen parameters changed')
    for name,digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Prior source changed:'+name)
    report.update(status='FROZEN_CORRECTED_REWARD_AUDIT_COMPLETE',parameters_after_sha256=after,
        parameters_unchanged=True,scale_gradient_eligible_coefficients=eligible,selected_coefficients=None,
        actor_updates=0,critic_updates=0,world_updates=0,optimizer_calls=0,default_replaced=False,
        limitations=['Coarse mode fingerprints do not certify every internal solver/contact branch.',
          'Finite-difference criteria are separate from reward-scale/gradient calibration.',
          'Model reference errors and source117 reverse action response remain unqualified.',
          'No coefficient is adopted and no training gain is established by this frozen diagnostic.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config();setting=cfg['corrected_reward_audit']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20
        or any(plan[k] for k in ('actor_updates','critic_updates','world_updates','default_replacement'))):
        raise ValueError('Only the authorized frozen reward audit may run')
    for key in ('physics','control'):
        if config_hash(cfg[key])!=plan[key+'_sha256']:raise ValueError('Preserved contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients'])!=plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    authenticate(repository_root(),plan['prerequisite'])
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_corrected_reward_audit',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text()
            for n in ('corrected_reward.py','audit_corrected_reward.py')},
            protocol=authenticate(repository_root(),plan['protocol']).read_text()),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            # Per-mode files retain all per-source differences; avoid duplicating them.
            report['runs']={k:dict(path=str(args.output_dir/(k+'.json')),
                sha256=file_hash(args.output_dir/(k+'.json')),
                **{x:v[x] for x in ('mean_objectives','gradient_norms','grid','valid_controls','wall_s')})
                for k,v in report.get('runs',{}).items() if (args.output_dir/(k+'.json')).exists()}
            report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
