"""Guarded native fixed-case evaluation; no optimizer or learned-model updates."""

import argparse
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .frozen_cases import validate_case_manifest, GROUPS
from .frozen_initialization import StandardDropAuditor
from .frozen_task import init_evaluation, fail_evaluation
from .frozen_seed import save_warmed_seed, load_warmed_seed
from .imagined_env import EpisodeContract, physical_bounds
from .migration import expand_actor_history, load_v321_actor, actor_mean
from .native_evaluation import build_evaluation_step
from .native_observation import NativeObservationAdapter
from .native_pair import assess_native_pair
from .native_stack import build_native_environment
from .physical_events import init_physical_events
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .teacher_initialization import prepare_warmup, begin_ball_trial
from .world_control import ControlContract


FAILURES={0:'none',1:'nonfinite',2:'other_ball_collision',3:'ball_domain',
          4:'racket_workspace',5:'racket_orientation',6:'joint_position'}


def json_numeric(value):
    """Retain missing/nonfinite physical labels as null, never invented zeros."""
    array=np.asarray(value)
    if array.dtype.kind in 'fc' and not np.isfinite(array).all():
        return np.where(np.isfinite(array),array,None).tolist()
    return array.tolist()


def endpoint_code(env,native,adapter,control,episode):
    _,pose=adapter._values(native)
    physics=SimpleNamespace(q_rad=native.data.qpos[:,env.arm_qadr],
        ball=SimpleNamespace(position_m=native.data.qpos[:,env.ball_qadr:env.ball_qadr+3]))
    _,tests=physical_bounds(physics,pose,control,episode)
    code=jnp.zeros(native.step_count.shape,jnp.int32)
    for name,value in reversed(tuple(zip(('ball','racket','orientation','joints'),(3,4,5,6)))):
        code=jnp.where(~tests[name],value,code)
    return code


def select_evaluation_actor(cfg, *, actor_override=None, actor_reference=None):
    """Preserve the baseline default; candidate callers must supply authenticated identity."""
    if actor_override is None:
        if actor_reference is not None:
            raise ValueError('Actor reference provided without parameters')
        reference = cfg['lineage']['actor_source']
        return expand_actor_history(load_v321_actor(repository_root(), reference)), reference
    from .migration import _validate_actor
    if not actor_reference or not {'path', 'sha256'}.issubset(actor_reference):
        raise ValueError('An explicit candidate requires its checkpoint identity')
    _validate_actor(actor_override, 969)
    return actor_override, actor_reference


def audit(cfg,args,guard,report,*,actor_override=None,actor_reference=None):
    from mjx_topology_execution import make_collect
    pinned=authenticate(repository_root(),cfg['frozen_evaluation_artifacts']['static_initialization'])
    if pinned!=(args.initialization_dir/'initialization.json').resolve():
        raise ValueError('Only the registered fixed-case initialization artifact may be evaluated')
    static=json.loads((args.initialization_dir/'initialization.json').read_text())
    if static['cases_reference']['sha256']!=cfg['frozen_evaluation_artifacts']['cases']['sha256']:
        raise ValueError('Fixed evaluation cases changed')
    manifest=json.loads(authenticate(repository_root(),static['cases_reference']).read_text())
    validate_case_manifest(manifest,cfg)
    if static['atlas_reference']!=cfg['collection']['initialization_atlas']:
        raise ValueError('Unregistered initialization atlas')
    stage=f'S{args.stage:02d}'
    ids=list(range(128)) if not args.smoke else [0,32,64,96]
    cases=[manifest['stages'][stage][i] for i in ids]
    rows=[static['rows'][i] for i in ids]
    if any(c['case_id']!=r['case_id'] for c,r in zip(cases,rows)):
        raise ValueError('Admission rows and fixed cases differ')
    goals=jnp.asarray([r['goals_xyz_m'] for r in cases],jnp.float32)
    n=len(cases)
    known=np.array([r['status']=='STATIC_INITIALIZATION_ADMITTED' for r in rows])
    bundle=build_native_environment(cfg,n_envs=n,stage_index=args.stage-1)
    env=bundle.environment
    control=ControlContract.from_native(env)
    episode=EpisodeContract.from_config(cfg)
    reset_keys=jax.random.split(jax.random.PRNGKey(args.seed),n)
    fresh,_=jax.jit(env.reset)(reset_keys)
    jax.block_until_ready(fresh)
    plain_scoped,execution=make_collect(lambda s,a:env.step(s,a)[0],mode='project')
    plain=jax.jit(plain_scoped)
    report.update(execution=execution,stage=stage,cases_sha256=manifest['cases_sha256'],
        initialization_sha256=file_hash(args.initialization_dir/'initialization.json'),
        actor_reference=cfg['lineage']['actor_source'],native_overrides=bundle.overrides,
        evaluation_scope='4_case_smoke' if args.smoke else 'fixed128',warmup_controls=128,
        unknown_not_replaced=True,geometry_execution='offline_initialization_only')
    context=dict(schema='wmj_fixed_warmed_population',cases_sha256=manifest['cases_sha256'],
        physics_sha256=config_hash(cfg['physics']),control_sha256=config_hash(cfg['control']),
        domain_sha256=config_hash(cfg['domain']),ids=ids,reset_seed=args.seed,warmup_controls=128,
        native_source_sha256=bundle.stack.spec.native_env_sha256,
        initialization_sha256=report['initialization_sha256'],warmup_ar1_enabled=False)
    cache=None if args.smoke else cfg['frozen_evaluation_artifacts'].get('warmed_seed')
    if cache:
        warmed,warm_rows=load_warmed_seed(cache,fresh,context)
        if [r['case_id'] for r in warm_rows]!=[r['case_id'] for r in cases]:
            raise ValueError('Frozen warmed admission IDs changed')
        known=np.array([r['status']=='ADMITTED' for r in warm_rows])
        report.update(warmed_seed=cache,warmup_replayed=False,warmup_wall_s=0.)
        print('[RESTORE] exact warmed native population and admission; no repeat warmup/audit',flush=True)
    else:
        if args.stage==5: raise ValueError('S05 requires a pinned zero-noise warmed population')
        auditor=StandardDropAuditor(cfg)
        q=jnp.asarray([r['q_rad'] if k else auditor.catalog.q_rad[0].tolist() for r,k in zip(rows,known)])
        warmed=jax.jit(lambda s,q:prepare_warmup(env,s,q))(fresh,q)
        print(f'[WARMUP] {stage} {n} lanes,128 zero-action native controls',flush=True)
        start=time.monotonic()
        for _ in range(128):
            guard.check()
            warmed=plain(warmed,jnp.zeros((n,7),jnp.float32))
        jax.block_until_ready(warmed)
        report['warmup_wall_s']=time.monotonic()-start
        warm_rows=[]
        actual_qpos=np.asarray(warmed.data.qpos);actual_qvel=np.asarray(warmed.data.qvel)
        for i,(case,row) in enumerate(zip(cases,rows)):
            guard.check()
            result=(auditor.audit_warmed(actual_qpos[i],actual_qvel[i],np.asarray(goals[i,0]),
                control.projection.reference_rotation) if known[i] else dict(status='UNKNOWN',reason='static_initialization_unknown'))
            known[i]&=result['status']=='ADMITTED'
            warm_rows.append(dict(case_id=case['case_id'],group=case['group'],**result))
        report['warmed_seed']=save_warmed_seed(args.output_dir/'warmed_seed.json',
            warmed,warm_rows,context,guard.budget)
        restored,restored_rows=load_warmed_seed(report['warmed_seed'],fresh,context)
        report['warmed_seed_roundtrip_exact']=(numeric_tree_digest(restored)==numeric_tree_digest(warmed)
            and restored_rows==warm_rows)
        if not report['warmed_seed_roundtrip_exact']: raise ValueError('Warmed native seed did not roundtrip')
    write_json_unique(args.output_dir/'warmup_admission.json',dict(rows=warm_rows,
        admitted=int(known.sum()),unknown=int((~known).sum()),policy_outputs_used=False),guard.budget)
    print(f'[ADMISSION] {int(known.sum())}/{n} after native warmup; no substitutions',flush=True)
    initial=jax.jit(lambda f,w,p:begin_ball_trial(env,f,w,p,jnp.zeros((n,3)),
        ar1_enabled=args.stage==5))(fresh,warmed,goals[:,0])
    report['camera_ar1_enabled']=bool(env.cfg.ball_position_noise_ar1_enabled)
    report['initial_camera_epsilon_m']=np.asarray(initial.ball_position_noise_epsilon_m).tolist()
    mass=np.asarray(initial.model.body_mass)
    ball_mass=mass[:,env.ball_body_id] if mass.ndim==2 else mass[env.ball_body_id]
    report['actual_ball_mass_kg_range']=[float(np.min(ball_mass)),float(np.max(ball_mass))]
    report['native_xml']=dict(path=str(bundle.stack.xml),sha256=file_hash(Path(bundle.stack.xml)))
    adapter=NativeObservationAdapter(env,initial,full_chain=True)
    history,_=adapter.initialize(initial)
    initial_task=init_evaluation(goals,jax.random.PRNGKey(args.seed+1),jnp.asarray(ids))
    initial_task=initial_task._replace(done=~jnp.asarray(known))
    initial_events=init_physical_events(n)
    task_step,identity=build_evaluation_step(env,bundle.stack,perturb=args.stage>=4)
    report['native_task_hook']=identity
    actor,report['actor_reference']=select_evaluation_actor(cfg,
        actor_override=actor_override,actor_reference=actor_reference)
    actor_digest=numeric_tree_digest(actor)

    def policy_step(native,history,task,events,keys,std):
        observation=adapter.observation(native,history,task.task.goal_xyz)
        split=jax.vmap(lambda key:jax.random.split(key,2))(keys)
        noise=jax.vmap(lambda key:jax.random.normal(key,(7,)))(split[:,1])
        action=actor_mean(actor,observation)+std*noise
        result=task_step(native,action,wmj_payload=(task,events),wmj_goals=goals)
        following=result[0];next_task,next_events=result[-1]['wmj_payload']
        following_history,_=adapter.advance(following,history)
        code=endpoint_code(env,following,adapter,control,episode)
        failed=(code!=0)&~task.done
        # A final goal does not hide a physical violation at this same200Hz
        # endpoint. Later padding cannot fail an already completed case.
        next_task=next_task._replace(done=next_task.done&~failed,success=next_task.success&~failed)
        next_task=fail_evaluation(next_task,jnp.where(failed,code,0))
        trace=dict(result[-1]['wmj_trace'],raw_action=jnp.broadcast_to(action,(5,n,7)))
        return following,following_history,next_task,next_events,split[:,0],trace

    scoped,_=make_collect(policy_step,mode='project')
    step=jax.jit(scoped)
    keys=jax.random.split(jax.random.PRNGKey(args.seed+2),n)
    # Compare the same control action, camera/controller context and data seed.
    began=time.monotonic()
    actual=step(initial,history,initial_task,initial_events,keys,jnp.asarray(0.,jnp.float32))
    jax.block_until_ready(actual)
    report['cold_task_step_s']=time.monotonic()-began
    # Reuse the exact compiled actor output, rather than recomputing an eager
    # matmul with different fusion/precision and calling it the same action.
    expected=plain(initial,actual[-1]['raw_action'][0])
    jax.block_until_ready(expected)
    pair=assess_native_pair(expected,actual[0],known,env,
        limits=cfg['frozen_evaluation_numerics']['absolute_tolerances'])
    report['zero_force_pair']=pair
    if not pair['pass_']:
        raise ValueError('Native evaluation hook changed zero-force physical outputs')
    print(f'[PAIR] zero-force native qpos/qvel within registered smoke tolerances: {pair}',flush=True)
    if args.stage>=4:
        delta_v=jnp.asarray([.03,.04,.1],jnp.float32)
        forced_task=initial_task._replace(schedule=initial_task.schedule._replace(
            remaining_steps=jnp.full((n,),20,jnp.int32),pulse_used=jnp.ones(n,bool),
            force_world_n=jnp.broadcast_to(delta_v*(.004/.020),(n,3))))
        left=(initial,history,initial_task,initial_events,keys)
        right=(initial,history,forced_task,initial_events,keys)
        applied=np.zeros(n,int);contacts=np.zeros(n,bool)
        for _ in range(5):
            before=step(*left,jnp.asarray(0.,jnp.float32))
            after=step(*right,jnp.asarray(0.,jnp.float32))
            left=before[:5];right=after[:5]
            applied+=np.asarray(after[-1]['pulse_applied']).sum(axis=0)
            contacts|=np.asarray(after[-1]['contact_entry']).any(axis=0)|np.asarray(before[-1]['contact_entry']).any(axis=0)
        measured=np.asarray(right[0].data.qvel[:,env.ball_vadr:env.ball_vadr+3]
                            -left[0].data.qvel[:,env.ball_vadr:env.ball_vadr+3])
        usable=known&~np.asarray(left[2].done)&~np.asarray(right[2].done)&~contacts
        error=float(np.max(np.abs(measured[usable]-np.asarray(delta_v)))) if usable.any() else None
        passed=bool(usable.any() and np.all(applied[usable]==20) and error<1e-4)
        report['force_delivery_probe']=dict(pass_=passed,known_lanes=int(known.sum()),
            free_flight_complete_lanes=int(usable.sum()),
            excluded_from_free_flight_equation=[dict(case_id=cases[i]['case_id'],
                unknown=bool(not known[i]),baseline_ended=bool(left[2].done[i]),
                forced_ended=bool(right[2].done[i]),contact=bool(contacts[i])) for i in np.flatnonzero(~usable)],
            applied_ticks=applied.tolist(),requested_delta_v_m_s=np.asarray(delta_v).tolist(),
            measured_delta_v_m_s=json_numeric(measured),max_abs_error_m_s=error,
            diagnostic_seed_override=True,counts_as_task_evidence=False)
        if not passed: raise ValueError('Native20ms external-force delivery probe failed')
        print(f'[FORCE]20 native1ms ticks,delta-v error={error:.3g}m/s',flush=True)
    all_modes={}
    for mode,std in (('deterministic',0.),('sampled',cfg['imagination']['exploration_std'])):
        native,paired,task,events=initial,history,initial_task,initial_events
        keys=jax.random.split(jax.random.PRNGKey(args.seed+2),n)
        event_rows=[];pulse_ticks=np.zeros(n,int);elapsed=time.monotonic()
        noise_count=0;noise_sum=np.zeros(3);noise_square_sum=np.zeros(3);hold_error=0.;invalid_noise_rows=0
        for tick in range(12000):
            if np.asarray(task.done).all(): break
            guard.check()
            active_before=~np.asarray(task.done)
            epsilon_before=np.asarray(native.ball_position_noise_epsilon_m)
            native,paired,task,events,keys,trace=step(native,paired,task,events,keys,jnp.asarray(std,jnp.float32))
            trace=jax.device_get(trace)
            observed=np.asarray(native.fixed_contact_ball_observed)&active_before
            epsilon=np.asarray(native.ball_position_noise_epsilon_m)
            finite_noise=np.all(np.isfinite(epsilon),axis=-1)
            invalid_noise_rows+=int((active_before&~finite_noise).sum())
            observed&=finite_noise
            noise_count+=int(observed.sum())
            noise_sum+=epsilon[observed].sum(axis=0)
            noise_square_sum+=(epsilon[observed]**2).sum(axis=0)
            held=active_before&~observed&finite_noise&np.all(np.isfinite(epsilon_before),axis=-1)
            if held.any(): hold_error=max(hold_error,float(np.max(np.abs(epsilon[held]-epsilon_before[held]))))
            pulse_ticks+=np.asarray(trace['pulse_applied']).sum(axis=0)
            selected=(trace['valid_release']|trace['valid_apex']|trace['pulse_start']|(trace['failure_code']!=0))
            for sub,lane in zip(*np.nonzero(selected)):
                record={k:json_numeric(v[sub,lane]) for k,v in trace.items()}
                record['nonfinite_fields']=[k for k,v in trace.items() if not np.isfinite(v[sub,lane]).all()]
                record.update(case_id=cases[lane]['case_id'],control_tick=tick,substep=int(sub))
                event_rows.append(record)
            if tick%200==199:
                print(f'[EVAL] {stage}/{mode} {(tick+1)*.005:.1f}s alive={int((~np.asarray(task.done)).sum())}',flush=True)
        state=jax.device_get(task)
        outcomes=[]
        for i,case in enumerate(cases):
            outcomes.append(dict(case_id=case['case_id'],group=case['group'],
                admission='ADMITTED' if known[i] else 'UNKNOWN',success=bool(state.success[i]) if known[i] else None,
                done=bool(state.done[i]),physical_failure=FAILURES[int(state.failure_code[i])],
                timed_out=bool(state.truncated[i]),elapsed_s=float(state.elapsed_ms[i])*.001,
                hits=int(state.hits[i]),apexes=int(state.apexes[i]),completed_goals=int(state.completed_goals[i]),
                successful_goals=int(state.successful_goals[i]),pulse_ticks=int(pulse_ticks[i])))
        groups={}
        for group in GROUPS:
            group_rows=[r for r in outcomes if r['group']==group]
            accepted=sum(r['admission']=='ADMITTED' for r in group_rows)
            successes=sum(r['success'] is True for r in group_rows)
            groups[group]=dict(planned=len(group_rows),admitted=accepted,unknown=len(group_rows)-accepted,
                success=successes,success_fraction_of_planned=successes/len(group_rows),
                pass_=len(group_rows)==32 and accepted==32 and successes/32>=.90)
        result=dict(stage=stage,mode=mode,exploration_std=std,groups=groups,cases=outcomes,
            events=event_rows,wall_s=time.monotonic()-elapsed,policy_updates=0,
            hit1=sum(r['hits']>=1 for r in outcomes),hit3=sum(r['hits']>=3 for r in outcomes),
            hit12=sum(r['hits']>=12 for r in outcomes),mean_hits_known=float(state.hits[known].mean()) if known.any() else None,
            failure_distribution={name:sum(r['admission']=='ADMITTED' and r['physical_failure']==name
                for r in outcomes) for name in FAILURES.values()},
            kl=None,explained_variance=None,optimizer_metrics_scope='not_applicable_frozen_evaluation',
            camera_noise=dict(ar1_enabled=report['camera_ar1_enabled'],fresh_samples=noise_count,
                epsilon_mean_m=(noise_sum/max(noise_count,1)).tolist(),
                epsilon_std_m=np.sqrt(np.maximum(noise_square_sum/max(noise_count,1)
                    -(noise_sum/max(noise_count,1))**2,0.)).tolist(),held_epsilon_max_change_m=hold_error,
                nonfinite_rows=invalid_noise_rows),
            complete=all(r['done'] for r in outcomes),pass_=all(g['pass_'] for g in groups.values()))
        digest=write_json_unique(args.output_dir/(mode+'.json'),result,guard.budget)
        all_modes[mode]=dict(sha256=digest,groups=groups,wall_s=result['wall_s'],complete=result['complete'])
        print(f'[RESULT] {stage}/{mode} hits={int(state.hits.sum())} success={int(state.success.sum())}/{n}',flush=True)
    if actor_digest!=numeric_tree_digest(actor):
        raise ValueError('Frozen actor changed during evaluation')
    report.update(status='COMPLETE',modes=all_modes,actor_sha256=actor_digest,actor_unchanged=True,
        task_gate_pass=all(g['pass_'] for m in all_modes.values() for g in m['groups'].values()),
        baseline_only=actor_override is None,policy_training_authorized=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--initialization-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--stage',type=int,choices=range(1,6),default=1)
    parser.add_argument('--seed',type=int,default=2026110925)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        sources=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_native_frozen_evaluation',status='INCOMPLETE',
            config_sha256=config_hash(cfg),training_version='WMJ-V1',gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            source_sha256={p.name:file_hash(p) for p in sources},actor_updates=0,critic_updates=0,world_updates=0)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            source={p.name:p.read_text() for p in sources}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc)
            raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__': main()
