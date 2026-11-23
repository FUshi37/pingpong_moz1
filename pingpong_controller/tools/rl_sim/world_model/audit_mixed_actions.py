"""All480 existing saved MJX forks against the frozen mixed numerical candidate."""
import json
import time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .action_effects import compare_effects, compare_with_repeat, model_event_trace, summarize_outcomes
from .artifacts import read_array_archive, write_array_archive_unique
from .audit_full_prior_world import AuditReader, release_cpu_heap
from .audit_mixed_precision import check
from .auxiliary_refresh import restore_auxiliary
from .contracts import authenticate, file_hash, repository_root
from .mixed_action_seed import read_action_seed
from .mixed_precision import make_runtime
from .model_numerics import precision_scope
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .physical_events import scan_physical_events
from .pilot_contract import action_gate
from .state_adapter import numeric_tree_digest
from .world_control import ControlContract


def pack_rows(rows: list, prefix: str) -> tuple:
    """Preserve every outcome/missing flag compactly, with no NaN sentinels."""
    arrays={};schema={}
    for key in rows[0]:
        example=next((r[key] for r in rows if r.get(key) is not None),0.)
        value=np.asarray([r[key] if r.get(key) is not None else np.zeros_like(example) for r in rows])
        arrays[prefix+'__'+key]=value
        arrays[prefix+'__'+key+'__present']=np.asarray([r.get(key) is not None for r in rows])
        schema[key]=dict(shape=list(value.shape[1:]),dtype=value.dtype.str)
    return arrays,schema


def pack_numeric_slabs(arrays: dict) -> tuple:
    """Losslessly group small arrays, retaining every shape, dtype and offset."""
    groups={}
    layout={}
    slabs={}
    for key,value in arrays.items():
        array=np.asarray(value)
        if array.dtype.kind not in 'biufc':raise ValueError('Numeric evidence only')
        groups.setdefault(array.dtype.str,[]).append((key,array))
    for index,(dtype,rows) in enumerate(sorted(groups.items())):
        name=f'slab_{index}';offset=0;parts=[]
        for key,value in rows:
            layout[key]=dict(slab=name,offset=offset,count=value.size,
                             shape=list(value.shape),dtype=dtype)
            parts.append(value.reshape(-1));offset+=value.size
        slabs[name]=np.concatenate(parts)
    return slabs,layout


def audit_actions(cfg, args, guard, report, plan):
    if args.mode!='mixed':raise ValueError('Saved action-fork worker is candidate-only')
    root=repository_root()/cfg['resources']['output_root']
    previous=json.loads(authenticate(repository_root(),plan['native_action_reference']).read_text())
    if previous['status']!='CANDIDATE_ACTION_RECHECK_COMPLETE':raise ValueError('Saved candidate comparison incomplete')
    evidence=json.loads(authenticate(repository_root(),cfg['bounded_imagination_pilot']['action_reference']).read_text())
    old_dir=root/'full_prior_action_effect_a04'
    for path,digest in evidence['evidence'].items():
        if path.startswith(old_dir.name+'/') and file_hash(root/path)!=digest:
            raise ValueError('Native action evidence changed')
    reader=AuditReader(root/'support_replay_full_a01',guard.budget,max_bytes=0)
    params,options,refs,_,_=load_components(cfg,reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    restored,_=restore_auxiliary(cfg,plan['auxiliary']);aux=dict(parameters=restored['parameters'],scale=restored['scale'])
    if previous['candidate_checkpoint']!=plan['auxiliary']:raise ValueError('Auxiliary candidate identity changed')
    env=build_native_environment(cfg,n_envs=1).environment
    control=ControlContract.from_native(env);frozen=numeric_tree_digest((params['robot'],aux))
    native={k:[] for k in ('base','plus','minus')};world={k:[] for k in native};repeats=[];cases=[];trials=[]
    names=list(reader.dataset.chunks)
    with jax.enable_x64(True):
        runtime=make_runtime('mixed',params['robot'],aux,options,control)
        def run(initial,context,events,valid_before,actions,goal):
            def tick(state,action):
                return runtime.world_step(runtime.robot,runtime.auxiliary,state,action,
                    context=context,control_contract=runtime.control,task_goal_xyz=goal,
                    ar1_enabled=False,**runtime.options)
            _,trace=jax.lax.scan(tick,initial,actions)
            physical=jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),trace['physics'])
            with precision_scope(options['matmul_precision']):
                et=model_event_trace(physical,surface=runtime.options['surface'],
                    ball_radius_m=options['ball_radius_m'],contact_contract=runtime.options['contact_contract'])
            finite=jnp.all(jnp.isfinite(physical['q_after_rad']),axis=-1)
            for name in ('ball_qpos_after','ball_qvel_after'):finite &= jnp.all(jnp.isfinite(et[name]),axis=-1)
            valid=valid_before[None] & (jnp.cumsum((~finite).astype(jnp.int32),axis=0)==0)
            _,labels=scan_physical_events(events,et,valid=valid,initialization_active=jnp.zeros_like(valid))
            return labels,trace['control']['accepted_action'],jnp.all(finite)
        compiled=jax.jit(run)
        for trial in range(10):
            check(guard);saved_path=old_dir/f'trial_{trial:02d}.json';saved=json.loads(saved_path.read_text())
            first=reader.dataset.chunks[names[2*trial]]
            if saved['seed_reference']!=first['seed_ref']:raise ValueError('Saved native initial source changed')
            lanes=np.array([r['lane'] for r in saved['cases']])
            for row in saved['cases']:
                group=first['groups'][row['lane']]
                if group['split']!='validation' or group['root_seed_family']!=row['root_seed_family']:
                    raise ValueError('Action comparison partition changed')
            # Restore under the original precision before promoting candidate state.
            with jax.enable_x64(False):
                initial,context,events,valid,goal=read_action_seed(saved['seed_reference'],env,options,
                    reader.report['context_sha256'],lanes)
            initial_hash=numeric_tree_digest((initial,context,events,valid,goal))
            initial=runtime.prepare_world(initial);context=runtime.prepare_context(context);events=runtime.prepare_events(events)
            chunks=[read_array_archive(reader.dataset.root/f'chunk_{name}.npz',
                reader.dataset.chunks[name]['archive'],names=['action__raw_normalized'])['action__raw_normalized'][:,lanes]
                for name in names[2*trial:2*trial+2]]
            base=np.concatenate(chunks);pulse=np.zeros_like(base)
            for i,row in enumerate(saved['cases']):
                if row['pulse_ms'] not in (10,20) or not any(np.isclose(row['epsilon'],e) for e in (.005,.01,.02)):
                    raise ValueError('Unregistered saved action pulse')
                pulse[:row['pulse_ms']//5,i,row['joint']]=row['epsilon']
            actions=dict(base=base,plus=base+pulse,minus=base-pulse)
            outcomes={};accepted={};times={}
            for branch in actions:
                check(guard);started=time.monotonic()
                labels,taken,finite=jax.device_get(compiled(initial,context,events,valid,jnp.asarray(actions[branch]),goal))
                times[branch]=time.monotonic()-started
                if not finite:raise FloatingPointError('Nonfinite mixed action fork')
                outcomes[branch]=summarize_outcomes(labels);accepted[branch]=taken
                native[branch].extend(saved['native'][branch]);world[branch].extend(outcomes[branch])
            rows=[dict(row,world_accepted_central_pulse_norm=float(np.linalg.norm(accepted['plus'][:,i]-accepted['minus'][:,i])))
                  for i,row in enumerate(saved['cases'])]
            cases.extend(rows);repeats.extend(saved['native_repeat'])
            trials.append(dict(trial=trial,native_reference=dict(path=str(saved_path),sha256=file_hash(saved_path)),
                initial_sha256=initial_hash,cases=48,times_s=times,comparison=compare_effects(saved['native'],outcomes)))
            print(f'[MIXED_ACTION] {trial+1}/10 all48 finite',flush=True);release_cpu_heap()
        compiled.clear_cache()
    comparison=compare_with_repeat(native,world,repeats,cases)
    gate=action_gate(previous['repeat_comparison'],comparison)
    arrays={};schemas={}
    for branch,rows in world.items():
        data,schema=pack_rows(rows,branch);arrays.update(data);schemas[branch]=schema
    for name in ('outgoing_velocity_m_s','actual_apex_m'):
        rows=comparison[name].pop('cases')
        # Some missing cases omit numerical fields; preserve availability separately.
        keys=set().union(*(r.keys() for r in rows))
        rows=[{k:r.get(k) for k in sorted(keys)} for r in rows]
        data,schema=pack_rows(rows,'repeat_'+name);arrays.update(data);schemas['repeat_'+name]=schema
    data,schema=pack_rows(cases,'case');arrays.update({k:v for k,v in data.items() if v.dtype.kind in 'biufc'})
    schemas['case_numeric']={k:v for k,v in schema.items() if np.dtype(v['dtype']).kind in 'biufc'}
    arrays,layout=pack_numeric_slabs(arrays)
    report.update(status='MIXED_ACTION_COMPUTE_COMPLETE_PENDING_ARCHIVE',trials=trials,schemas=schemas,
        archive_layout=layout,
        comparison=compare_effects(native,world),repeat_comparison=comparison,action_gate=gate,
        previous_reference=plan['native_action_reference'],native_steps_executed=0,
        model_references=refs,auxiliary=plan['auxiliary'],parameters_before_sha256=frozen,
        parameters_after_sha256=numeric_tree_digest((params['robot'],aux)),task_qualified=False)
    if report['parameters_after_sha256']!=frozen:raise ValueError('Frozen action model changed')
    report['outcome_archive']=write_array_archive_unique(args.output_dir/'action_outcomes.npz',arrays,guard.budget)
    report['parameters_unchanged']=True
    report['status']='FROZEN_MIXED_ACTION_EFFECTS_COMPLETE'
