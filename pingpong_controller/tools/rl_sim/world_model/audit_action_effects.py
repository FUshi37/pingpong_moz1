"""Same-full-state MJX/model action pulses with frozen model and raw continuation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v92 import rotation_matrix_to_6d
from .action_effects import compare_effects, compare_with_repeat, model_event_trace, summarize_outcomes
from .audit_full_prior_world import release_cpu_heap
from .artifacts import read_array_archive, write_json_unique
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .collection_labels import label_task_bounds
from .contracts import config_hash, file_hash, load_config, repository_root
from .full_prior_seeds import initial_full_history
from .full_prior_world import FullPriorWorldState, full_prior_world_rollout
from .model_numerics import precision_scope
from .native_observation import NativeObservationAdapter, camera_from_native
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .physical_events import init_physical_events, scan_physical_events
from .replay_support import _restore
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader
from .teacher_trace import instrument_environment
from .trajectory_data import TrajectoryDataset
from .world_control import ControlContract, control_from_native
from .world_physics import PhysicsState


def clone_batch_lanes(state,mapping):
    """Copy all native model/state/RNG lanes; reject unregistered vector axes."""
    mapping=np.asarray(mapping)
    if mapping.ndim!=1 or mapping.dtype.kind not in 'iu' or np.any(mapping<0) or np.any(mapping>=len(mapping)):
        raise ValueError('Expected a complete valid lane mapping')
    def clone(value):
        if value.ndim==0:
            return value
        if value.shape[0]!=len(mapping):
            raise ValueError('Unknown unbatched native state axis: '+str(value.shape))
        return value[mapping]
    return jax.tree.map(clone,state)


def initial_world(full, env, options, lanes):
    """Select a complete physical reset; no approximation of old history/state."""
    native=full['native']
    support=np.array([env.mj_model.joint(f'LegWaist-{i}').id for i in range(6)])
    qadr=np.r_[np.asarray(env.arm_qadr),env.mj_model.jnt_qposadr[support]]
    vadr=np.r_[np.asarray(env.arm_vadr),env.mj_model.jnt_dofadr[support]]
    control,context=control_from_native(native)
    control,context=jax.tree.map(lambda x:x[lanes],(control,context))
    camera=jax.tree.map(lambda x:x[lanes],camera_from_native(native))
    q,dq=native.data.qpos[lanes][:,qadr],native.data.qvel[lanes][:,vadr]
    bq=native.data.qpos[lanes,env.ball_qadr:env.ball_qadr+7]
    bv=native.data.qvel[lanes,env.ball_vadr:env.ball_vadr+6]
    event=jax.tree.map(lambda x:x[lanes],full['events'])
    ball=ContactState(bq[:,:3],bv[:,:3],world_spin(bq[:,3:],bv[:,3:]),
                      event.contact_active,event.contact_duration_steps*.001)
    observation=initial_full_history(camera,q,chain=options['chain'])
    pose=jnp.concatenate((native.data.site_xpos[lanes,env.racket_site_id],
        rotation_matrix_to_6d(native.data.site_xmat[lanes,env.racket_site_id].reshape(-1,3,3))),axis=-1)
    return FullPriorWorldState(PhysicsState(q,dq,ball),control,observation,pose),context,event


def audit(cfg,args,guard,report):
    from mjx_topology_execution import make_collect
    root=repository_root()/cfg['resources']['output_root']
    reader=SupportReplayReader(root/'support_replay_full_a01',guard.budget,max_bytes=0)
    parameters,options,refs,auxiliary,aux_ref=load_components(cfg,reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    collection=Path(reader.report['source_collection'])
    source=json.loads((collection/'result.json').read_text())
    dataset=TrajectoryDataset.open(collection/'dataset',guard.budget)
    n=1024
    bundle=build_native_environment(cfg,n_envs=n); env=bundle.environment
    contract=ControlContract.from_native(env)
    fresh,_=jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']),n))
    jax.block_until_ready(fresh)
    context_hash=source['context']['sha256']
    model=_restore(source['context']['reference'],fresh.model,context_hash)
    history,_=NativeObservationAdapter(env,fresh).initialize(fresh)
    template=dict(native=fresh._replace(model=None),history=history,events=init_physical_events(n),
                  ball_model_valid=jnp.ones(n,bool),goal=jnp.zeros((n,3),jnp.float32))
    observer=instrument_environment(env,bundle.stack,support_trace=True)

    def native_transition(state,action,lanes):
        native,event,valid_before=state
        after,_,_,_,metrics=env.step(native,action)
        trace=metrics['wmj_trace']
        finite=jnp.ones((5,n),bool)
        for name in ('q_after','dq_after','support_q_after','support_dq_after','ball_qpos_after','ball_qvel_after'):
            finite &= jnp.all(jnp.isfinite(trace[name]),axis=-1)
        valid=valid_before[None] & (jnp.cumsum((trace['other_ball_collision']|~finite).astype(jnp.int32),axis=0)==0)
        event,labels=scan_physical_events(event,trace,valid=valid,initialization_active=jnp.zeros_like(valid))
        return (after,event,valid[-1]),(jax.tree.map(lambda x:x[:,lanes],labels),after.prev_action[lanes])

    scoped,execution=make_collect(native_transition,mode='project')
    native_step=jax.jit(scoped)

    @jax.jit
    def world_run(initial,context,events,valid_before,actions,goal):
        _,trace=full_prior_world_rollout(parameters['robot'],auxiliary,initial,actions,
            context=context,control_contract=contract,task_goal_xyz=goal,ar1_enabled=False,**options)
        physical=jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),trace['physics'])
        with precision_scope(options['matmul_precision']):
            event_trace=model_event_trace(physical,surface=options['surface'],
                ball_radius_m=options['ball_radius_m'],contact_contract=options['contact_contract'])
        finite=jnp.ones_like(event_trace['contact_overlap'])
        for name in ('ball_qpos_after','ball_qvel_after'):
            finite &= jnp.all(jnp.isfinite(event_trace[name]),axis=-1)
        finite &= jnp.all(jnp.isfinite(physical['q_after_rad']),axis=-1)
        valid=valid_before[None] & (jnp.cumsum((~finite).astype(jnp.int32),axis=0)==0)
        _,labels=scan_physical_events(events,event_trace,valid=valid,initialization_active=jnp.zeros_like(valid))
        return labels,trace['control']['accepted_action']

    names=list(dataset.chunks)
    rng=np.random.default_rng(2026110914)
    all_native={k:[] for k in ('base','plus','minus')};all_world={k:[] for k in all_native}
    records=[];all_repeat=[];all_cases=[]
    report.update(references=refs,auxiliary_reference=aux_ref,execution=execution,observer=observer,
        source_collection=str(collection),source_result_sha256=file_hash(collection/'result.json'),
        native_parallelism=n,learned_contact_residual=False,physical_modifications=False)
    for trial in range(args.trials):
        guard.check()
        first=dataset.chunks[names[2*trial]]
        eligible=[i for i,g in enumerate(first['groups']) if g['split']=='validation' and g['generator']!='robot_identification']
        if len(eligible)<48:
            raise ValueError('Preregistered48 non-robot-ID validation lanes unavailable')
        lanes=np.sort(rng.choice(eligible,48,replace=False))
        full=_restore(first['seed_ref'],template,context_hash)
        full['native']=full['native']._replace(model=model)
        initial,context,initial_events=initial_world(full,env,options,lanes)
        baseline=np.concatenate([read_array_archive(dataset.root/f'chunk_{key}.npz',dataset.chunks[key]['archive'],
            names=['action__raw_normalized'])['action__raw_normalized'] for key in names[2*trial:2*trial+2]])
        assignment=rng.permutation(48)
        joint=assignment%7
        epsilon=np.array([.005,.01,.02],np.float32)[(assignment//7)%3]
        duration=np.array([2,4])[(assignment//21)%2]
        pulse=np.zeros_like(baseline)
        for i,lane in enumerate(lanes):
            pulse[:duration[i],lane,joint[i]]=epsilon[i]
        native_outcomes={};world_outcomes={};native_actions={};world_actions={}
        native=full['native']
        bounds=label_task_bounds(cfg,dict(q_rad=np.asarray(native.data.qpos[:,env.arm_qadr])[None],
            ball_qpos=np.asarray(native.data.qpos[:,env.ball_qadr:env.ball_qadr+7])[None],
            racket_position_m=np.asarray(native.data.site_xpos[:,env.racket_site_id])[None],
            racket_rotation_world=np.asarray(native.data.site_xmat[:,env.racket_site_id]).reshape(1,n,3,3)),
            reference_rotation=np.asarray(env.fixed_contact_racket_reference_rotation),
            joint_lower_rad=np.asarray(env.arm_lo),joint_upper_rad=np.asarray(env.arm_hi))
        simultaneous_labels={};duplicate_check=None
        if args.paired_native_batch:
            # Baseline/+/-/duplicate share one solver invocation and stop clock.
            mapping=np.arange(n)
            mapping[:192]=np.tile(lanes,4)
            native_full=clone_batch_lanes((full['native'],full['events'],full['ball_model_valid']),mapping)
            branch_actions=baseline[:,mapping].copy()
            branch_actions[:,48:96]+=pulse[:,lanes]
            branch_actions[:,96:144]-=pulse[:,lanes]
            carry=native_full;labels=[];accepted=[];repeat_state_errors=[]
            # Include initial equality and each full-state endpoint, rather than
            # interpreting event-gated zero padding as an outcome distance.
            def repeat_error(native):
                fields=dict(qpos=native.data.qpos,qvel=native.data.qvel,
                            accepted_action=native.prev_action)
                return {key:float(np.max(np.abs(np.asarray(value[:48])-np.asarray(value[144:192]))))
                        for key,value in fields.items()}
            initial_repeat_error=repeat_error(carry[0])
            if any(value!=0. for value in initial_repeat_error.values()):
                raise ValueError('Cloned baseline states differ before stepping')
            for t in range(128):
                carry,(label,action_taken)=native_step(carry,jnp.asarray(branch_actions[t]),jnp.arange(192))
                label,action_taken=jax.device_get((label,action_taken))
                labels.append(label);accepted.append(action_taken)
                repeat_state_errors.append(repeat_error(carry[0]))
                if t%32==0:
                    guard.check()
                    print(f'[PAIRED_ACTION_EFFECT] trial={trial+1} tick={t}',flush=True)
            joined={k:np.concatenate([v[k] for v in labels]) for k in labels[0]}
            accepted=np.stack(accepted)
            for index,branch in enumerate(('base','plus','minus','repeat')):
                subset=slice(index*48,(index+1)*48)
                simultaneous_labels[branch]={k:v[:,subset] for k,v in joined.items()}
                native_actions[branch]=accepted[:,subset]
            duplicate_check={k:float(np.max(np.abs(np.asarray(simultaneous_labels['base'][k],np.float64)
                -np.asarray(simultaneous_labels['repeat'][k],np.float64)))) for k in joined}
            duplicate_check['accepted_action']=float(np.max(np.abs(native_actions['base']-native_actions['repeat'])))
            # Preserve the full diagnostics even if this negative control fails.
            write_json_unique(args.output_dir/f'duplicate_trial_{trial:02d}.json',
                dict(maximum_errors=duplicate_check,initial_state_errors=initial_repeat_error,
                    endpoint_state_errors=repeat_state_errors,
                    original_repeat_gate_passed=all(value<=1e-6 for value in duplicate_check.values())),guard.budget)
            # A failed repeat gate remains failed. Complete the diagnostic and
            # report event-aligned drift; do not silently qualify small effects.
            del carry,labels,joined,accepted,native_full,branch_actions
            release_cpu_heap()
        for branch,sign in (('base',0.),('plus',1.),('minus',-1.)):
            guard.check()
            actions=baseline+sign*pulse
            if args.paired_native_batch:
                native_labels=simultaneous_labels[branch]
            else:
                carry=(full['native'],full['events'],full['ball_model_valid'])
                labels=[];accepted=[]
                for t in range(128):
                    carry,(label,action_taken)=native_step(carry,jnp.asarray(actions[t]),jnp.asarray(lanes))
                    label,action_taken=jax.device_get((label,action_taken))
                    labels.append(label);accepted.append(action_taken)
                    if t%32==0:
                        guard.check()
                        print(f'[ACTION_EFFECT] trial={trial+1} branch={branch} tick={t}',flush=True)
                native_labels={k:np.concatenate([v[k] for v in labels]) for k in labels[0]}
                native_actions[branch]=np.stack(accepted)
                del carry,labels,accepted
            native_outcomes[branch]=summarize_outcomes(native_labels)
            predicted,world_actions[branch]=jax.device_get(world_run(initial,context,initial_events,full['ball_model_valid'][lanes],
                jnp.asarray(actions[:,lanes]),full['goal'][lanes]))
            world_outcomes[branch]=summarize_outcomes(predicted)
            all_native[branch].extend(native_outcomes[branch]);all_world[branch].extend(world_outcomes[branch])
            del native_labels, predicted
            release_cpu_heap()
        cases=[dict(lane=int(lane),root_seed_family=first['groups'][lane]['root_seed_family'],
            region_id=first['groups'][lane]['region_id'],generator=first['groups'][lane]['generator'],
            native_initial_task_eligible=bool(bounds['initial_task_eligible'][lane]),
            joint=int(joint[i]),epsilon=float(epsilon[i]),pulse_ms=int(duration[i]*5),
            native_accepted_central_pulse_norm=float(np.linalg.norm(native_actions['plus'][:,i]-native_actions['minus'][:,i])),
            world_accepted_central_pulse_norm=float(np.linalg.norm(world_actions['plus'][:,i]-world_actions['minus'][:,i])),
            effective_clipped_central_pulse_norm=float(np.linalg.norm(np.clip(baseline[:,lane]+pulse[:,lane],-1,1)
                -np.clip(baseline[:,lane]-pulse[:,lane],-1,1)))) for i,lane in enumerate(lanes)]
        result=dict(trial=trial,cases=cases,native=native_outcomes,world=world_outcomes,
            comparison=compare_effects(native_outcomes,world_outcomes),seed_reference=first['seed_ref'],
            native_comparison_mode='same_batch_four_way' if args.paired_native_batch else 'separate_batches',
            duplicate_baseline=duplicate_check,
            excluded_robot_identification=sum(g['split']=='validation' and g['generator']=='robot_identification' for g in first['groups']),
            eligible_nonrobot_validation=len(eligible))
        if args.paired_native_batch:
            repeat_outcomes=summarize_outcomes(simultaneous_labels['repeat'])
            result.update(native_repeat=repeat_outcomes,
                repeat_comparison=compare_with_repeat(native_outcomes,world_outcomes,repeat_outcomes,cases))
            all_repeat.extend(repeat_outcomes);all_cases.extend(cases)
        digest=write_json_unique(args.output_dir/f'trial_{trial:02d}.json',result,guard.budget)
        records.append(dict(trial=trial,sha256=digest,comparison=result['comparison']))
        report.update(trials=records,comparison=compare_effects(all_native,all_world))
        if args.paired_native_batch:
            report['repeat_comparison']=compare_with_repeat(all_native,all_world,all_repeat,all_cases)
        write_json_unique(args.output_dir/f'progress_trial_{trial:02d}.json',report,guard.budget)
    report['status']='ACTION_EFFECT_DIAGNOSTIC_COMPLETE'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--trials',type=int,default=1,choices=(1,10))
    parser.add_argument('--paired-native-batch',action='store_true')
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_full_prior_action_effect_audit_v1',training_version='WMJ-V1',status='INCOMPLETE',
            config_sha256=config_hash(cfg),source_sha256={p.name:file_hash(p) for p in paths},gpu_uuid=args.gpu_uuid,
            actor_updates=0,critic_updates=0,world_qualified=False,
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            source={p.name:p.read_text() for p in paths}),guard.budget)
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


if __name__=='__main__':
    main()
