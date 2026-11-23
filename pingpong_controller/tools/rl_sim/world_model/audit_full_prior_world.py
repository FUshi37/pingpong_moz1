"""Guarded held-out integration audit: own controls versus recorded commands."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from fixed_plane_v92 import rotation_matrix_to_6d
from .artifacts import discard_file_cache, write_json_unique
from .contracts import config_hash, file_hash, load_config, repository_root
from .control_seed import read_control_seed
from .control_projection import project_reference
from .fit_robot import summarize_errors
from .full_prior_seeds import (initial_full_history, read_initial_camera,
    reconstruct_teacher_bank, select_teacher_seeds)
from .full_prior_world import FullPriorWorldState, full_prior_world_rollout, measured_racket_pose
from .model_numerics import precision_scope
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .physics_data import read_physics_trial
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader
from .tangent_coupling import make_predictor
from .world_control import ControlContract
from .world_physics import PhysicsState


STATE_NAMES = ('q_ref_rad', 'accepted_action', 'racket_position_m',
    'racket_rotation_world', 'goal_xyz_m', 'observed_ball_position_m',
    'ball_source_time_us', 'ball_observed', 'time_us', 'task_within_task_bounds')


class AuditReader(SupportReplayReader):
    """Release per-file OS cache after bounded reads; preserve authenticated bytes."""

    def read(self, chunk_id, *, names=None):
        arrays = super().read(chunk_id,names=names)
        record = self.records[chunk_id]
        for path in (Path(record['base_archive_path']),self.root/record['archive_file']):
            discard_file_cache(path)
        return arrays


def read_selected_trial(reader, names, lanes):
    def read_selected(name, *, names):
        arrays=reader.read(name,names=names)
        # Authenticate complete archives, then select before joining trials.
        return {key:(value[:,:,lanes].copy() if key.startswith('substep__')
                     else value[:,lanes].copy()) for key,value in arrays.items()}
    selected_reader=SimpleNamespace(records=reader.records,read=read_selected)
    raw, record = read_physics_trial(selected_reader, *names)
    chunks = [read_selected(name, names={'state__'+s for s in STATE_NAMES}
        | {'action__raw_normalized', 'action__accepted_normalized'}) for name in names]
    values = {key:np.concatenate((chunks[0][key][:-1], chunks[1][key]))
              for key in chunks[0] if key.startswith('state__')}
    for key in ('action__raw_normalized', 'action__accepted_normalized'):
        values[key] = np.concatenate([c[key] for c in chunks])
    if not np.array_equal(values['state__time_us'],
                          np.broadcast_to(np.arange(129)[:,None]*5000, (129,len(lanes)))):
        raise ValueError('Full history reconstruction needs a continuous reset-to-640 ms trial')
    return raw, values, record


def make_bank_builder(options, control_contract):
    @jax.jit
    def build(initial, endpoints, cached, reference, accepted, context):
        return reconstruct_teacher_bank(initial, endpoints, cached, reference, accepted,
            context=context, control_contract=control_contract, chain=options['chain'], ar1_enabled=False)
    return build


def cached_pose_sequence(position, rotation):
    """Preserve arbitrary leading time/lane axes around the native 6D helper."""
    rotation = jnp.asarray(rotation)
    encoded = rotation_matrix_to_6d(rotation.reshape(-1,3,3)).reshape(rotation.shape[:-2]+(6,))
    return jnp.concatenate((jnp.asarray(position),encoded),axis=-1)


def teacher_seeds(reader, record, lanes, raw, values, options, builder):
    reference = record['first_seed_ref']
    control, context = read_control_seed(reference,
        context_sha256=reader.report['context_sha256'], lanes=lanes)
    camera = read_initial_camera(reference, context_sha256=reader.report['context_sha256'], lanes=lanes)
    batch = prepare_position_contact_batch(None, jax.tree.map(jnp.asarray, raw))
    initial_physics = PhysicsState(jnp.asarray(raw['q_before'][0]), jnp.asarray(raw['dq_before'][0]),
                                  jax.tree.map(lambda x:x[0], batch['before']))
    endpoints = PhysicsState(jnp.asarray(raw['q_after'][4::5]), jnp.asarray(raw['dq_after'][4::5]),
                             jax.tree.map(lambda x:x[4::5], batch['after']))
    pose = cached_pose_sequence(values['state__racket_position_m'],values['state__racket_rotation_world'])
    observation = initial_full_history(camera, initial_physics.q_rad, chain=options['chain'])
    initial = FullPriorWorldState(initial_physics, control, observation, pose[0])
    bank, commands = builder(initial, endpoints, pose[1:],
        jnp.asarray(values['state__q_ref_rad'][1:]),
        jnp.asarray(values['action__accepted_normalized']), context)
    camera_bank = jax.device_get(bank.observation.camera)
    differences = dict(camera_position_m=np.abs(camera_bank.held_position_m
        -values['state__observed_ball_position_m']).ravel())
    if (not np.array_equal(camera_bank.last_sample_step*5000, values['state__ball_source_time_us'])
            or not np.array_equal(camera_bank.observed, values['state__ball_observed'])):
        raise ValueError('Reconstructed camera source clock/visibility differs from native replay')
    for key in ('q', 'dq', 'ddq'):
        prediction = np.asarray(commands[key]).reshape(raw['delayed_'+key+'_command'].shape)
        differences['delayed_'+key] = np.abs(prediction-raw['delayed_'+key+'_command']).ravel()
    tolerances = dict(camera_position_m=1e-6, delayed_q=1e-5, delayed_dq=1e-4, delayed_ddq=.002)
    if any(not np.isfinite(v).all() or v.max() > tolerances[k] for k,v in differences.items()):
        raise ValueError('Reconstructed full seed is not coherent: '
                         +str({k:float(v.max()) for k,v in differences.items()}))
    # Validate timestamp-paired endpoint history against independent time lookup.
    history = jax.tree.map(lambda x:np.asarray(x[-1]), bank.observation.history.paired)
    full_q = np.concatenate((raw['q_before'][:1], raw['q_after'][4::5]))
    times = np.maximum(history.sample_time_us//5000, 0)
    source_q = full_q[times, np.arange(len(lanes))[:, None]]
    expected_pose = np.asarray(measured_racket_pose(options['chain'], source_q.reshape(-1,13))).reshape(len(lanes),64,9)
    pair_error = np.abs(history.racket_pose-expected_pose)[history.present & history.aligned]
    differences['paired_pose_source_lookup'] = pair_error.ravel()
    if pair_error.size and pair_error.max() > 2e-6:
        raise ValueError('Full-chain racket history lost source-time alignment')
    # Independent native endpoint site labels establish the inherited prefix
    # timing. Contact substep poses still use their 1 ms before-state.
    endpoint_pose = np.asarray(measured_racket_pose(options['chain'],
        full_q.reshape(-1,13))).reshape(pose.shape)
    prefix_error = np.abs(endpoint_pose-np.asarray(pose))
    differences['native_endpoint_prefix_pose'] = prefix_error.ravel()
    # Native GPU forward and the private full-FP32 FK use different arithmetic.
    # 0.3 mm / 3e-4 rotation components are a small fraction of the task budget;
    # do not demand artificial micrometer equality across those paths.
    if not np.isfinite(prefix_error).all() or prefix_error.max() > 3e-4:
        raise ValueError('Native refreshed site disagrees with endpoint full-chain FK')
    return bank, context, {k:summarize_errors([v]) for k,v in differences.items()}


def projection_diagnostic(context, actions, measured, references, *, chain, contract):
    def tick(unused, values):
        action, q, reference = values
        scale = contract.projection.velocity_limit*contract.action_velocity_scale*context.action_scale_mult[:,None]
        nominal = jnp.clip(jnp.clip(action,-1.,1.)*scale,
                           -contract.projection.velocity_limit,contract.projection.velocity_limit)
        result = project_reference(chain,contract.projection,q,reference,nominal)
        return unused,result
    return jax.lax.scan(tick,None,(actions,measured,references))[1]


def audit(cfg, args, guard, report):
    root = repository_root()/cfg['resources']['output_root']
    reader = AuditReader(root/'support_replay_full_a01', guard.budget, max_bytes=0)
    parameters, options, refs, auxiliary, aux_ref = load_components(cfg, reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    bundle = build_native_environment(cfg, n_envs=1)
    contract = ControlContract.from_native(bundle.environment)
    builder = make_bank_builder(options, contract)
    component = make_predictor(parameters, options, auxiliary, tangent_enabled=True, learned=False)
    project = jax.jit(lambda context,a,q,r:projection_diagnostic(context,a,q,r,
        chain=options['chain'],contract=contract))

    @jax.jit
    def evaluate(seed, context, actions, goal, raw):
        after, trace = full_prior_world_rollout(parameters['robot'], auxiliary, seed, actions,
            context=context, control_contract=contract, task_goal_xyz=goal,
            ar1_enabled=False, **options)
        physical = jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),trace['physics'])
        batch = prepare_position_contact_batch(None,raw)
        metrics, events = ball_metrics(physical['contact'],batch)
        metrics.update(robot_metrics(physical,raw,batch))
        finite = jnp.all(jnp.isfinite(trace['actor_observation']),axis=-1)
        for value in physical['contact']['after'][:3]:
            finite = finite & jnp.all(jnp.isfinite(value.reshape((actions.shape[0],5,-1,3))),axis=(1,3))
        # Common-command model uses exactly the same branch initial physical state.
        with precision_scope(options['matmul_precision']):
            common = component(raw)['predicted_robot']
        control_errors = {key:jnp.max(jnp.abs(trace['control']['delayed_commands'][key]
            -raw['delayed_'+key+'_command'].reshape(trace['control']['delayed_commands'][key].shape)),axis=(1,3))
            for key in ('q','dq','ddq')}
        return dict(own_control=dict(metrics=metrics,events=events,nonfinite_windows=jnp.sum(~jnp.all(finite,axis=0))),
            common_command=common, control_errors=control_errors,
            projection=trace['control']['projection'],
            joint_max_error_rad=jnp.max(jnp.abs(physical['q_after_rad']-raw['q_after']),axis=-1).reshape(actions.shape[0],5,-1).max(axis=1),
            racket_position_error_m=jnp.linalg.norm(physical['racket_before']['position']
                -raw['racket_site_position_before_m'],axis=-1).reshape(actions.shape[0],5,-1).max(axis=1),
            history_valid_count=jnp.sum(after.observation.history.paired.present,axis=-1))

    rng = np.random.default_rng(2026110913)
    names = list(reader.records)
    pieces, region_pieces, events, nonfinite = {}, {}, {}, {}
    reports, windows, control_diagnostics = [], [], []
    for trial in range(min(args.trials,len(names)//2)):
        guard.check()
        keys = names[2*trial:2*trial+2]
        record = reader.records[keys[0]]
        eligible = [i for i,g in enumerate(record['record']['groups']) if g['split']=='validation']
        lanes = np.sort(rng.choice(eligible,32,replace=False))
        raw, values, _ = read_selected_trial(reader,keys,lanes)
        bank, context, seed_checks = teacher_seeds(reader,record,lanes,raw,values,options,builder)
        projection = jax.device_get(project(context,jnp.asarray(values['action__raw_normalized']),
            jnp.asarray(raw['q_before'][::5]),jnp.asarray(values['state__q_ref_rad'][:-1])))
        ref_error = np.max(np.abs(projection.reference_q-values['state__q_ref_rad'][1:]),axis=-1)
        native_hold = np.all(values['state__q_ref_rad'][1:]==values['state__q_ref_rad'][:-1],axis=-1)
        changed = (native_hold != projection.fallback_to_hold) & (ref_error>1e-4)
        cases = [dict(control=int(t),lane=int(lanes[i]),reference_max_error_rad=float(ref_error[t,i]),
            native_hold=bool(native_hold[t,i]),world_hold=bool(projection.fallback_to_hold[t,i]),
            current_safe=bool(projection.current_safe[t,i]),position_violation_m=float(projection.position_violation[t,i]),
            orientation_violation_rad=float(projection.orientation_violation[t,i]),
            linear_violation=float(projection.linear_max_violation[t,i])) for t,i in np.argwhere(changed)]
        reports.append(dict(trial=trial,lanes=lanes.tolist(),seed_checks=seed_checks,
            projection_reference_error=summarize_errors([ref_error.ravel()]),
            projection_switch_disagreements=cases))
        region = np.asarray([record['record']['groups'][i]['region_id'] for i in lanes])
        for horizon in (1,8,32,64,128):
            guard.check()
            starts = []
            for lane in range(len(lanes)):
                # Retain every selected validation family, including misses and invalid tails.
                release = np.flatnonzero(raw['event_controlled_release'][:,lane])//5
                candidates = sorted({s for e in release for s in range(max(0,e-horizon+1),min(e,128-horizon)+1)})
                starts.append(int(rng.choice(candidates)) if candidates else int(rng.integers(129-horizon)))
            starts = np.asarray(starts)
            times = starts[None,:]*5+np.arange(horizon*5)[:,None]
            selected = {k:v[times,np.arange(len(lanes))[None,:]] for k,v in raw.items()}
            action_times = starts[None,:]+np.arange(horizon)[:,None]
            actions = values['action__raw_normalized'][action_times,np.arange(len(lanes))[None,:]]
            seed = select_teacher_seeds(bank,starts)
            result = jax.device_get(evaluate(seed,context,jnp.asarray(actions),
                jnp.asarray(values['state__goal_xyz_m'][starts,np.arange(len(lanes))]),
                jax.tree.map(jnp.asarray,selected)))
            windows.append(dict(trial=trial,horizon=horizon,lanes=lanes.tolist(),starts=starts.tolist()))
            control_diagnostics.append(dict(trial=trial,horizon=horizon,
                delayed_command_max_per_control={k:summarize_errors([v.ravel()])
                    for k,v in result['control_errors'].items()},
                history_present_count=result['history_valid_count'].tolist()))
            own_reference = result['projection'].reference_q
            actual_reference = values['state__q_ref_rad'][action_times+1,np.arange(len(lanes))[None,:]]
            reference_error = np.max(np.abs(own_reference-actual_reference),axis=-1)
            native_holds = np.all(actual_reference == values['state__q_ref_rad'][
                action_times,np.arange(len(lanes))[None,:]],axis=-1)
            switch = (native_holds != result['projection'].fallback_to_hold) & (reference_error>1e-4)
            details=[]
            for local,lane in enumerate(lanes):
                def first(mask):
                    indices=np.flatnonzero(mask)
                    return int(indices[0]) if len(indices) else None
                details.append(dict(lane=int(lane),region_id=int(region[local]),
                    generator=record['record']['groups'][lane]['generator'],
                    start_control=int(starts[local]),
                    native_initial_task_eligible=bool(values['state__task_within_task_bounds'][starts[local],local]),
                    native_first_task_bound_failure=first(~values['state__task_within_task_bounds'][
                        starts[local]:starts[local]+horizon+1,local]),
                    own_reference_error_max_rad=float(reference_error[:,local].max()),
                    own_joint_error_max_rad=float(result['joint_max_error_rad'][:,local].max()),
                    own_racket_error_max_m=float(result['racket_position_error_m'][:,local].max()),
                    first_reference_error_over_1_mrad=first(reference_error[:,local]>.001),
                    first_joint_error_over_1_mrad=first(result['joint_max_error_rad'][:,local]>.001),
                    first_racket_error_over_1_mm=first(result['racket_position_error_m'][:,local]>.001),
                    first_hold_switch_disagreement=first(switch[:,local]),
                    hold_switch_disagreements=int(switch[:,local].sum())))
            control_diagnostics[-1]['lanes']=details
            for mode in ('own_control','common_command'):
                key = str(horizon)+'/'+mode
                nonfinite[key] = nonfinite.get(key,0)+int(result[mode]['nonfinite_windows'])
                for name,count in result[mode]['events'].items():
                    events.setdefault(key,{}).setdefault(name,0)
                    events[key][name] += int(count)
                for metric,(value,mask) in result[mode]['metrics'].items():
                    name = key+'/'+metric
                    pieces.setdefault(name,[]).append(value[mask])
                    region_pieces.setdefault(name,[]).append(np.broadcast_to(region,mask.shape)[mask])
            print(f'[INTEGRATION] trial={trial+1} H={horizon} '
                  f'nonfinite={int(result["own_control"]["nonfinite_windows"])}',flush=True)
            release_cpu_heap()
        # Commit completed evidence before reading the next large acquisition.
        progress = dict(trials=reports,windows=windows,events=events,nonfinite_windows=nonfinite,
            control_diagnostics=control_diagnostics,
            metrics={key:summarize_errors(value) for key,value in pieces.items()})
        write_json_unique(args.output_dir/f'progress_trial_{trial:02d}.json',progress,guard.budget)
        del bank, raw, values, selected, seed, result
        release_cpu_heap()
    summary, regions = {}, {}
    for key,values in pieces.items():
        summary[key] = summarize_errors(values)
        region = np.concatenate(region_pieces[key])
        errors = np.concatenate(values)
        regions[key] = {str(int(r)):summarize_errors([errors[region==r]]) for r in np.unique(region)}
    report.update(status='INTEGRATION_DIAGNOSTIC_COMPLETE',dataset_sha256=reader.identity,
        references=refs,auxiliary_reference=aux_ref,metrics=summary,by_region=regions,
        events=events,nonfinite_windows=nonfinite,trials=reports,windows=windows,
        control_diagnostics=control_diagnostics,
        window_selection_sha256=config_hash(windows),learned_contact_residual=False)


def release_cpu_heap():
    """Return unused glibc arenas after compilation/reads; no tensor is changed."""
    import ctypes
    gc.collect()
    libc=ctypes.CDLL(None)
    if hasattr(libc,'malloc_trim'):
        libc.malloc_trim(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--trials',type=int,default=10,choices=(1,10))
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_full_prior_integration_audit_v1',training_version='WMJ-V1',status='INCOMPLETE',
            config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,actor_updates=0,critic_updates=0,
            world_qualified=False,scope='heldout_own_control_and_common_command_same_full_seeds',
            source_sha256={p.name:file_hash(p) for p in sources},
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report,source={p.name:p.read_text() for p in sources}),guard.budget)
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
