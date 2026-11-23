"""Frozen free-attitude control and model audit on independent training sources."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .artifacts import read_array_archive
from .audit_full_prior_world import AuditReader, release_cpu_heap
from .audit_new_pose_world import native_world, finite_intervals
from .auxiliary_refresh import restore_auxiliary
from .ball_flight import ballistic_apex
from .build_training_seed_bank import load_training_bank
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .free_imagination import make_runtime, select_verified_state
from .free_orientation import install_position_only_reference
from .free_orientation import free_physical_bounds
from .imagined_env import EpisodeContract, finite_physics
from .imagined_rewards import RewardContract
from .migration import actor_mean
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .native_observation import NativeObservationAdapter
from .native_pair import assess_native_pair
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .paired_precision import load_actor
from .physical_events import init_physical_events, scan_physical_events
from .runtime_guard import RunGuard
from .state_adapter import concatenate_phase_group, numeric_tree_digest
from .teacher_trace import instrument_environment
from .train_pose_trial import phase_budget
from .world_control import ControlContract, control_from_native


def take_native(tree, indices, count):
    def take(value):
        if value.ndim == 0:
            return value
        if value.shape[0] != count:
            raise ValueError('Unregistered native lane axis')
        return value[indices]
    return jax.tree.map(take, tree)


def restore_selected(reference, template, lanes, context, *, prefix=''):
    path = Path(reference['manifest'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Original source reference changed')
    manifest = json.loads(path.read_text())
    paths = {prefix + jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(template)[0]}
    names = {f'leaf_{i:06d}' for i, row in enumerate(manifest['leaves']) if row['path'] in paths}
    arrays = read_array_archive(path.parent / manifest['archive_file'], manifest['archive'], names=names)
    return select_verified_state(template, arrays, manifest, prefix=prefix, lanes=lanes,
                                 original_count=1024, context_sha256=context)


def restore_native_sources(env, rows, source, seed):
    n = len(rows)
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(seed), n))
    context = source['context']['sha256']
    model = restore_selected(source['context']['reference'], fresh.model,
                             [r['lane'] for r in rows], context)
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(row['initial_seed_reference']['manifest'], []).append(i)
    pieces, order = [], []
    for indices in groups.values():
        selected = [rows[i] for i in indices]
        template = take_native(fresh._replace(model=None), np.arange(len(indices)), n)
        pieces.append(restore_selected(selected[0]['initial_seed_reference'], template,
            [r['lane'] for r in selected], context, prefix="['native']"))
        order.extend(indices)
    native = concatenate_phase_group(pieces)
    native = take_native(native, np.argsort(order), n)._replace(model=model)
    if np.any(np.asarray(native.step_count) != 0):
        raise ValueError('This population must retain genuine control0 resets')
    return native


def pose_angles(pose, reference):
    a, b = pose[..., 3:6], pose[..., 6:9]
    rotation = np.stack((a, b, np.cross(a, b)), axis=-1)
    relative = np.einsum('ji,...jk->...ik', np.asarray(reference), rotation)
    return np.rad2deg(np.stack((np.arctan2(-relative[..., 1, 2], relative[..., 2, 2]),
        np.arcsin(np.clip(relative[..., 0, 2], -1., 1.))), axis=-1))


def maximum_tree_delta(a, b):
    return max(float(np.max(np.abs(np.asarray(x).astype(float)-np.asarray(y).astype(float))))
               for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)) if np.size(x))


def endpoint_bounds(env, native, adapter, control, episode):
    """Trace only native endpoint values; full seed validation remains on host."""
    _, pose = adapter._values(native)
    physics = SimpleNamespace(q_rad=native.data.qpos[:, env.arm_qadr],
        ball=SimpleNamespace(position_m=native.data.qpos[:, env.ball_qadr:env.ball_qadr+3]))
    return free_physical_bounds(physics, pose, control, episode)


def audit(cfg, args, guard, report, plan):
    from mjx_topology_execution import make_collect
    root = repository_root() / cfg['resources']['output_root']
    rows = plan['source_rows']; n = len(rows)
    bundle = build_native_environment(cfg, n_envs=n, stage_index=0)
    env = bundle.environment
    report['native_projection'] = install_position_only_reference(env, bundle.stack)
    control = ControlContract.from_native(env)
    reader = AuditReader(root / 'support_replay_full_a01', guard.budget, max_bytes=0)
    parameters, options, refs, old_aux, old_aux_ref = load_components(cfg, reader,
        robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    bank, metadata = load_training_bank(cfg, plan['training_bank'], reader=reader, options=options,
        control_contract=control, model_references=dict(refs, auxiliary=old_aux_ref))
    if [metadata['source_rows'][i] for i in plan['source_indices']] != rows:
        raise ValueError('Independent source selection changed')
    initial_course, _ = bank.gather(np.asarray(plan['source_indices']))
    print('[FREE_SETUP] authenticated models and independent bank restored', flush=True)
    auxiliary, _ = restore_auxiliary(cfg, plan['auxiliary'])
    auxiliary = dict(parameters=auxiliary['parameters'], scale=auxiliary['scale'])
    robot = parameters['robot']; actor, actor_meta = load_actor(cfg, plan['actor'])
    frozen = numeric_tree_digest((actor, robot, auxiliary))
    runtime = make_runtime(robot, auxiliary, options, control)
    episode = EpisodeContract.from_config(cfg)
    reward = RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    source = json.loads((Path(reader.report['source_collection']) / 'result.json').read_text())
    native = restore_native_sources(env, rows, source, plan['seed'])
    adapter = NativeObservationAdapter(env, native, full_chain=True)
    history, _ = adapter.initialize(native)
    events = init_physical_events(n)
    initial, context = native_world(env, adapter, native, history, events)
    goals = jnp.asarray([r['first_goal_xyz_m'] for r in rows], jnp.float32)
    delta = dict(physics=maximum_tree_delta(initial.physics, initial_course.world.physics),
        control=maximum_tree_delta(initial.control, initial_course.world.control),
        observation=maximum_tree_delta(initial.observation, initial_course.world.observation),
        cached_pose=maximum_tree_delta(initial.cached_racket_pose, initial_course.world.cached_racket_pose),
        context=maximum_tree_delta(context, initial_course.context))
    observation_deltas = {jax.tree_util.keystr(p): float(np.max(np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float))))
        for (p,a),b in zip(jax.tree_util.tree_flatten_with_path(initial.observation)[0],
                           jax.tree.leaves(initial_course.world.observation)) if np.size(a)}
    if max(v for k,v in delta.items() if k!='observation') > 1e-5:
        raise ValueError('Original native training seed and complete bank disagree: ' + str(delta))
    # These are authenticated true reset states, with one honest history row.
    # Keep the actual native observer history as the common seed rather than
    # rejecting tiny reconstructed-FK differences or changing either FK path.
    initial_course = initial_course._replace(world=initial, context=context)
    print('[FREE_SETUP] common physical/control/history seed ready; deltas=' + str(delta), flush=True)
    report.update(initial_source_delta=delta, initial_native_sha256=numeric_tree_digest(native),
        bank_vs_native_observation_deltas=observation_deltas,
        common_history='actual full-chain native reset observer history used by both recursions',
        initial_world_sha256=numeric_tree_digest(initial), model_references=dict(refs, auxiliary=plan['auxiliary']),
        actor_metadata=actor_meta, frozen_parameters_before_sha256=frozen,
        source_collection_sha256=file_hash(Path(reader.report['source_collection'])/'result.json'),
        native_xml=dict(path=str(bundle.stack.xml), sha256=file_hash(Path(bundle.stack.xml))))
    original = env._step_impl
    report['observer'] = instrument_environment(env, bundle.stack, support_trace=True)

    def collect(current, paired, event):
        action = actor_mean(actor, adapter.observation(current, paired, goals))
        after, _, _, _, metrics = env.step(current, action)
        trace = metrics['wmj_trace']
        finite = finite_intervals(trace)
        event, labels = scan_physical_events(event, trace, valid=finite & ~trace['other_ball_collision'],
                                             initialization_active=jnp.zeros_like(finite))
        paired, _ = adapter.advance(after, paired)
        within, bounds = endpoint_bounds(env, after, adapter, control, episode)
        usable = jnp.all(finite & ~trace['other_ball_collision'], axis=0)
        return after, paired, event, action, trace, labels, within & usable, bounds

    scoped, execution = make_collect(collect, mode='project')
    step = jax.jit(scoped); report['execution'] = execution
    guard.check(); started = time.monotonic()
    first = step(native, history, events); jax.block_until_ready(first)
    env._step_impl = original
    plain_scoped, _ = make_collect(lambda s, a: env.step(s, a)[0], mode='project')
    plain = jax.jit(plain_scoped)
    expected = plain(native, first[3]); jax.block_until_ready(expected)
    pair = assess_native_pair(expected, first[0], np.ones(n, bool), env,
                             limits=cfg['frozen_evaluation_numerics']['absolute_tolerances'])
    report['native_pair'] = pair
    if not pair['pass_']:
        raise ValueError('Native observation instrumentation failed numerical pairing')
    del env._wmj_trace_identity
    instrument_environment(env, bundle.stack, support_trace=True)
    states = [jax.device_get(initial)]; event_states = [jax.device_get(events)]
    actions, traces, labels, okay, native_bounds = [], [], [], [], []
    for tick in range(128):
        guard.check()
        result = first if tick == 0 else step(native, history, events)
        native, history, events = result[:3]
        action, trace, label, good, bounds = jax.device_get(result[3:])
        world, following_context = native_world(env, adapter, native, history, events)
        if tick in (0,127) and numeric_tree_digest(following_context)!=numeric_tree_digest(context):
            raise ValueError('Native episode context changed during control replay')
        states.append(jax.device_get(world)); event_states.append(jax.device_get(events))
        actions.append(action); traces.append(trace); labels.append(label); okay.append(good); native_bounds.append(bounds)
        if tick % 32 == 31:
            print(f'[FREE_NATIVE] {tick+1}/128 controls', flush=True)
    report['native_wall_s'] = time.monotonic()-started
    world_bank = jax.tree.map(lambda *x: np.stack(x), *states)
    events_bank = jax.tree.map(lambda *x: np.stack(x), *event_states)
    raw = jax.tree.map(lambda *x: np.stack(x), *traces)
    observed_labels = jax.tree.map(lambda *x: np.stack(x), *labels)
    actions = np.stack(actions); okay = np.stack(okay)
    prefix = np.cumprod(okay, axis=0).astype(bool)
    native_angles = pose_angles(world_bank.cached_racket_pose[1:], control.projection.reference_rotation)
    report['native_population'] = dict(source_count=n, unique_regions=len({r['region_id'] for r in rows}),
        first_failure_control=[int(np.flatnonzero(~okay[:, i])[0]+1) if (~okay[:, i]).any() else None for i in range(n)],
        valid_prefix_controls=int(prefix.sum()), max_abs_roll_pitch_deg=np.max(np.abs(native_angles), axis=(0, 1)).tolist(),
        prefix_max_abs_roll_pitch_deg=np.max(np.abs(native_angles[prefix]), axis=0).tolist() if prefix.any() else None,
        prefix_over_2deg_controls=int((prefix & (np.abs(native_angles).max(axis=-1)>2)).sum()),
        other_ball_collision_substeps=int(raw['other_ball_collision'].sum()),
        controlled_releases=int((observed_labels['controlled_release'] & prefix[:, None]).sum()),
        bounded_angle_admission_used=False)
    step.clear_cache(); plain.clear_cache(); jax.clear_caches(); release_cpu_heap()

    @jax.jit
    def replay_controller(initial_control, measured, raw_actions):
        def tick(current, inputs):
            return runtime.controller(current, inputs[0], inputs[1], context,
                                      chain=options['chain'], contract=control)
        def capture(current, inputs):
            after, tr = tick(current, inputs)
            return after, (after, tr['delayed_commands'])
        return jax.lax.scan(capture, initial_control, (raw_actions, measured))

    _, (replayed, commands) = jax.device_get(replay_controller(initial.control,
        jnp.asarray(world_bank.physics.q_rad[:-1]), jnp.asarray(actions)))
    errors = dict(reference_q_rad=np.max(np.abs(replayed.reference_q_rad-world_bank.control.reference_q_rad[1:]), axis=-1),
        accepted_normalized=np.max(np.abs(replayed.previous_accepted_action-world_bank.control.previous_accepted_action[1:]), axis=-1))
    for name, key in [('q', 'delayed_q_rad'), ('dq', 'delayed_dq_rad_s'), ('ddq', 'delayed_ddq_rad_s2')]:
        errors[key] = np.max(np.abs(commands[name]-raw[f'delayed_{name}_command']), axis=(1, 3))
    tolerances = plan['control_tolerances']
    control_metrics = {key:dict(all=distribution(value), task_prefix=distribution(value[prefix])) for key, value in errors.items()}
    control_pass = all(v['all']['nonfinite']==0 and v['all']['maximum']<=tolerances[k] for k,v in control_metrics.items())
    report['control_replay'] = dict(metrics=control_metrics, tolerances=tolerances, pass_=control_pass,
        full_memory_max_abs=maximum_tree_delta(replayed, jax.tree.map(lambda x:x[1:], world_bank.control)),
        scope='persistent own RMP/filter/delay memory with native measured q and identical raw actions')
    replay_controller.clear_cache(); release_cpu_heap()

    entries = raw['contact_overlap'].reshape(640, n)
    first_contact = np.where(entries.any(axis=0), entries.argmax(axis=0)//5, -1)
    starts = np.stack((np.zeros(n, int), np.maximum(first_contact-2, 0), np.full(n,32), np.full(n,64)))
    window_lanes = np.tile(np.arange(n), 4); window_starts = starts.ravel()
    enabled = np.ones(4*n, bool); enabled[n:2*n] = first_contact>=0
    seed_world = jax.tree.map(lambda x:jnp.asarray(x[window_starts, window_lanes]), world_bank)
    seed_events = jax.tree.map(lambda x:jnp.asarray(x[window_starts, window_lanes]), events_bank)
    window_context = jax.tree.map(lambda x:x[window_lanes], context)
    window_goal = goals[window_lanes]
    times = np.arange(128)[:,None]+window_starts[None]
    available = (times<128)&enabled[None]
    safe_times = np.minimum(times,127)
    window_actions = np.where(available[...,None], actions[safe_times,window_lanes], 0.)

    @jax.jit
    def predict(state, event, raw_actions):
        def tick(carry, action):
            before, event = carry
            after, trace = runtime.world_step(robot, auxiliary, before, action,
                context=window_context, control_contract=control, task_goal_xyz=window_goal,
                ar1_enabled=False, **options)
            event_trace = model_event_trace(trace['physics'], surface=options['surface'],
                ball_radius_m=options['ball_radius_m'], contact_contract=options['contact_contract'])
            valid = jnp.broadcast_to(finite_physics(after.physics), (5, len(window_lanes)))
            event, labels = scan_physical_events(event, event_trace, valid=valid,
                                                 initialization_active=jnp.zeros_like(valid))
            within, _ = runtime.bounds(after.physics, after.cached_racket_pose, control, episode)
            # Keep dense traces only in RAM. No teacher correction enters recursion.
            observed = dict(position=after.physics.ball.position_m, velocity=after.physics.ball.velocity_m_s,
                q=after.physics.q_rad, dq=after.physics.dq_rad_s, pose=after.cached_racket_pose,
                reference=after.control.reference_q_rad, observation=trace['actor_observation'],
                within=within, finite=finite_physics(after.physics), labels=labels)
            return (after, event), observed
        return jax.lax.scan(tick, (state, event), raw_actions)

    guard.check(); started=time.monotonic()
    _, prediction = jax.device_get(predict(seed_world, seed_events, jnp.asarray(window_actions)))
    report['world_compile_and_rollout_s'] = time.monotonic()-started
    truth = jax.tree.map(lambda x:x[np.minimum(times+1,128),window_lanes], world_bank)
    predicted_apex = np.asarray(ballistic_apex(jnp.asarray(prediction['position']),jnp.asarray(prediction['velocity']))[0])
    true_apex = np.asarray(ballistic_apex(jnp.asarray(truth.physics.ball.position_m),jnp.asarray(truth.physics.ball.velocity_m_s))[0])
    metrics = dict(ball_position_m=np.linalg.norm(prediction['position']-truth.physics.ball.position_m,axis=-1),
        ball_velocity_m_s=np.linalg.norm(prediction['velocity']-truth.physics.ball.velocity_m_s,axis=-1),
        ballistic_apex_m=np.linalg.norm(predicted_apex-true_apex,axis=-1),
        racket_position_m=np.linalg.norm(prediction['pose'][...,:3]-truth.cached_racket_pose[...,:3],axis=-1),
        joint_position_rad=np.max(np.abs(prediction['q']-truth.physics.q_rad),axis=-1),
        joint_velocity_rad_s=np.max(np.abs(prediction['dq']-truth.physics.dq_rad_s),axis=-1),
        control_reference_rad=np.max(np.abs(prediction['reference']-truth.control.reference_q_rad),axis=-1))
    native_prefix = prefix[safe_times,window_lanes]&available
    model_prefix = np.cumprod(prediction['within']&prediction['finite'],axis=0).astype(bool)&available
    angles = np.abs(native_angles[safe_times,window_lanes]).max(axis=-1)
    results = {}
    for h in plan['horizons']:
        index=h-1; scopes=dict(available=available[index], native_task_prefix=native_prefix[index],
            common_task_prefix=native_prefix[index]&model_prefix[index],
            native_prefix_over_2deg=native_prefix[index]&(angles[index]>2.))
        upward = (prediction['velocity'][index, :, 2]>0) & (truth.physics.ball.velocity_m_s[index, :, 2]>0)
        results[str(h)] = {scope:{key:distribution(value[index][mask & upward if key=='ballistic_apex_m' else mask]) for key,value in metrics.items()}
                          for scope,mask in scopes.items()}
    report['horizon_errors'] = results
    report['windows'] = dict(starts=starts.tolist(), first_native_contact_control=first_contact.tolist(),
        enabled=enabled.tolist(), count=4*n, native_prefix_end_counts={str(h):int(native_prefix[h-1].sum()) for h in plan['horizons']},
        common_prefix_end_counts={str(h):int((native_prefix[h-1]&model_prefix[h-1]).sum()) for h in plan['horizons']},
        nonfinite_prediction_controls=int((available & ~prediction['finite']).sum()))
    # Pair first controlled releases within each horizon, keeping missing and
    # different contact counts visible; do not compare release vs an arbitrary clock.
    release_reports = {}
    for h in plan['horizons']:
        pairs=[]; missing_native=missing_model=unequal=0
        for w in range(4*n):
            s=window_starts[w]; lane=window_lanes[w]
            if not enabled[w] or s+h>128:
                continue
            actual=observed_labels['controlled_release'][s:s+h,:,lane].reshape(-1)
            model=prediction['labels']['controlled_release'][:h,:,w].reshape(-1)
            ai=np.flatnonzero(actual); mi=np.flatnonzero(model)
            missing_native+=not bool(len(ai)); missing_model+=not bool(len(mi)); unequal+=len(ai)!=len(mi)
            if not len(ai) or not len(mi):
                continue
            a=int(ai[0]); m=int(mi[0]); ac=a//5; mc=m//5
            if not prefix[s+ac,lane] or not native_prefix[mc,w]:
                continue
            ap=observed_labels['release_position_m'][s+ac,a%5,lane]
            av=observed_labels['release_velocity_m_s'][s+ac,a%5,lane]
            mp=prediction['labels']['release_position_m'][mc,m%5,w]
            mv=prediction['labels']['release_velocity_m_s'][mc,m%5,w]
            def apex(p,v):
                t=max(float(v[2]),0.)/9.81
                return p+v*t+np.array([0.,0.,-.5*9.81*t*t])
            pairs.append(dict(window=w, source_index=plan['source_indices'][lane],
                native_release_substep=a,model_release_substep=m, native_count=int(len(ai)),model_count=int(len(mi)),
                velocity_error_m_s=float(np.linalg.norm(mv-av)),apex_error_m=float(np.linalg.norm(apex(mp,mv)-apex(ap,av))),
                native_position_m=ap.tolist(),native_velocity_m_s=av.tolist(),model_position_m=mp.tolist(),
                model_velocity_m_s=mv.tolist(),goal_xyz_m=np.asarray(goals[lane]).tolist(),
                native_roll_pitch_deg=native_angles[s+ac,lane].tolist()))
        release_reports[str(h)]=dict(paired=len(pairs),missing_native=missing_native,missing_model=missing_model,
            unequal_event_counts=unequal,velocity_error_m_s=distribution([p['velocity_error_m_s'] for p in pairs]),
            apex_error_m=distribution([p['apex_error_m'] for p in pairs]),pairs=pairs)
    report['release_alignment'] = release_reports
    predict.clear_cache(); jax.clear_caches(); release_cpu_heap()

    @jax.jit
    def course_rollout(state, raw_actions):
        def tick(current, action):
            after,tr=runtime.course_step(robot,auxiliary,current,action,current.task.goal_xyz,
                stage_id='S01',control_contract=control,episode_contract=episode,reward_contract=reward,**options)
            return after,dict(valid=tr.valid,terminated=tr.terminated,model_truncated=tr.model_truncated,
                reward=tr.reward,angle_check=tr.task_events['bound_orientation'],
                pose=after.world.cached_racket_pose)
        return jax.lax.scan(tick,state,raw_actions)
    guard.check()
    final, course = jax.device_get(course_rollout(initial_course,jnp.asarray(actions)))
    report['course_no_update'] = dict(valid_controls=int(course['valid'].sum()),
        physical_failures=int(course['terminated'].sum()),model_truncations=int(course['model_truncated'].sum()),
        finite_rewards=bool(np.isfinite(course['reward']).all()),
        no_angle_only_failure=bool(np.all(course['angle_check'][course['valid']])),
        final_history_present=np.asarray(final.world.observation.history.paired.valid).sum(axis=-1).tolist()
            if hasattr(final.world.observation.history.paired,'valid') else None,
        terminated_lanes=np.flatnonzero(final.terminated).tolist())
    course_rollout.clear_cache()
    after_hash=numeric_tree_digest((actor,robot,auxiliary))
    if after_hash!=frozen:
        raise ValueError('Frozen parameters changed')
    report.update(status='FREE_IMAGINATION_FROZEN_AUDIT_COMPLETE',control_replay_pass=control_pass,
        frozen_parameters_after_sha256=after_hash, parameters_unchanged=True,
        task_qualified=False, world_qualified=False, promoted=False,
        limitations=['32 pre-registered independent training sources, not full-domain validation',
            'Same native actions, own model state/control/history; no policy adaptation',
            'Windows after first native failure are diagnostic only',
            'World model does not simulate other-body collisions',
            'Continuous ballistic apex metric differs slightly from actual 1ms discrete-flight apex'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config();setting=cfg['free_imagination_candidate']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20:
        raise ValueError('Frozen free-imagination phase is not authorized')
    if any(plan[k] for k in ('actor_updates','critic_updates','world_updates','default_replacement')):
        raise ValueError('Only frozen integration is authorized')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_free_imagination_frozen_audit',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            command=subprocess.list2cmdline(['python','-m','world_model.audit_free_imagination',
                '--output-dir',str(args.output_dir),'--gpu-uuid',args.gpu_uuid]),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in paths},actor_updates=0,critic_updates=0,world_updates=0)
        write_compact_json(args.output_dir/'implementation.json',dict(report=report,sources={
            name:(Path(__file__).parent/name).read_text() for name in ('free_imagination.py','audit_free_imagination.py')}),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
