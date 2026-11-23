"""Frozen model audit on independently cached flat validation poses; no updates."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_array_archive_unique, write_json_unique
from .audit_full_prior_world import AuditReader, release_cpu_heap
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .evaluate_frozen import endpoint_code
from .fit_robot import summarize_errors
from .frozen_cases import validate_case_manifest, GROUPS
from .frozen_seed import load_warmed_seed
from .full_prior_world import FullPriorWorldState, full_prior_world_rollout, world_actor_observation
from .imagined_env import EpisodeContract
from .migration import actor_mean, expand_actor_history, load_v321_actor
from .native_observation import NativeObservationAdapter
from .native_pair import assess_native_pair
from .native_stack import build_native_environment
from .new_pose_audit import diagnostic_starts, prefix_validity, select_windows
from .normal_checkpoint import load_components
from .physical_events import init_physical_events, scan_physical_events
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state
from .teacher_initialization import begin_ball_trial
from .teacher_trace import instrument_environment
from .world_control import ControlContract, control_from_native
from .world_physics import PhysicsState


def native_world(env, adapter, native, history, events):
    """Extract a same-time full world seed, including measured support motion."""
    control, context = control_from_native(native)
    qpos = native.data.qpos[:, env.ball_qadr:env.ball_qadr + 7]
    qvel = native.data.qvel[:, env.ball_vadr:env.ball_vadr + 6]
    support = [env.mj_model.joint(f'LegWaist-{i}').id for i in range(6)]
    vadr = np.concatenate((np.asarray(env.arm_vadr), env.mj_model.jnt_dofadr[support]))
    physics = PhysicsState(native.data.qpos[:, adapter.feedback_qadr], native.data.qvel[:, vadr],
        ContactState(qpos[:, :3], qvel[:, :3], world_spin(qpos[:, 3:], qvel[:, 3:]),
                     events.contact_active, events.contact_duration_steps.astype(jnp.float32) * .001))
    return FullPriorWorldState(physics, control, adapter.world_observation_state(native, history),
                              adapter._values(native)[1]), context


def finite_intervals(trace):
    """All recorded numeric physics fields must be usable at a label interval."""
    valid = jnp.ones(trace['contact_overlap'].shape, bool)
    for value in trace.values():
        finite = jnp.isfinite(value)
        if value.ndim > 2:
            finite = jnp.all(finite, axis=tuple(range(2, value.ndim)))
        valid &= finite
    return valid


def archive_finite(path, arrays, guard):
    """Save continuous streams, retaining element masks for unusable values."""
    stored, missing = {}, {}
    for name, value in arrays.items():
        value = np.asarray(value)
        finite = np.isfinite(value)
        stored[name] = np.where(finite, value, 0) if not finite.all() else value
        if not finite.all():
            stored[name + '__finite'] = finite
            missing[name] = int((~finite).sum())
    result = write_array_archive_unique(path, stored, guard.budget)
    return dict(path=str(path), archive=result, nonfinite_replaced_with_zero_and_mask=missing)


def audit(cfg, args, guard, report):
    from mjx_topology_execution import make_collect

    root = repository_root()
    output_root = root / cfg['resources']['output_root']
    refs = cfg['frozen_evaluation_artifacts']
    static = json.loads(authenticate(root, refs['static_initialization']).read_text())
    manifest = json.loads(authenticate(root, refs['cases']).read_text())
    validate_case_manifest(manifest, cfg)
    cases = manifest['stages']['S01']
    if static['cases_reference'] != refs['cases'] or len(cases) != 128:
        raise ValueError('Fixed validation population identity changed')
    n = len(cases)
    goals = jnp.asarray([row['goals_xyz_m'][0] for row in cases], jnp.float32)
    bundle = build_native_environment(cfg, n_envs=n, stage_index=0)
    env = bundle.environment
    control = ControlContract.from_native(env)
    episode = EpisodeContract.from_config(cfg)
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(args.seed), n))
    jax.block_until_ready(fresh)
    cache_context = dict(schema='wmj_fixed_warmed_population', cases_sha256=manifest['cases_sha256'],
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        domain_sha256=config_hash(cfg['domain']), ids=list(range(n)), reset_seed=args.seed,
        warmup_controls=128, native_source_sha256=bundle.stack.spec.native_env_sha256,
        initialization_sha256=refs['static_initialization']['sha256'], warmup_ar1_enabled=False)
    warmed, rows = load_warmed_seed(refs['warmed_seed'], fresh, cache_context)
    if [r['case_id'] for r in rows] != [c['case_id'] for c in cases]:
        raise ValueError('Cached admission and fixed case identities differ')
    known = np.array([r['status'] == 'ADMITTED' for r in rows])
    native = jax.jit(lambda f, w: begin_ball_trial(env, f, w, goals, jnp.zeros((n, 3)),
                                                ar1_enabled=False))(fresh, warmed)
    adapter = NativeObservationAdapter(env, native, full_chain=True)
    history, _ = adapter.initialize(native)
    events = init_physical_events(n)
    actor = expand_actor_history(load_v321_actor(root, cfg['lineage']['actor_source']))
    actor_digest = numeric_tree_digest(actor)
    first, context = native_world(env, adapter, native, history, events)
    bank = [jax.device_get(first)]
    original = env._step_impl
    observer = instrument_environment(env, bundle.stack, support_trace=True)

    def collect(current, paired, event_state):
        observation = adapter.observation(current, paired, goals)
        action = actor_mean(actor, observation)
        result = env.step(current, action)
        following = result[0]
        trace = result[-1]['wmj_trace']
        valid = finite_intervals(trace) & ~trace['other_ball_collision']
        following_events, labels = scan_physical_events(event_state, trace, valid=valid,
            initialization_active=jnp.zeros_like(valid))
        following_history, _ = adapter.advance(following, paired)
        code = endpoint_code(env, following, adapter, control, episode)
        return following, following_history, following_events, trace, labels, action, code

    scoped, execution = make_collect(collect, mode='project')
    step = jax.jit(scoped)
    report.update(native_observer=observer, execution=execution, native_overrides=bundle.overrides,
        cases_reference=refs['cases'], warmed_seed=refs['warmed_seed'],
        admitted=int(known.sum()), unknown=int((~known).sum()),
        actor_reference=cfg['lineage']['actor_source'],
        native_xml=dict(path=str(bundle.stack.xml), sha256=file_hash(Path(bundle.stack.xml))),
        prediction_scope='common_raw_actions_with_own_predicted_control_physics_and_history',
        primary_scope='before_first_physical_or_task_bound_failure',
        auxiliary_scope='finite_physics_including_post_task_failure_diagnostic_only')
    started = time.monotonic()
    first_result = step(native, history, events)
    jax.block_until_ready(first_result)
    # Compile the preserved native method separately, with the identical action.
    env._step_impl = original
    plain_scoped, _ = make_collect(lambda s, a: env.step(s, a)[0], mode='project')
    plain = jax.jit(plain_scoped)
    expected = plain(native, first_result[-2])
    jax.block_until_ready(expected)
    pair = assess_native_pair(expected, first_result[0], known, env,
                              limits=cfg['frozen_evaluation_numerics']['absolute_tolerances'])
    report['native_pair'] = pair
    if not pair['pass_']:
        raise ValueError('Native trace observer failed the registered numerical pairing')
    # The observed step executable is already compiled, but restore its bound
    # source so a retrace cannot silently call the uninstrumented method.
    del env._wmj_trace_identity
    instrument_environment(env, bundle.stack, support_trace=True)
    traces, labels, actions, codes = [], [], [], []
    for tick in range(128):
        guard.check()
        result = first_result if tick == 0 else step(native, history, events)
        native, history, events = result[:3]
        trace, label, action, code = jax.device_get(result[3:])
        traces.append(trace)
        labels.append(label)
        actions.append(action)
        codes.append(code)
        state, following_context = native_world(env, adapter, native, history, events)
        if tick in (0, 127) and numeric_tree_digest(context) != numeric_tree_digest(following_context):
            raise ValueError('Native episode control context changed')
        bank.append(jax.device_get(state))
        if tick % 32 == 31:
            print(f'[NATIVE] {tick + 1}/128 controls; full state and1ms labels captured', flush=True)
    report['native_capture_wall_s'] = time.monotonic() - started
    if numeric_tree_digest(actor) != actor_digest:
        raise ValueError('Frozen actor changed')
    raw = {k: np.concatenate([r[k] for r in traces]) for k in traces[0]}
    for key in ('q_before', 'q_after', 'dq_before', 'dq_after'):
        raw[key] = np.concatenate((raw[key], raw['support_' + key]), axis=-1)
    event = {k: np.concatenate([r[k] for r in labels]) for k in labels[0]}
    for key in ('contact_entry', 'contact_release', 'controlled_release', 'contact_duration_steps'):
        raw['event_' + key] = event[key]
    finite = np.ones((640, n), bool)
    for value in raw.values():
        finite &= np.all(np.isfinite(value).reshape(640, n, -1), axis=-1)
    physical, prefix = prefix_validity(finite, raw['other_ball_collision'], np.stack(codes), known)
    raw['ball_model_valid'] = physical
    actions = np.stack(actions)
    bank = jax.tree.map(lambda *x: np.stack(x), *bank)
    report['native_streams'] = archive_finite(args.output_dir / 'native_streams.npz',
        {**{'substep__' + k: v for k, v in raw.items()}, 'task_prefix_valid': prefix,
         'raw_action': actions, 'endpoint_failure_code': np.stack(codes), 'admitted': known}, guard)
    report['native_cases'] = []
    for lane, case in enumerate(cases):
        failures = np.flatnonzero(~prefix[:, lane])
        releases = np.flatnonzero(event['controlled_release'][:, lane] & physical[:, lane])
        report['native_cases'].append(dict(case_id=case['case_id'], group=case['group'],
            admission='ADMITTED' if known[lane] else 'UNKNOWN',
            first_failure_ms=int(failures[0] + 1) if known[lane] and len(failures) else None,
            diagnostic_release_ms=(releases + 1).tolist(),
            valid_task_releases=int(np.sum(event['controlled_release'][:, lane] & prefix[:, lane]))))
    # Release native compiler memory before creating model rollout executables.
    del first_result, expected, fresh, warmed, native, history, events, first, state
    del traces, labels, step, plain, scoped, plain_scoped
    jax.clear_caches()
    gc.collect()
    release_cpu_heap()
    reader = AuditReader(output_root / 'support_replay_full_a01', guard.budget, max_bytes=0)
    parameters, options, model_refs, auxiliary, aux_ref = load_components(cfg, reader,
        robot_checkpoint=output_root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=output_root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=output_root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    frozen = numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary))
    report['model_references'] = dict(model_refs, auxiliary=aux_ref)

    @jax.jit
    def predict(seed, ctx, action, goal, actual):
        after, trace = full_prior_world_rollout(parameters['robot'], auxiliary, seed, action,
            context=ctx, control_contract=control, task_goal_xyz=goal, ar1_enabled=False, **options)
        physics = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trace['physics'])
        batch = prepare_position_contact_batch(None, actual)
        metrics, event_counts = ball_metrics(physics['contact'], batch)
        metrics.update(robot_metrics(physics, actual, batch))
        # Return masks/values for primary and diagnostic scopes on the host.
        model_finite = jnp.all(jnp.isfinite(trace['actor_observation']), axis=-1)
        for value in (physics['q_after_rad'], physics['dq_after_rad_s'],
                      physics['contact']['after'].position_m,
                      physics['contact']['after'].velocity_m_s):
            per_substep = jnp.all(jnp.isfinite(value), axis=-1)
            model_finite &= jnp.all(per_substep.reshape(action.shape[0], 5, -1), axis=1)
        return dict(metrics=metrics, event_counts=event_counts,
            model_finite=model_finite, final_observation=trace['actor_observation'][-1],
            final_history_present=after.observation.history.paired.present,
            entry=physics['contact']['entry'], release=physics['contact']['release'])

    summaries = {}
    lanes = np.flatnonzero(known)
    for horizon in (1, 8, 32, 64, 128):
        guard.check()
        starts = diagnostic_starts(event['controlled_release'] & physical, horizon)
        # Nonfinite seeds cannot be meaningfully simulated. Record and exclude
        # them before transfer; never repair them or replace their case IDs.
        seed = jax.tree.map(lambda x: x[starts, np.arange(n)], bank)
        seed_finite = np.ones(n, bool)
        for value in jax.tree.leaves(seed):
            seed_finite &= np.all(np.isfinite(value).reshape(n, -1), axis=-1)
        selected_lanes = lanes[seed_finite[lanes]]
        if not len(selected_lanes):
            summaries[str(horizon)] = dict(status='NO_FINITE_ADMITTED_SEEDS', excluded=lanes.tolist())
            continue
        selected_seed = jax.tree.map(lambda x: x[selected_lanes], seed)
        selected_context = jax.tree.map(lambda x: np.asarray(x)[selected_lanes], context)
        actual = {k: select_windows(v, starts * 5, horizon * 5)[:, selected_lanes] for k, v in raw.items()}
        primary = select_windows(prefix, starts * 5, horizon * 5)[:, selected_lanes]
        predicted = jax.device_get(predict(jax.tree.map(jnp.asarray, selected_seed),
            jax.tree.map(jnp.asarray, selected_context),
            jnp.asarray(select_windows(actions, starts, horizon)[:, selected_lanes]), goals[selected_lanes],
            jax.tree.map(jnp.asarray, actual)))
        final_true = jax.tree.map(lambda x: jnp.asarray(x[starts[selected_lanes] + horizon, selected_lanes]), bank)
        true_observation = np.asarray(world_actor_observation(final_true, goals[selected_lanes]))
        observation_error = np.abs(predicted['final_observation'] - true_observation)
        metrics = dict(predicted['metrics'])
        metrics['endpoint_actor_prefix_abs'] = (observation_error[:, :73].max(axis=-1), actual['ball_model_valid'][-1])
        metrics['endpoint_actor_history_abs'] = (observation_error[:, 73:].max(axis=-1), actual['ball_model_valid'][-1])
        result = dict(horizon=horizon, starts=starts.tolist(), case_ids=[cases[i]['case_id'] for i in selected_lanes],
            excluded_nonfinite_seed_ids=[cases[i]['case_id'] for i in lanes[~seed_finite[lanes]]],
            nonfinite_model_windows=int((~predicted['model_finite'].all(axis=0)).sum()),
            task_prefix_complete_windows=int(primary.all(axis=0).sum()),
            finite_physics_complete_windows=int(actual['ball_model_valid'].all(axis=0).sum()),
            metrics={}, groups={}, events={})
        for scope in ('task_prefix', 'diagnostic_physics'):
            validity = primary if scope == 'task_prefix' else actual['ball_model_valid']
            scoped = {}
            for key, (value, mask) in metrics.items():
                use = mask & (validity[-1] if value.ndim == 1 else validity)
                scoped[key] = summarize_errors([value[use]])
            result['metrics'][scope] = scoped
            result['groups'][scope] = {}
            for group in GROUPS:
                group_lanes = np.array([cases[i]['group'] == group for i in selected_lanes])
                result['groups'][scope][group] = {
                    key: summarize_errors([value[mask & (validity[-1] if value.ndim == 1 else validity) & group_lanes]])
                    for key, (value, mask) in metrics.items()}
            result['events'][scope] = {}
            for edge in ('entry', 'release'):
                true = actual['event_contact_' + edge]
                estimated = predicted[edge]
                result['events'][scope][edge] = dict(true=int((true & validity).sum()),
                    predicted=int((estimated & validity).sum()),
                    same_interval_missed=int((true & ~estimated & validity).sum()),
                    same_interval_extra=int((estimated & ~true & validity).sum()))
        # Full branch seeds preserve history and memory; continuous streams above
        # avoid repeating flattened actor histories for every transition.
        packed, seed_manifest = pack_numeric_state(dict(world=selected_seed, context=selected_context),
                                                    context_sha256=config_hash(report['model_references']))
        seed_archive = write_array_archive_unique(args.output_dir / f'seeds_h{horizon}.npz', packed, guard.budget)
        result['seeds'] = dict(archive_file=f'seeds_h{horizon}.npz', archive=seed_archive, state=seed_manifest)
        result['metric_arrays'] = archive_finite(args.output_dir / f'errors_h{horizon}.npz',
            {**{k: v[0] for k, v in metrics.items()}, **{k + '__valid': v[1] for k, v in metrics.items()},
             'task_prefix_valid': primary, 'model_finite': predicted['model_finite'],
             'native_valid': actual['ball_model_valid'], 'case_indices': selected_lanes}, guard)
        digest = write_json_unique(args.output_dir / f'horizon_{horizon}.json', result, guard.budget)
        summaries[str(horizon)] = dict(sha256=digest, metrics=result['metrics'], groups=result['groups'],
            events=result['events'], task_prefix_complete_windows=result['task_prefix_complete_windows'],
            nonfinite_model_windows=result['nonfinite_model_windows'])
        print(f'[WORLD] H={horizon}, complete task prefixes={result["task_prefix_complete_windows"]}, '
              f'nonfinite={result["nonfinite_model_windows"]}', flush=True)
    if frozen != numeric_tree_digest(dict(robot=parameters['robot'], auxiliary=auxiliary)):
        raise ValueError('Frozen world parameters changed')
    report.update(status='FROZEN_NEW_POSE_AUDIT_COMPLETE', horizons=summaries,
        frozen_before_sha256=frozen, frozen_after_sha256=frozen, actor_sha256=actor_digest,
        promoted=False, training_seed_bank_contains_validation_sources=False,
        limitations=['This is a fixed-policy common-action dynamics diagnostic, not learned task success',
                     'Same-interval contact disagreements include timing shifts',
                     'Unknown initialization cases remain unknown and are not replaced',
                     'No action-effect perturbation or model optimization is performed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=2026110925)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_new_pose_frozen_world_audit', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources}, seed=args.seed,
            actor_updates=0, critic_updates=0, world_updates=0)
        write_json_unique(args.output_dir / 'implementation.json',
                          dict(report=report, source={p.name: p.read_text() for p in sources}), guard.budget)
        write_json_unique(args.output_dir / 'config_snapshot.json', cfg, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
