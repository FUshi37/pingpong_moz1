"""Frozen auxiliary candidate against authenticated, saved native action forks."""

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import compare_effects, compare_with_repeat, model_event_trace, summarize_outcomes
from .artifacts import read_array_archive, write_json_unique
from .audit_action_effects import initial_world
from .audit_full_prior_world import release_cpu_heap
from .auxiliary_refresh import restore_auxiliary
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .full_prior_world import full_prior_world_rollout
from .model_numerics import precision_scope
from .native_observation import NativeObservationAdapter
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .physical_events import init_physical_events, scan_physical_events
from .pilot_contract import action_gate, counterfactual_actions
from .replay_support import _restore
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .support_data import SupportReplayReader
from .trajectory_data import TrajectoryDataset
from .world_control import ControlContract


def audit(cfg, args, guard, report):
    root = repository_root() / cfg['resources']['output_root']
    settings = cfg['bounded_imagination_pilot']
    evidence = json.loads(authenticate(repository_root(), settings['action_reference']).read_text())
    old_dir = root / 'full_prior_action_effect_a04'
    old_report = json.loads((old_dir / 'result.json').read_text())
    for relative, digest in evidence['evidence'].items():
        if relative.startswith(old_dir.name + '/') and file_hash(root / relative) != digest:
            raise ValueError('Saved native action evidence changed')
    reader = SupportReplayReader(root / 'support_replay_full_a01', guard.budget, max_bytes=0)
    params, options, refs, parent, parent_ref = load_components(cfg, reader,
        robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    restored, _ = restore_auxiliary(cfg, settings['candidate_checkpoint'])
    auxiliary = dict(parameters=restored['parameters'], scale=restored['scale'])
    # Match the established loader's inference object exactly.
    if set(auxiliary) != set(parent):
        raise ValueError('Unexpected auxiliary inference layout')
    frozen = numeric_tree_digest((params, auxiliary))
    collection = Path(reader.report['source_collection'])
    source = json.loads((collection / 'result.json').read_text())
    dataset = TrajectoryDataset.open(collection / 'dataset', guard.budget)
    bundle = build_native_environment(cfg, n_envs=1024)
    env = bundle.environment
    control = ControlContract.from_native(env)
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']), 1024))
    jax.block_until_ready(fresh)
    history, _ = NativeObservationAdapter(env, fresh).initialize(fresh)
    template = dict(native=fresh._replace(model=None), history=history,
        events=init_physical_events(1024), ball_model_valid=jnp.ones(1024, bool),
        goal=jnp.zeros((1024, 3), jnp.float32))

    @jax.jit
    def run(initial, context, events, valid_before, actions, goal):
        _, trace = full_prior_world_rollout(params['robot'], auxiliary, initial, actions,
            context=context, control_contract=control, task_goal_xyz=goal,
            ar1_enabled=False, **options)
        physical = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), trace['physics'])
        with precision_scope(options['matmul_precision']):
            event_trace = model_event_trace(physical, surface=options['surface'],
                ball_radius_m=options['ball_radius_m'], contact_contract=options['contact_contract'])
        finite = jnp.all(jnp.isfinite(physical['q_after_rad']), axis=-1)
        for name in ('ball_qpos_after', 'ball_qvel_after'):
            finite &= jnp.all(jnp.isfinite(event_trace[name]), axis=-1)
        valid = valid_before[None] & (jnp.cumsum((~finite).astype(jnp.int32), axis=0) == 0)
        _, labels = scan_physical_events(events, event_trace, valid=valid,
            initialization_active=jnp.zeros_like(valid))
        return labels, trace['control']['accepted_action'], jnp.all(finite)

    names = list(dataset.chunks)
    native = {k: [] for k in ('base', 'plus', 'minus')}
    world = {k: [] for k in native}
    repeats, cases, records = [], [], []
    report.update(model_references=refs, parent_checkpoint=parent_ref,
        candidate_checkpoint=settings['candidate_checkpoint'], native_steps_executed=0,
        native_evidence=settings['action_reference'], frozen_before_sha256=frozen)
    for trial in range(10):
        guard.check()
        saved = json.loads((old_dir / f'trial_{trial:02d}.json').read_text())
        first = dataset.chunks[names[2 * trial]]
        if saved['seed_reference'] != first['seed_ref']:
            raise ValueError('Counterfactual initial seed reference changed')
        lanes = np.array([row['lane'] for row in saved['cases']])
        for row in saved['cases']:
            group = first['groups'][row['lane']]
            if group['split'] != 'validation' or group['root_seed_family'] != row['root_seed_family']:
                raise ValueError('Native evidence has an unexpected data partition')
        full = _restore(first['seed_ref'], template, source['context']['sha256'])
        initial, context, events = initial_world(full, env, options, lanes)
        baseline = np.concatenate([read_array_archive(dataset.root / f'chunk_{name}.npz',
            dataset.chunks[name]['archive'], names=['action__raw_normalized'])['action__raw_normalized']
            for name in names[2 * trial:2 * trial + 2]])
        actions = counterfactual_actions(baseline, saved['cases'])
        outcomes, accepted = {}, {}
        for branch in native:
            labels, taken, finite = jax.device_get(run(initial, context, events,
                full['ball_model_valid'][lanes], jnp.asarray(actions[branch]), full['goal'][lanes]))
            if not finite:
                raise FloatingPointError('Nonfinite candidate action-effect trajectory')
            outcomes[branch] = summarize_outcomes(labels)
            accepted[branch] = taken
            native[branch].extend(saved['native'][branch])
            world[branch].extend(outcomes[branch])
        current_cases = [dict(row, world_accepted_central_pulse_norm=float(np.linalg.norm(
            accepted['plus'][:, i] - accepted['minus'][:, i]))) for i, row in enumerate(saved['cases'])]
        cases.extend(current_cases)
        repeats.extend(saved['native_repeat'])
        result = dict(trial=trial, cases=current_cases, world=outcomes,
            native_reference=dict(path=str(old_dir / f'trial_{trial:02d}.json'),
                sha256=file_hash(old_dir / f'trial_{trial:02d}.json')),
            comparison=compare_effects(saved['native'], outcomes),
            repeat_comparison=compare_with_repeat(saved['native'], outcomes, saved['native_repeat'], current_cases))
        digest = write_json_unique(args.output_dir / f'trial_{trial:02d}.json', result, guard.budget)
        records.append(dict(trial=trial, sha256=digest))
        print(f'[ACTION_RECHECK] {trial+1}/10, all48 candidate cases finite', flush=True)
        del full, baseline, labels, accepted
        release_cpu_heap()
    comparison = compare_with_repeat(native, world, repeats, cases)
    report.update(status='CANDIDATE_ACTION_RECHECK_COMPLETE', trials=records,
        comparison=compare_effects(native, world), repeat_comparison=comparison,
        pilot_gate=action_gate(old_report['repeat_comparison'], comparison),
        frozen_after_sha256=numeric_tree_digest((params, auxiliary)), all_predictions_finite=True)
    if report['frozen_after_sha256'] != frozen:
        raise ValueError('Frozen world parameters changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 1800.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_candidate_action_recheck', status='INCOMPLETE',
            training_version='WMJ-V1', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in sources},
            actor_updates=0, critic_updates=0, world_updates=0, world_qualified=False)
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
