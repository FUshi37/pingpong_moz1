"""Augment immutable 128-control teacher trials with native support/FK labels.

Replay every recorded raw action from the complete first seed. Do not restore
at the middle storage boundary, and do not combine old and new trajectories.
Lossless overlays store all canonical changes and the added support streams.
These augmented streams qualify dynamics only; old arm-only history seeds do
not become thirteen-joint observation-history seeds by relabeling them.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .collect_teacher import compact_state, _stack_cpu
from .collection_labels import label_task_bounds
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import dataset_identity
from .replay_overlay import encode_overlay, decode_overlay
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, restore_numeric_state
from .trajectory_data import TrajectoryDataset


def _restore(reference, template, context):
    path = Path(reference['manifest'])
    if file_hash(path) != reference['sha256']:
        raise ValueError('Seed manifest changed')
    manifest = json.loads(path.read_text())
    arrays = read_array_archive(path.parent/manifest['archive_file'], manifest['archive'])
    return restore_numeric_state(template, arrays, manifest, context_sha256=context)


def replay(cfg, args, guard, report):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .physical_events import init_physical_events, scan_physical_events
    from .teacher_trace import instrument_environment
    from mjx_topology_execution import make_collect

    dataset = TrajectoryDataset.open(args.collection/'dataset', guard.budget)
    source = json.loads((args.collection/'result.json').read_text())
    if (dataset.identity['control_sha256'] != config_hash(cfg['control'])
            or dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Cannot replay a different control/contact contract')
    names = list(dataset.chunks)
    if len(names) % 2:
        raise ValueError('Replay requires complete two-chunk trials')
    n = dataset.chunks[names[0]]['lanes']
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']), n))
    jax.block_until_ready(fresh)
    context = source['context']['sha256']
    model = _restore(source['context']['reference'], fresh.model, context)
    if numeric_tree_digest(model) != context:
        raise ValueError('Restored episode model differs from collection context')
    adapter = NativeObservationAdapter(env, fresh)
    history, _ = adapter.initialize(fresh)
    template = dict(native=fresh._replace(model=None), history=history,
        events=init_physical_events(n), ball_model_valid=jnp.ones(n, bool),
        goal=jnp.zeros((n, 3), jnp.float32))
    observer = instrument_environment(env, bundle.stack, support_trace=True)
    support = env._wmj_support_trace

    def values(native, goal):
        return {**compact_state(env, native, goal),
            'support_q_rad': native.data.qpos[:, support['qadr']],
            'support_dq_rad_s': native.data.qvel[:, support['vadr']]}

    def transition(full, action):
        after, _, _, native_done, metrics = env.step(full['native'], action)
        history, diagnostics = adapter.advance(after, full['history'])
        trace = metrics['wmj_trace']
        finite = jnp.ones((5, n), bool)
        for key in ('q_after', 'dq_after', 'support_q_after', 'support_dq_after',
                    'ball_qpos_after', 'ball_qvel_after'):
            finite &= jnp.all(jnp.isfinite(trace[key]), axis=-1)
        valid = full['ball_model_valid'][None, :] & (
            jnp.cumsum((trace['other_ball_collision'] | ~finite).astype(jnp.int32), axis=0) == 0)
        events, labels = scan_physical_events(full['events'], trace, valid=valid,
                                              initialization_active=jnp.zeros_like(valid))
        streams = {**trace, **{f'event_{key}': val for key, val in labels.items()},
                   'ball_model_valid': valid, 'robot_model_valid': finite}
        actions = dict(raw_normalized=action, clipped_normalized=jnp.clip(action, -1., 1.),
            projected_normalized=after.prev_action, accepted_normalized=after.prev_action,
            native_done_diagnostic=native_done)
        next_full = dict(native=after, history=history, events=events,
                        ball_model_valid=valid[-1], goal=full['goal'])
        return next_full, values(after, full['goal']), actions, streams, jnp.all(finite, axis=0)

    scoped, execution = make_collect(transition, mode='project')
    step = jax.jit(scoped)
    report.update(source_dataset_sha256=dataset_identity(dataset),
        source_collection=str(args.collection.absolute()),
        source_result_sha256=file_hash(args.collection/'result.json'),
        geometry=source['geometry'], observer=observer, execution=execution,
        context_sha256=context, n_envs=n, expected_chunks=len(names),
        observation_seed_qualification='legacy_arm_only_history_not_full_chain_policy_seed')
    write_json_unique(args.output_dir/'replay_manifest.json', report, guard.budget)
    count = min(len(names), args.diagnostic_trials*2) if args.diagnostic_trials else len(names)
    for index in range(0, count, 2):
        guard.check()
        first, second = (dataset.chunks[names[index+i]] for i in (0, 1))
        if (first['groups'] != second['groups'] or first['context_sha256'] != second['context_sha256']
                or first['lanes'] != n or second['lanes'] != n):
            raise ValueError('Trial pair has incompatible grouping/context')
        full = _restore(first['seed_ref'], template, context)
        full['native'] = full['native']._replace(model=model)
        for half in range(2):
            guard.check()
            chunk_id = names[index+half]
            record = dataset.chunks[chunk_id]
            base_path = dataset.root/f'chunk_{chunk_id}.npz'
            base = read_array_archive(base_path, record['archive'])
            initial = jax.device_get(values(full['native'], full['goal']))
            state_rows, action_rows, traces, valid_rows = [initial], [], [], []
            started = time.monotonic()
            for tick in range(record['steps']):
                full, state, action, trace, valid = step(full, jnp.asarray(base['action__raw_normalized'][tick]))
                state, action, trace, valid = jax.device_get((state, action, trace, valid))
                if not np.all(valid):
                    raise FloatingPointError('Nonfinite replay; prior committed overlays remain protected')
                state_rows.append(state)
                action_rows.append(action)
                traces.append(trace)
                valid_rows.append(valid)
                if tick % 32 == 0:
                    guard.check()
                    print(f'[REPLAY] {chunk_id} tick={tick}/64', flush=True)
            states, actions, substeps = map(_stack_cpu, (state_rows, action_rows, traces))
            bounds = label_task_bounds(cfg, states,
                reference_rotation=np.asarray(env.fixed_contact_racket_reference_rotation),
                joint_lower_rad=np.asarray(env.arm_lo), joint_upper_rad=np.asarray(env.arm_hi))
            for key, value in bounds.items():
                if value.shape[:2] == (record['steps']+1, n):
                    states[f'task_{key}'] = value
            canonical = dict(valid=np.stack(valid_rows))
            for prefix, rows in (('state', states), ('action', actions), ('substep', substeps)):
                canonical.update({f'{prefix}__{key}': value for key, value in rows.items()})
            if not set(base).issubset(canonical):
                raise ValueError('Replay omitted an original trajectory stream')
            arrays, layout = encode_overlay(base, canonical)
            restored = decode_overlay(base, arrays, layout)
            if any(restored[key].tobytes() != value.tobytes() for key, value in canonical.items()):
                raise ValueError('Replay storage codec failed exact reconstruction')
            del restored
            archive = write_array_archive_unique(args.output_dir/f'chunk_{chunk_id}.npz', arrays, guard.budget)
            differences = {key: float(np.max(np.abs(canonical[key]-base[key]))) for key in (
                'substep__q_after', 'substep__dq_after', 'substep__ball_qpos_after', 'substep__ball_qvel_after')}
            changes = {key: int(np.count_nonzero(canonical[key] != base[key])) for key in (
                'substep__event_contact_entry', 'substep__event_contact_release')}
            metadata = dict(schema='wmj_support_replay_overlay_v1', chunk_id=chunk_id,
                base_archive_path=str(base_path.absolute()), base_archive=record['archive'],
                base_record_sha256=file_hash(base_path.with_suffix('.json')),
                archive=archive, archive_file=f'chunk_{chunk_id}.npz', layout=layout,
                record=record, first_seed_ref=first['seed_ref'],
                replay_controls_before_chunk=half*64,
                source_dataset_sha256=report['source_dataset_sha256'],
                canonical_labels='all_streams_from_one_continuous_native_replay',
                maximum_original_difference=differences, changed_event_bits=changes)
            digest = write_json_unique(args.output_dir/f'chunk_{chunk_id}.json', metadata, guard.budget)
            report['chunks'].append(dict(chunk_id=chunk_id, sha256=digest,
                stored_bytes=archive['stored_bytes'], wall_s=time.monotonic()-started,
                maximum_original_difference=differences, changed_event_bits=changes))
            print(f'[SAVE] {chunk_id} bytes={archive["stored_bytes"]}; differences={differences}', flush=True)
            write_json_unique(args.output_dir/f'progress_{chunk_id}.json', dict(chunks=report['chunks']), guard.budget)
            del arrays, canonical, base, state_rows, action_rows, traces, states, actions, substeps
    report['status'] = 'DIAGNOSTIC_COMPLETE' if args.diagnostic_trials else 'REPLAY_COMPLETE'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--diagnostic-trials', type=int, choices=(1,))
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    args = parser.parse_args(argv)
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_support_replay_v1', training_version='WMJ-V1', status='INCOMPLETE',
            config=cfg, config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={p.name: file_hash(p) for p in paths}, chunks=[],
            actor_updates=0, critic_updates=0, physics_modified=False,
            arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            {**report, 'source_snapshot': {p.name: p.read_text() for p in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device is required')
            replay(cfg, args, guard, report)
        except InterruptedError as exc:
            report.update(status='INTERRUPTED', error=str(exc))
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
