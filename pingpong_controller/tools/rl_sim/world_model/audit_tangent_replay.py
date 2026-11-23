"""Same-replay held-out normal/full prior diagnostic; no cross-replay labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_json_unique
from .audit_tangent_prior import prior_errors
from .collect_solver_auxiliary import solver_labels
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .model_numerics import precision_scope
from .normal_checkpoint import load_components
from .replay_support import _restore
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader
from .trajectory_data import TrajectoryDataset


def diagnostic_raw(trace: dict) -> dict:
    """Build before/after contact arrays from one native solver interval."""
    raw = dict(trace)
    for key in ('q_before', 'dq_before'):
        raw[key] = jnp.concatenate((trace[key], trace['support_'+key]), axis=-1)
    shape = trace['contact_overlap'].shape
    for key in ('event_contact_entry', 'event_contact_release', 'event_controlled_release'):
        raw[key] = jnp.zeros(shape, bool)
    raw['event_contact_duration_steps'] = jnp.zeros(shape, jnp.int32)
    raw['ball_model_valid'] = jnp.ones(shape, bool)
    # Event placeholders are unused by the one-step prior equation; they do
    # not create contact-event labels or qualify multi-step prediction.
    return raw


def audit(cfg, args, guard, report):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .physical_events import init_physical_events
    from . import teacher_trace
    from mjx_topology_execution import make_collect

    root = repository_root()/cfg['resources']['output_root']
    reader = SupportReplayReader(root/'support_replay_full_a01', guard.budget, max_bytes=0)
    _, options, refs, auxiliary, aux_ref = load_components(cfg, reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    collection = Path(reader.report['source_collection'])
    source = json.loads((collection/'result.json').read_text())
    dataset = TrajectoryDataset.open(collection/'dataset', guard.budget)
    if (dataset.identity['control_sha256'] != config_hash(cfg['control'])
            or dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Teacher contract mismatch')
    names, n = list(dataset.chunks), 1024
    bundle = build_native_environment(cfg, n_envs=n)
    env = bundle.environment
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(source['seed']), n))
    jax.block_until_ready(fresh)
    context = source['context']['sha256']
    model = _restore(source['context']['reference'], fresh.model, context)
    history, _ = NativeObservationAdapter(env, fresh).initialize(fresh)
    template = dict(native=fresh._replace(model=None), history=history,
        events=init_physical_events(n), ball_model_valid=jnp.ones(n, bool),
        goal=jnp.zeros((n, 3), jnp.float32))
    original_loop = teacher_trace.recorded_loop

    def observed_loop(substep, initial, count, sample):
        def observe(index, previous, current):
            trace = sample(index, previous, current)
            labels = solver_labels(env, current[0])
            with precision_scope('highest'):
                errors = prior_errors(options, auxiliary, diagnostic_raw(trace), labels['target'])
            return dict(diagnostic=dict(metrics=errors['metrics'], near_rim=errors['near_rim'],
                overlap=labels['overlap'], other=labels['other'], pairs=labels['active_pair_count']))
        return original_loop(substep, initial, count, observe)

    teacher_trace.recorded_loop = observed_loop
    observer = teacher_trace.instrument_environment(env, bundle.stack, support_trace=True)

    def transition(state, action):
        after, _, _, _, metrics = env.step(state, action)
        return after, metrics['wmj_trace']['diagnostic']

    scoped, execution = make_collect(transition, mode='project')
    step = jax.jit(scoped)
    report.update(references=refs, auxiliary_reference=aux_ref, observer=observer, execution=execution,
        source_collection=str(collection), source_result_sha256=file_hash(collection/'result.json'),
        source_dataset_identity=dataset.identity, trials=[], native_parallelism=n,
        physical_modifications=False, label_alignment='same_solver_interval_in_same_replay')
    values, regions, rims, identities = {}, [], [], []
    try:
        for trial in range(len(names)//2):
            guard.check()
            first, second = (dataset.chunks[names[trial*2+i]] for i in (0, 1))
            if first['groups'] != second['groups']:
                raise ValueError('Cannot join unrelated acquisitions')
            full = _restore(first['seed_ref'], template, context)
            native = full['native']._replace(model=model)
            valid = np.asarray(full['ball_model_valid']).copy()
            selected_lanes = np.array([g['split'] == 'validation' for g in first['groups']])
            region = np.array([g['region_id'] for g in first['groups']], np.int32)
            counts = dict(retained=0, invalid_validation=0, eligible_validation=0)
            started = time.monotonic()
            for half in range(2):
                record = dataset.chunks[names[trial*2+half]]
                actions = read_array_archive(dataset.root/f'chunk_{names[trial*2+half]}.npz',
                    record['archive'], names=['action__raw_normalized'])['action__raw_normalized']
                for tick in range(record['steps']):
                    native, result = step(native, jnp.asarray(actions[tick]))
                    result = jax.device_get(result)
                    allowed = valid[None] & (np.cumsum(result['other'], axis=0) == 0)
                    valid = allowed[-1]
                    selected = allowed & selected_lanes[None] & result['overlap']
                    if np.any(result['pairs'][selected] > 1):
                        raise ValueError('Multiple contacts need an explicit equation definition')
                    substeps, lanes = np.nonzero(selected)
                    for name, metrics in result['metrics'].items():
                        for key, value in metrics.items():
                            values.setdefault(name, {}).setdefault(key, []).append(value[selected])
                    regions.append(region[lanes])
                    rims.append(result['near_rim'][selected])
                    identities.extend((trial, (half*64+tick)*5+int(s), int(l)) for s,l in zip(substeps,lanes))
                    counts['retained'] += len(lanes)
                    counts['eligible_validation'] += int(np.sum(allowed & selected_lanes[None]))
                    counts['invalid_validation'] += int(np.sum(~allowed & selected_lanes[None]))
                    if tick % 16 == 0:
                        guard.check()
                        print(f'[TANGENT_REPLAY] trial={trial+1}/10 half={half} tick={tick}/64', flush=True)
            row = dict(trial=trial, counts=counts, source_chunks=names[trial*2:trial*2+2],
                wall_s=time.monotonic()-started, seed_reference=first['seed_ref'])
            report['trials'].append(row)
            write_json_unique(args.output_dir/f'trial_{trial:02d}.json', row, guard.budget)
    finally:
        teacher_trace.recorded_loop = original_loop
    region, rim = np.concatenate(regions), np.concatenate(rims)
    values = {n: {k: np.concatenate(v) for k,v in metrics.items()} for n,metrics in values.items()}
    def summarize(mask):
        return {n: {k: summarize_errors([v[mask]]) for k,v in metrics.items()} for n,metrics in values.items()}
    report.update(selected_rows_sha256=config_hash(identities), selected_rows=len(region),
        overall=summarize(np.ones(len(region), bool)),
        by_region={str(int(r)): summarize(region == r) for r in np.unique(region)},
        by_geometry={'near_rim': summarize(rim), 'interior': summarize(~rim)},
        status='SAME_REPLAY_TANGENT_DIAGNOSTIC_COMPLETE_NOT_WORLD_QUALIFIED')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 1800.) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_tangent_same_replay_diagnostic_v1', training_version='WMJ-V1',
            status='INCOMPLETE', config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            source_sha256={p.name:file_hash(p) for p in paths}, actor_updates=0, critic_updates=0,
            world_qualified=False, final_test_used=False, scope='heldout_validation_contacts_same_replay_1ms_prior_diagnostic',
            arguments={k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report, source_snapshot={p.name:p.read_text() for p in paths}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
