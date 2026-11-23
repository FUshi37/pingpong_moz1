"""Read-only replay labels for the authorized independent 47->6 diagnostic."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive, write_array_archive_unique, write_json_unique
from .contracts import config_hash, file_hash, load_config
from .fit_robot import dataset_identity, summarize_errors
from .replay_support import _restore
from .runtime_guard import RunGuard
from .solver_auxiliary import execution_features, normal_projection_coefficients
from .teacher_trace import contact_pair_masks, point_twist
from .trajectory_data import TrajectoryDataset


def solver_labels(env, current):
    """Observe native J*qacc at the preintegration geometry and solver time."""
    support = env._wmj_support_trace
    ancestors = support['ancestor_ids']
    origin = current.geom_xpos[:, env.racket_contact_geom_id]
    linear, angular = point_twist(origin, current.xaxis[:, ancestors],
        current.xanchor[:, ancestors], current.qacc[:, support['ancestor_dof']],
        support['hinge_mask'])
    terms = jnp.concatenate((linear, angular), axis=-1)
    pair, other = contact_pair_masks(current.contact.geom, current.contact.dist,
        ball_geom_id=env.ball_geom_id, racket_geom_id=env.racket_contact_geom_id)
    slot = jnp.argmin(jnp.where(pair, current.contact.dist, 1e6), axis=-1)
    lane = jnp.arange(pair.shape[0])
    address = jnp.asarray(current.contact.efc_address)[slot]
    jacobian = current.efc_J[lane, jnp.maximum(address, 0)]
    normal = jacobian[:, env.ball_vadr:env.ball_vadr+3]
    coefficients = normal_projection_coefficients(normal, current.contact.pos[lane, slot]-origin)
    ball_term = jnp.sum(normal*current.qacc[:, env.ball_vadr:env.ball_vadr+3], axis=-1)
    robot_term = jnp.sum(jacobian*current.qacc, axis=-1)-ball_term
    distance = current.contact.dist[lane, slot]
    overlap = jnp.any(pair, axis=-1) & (distance <= 0.) & (address >= 0)
    return dict(target=terms, normal_coefficients=coefficients,
        robot_normal_term=robot_term,
        closure_error=jnp.sum(coefficients*terms, axis=-1)-robot_term,
        aref=current.efc_aref[lane, jnp.maximum(address, 0)],
        regularizer=1/jnp.maximum(current.efc_D[lane, jnp.maximum(address, 0)], 1e-20),
        gravity=normal[:, 2]*-9.81,
        force_n=current.efc_force[lane, jnp.maximum(address, 0)],
        distance_m=distance, overlap=overlap, other=jnp.any(other, axis=-1),
        active_pair_count=jnp.sum(pair & (current.contact.dist <= 0.), axis=-1))


def collect(cfg, args, guard, report):
    from .native_stack import build_native_environment
    from .native_observation import NativeObservationAdapter
    from .physical_events import init_physical_events
    from . import teacher_trace
    from mjx_topology_execution import make_collect

    source = json.loads((args.collection/'result.json').read_text())
    dataset = TrajectoryDataset.open(args.collection/'dataset', guard.budget)
    if (dataset.identity['control_sha256'] != config_hash(cfg['control'])
            or dataset.identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Collection contract mismatch')
    names = list(dataset.chunks)
    if len(names) % 2 or not 1 <= args.trials <= len(names)//2:
        raise ValueError('Require complete, bounded two-chunk acquisitions')
    n = dataset.chunks[names[0]]['lanes']
    if n != 1024:
        raise ValueError('Native parallelism must remain 1024')
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
            return dict(auxiliary={**solver_labels(env, current[0]),
                                    'features': execution_features(trace)})
        return original_loop(substep, initial, count, observe)

    teacher_trace.recorded_loop = observed_loop
    observer = teacher_trace.instrument_environment(env, bundle.stack, support_trace=True)

    def transition(state, action):
        after, _, _, _, metrics = env.step(state, action)
        return after, metrics['wmj_trace']['auxiliary']

    scoped, execution = make_collect(transition, mode='project')
    step = jax.jit(scoped)
    rng = np.random.default_rng(args.seed)
    report.update(source_collection=str(args.collection), source_dataset_sha256=dataset_identity(dataset),
        source_result_sha256=file_hash(args.collection/'result.json'), context_sha256=context,
        observer=observer, execution=execution, n_envs=n, records=[],
        label_definition='world geom-origin J*qacc linear3/angular3; includes base; excludes Jdot*dq',
        input_definition='existing47 execution features at preintegration time, current delayed commands',
        sampling='all valid overlaps; independent2percent of valid nonoverlaps; test excluded',
        physical_modifications=False, source_groups_preserved=True)
    closures = []
    try:
        for trial in range(args.trials):
            guard.check()
            first, second = (dataset.chunks[names[trial*2+i]] for i in (0, 1))
            if first['groups'] != second['groups'] or first['context_sha256'] != second['context_sha256']:
                raise ValueError('Cannot join unrelated acquisition halves')
            full = _restore(first['seed_ref'], template, context)
            native = full['native']._replace(model=model)
            valid = np.asarray(full['ball_model_valid']).copy()
            split = np.array([{'train': 0, 'validation': 1, 'test': 2}[g['split']]
                              for g in first['groups']], np.int8)
            regions = np.array([g['region_id'] for g in first['groups']], np.int32)
            parts, counts = {}, dict(eligible=0, test_excluded=0, invalid=0, retained=0)
            started = time.monotonic()
            for half in range(2):
                name = names[trial*2+half]
                record = dataset.chunks[name]
                actions = read_array_archive(dataset.root/f'chunk_{name}.npz', record['archive'],
                    names=['action__raw_normalized'])['action__raw_normalized']
                for tick in range(record['steps']):
                    native, rows = step(native, jnp.asarray(actions[tick]))
                    rows = jax.device_get(rows)
                    if not all(np.isfinite(v).all() for v in rows.values()):
                        raise FloatingPointError('Nonfinite replay label; do not silently discard')
                    allowed = valid[None] & (np.cumsum(rows.pop('other'), axis=0) == 0)
                    valid = allowed[-1]
                    overlap = rows['overlap']
                    closure_mask = allowed & overlap & (split[None] != 2)
                    closures.append(rows['closure_error'][closure_mask])
                    if np.any(rows['active_pair_count'][closure_mask] > 1):
                        raise ValueError('Multiple simultaneous ball/racket rows require explicit aggregation')
                    select = allowed & (split[None] != 2) & (overlap | (rng.random((5, n)) < .02))
                    substep_id, lane = np.nonzero(select)
                    for key, value in rows.items():
                        parts.setdefault(key, []).append(value[select])
                    extras = dict(split=split[lane], region_id=regions[lane],
                        family_id=(trial*n+lane).astype(np.int32),
                        substep_id=((half*64+tick)*5+substep_id).astype(np.int32))
                    for key, value in extras.items():
                        parts.setdefault(key, []).append(value)
                    counts['eligible'] += int(np.sum(allowed & (split[None] != 2)))
                    counts['test_excluded'] += int(np.sum(np.broadcast_to(split[None] == 2, (5, n))))
                    counts['invalid'] += int(np.sum(~allowed & (split[None] != 2)))
                    counts['retained'] += len(lane)
                    if tick % 16 == 0:
                        guard.check()
                        print(f'[LABEL] trial={trial+1}/{args.trials} {name} tick={tick}/64 rows={counts["retained"]}', flush=True)
            arrays = {k: np.concatenate(v) for k, v in parts.items()}
            closure = summarize_errors([np.abs(arrays['closure_error'][arrays['overlap']])])
            # This is the Lipschitz upper bound through the scalar soft-force
            # equation, including its max(force,0). Judge numerical closure in
            # ball-velocity units, not an arbitrary acceleration threshold.
            closure_dv = np.abs(arrays['closure_error'])*.001/.004/(250.+arrays['regularizer'])
            closure_velocity = summarize_errors([closure_dv[arrays['overlap']]])
            filename = f'trial_{trial:02d}.npz'
            archive = write_array_archive_unique(args.output_dir/filename, arrays, guard.budget)
            record = dict(archive_file=filename, archive=archive, trial=trial,
                source_chunks=names[trial*2:trial*2+2], seed_reference=first['seed_ref'],
                groups=first['groups'], counts=counts, closure=closure,
                closure_velocity_bound_m_s=closure_velocity,
                train_rows=int(np.sum(arrays['split'] == 0)),
                validation_rows=int(np.sum(arrays['split'] == 1)),
                wall_s=time.monotonic()-started)
            report['records'].append(record)
            write_json_unique(args.output_dir/f'trial_{trial:02d}.json', record, guard.budget)
            if (closure['nonfinite'] or closure_velocity['p95'] > .001
                    or closure_velocity['maximum'] > .005):
                raise ValueError(f'Label reconstruction exceeds numerical velocity budget: {closure_velocity}')
            print(f'[LABEL_SAVE] trial={trial} rows={counts["retained"]} closure_max={closure["maximum"]:.6g}', flush=True)
    finally:
        teacher_trace.recorded_loop = original_loop
    report['native_row_closure_m_s2'] = summarize_errors([np.abs(np.concatenate(closures))])
    report['label_closure_velocity_gate'] = dict(p95_max_m_s=.001, maximum_m_s=.005,
                                                all_trials_passed=True)
    report['status'] = 'LABELS_COMPLETE_INDEPENDENT_DIAGNOSTIC_ONLY'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--collection', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2026110908)
    parser.add_argument('--max-wall-s', type=float, default=1800.)
    args = parser.parse_args()
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_solver_auxiliary_labels_v1', training_version='WMJ-V1',
            status='INCOMPLETE', timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            config_sha256=config_hash(cfg), physics_sha256=config_hash(cfg['physics']),
            control_sha256=config_hash(cfg['control']), gpu_uuid=args.gpu_uuid,
            source_sha256={p.name: file_hash(p) for p in paths},
            actor_updates=0, critic_updates=0, world_model_ready=False,
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report, sources={p.name: p.read_text() for p in paths}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            report['jax_devices'] = [str(d) for d in jax.devices()]
            collect(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
