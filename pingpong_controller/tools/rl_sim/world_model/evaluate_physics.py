"""Paired held-out native-robot versus own-robot contact prediction.

Commands remain recorded delayed commands. This diagnoses coupled physical
components, not full policy/RMP feedback or completed task performance.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .contact_data import contact_window_rows, take_contact_windows
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .physics_components import load_physics_components
from .physics_data import read_physics_trial
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact import position_contact_rollout
from .position_contact_data import prepare_position_contact_batch
from .runtime_guard import RunGuard
from .support_data import SupportReplayReader
from .world_physics import PhysicsState, physics_rollout
from .model_numerics import precision_scope


def _summaries(entries: dict) -> dict:
    overall, regional, buckets = {}, {}, {}
    for metric, pieces in entries.items():
        values = np.concatenate(pieces['values'])
        regions = np.concatenate(pieces['regions'])
        conditions = np.concatenate(pieces['buckets'])
        overall[metric] = summarize_errors([values])
        for region in np.unique(regions):
            regional.setdefault(str(int(region)), {})[metric] = summarize_errors([values[regions == region]])
        for condition in np.unique(conditions):
            buckets.setdefault(str(condition), {})[metric] = summarize_errors([values[conditions == condition]])
    return dict(overall=overall, by_region=regional, by_bucket=buckets)


def evaluate(reader, parameters, options, guard, *, seed: int, progress=None, predictor=None) -> dict:
    """Use identical fixed windows in both modes, retaining every selected lane."""
    rng = np.random.default_rng(seed)
    names = list(reader.records)
    horizons = (1, 8, 32, 64, 128)
    modes = ('recorded_native_robot', 'predicted_robot')
    measures = {h:{mode:{} for mode in modes} for h in horizons}
    events = {h:{mode:{} for mode in modes} for h in horizons}
    invalid = {h:{mode:0 for mode in modes} for h in horizons}
    populations = {h:{} for h in horizons}
    identities = {h:hashlib.sha256() for h in horizons}
    calls = []

    def predict_unscoped(raw):
        if predictor is not None:
            return predictor(raw)
        batch = prepare_position_contact_batch(None, raw)
        ball = jax.tree.map(lambda value:value[0], batch['before'])
        _, native = position_contact_rollout(parameters['contact'], options['surface'], ball,
            batch['racket'], ball_radius_m=options['ball_radius_m'],
            position_scale=options['position_scale'], contract=options['contact_contract'])
        initial = PhysicsState(raw['q_before'][0], raw['dq_before'][0], ball)
        commands = {key:raw['delayed_'+key+'_command'] for key in ('q', 'dq', 'ddq')}
        _, own = physics_rollout(parameters['robot'], parameters['contact'], initial, commands, **options)
        output = {}
        for mode, contact in (('recorded_native_robot', native), ('predicted_robot', own['contact'])):
            metrics, edge_counts = ball_metrics(contact, batch)
            finite = jnp.ones(raw['q_before'].shape[:2], bool)
            for value in contact['after'][:3]:
                finite = finite & jnp.all(jnp.isfinite(value), axis=-1)
            if mode == 'predicted_robot':
                metrics.update(robot_metrics(own, raw, batch))
                finite = finite & jnp.all(jnp.isfinite(own['q_after_rad']), axis=-1)
                finite = finite & jnp.all(jnp.isfinite(own['dq_after_rad_s']), axis=-1)
            output[mode] = dict(metrics=metrics, events=edge_counts,
                                nonfinite_windows=jnp.sum(~jnp.all(finite, axis=0)))
        return output

    @jax.jit
    def predict(raw):
        with precision_scope(options.get('matmul_precision')):
            return predict_unscoped(raw)

    started = time.monotonic()
    for trial in range(0, len(names), 2):
        guard.check()
        data, record = read_physics_trial(reader, names[trial], names[trial+1])
        for horizon in horizons:
            guard.check()
            rows = contact_window_rows(data, record, controls=horizon, split='validation')
            independent = {}
            for index, row in enumerate(rows):
                independent.setdefault(row['independent_event_id'], []).append(index)
            groups = list(independent.values())
            if not groups:
                raise ValueError(f'No validation windows at horizon{horizon}')
            selected = [int(rng.choice(groups[int(i)])) for i in rng.integers(len(groups), size=128)]
            raw, refs = take_contact_windows(data, rows, selected, controls=horizon)
            identities[horizon].update(json.dumps(refs, sort_keys=True).encode())
            region = np.asarray([row['region_id'] for row in refs])
            bucket = np.asarray([row['bucket'] for row in refs])
            for name in np.unique(bucket):
                populations[horizon][str(name)] = populations[horizon].get(str(name), 0)+int(np.sum(bucket == name))
            device_raw = jax.tree.map(jnp.asarray, raw)
            jax.block_until_ready(device_raw)
            tick = time.monotonic()
            result = jax.device_get(predict(device_raw))
            calls.append(dict(trial=trial//2, controls=horizon,
                compile_included=trial == 0, paired_predict_and_transfer_s=time.monotonic()-tick))
            for mode in modes:
                invalid[horizon][mode] += int(result[mode]['nonfinite_windows'])
                for edge, count in result[mode]['events'].items():
                    events[horizon][mode][edge] = events[horizon][mode].get(edge, 0)+int(count)
                for metric, (values, mask) in result[mode]['metrics'].items():
                    entry = measures[horizon][mode].setdefault(metric, dict(values=[], regions=[], buckets=[]))
                    entry['values'].append(values[mask])
                    entry['regions'].append(np.broadcast_to(region, values.shape)[mask])
                    entry['buckets'].append(np.broadcast_to(bucket, values.shape)[mask])
            del raw, device_raw, result
        del data
        if progress is not None:
            progress(trial//2+1)
        print(f'[VALIDATE] common-command trial{trial//2+1}/{len(names)//2}', flush=True)
    result = {}
    for horizon in horizons:
        result[str(horizon)] = dict(window_selection_sha256=identities[horizon].hexdigest(),
            sampled_windows_by_bucket=populations[horizon],
            modes={mode:{**_summaries(measures[horizon][mode]),
                'event_counts':events[horizon][mode], 'nonfinite_windows':invalid[horizon][mode]}
                for mode in modes})
    return dict(horizons=result, calls=calls, wall_s=time.monotonic()-started,
        source='validation_only', seed=seed, windows_per_trial_horizon=128,
        edge_comparison='exact_1ms_labels; timing shifts count as missed_and_extra',
        apex_label='analytic_no_drag_apex_from_observed_controlled_release',
        precontact_label='native_before_states0..25ms_before_entry_inside_window',
        nonfinite_handling='retained_in_counts; quantiles_describe_finite_values_only')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--support-replay', type=Path, required=True)
    parser.add_argument('--robot-checkpoint', type=Path, required=True)
    parser.add_argument('--contact-checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--seed', type=int, default=2026120607)
    parser.add_argument('--max-wall-s', type=float, default=900.)
    parser.add_argument('--diagnostic-matmul-precision', choices=('highest',),
        help='Explicit evaluation-only precision probe; never rewrites checkpoint semantics')
    args = parser.parse_args(argv)
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_coupled_physics_evaluation_v1', training_version='WMJ-V1',
            status='INCOMPLETE', config=cfg, config_sha256=config_hash(cfg),
            gpu_uuid=args.gpu_uuid, absolute_output_dir=str(args.output_dir.absolute()),
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository_root(), text=True).strip(),
            source_sha256={path.name:file_hash(path) for path in paths},
            arguments={key:str(value) if isinstance(value, Path) else value for key,value in vars(args).items()},
            invocation=sys.argv, completed_trials=0, actor_updates=0, critic_updates=0,
            scope='recorded_delayed_commands_own_joint_and_ball_feedback', world_model_ready=False)
        write_json_unique(args.output_dir/'manifest.json',
            {**report, 'source_snapshot':{path.name:path.read_text() for path in paths}}, guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA GPU is required')
            report['devices'] = [str(device) for device in jax.devices()]
            reader = SupportReplayReader(args.support_replay, guard.budget, max_bytes=0)
            parameters, options, references = load_physics_components(cfg, reader,
                robot_checkpoint=args.robot_checkpoint, contact_checkpoint=args.contact_checkpoint)
            report.update(components=references, dataset_sha256=reader.identity)
            report['inference_precision_override'] = args.diagnostic_matmul_precision
            precision = (nullcontext() if args.diagnostic_matmul_precision is None
                         else jax.default_matmul_precision(args.diagnostic_matmul_precision))
            with precision:
                result = evaluate(reader, parameters, options, guard, seed=args.seed,
                    progress=lambda count:report.update(completed_trials=count))
            write_json_unique(args.output_dir/'validation.json', result, guard.budget)
            report.update(validation=result, status='COUPLED_PHYSICS_EVALUATED_NOT_WORLD_QUALIFIED')
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
