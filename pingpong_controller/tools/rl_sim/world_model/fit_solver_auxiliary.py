"""Bounded supervised diagnostic; never installs a component in the world."""

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
from .fit_robot import summarize_errors
from .neural import adam_update, init_adam, init_mlp
from .runtime_guard import RunGuard
from .solver_auxiliary import WIDTHS, normal_velocity_error, predict, scale_from_statistics, supervised_loss
from .state_adapter import pack_numeric_state
from .trajectory_data import TrainNormalizer


def load_labels(path, cfg, guard):
    report = json.loads((path/'result.json').read_text())
    if (report.get('status') != 'LABELS_COMPLETE_INDEPENDENT_DIAGNOSTIC_ONLY'
            or report['physics_sha256'] != config_hash(cfg['physics'])
            or report['control_sha256'] != config_hash(cfg['control'])):
        raise ValueError('Require complete, consistent, closed native solver labels')
    parts = {}
    families = {'train': set(), 'validation': set()}
    for record in report['records']:
        guard.check()
        arrays = read_array_archive(path/record['archive_file'], record['archive'])
        projected = np.sum(arrays['target']*arrays['normal_coefficients'], axis=-1)
        closure = np.abs((projected-arrays['robot_normal_term'])[arrays['overlap']])*.25/(
            250.+arrays['regularizer'][arrays['overlap']])
        if (not len(closure) or not np.isfinite(closure).all()
                or np.quantile(closure, .95) > .001 or np.max(closure) > .005):
            raise ValueError('Native label reconstruction exceeds velocity-error budget')
        if not np.isin(arrays['split'], [0, 1]).all():
            raise ValueError('Final-test rows must remain sealed')
        for split in families:
            families[split].update(g['root_seed_family'] for g in record['groups'] if g['split'] == split)
        for key, value in arrays.items():
            parts.setdefault(key, []).append(value)
    if families['train'] & families['validation']:
        raise ValueError('Family leakage between training and validation')
    rows = {key: np.concatenate(value) for key, value in parts.items()}
    if rows['features'].shape[1:] != (47,) or rows['target'].shape[1:] != (6,):
        raise ValueError('Expected exactly47 input and6 output labels')
    return rows, report


def family_pool(rows, mask):
    """Group valid row indexes, enabling family-uniform sampling in each bin."""
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        raise ValueError('Both contact and noncontact training bins are required')
    order = np.argsort(rows['family_id'][indexes], kind='stable')
    indexes = indexes[order]
    _, starts, counts = np.unique(rows['family_id'][indexes], return_index=True, return_counts=True)
    return indexes, starts, counts


def sample_pool(rng, pool, size):
    indexes, starts, counts = pool
    groups = rng.integers(len(starts), size=size)
    offsets = (rng.random(size)*counts[groups]).astype(np.int64)
    return indexes[starts[groups]+offsets]


def summarize_prediction(prediction, rows):
    """Retain both six-term errors and teacher-conditioned scalar diagnostics."""
    error = prediction-rows['target']
    finite = np.isfinite(error).all(axis=-1)
    result = dict(count=len(error), nonfinite=int(np.sum(~finite)))
    for name, block in (('linear_m_s2', slice(0, 3)), ('angular_rad_s2', slice(3, 6))):
        values = error[:, block]
        result[name] = summarize_errors([np.linalg.norm(values, axis=-1)])
        result[name]['rmse_components'] = float(np.sqrt(np.mean(values[finite]**2))) if finite.any() else None
    contact = rows['overlap']
    normal = np.sum(prediction*rows['normal_coefficients'], axis=-1)-rows['robot_normal_term']
    result['contact_normal_term_m_s2'] = summarize_errors([np.abs(normal[contact])])
    velocity = normal_velocity_error(prediction, rows)
    result['positive_force_scalar_velocity_error_m_s'] = summarize_errors([
        np.abs(velocity[contact & (rows['force_n'] > 1e-5)])])
    result['overlap_scalar_velocity_error_m_s'] = summarize_errors([np.abs(velocity[contact])])
    return result


def evaluate(parameters, scale, rows, indexes):
    selected = {k: v[indexes] for k, v in rows.items()}
    compiled = jax.jit(predict)
    chunks = []
    for start in range(0, len(indexes), 2048):
        features = selected['features'][start:start+2048]
        padded = np.pad(features, ((0, 2048-len(features)), (0, 0)))
        chunks.append(np.asarray(compiled(parameters, scale, jnp.asarray(padded)))[:len(features)])
    prediction = np.concatenate(chunks)
    return summarize_prediction(prediction, selected), prediction


def fit(cfg, args, guard, report):
    rows, labels = load_labels(args.labels, cfg, guard)
    train, valid = rows['split'] == 0, rows['split'] == 1
    if not train.any() or not valid.any():
        raise ValueError('Need disjoint training and validation populations')
    inputs, outputs = TrainNormalizer(47), TrainNormalizer(6)
    inputs.update(rows['features'], train, split='train')
    outputs.update(rows['target'], train, split='train')
    stats = dict(inputs=inputs.to_dict(), outputs=outputs.to_dict())
    scale = scale_from_statistics(**stats)
    params = init_mlp(jax.random.PRNGKey(args.seed), WIDTHS, zero_output=True)
    optimizer = init_adam(params)
    rng = np.random.default_rng(args.seed)
    pools = [family_pool(rows, train & mask) for mask in (rows['overlap'], ~rows['overlap'])]
    report.update(label_result_sha256=file_hash(args.labels/'result.json'),
        label_source_dataset_sha256=labels['source_dataset_sha256'], normalization=stats,
        split_counts={'train': int(train.sum()), 'validation': int(valid.sum())},
        train_family_count=int(len(np.unique(rows['family_id'][train]))),
        validation_family_count=int(len(np.unique(rows['family_id'][valid]))),
        train_overlap_rows=int(np.sum(train & rows['overlap'])),
        validation_overlap_rows=int(np.sum(valid & rows['overlap'])),
        width=list(WIDTHS), output_definition=labels['label_definition'],
        recipe=dict(updates=args.updates, batch_size=512, learning_rate=3e-4,
            gradient_clip=1., precision='highest', loss='six_normalized_component_MSE',
            sampling='256contact+256noncontact; family_uniform_within_bin',
            normalization_source='train', checkpoint_interval=1024),
        integration_in_world=False, contact_input_dim=39, actor_updates=0, critic_updates=0)
    valid_index = np.flatnonzero(valid)
    validation_rows = {k: v[valid] for k, v in rows.items()}
    baselines = {
        'zero': np.zeros_like(validation_rows['target']),
        'train_mean': np.broadcast_to(np.asarray(scale.residual_mean), validation_rows['target'].shape),
        'native_six_term_oracle': validation_rows['target'],
    }
    # A cheap affine comparator uses only the same47 inputs and training rows.
    x = ((rows['features'][train]-np.asarray(scale.input_mean))/np.asarray(scale.input_scale)).astype(np.float64)
    x = np.column_stack((x, np.ones(len(x))))
    y = ((rows['target'][train]-np.asarray(scale.residual_mean))/np.asarray(scale.residual_scale)).astype(np.float64)
    ridge = np.eye(48)*1e-4*len(x)
    ridge[-1, -1] = 0.
    affine = np.linalg.solve(x.T@x+ridge, x.T@y).astype(np.float32)
    xv = (validation_rows['features']-np.asarray(scale.input_mean))/np.asarray(scale.input_scale)
    baselines['affine47'] = (np.column_stack((xv, np.ones(len(xv))))@affine)*np.asarray(scale.residual_scale)+np.asarray(scale.residual_mean)
    del x, y
    report['baselines_validation'] = {k: summarize_prediction(v, validation_rows) for k, v in baselines.items()}
    write_json_unique(args.output_dir/'prerun.json', report, guard.budget)

    @jax.jit
    def update(parameters, state, features, targets):
        loss, gradient = jax.value_and_grad(supervised_loss)(parameters, scale, features, targets)
        parameters, state, diagnostics = adam_update(parameters, gradient, state,
            learning_rate=3e-4, max_gradient_norm=1.)
        return parameters, state, loss, diagnostics['gradient_norm']

    signature = config_hash(dict(widths=WIDTHS, recipe=report['recipe'], label_hash=report['label_result_sha256'],
                                 source=report['source_sha256']))
    report['validation_history'], report['checkpoints'] = [], []
    started = time.monotonic()
    for step in range(1, args.updates+1):
        if step % 64 == 1:
            guard.check()
        index = np.concatenate([sample_pool(rng, pool, 256) for pool in pools])
        params, optimizer, loss, norm = update(params, optimizer,
            jnp.asarray(rows['features'][index]), jnp.asarray(rows['target'][index]))
        if step % 256 == 0 or step == 1:
            loss_value, norm_value = map(float, (loss, norm))
            if not np.isfinite([loss_value, norm_value]).all():
                raise FloatingPointError('Nonfinite auxiliary fit')
            print(f'[FIT_AUX] u={step}/{args.updates} loss={loss_value:.6g} grad={norm_value:.6g}', flush=True)
        if step % 1024 == 0 or step == args.updates:
            guard.check()
            metrics, prediction = evaluate(params, scale, rows, valid_index)
            report['validation_history'].append(dict(updates=step, metrics=metrics))
            arrays, state = pack_numeric_state(dict(parameters=params, optimizer=optimizer, scale=scale), context_sha256=signature)
            archive_file = f'auxiliary_u{step:06d}.npz'
            archive = write_array_archive_unique(args.output_dir/archive_file, arrays, guard.budget)
            metadata = dict(schema='wmj_solver_auxiliary_checkpoint_v1', training_version='WMJ-V1',
                model_signature=signature, archive_file=archive_file, archive=archive, state=state,
                numpy_rng=rng.bit_generator.state, updates=step, normalization=stats,
                source_sha256=report['source_sha256'], label_result_sha256=report['label_result_sha256'],
                recipe=report['recipe'], width=list(WIDTHS), integration_in_world=False)
            meta_file = f'auxiliary_u{step:06d}.json'
            digest = write_json_unique(args.output_dir/meta_file, metadata, guard.budget)
            report['checkpoints'].append(dict(path=str(args.output_dir/meta_file), sha256=digest))
            print(f'[VALIDATE_AUX] u={step} normal_scalar_v_p95={metrics["positive_force_scalar_velocity_error_m_s"]["p95"]:.6g}', flush=True)
    report['fit_wall_s'] = time.monotonic()-started
    report['validation_final'] = metrics
    report['validation_by_region'] = {}
    for region in np.unique(validation_rows['region_id']):
        mask = validation_rows['region_id'] == region
        report['validation_by_region'][str(int(region))] = summarize_prediction(prediction[mask],
            {k: v[mask] for k, v in validation_rows.items()})
    report['validation_by_contact'] = {}
    for name, mask in (('overlap', validation_rows['overlap']), ('nonoverlap', ~validation_rows['overlap'])):
        report['validation_by_contact'][name] = summarize_prediction(prediction[mask], {k: v[mask] for k, v in validation_rows.items()})
    baseline = report['baselines_validation']
    error = metrics['positive_force_scalar_velocity_error_m_s']['p95']
    screen = dict(finite=metrics['nonfinite'] == 0,
        scalar_velocity_p95_under_008=error <= .08,
        scalar_velocity_p95_half_zero=error <= .5*baseline['zero']['positive_force_scalar_velocity_error_m_s']['p95'])
    for name in ('linear_m_s2', 'angular_rad_s2'):
        screen[name+'_rmse_half_mean'] = metrics[name]['rmse_components'] <= .5*baseline['train_mean'][name]['rmse_components']
    report['independent_screen'] = screen
    report['status'] = 'INDEPENDENT_SCREEN_PASS_NOT_INTEGRATED' if all(screen.values()) else 'INDEPENDENT_SCREEN_FAIL_NOT_INTEGRATED'
    report['world_model_ready'] = False
    # Save paired predictions once, allowing independent metric recomputation.
    archive = write_array_archive_unique(args.output_dir/'validation_predictions.npz',
        dict(prediction=prediction, affine_prediction=baselines['affine47'],
             **validation_rows), guard.budget)
    report['prediction_archive'] = archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--updates', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=2026110909)
    parser.add_argument('--max-wall-s', type=float, default=900.)
    args = parser.parse_args()
    if not 1 <= args.updates <= 20000:
        raise ValueError('Independent fit budget must be1..20000 updates')
    cfg = load_config()
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, args.max_wall_s) as guard:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        paths = [Path(__file__), Path(__file__).with_name('solver_auxiliary.py'), Path(__file__).with_name('neural.py')]
        report = dict(schema='wmj_solver_auxiliary_fit_v1', training_version='WMJ-V1', status='INCOMPLETE',
            timestamp=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), config_sha256=config_hash(cfg),
            gpu_uuid=args.gpu_uuid, source_sha256={p.name: file_hash(p) for p in paths},
            arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report, sources={p.name: p.read_text() for p in paths}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            report['jax_devices'] = [str(d) for d in jax.devices()]
            fit(cfg, args, guard, report)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_json_unique(args.output_dir/'result.json', report, guard.budget)
            print(f'[SAVE] {args.output_dir}; status={report["status"]}', flush=True)


if __name__ == '__main__':
    main()
