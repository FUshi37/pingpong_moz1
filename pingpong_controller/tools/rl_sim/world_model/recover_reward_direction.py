"""Complete only the authorized sampled probes with lossless bounded storage."""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import jax
import numpy as np

from . import audit_reward_direction, reward_direction_rollout
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .native_entry_collision import write_compact_json
from .runtime_guard import RunGuard
from .train_pose_trial import phase_budget


def write_gzip_json(path: Path, payload: dict, budget) -> str:
    """Validate before publishing, reserving both temporary and final file bytes."""
    path = Path(path).absolute()
    if not path.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Compressed output must remain inside the budget root')
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    raw = (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                      separators=(',', ':')) + '\n').encode()
    data = gzip.compress(raw, mtime=0)
    budget.check(additional_bytes=2 * len(data) + 8192)
    descriptor, temporary = tempfile.mkstemp(prefix='.wmj-lossless-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        recovered = gzip.decompress(Path(temporary).read_bytes())
        if recovered != raw:
            raise ValueError('Lossless archive round trip failed')
        json.loads(recovered)
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return hashlib.sha256(data).hexdigest()
    finally:
        Path(temporary).unlink(missing_ok=True)


def sampled_collector():
    """Move the identical observer check to the first sampled control only."""
    original = reward_direction_rollout.collect_branches
    tree = ast.parse(inspect.getsource(original))
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.Constant) and node.value == 'deterministic']
    if len(matches) != 1:
        raise ValueError('Collector pairing anchor changed')
    matches[0].value = 'sampled'
    scope = dict(original.__globals__)
    exec(compile(ast.fix_missing_locations(tree), '<WMJ-sampled-pairing>', 'exec'), scope)
    return scope[original.__name__]


def sampled_audit():
    """Keep the original gradient and probes; change storage and row accounting."""
    original = audit_reward_direction.audit
    tree = ast.parse(inspect.getsource(original))
    extensions = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Constant) and n.value == '.json']
    counts = [n for n in ast.walk(tree)
              if isinstance(n, ast.Constant) and n.value == 49152]
    if len(extensions) != 1 or len(counts) != 2:
        raise ValueError('Audit storage or control-count anchor changed')
    extensions[0].value = '.json.gz'
    for node in counts:
        node.value = 24576
    scope = dict(original.__globals__, collect_branches=sampled_collector(),
                 write_compact_json=write_gzip_json)
    exec(compile(ast.fix_missing_locations(tree), '<WMJ-sampled-lossless>', 'exec'), scope)
    return scope[original.__name__]


def gradient_recomputation_report(previous: dict, current: dict) -> dict:
    """Record numerical re-evaluation; this does not certify vector equality.

    The first attempt did not save its full direction vector. Norm and scalar
    objective agreement cannot establish vector agreement, and a tiny scalar
    mismatch alone cannot establish a changed experiment. Source, frozen-state
    and noise identities are checked separately; native outcome gates stay intact.
    """
    old = np.asarray(previous['per_source_objectives'], float)
    new = np.asarray(current['per_source_objectives'], float)
    if old.shape != new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError('Invalid recomputed objective layout or values')
    fields = ('norm', 'objective')
    if any(not np.isfinite(row[key]) for row in (previous, current) for key in fields):
        raise ValueError('Nonfinite recomputed objective or gradient norm')
    if min(previous['norm'], current['norm']) <= 0.:
        raise ValueError('Zero or negative direction norm')
    delta = np.abs(new - old)
    return dict(status='NUMERIC_RECOMPUTATION_RECORDED_NOT_VECTOR_EQUIVALENCE',
                direction_hash_equal=current['normalized_direction_sha256'] == previous['normalized_direction_sha256'],
                norm_absolute_difference=abs(current['norm'] - previous['norm']),
                norm_relative_difference=abs(current['norm'] / previous['norm'] - 1.),
                objective_absolute_difference=abs(current['objective'] - previous['objective']),
                objective_relative_difference=abs(current['objective'] - previous['objective']) /
                    max(abs(previous['objective']), 1e-12),
                source_objective_max_abs=float(delta.max()),
                source_objective_p95_abs=float(np.quantile(delta, .95)),
                counts={key: dict(previous=previous[key], current=current[key]) for key in
                        ('valid_controls', 'release_count', 'physical_failures')},
                full_vector_comparison_available=False,
                scope='Same algorithm and inputs, independently recomputed direction. No numeric outcome gate relaxed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config()
    setting = cfg['reward_direction_audit']['sampled_recovery']
    recovery = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    plan = json.loads(authenticate(repository_root(), recovery['original_plan']).read_text())
    previous = json.loads(authenticate(repository_root(), recovery['partial_result']).read_text())
    authenticate(repository_root(), recovery['retained_deterministic'])
    if (not setting['authorized'] or cfg['reward_direction_audit']['default_enabled']
            or recovery['modes'] != ['sampled'] or plan['max_phase_bytes'] != 2 * 2**20
            or any(plan[k] for k in ('actor_updates', 'critic_updates', 'world_updates',
                                    'default_replacement'))):
        raise ValueError('Only the approved sampled recovery may run')
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key + '_sha256']:
            raise ValueError('Preserved contract changed')
    if (config_hash(cfg['imagination_implementation']['reward']['coefficients'])
            != plan['default_reward_sha256']):
        raise ValueError('Default reward changed')
    for name, digest in previous['source_sha256'].items():
        if file_hash(Path(__file__).parent / name) != digest:
            raise ValueError('Previous implementation changed: ' + name)
    # Preserve the original phase start and quota, including the incomplete run.
    plan = dict(plan, policy_modes={'sampled': plan['policy_modes']['sampled']})
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, recovery['max_worker_wall_s'],
                  startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan)
        guard.budget.check(additional_bytes=recovery['peak_write_reservation_bytes'])
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_v1_reward_direction_sampled_recovery', status='INCOMPLETE',
                      training_version='WMJ-V1', recovery_plan=setting['plan'],
                      original_plan=recovery['original_plan'], gpu_uuid=args.gpu_uuid,
                      config_sha256=config_hash(cfg), source_commit=plan['source_commit'],
                      source_sha256={f.name: file_hash(f) for f in
                                     sorted(Path(__file__).parent.glob('*.py'))},
                      retained_deterministic=recovery['retained_deterministic'])
        write_gzip_json(args.output_dir / 'config_snapshot.json.gz', cfg, guard.budget)
        write_compact_json(args.output_dir / 'implementation.json',
                           dict(source=Path(__file__).read_text(),
                                previous_implementation=recovery['previous_implementation']),
                           guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('One CUDA GPU required')
            sampled_audit()(cfg, args, guard, report, plan)
            report['recovery_identity'] = {
                name: report[name] == previous[name] for name in
                ('parameters_before_sha256', 'common_initial_course_sha256',
                 'exploration_noise_sha256', 'initial_native_sha256')}
            if not all(report['recovery_identity'].values()):
                raise ValueError('Sampled recovery source/model/noise identity changed')
            old = previous['gradient_directions']['sampled']
            new = report['gradient_directions']['sampled']
            report['recovery_gradient'] = gradient_recomputation_report(old, new)
            details = report['comparisons']['sampled']['details']
            raw = gzip.decompress(Path(details['path']).read_bytes())
            details.update(encoding='gzip', uncompressed_bytes=len(raw),
                           uncompressed_sha256=hashlib.sha256(raw).hexdigest())
            report['status'] = 'FROZEN_SAMPLED_RECOVERY_COMPLETE'
        except Exception as exc:
            report.update(status='INCOMPLETE', error=repr(exc))
            raise
        finally:
            report['resources'] = guard.report()
            write_compact_json(args.output_dir / 'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
