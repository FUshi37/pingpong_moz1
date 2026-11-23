"""Preparation-only native pairing diagnostic; no policy step or parameter update."""

import argparse
import inspect
import json
from pathlib import Path
from types import FunctionType, SimpleNamespace

import jax
import numpy as np

from . import evaluate_frozen
from .contracts import authenticate, file_hash, load_config, repository_root
from .free_orientation import install_position_only_reference, phase_plan
from .native_entry_collision import write_compact_json
from .native_stack import build_native_environment
from .paired_precision import load_actor
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget


class Prepared(Exception):
    """Stop before the first cold or rollout policy invocation."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = load_config(); plan = phase_plan(cfg)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 1800., startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan); args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_v1_orientation_initial_pairing', status='INCOMPLETE',
            plan=cfg['orientation_limit_trial']['plan'], actor_updates=0, critic_updates=0,
            world_updates=0, policy_calls=0, source_sha256=file_hash(Path(__file__)), preparations={})
        write_compact_json(args.output_dir/'implementation.json', dict(source=Path(__file__).read_text(),
            arguments={k:str(v) for k,v in vars(args).items()}, plan=report['plan']), guard.budget)
        try:
            if jax.default_backend() != 'gpu':
                raise ValueError('CUDA preparation required')
            actor, _ = load_actor(cfg, plan['actor'])
            stack = None; states = {}
            for mode in ('bounded_2deg', 'free_orientation'):
                def build(*values, **kwargs):
                    nonlocal stack
                    bundle = build_native_environment(*values, **kwargs, stack=stack)
                    stack = bundle.stack
                    if mode == 'free_orientation':
                        install_position_only_reference(bundle.environment, stack)
                    return bundle

                def jit(function, *values, **kwargs):
                    closure = inspect.getclosurevars(function).nonlocals
                    policy = next((v for v in closure.values() if callable(v)
                        and getattr(v, '__name__', '') == 'policy_step'), None)
                    if policy is None:
                        return jax.jit(function, *values, **kwargs)

                    def capture(*inputs, **options):
                        states[mode] = jax.device_get(inputs[:5])
                        report['preparations'][mode] = dict(native_sha256=numeric_tree_digest(inputs[0]),
                            history_task_rng_sha256=numeric_tree_digest(inputs[1:5]))
                        raise Prepared()
                    return capture

                namespace = dict(evaluate_frozen.audit.__globals__, build_native_environment=build,
                    jax=SimpleNamespace(**dict(jax.__dict__, jit=jit)),
                    write_json_unique=lambda *unused: 'captured_without_duplicate_artifacts')
                entry = FunctionType(evaluate_frozen.audit.__code__, namespace, evaluate_frozen.audit.__name__,
                    evaluate_frozen.audit.__defaults__, evaluate_frozen.audit.__closure__)
                entry.__kwdefaults__ = evaluate_frozen.audit.__kwdefaults__
                invocation = SimpleNamespace(output_dir=args.output_dir,
                    initialization_dir=authenticate(repository_root(),
                        cfg['frozen_evaluation_artifacts']['static_initialization']).parent,
                    stage=1, seed=cfg['frozen_evaluation']['seed'], smoke=False)
                try:
                    entry(cfg, invocation, guard, {}, actor_override=actor, actor_reference=plan['actor'])
                except Prepared:
                    pass
                else:
                    raise ValueError('Preparation did not stop before policy evaluation')
                guard.check()
            left, left_tree = jax.tree_util.tree_flatten_with_path(states['bounded_2deg'])
            right, right_tree = jax.tree_util.tree_flatten_with_path(states['free_orientation'])
            if left_tree != right_tree:
                raise ValueError('Initial state schemas differ')
            differences = []
            for (path, x), (_, y) in zip(left, right):
                x, y = np.asarray(x), np.asarray(y)
                if not np.array_equal(x, y):
                    delta = np.abs(x.astype(float) - y.astype(float))
                    differences.append(dict(path=jax.tree_util.keystr(path), shape=list(x.shape),
                        dtype=str(x.dtype), max_abs_difference=float(delta.max()),
                        changed_elements=int(np.count_nonzero(x != y))))
            historical = {}
            for mode, dirname in [('bounded_2deg', 'bounded'), ('free_orientation', 'free')]:
                path = guard.budget.root / f'orientation_{dirname}_native_20261110_a01/result.json'
                old = json.loads(path.read_text())
                historical[mode] = dict(reference=dict(path=str(path), sha256=file_hash(path)),
                    recorded_native_sha256=old['initial_native_sha256'],
                    recorded_history_task_rng_sha256=old['initial_history_task_rng_sha256'],
                    matches_recorded_full_native=old['initial_native_sha256'] == report['preparations'][mode]['native_sha256'])
            report.update(status='PREPARATION_COMPARISON_COMPLETE', leaf_count=len(left),
                differences=differences, historical=historical,
                same_cached_warmed_seed=cfg['frozen_evaluation_artifacts']['warmed_seed'])
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_compact_json(args.output_dir/'result.json', report, guard.budget)


if __name__ == '__main__':
    main()
