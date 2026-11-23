"""Frozen V172 paired source/new-space and directional exposure CUDA screens."""
import argparse
import ast
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import types
import mjx_juggle_env as environment
import train_juggle_mjx_curriculum as trainer
from preflight_fixed_plane_v225 import authenticate, course
from fixed_plane_v225_near_side import PROFILE, PARENT_PROFILE, PARENT_SHA256, recovery_gate


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--checkpoint-sha', default=PARENT_SHA256)
    cli.add_argument('--out', type=Path, required=True)
    cli.add_argument('--cases', nargs='+', default=['parent', '1', '8', '11'])
    cli.add_argument('--seeds', nargs='+', type=int, default=[20266501, 20266502])
    cli.add_argument('--n-envs', type=int, default=288)
    cli.add_argument('--steps', type=int, default=1200)
    cli.add_argument('--legacy-source', type=Path)
    cli.add_argument('--reuse-runtime', action='store_true')
    cli.add_argument('--nominal-only', action='store_true')
    args = cli.parse_args()
    if jax.default_backend() != 'gpu':
        raise ValueError('V225 GPU evidence requires CUDA')
    source, sha = authenticate(args.checkpoint, args.checkpoint_sha)
    params = jax.tree.map(jnp.asarray, source['params'])
    settings = argparse.Namespace(**source['args'])
    settings.advance_eval_n_envs = args.n_envs
    settings.advance_eval_steps = args.steps
    settings.advance_eval_min_episodes = args.n_envs//2
    settings.advance_eval_stochastic = True
    settings.advance_validation_mode = 'block'
    settings.bounded_validation_memory = not args.reuse_runtime
    stages = course()
    if args.legacy_source:
        tree = ast.parse(args.legacy_source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'MjxJuggleEnv')
        method_names = {'reset', '_step_impl', '_make_obs', '_augment_obs', '_apply_observation_pipeline'}
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in method_names]
        if {n.name for n in methods} != method_names:
            raise ValueError('Archived environment methods incomplete')
        namespace = dict(vars(environment))
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(args.legacy_source), 'exec'), namespace)
        original_env = trainer.MjxJuggleEnv
        def legacy_env(*a, **k):
            env = original_env(*a, **k)
            if env.v225_enabled:
                raise ValueError('Legacy methods must only evaluate historical profiles')
            for name in method_names:
                setattr(env, name, types.MethodType(namespace[name], env))
            return env
        trainer.MjxJuggleEnv = legacy_env
    args.out.mkdir(parents=True, exist_ok=False)
    manifest = dict(profile=PROFILE, source_sha256=sha,
        devices=[str(x) for x in jax.devices()], cases={}, n_envs=args.n_envs,
        steps=args.steps, seeds=args.seeds, simulation_only=True,
        legacy_source=str(args.legacy_source) if args.legacy_source else None)
    original_runtime = trainer._get_advance_validation_runtime
    captured = {}
    def capture(*a, **k):
        env, rollout = original_runtime(*a, **k)
        def run(params, key):
            result = rollout(params, key)
            captured['outputs'] = result
            return result
        return env, run
    trainer._get_advance_validation_runtime = capture
    for case in args.cases:
        selected = [course(PARENT_PROFILE)[9]] if case == 'parent' else [stages[int(case)-1]]
        if args.nominal_only:
            if case == 'parent':
                raise ValueError('Nominal-cohort ablation requires a V225 lesson')
            selected = [replace(selected[0], cfg=replace(selected[0].cfg,
                v225_nominal_fraction=1., v225_mirror_fraction=0.))]
        directory = args.out/f'case_{case}'; directory.mkdir()
        (directory/'config.json').write_text(json.dumps(asdict(selected[0].cfg), indent=2))
        reports = []
        cache = {} if args.reuse_runtime else None
        for seed in args.seeds:
            print('CASE_START', case, seed, flush=True)
            block = trainer.run_advance_validation(settings, 1, selected, params,
                jax.random.PRNGKey(seed), runtime_cache=cache)
            output = captured.pop('outputs')
            names = ['terminated', 'truncated', 'hit_event_count', 'hit_count',
                     'ball_racket_anchor_xy_dist_m']
            if case != 'parent':
                names += [name for name in output['metrics'] if name.startswith('v225/')]
            arrays = jax.device_get({name: output['metrics'][name] for name in names})
            np.savez_compressed(directory/f'metrics_{seed}.npz', **arrays)
            extra = recovery_gate(arrays, selected[0].cfg) if case != 'parent' else {}
            result = dict(seed=seed, block=block, recovery=extra)
            (directory/f'block_{seed}.json').write_text(json.dumps(result, indent=2))
            reports.append(result)
            print('CASE_RESULT', case, seed, 'passed', block.get('advance_eval/passed'),
                  'hits', block.get('advance_eval/mean_hits'), 'len', block.get('advance_eval/mean_len_frac'),
                  'recovery', json.dumps(extra), flush=True)
            del output, arrays
            if not args.reuse_runtime:
                jax.clear_caches(); gc.collect()
        if cache is not None:
            cache.clear()
        jax.clear_caches(); gc.collect()
        manifest['cases'][case] = reports
        (args.out/'progress.json').write_text(json.dumps(manifest, indent=2))
    manifest['artifact_hashes'] = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.rglob('*') if p.is_file()}
    (args.out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
