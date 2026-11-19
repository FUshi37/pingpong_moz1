#!/usr/bin/env python3
"""Explicit64-update S15 probe; paired short evaluation never enters curriculum gates."""
from __future__ import annotations

from collections import Counter
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

EVALUATION_SEED = 2026111617
EVALUATION_ROLLOUTS = 20
BUDGET = 64
BASELINE_JSON = None
EXTRA_METRICS = (
    'done/racket_z_out_of_bounds', 'done/pending_hit_recontact', 'done/ball_too_high',
    'done/ball_x_out_of_bounds', 'done/ball_y_out_of_bounds',
    'done/ball_view_x_too_high', 'done/ball_view_x_too_low',
    'done/ball_view_y_too_high', 'done/ball_view_y_too_low',
    'done/ball_view_z_too_high', 'done/ball_view_z_too_low',
    'wind/startup_confirmed_hits', 'dynamics/release_event',
    'dynamics/release_error_vx_sq_sum', 'dynamics/release_error_vy_sq_sum',
    'dynamics/release_error_vz_sq_sum', 'dynamics/braking_cost', 'racket_z')


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


class FirstEpisodes:
    """Ignore every sample after a lane's first physical episode ends."""

    def __init__(self, modes: np.ndarray, dt_s: float):
        self.modes = np.asarray(modes, int)
        self.dt_s = dt_s
        self.alive = np.ones(len(modes), bool)
        self.steps = np.zeros(len(modes), int)
        self.hits = np.zeros(len(modes))
        self.recent = np.zeros(len(modes), bool)
        self.quality = [Counter() for _ in range(4)]
        self.failures = [Counter() for _ in range(4)]
        self.extra = [Counter() for _ in range(4)]

    def add(self, transitions: Any) -> None:
        from fixed_plane_v321_long_episode import QUALITY_FIELDS
        done = np.asarray(transitions.done, bool)
        hits = np.asarray(transitions.hit_count)
        metrics = {k: np.asarray(v) for k, v in transitions.metrics.items()
                   if k in QUALITY_FIELDS or k in EXTRA_METRICS or k == 'v321/recent_hit'}
        for tick in range(done.shape[0]):
            valid = self.alive.copy()
            self.steps += valid
            self.hits[valid] = hits[tick, valid]
            self.recent[valid] = metrics['v321/recent_hit'][tick, valid] > .5
            for mode in range(4):
                selected = valid & (self.modes == mode)
                for name in QUALITY_FIELDS:
                    self.quality[mode][name] += float(metrics[name][tick, selected].sum())
                for name in EXTRA_METRICS:
                    if name not in metrics:
                        continue
                    if name.startswith('done/'):
                        self.failures[mode][name] += int(metrics[name][tick, selected].sum())
                    else:
                        windy = selected & (metrics.get('wind/startup_confirmed_hits',
                            np.zeros_like(done))[tick] >= 5.)
                        self.extra[mode][name] += float(metrics[name][tick, windy].sum())
            self.alive &= ~done[tick]

    def summary(self) -> dict:
        from fixed_plane_v321_long_episode import quality_summary
        result = {}
        for index, name in enumerate(('clean', 'constant', 'slow', 'gust')):
            selected = self.modes == index
            n = int(selected.sum())
            if n == 0:
                continue
            count = self.extra[index].get('dynamics/release_event', 0.)
            result[name] = dict(n=n, survived=int((self.alive & self.recent & selected).sum()),
                survival=float((self.alive & self.recent)[selected].mean()),
                physical_survival=float(self.alive[selected].mean()),
                hits_mean=float(self.hits[selected].mean()),
                hit1=float((self.hits[selected] >= 1).mean()),
                hit3=float((self.hits[selected] >= 3).mean()),
                hit12=float((self.hits[selected] >= 12).mean()),
                mean_seconds=float(self.steps[selected].mean()) * self.dt_s,
                quality=quality_summary(self.quality[index]), failures=dict(self.failures[index]),
                releases_after5=count,
                outgoing_error_rms_m_s={axis: float(np.sqrt(self.extra[index].get(
                    f'dynamics/release_error_v{axis}_sq_sum', 0.) / max(count, 1.))) for axis in 'xyz'})
        return result


def promotion_decision(before: dict, after: dict) -> dict:
    """Require measured physical improvement, not return or model-score improvement alone."""
    failed = []
    for name in ('clean', 'constant'):
        old, new = before[name], after[name]
        if new['n'] != old['n']:
            failed.append(name + '/unpaired_population')
        if new['survival'] < old['survival'] - .0100001:
            failed.append(name + '/survival_regression')
        for key, tolerance in (('in_band', .01),):
            if new['quality'].get(key, 0.) < old['quality'].get(key, 0.) - tolerance:
                failed.append(name + '/' + key)
        if new['quality'].get('center_m', np.inf) > old['quality'].get('center_m', 0.) + .0005:
            failed.append(name + '/center')
        for key, absolute in (('racket_vxy_rms_m_s', .002), ('racket_omega_rms_rad_s', .02)):
            if new['quality'].get(key, np.inf) > old['quality'].get(key, 0.) * 1.1 + absolute:
                failed.append(name + '/' + key)
        if sum(v for k, v in new['failures'].items() if 'view' in k) > sum(
                v for k, v in old['failures'].items() if 'view' in k):
            failed.append(name + '/view')
    old, new = before['constant'], after['constant']
    old_z = old['failures'].get('done/racket_z_out_of_bounds', 0)
    new_z = new['failures'].get('done/racket_z_out_of_bounds', 0)
    if new_z > old_z:
        failed.append('constant/Z_regression')
    improved = (new['survival'] >= old['survival'] + .02
                or (old_z > 0 and new_z <= .9 * old_z))
    if not improved:
        failed.append('no_physical_target_improvement')
    return dict(passed=not failed, failures=failed, evaluation_seconds=12.8,
                scope='early-wind functional training test; no stage graduation claim')


def release_compiler_memory() -> None:
    import jax
    import ctypes
    jax.clear_caches()
    gc.collect()
    ctypes.CDLL('libc.so.6').malloc_trim(0)


def install_probe() -> None:
    import jax
    import jax.numpy as jnp
    import train_fixed_plane_wind as driver
    from fixed_plane_wind_clean_rehearsal import DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX
    original_factory = driver.make_train_functions

    def factory(stack, env, args):
        if args.wind_training_mix not in (DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX) or env.n_envs != 1024 or env.obs_dim != 114:
            raise ValueError('Guidance probe requires native114-D/1024env S15 dynamics mode')
        original_make = stack.ppo.make_train_fns

        def capture(**kwargs):
            kwargs['full_rollout_metric_names'] += EXTRA_METRICS
            return original_make(**kwargs)

        stack.ppo.make_train_fns = capture
        try:
            collect_fn, update_fn = original_factory(stack, env, args)
        finally:
            stack.ppo.make_train_fns = original_make
        collect_exe, update_exe, initial_eval = None, None, None
        updates, before = 0, None
        directory = Path(args.save_dir)

        def evaluate(params, label):
            nonlocal initial_eval
            if initial_eval is None:
                rng, reset = jax.random.split(jax.random.PRNGKey(EVALUATION_SEED))
                core, obs = jax.jit(env.reset)(jax.random.split(reset, env.n_envs))
                core, obs = driver.strong_arrays((core, obs))
                initial_eval = stack.ppo.RunnerState(core, obs, env.get_critic_obs(core, obs),
                    rng, jnp.zeros(env.n_envs), jnp.zeros(env.n_envs, jnp.int32))
            modes = np.asarray(initial_eval.env_state.wind_mode)
            tally = FirstEpisodes(modes, env.dt)
            runner = initial_eval
            for index in range(EVALUATION_ROLLOUTS):
                runner, transitions = collect_exe(params, runner)
                from fixed_plane_v321_long_episode import QUALITY_FIELDS
                selected = set(QUALITY_FIELDS + EXTRA_METRICS + ('v321/recent_hit',))
                # Do not copy the large actor/critic rollout tensors to host RAM.
                done, hits, metrics = jax.device_get((transitions.done, transitions.hit_count,
                    {k: v for k, v in transitions.metrics.items() if k in selected}))
                tally.add(SimpleNamespace(done=done, hit_count=hits, metrics=metrics))
                if index % 4 == 0:
                    print(f'[guidance probe evaluation] {label} {(index+1)*.64:.2f}/12.8s', flush=True)
            assert np.all(np.asarray(initial_eval.env_state.step_count) == 0)
            result = tally.summary()
            write_json(directory / f'probe_eval_{label}.json', result)
            print(f'[guidance probe evaluation complete] {label} {result}', flush=True)
            return result

        def collect(params, runner):
            nonlocal collect_exe, before
            if collect_exe is None:
                print('[guidance probe] compiling collection executable', flush=True)
                collect_exe = collect_fn.lower(params, runner).compile()
                release_compiler_memory()
                if BASELINE_JSON is None:
                    before = evaluate(params, 'before')
                else:
                    before = json.loads(BASELINE_JSON.read_text())
                    write_json(directory / 'probe_eval_before.json', before)
                    write_json(directory / 'probe_baseline_reuse.json', dict(path=str(BASELINE_JSON),
                        parity_evidence='NATIVE_QUALITY_REPAIR_PARITY.json',
                        parent_sha256='9153f7b916a05d807d19352ef3e7ee96eada7f4e1c5412071b22517a185be881',
                        evaluation_seed=EVALUATION_SEED))
                    print('[guidance probe] reused exact frozen parent baseline via explicit CLI', flush=True)
                release_compiler_memory()
            return collect_exe(params, runner)

        def update(state, runner, transitions):
            nonlocal updates, update_exe
            if update_exe is None:
                print('[guidance probe] compiling optimizer executable', flush=True)
                update_exe = update_fn.lower(state, runner, transitions).compile()
                release_compiler_memory()
            previous = int(state.opt.t)
            state, losses = update_exe(state, runner, transitions)
            values = {k: float(v) for k, v in jax.device_get(losses).items()}
            if int(state.opt.t) != previous + 32 or not all(np.isfinite(v) for v in values.values()):
                raise FloatingPointError('Probe requires32finite Adam minibatches')
            updates += 1
            prefix = stack.ppo.COMPACT_METRIC_MEAN_PREFIX
            means = {k[len(prefix):]: float(np.asarray(v).mean()) for k, v in
                     jax.device_get({k: v for k, v in transitions.metrics.items()
                                     if k.startswith(prefix)}).items()}
            record = dict(update=updates, adam_t=int(state.opt.t), loss=values,
                          means={k: v for k, v in means.items() if any(s in k for s in (
                              'done/', 'dynamics/', 'v321/finite', 'arm_acc', 'arm_vel'))})
            with (directory / 'probe_progress.jsonl').open('a') as handle:
                handle.write(json.dumps(record) + '\n')
            if updates == BUDGET:
                after = evaluate(state.params, 'after')
                decision = promotion_decision(before, after)
                write_json(directory / 'probe_comparison.json', dict(before=before, after=after,
                    decision=decision, state_digest=driver.numeric_digest(state), updates=updates,
                    adam_t=int(state.opt.t), evaluation_seed=EVALUATION_SEED))
                print(f'[guidance probe decision] {decision}', flush=True)
            return state, losses

        return collect, update

    driver.make_train_functions = factory


if __name__ == '__main__':
    import sys
    if '--probe-baseline-json' in sys.argv:
        import hashlib
        index = sys.argv.index('--probe-baseline-json')
        BASELINE_JSON = Path(sys.argv[index + 1]).resolve()
        del sys.argv[index:index + 2]
        source = Path(sys.argv[sys.argv.index('--resume') + 1])
        if (hashlib.sha256(source.read_bytes()).hexdigest() !=
                '9153f7b916a05d807d19352ef3e7ee96eada7f4e1c5412071b22517a185be881'
                or BASELINE_JSON.name != 'probe_eval_after.json'
                or BASELINE_JSON.parent.name != 'fplc_v325_gpu0_dynamics_probe_20261116_r1'):
            raise ValueError('Cached baseline requires the pinned first-test native learner')
    if '--probe-updates' not in sys.argv or sys.argv[sys.argv.index('--probe-updates')+1] != str(BUDGET):
        raise ValueError('This explicit probe must use exactly64updates')
    install_probe()
    import train_fixed_plane_wind
    train_fixed_plane_wind.main(325)
