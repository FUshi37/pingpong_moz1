"""Fresh fixed-policy wind tests and per-domain endpoint accounting."""
from __future__ import annotations

from collections import deque
import time
import subprocess

import numpy as np

from fixed_plane_v321_long_episode import EndpointWindow, QUALITY_FIELDS, quality_summary
from fixed_plane_v322_condition_gates import condition_gate_settings, condition_ids
from fixed_plane_v325_wind import MODE_NAMES, make_profile_config


def current_domain_mask(done, valid_at_entry):
    """Exclude episodes spanning a domain change until their first natural reset."""
    done = np.asarray(done, bool)
    prior_resets = np.cumsum(done, axis=0) - done
    return prior_resets.astype(bool) | valid_at_entry[None, :], valid_at_entry | done.any(axis=0)


class WindEndpointWindow:
    """Keep one outcome per episode, together with its fixed wind-mode label."""
    def __init__(self, seconds, dt_s, capacity=4096):
        self.window = EndpointWindow(seconds, dt_s, capacity)
        self.modes = deque(maxlen=capacity)

    def add(self, lengths, hits, done, recent, modes, *, terminated):
        selected = (lengths == self.window.steps) | (done & (lengths < self.window.steps))
        self.modes.extend(np.asarray(modes, int)[selected].tolist())
        self.window.add(lengths, hits, done, recent, terminated=terminated)

    def summary(self):
        rows = np.asarray(self.window.outcomes, float)
        modes = np.asarray(self.modes)
        result = {}
        for index, name in enumerate(MODE_NAMES):
            values = rows[modes == index]
            result[name] = ({'n': len(values), 'len_fraction': float(values[:, 0].mean()),
                'hits_mean': float(values[:, 1].mean()), 'survival': float(values[:, 2].mean())}
                if len(values) else {'n': 0})
        return result


def endpoints_passed(summaries, lesson, origin):
    return all(summaries[name].get('n', 0) >= 64
        and summaries[name].get('survival', 0.) >= origin.min_survival
        and summaries[name].get('len_fraction', 0.) >= origin.min_length_fraction
        and summaries[name].get('hits_mean', 0.) / lesson.seconds >= 1.2
        for name, weight in zip(MODE_NAMES, lesson.mode_weights) if weight > 0.)


def build_evaluator(env, ppo, args, prefix, stochastic):
    """Freeze first-failure lanes and sum physical quality per environment."""
    import jax
    import jax.numpy as jnp
    from train_fixed_plane_v321 import freeze_lanes

    @jax.jit
    def chunk(params, state, obs, active, rng):
        def tick(carry, _):
            state, obs, active, rng = carry
            rng, key = jax.random.split(rng)
            action = ppo.policy_mean(params, obs)
            if stochastic:
                action += jnp.exp(ppo.effective_log_std(params['log_std'],
                    args.min_log_std, args.max_log_std)) * jax.random.normal(key, action.shape)
            after, next_obs, _, _, metrics = env.step(state, action)
            finite = metrics[prefix+'/finite'] > .5
            failed = (metrics['terminated'] > .5) | ~finite
            packet = {name: jnp.where(active, metrics[name], 0.) for name in QUALITY_FIELDS}
            packet.update({name: jnp.where(active, value, 0.)
                for name, value in metrics.items() if name.startswith('done/')})
            packet['numerical_error'] = (active & ~finite).astype(jnp.float32)
            packet['valid_steps'] = active.astype(jnp.float32)
            packet['wind_force_clipped_steps'] = jnp.where(active,
                metrics['wind/entry_force_clipped'], 0.)
            return (freeze_lanes(state, after, active, jax, jnp),
                    freeze_lanes(obs, next_obs, active, jax, jnp), active & ~failed, rng), packet
        carry, packets = jax.lax.scan(tick, (state, obs, active, rng), None, length=200)
        return carry, jax.tree.map(lambda x: jnp.sum(x, axis=0), packets)
    return chunk


def evaluate_policy(stack, args, params, stage, seed, stop, *, quiet=False,
                    seconds=None, n_envs=1024, modes=('deterministic', 'stochastic')):
    """Evaluate unchanged native cohorts, with separate clean/windy buckets."""
    import jax
    import jax.numpy as jnp
    from bounded_validation_memory import release_caches
    lesson = stack.spec.course[stage]
    seconds = lesson.seconds if seconds is None else seconds
    cfg = make_profile_config(stack.native_cfg, lesson, stack.spec.version, enabled=not quiet)
    env = stack.env_class(stack.xml, n_envs=n_envs, cfg=cfg)
    from train_fixed_plane_wind import base_version, metric_prefix
    prefix = metric_prefix(stack.spec)
    results = {}
    for mode in modes:
        release_caches()
        rng, reset_key = jax.random.split(jax.random.PRNGKey(seed))
        state, obs = jax.jit(env.reset)(jax.random.split(reset_key, n_envs))
        labels = np.asarray(state.wind_mode)
        conditions = (condition_ids(np.asarray(state.v225_group),
            np.asarray(state.v225_reset_y_offset_m)) if base_version(stack.spec) == 326
            else np.zeros(n_envs, int))
        active = jnp.ones(n_envs, bool)
        evaluator = build_evaluator(env, stack.ppo, args, prefix, mode == 'stochastic')
        sums = {}
        started = time.monotonic()
        for second in range(1, int(seconds)+1):
            if stop[0]:
                return dict(passed=False, complete=False, reason='interrupted', modes=results)
            (state, obs, active, rng), packet = evaluator(params, state, obs, active, rng)
            for name, value in jax.device_get(packet).items():
                sums[name] = sums.get(name, np.zeros(n_envs, float)) + value
            if stack.spec.version in (325, 326):
                from fixed_plane_wind_fresh_validation import impossible_frozen_gate
                impossible = impossible_frozen_gate(np.asarray(active), labels,
                    sums['valid_steps'], second, lesson, stack.spec.origin, quiet=quiet,
                    conditions=conditions if base_version(stack.spec) == 326 else None)
                if impossible:
                    return dict(passed=False, complete=False, observed_s=second,
                        quiet=quiet, fixed_policy=True, modes=results,
                        reason='irrecoverable_frozen_longevity', failures=impossible,
                        partial_action_mode=mode,
                        partial_failure_causes={k:int(v.sum()) for k,v in sums.items()
                                                if k.startswith('done/') and v.sum()})
            if second % 10 == 0 or second == int(seconds):
                print(f'[V{stack.spec.version} frozen {mode}] S{stage+1} quiet={quiet} '
                    f't={second}/{seconds:g}s alive={int(np.asarray(active).sum())}/{n_envs} '
                    f'wall={time.monotonic()-started:.1f}s', flush=True)
                temperature = float(subprocess.check_output(['nvidia-smi', '-i',
                    stack.spec.origin.gpu_uuid, '--query-gpu=temperature.gpu',
                    '--format=csv,noheader,nounits'], text=True))
                if temperature >= 78.:
                    stop[0] = True
                    return dict(passed=False, complete=False, reason='gpu_temperature', modes=results)
        age = np.asarray(state.step_count)*env.dt-np.asarray(state.last_counted_hit_time)
        success = np.asarray(active) & (np.asarray(state.hit_count) > 0) & (age <= 1.2)

        def summarize(mask):
            n = int(mask.sum())
            if not n:
                return {'n': 0}
            return dict(n=n, survival=float(success[mask].mean()),
                len_fraction=float(np.minimum(sums['valid_steps'][mask]*env.dt/seconds, 1.).mean()),
                hits_mean=float(np.asarray(state.hit_count)[mask].mean()),
                hit1=float((np.asarray(state.hit_count)[mask] >= 1).mean()),
                hit3=float((np.asarray(state.hit_count)[mask] >= 3).mean()),
                hit12=float((np.asarray(state.hit_count)[mask] >= 12).mean()),
                numerical_errors=int(sums['numerical_error'][mask].sum()),
                force_clipped_entry_fraction=float(sums['wind_force_clipped_steps'][mask].sum()
                    / max(1., sums['valid_steps'][mask].sum())),
                quality=quality_summary({k: float(v[mask].sum()) for k, v in sums.items()}),
                failures={k: int(v[mask].sum()) for k, v in sums.items() if k.startswith('done/')})

        nominal = conditions == 0
        result = dict(all=summarize(np.ones(n_envs, bool)),
            wind={name: summarize(nominal & (labels == index))
                  for index, name in enumerate(MODE_NAMES)},
            conditions={name: summarize(conditions == index)
                        for index, name in enumerate(condition_gate_settings())})
        results[mode] = result
        del evaluator, state, obs
    return grade_validation(stack, stage, results, quiet=quiet, seconds=seconds, n_envs=n_envs)


def grade_validation(stack, stage, results, *, quiet, seconds, n_envs):
    """Recompute every gate; a short probe cannot pass a full lesson."""
    lesson, origin = stack.spec.course[stage], stack.spec.origin
    failures = []
    if seconds != lesson.seconds:
        failures.append('duration_is_probe_only')
    if set(results) != {'deterministic', 'stochastic'}:
        failures.append('missing_action_mode')
    for mode, result in results.items():
        overall = result.get('all', {})
        if overall.get('n') != n_envs or overall.get('numerical_errors', 1) != 0:
            failures.append(f'{mode}/incomplete_or_nonfinite')
        wind_count = sum(item.get('n', 0) for item in result.get('wind', {}).values())
        expected_wind_count = (result.get('conditions', {}).get('nominal', {}).get('n', -1)
                              if stack.spec.version == 326 else n_envs)
        if wind_count != expected_wind_count:
            failures.append(f'{mode}/incomplete_wind_population')
        groups = {f'wind/{name}': (measured, dict(min_n=32,
            min_survival=origin.min_survival, min_length_fraction=origin.min_length_fraction))
            for index, name in enumerate(MODE_NAMES)
            for measured in (result.get('wind', {}).get(name, {}),)
            if (quiet and name == 'clean') or (not quiet and lesson.mode_weights[index] > 0.)}
        if set(result.get('wind', {})) != set(MODE_NAMES):
            failures.append(f'{mode}/missing_wind_buckets')
        if stack.spec.version == 326:
            if (set(result.get('conditions', {})) != set(condition_gate_settings())
                    or sum(item.get('n', 0) for item in result.get('conditions', {}).values()) != n_envs):
                failures.append(f'{mode}/incomplete_condition_population')
            for name, limits in condition_gate_settings().items():
                groups['condition/'+name] = (result.get('conditions', {}).get(name, {}), limits)
        for name, (measured, limits) in groups.items():
            if (measured.get('n', 0) < limits['min_n']
                    or not np.isfinite(measured.get('survival', np.nan))
                    or measured.get('survival', 0.) < limits['min_survival']
                    or not np.isfinite(measured.get('len_fraction', np.nan))
                    or measured.get('len_fraction', 0.) < limits['min_length_fraction']
                    or not np.isfinite(measured.get('hits_mean', np.nan))
                    or measured.get('hits_mean', 0.) / seconds < 1.2):
                failures.append(f'{mode}/{name}/longevity')
            # Match M13's original nominal motion-quality scope while retaining
            # its independently preregistered noncentral longevity floors.
            if name.startswith('wind/') and not stack.helpers.quality_passed(measured.get('quality', {})):
                failures.append(f'{mode}/{name}/quality')
    return dict(passed=not failures, complete=True, observed_s=seconds,
        quiet=quiet, modes=results, failures=failures,
        fixed_policy=True, hardware_coverage=False)
