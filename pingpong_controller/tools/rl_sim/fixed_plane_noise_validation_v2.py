"""Opt-in paired validation of frozen V323/V324 policies in their native domain."""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from fixed_plane_v321_long_episode import QUALITY_FIELDS, quality_passed, quality_summary
from fixed_plane_v322_condition_gates import (
    CONDITION_NAMES, condition_gate_settings, condition_ids,
)
from fixed_plane_v323_ball_obs_noise import spec_for

PROTOCOL = 'paired_validation_v2'
CASE_COUNT = 1024
SEEDS = (2026110407, 2026110417)
ACTION_SEED = 2026110427
QUALITY_MAX = dict(racket_vxy_rms_m_s=.055, racket_omega_rms_rad_s=.28,
    center_m=.0055, adjacent_jump_m=.012, recurrent_vxy_m_s=.05,
    recurrent_vxy_rms_m_s=.055, ball_vxy_m_s=.035, ball_vxy_rms_m_s=.045)


class ValidationMemoryLease:
    """Serialize cold compilation and validation, keeping hot PPO updates parallel."""
    def __init__(self, version: int, path: Path | None = None):
        self.version = version
        self.path = path or Path(f'/tmp/pingpong_noise_validation_v2_{os.getuid()}.lock')
        self.stream = None

    def acquire(self, stop: list[bool] | None = None) -> bool:
        descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        self.stream = os.fdopen(descriptor, 'a+')
        waiting = False
        while True:
            if stop is not None and stop[0]:
                self.release()
                return False
            try:
                fcntl.flock(self.stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if not waiting:
                    print(f'[V{self.version} memory slot] waiting for the other trainer\'s '
                          'compilation/validation; PPO state retained', flush=True)
                    waiting = True
                if stop is not None and stop[0]:
                    self.release()
                    return False
                time.sleep(1.)

    def release(self) -> None:
        if self.stream is not None:
            fcntl.flock(self.stream, fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


def profile_for(spec: Any, protocol: str) -> str:
    """Keep the original gate identity immutable and version the new contract."""
    if protocol not in ('legacy', PROTOCOL, 'training_quality_v3'):
        raise ValueError('Unknown noise validation protocol')
    if protocol == 'training_quality_v3':
        return spec.profile + '_training_v3'
    return spec.profile if protocol == 'legacy' else spec.profile + '_validation_v2'


def protocol_contract(version: int) -> dict:
    """Preregister population and gates before inspecting candidate outcomes."""
    spec = spec_for(version)
    return dict(protocol=PROTOCOL, profile=profile_for(spec, PROTOCOL),
        case_count=CASE_COUNT, seeds=list(SEEDS), action_seed=ACTION_SEED,
        retry_updates=[100, 1000], execution='mjx_topology_execution_v2',
        domain='unchanged native training reset/physics/noise configuration',
        pairing='identical reset keys and action innovations across modes and checkpoints',
        lessons_s=[30., 1200.], first_stage_primary='sampled',
        final_stage_required=['sampled', 'deterministic'],
        quality_max=dict(QUALITY_MAX), apex_band_m=[1.4, 1.45], min_in_band=.99,
        min_steady_events=100, min_hits_per_s=1.2,
        min_survival=spec.min_survival, min_length_fraction=spec.min_length_fraction,
        condition_gates=condition_gate_settings() if version == 324 else None,
        first_stage_conditions=['nominal'] if version == 324 else ['all'],
        final_stage_conditions=list(CONDITION_NAMES) if version == 324 else ['all'],
        trigger=dict(window_updates=32, readiness_hold=16, min_steady_events=100,
                     physical_quality='decided by frozen validation'),
        mixed_policy_endpoints='telemetry only; never block a fresh frozen-policy test',
        first_stage_release_qualified=False, hardware_coverage=False)


def validation_trigger_ready(window_size: int, quality: dict,
                             elapsed_s: float, deadline_s: float) -> bool:
    """Request a current-policy test once finite, populated telemetry is ready."""
    fields = set(QUALITY_MAX) | {'apex_m', 'in_band', 'steady_events'}
    return (window_size == 32 and elapsed_s >= deadline_s
            and quality.get('steady_events', 0.) >= 100
            and fields.issubset(quality)
            and all(math.isfinite(quality[name]) for name in fields))


def _grade_mode(version: int, stage_index: int, result: dict) -> dict:
    spec = spec_for(version)
    expected_s = spec.course[stage_index].seconds
    all_cases = result.get('all', {})
    failed = []
    if not result.get('complete') or result.get('observed_s') != expected_s:
        failed.append('incomplete_duration')
    if all_cases.get('n') != CASE_COUNT:
        failed.append('incomplete_population')
    if all_cases.get('numerical_errors', math.inf) != 0:
        failed.append('nonfinite_simulation')
    primary = (result.get('conditions', {}).get('nominal', {})
               if version == 324 and stage_index == 0 else all_cases)
    q = primary.get('quality', {})
    if not quality_passed(q):
        for key, limit in QUALITY_MAX.items():
            if not math.isfinite(q.get(key, math.inf)) or q[key] > limit:
                failed.append(f'quality/{key}')
        if not 1.4 <= q.get('apex_m', 0.) <= 1.45:
            failed.append('quality/apex_m')
        if not math.isfinite(q.get('in_band', float('nan'))) or q['in_band'] < .99:
            failed.append('quality/in_band')
        if not math.isfinite(q.get('steady_events', float('nan'))) or q['steady_events'] < 100:
            failed.append('quality/steady_events')
    if version == 323:
        cohorts = {'all': all_cases}
        limits = {'all': dict(min_n=CASE_COUNT, min_survival=spec.min_survival,
                              min_length_fraction=spec.min_length_fraction)}
    else:
        cohorts = result.get('conditions', {})
        limits = condition_gate_settings()
        if (set(cohorts) != set(CONDITION_NAMES)
                or sum(c.get('n', 0) for c in cohorts.values()) != CASE_COUNT):
            failed.append('incomplete_condition_population')
        if stage_index == 0:
            limits = {'nominal': limits['nominal']}
    for name, bounds in limits.items():
        measured = cohorts.get(name, {})
        if measured.get('n', 0) < bounds['min_n']:
            failed.append(f'{name}/sample_count')
        for field, minimum in (('survival', bounds['min_survival']),
                               ('len_fraction', bounds['min_length_fraction'])):
            value = measured.get(field, float('nan'))
            if not math.isfinite(value) or value < minimum:
                failed.append(f'{name}/{field}')
    if primary.get('hits_mean', 0.) / expected_s < 1.2:
        failed.append('primary/hits_per_s')
    return dict(passed=not failed, failed=failed)


def grade_validation(version: int, stage_index: int, modes: dict) -> dict:
    """Separate a sampled-policy curriculum bridge from final deterministic readiness."""
    if stage_index not in (0, 1):
        raise ValueError('Only the two registered noise lessons are supported')
    grades = {mode: _grade_mode(version, stage_index, modes.get(mode, {}))
              for mode in ('sampled', 'deterministic')}
    complete = all(modes.get(mode, {}).get('complete', False)
                   and modes[mode].get('observed_s') == spec_for(version).course[stage_index].seconds
                   and modes[mode].get('all', {}).get('n') == CASE_COUNT
                   and modes[mode].get('all', {}).get('numerical_errors', math.inf) == 0
                   for mode in grades)
    if version == 324:
        complete = complete and all(
            set(modes[mode].get('conditions', {})) == set(CONDITION_NAMES)
            and sum(c.get('n', 0) for c in modes[mode]['conditions'].values()) == CASE_COUNT
            and all(c.get('n', 0) >= 64 for c in modes[mode]['conditions'].values())
            for mode in grades)
    passed = (complete and grades['sampled']['passed']
              and (stage_index == 0 or grades['deterministic']['passed']))
    return dict(passed=bool(passed), complete=bool(complete), mode_gates=grades,
        deterministic_ready=grades['deterministic']['passed'],
        graduation_kind=('sampled_curriculum_bridge' if stage_index == 0
                         else 'sampled_and_deterministic_long_acceptance'),
        final_acceptance_passed=bool(stage_index == 1 and passed), hardware_coverage=False)


def reset_keys():
    """Two fixed seed panels; there is no training-update-dependent seed."""
    import jax
    import jax.numpy as jnp
    return jnp.concatenate([jax.random.split(jax.random.PRNGKey(seed), CASE_COUNT//2)
                            for seed in SEEDS])


def inference_params(params: dict) -> dict:
    """Actor inference never depends on a parent's differently sized critic."""
    # Keep every actor-composition field, including frozen teachers and gates.
    return {name: value for name,value in params.items()
            if name not in ('v', 'cmdp_cost_v')}


def build_chunk(env: Any, ppo: Any, finite_key: str, min_log_std: float,
                max_log_std: float | None, *, execution: str = 'topology_v2',
                steps: int = 200):
    """Use PPO action input, first-terminal freezing and native metrics."""
    import jax
    import jax.numpy as jnp
    from train_fixed_plane_v321 import freeze_lanes

    @jax.jit
    def chunk(params, state, obs, active, rng, sampled):
        def tick(carry, _):
            before, observation, live, key = carry
            key, noise_key, _ = jax.random.split(key, 3)
            mean = ppo.policy_mean(params, observation)
            noise = jnp.exp(ppo.effective_log_std(
                params['log_std'], min_log_std, max_log_std)) * jax.random.normal(noise_key, mean.shape)
            action = mean + jnp.where(sampled, noise, jnp.zeros_like(noise))
            action = ppo.policy_action_reward_input(action,
                expose_raw_action_for_reward=bool(getattr(env.cfg,
                    'fixed_contact_include_action_clip_excess_penalty', False)))
            after, next_obs, reward, _, metrics = env.step(before, action)
            finite = ((metrics[finite_key] > .5)
                & jnp.all(jnp.isfinite(next_obs), axis=-1) & jnp.isfinite(reward)
                & jnp.all(jnp.isfinite(after.data.qpos), axis=-1)
                & jnp.all(jnp.isfinite(after.data.qvel), axis=-1))
            for name in QUALITY_FIELDS:
                if name in metrics:
                    finite &= jnp.isfinite(metrics[name])
            failed = (metrics['terminated'] > .5) | ~finite
            packet = {name: jnp.where(live, metrics[name], 0.)
                      for name in QUALITY_FIELDS if name in metrics}
            packet.update({name: jnp.where(live, value, 0.)
                for name, value in metrics.items() if name.startswith('done/')})
            packet['numerical_errors'] = (live & ~finite).astype(jnp.float32)
            packet['valid_steps'] = live.astype(jnp.float32)
            packet['reward_sum'] = jnp.where(live, reward, 0.)
            state = freeze_lanes(before, after, live, jax, jnp)
            obs = freeze_lanes(observation, next_obs, live, jax, jnp)
            return (state, obs, live & ~failed, key), packet
        carry, packets = jax.lax.scan(tick, (state, obs, active, rng), None, length=steps)
        return carry, jax.tree.map(lambda value: jnp.sum(value, axis=0), packets)

    if execution == 'legacy':
        return chunk
    if execution != 'topology_v2':
        raise ValueError('Unsupported validation execution mode')
    from mjx_topology_execution import make_collect
    return make_collect(chunk, 'project')[0]


@dataclass
class EvaluationContext:
    """One reusable frozen population and one compiled inference graph."""
    stack: Any
    env: Any
    state: Any
    obs: Any
    chunk: Any
    condition: np.ndarray | None
    case_keys_sha256: str


def make_context(stack: Any, args: Any) -> EvaluationContext:
    import jax
    import numpy as np
    from bounded_validation_memory import release_caches
    env = stack.env_class(stack.xml, n_envs=CASE_COUNT, cfg=stack.cfg)
    keys = reset_keys()
    state, obs = jax.jit(env.reset)(keys)
    jax.block_until_ready((state, obs))
    labels = (condition_ids(np.asarray(state.v225_group), np.asarray(state.v225_reset_y_offset_m))
              if stack.spec.version == 324 else None)
    if labels is not None:
        counts = np.bincount(labels, minlength=5)
        if np.any(counts < 64):
            raise ValueError(f'Preregistered native population lacks condition coverage: {counts}')
    digest = hashlib.sha256(np.asarray(keys).tobytes()).hexdigest()
    release_caches()
    chunk = build_chunk(env, stack.ppo, f'v{stack.spec.version-2}/finite',
                        args.min_log_std, args.max_log_std)
    return EvaluationContext(stack, env, state, obs, chunk, labels, digest)


def _summarize(ctx: EvaluationContext, state: Any, active: Any, sums: dict,
               seconds: float, selected: np.ndarray) -> dict:
    n = int(selected.sum())
    if not n:
        return dict(n=0)
    duration = np.asarray(state.step_count)[selected]*ctx.env.dt
    hits = np.asarray(state.hit_count)[selected]
    age = (np.asarray(state.step_count)*ctx.env.dt
           - np.asarray(state.last_counted_hit_time))[selected]
    recent = (hits > 0) & (age >= 0.) & (age <= 1.2)
    return dict(n=n, survival=float((np.asarray(active)[selected] & recent).mean()),
        len_fraction=float(np.minimum(duration/seconds, 1.).mean()),
        hits_mean=float(hits.mean()), hits_min=int(hits.min()),
        hit1=float((hits >= 1).mean()), hit3=float((hits >= 3).mean()),
        hit12=float((hits >= 12).mean()),
        numerical_errors=float(sums['numerical_errors'][selected].sum()),
        quality=quality_summary({k: float(v[selected].sum()) for k,v in sums.items()
                                 if k in QUALITY_FIELDS}),
        failures={k: float(v[selected].sum()) for k,v in sums.items()
                  if k.startswith('done/') and v[selected].sum()})


def evaluate_policy(ctx: EvaluationContext, params: dict, stage_index: int,
                    stop: list[bool], *, label: str = 'candidate') -> dict:
    """Freeze one actor for both modes on exactly the same cases and innovations."""
    import jax
    import jax.numpy as jnp
    version = ctx.stack.spec.version
    seconds = ctx.stack.spec.course[stage_index].seconds
    params = jax.tree.map(jnp.asarray, inference_params(params))
    modes = {}
    for mode in ('sampled', 'deterministic'):
        state, obs = ctx.state, ctx.obs
        active = jnp.ones(CASE_COUNT, dtype=bool)
        rng = jax.random.PRNGKey(ACTION_SEED)
        sums = {}
        started = time.monotonic()
        observed = 0
        for second in range(1, int(seconds)+1):
            if stop[0]:
                break
            (state, obs, active, rng), packet = ctx.chunk(params, state, obs,
                active, rng, jnp.asarray(mode == 'sampled'))
            for name, value in jax.device_get(packet).items():
                sums[name] = sums.get(name, np.zeros(CASE_COUNT, dtype=np.float64)) + value
            observed = second
            if second % 5 == 0 or second == int(seconds):
                print(f'[V{version} paired eval] {label}/{mode} t={second}/{seconds:g}s '
                      f'alive={int(np.asarray(active).sum())}/{CASE_COUNT} '
                      f'wall={time.monotonic()-started:.1f}s', flush=True)
        if not sums:
            modes[mode] = dict(complete=False, observed_s=0., all={})
            continue
        all_cases = np.ones(CASE_COUNT, dtype=bool)
        modes[mode] = dict(complete=observed == int(seconds), observed_s=float(observed),
            all=_summarize(ctx, state, active, sums, seconds, all_cases),
            conditions=({name: _summarize(ctx, state, active, sums, seconds, ctx.condition == i)
                for i,name in enumerate(CONDITION_NAMES)} if ctx.condition is not None else {}),
            panels={str(seed): _summarize(ctx, state, active, sums, seconds,
                np.arange(CASE_COUNT)//(CASE_COUNT//2) == i) for i,seed in enumerate(SEEDS)})
    return dict(protocol=PROTOCOL, case_keys_sha256=ctx.case_keys_sha256,
        seeds=list(SEEDS), action_seed=ACTION_SEED, case_count=CASE_COUNT,
        label=label, modes=modes, **grade_validation(version, stage_index, modes))


def audit_evaluator(stack: Any, args: Any, train_state: Any, runtime: Any,
                    *, stop: list[bool] | None = None) -> dict:
    """Measure execution parity and parent/current policies before promotion."""
    import json
    import jax
    import jax.numpy as jnp
    from bounded_validation_memory import release_caches
    from train_fixed_plane_ball_obs_noise import numeric_digest

    before = numeric_digest(train_state)
    stop = stop if stop is not None else [False]
    ctx = make_context(stack, args)
    params = inference_params(train_state.params)
    active = jnp.ones(CASE_COUNT, dtype=bool)
    rng = jax.random.PRNGKey(ACTION_SEED)
    comparisons = []
    for sampled in (False, True):
        signatures = {}
        for execution in ('legacy', 'topology_v2'):
            if stop[0]:
                raise InterruptedError('Validation qualification cancelled')
            release_caches()
            chunk = build_chunk(ctx.env, stack.ppo, f'v{stack.spec.version-2}/finite',
                args.min_log_std, args.max_log_std, execution=execution)
            result = chunk(params, ctx.state, ctx.obs, active, rng, jnp.asarray(sampled))
            signatures[execution] = numeric_digest(result)
            del chunk, result
            print(f'[V{stack.spec.version} validation arithmetic] sampled={sampled} '
                  f'execution={execution} signature={signatures[execution]}', flush=True)
        comparisons.append(dict(sampled=sampled, **signatures))
        if signatures['legacy'] != signatures['topology_v2']:
            (runtime/'validation_pair_attempt.json').write_text(json.dumps(comparisons, indent=2))
            raise ValueError('New validation must retain exact physical state and gate statistics')
    release_caches()
    parent = evaluate_policy(ctx, stack.parent['params'], 0, stop, label=stack.spec.parent_model)
    candidate = evaluate_policy(ctx, train_state.params, 0, stop, label='resume_checkpoint')
    after = numeric_digest(train_state)
    report = dict(protocol=protocol_contract(stack.spec.version), comparisons=comparisons,
        reference_checkpoint_sha256=stack.spec.parent_sha256,
        learner_before=before, learner_after=after,
        parent=parent, candidate=candidate)
    (runtime/'validation_pair_attempt.json').write_text(json.dumps(report, indent=2))
    validate_evaluator_audit(report, stack.spec.version)
    del ctx
    release_caches()
    return report


def validate_evaluator_audit(report: dict, version: int) -> None:
    """Check raw pairing facts and complete populations, not a declared pass flag."""
    import re
    if report.get('protocol') != protocol_contract(version):
        raise ValueError('Validation population or gate contract changed')
    if report.get('reference_checkpoint_sha256') != spec_for(version).parent_sha256:
        raise ValueError('Wrong parent checkpoint in paired validation')
    comparisons = report.get('comparisons', [])
    if len(comparisons) != 2 or {r.get('sampled') for r in comparisons} != {False, True}:
        raise ValueError('Both action modes require measured execution pairing')
    for row in comparisons:
        if (not isinstance(row.get('legacy'), str)
                or not re.fullmatch('[a-f0-9]{64}', row['legacy'])
                or row['legacy'] != row.get('topology_v2')):
            raise ValueError('Validation execution arithmetic changed')
    if (not isinstance(report.get('learner_before'), str)
            or not re.fullmatch('[a-f0-9]{64}', report['learner_before'])
            or report['learner_before'] != report.get('learner_after')):
        raise ValueError('Validation modified the learner state')
    keys = np.asarray(reset_keys())
    key_hash = hashlib.sha256(keys.tobytes()).hexdigest()
    for name in ('parent', 'candidate'):
        measured = report.get(name, {})
        if (measured.get('case_keys_sha256') != key_hash
                or measured.get('seeds') != list(SEEDS)
                or measured.get('case_count') != CASE_COUNT
                or measured.get('action_seed') != ACTION_SEED
                or not grade_validation(version, 0, measured.get('modes', {}))['complete']):
            raise ValueError('Incomplete or unpaired parent/candidate validation audit')
        for mode in ('sampled', 'deterministic'):
            panels = measured['modes'][mode].get('panels', {})
            if (set(panels) != {str(seed) for seed in SEEDS}
                    or any(p.get('n') != CASE_COUNT//2 for p in panels.values())):
                raise ValueError('Missing preregistered validation seed panel')
