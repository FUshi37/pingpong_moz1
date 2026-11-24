#!/usr/bin/env python3
"""Opt-in long-episode continuation of the authenticated M11 simulator.

Use run_gpu1_fixed_plane_v322_long_episode.sh. No step/update budget is applied
to formal training. Metrics and W&B SDK files are temporary RAM files only.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from fixed_plane_v322_condition_gates import (
    ConditionEndpointWindow, condition_gate_settings, condition_gates_passed,
    centered_acceptance_config, required_conditions,
)

from fixed_plane_v322_long_episode import (
    COURSE, PROFILE, PARENT_SHA256, XML_SHA256, GPU_UUID, MAX_HIT_GAP_S,
    CONFIG_OVERRIDES, QUALITY_FIELDS, EndpointWindow, CheckpointRetention,
    make_config, survival_reward, quality_summary, quality_passed,
    PARENT_RUN, PARENT_FILE, PPO_EXPECTED, REPLAY_HASHES, EVAL_ENVS,
    EVAL_SECONDS, EVAL_WALL_BUDGET_S, EVAL_LESSON, EVAL_RETRY_UPDATES, should_validate,
    restore_nested_config,
    REWARD_PRESETS, reward_adjustments, validate_reward_resume,
    STARTUP_PREPARATION_SETTINGS, CONTACT_QUALITY_SETTINGS,
    LOSS_PRESETS, loss_settings, resolve_loss_resume,
    GATE_PRESETS, course_for, evaluation_lesson_for, resolve_gate_resume,
    STARTUP_RELEASE_SETTINGS, STARTUP_TRANSFER_SETTINGS, STARTUP_LEAD_SETTINGS,
    SHARED_PREPARATION_SETTINGS, SHARED_EARLY_SETTINGS, NATIVE_THIRD_FOCUS_SETTINGS, NATIVE_REFERENCE_PREPARATION_SETTINGS,
    NATIVE_REFERENCE_GAIN6_SETTINGS,
    THIRD_RETURN_PREPARATION_SETTINGS,
    PROJECTED_RETURN_PREPARATION_SETTINGS,
    ACQUISITION_XY_SETTINGS, WEAK_RELEASE_OUTCOME_SETTINGS,
    REFERENCE_TIMING_SETTINGS, reference_timing_active,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration_json(value) -> str:
    """Canonicalize nested checkpoint dataclasses without dropping fields."""
    return json.dumps(value, sort_keys=True, default=lambda x:
        asdict(x) if is_dataclass(x) else np.asarray(x).tolist())


def load_stack(repo: Path, ppo_source: Path):
    """Use immutable M11 dynamics plus the explicit accept-every-minibatch PPO."""
    run = repo / 'pingpong_controller/outputs/rl_sim' / PARENT_RUN
    snapshot = run / 'source_snapshot/pingpong_controller/tools/rl_sim'
    checkpoint = run / PARENT_FILE
    xml = snapshot / 'moz1_pd_1ms_v98.xml'
    for path, expected in ((checkpoint, PARENT_SHA256), (xml, XML_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Authenticated M11 input changed: {path}')
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location('train_juggle_mjx_ppo', ppo_source)
    ppo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ppo
    spec.loader.exec_module(ppo)
    from validate_juggle_mjx_ppo import env_config_from_checkpoint, load_checkpoint, parse_args
    parent = load_checkpoint(checkpoint)
    cfg = env_config_from_checkpoint(parent, parse_args([]))
    from fixed_contact_plane_training import HighBallPredictorConfig, ContactTargetConfig
    cfg = restore_nested_config(cfg, HighBallPredictorConfig, ContactTargetConfig)
    restored = asdict(cfg)
    for key, value in parent['env_cfg'].items():
        if key not in restored or configuration_json(value) != configuration_json(restored[key]):
            raise ValueError(f'M11 saved config mismatch: {key}')
    if (parent['args']['curriculum_profile'] != 'fixed_plane_limit_cycle_qvel_rmp_v319_mean0265_completion'
            or parent['stage_name'] != 'fplc_v281_s17_unchanged_consolidation'
            or parent['stage_update'] != 57 or parent['step'] != 4127719424
            or (parent['obs_dim'], parent['critic_obs_dim'], parent['act_dim']) != (72, 393, 7)):
        raise ValueError('Wrong M11 identity')
    for key, expected in REPLAY_HASHES.items():
        if sha(Path(parent['args'][key])) != expected:
            raise ValueError(f'M11 replay provenance changed: {key}')
    if parent.get('stage_actor_anchor_reference_params') is None:
        raise ValueError('M11 must retain its exact S17 actor replay KL origin')
    return ppo, parent, cfg, xml, snapshot


def published_reference_vertical_velocity(env, before, after, active,
                                          current_reference_z_m):
    """Measure reference-only FK change using the same current physical base.

    The current pose reuses native V304's exact reference FK. The previous
    reference is an immutable kinematic copy using the current physical base.
    Neither enters state, observations or control. Outside the cue the new
    diagnostic is zero; every native sample and metric remains unchanged.
    """
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    def measure(_):
        previous_data = after.data.replace(
            qpos=after.data.qpos.at[:,env.arm_qadr].set(before.arm_cmd_q))
        kinematics = jax.vmap(mjx.kinematics)
        previous_fk = kinematics(after.model,previous_data)
        velocity = (current_reference_z_m
            -previous_fk.site_xpos[:,env.racket_site_id,2])/env.dt
        return jnp.where(active,velocity,0.)

    return jax.lax.cond(jnp.any(active),measure,
        lambda _:jnp.zeros(after.data.qpos.shape[0],dtype=after.data.qpos.dtype),None)


def environment_class(base, *, reward_preset: str = 'baseline',
                      grouped_conditions: bool = False):
    """Adapt survival and optional audited costs; preserve native transitions."""
    import jax.numpy as jnp
    requested_reward_preset = reward_preset
    if reward_preset in ('late_cycle_repair', 'late_center_precision'):
        reward_preset = 'reference_timing'

    class LongEpisodeEnv(base):
        def step(self, state, action):
            after, obs, reward, done, metrics = super().step(state, action)
            age_s = after.step_count * self.dt - after.last_counted_hit_time
            alive_reward = survival_reward(after.hit_count, age_s,
                metrics['terminated'].astype(bool), self.dt, xp=jnp)
            old_reward = metrics['reward/fixed_contact_survival_reward']
            reward = reward + alive_reward - old_reward
            metrics['reward/fixed_contact_survival_reward'] = alive_reward
            if reward_preset in ('startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
                metrics['v322_pending_after'] = after.pending_hit.astype(jnp.float32)
                metrics['v322_pending_age_s'] = after.pending_hit_steps * self.dt
                metrics['v322_gravity_m_s2'] = jnp.abs(state.dr_gravity_z)
            if reward_preset == 'reference_timing':
                active = reference_timing_active(metrics,state.hit_count,xp=jnp)
                metrics['v322_reference_vz_m_s'] = published_reference_vertical_velocity(
                    self,state,after,active,
                    metrics['racket_z']+metrics['v304_reference_lead_m'])
            if requested_reward_preset in ('late_cycle_repair', 'late_center_precision'):
                metrics['v322_previous_apex_error_m'] = state.late_physical_prev_signed_apex_error_m
                metrics['v322_current_apex_error_m'] = after.late_physical_prev_signed_apex_error_m
                metrics['v322_previous_apex_valid'] = state.late_physical_prev_apex_valid.astype(jnp.float32)
            correction, terms = reward_adjustments(metrics, requested_reward_preset,
                jnp.zeros_like(reward), hit_count=after.hit_count,
                previous_hits=None if state is None else state.hit_count,
                dt_s=self.dt, xp=jnp)
            reward = reward + correction
            metrics.update(terms)
            metrics['reward/total'] = reward
            if grouped_conditions:
                # Saved initial labels stay fixed even if a nominal flight
                # later drifts near the robot. They never enter observations.
                metrics['v322/initial_group'] = state.v225_group.astype(jnp.float32)
                metrics['v322/initial_y_m'] = state.v225_reset_y_offset_m
            metrics['v322/recent_hit'] = (
                (after.hit_count > 0) & (age_s >= 0.) & (age_s <= MAX_HIT_GAP_S)
            ).astype(jnp.float32)
            recurrent = after.hit_count >= 2
            metrics['recurrent_hit_racket_vxy_sum'] = jnp.where(
                recurrent, metrics['hit_racket_vxy_sum'], 0.)
            metrics['recurrent_hit_racket_vxy_sq_sum'] = jnp.where(
                recurrent, metrics['hit_racket_vxy_sq_sum'], 0.)
            metrics['v322/finite'] = (jnp.all(jnp.isfinite(obs), axis=-1)
                & jnp.isfinite(reward)).astype(jnp.float32)
            return after, obs, reward, done, metrics

    return LongEpisodeEnv


def freeze_lanes(old, new, active, jax, jnp):
    """Keep first-terminal lane states; preserve native shared scalar clocks."""
    return jax.tree.map(lambda a, b: b if b.ndim == 0 else jnp.where(
        active.reshape((len(active),) + (1,) * (b.ndim - 1)), b, a), old, new)


def build_evaluator(env, ppo, min_log_std, max_log_std):
    """Equal deterministic/stochastic groups, one episode per seed."""
    import jax
    import jax.numpy as jnp

    @jax.jit
    def run_chunk(params, state, obs, active, rng):
        def body(carry, _):
            state, obs, active, rng = carry
            rng, key = jax.random.split(rng)
            mean = ppo.policy_mean(params, obs)
            noise = jnp.exp(ppo.effective_log_std(
                params['log_std'], min_log_std, max_log_std)) * jax.random.normal(key, mean.shape)
            action = mean + jnp.where((jnp.arange(env.n_envs) >= env.n_envs // 2)[:, None], noise, 0.)
            # Same raw-action reward input as training; the environment applies
            # the original safety projection and actuator constraints.
            after, next_obs, _, done, metrics = env.step(state, action)
            finite = metrics['v322/finite'] > .5
            failed = (metrics['terminated'] > 0.) | ~finite
            packet = {name: jnp.where(active, metrics[name], 0.)
                      for name in QUALITY_FIELDS if name in metrics}
            packet.update({name: jnp.where(active, value, 0.)
                for name, value in metrics.items() if name.startswith('done/')})
            packet['failed'] = (active & failed).astype(jnp.float32)
            packet['numerical_error'] = (active & ~finite).astype(jnp.float32)
            packet['valid_step'] = active.astype(jnp.float32)
            state = freeze_lanes(state, after, active, jax, jnp)
            obs = freeze_lanes(obs, next_obs, active, jax, jnp)
            # Native truncation is not a physical failure at the final 1200 s.
            return (state, obs, active & ~failed, rng), packet

        carry, packets = jax.lax.scan(body, (state, obs, active, rng), None, length=200)
        return carry, jax.tree.map(lambda value: jnp.sum(value, axis=0), packets)

    return run_chunk


def probe_reward_pair(base, xml, cfg, ppo, params, opts):
    """Compare two frozen-policy CUDA trajectories; write only a small audit."""
    import jax
    import jax.numpy as jnp
    baseline = base(xml, n_envs=16, cfg=cfg)

    @jax.jit
    def chunk(state, obs, active):
        def tick(carry, _):
            before, observation, live = carry
            action = jnp.tanh(ppo.policy_mean(params, observation))
            native = baseline.step(before, action)

            class NativeTransition:
                # Physics is identical for a frozen actor when only reward
                # changes. Replay the same real native frame through both
                # adapters, checking unchanged input identity as well as all
                # outputs; avoid compiling duplicate MuJoCo/RMP graphs.
                dt = baseline.dt

                def __getattr__(self, name):
                    return getattr(baseline, name)

                def step(self, passed_state, passed_action):
                    if passed_state is not before or passed_action is not action:
                        raise AssertionError('Reward adapter changed native step inputs')
                    return (*native[:4], dict(native[4]))

            old_preset = ('late_cycle_repair' if opts.reward_preset == 'late_center_precision'
                else 'reference_timing' if opts.reward_preset == 'late_cycle_repair' else 'baseline')
            old = environment_class(NativeTransition, reward_preset=old_preset)().step(before, action)
            new = environment_class(NativeTransition, reward_preset=opts.reward_preset)().step(
                before, action)
            # Opt-in reward-only telemetry exists on the new adapter; native
            # channels are verified unchanged separately below.
            delta, _ = reward_adjustments(new[4], opts.reward_preset,
                jnp.zeros_like(old[2]), hit_count=old[0].hit_count,
                previous_hits=before.hit_count, dt_s=baseline.dt, xp=jnp)
            if opts.reward_preset in ('late_cycle_repair', 'late_center_precision'):
                inherited, _ = reward_adjustments(old[4], old_preset,
                    jnp.zeros_like(old[2]), hit_count=old[0].hit_count,
                    previous_hits=before.hit_count, dt_s=baseline.dt, xp=jnp)
                delta = delta-inherited
            same_state = jnp.all(jnp.stack(jax.tree.leaves(jax.tree.map(
                lambda x, y: jnp.array_equal(x, y), old[0], new[0]))))
            same_metrics = jnp.all(jnp.stack([
                jnp.array_equal(old[4][k], new[4][k]) for k in old[4]
                if k not in ('reward/total','reward/v322_late_center_penalty',
                             'reward/v322_late_energy_contraction_penalty')]))
            packet = dict(
                state_equal=same_state, native_metrics_equal=same_metrics,
                observation_equal=jnp.array_equal(old[1], new[1]),
                done_equal=jnp.array_equal(old[3], new[3]),
                reward_error=jnp.max(jnp.abs(new[2] - (old[2] + delta))),
                finite=jnp.all(jnp.isfinite(new[2])),
                protected_launch_unchanged=jnp.all(jnp.where(
                    (before.hit_count == 0 if opts.reward_preset in ('stability_launch', 'contact_quality', 'balanced_contact', 'startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing')
                     else old[0].hit_count < 4), delta == 0., True)),
                corrected_steps=jnp.sum(live & (delta != 0.)),
                delta_sum=jnp.sum(jnp.where(live, delta, 0.)),
                hits=jnp.sum(jnp.where(live, old[4]['hit_event_count'], 0.)),
                failures=jnp.sum(live & old[3]))
            next_state = freeze_lanes(before, old[0], live, jax, jnp)
            next_obs = freeze_lanes(observation, old[1], live, jax, jnp)
            return (next_state, next_obs, live & ~old[3]), packet
        return jax.lax.scan(tick, (state, obs, active), None, length=200)

    reset = jax.jit(baseline.reset)
    results = []
    for seed in (2026103001, 2026103002):
        state, obs = reset(jax.random.split(jax.random.PRNGKey(seed), 16))
        active = jnp.ones(16, dtype=bool)
        checks = dict(state_equal=True, native_metrics_equal=True,
                      observation_equal=True, done_equal=True, finite=True,
                      protected_launch_unchanged=True)
        counts = dict(corrected_steps=0., delta_sum=0., hits=0., failures=0.)
        error = 0.
        for second in range(1, 7):
            (state, obs, active), packet = chunk(state, obs, active)
            packet = jax.device_get(packet)
            for name in checks:
                checks[name] &= bool(np.all(packet[name]))
            for name in counts:
                counts[name] += float(np.sum(packet[name]))
            error = max(error, float(np.max(packet['reward_error'])))
            print(f'[V322 reward pair] seed={seed} t={second}/6s '
                  f'alive={int(active.sum())}/16 checks={all(checks.values())} '
                  f'reward_error={error:g}', flush=True)
        results.append(dict(seed=seed, checks=checks, **counts,
                            reward_error=error, alive=int(active.sum())))
    passed = all(all(r['checks'].values()) and r['reward_error'] <= 1e-4
                 and r['corrected_steps'] > 0 for r in results)
    report = dict(passed=passed, frozen_checkpoint=str(opts.resume),
        checkpoint_sha256=sha(opts.resume), reward_preset=opts.reward_preset,
        pair_method='one native physical step, two independent reward adapters; '
                    'assert identical native inputs and compare full outputs',
        envs=16, seconds=6, devices=[str(d) for d in jax.devices()], results=results)
    (opts.runtime / 'reward_pair_result.json').write_text(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(f'V322 frozen reward-only pair failed: {report}')


def evaluate(env, evaluator, params, lesson, seed, stop, log):
    """Freeze weights for a fresh full-window test; no reset after failure."""
    import jax
    import jax.numpy as jnp

    started = time.monotonic()
    rng, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, env.n_envs))
    active = jnp.ones(env.n_envs, bool)
    sums = {}
    for second in range(1, int(lesson.seconds) + 1):
        if stop[0]:
            return None
        if time.monotonic()-started >= EVAL_WALL_BUDGET_S:
            return dict(passed=False, complete=False, observed_s=second-1,
                        reason='validation_wall_budget_exhausted',
                        wall_s=time.monotonic()-started)
        (state, obs, active, rng), packet = evaluator(params, state, obs, active, rng)
        packet = jax.device_get(packet)
        for name, value in packet.items():
            sums[name] = sums.get(name, np.zeros(env.n_envs, dtype=np.float64)) + value
        if second % 10 == 0 or second == int(lesson.seconds):
            print(f'[V322 eval] {lesson.name} t={second}/{lesson.seconds:g}s '
                  f'alive={int(np.asarray(active).sum())}/{env.n_envs} '
                  f'wall={time.monotonic()-started:.1f}s', flush=True)
        # A failed lane is never replaced. Stop hopeless validation early,
        # explicitly mark incomplete; it can never satisfy advancement.
        group_alive = np.asarray(active).reshape(2, -1).mean(axis=1)
        if np.any(group_alive < lesson.min_survival):
            return dict(passed=False, complete=False, observed_s=second,
                deterministic_survival=float(group_alive[0]),
                stochastic_survival=float(group_alive[1]),
                reason='survival_below_required_bound',
                failures={k: float(v.sum()) for k, v in sums.items() if k.startswith('done/')})
    age = np.asarray(state.step_count) * env.dt - np.asarray(state.last_counted_hit_time)
    recent = (np.asarray(state.hit_count) > 0) & (age <= MAX_HIT_GAP_S)
    results = {}
    for index, name in enumerate(('deterministic', 'stochastic')):
        selection = slice(index * env.n_envs // 2, (index+1) * env.n_envs // 2)
        q = quality_summary({k: float(v[selection].sum()) for k, v in sums.items()})
        survival = float((np.asarray(active)[selection] & recent[selection]).mean())
        duration = np.asarray(state.step_count)[selection] * env.dt
        fraction = float(np.minimum(duration / lesson.seconds, 1.).mean())
        hits = np.asarray(state.hit_count)[selection]
        results[name] = dict(quality=q, survival=survival,
            len_fraction=fraction, hits_mean=float(hits.mean()),
            hits_min=int(hits.min()),
            passed=(quality_passed(q) and survival >= lesson.min_survival
                    and fraction >= lesson.min_length_fraction
                    and float(hits.mean()) / lesson.seconds >= 1.2))
    return dict(complete=True, observed_s=lesson.seconds,
        passed=all(r['passed'] for r in results.values()), groups=results,
        failures={k: float(v.sum()) for k, v in sums.items() if k.startswith('done/')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--save-dir', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--seed', type=int, default=2026102907)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--reward-probe', action='store_true')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--reward-preset', choices=tuple(REWARD_PRESETS), default='baseline')
    parser.add_argument('--allow-reward-change', action='store_true')
    parser.add_argument('--loss-preset', choices=tuple(LOSS_PRESETS), default=None)
    parser.add_argument('--allow-loss-change', action='store_true')
    parser.add_argument('--gate-preset', choices=tuple(GATE_PRESETS), default=None)
    parser.add_argument('--allow-gate-change', action='store_true')
    parser.add_argument('--quality-repair', action='store_true')
    parser.add_argument('--long-qualification-audit', type=Path)
    parser.add_argument('--packed-host-metrics', action='store_true',
        help='Opt in to packed_host_metrics_v1; retain every rollout metric')
    parser.add_argument('--mjx-line-search-schedule',
        choices=('legacy', 'chunk5_v1'), default='legacy',
        help='Opt in to validated MJX loop scheduling; physical caps unchanged')
    parser.add_argument('--mjx-execution-mode',
        choices=('legacy', 'topology_v2'), default='legacy',
        help='Opt in to topology/unique-scatter/projection scheduling')
    opts = parser.parse_args()
    if (opts.mjx_execution_mode != 'legacy'
            and opts.mjx_line_search_schedule != 'legacy'):
        parser.error('Choose one MJX execution option; do not stack schedules')
    if opts.reward_probe and (opts.mjx_line_search_schedule != 'legacy'
                             or opts.mjx_execution_mode != 'legacy'):
        parser.error('MJX scheduling opt-in applies to training/smoke only')
    if (os.environ.get('PP_MEMORY_SCOPE_ACTIVE') != 'YES'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID):
        raise ValueError('Use the isolated GPU1 launcher')
    if opts.runtime.resolve().parent != Path('/dev/shm'):
        raise ValueError('Runtime data must stay in /dev/shm')
    ppo, parent, parent_cfg, xml, snapshot = load_stack(
        opts.repo_root, opts.runtime / 'source/train_juggle_mjx_ppo.py')
    import jax
    import jax.numpy as jnp
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('CUDA GPU1 is mandatory')
    from mjx_juggle_env import MjxJuggleEnv
    from bounded_validation_memory import release_caches
    cfg = make_config(parent_cfg)
    args = argparse.Namespace(**parent['args'])
    args.curriculum_profile = PROFILE
    args.save_dir = opts.save_dir
    args.seed = opts.seed
    args.xml = xml
    args.wandb_name = opts.run_id
    args.wandb_id = opts.run_id
    args.wandb_mode = 'online'
    args.wandb_resume = 'never'
    args.resume_from = opts.resume or snapshot.parents[3] / PARENT_FILE
    args.resume_start_stage = COURSE[0].name
    args.resume_curriculum_state = False
    args.reset_optimizer_on_resume = False
    args.reset_critic_on_resume = False
    args.apply_every_minibatch = True
    args.total_update_limit = None
    args.max_stage_updates = -1
    args.max_stages = len(COURSE)
    args.convergence_hold_updates = 16
    args.convergence_min_episodes = 1024
    args.save_every_updates = 100
    args.archive_every_updates = 0
    args.training_entrypoint = 'train_fixed_plane_v322.py'
    args.advance_eval_n_envs = EVAL_ENVS
    args.advance_eval_min_episodes = EVAL_ENVS
    args.advance_retention_stage_name = None
    args.advance_eval_stochastic = True
    args.conditional_reset_done = True
    expected = PPO_EXPECTED
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f'M11 PPO source mismatch {name}')
    payload = parent
    stage_index = 0
    update_idx = 0
    if opts.resume:
        from validate_juggle_mjx_ppo import load_checkpoint
        payload = load_checkpoint(opts.resume)
        if (payload['args']['curriculum_profile'] != PROFILE
                or payload.get('v322_contract', {}).get('parent_sha256') != PARENT_SHA256
                or payload.get('v322_contract', {}).get('config_overrides') != CONFIG_OVERRIDES
                or (payload['obs_dim'], payload['critic_obs_dim'], payload['act_dim']) != (72, 393, 7)):
            raise ValueError('Only an exact V322 checkpoint can resume this course')
        if configuration_json(payload['env_cfg']) != configuration_json(asdict(cfg)):
            raise ValueError('V322 resume environment configuration changed')
        validate_reward_resume(payload.get('v322_contract', {}), opts.reward_preset,
                               allow_change=opts.allow_reward_change)
        stage_index = int(payload['stage_index']) + int(payload.get('stage_converged', False))
        update_idx = int(payload['global_update'])
        if stage_index >= len(COURSE):
            raise ValueError('This V322 checkpoint already completed all lessons')
    loss_preset = resolve_loss_resume(payload.get('v322_contract', {}),
        payload['args'], opts.loss_preset, allow_change=opts.allow_loss_change)
    gate_preset = resolve_gate_resume(payload.get('v322_contract', {}),
        opts.gate_preset, allow_change=opts.allow_gate_change)
    if (opts.mjx_line_search_schedule != 'legacy'
            or opts.mjx_execution_mode != 'legacy'):
        from mjx_line_search_schedule import validate_v322_execution_contract
        validate_v322_execution_contract(
            opts.reward_preset, loss_preset, gate_preset)
    inherited_long = None
    if opts.quality_repair:
        from fixed_plane_v322_quality_repair import validate_quality_repair_resume
        if (opts.resume is None or opts.long_qualification_audit is None
                or opts.reward_preset not in ('late_cycle_repair','late_center_precision')
                or loss_preset != 'anchor1' or gate_preset != 'deployment_groups'):
            raise ValueError('Use the reviewed late-cycle quality-repair continuation')
        inherited_long = validate_quality_repair_resume(payload,
            json.loads(opts.long_qualification_audit.read_text()), sha(opts.resume))
    course = course_for(gate_preset)
    eval_lesson = evaluation_lesson_for(gate_preset)
    grouped_conditions = gate_preset == 'deployment_groups'
    group_settings = condition_gate_settings() if grouped_conditions else None
    expected = loss_settings(loss_preset)
    # Only this explicitly authorized loss scalar differs from the M11 learner.
    args.actor_anchor_replay_kl_coef = expected['actor_anchor_replay_kl_coef']
    params = jax.tree.map(jnp.asarray, payload['params'])
    train_state = ppo.TrainState(params, jax.tree.map(jnp.asarray, payload['opt']))
    reference_params = jax.tree.map(jnp.asarray, parent['stage_actor_anchor_reference_params'])
    global_step = int(payload['step'])
    if opts.quality_repair and int(train_state.opt.t) != inherited_long['resume_origin']['Adam_t']:
        raise ValueError('Inherited quality repair Adam state mismatch')
    if opts.reward_probe:
        if opts.resume is None or opts.reward_preset == 'baseline':
            raise ValueError('Reward probe requires a frozen V322 source and reward opt-in')
        probe_reward_pair(MjxJuggleEnv, xml, cfg, ppo, params, opts)
        return
    env_class = environment_class(MjxJuggleEnv, reward_preset=opts.reward_preset,
                                  grouped_conditions=grouped_conditions)
    env = env_class(xml, n_envs=1024, cfg=cfg)
    if (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) != (72, 393, 7, 240000):
        raise ValueError('V322 layout/horizon mismatch')
    # Parent actor/critic and Adam are retained exactly. Critic time features
    # retain fractional episode semantics, now normalized by 240000 steps.
    rng, reset_key = jax.random.split(jax.random.PRNGKey(opts.seed))
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, env.n_envs))
    critic_obs = env.get_critic_obs(state, obs)
    jax.block_until_ready(obs)
    runner = ppo.RunnerState(state, obs, critic_obs, rng,
        jnp.zeros(env.n_envs), jnp.zeros(env.n_envs, dtype=jnp.int32))
    replay_obs = np.load(args.counterfactual_replay_obs)
    replay_actions = np.load(args.counterfactual_replay_actions)
    if replay_obs.shape[1] != 72 or replay_actions.shape != (len(replay_obs), 7):
        raise ValueError('Wrong original M11 BC replay dimensions')
    kwargs = {k: getattr(args, k) for k in (
        'n_steps', 'update_epochs', 'minibatch_size', 'gamma', 'gae_lambda',
        'learning_rate', 'clip_range', 'vf_coef', 'ent_coef', 'max_grad_norm',
        'min_log_std', 'max_log_std', 'target_kl', 'time_limit_bootstrap')}
    full_metrics = ('v322/recent_hit',) + (('v322/initial_group', 'v322/initial_y_m')
                                         if grouped_conditions else ())
    collect, update = ppo.make_train_fns(env=env, **kwargs,
        compact_rollout_metrics=True, full_rollout_metric_names=full_metrics,
        conditional_reset_done=True, apply_every_minibatch=True,
        counterfactual_replay_obs=jnp.asarray(replay_obs),
        counterfactual_replay_actions=jnp.asarray(replay_actions),
        counterfactual_supervision_coef=.05, counterfactual_vxy_weight_mode='uniform',
        reference_params=reference_params,
        actor_anchor_replay_obs=jnp.asarray(np.load(args.actor_anchor_replay_obs)),
        actor_anchor_kl_coef=args.actor_anchor_kl_coef,
        actor_anchor_replay_kl_coef=args.actor_anchor_replay_kl_coef,
        actor_anchor_replay_max_kl=args.actor_anchor_replay_max_kl,
        residual_l2_coef=args.residual_l2_coef,
        failure_focus_hit_threshold=args.failure_focus_hit_threshold,
        failure_focus_weight=args.failure_focus_weight,
        failure_focus_tail_steps=args.failure_focus_tail_steps,
        failure_focus_balanced_mass=args.failure_focus_balanced_mass)
    solver_execution_optimization = None
    if opts.mjx_line_search_schedule != 'legacy':
        if env.model.opt.iterations != 100 or env.model.opt.ls_iterations != 50:
            raise ValueError('MJX schedule requires the original 100/50 solver caps')
        from mjx_line_search_schedule import make_scheduled_collect
        collect, solver_execution_optimization = make_scheduled_collect(collect)
        args.mjx_line_search_schedule = opts.mjx_line_search_schedule
    if opts.mjx_execution_mode != 'legacy':
        if env.model.opt.iterations != 100 or env.model.opt.ls_iterations != 50:
            raise ValueError('MJX execution requires the original 100/50 caps')
        from mjx_topology_execution import make_collect
        collect, solver_execution_optimization = make_collect(collect, 'project')
        args.mjx_execution_mode = opts.mjx_execution_mode
    packed_metric_reader = None
    if opts.packed_host_metrics:
        from packed_rollout_metrics import (
            EXECUTION_VERSION, make_packed_metric_reader,
        )
        packed_metric_reader = make_packed_metric_reader(
            prefix=ppo.COMPACT_METRIC_MEAN_PREFIX)
        args.packed_host_metrics = True
    contract = dict(profile=PROFILE, parent_sha256=PARENT_SHA256,
        xml_sha256=XML_SHA256, gpu_uuid=GPU_UUID, config_overrides=CONFIG_OVERRIDES,
        reward_preset=opts.reward_preset,
        loss_preset=loss_preset,
        loss_overrides=LOSS_PRESETS[loss_preset],
        loss_change_from=payload.get('v322_contract', {}).get('loss_preset', 'baseline'),
        reward_multipliers=REWARD_PRESETS[opts.reward_preset],
        startup_preparation_settings=(STARTUP_PREPARATION_SETTINGS
            if opts.reward_preset in ('stability_launch', 'contact_quality', 'balanced_contact', 'startup_release', 'startup_transfer') else None),
        contact_quality_settings=(CONTACT_QUALITY_SETTINGS
            if opts.reward_preset in ('contact_quality', 'balanced_contact', 'startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else None),
        startup_release_settings=(STARTUP_RELEASE_SETTINGS
            if opts.reward_preset in ('startup_release', 'startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else None),
        startup_transfer_settings=(STARTUP_TRANSFER_SETTINGS
            if opts.reward_preset in ('startup_transfer', 'native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else None),
        native_preparation_restored=(opts.reward_preset in ('native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing')),
        startup_lead_settings=(STARTUP_LEAD_SETTINGS if opts.reward_preset in ('startup_lead', 'shared_prepare', 'shared_early_prepare') else None),
        shared_preparation_settings=(SHARED_EARLY_SETTINGS if opts.reward_preset in ('shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else SHARED_PREPARATION_SETTINGS if opts.reward_preset == 'shared_prepare' else None),
        startup_diagnostic_seconds=(6. if opts.reward_preset in ('native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else None),
        native_third_focus_settings=(NATIVE_THIRD_FOCUS_SETTINGS
            if opts.reward_preset == 'native_third_focus' else None),
        native_reference_preparation_settings=(NATIVE_REFERENCE_GAIN6_SETTINGS
            if opts.reward_preset in ('native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing') else NATIVE_REFERENCE_PREPARATION_SETTINGS
            if opts.reward_preset == 'native_reference_prepare' else None),
        reward_change_from=payload.get('v322_contract', {}).get('reward_preset', 'baseline'),
        survival_reward='40*dt after a confirmed hit no older than 1.2s; physical terminal step zero',
        max_hit_gap_s=MAX_HIT_GAP_S, course=[asdict(s) for s in course],
        gate_preset=gate_preset, eval_lesson=asdict(eval_lesson),
        condition_gate_settings=group_settings,
        condition_gate_required_by_stage=([required_conditions(i) for i in range(4)]
            if grouped_conditions else None),
        condition_gate_calibration=('original M11 native sampled6s; fixed Wilson95 '
            'lower rounded down5pp, floor50%; nominal95% length/93% survival'
            if grouped_conditions else None),
        condition_gate_evidence_sha256=('62a9c5c2ac3865bd6fd6a9b903cc228ddf2da5077789f8252571d4b6f3b3cc0c'
            if grouped_conditions else None),
        condition_labels=('native initial group and saved initial world-Y offset; '
            'nominal drift failures remain nominal; every failure counted'
            if grouped_conditions else None),
        final_acceptance_reset_overrides=(dict(v225_nominal_fraction=1.,
            v225_mirror_fraction=0., racket_launch_xy_jitter=.001)
            if grouped_conditions else None),
        no_new_dr=True, actor_critic_action=[72, 393, 7],
        ppo=expected, every_minibatch_applied=True,
        independent_eval=('final stage only: 8 mean + 8 sampled,60 seconds, '
            'centered +/-1mm reset control; native physics, no reset after failure'
            if grouped_conditions else
            'final stage only: 8 mean + 8 sampled, 60 seconds, M11 domain, no reset'),
        validation_budget_s=EVAL_WALL_BUDGET_S, validation_retry_updates=EVAL_RETRY_UPDATES,
        graduation='S1-S3 rolling training gates; S4 1200s mixed-policy endpoints plus 60s frozen acceptance; not a frozen 1200s proof',
        metrics='rolling32 event-weighted quality; deadline endpoints rolling4096',
        actor_anchor_origin='exact serialized M11 S17 stage_actor_anchor_reference_params',
        storage='W&B online, tmpfs spool/logs, every100 updates checkpoint; every5 new saves keep best; all converged retained',
        parent_step=int(parent['step']), source_checkpoint=str(args.resume_from),
        source_sha256={str(p.relative_to(opts.runtime)): sha(p)
                       for p in (opts.runtime / 'source').glob('*.py')},
        frozen_parent_source=str(snapshot),
        frozen_parent_source_sha256={p.name: sha(p) for p in snapshot.glob('*.py')},
        replay_sha256={str(getattr(args,k)):v for k,v in REPLAY_HASHES.items()})
    if opts.reward_preset == 'native_return_prepare':
        contract['third_return_preparation_settings'] = THIRD_RETURN_PREPARATION_SETTINGS
    if opts.reward_preset in ('projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        contract['projected_return_preparation_settings'] = PROJECTED_RETURN_PREPARATION_SETTINGS
    if opts.reward_preset in ('projected_return_acquisition', 'weak_release_outcome', 'reference_timing'):
        contract['acquisition_xy_settings'] = ACQUISITION_XY_SETTINGS
    if opts.reward_preset in ('weak_release_outcome', 'reference_timing'):
        contract['weak_release_outcome_settings'] = WEAK_RELEASE_OUTCOME_SETTINGS
    if opts.reward_preset == 'reference_timing':
        contract['reference_timing_settings'] = REFERENCE_TIMING_SETTINGS
        contract['reference_velocity_source'] = 'exact site FK difference of published reference joints using the same current base; reward only'
    if opts.reward_preset in ('late_cycle_repair','late_center_precision'):
        from fixed_plane_v322_late_cycle_reward import LATE_CYCLE_SETTINGS, LATE_CENTER_PRECISION_SETTINGS
        contract['late_cycle_settings'] = (LATE_CENTER_PRECISION_SETTINGS
            if opts.reward_preset == 'late_center_precision' else LATE_CYCLE_SETTINGS)
        contract['inherited_reward_preset'] = 'reference_timing'
        contract['reference_timing_settings'] = REFERENCE_TIMING_SETTINGS
        for key in ('contact_quality_settings', 'startup_release_settings',
                    'startup_transfer_settings', 'native_preparation_restored',
                    'shared_preparation_settings', 'startup_diagnostic_seconds',
                    'native_reference_preparation_settings',
                    'projected_return_preparation_settings', 'acquisition_xy_settings',
                    'weak_release_outcome_settings', 'reference_velocity_source'):
            contract[key] = payload['v322_contract'].get(key)
    if opts.quality_repair:
        contract['quality_repair'] = inherited_long
        contract['independent_eval'] = ('32cases/60s; paired center controls and '
            'native in-flight recovery; 8mean+8sampled per arm; original numerical gates')
        contract['graduation'] = ('inherited u31201 long qualification; original32-update '
            'quality/16hold plus original32-case60s mean/sampled runtime gates; '
            'no new1200s clock; old all-five/stage flags remain unchanged')
    if opts.packed_host_metrics:
        contract['execution_optimization'] = dict(
            version=EXECUTION_VERSION, scope='host metric reduction/transfer only',
            all_metric_channels_retained=True, default_enabled=False)
    if solver_execution_optimization is not None:
        contract['solver_execution_optimization'] = solver_execution_optimization
    audit_path = opts.runtime / 'contract.json'
    audit_path.write_text(json.dumps(contract, indent=2))
    wandb_run = None
    if not opts.smoke:
        import wandb
        wandb_run = wandb.init(project='pingpong-mjx', id=opts.run_id,
            name=opts.run_id, resume='never', mode='online', dir=str(opts.runtime),
            config={**contract, 'env_cfg': asdict(cfg), 'seed': opts.seed,
                    'command': sys.argv, 'devices': [str(d) for d in jax.devices()]},
            settings=wandb.Settings(x_disable_stats=True),
            tags=['v322', 'M11', '1200s', 'gpu1', 'no-new-DR'])
        print(f'[V322] W&B {wandb_run.url}', flush=True)
        wandb_run.define_metric('update')
        wandb_run.define_metric('*', step_metric='update')
        artifact = wandb.Artifact(f'{opts.run_id}-source', type='source')
        artifact.add_dir(str(opts.runtime / 'source'))
        artifact.add_file(str(audit_path))
        artifact.add_file(str(opts.runtime / 'command.sh'))
        wandb_run.log_artifact(artifact)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    validation_pause = [False]
    quality_admission = None
    if opts.quality_repair:
        from fixed_plane_v322_quality_repair import QualityRepairAdmission
        quality_admission = QualityRepairAdmission(update_idx, global_step)
        signal.signal(signal.SIGUSR1, lambda *_: validation_pause.__setitem__(0, True))
        signal.signal(signal.SIGUSR2, lambda *_: validation_pause.__setitem__(0, False))
    quality_window = deque(maxlen=32)
    endpoints = [EndpointWindow(s.seconds, env.dt) for s in course]
    condition_endpoints = ([ConditionEndpointWindow(s.seconds, env.dt) for s in course]
                           if grouped_conditions else None)
    # Read-only startup telemetry: no short training episode, reset, loss
    # weighting, deadline gate or advancement rule is changed.
    startup_diagnostic = (ConditionEndpointWindow(6., env.dt)
        if grouped_conditions and opts.reward_preset in ('native_prepare', 'startup_lead', 'shared_prepare', 'shared_early_prepare', 'native_third_focus', 'native_reference_prepare', 'native_reference_gain6', 'native_return_prepare', 'projected_return_prepare', 'projected_return_acquisition', 'weak_release_outcome', 'reference_timing', 'late_cycle_repair','late_center_precision') else None)
    retention = CheckpointRetention(opts.save_dir)
    if not opts.smoke:
        opts.save_dir.mkdir(parents=True, exist_ok=False)
    row = {}
    best_eval = None
    eval_env = None
    evaluator = None
    last_eval_update = update_idx-EVAL_RETRY_UPDATES
    validation_wall_total_s = 0.
    stage_entry_update = update_idx
    full_start_update = update_idx
    hold = 0

    def save(converged=False, *, validation=False):
        if opts.smoke:
            return
        lesson = course[stage_index]
        args.advance_eval_steps = round(EVAL_SECONDS / env.dt)
        score = (stage_index, float(quality_passed(row.get('quality', {}))),
                 row.get('endpoint', {}).get('survival', 0.),
                 row.get('endpoint', {}).get('len_fraction', 0.),
                 row.get('ongoing_s_mean', 0.))
        if grouped_conditions:
            conditions = row.get('conditions', {})
            primary = conditions.get('nominal', {})
            margin = min(min(conditions.get(name, {}).get('survival', 0.) / group_settings[name]['min_survival'],
                             conditions.get(name, {}).get('len_fraction', 0.) / group_settings[name]['min_length_fraction'])
                         for name in required_conditions(stage_index))
            score = (stage_index, float(quality_passed(row.get('quality', {}))),
                     margin, primary.get('survival', 0.),
                     primary.get('len_fraction', 0.), row.get('ongoing_s_mean', 0.))
        path = opts.save_dir / (f'converged_s{stage_index+1}_{int(lesson.seconds)}s.pkl'
            if converged else f'periodic_u{update_idx:08d}.pkl')
        if validation:
            path = opts.runtime / f'quality_validation_source_u{update_idx}.pkl'
        if converged and path.exists():
            raise FileExistsError(path)
        ppo.save_checkpoint(path, train_state, args, env, global_step,
            extra=dict(stage_index=stage_index, stage_name=lesson.name,
                stage_update=update_idx-stage_entry_update, global_update=update_idx,
                stage_converged=converged, v322_contract=contract,
                stage_condition_scope=(required_conditions(stage_index)
                    if grouped_conditions else None),
                stage_convergence_evidence=('rolling_training_endpoints_and_60s_frozen_acceptance'
                    if stage_index == len(course)-1 else 'rolling_training_endpoints_only'),
                selection_score=score, selection_kind='mixed_policy_training_rank',
                last_summary=row, independent_evaluation=best_eval,
                stage_actor_anchor_reference_params=jax.device_get(reference_params)))
        if not converged and not validation:
            retention.register(path, score)
        print(f'[V322 save] {path.name}; periodic_kept={len(retention.saved)}', flush=True)
        return path

    def log(metrics):
        if wandb_run:
            def flatten(values, prefix=''):
                result = {}
                for key, value in values.items():
                    name = f'{prefix}/{key}' if prefix else key
                    if isinstance(value, dict):
                        result.update(flatten(value, name))
                    else:
                        result[name] = value
                return result
            # W&B transport steps are independent from true optimizer steps:
            # evaluation may add a second record at the same training update.
            wandb_run.log({'global_step': global_step, 'update': update_idx,
                           **flatten(metrics)})

    print(f'[V322] CUDA={jax.devices()} n_envs=1024 n_steps=128 '
          f'horizon=1200s obs/critic/action=72/393/7 resume_step={global_step} '
          f'Adam_t={int(train_state.opt.t)} reward_preset={opts.reward_preset} '
          f'loss_preset={loss_preset} replay_KL_coef={args.actor_anchor_replay_kl_coef:g} '
          f'gate_preset={gate_preset} '
          f'smoke={opts.smoke}', flush=True)
    # Avoid retaining reset/constructor executables alongside rollout compile.
    jax.block_until_ready((runner, train_state))
    release_caches()
    try:
        while not stop[0]:
            lesson = course[stage_index]
            t0 = time.monotonic()
            previous_adam_t = int(train_state.opt.t)
            runner, transitions = collect(train_state.params, runner)
            train_state, losses = update(train_state, runner, transitions)
            loss = {k: float(v) for k, v in jax.device_get(losses).items()}
            if not all(np.isfinite(v) for v in loss.values()):
                raise FloatingPointError(f'Nonfinite PPO diagnostics {loss}')
            if int(train_state.opt.t) - previous_adam_t != 32:
                raise RuntimeError('An optimizer minibatch was skipped')
            update_idx += 1
            global_step += 1024 * 128
            done = np.asarray(transitions.done, dtype=bool)
            lengths = np.asarray(transitions.episode_length)
            hits = np.asarray(transitions.hit_count)
            recent = np.asarray(transitions.metrics['v322/recent_hit']) > .5
            terminated = np.asarray(transitions.terminated, dtype=bool)
            for endpoint in endpoints:
                endpoint.add(lengths, hits, done, recent, terminated=terminated)
            if grouped_conditions:
                initial_groups = np.asarray(transitions.metrics['v322/initial_group'])
                initial_y_m = np.asarray(transitions.metrics['v322/initial_y_m'])
                for point in condition_endpoints:
                    point.add(lengths, hits, done, recent, initial_groups, initial_y_m,
                              terminated=terminated)
                if startup_diagnostic is not None:
                    startup_diagnostic.add(lengths, hits, done, recent, initial_groups,
                                           initial_y_m, terminated=terminated)
            if packed_metric_reader is None:
                means = {k.removeprefix(ppo.COMPACT_METRIC_MEAN_PREFIX): float(v)
                    for k, v in jax.device_get(jax.tree.map(jnp.mean, transitions.metrics)).items()
                    if k.startswith(ppo.COMPACT_METRIC_MEAN_PREFIX)}
            else:
                means = packed_metric_reader(transitions.metrics)
            if means['v322/finite'] < 1.:
                raise FloatingPointError('Nonfinite simulation observation/reward')
            missing = set(QUALITY_FIELDS) - set(means)
            if missing:
                raise ValueError(f'Missing quality metric channels: {missing}')
            quality_window.append(means)
            quality = quality_summary({k: sum(m.get(k, 0.) for m in quality_window)
                * 1024 * 128 for k in QUALITY_FIELDS})
            endpoint = endpoints[stage_index].summary()
            age = np.asarray(runner.running_length) * env.dt
            row = dict(update=update_idx, stage=stage_index+1, quality=quality,
                endpoint=endpoint, ongoing_s_mean=float(age.mean()),
                ongoing_s_max=float(age.max()), completed_episodes=int(done.sum()))
            conditions = (condition_endpoints[stage_index].summary()
                          if grouped_conditions else None)
            if grouped_conditions:
                row['conditions'] = conditions
                if sum(summary['n'] for summary in conditions.values()) != endpoint['n']:
                    raise RuntimeError('Condition accounting differs from overall outcomes')
            flat = {**loss, 'update': update_idx, 'stage': stage_index+1,
                'episode_limit_s': 1200., 'episodes/completed': int(done.sum()),
                'ongoing/mean_s': float(age.mean()), 'ongoing/max_s': float(age.max()),
                'sps': 131072 / (time.monotonic()-t0),
                **{f'quality/{k}': v for k, v in quality.items()},
                **{f'endpoint/{k}': v for k, v in endpoint.items()},
                **{f'metric/{k}': v for k, v in means.items()}}
            if grouped_conditions:
                flat.update({f'condition/{name}/{key}': value
                    for name, summary in conditions.items() for key, value in summary.items()})
                flat['condition/all_gates_passed'] = int(condition_gates_passed(conditions, group_settings))
            if startup_diagnostic is not None:
                startup_summary = startup_diagnostic.summary()
                flat.update({f'startup6/{name}/{key}': value
                    for name, summary in startup_summary.items()
                    for key, value in summary.items()})
            if done.any():
                flat.update({'completed/mean_s': float(lengths[done].mean() * env.dt),
                    'completed/mean_hits': float(hits[done].mean()),
                    'completed/return': float(np.asarray(transitions.episode_return)[done].mean())})
                flat.update({f'completed/hit{k}_rate': float((hits[done] >= k).mean())
                             for k in (1, 3, 12)})
            for i, point in enumerate(endpoints):
                flat.update({f'survival_{int(course[i].seconds)}s/{k}': v
                             for k, v in point.summary().items()})
            enough_time = (update_idx-full_start_update)*.64 >= lesson.seconds
            endpoint_pass = (condition_gates_passed(conditions, group_settings,
                                required=required_conditions(stage_index))
                if grouped_conditions else
                endpoint.get('survival', 0.) >= lesson.min_survival
                and endpoint.get('len_fraction', 0.) >= lesson.min_length_fraction)
            eligible = (len(quality_window) == 32 and endpoint['n'] >= 1024
                and enough_time and quality_passed(quality)
                and endpoint_pass
                and endpoint.get('hits_mean', 0.) / lesson.seconds >= 1.2)
            hold = hold+1 if eligible else 0
            flat['convergence/hold'] = hold
            flat['convergence/eligible'] = int(eligible)
            if quality_admission is not None:
                repair_report = quality_admission.add(update=update_idx,
                    step=global_step, quality=quality, quality_window=len(quality_window),
                    loss=loss, finite=means['v322/finite'] == 1.)
                row['quality_repair'] = repair_report
                flat.update({f'quality_repair/{k}': v for k,v in repair_report.items()
                    if isinstance(v, (bool,int,float))})
                (opts.runtime/'quality_repair_latest.json').write_text(json.dumps(repair_report,indent=2))
            log(flat)
            def display_quality(name):
                return f'{quality[name]:.4f}' if name in quality else 'pending'
            shown_endpoint = conditions['nominal'] if grouped_conditions else endpoint
            shown_label = 'nominal_' if grouped_conditions else ''
            overall_context = (f' overall_P={endpoint.get("survival", "pending")} '
                f'overall_len={endpoint.get("len_fraction", "pending")}'
                if grouped_conditions else '')
            repair_context = (f' repair_quality_hold={repair_report["hold"]}/16 '
                f'center_mm={quality.get("center_m", float("nan"))*1000:.3f} '
                f'inband={quality.get("in_band", float("nan")):.4%} '
                'long_qualification=inherited_u31201'
                if opts.quality_repair else '')
            print(f'[V322] S{stage_index+1}/{len(course)} u={update_idx} '
                f'episodes={int(done.sum())} alive_s={age.mean():.2f}/{age.max():.2f} '
                f'{shown_label}P({lesson.seconds:g}s)={shown_endpoint.get("survival", "pending")} '
                f'{shown_label}len={shown_endpoint.get("len_fraction", "pending")} '
                f'apex={display_quality("apex_m")} '
                f'rec_vxy={display_quality("recurrent_vxy_m_s")} '
                f'rms={display_quality("racket_vxy_rms_m_s")} '
                f'kl={loss["approx_kl"]:.5f} hold={hold}/16 '
                f'sps={flat["sps"]:.0f} Adam+32{overall_context}{repair_context}', flush=True)
            if grouped_conditions and update_idx % 10 == 0:
                print('[V322 conditions] ' + ' '.join(
                    f'{name}:n={summary["n"]},P={summary.get("survival", 0.):.3f},'
                    f'L={summary.get("len_fraction", 0.):.3f}'
                    for name, summary in conditions.items()), flush=True)
                if startup_diagnostic is not None:
                    print('[V322 startup6 diagnostic] ' + ' '.join(
                        f'{name}:n={summary["n"]},P={summary.get("survival", 0.):.3f}'
                        for name, summary in startup_summary.items()), flush=True)
            del transitions
            if update_idx % 100 == 0:
                save()
            if opts.smoke and update_idx-full_start_update >= 3:
                (opts.runtime / 'smoke_result.json').write_text(json.dumps(
                    dict(passed=True, updates=3, latest=row, losses=loss,
                         devices=[str(d) for d in jax.devices()],
                         actor_dim=72, critic_dim=393, envs=1024,
                         episode_steps=env.max_steps, adam_applied_per_update=32), indent=2))
                break
            if not opts.quality_repair and hold >= 16 and stage_index < len(course)-1:
                best_eval = {'not_required': True, 'evidence': 'rolling_training_endpoints',
                             'frozen_test': False}
                save(converged=True)
                stage_index += 1
                stage_entry_update = update_idx
                hold = 0
            elif not opts.quality_repair and hold >= 16 and should_validate(stage_index, update_idx, last_eval_update):
                # Validation has a separate batch size and compiled program;
                # release training compilation caches before building it.
                jax.block_until_ready((runner, train_state))
                release_caches()
                if eval_env is None:
                    eval_cfg = centered_acceptance_config(cfg) if grouped_conditions else cfg
                    eval_env = env_class(xml, n_envs=EVAL_ENVS, cfg=eval_cfg)
                    evaluator = build_evaluator(eval_env, ppo, args.min_log_std, args.max_log_std)
                validation_started = time.monotonic()
                best_eval = evaluate(eval_env, evaluator, train_state.params,
                    eval_lesson, opts.seed + 100000 + update_idx, stop, log)
                validation_wall_s = time.monotonic()-validation_started
                validation_wall_total_s += validation_wall_s
                last_eval_update = update_idx
                if best_eval is not None:
                    best_eval.update(wall_s=validation_wall_s,
                                     cumulative_wall_s=validation_wall_total_s,
                                     n_envs=EVAL_ENVS)
                    log({'validation': best_eval})
                print(f'[V322 eval result] {json.dumps(best_eval)}', flush=True)
                release_caches()
                if best_eval and best_eval['passed']:
                    save(converged=True)
                    stage_index += 1
                    if stage_index == len(course):
                        print('[V322] Four training duration gates and final 60s frozen acceptance passed.', flush=True)
                        break
                    stage_entry_update = update_idx
                    hold = 0
                    # Same training runner, weights, Adam and RNG continue.
            temperature = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True))
            if temperature >= 78:
                print('[V322] temperature guard requested safe stop', flush=True)
                stop[0] = True
            if (opts.quality_repair and validation_pause[0] and not stop[0]
                    and repair_report['ready_for_frozen_evaluation']):
                # Pause only after all32 minibatches. Preserve the entire live
                # runner, RNG and optimizer; failed validation resumes them.
                validation_path = save(validation=True)
                jax.block_until_ready((runner, train_state))
                release_caches()
                marker = dict(paused=True, pid=os.getpid(), update=update_idx,
                    step=global_step, Adam_t=int(train_state.opt.t),
                    source=str(validation_path), sha256=sha(validation_path),
                    quality_repair=repair_report)
                (opts.runtime/'quality_validation_ready.json').write_text(json.dumps(marker,indent=2))
                print(f'[V322 quality validation pause] u={update_idx}; live runner/Adam/RNG retained',flush=True)
                while validation_pause[0] and not stop[0]:
                    time.sleep(.5)
                marker['paused'] = False
                (opts.runtime/'quality_validation_ready.json').write_text(json.dumps(marker,indent=2))
        if stop[0] and stage_index < len(course):
            save()
    finally:
        retention.prune()
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()
