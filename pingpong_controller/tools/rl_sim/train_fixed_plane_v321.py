#!/usr/bin/env python3
"""Opt-in long-episode continuation of the authenticated M04 simulator.

Use run_gpu1_fixed_plane_v321_long_episode.sh. No step/update budget is applied
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

from fixed_plane_v321_long_episode import (
    COURSE, PROFILE, PARENT_SHA256, XML_SHA256, GPU_UUID, MAX_HIT_GAP_S,
    CONFIG_OVERRIDES, QUALITY_FIELDS, EndpointWindow, CheckpointRetention,
    make_config, survival_reward, quality_summary, quality_passed,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configuration_json(value) -> str:
    """Canonicalize nested checkpoint dataclasses without dropping fields."""
    return json.dumps(value, sort_keys=True, default=lambda x:
        asdict(x) if is_dataclass(x) else np.asarray(x).tolist())


def load_stack(repo: Path, ppo_source: Path):
    """Use immutable M04 dynamics plus the explicit accept-every-minibatch PPO."""
    run = repo / 'pingpong_controller/outputs/rl_sim/fplc_v172_gpu1_quality_gates_formal_20261009_r1'
    snapshot = run / 'source_snapshot/pingpong_controller/tools/rl_sim'
    checkpoint = run / '10_fplc_v159_gpu1_s10_clean_commit.pkl'
    xml = snapshot / 'moz1_pd_1ms_v98.xml'
    for path, expected in ((checkpoint, PARENT_SHA256), (xml, XML_SHA256)):
        if sha(path) != expected:
            raise ValueError(f'Authenticated M04 input changed: {path}')
    sys.path.insert(0, str(snapshot))
    spec = importlib.util.spec_from_file_location('train_juggle_mjx_ppo', ppo_source)
    ppo = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ppo
    spec.loader.exec_module(ppo)
    from validate_juggle_mjx_ppo import env_config_from_checkpoint, load_checkpoint, parse_args
    parent = load_checkpoint(checkpoint)
    cfg = env_config_from_checkpoint(parent, parse_args([]))
    restored = asdict(cfg)
    for key, value in parent['env_cfg'].items():
        if key not in restored or configuration_json(value) != configuration_json(restored[key]):
            raise ValueError(f'M04 saved config mismatch: {key}')
    if (parent['args']['curriculum_profile'] != 'fixed_plane_limit_cycle_qvel_rmp_gpu1_v172'
            or parent['stage_name'] != 'fplc_v159_gpu1_s10_clean_commit'
            or parent['stage_update'] != 314 or parent['step'] != 1334837248
            or (parent['obs_dim'], parent['critic_obs_dim'], parent['act_dim']) != (72, 393, 7)):
        raise ValueError('Wrong M04 identity')
    return ppo, parent, cfg, xml, snapshot


def environment_class(base):
    """Only adjust dense survival credit and expose small diagnostic channels."""
    import jax.numpy as jnp

    class LongEpisodeEnv(base):
        def step(self, state, action):
            after, obs, reward, done, metrics = super().step(state, action)
            age_s = after.step_count * self.dt - after.last_counted_hit_time
            alive_reward = survival_reward(after.hit_count, age_s,
                metrics['terminated'].astype(bool), self.dt, xp=jnp)
            old_reward = metrics['reward/fixed_contact_survival_reward']
            reward = reward + alive_reward - old_reward
            metrics['reward/fixed_contact_survival_reward'] = alive_reward
            metrics['reward/total'] = reward
            metrics['v321/recent_hit'] = (
                (after.hit_count > 0) & (age_s >= 0.) & (age_s <= MAX_HIT_GAP_S)
            ).astype(jnp.float32)
            recurrent = after.hit_count >= 2
            metrics['recurrent_hit_racket_vxy_sum'] = jnp.where(
                recurrent, metrics['hit_racket_vxy_sum'], 0.)
            metrics['recurrent_hit_racket_vxy_sq_sum'] = jnp.where(
                recurrent, metrics['hit_racket_vxy_sq_sum'], 0.)
            metrics['v321/finite'] = (jnp.all(jnp.isfinite(obs), axis=-1)
                & jnp.isfinite(reward)).astype(jnp.float32)
            return after, obs, reward, done, metrics

    return LongEpisodeEnv


def freeze_lanes(old, new, active, jax, jnp):
    """Keep first-terminal lane states; preserve native shared scalar clocks."""
    return jax.tree.map(lambda a, b: b if b.ndim == 0 else jnp.where(
        active.reshape((len(active),) + (1,) * (b.ndim - 1)), b, a), old, new)


def build_evaluator(env, ppo, min_log_std, max_log_std):
    """64 deterministic and 64 stochastic lanes, one episode per seed."""
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
            finite = metrics['v321/finite'] > .5
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


def evaluate(env, evaluator, params, lesson, seed, stop, log):
    """Freeze weights for a fresh full-window test; no reset after failure."""
    import jax
    import jax.numpy as jnp

    rng, reset_key = jax.random.split(jax.random.PRNGKey(seed))
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, env.n_envs))
    active = jnp.ones(env.n_envs, bool)
    sums = {}
    started = time.monotonic()
    for second in range(1, int(lesson.seconds) + 1):
        if stop[0]:
            return None
        (state, obs, active, rng), packet = evaluator(params, state, obs, active, rng)
        packet = jax.device_get(packet)
        for name, value in packet.items():
            sums[name] = sums.get(name, np.zeros(env.n_envs, dtype=np.float64)) + value
        if second % 10 == 0 or second == int(lesson.seconds):
            print(f'[V321 eval] {lesson.name} t={second}/{lesson.seconds:g}s '
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
    parser.add_argument('--seed', type=int, default=2026102807)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', type=Path)
    opts = parser.parse_args()
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
    args.resume_from = opts.resume or snapshot.parents[3] / '10_fplc_v159_gpu1_s10_clean_commit.pkl'
    args.apply_every_minibatch = True
    args.total_update_limit = None
    args.max_stage_updates = -1
    args.max_stages = len(COURSE)
    args.convergence_hold_updates = 16
    args.convergence_min_episodes = 1024
    args.save_every_updates = 100
    args.archive_every_updates = 0
    args.training_entrypoint = 'train_fixed_plane_v321.py'
    args.advance_eval_n_envs = 128
    args.advance_eval_min_episodes = 128
    args.advance_retention_stage_name = None
    args.advance_eval_stochastic = True
    args.conditional_reset_done = True
    expected = dict(n_envs=1024, n_steps=128, minibatch_size=16384,
        update_epochs=4, learning_rate=1.5e-5, gamma=.9995, gae_lambda=.99,
        clip_range=.12, target_kl=.006, ent_coef=.001, vf_coef=.05,
        min_log_std=-3., max_grad_norm=.5, counterfactual_supervision_coef=.05)
    for name, value in expected.items():
        if getattr(args, name) != value:
            raise ValueError(f'M04 PPO source mismatch {name}')
    payload = parent
    stage_index = 0
    update_idx = 0
    if opts.resume:
        from validate_juggle_mjx_ppo import load_checkpoint
        payload = load_checkpoint(opts.resume)
        if (payload['args']['curriculum_profile'] != PROFILE
                or payload.get('v321_contract', {}).get('parent_sha256') != PARENT_SHA256
                or payload.get('v321_contract', {}).get('config_overrides') != CONFIG_OVERRIDES
                or (payload['obs_dim'], payload['critic_obs_dim'], payload['act_dim']) != (72, 393, 7)):
            raise ValueError('Only an exact V321 checkpoint can resume this course')
        if configuration_json(payload['env_cfg']) != configuration_json(asdict(cfg)):
            raise ValueError('V321 resume environment configuration changed')
        for name, value in expected.items():
            if payload['args'].get(name) != value:
                raise ValueError(f'V321 resume PPO mismatch: {name}')
        stage_index = int(payload['stage_index']) + int(payload.get('stage_converged', False))
        update_idx = int(payload['global_update'])
        if stage_index >= len(COURSE):
            raise ValueError('This V321 checkpoint already completed all lessons')
    params = jax.tree.map(jnp.asarray, payload['params'])
    train_state = ppo.TrainState(params, jax.tree.map(jnp.asarray, payload['opt']))
    global_step = int(payload['step'])
    env_class = environment_class(MjxJuggleEnv)
    env = env_class(xml, n_envs=1024, cfg=cfg)
    if (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) != (72, 393, 7, 240000):
        raise ValueError('V321 layout/horizon mismatch')
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
        raise ValueError('Wrong original M04 BC replay dimensions')
    kwargs = {k: getattr(args, k) for k in (
        'n_steps', 'update_epochs', 'minibatch_size', 'gamma', 'gae_lambda',
        'learning_rate', 'clip_range', 'vf_coef', 'ent_coef', 'max_grad_norm',
        'min_log_std', 'max_log_std', 'target_kl', 'time_limit_bootstrap')}
    collect, update = ppo.make_train_fns(env=env, **kwargs,
        compact_rollout_metrics=True, full_rollout_metric_names=('v321/recent_hit',),
        conditional_reset_done=True, apply_every_minibatch=True,
        counterfactual_replay_obs=jnp.asarray(replay_obs),
        counterfactual_replay_actions=jnp.asarray(replay_actions),
        counterfactual_supervision_coef=.05, counterfactual_vxy_weight_mode='uniform')
    contract = dict(profile=PROFILE, parent_sha256=PARENT_SHA256,
        xml_sha256=XML_SHA256, gpu_uuid=GPU_UUID, config_overrides=CONFIG_OVERRIDES,
        survival_reward='40*dt after a confirmed hit no older than 1.2s; physical terminal step zero',
        max_hit_gap_s=MAX_HIT_GAP_S, course=[asdict(s) for s in COURSE],
        no_new_dr=True, actor_critic_action=[72, 393, 7],
        ppo=expected, every_minibatch_applied=True, independent_eval='64 mean + 64 sampled, no reset, original M04 domain',
        metrics='rolling32 event-weighted quality; deadline endpoints rolling4096; frozen independent validation',
        storage='W&B online, tmpfs spool/logs, every100 updates checkpoint; every5 new saves keep best; all converged retained',
        parent_step=int(parent['step']), source_checkpoint=str(args.resume_from),
        source_sha256={str(p.relative_to(opts.runtime)): sha(p)
                       for p in (opts.runtime / 'source').glob('*.py')},
        frozen_parent_source=str(snapshot),
        frozen_parent_source_sha256={p.name: sha(p) for p in snapshot.glob('*.py')},
        replay_sha256={str(p): sha(Path(p)) for p in
                      (args.counterfactual_replay_obs, args.counterfactual_replay_actions)})
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
            tags=['v321', 'M04', '1200s', 'gpu1', 'no-new-DR'])
        print(f'[V321] W&B {wandb_run.url}', flush=True)
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
    quality_window = deque(maxlen=32)
    endpoints = [EndpointWindow(s.seconds, env.dt) for s in COURSE]
    retention = CheckpointRetention(opts.save_dir)
    if not opts.smoke:
        opts.save_dir.mkdir(parents=True, exist_ok=False)
    row = {}
    best_eval = None
    eval_env = None
    evaluator = None
    last_eval_update = update_idx
    stage_entry_update = update_idx
    full_start_update = update_idx
    hold = 0

    def save(converged=False):
        if opts.smoke:
            return
        lesson = COURSE[stage_index]
        args.advance_eval_steps = round(lesson.seconds / env.dt)
        score = (stage_index, float(quality_passed(row.get('quality', {}))),
                 row.get('endpoint', {}).get('survival', 0.),
                 row.get('endpoint', {}).get('len_fraction', 0.),
                 row.get('ongoing_s_mean', 0.))
        path = opts.save_dir / (f'converged_s{stage_index+1}_{int(lesson.seconds)}s.pkl'
            if converged else f'periodic_u{update_idx:08d}.pkl')
        if converged and path.exists():
            raise FileExistsError(path)
        ppo.save_checkpoint(path, train_state, args, env, global_step,
            extra=dict(stage_index=stage_index, stage_name=lesson.name,
                stage_update=update_idx-stage_entry_update, global_update=update_idx,
                stage_converged=converged, v321_contract=contract,
                selection_score=score, selection_kind='mixed_policy_training_rank',
                last_summary=row, independent_evaluation=best_eval))
        if not converged:
            retention.register(path, score)
        print(f'[V321 save] {path.name}; periodic_kept={len(retention.saved)}', flush=True)

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

    print(f'[V321] CUDA={jax.devices()} n_envs=1024 n_steps=128 '
          f'horizon=1200s obs/critic/action=72/393/7 resume_step={global_step} '
          f'Adam_t={int(train_state.opt.t)} smoke={opts.smoke}', flush=True)
    # Avoid retaining reset/constructor executables alongside rollout compile.
    jax.block_until_ready((runner, train_state))
    release_caches()
    try:
        while not stop[0]:
            lesson = COURSE[stage_index]
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
            recent = np.asarray(transitions.metrics['v321/recent_hit']) > .5
            terminated = np.asarray(transitions.terminated, dtype=bool)
            for endpoint in endpoints:
                endpoint.add(lengths, hits, done, recent, terminated=terminated)
            means = {k.removeprefix(ppo.COMPACT_METRIC_MEAN_PREFIX): float(v)
                for k, v in jax.device_get(jax.tree.map(jnp.mean, transitions.metrics)).items()
                if k.startswith(ppo.COMPACT_METRIC_MEAN_PREFIX)}
            if means['v321/finite'] < 1.:
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
            flat = {**loss, 'update': update_idx, 'stage': stage_index+1,
                'episode_limit_s': 1200., 'episodes/completed': int(done.sum()),
                'ongoing/mean_s': float(age.mean()), 'ongoing/max_s': float(age.max()),
                'sps': 131072 / (time.monotonic()-t0),
                **{f'quality/{k}': v for k, v in quality.items()},
                **{f'endpoint/{k}': v for k, v in endpoint.items()},
                **{f'metric/{k}': v for k, v in means.items()}}
            if done.any():
                flat.update({'completed/mean_s': float(lengths[done].mean() * env.dt),
                    'completed/mean_hits': float(hits[done].mean()),
                    'completed/return': float(np.asarray(transitions.episode_return)[done].mean())})
                flat.update({f'completed/hit{k}_rate': float((hits[done] >= k).mean())
                             for k in (1, 3, 12)})
            for i, point in enumerate(endpoints):
                flat.update({f'survival_{int(COURSE[i].seconds)}s/{k}': v
                             for k, v in point.summary().items()})
            enough_time = (update_idx-full_start_update)*.64 >= lesson.seconds
            eligible = (len(quality_window) == 32 and endpoint['n'] >= 1024
                and enough_time and quality_passed(quality)
                and endpoint.get('survival', 0.) >= lesson.min_survival
                and endpoint.get('len_fraction', 0.) >= lesson.min_length_fraction
                and endpoint.get('hits_mean', 0.) / lesson.seconds >= 1.2)
            hold = hold+1 if eligible else 0
            flat['convergence/hold'] = hold
            flat['convergence/eligible'] = int(eligible)
            log(flat)
            def display_quality(name):
                return f'{quality[name]:.4f}' if name in quality else 'pending'
            print(f'[V321] S{stage_index+1}/{len(COURSE)} u={update_idx} '
                f'episodes={int(done.sum())} alive_s={age.mean():.2f}/{age.max():.2f} '
                f'P({lesson.seconds:g}s)={endpoint.get("survival", "pending")} '
                f'len={endpoint.get("len_fraction", "pending")} '
                f'apex={display_quality("apex_m")} '
                f'rec_vxy={display_quality("recurrent_vxy_m_s")} '
                f'rms={display_quality("racket_vxy_rms_m_s")} '
                f'kl={loss["approx_kl"]:.5f} hold={hold}/16 '
                f'sps={flat["sps"]:.0f} Adam+32', flush=True)
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
            if hold >= 16 and update_idx-last_eval_update >= 200:
                # Validation has a separate batch size and compiled program;
                # release training compilation caches before building it.
                jax.block_until_ready((runner, train_state))
                release_caches()
                if eval_env is None:
                    eval_env = env_class(xml, n_envs=128, cfg=cfg)
                    evaluator = build_evaluator(eval_env, ppo, args.min_log_std, args.max_log_std)
                best_eval = evaluate(eval_env, evaluator, train_state.params,
                    lesson, opts.seed + 100000 + update_idx, stop, log)
                last_eval_update = update_idx
                if best_eval is not None:
                    log({'validation': best_eval})
                print(f'[V321 eval result] {json.dumps(best_eval)}', flush=True)
                release_caches()
                if best_eval and best_eval['passed']:
                    save(converged=True)
                    stage_index += 1
                    if stage_index == len(COURSE):
                        print('[V321] All four duration lessons passed frozen validation.', flush=True)
                        break
                    stage_entry_update = update_idx
                    hold = 0
                    # Same training runner, weights, Adam and RNG continue.
            temperature = float(subprocess.check_output(['nvidia-smi', '-i', GPU_UUID,
                '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True))
            if temperature >= 78:
                print('[V321] temperature guard requested safe stop', flush=True)
                stop[0] = True
        if stop[0] and stage_index < len(COURSE):
            save()
    finally:
        retention.prune()
        if wandb_run:
            wandb_run.finish()


if __name__ == '__main__':
    main()
