"""Frozen train/heldout source comparison; no optimizer or model updates."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import frozen_value, lambda_returns
from .artifacts import write_array_archive_unique, read_array_archive
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_layout
from .critic_distribution import failure_horizon, contact_phase, stratified_summary
from .critic_targets import checkpoint_critics_and_pool, lambda_mixture_oracle
from .imagined_rollout import window_batch
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_critic_fit import make_forward, check_window, take
from .train_pose_trial import phase_budget


def check_plan(cfg: dict) -> tuple:
    setting = cfg['critic_distribution_audit']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    source = json.loads(authenticate(repository_root(), plan['source_plan']).read_text())
    if (not setting['authorized'] or not plan['authorized'] or setting['default_enabled']
            or plan['training_version'] != 'WMJ-V1' or plan['horizon'] != 128
            or plan['microbatch'] != 32 or plan['max_phase_bytes'] != 2*2**20
            or any(plan[k] for k in ('actor_updates', 'critic_updates', 'world_updates', 'default_replacement'))
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or plan['modes'] != dict(deterministic=0., sampled=.03)):
        raise ValueError('Frozen audit scope differs from authorization')
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Prior implementation changed: '+name)
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key+'_sha256']:
            raise ValueError('Physical or control contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    train, held = source['train_indices'], source['heldout_indices']
    if (len(set(train)) != 128 or len(set(held)) != 32 or set(train) & set(held)
            or source['source_indices'] != train+held):
        raise ValueError('Registered train/heldout source partition changed')
    authenticate(repository_root(), plan['prerequisite'])
    authenticate(repository_root(), plan['protocol'])
    return plan, source


def window_values(window, networks: dict, source_rows: list, source_indices: list,
                  layout: dict) -> tuple:
    """Compute both Critics on exactly one shared frozen trajectory."""
    batch = window_batch(window)
    valid = np.asarray(batch.valid, bool)
    failed = np.asarray(window.transitions.terminated, bool)
    complete, remaining = failure_horizon(valid, failed)
    mc = np.asarray(lambda_returns(batch, jnp.zeros_like(batch.reward), lambda_value=1.))
    fields = {r['name']:r for r in layout['fields']}

    def feature(name):
        row = fields[name]
        return np.asarray(window.critic_inputs[..., row['start']:row['end']])*row['divisor']

    phase = contact_phase(feature('ball_contact')[..., 0] > .5,
        np.asarray(window.transitions.task_events['valid_release']), feature('ball_velocity')[..., 2])
    identity = np.broadcast_to(np.asarray(source_indices), valid.shape)
    regions = np.broadcast_to(np.asarray([r['region_id'] for r in source_rows]), valid.shape)
    ticks = np.broadcast_to(np.arange(len(valid))[:, None], valid.shape)
    rows = dict(source=identity[valid].astype(np.int16), region=regions[valid].astype(np.int16),
        tick=ticks[valid].astype(np.int16), complete=complete[valid], remaining=remaining[valid],
        phase=phase[valid], mc=np.where(complete, mc, 0.)[valid],
        reward=np.asarray(batch.reward)[valid], discount=np.asarray(batch.discount)[valid],
        trace=np.asarray(batch.trace_continue)[valid], failed=failed[valid])
    oracle_error = {}
    for arm, net in networks.items():
        online = np.asarray(frozen_value(net['critic'], window.critic_inputs))
        target = np.asarray(frozen_value(net['target_critic'], window.critic_inputs))
        following = frozen_value(net['target_critic'], window.transitions.next_critic_features)
        own = np.asarray(lambda_returns(batch, following, lambda_value=.99))
        oracle = lambda_mixture_oracle(batch, np.asarray(following), .99)
        np.testing.assert_allclose(own[valid], oracle[valid], atol=5e-5, rtol=1e-5)
        oracle_error[arm] = float(np.max(np.abs(own[valid]-oracle[valid])))
        rows.update({arm+'_online':online[valid], arm+'_target':target[valid],
                     arm+'_lambda':own[valid], arm+'_next_target':np.asarray(following)[valid]})
    sources = [dict(source_index=int(source), region_id=int(row['region_id']),
        root_seed_family=row['root_seed_family'], goal_xyz_m=row['first_goal_xyz_m'],
        valid_controls=int(valid[:, lane].sum()),
        failure_control=int(np.argmax(failed[:, lane])+1) if failed[:, lane].any() else None,
        releases=int(np.asarray(window.transitions.task_events['valid_release'])[:, lane].sum()))
        for lane, (source, row) in enumerate(zip(source_indices, source_rows))]
    return rows, dict(sources=sources, lambda_oracle_max_abs=oracle_error,
        trajectory_sha256=numeric_tree_digest((window.critic_inputs, window.transitions)))


def run(cfg, args, guard, report, plan, source):
    data = make_prepare()(cfg, args, guard, report, source)
    layout = course_critic_layout(data['initial'])
    if layout['dimension'] != 2602:
        raise ValueError('Critic input layout changed')
    report['critic_layout'] = layout
    networks = {}
    for arm, reference in plan['checkpoints'].items():
        loaded = checkpoint_critics_and_pool(reference, data['initial'])
        if (loaded['counters']['critic_updates'] != 1144 or loaded['counters']['actor_updates'] != 30
                or loaded['metadata']['arm'] != arm or loaded['metadata']['plan'] != plan['paired_plan']):
            raise ValueError('Unexpected frozen Critic endpoint')
        networks[arm] = {key:loaded[key] for key in ('critic', 'target_critic')}
        del loaded
    frozen = numeric_tree_digest((data['actor'], data['robot'], data['auxiliary'], networks))
    initial_digest = numeric_tree_digest(data['initial'])
    forward = make_forward(data)
    jax.clear_caches(); release_cpu_heap()
    report.update(frozen_before_sha256=frozen, initial_sha256=initial_digest, groups={}, chunks=[])
    for split, begin, end, offset in (('train', 0, 128, 4), ('heldout', 128, 160, 3)):
        keys = jax.random.split(jax.random.PRNGKey(source['seed']+offset), end-begin)
        for mode, std in plan['modes'].items():
            all_rows, sources = [], []
            for start in range(begin, end, 32):
                guard.check(); started = time.monotonic()
                window = forward(take(data['initial'], slice(start, start+32)),
                    keys[start-begin:start-begin+32], std=std, horizon=128)
                jax.block_until_ready(window); check_window(window)
                rows, detail = window_values(window, networks, source['source_rows'][start:start+32],
                    source['source_indices'][start:start+32], layout)
                path = args.output_dir/f'{split}_{mode}_{start-begin:03d}.npz'
                manifest = write_array_archive_unique(path, rows, guard.budget)
                restored = read_array_archive(path, manifest)
                if not all(np.array_equal(rows[k], restored[k]) for k in rows):
                    raise ValueError('Numeric evidence round trip failed')
                report['chunks'].append(dict(split=split, mode=mode, path=str(path), archive=manifest,
                    wall_s=time.monotonic()-started, **detail))
                sources.extend(detail['sources']); all_rows.append(rows)
                print(f'[CRITIC_DISTRIBUTION] {split}/{mode} {start-begin+32}/{end-begin}', flush=True)
                del window, restored
            joined = {k:np.concatenate([r[k] for r in all_rows]) for k in all_rows[0]}
            report['groups'][split+'/'+mode] = dict(stratified_summary(joined), sources=sources)
            del all_rows, joined
    report['frozen_after_sha256'] = numeric_tree_digest((data['actor'], data['robot'], data['auxiliary'], networks))
    if (report['frozen_after_sha256'] != frozen
            or numeric_tree_digest(data['initial']) != initial_digest):
        raise ValueError('Frozen network or reset bank mutated')
    report.update(status='FROZEN_DISTRIBUTION_AUDIT_COMPLETE',
        scheduled_imagined_controls=160*128*2,
        valid_imagined_controls=sum(r['valid_rows'] for r in report['groups'].values()),
        physically_failed_cases=sum(r['complete_sources'] for r in report['groups'].values()),
        native_control_steps=0, frozen_parameters_unchanged=True,
        scope='Source comparison only; no optimizer updates, model or policy promotion')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args(); cfg = load_config(); plan, source = check_plan(cfg)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, plan['max_worker_wall_s'],
                  startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        report = dict(schema='wmj_v1_critic_distribution_audit', status='INCOMPLETE',
            training_version='WMJ-V1', plan=cfg['critic_distribution_audit']['plan'],
            gpu_uuid=args.gpu_uuid, arguments=sys.argv, config_sha256=config_hash(cfg),
            actor_updates=0, critic_updates=0, world_updates=0, default_replaced=False, wandb=None,
            versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','mujoco','numpy')},
            source_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')})
        write_gzip_json(args.output_dir/'inputs.json.gz', dict(config=cfg, plan=plan,
            implementation={p: (Path(__file__).parent/p).read_text()
                for p in ('audit_critic_distribution.py','critic_distribution.py')}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            report['jax_devices'] = [str(d) for d in jax.devices()]
            run(cfg, args, guard, report, plan, source)
        except Exception as exc:
            report['error'] = repr(exc)
            raise
        finally:
            report['resources'] = guard.report()
            write_gzip_json(args.output_dir/'result.json.gz', report, guard.budget)


if __name__ == '__main__':
    main()
