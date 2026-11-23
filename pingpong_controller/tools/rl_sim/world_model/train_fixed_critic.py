"""Opt-in fixed-data value fitting; parent EMA labels never move."""

from __future__ import annotations

import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import frozen_value, init_critic
from .artifacts import read_array_archive, write_array_archive_unique
from .audit_critic_distribution import window_values
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_layout
from .critic_distribution import stratified_summary
from .critic_fit import make_update
from .critic_targets import activation_summary
from .fixed_critic_fit import invariant_digest, next_draw, update_fixed_state
from .imagination_checkpoint import restore_checkpoint, save_checkpoint
from .imagination_learner import init_learner
from .persistent_pool import init_pool
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest, pack_numeric_state, restore_numeric_state
from .train_critic_fit import check_window, make_forward, storage_reservation, take
from .train_pose_trial import phase_budget


def check_resource_continuation(plan: dict) -> dict:
    """Authenticate a resource-only bridge from the interrupted176-step state."""
    resume = plan['resource_continuation']
    original = json.loads(authenticate(repository_root(), resume['original_plan']).read_text())
    allowed = {'authorization', 'max_phase_bytes', 'total_cap_bytes',
               'budget_validator_revision', 'resource_continuation'}
    if (set(plan) != set(original) | {'resource_continuation'}
            or any(plan[k] != value for k, value in original.items() if k not in allowed)):
        raise ValueError('Resource continuation changed the learning experiment')
    prior = json.loads(gzip.decompress(authenticate(repository_root(), resume['prior_run']).read_bytes()))
    payload = json.loads(authenticate(repository_root(), resume['checkpoint']).read_text())
    if (resume['completed_updates'] != 176 or resume['remaining_updates'] != 336
            or resume['psi_host_gib'] != 8 or resume['psi_scope_gib'] != 8
            or plan['max_phase_bytes'] != 48*2**20 or plan['total_cap_bytes'] != 6710886400
            or prior['actual_added_critic_updates'] != 176 or prior['status'] != 'INCOMPLETE'
            or prior['checkpoint'] != resume['checkpoint']
            or prior['plan'] != resume['original_plan']
            or payload['metadata']['plan'] != resume['original_plan']
            or payload['metadata']['fixed_fit_transactions'] != 44
            or payload['metadata']['critic_updates'] != 1320
            or payload['runtime_signature'] != resume['signature']):
        raise ValueError('Resource resume does not match the approved interruption')
    guard_path = Path(__file__).parent.parent/'training_memory_scope.py'
    if file_hash(guard_path) != resume['memory_guard_revision']['after_sha256']:
        raise ValueError('Resource resume guard changed')
    return prior


def check_plan(cfg: dict) -> tuple:
    setting = cfg['critic_fixed_data_fit']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    source = json.loads(authenticate(repository_root(), plan['source_plan']).read_text())
    recipe = dict(cfg['imagination'], target_critic_ema=.05)
    continuation = plan.get('resource_continuation')
    if (not setting['authorized'] or setting['default_enabled'] or not plan['authorized']
            or plan['maximum_critic_updates'] != 512 or plan['parent_critic_updates'] != 1144
            or plan['training_sources'] != 128 or plan['heldout_sources'] != 32
            or plan['horizon'] != 128 or plan['rollout_microbatch'] != 32
            or plan['training_mode'] != 'sampled' or plan['training_noise_std'] != .03
            or plan['training_noise_seed_offset'] != 4 or plan['heldout_noise_seed_offset'] != 3
            or plan['evaluation_modes'] != dict(deterministic=0., sampled=.03)
            or plan['evaluation_updates'] != [0, 128, 256, 512]
            or plan['recipe'] != recipe
            or plan['max_phase_bytes'] != (48 if continuation else 32)*2**20
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or any(plan[k] for k in ('actor_updates', 'world_updates', 'default_replacement'))):
        raise ValueError('Fixed-data fit differs from approved scope')
    if continuation:
        if setting.get('resource_resume_plan') != setting['plan']:
            raise ValueError('Resource continuation registration differs')
        check_resource_continuation(plan)
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Prior implementation changed: '+name)
    if file_hash(Path(__file__).parent/'contracts.py') != plan['budget_validator_revision']['after_sha256']:
        raise ValueError('Exact budget extension changed')
    for name in ('physics', 'control'):
        if config_hash(cfg[name]) != plan[name+'_sha256']:
            raise ValueError('Preserved physical/control contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    train, held = source['train_indices'], source['heldout_indices']
    if (len(set(train)) != 128 or len(set(held)) != 32 or set(train) & set(held)
            or source['source_indices'] != train+held
            or (set(train) | set(held)) & set(source['excluded_indices'])):
        raise ValueError('Fixed source partition changed')
    for name in ('protocol', 'prerequisite', 'distribution_plan', 'distribution_run'):
        authenticate(repository_root(), plan[name])
    return plan, source


def pair_rows(current: dict, previous: dict, tolerance: dict) -> dict:
    """Pair against a50 evidence without demanding floating-point bit equality."""
    maximum = 0.
    for key, value in current.items():
        old = previous[key]
        if value.dtype.kind == 'f':
            np.testing.assert_allclose(value, old, atol=tolerance['atol'], rtol=tolerance['rtol'])
            maximum = max(maximum, float(np.max(np.abs(value-old))))
        elif not np.array_equal(value, old):
            raise ValueError('Replay changed a source, phase, failure or validity: '+key)
    return dict(discrete_exact=True, maximum_float_abs=maximum, passed=True)


def restore_gate(state: dict, restored: dict, count: int) -> dict:
    if numeric_tree_digest(state) != numeric_tree_digest(restored):
        raise ValueError('Full fixed-data state changed after restore')
    if not np.array_equal(next_draw(state['learner'], count), next_draw(restored['learner'], count)):
        raise ValueError('Restored fixed-data sampler differs')
    return dict(full_numeric_state_exact=True, next_sample_indices_exact=True)


def evaluate(learner, datasets: dict) -> tuple:
    result, predictions = {}, {}
    for name, data in datasets.items():
        rows = dict(data['rows'])
        x = jnp.asarray(data['inputs'])
        rows['candidate_online'] = np.asarray(frozen_value(learner.critic, x))
        rows['candidate_target'] = np.asarray(frozen_value(learner.target_critic, x))
        # This is the parent-fixed lambda label, not the evolving EMA's target.
        rows['candidate_lambda'] = rows.pop('ema005_lambda')
        for key in ('ema005_online', 'ema005_target', 'ema005_next_target'):
            rows.pop(key)
        result[name] = dict(stratified_summary(rows, arms=('candidate',)),
            lambda_semantics='immutable_parent_EMA005_labels',
            input_sha256=data['input_sha256'], label_sha256=data['label_sha256'],
            online_activations=activation_summary(learner.critic, data['inputs']),
            target_activations=activation_summary(learner.target_critic, data['inputs']))
        predictions[name] = rows
    return result, predictions


def compare(baseline: dict, final: dict, criteria: dict) -> dict:
    train_key = 'train/sample'
    before = baseline[train_key]['overall']['candidate']['online_vs_lambda']['rmse']
    after = final[train_key]['overall']['candidate']['online_vs_lambda']['rmse']
    result = dict(training_fixed_lambda_rmse_ratio=after/before,
                  training_fit_gate=after/before <= criteria['training_frozen_lambda_rmse_ratio_max'],
                  heldout={})
    for key in ('heldout/deterministic', 'heldout/sampled'):
        old, new = baseline[key], final[key]
        first, last = old['overall']['candidate'], new['overall']['candidate']
        ratio = last['online_vs_mc']['rmse']/first['online_vs_mc']['rmse']
        ev = last['online_vs_mc']['explained_variance']
        bias_delta = (abs(new['time_to_failure']['terminal']['candidate']['online_vs_mc']['bias'])
            - abs(old['time_to_failure']['terminal']['candidate']['online_vs_mc']['bias']))
        gates = dict(rmse=ratio <= criteria['heldout_mc_rmse_ratio_max'],
            explained_variance=ev is not None and ev >= criteria['heldout_mc_explained_variance_min'],
            terminal_bias=bias_delta <= criteria['heldout_terminal_abs_bias_regression_max'])
        result['heldout'][key] = dict(online_mc_rmse_ratio=ratio,
            terminal_abs_bias_change=bias_delta, gates=gates, passed=all(gates.values()))
    result['all_gates_passed'] = result['training_fit_gate'] and all(
        value['passed'] for value in result['heldout'].values())
    return result


def run(cfg, args, guard, report, plan, source):
    continuation = plan.get('resource_continuation')
    if continuation and not args.resume:
        raise ValueError('The resource extension only authorizes an exact176-step resume')
    data = make_prepare()(cfg, args, guard, report, source)
    layout = course_critic_layout(data['initial'])
    if layout['dimension'] != 2602:
        raise ValueError('Critic input layout changed')
    train, held = take(data['initial'], slice(0, 128)), take(data['initial'], slice(128, 160))
    keys = jax.random.split(jax.random.PRNGKey(source['seed']+1), 128)
    learner = init_learner(data['actor'], init_critic(jax.random.PRNGKey(0), 2602),
                          init_pool(train, keys), seed=source['seed']+2)
    template = dict(learner=learner, reset_bank=train, validation_bank=held,
        current_source_indices=jnp.arange(128, dtype=jnp.int32), fit_transactions=jnp.array(0, jnp.int32))
    parent = restore_checkpoint(plan['parent_checkpoint'], template, signature=plan['parent_signature'])
    manifest = json.loads(authenticate(repository_root(), plan['parent_checkpoint']).read_text())
    if (manifest['metadata']['plan'] != plan['parent_plan'] or int(parent['fit_transactions']) != 256
            or int(parent['learner'].critic_updates) != 1144
            or int(parent['learner'].critic_optimizer.step) != 1144
            or int(parent['learner'].actor_updates) != 30
            or numeric_tree_digest(parent['learner'].actor) != numeric_tree_digest(data['actor'])):
        raise ValueError('Expected complete EMA005 Critic1144/Actor30 parent')
    restored = restore_checkpoint(plan['parent_checkpoint'], template, signature=plan['parent_signature'])
    gate = dict(parent_disk_restore=restore_gate(parent, restored, 1))
    del restored, template, learner
    frozen_world = numeric_tree_digest((data['robot'], data['auxiliary']))
    initial_hash = numeric_tree_digest(data['initial'])
    parent_hash = numeric_tree_digest(parent)
    old_report = json.loads(gzip.decompress(authenticate(repository_root(), plan['distribution_run']).read_bytes()))
    previous = {(r['split'], r['mode'], int(Path(r['path']).stem.rsplit('_', 1)[1])):r
                for r in old_report['chunks']}
    forward = make_forward(data)
    jax.clear_caches(); release_cpu_heap()
    networks = {'ema005':dict(critic=parent['learner'].critic, target_critic=parent['learner'].target_critic)}
    datasets = {}; report['replayed_chunks'] = []
    for split, mode, begin, end, offset, std in (
            ('train', 'sampled', 0, 128, 4, .03),
            ('heldout', 'deterministic', 128, 160, 3, 0.),
            ('heldout', 'sampled', 128, 160, 3, .03)):
        keys = jax.random.split(jax.random.PRNGKey(source['seed']+offset), end-begin)
        inputs, chunks = [], []
        for start in range(begin, end, 32):
            guard.check()
            window = forward(take(data['initial'], slice(start, start+32)),
                keys[start-begin:start-begin+32], std=std, horizon=128)
            jax.block_until_ready(window); check_window(window)
            rows, detail = window_values(window, networks, source['source_rows'][start:start+32],
                source['source_indices'][start:start+32], layout)
            old = previous[split, mode, start-begin]
            old_rows = read_array_archive(Path(old['path']), old['archive'])
            pairing = pair_rows(rows, old_rows, plan['replay_pair_tolerance'])
            if detail['sources'] != old['sources']:
                raise ValueError('Source outcomes differ from approved existing trajectories')
            x = np.asarray(window.critic_inputs)[np.asarray(window.transitions.valid, bool)]
            inputs.append(x); chunks.append(rows)
            report['replayed_chunks'].append(dict(split=split, mode=mode, start=start-begin,
                previous_archive=old['archive'], pairing=pairing, **detail))
            print(f'[FIXED_DATA] {split}/{mode} {start-begin+32}/{end-begin}', flush=True)
            del window, old_rows
        joined = {key:np.concatenate([r[key] for r in chunks]) for key in chunks[0]}
        x = np.concatenate(inputs); x.setflags(write=False)
        label = joined['ema005_lambda']; label.setflags(write=False)
        name = 'train/sample' if split == 'train' else split+'/'+mode
        datasets[name] = dict(inputs=x, rows=joined, input_sha256=numeric_tree_digest(x),
                              label_sha256=numeric_tree_digest(label))
    del forward, networks, inputs, chunks, x, label, joined
    jax.clear_caches(); release_cpu_heap()
    training = datasets['train/sample']
    state = dict(parent, fixed_lambda_targets=jnp.asarray(training['rows']['ema005_lambda']),
                 fixed_fit_transactions=jnp.array(0, jnp.int32))
    frozen = invariant_digest(state)
    dataset_refs = {k:{n:v[n] for n in ('input_sha256', 'label_sha256')} for k,v in datasets.items()}
    signature = config_hash(dict(plan=report['plan'], sources=report['source_sha256'],
        layout=layout, models=report['model_references'], datasets=dataset_refs))
    report.update(runtime_signature=signature, dataset_digests=dataset_refs,
        frozen_before_sha256=frozen, parent_full_state_sha256=parent_hash,
        frozen_world_before_sha256=frozen_world, critic_layout=layout)
    baseline, baseline_rows = evaluate(parent['learner'], datasets)
    report['evaluations'] = {'0':baseline}
    if args.resume:
        ref = dict(path=str(args.resume.resolve()), sha256=file_hash(args.resume))
        payload = json.loads(args.resume.read_text())
        expected_plan = continuation['original_plan'] if continuation else report['plan']
        restore_signature = continuation['signature'] if continuation else signature
        if payload['metadata']['plan'] != expected_plan:
            raise ValueError('Resume belongs to another fixed-data experiment')
        if continuation:
            if ref != continuation['checkpoint'] or payload['metadata']['fixed_datasets'] != dataset_refs:
                raise ValueError('Resume checkpoint or reconstructed fixed data differs')
            prior = check_resource_continuation(plan)
            if prior['model_references'] != report['model_references']:
                raise ValueError('Resume changed frozen world/Actor references')
            report['evaluations']['128'] = prior['evaluations']['128']
            report['inherited_evaluation_128'] = continuation['prior_run']
        state = restore_checkpoint(ref, state, signature=restore_signature)
        if invariant_digest(state) != frozen:
            raise ValueError('Resume changed fixed labels, Actor, pool or reset banks')
        report['resumed_from'] = ref
    initial_transactions = int(state['fixed_fit_transactions'])
    if continuation and initial_transactions != 44:
        raise ValueError('Expected exactly176 previously completed updates')
    if not 0 <= initial_transactions <= 128 or int(state['learner'].critic_updates) != 1144+4*initial_transactions:
        raise ValueError('Resume exceeds total approved Critic updates')
    report.update(initial_phase_updates=4*initial_transactions,
        actual_added_critic_updates=4*initial_transactions, worker_critic_updates=0)
    arrays, tree = pack_numeric_state(state, context_sha256=signature)
    restored = restore_numeric_state(state, arrays, tree, context_sha256=signature)
    gate['augmented_state_restore'] = restore_gate(state, restored, len(training['inputs']))
    del arrays, tree, restored
    if numeric_tree_digest(parent) != parent_hash or numeric_tree_digest(data['initial']) != initial_hash:
        raise ValueError('No-update preparation mutated the parent or source bank')
    reservation = storage_reservation(state)
    gate.update(status='FIXED_DATA_NO_UPDATE_PASS', updates=0,
        immutable_training_rows=len(training['inputs']),
        lambda_oracle_max_abs=max(r['lambda_oracle_max_abs']['ema005'] for r in report['replayed_chunks']),
        writer_reservation=dict(reservation, **guard.budget.check(
            additional_bytes=reservation['single_archive_writer_bytes']+4*2**20)))
    report['admission'] = gate
    write_gzip_json(args.output_dir/'admission.json.gz', gate, guard.budget)
    update_fn = make_update(plan['recipe'])
    report['transactions'] = []; saved = initial_transactions

    def save_current(interrupted=False):
        nonlocal saved
        current = int(state['fixed_fit_transactions']); added = 4*current
        reference = save_checkpoint(args.output_dir/f'critic_fixed_added{added:03d}.json', state,
            signature=signature, budget=guard.budget, metadata=dict(plan=report['plan'],
                parent=plan['parent_checkpoint'], critic_updates=1144+added, actor_updates=30,
                fixed_fit_transactions=current, parent_fit_transactions=256,
                fixed_datasets=dataset_refs, interrupted=interrupted, teacher_learning_rows=0))
        replay = restore_checkpoint(reference, state, signature=signature)
        report['final_checkpoint_restore'] = restore_gate(state, replay, len(training['inputs']))
        report['checkpoint'] = reference; saved = current

    try:
        for transaction in range(initial_transactions+1, 129):
            guard.check(); started = time.monotonic()
            state, info = update_fixed_state(state, training['inputs'], update_fn,
                initial_updates=1144, maximum_updates=512, authorized=True)
            added = 4*transaction
            report['transactions'].append(dict(added_critic_updates=added,
                wall_s=time.monotonic()-started, **info))
            report['actual_added_critic_updates'] = added
            report['worker_critic_updates'] = added-4*initial_transactions
            if added in plan['evaluation_updates']:
                if invariant_digest(state) != frozen:
                    raise ValueError('Fixed dataset, Actor or physical memory changed')
                evaluation, rows = evaluate(state['learner'], datasets)
                report['evaluations'][str(added)] = evaluation
                write_gzip_json(args.output_dir/f'evaluation_{added:03d}.json.gz', evaluation, guard.budget)
                if added == 512:
                    for name, current_rows in rows.items():
                        evidence = dict(current_rows,
                            initial_online=baseline_rows[name]['candidate_online'],
                            initial_target=baseline_rows[name]['candidate_target'])
                        path = args.output_dir/(name.replace('/', '_')+'_values.npz')
                        archive = write_array_archive_unique(path, evidence, guard.budget)
                        report.setdefault('prediction_archives', {})[name] = dict(path=str(path), archive=archive)
            if transaction == 1 or transaction % 8 == 0:
                print(f'[FIXED_CRITIC] {added}/512 loss={np.mean(info["loss"]):.6f}', flush=True)
        save_current()
        report['comparison'] = compare(baseline, report['evaluations']['512'], plan['acceptance'])
        report.update(status='FIXED_DATA_CRITIC_512_COMPLETE',
            training_rows=len(training['inputs']), replayed_valid_controls=sum(len(d['inputs']) for d in datasets.values()),
            sampled_training_rows=512*plan['recipe']['critic_batch'],
            worker_sampled_training_rows=(512-4*initial_transactions)*plan['recipe']['critic_batch'],
            scheduled_imagined_controls=192*128, native_control_steps=0,
            labels_recomputed_after_admission=False, pool_advanced_by_fixed_fit=False)
    except Exception:
        if int(state['fixed_fit_transactions']) > saved:
            save_current(interrupted=True)
        raise
    finally:
        report['frozen_after_sha256'] = invariant_digest(state)
        report['frozen_world_after_sha256'] = numeric_tree_digest((data['robot'], data['auxiliary']))
        if report['frozen_after_sha256'] != frozen or report['frozen_world_after_sha256'] != frozen_world:
            raise ValueError('Frozen parameters, labels, or environment memory changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args(); cfg = load_config(); plan, source = check_plan(cfg)
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, plan['max_worker_wall_s'], startup_reserve_bytes=2**20) as guard:
        phase_budget(guard, plan)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        launcher = Path(__file__).parent.parent/('run_wmj_v1_fixed_critic_resume.sh'
            if plan.get('resource_continuation') else 'run_wmj_v1_fixed_critic.sh')
        report = dict(schema='wmj_v1_fixed_data_critic_fit', status='INCOMPLETE', training_version='WMJ-V1',
            plan=cfg['critic_fixed_data_fit']['plan'], config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid,
            actor_updates=0, world_updates=0, default_replaced=False, wandb=None,
            arguments=sys.argv, source_commit=plan['source_commit'],
            launcher=dict(path=str(launcher), sha256=file_hash(launcher)),
            versions={k:importlib.metadata.version(k) for k in ('jax', 'jaxlib', 'mujoco', 'numpy')},
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        write_gzip_json(args.output_dir/'inputs.json.gz', dict(config=cfg, plan=plan,
            implementation={name:(Path(__file__).parent/name).read_text()
                for name in ('fixed_critic_fit.py', 'train_fixed_critic.py')}), guard.budget)
        try:
            if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
                raise ValueError('Exactly one CUDA device required')
            if plan.get('resource_continuation'):
                scope_log = Path(str(args.output_dir)+'_scope')/'memory.jsonl'
                with scope_log.open() as stream:
                    policy = json.loads(stream.readline())['psi_stop_policy']
                if (policy['host_available_at_most_bytes'] != 8*2**30
                        or not policy.get('fixed_critic_resume_only')):
                    raise ValueError('Use the approved isolated8GiB PSI resume launcher')
                report['resource_continuation_policy'] = policy
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
