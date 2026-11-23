"""Read-only four-family Critic input and generalization audit on fixed data."""

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

from .actor_critic import frozen_value
from .artifacts import read_array_archive, write_array_archive_unique
from .audit_critic_distribution import window_values
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .course_env import course_critic_layout
from .critic_distribution import PHASES, compact_regression
from .critic_input_audit import (ACTIVE_STD, STAT_NAMES, component_stats,
    distribution_summary, input_gradients, input_groups, nearest_training_rows,
    sequential_storage_reservation)
from .critic_targets import checkpoint_critics_and_pool
from .recover_reward_direction import write_gzip_json
from .reward_direction import make_prepare
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_critic_fit import make_forward, check_window, take
from .train_fixed_critic import pair_rows
from .train_pose_trial import phase_budget


def stats(values):
    values = np.asarray(values, float)
    if not values.size:
        return dict(count=0)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite diagnostic statistic')
    return dict(count=values.size, mean=float(values.mean()),
        p50=float(np.median(values)), p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def check_plan(cfg):
    setting = cfg['critic_input_audit']
    plan = json.loads(authenticate(repository_root(), setting['plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or not plan['authorized']
            or plan['max_phase_bytes'] != 2**20 or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or plan['horizon'] != 128 or plan['microbatch'] != 32
            or plan['training_mode'] != 'sampled' or plan['gradient_batch'] != 128
            or plan['updates'] != dict(actor=0, critic=0, world=0)):
        raise ValueError('Frozen input audit differs from approved scope')
    if plan['resource_policy'] != cfg['resources']['psi_stop_policy']:
        raise ValueError('Recorded WMJ PSI policy changed')
    if plan.get('resource_retry') and file_hash(Path(__file__).parent.parent/'training_memory_scope.py') != plan['resource_retry']['guard_sha256']:
        raise ValueError('Confirmed WMJ memory guard changed')
    for name, digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name) != digest:
            raise ValueError('Preserved world implementation changed: '+name)
    for key in ('physics', 'control'):
        if config_hash(cfg[key]) != plan[key+'_sha256']:
            raise ValueError('Physical/control contract changed')
    if config_hash(cfg['imagination_implementation']['reward']['coefficients']) != plan['default_reward_sha256']:
        raise ValueError('Default reward changed')
    authenticate(repository_root(), plan['prerequisite'])
    authenticate(repository_root(), plan['protocol'])
    source = json.loads(authenticate(repository_root(), plan['source_plan']).read_text())
    if (len(set(source['train_indices'])) != 128 or len(set(source['heldout_indices'])) != 32
            or set(source['train_indices']) & set(source['heldout_indices'])
            or source['source_indices'] != source['train_indices']+source['heldout_indices']):
        raise ValueError('Fixed source split changed')
    return plan, source


def gradient_summary(parameters, x, scale, groups, rows, layout, guard):
    """Accumulate derivatives without saving a redundant full N x2602 matrix."""
    extra = dict(actor_prefix=np.arange(73), actor_history=np.arange(73, 969),
        history_geometry=np.array([73+14*i+j for i in range(64) for j in range(12)]),
        history_age=np.arange(85, 969, 14), history_valid=np.arange(86, 969, 14))
    blocks = dict(groups, **extra)
    raw = {k:[] for k in blocks}; standardized = {k:[] for k in blocks}
    sums = np.zeros(x.shape[1]); squares = np.zeros_like(sums)
    for start in range(0, len(x), 128):
        guard.check(); n = min(128, len(x)-start)
        batch = np.zeros((128, x.shape[1]), np.float32); batch[:n] = x[start:start+n]
        gradient = np.asarray(input_gradients(parameters, jnp.asarray(batch)))[:n]
        if not np.isfinite(gradient).all():
            raise FloatingPointError('Nonfinite frozen value-input derivative')
        g = gradient.astype(float); sums += g.sum(0); squares += (g*g).sum(0)
        for name, ids in blocks.items():
            raw[name].append(np.sqrt(np.mean(g[:, ids]**2, axis=1)))
            standardized[name].append(np.sqrt(np.sum((g[:, ids]*scale[ids])**2, axis=1)))
    raw = {k:np.concatenate(v) for k,v in raw.items()}
    standardized = {k:np.concatenate(v) for k,v in standardized.items()}
    total = sum(standardized[k]**2 for k in groups)
    report = dict(groups={}, sources={}, phases={})
    for name in blocks:
        report['groups'][name] = dict(raw_component_rms=stats(raw[name]),
            one_training_std_l2=stats(standardized[name]),
            mean_scaled_squared_share=float(np.mean(standardized[name]**2/np.maximum(total, 1e-30))))
    for name, mask in [('sources', rows['source']), ('phases', rows['phase'])]:
        for label in np.unique(mask):
            chosen = mask == label
            report[name][str(int(label))] = {k:dict(raw_rms=float(raw[k][chosen].mean()),
                scaled_l2=float(standardized[k][chosen].mean())) for k in groups}
    rms = np.sqrt(squares/len(x))
    report['fields'] = {f['name']:dict(raw_rms=float(np.sqrt(np.mean(rms[f['start']:f['end']]**2))),
        scaled_l2=float(np.linalg.norm(rms[f['start']:f['end']]*scale[f['start']:f['end']])))
        for f in layout['fields']}
    return report, np.stack((sums/len(x), rms)).astype(np.float32)


def derivative_check(parameters, inputs, training_std, groups):
    """Small encoded-feature finite differences, not physical interventions."""
    x = jnp.asarray(inputs[np.linspace(0, len(inputs)-1, 16, dtype=int)])
    gradient = np.asarray(input_gradients(parameters, x)); result = {}
    for name, indices in groups.items():
        direction = np.zeros(inputs.shape[1], np.float32)
        direction[indices] = training_std[indices]*np.where(indices % 2, 1., -1.)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        analytic = gradient @ direction
        values = {}
        for epsilon in (.01, .001):
            finite = np.asarray((frozen_value(parameters, x+epsilon*direction)
                                 - frozen_value(parameters, x-epsilon*direction))/(2*epsilon))
            values[str(epsilon)] = dict(abs_error=stats(np.abs(analytic-finite)),
                derivative_abs=stats(np.abs(analytic)),
                resolvable_direction_agreement=float(np.mean(np.sign(analytic[np.abs(analytic)>1e-3])
                    == np.sign(finite[np.abs(analytic)>1e-3]))) if np.any(np.abs(analytic)>1e-3) else None)
        result[name] = values
    return result


def coverage_report(train, query, training_stats, groups, rows, training_rows, guard):
    report = {}; std = training_stats[1]
    for name, ids in groups.items():
        guard.check()
        distance, nearest = nearest_training_rows(train[:, ids], query[:, ids],
            training_rows['source'], rows['source'], training_rows['phase'], rows['phase'],
            std[ids] > ACTIVE_STD, std[ids])
        present = nearest >= 0
        gap = np.full(len(query), np.nan)
        gap[present] = np.abs(rows['candidate_lambda'][present]
                              - training_rows['candidate_lambda'][nearest[present]])
        by_source = {}
        for source in np.unique(rows['source']):
            mask = (rows['source'] == source) & present
            nearest_sources, counts = np.unique(training_rows['source'][nearest[mask]], return_counts=True)
            by_source[str(int(source))] = dict(distance=stats(distance[mask]),
                lambda_gap=stats(gap[mask]),
                most_frequent_training_source=int(nearest_sources[np.argmax(counts)]) if len(counts) else None)
        errors = np.abs(rows['candidate_online']-rows['mc'])
        valid = present & rows['complete']
        correlation = None
        if valid.sum() > 1 and np.std(distance[valid]) > 1e-12 and np.std(errors[valid]) > 1e-12:
            correlation = float(np.corrcoef(distance[valid], errors[valid])[0, 1])
        report[name] = dict(distance=stats(distance[present]), missing_phase_neighbor=int((~present).sum()),
            lambda_gap=stats(gap[present]), distance_abs_value_error_pearson=correlation,
            sources=by_source, phases={PHASES[int(p)]:stats(distance[present & (rows['phase']==p)])
                                    for p in np.unique(rows['phase'])})
    return report


def run(cfg, args, guard, report, plan, source):
    data = make_prepare()(cfg, args, guard, report, source)
    layout = course_critic_layout(data['initial']); groups = input_groups(layout)
    if {k:len(v) for k,v in groups.items()} != plan['group_widths']:
        raise ValueError('Diagnostic group layout changed')
    prior = json.loads(gzip.decompress(authenticate(repository_root(), plan['fixed_fit_run']).read_bytes()))
    networks = {}
    for name, ref in plan['checkpoints'].items():
        loaded = checkpoint_critics_and_pool(ref, data['initial'])
        if loaded['counters']['critic_updates'] != (1144 if name == 'parent' else 1656):
            raise ValueError('Expected exact parent/final Critic endpoint')
        networks[name] = loaded['critic']
        if name == 'parent':
            target = loaded['target_critic']
        del loaded
    frozen = numeric_tree_digest((data['actor'], data['robot'], data['auxiliary'], networks, target))
    initial_digest = numeric_tree_digest(data['initial'])
    report.update(frozen_before_sha256=frozen, initial_sha256=initial_digest,
                  layout=layout, groups={k:v.tolist() for k,v in groups.items()}, chunks=[])
    old_distribution = json.loads(gzip.decompress(authenticate(repository_root(), plan['distribution_run']).read_bytes()))
    previous = {(v['split'], v['mode'], int(Path(v['path']).stem.rsplit('_',1)[1])):v for v in old_distribution['chunks']}
    forward = make_forward(data); datasets = {}
    jax.clear_caches(); release_cpu_heap()
    for split, mode, begin, end, offset, std in (
            ('train','sampled',0,128,4,.03), ('heldout','deterministic',128,160,3,0.),
            ('heldout','sampled',128,160,3,.03)):
        keys = jax.random.split(jax.random.PRNGKey(source['seed']+offset), end-begin)
        inputs = []
        for start in range(begin, end, 32):
            guard.check()
            window = forward(take(data['initial'], slice(start,start+32)),
                keys[start-begin:start-begin+32], std=std, horizon=128)
            jax.block_until_ready(window); check_window(window)
            rows, detail = window_values(window, {'ema005':dict(critic=networks['parent'],target_critic=target)},
                source['source_rows'][start:start+32], source['source_indices'][start:start+32], layout)
            old = previous[split,mode,start-begin]
            pairing = pair_rows(rows, read_array_archive(Path(old['path']), old['archive']), plan['pair_tolerance'])
            inputs.append(np.asarray(window.critic_inputs)[np.asarray(window.transitions.valid, bool)])
            report['chunks'].append(dict(split=split,mode=mode,start=start-begin,pairing=pairing,**detail))
            del window, rows
        name = 'train/sample' if split == 'train' else split+'/'+mode
        x = np.concatenate(inputs); x.setflags(write=False)
        if numeric_tree_digest(x) != prior['dataset_digests'][name]['input_sha256']:
            raise ValueError('Existing fixed feature data did not reproduce')
        ref = prior['prediction_archives'][name]
        rows = read_array_archive(Path(ref['path']), ref['archive'])
        if len(rows['source']) != len(x):
            raise ValueError('Feature/value row alignment changed')
        datasets[name] = dict(x=x, rows=rows)
        print('[CRITIC_INPUTS] fixed data ready '+name, flush=True)
    # The model rollout is over. Keep only frozen network leaves for the final
    # integrity check and release the native/runtime/history construction graph.
    frozen_tensors = (data['actor'],data['robot'],data['auxiliary'],networks,target)
    report['initial_after_replay_sha256'] = numeric_tree_digest(data['initial'])
    if report['initial_after_replay_sha256'] != initial_digest:
        raise ValueError('Initial source memory changed during replay')
    def resident_bytes():
        return int(next(v.split()[1] for v in Path('/proc/self/status').read_text().splitlines()
                        if v.startswith('VmRSS:')))*1024
    before_release = resident_bytes()
    del forward, inputs, x, rows, data, prior, old_distribution, previous
    jax.clear_caches(); release_cpu_heap()
    report['temporary_memory_release'] = dict(before_rss_bytes=before_release,
        after_rss_bytes=resident_bytes(), method='drop_finished_native_runtime_and_seed_temporaries;clear_jax_caches;gc;malloc_trim')
    training = datasets['train/sample']; training_stats = component_stats(training['x'])
    # Exactly constant/near-constant coordinates contribute zero to std-scaled sensitivity.
    scale = np.where(training_stats[1] > ACTIVE_STD, training_stats[1], 0.)
    report['analysis'] = {}; component_rows = []; component_labels = []
    for name, dataset in datasets.items():
        guard.check(); x, rows = dataset['x'], dataset['rows']; started = time.monotonic()
        component_rows.append(component_stats(x)); component_labels.extend(name+'/input/'+k for k in STAT_NAMES)
        history = x[:,73:969].reshape(-1,64,14)
        result = dict(distribution={k:distribution_summary(x,training_stats,ids) for k,ids in groups.items()},
            history_valid_slots=stats(np.sum(history[:,:,13]>.5,axis=1)), gradients={}, value_errors={})
        for arm, parameters in networks.items():
            predicted = np.asarray(frozen_value(parameters,jnp.asarray(x)))
            expected = rows['initial_online' if arm=='parent' else 'candidate_online']
            np.testing.assert_allclose(predicted, expected, atol=5e-5,rtol=1e-5)
            complete=rows['complete']; failed=rows['failed']
            result['value_errors'][arm]=dict(mc=compact_regression(predicted[complete],rows['mc'][complete]),
                terminal=compact_regression(predicted[failed],rows['mc'][failed]))
            grad, components = gradient_summary(parameters,x,scale,groups,rows,layout,guard)
            result['gradients'][arm] = grad; component_rows.append(components)
            component_labels.extend(name+'/'+arm+'/gradient/'+k for k in ('mean','rms'))
            if name == 'train/sample':
                result.setdefault('finite_difference',{})[arm] = derivative_check(parameters,x,scale,groups)
        result['coverage'] = coverage_report(training['x'],x,training_stats,groups,rows,training['rows'],guard)
        result['wall_s'] = time.monotonic()-started
        report['analysis'][name] = result
        print('[CRITIC_INPUTS] analysis complete '+name, flush=True)
    report['component_rows'] = component_labels
    report['frozen_after_sha256'] = numeric_tree_digest(frozen_tensors)
    if report['frozen_after_sha256'] != frozen:
        raise ValueError('Frozen parameters or initial data changed')
    report.update(status='FROZEN_CRITIC_INPUT_AUDIT_COMPLETE',actor_updates=0,critic_updates=0,world_updates=0,
        native_control_steps=0,scheduled_imagined_controls=192*128,
        valid_imagined_controls=sum(len(v['x']) for v in datasets.values()),
        scope='Input diagnostics only; no input deletion, normalization, model or reward changes')
    matrix = np.concatenate(component_rows)
    codes = np.empty(layout['dimension'],np.int8)
    for code, ids in enumerate(groups.values()): codes[ids]=code
    arrays = dict(components=matrix,group_code=codes)
    reservation = sequential_storage_reservation(arrays, report)
    guard.budget.check(additional_bytes=reservation['required_peak_bytes'])
    report['sequential_write_reservation'] = reservation
    path=args.output_dir/'component_statistics.npz'
    archive=write_array_archive_unique(path,arrays,guard.budget)
    restored=read_array_archive(path,archive)
    if not all(np.array_equal(arrays[k],restored[k]) for k in arrays):
        raise ValueError('Component evidence changed on round trip')
    report['component_archive']=dict(path=str(path),archive=archive)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args(); cfg=load_config(); plan,source=check_plan(cfg)
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,plan['max_worker_wall_s'],startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan); args.output_dir.mkdir(parents=True,exist_ok=False)
        launcher=Path(__file__).parent.parent/'run_wmj_v1_critic_inputs.sh'
        report=dict(schema='wmj_v1_critic_input_audit',status='INCOMPLETE',training_version='WMJ-V1',
            plan=cfg['critic_input_audit']['plan'],config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,
            arguments=sys.argv,wandb=None,default_replaced=False,source_commit=plan['source_commit'],
            launcher=dict(path=str(launcher),sha256=file_hash(launcher)),
            versions={k:importlib.metadata.version(k) for k in ('jax','jaxlib','mujoco','numpy')},
            source_sha256={p.name:file_hash(p) for p in Path(__file__).parent.glob('*.py')},
            limitations=plan['limitations'])
        write_gzip_json(args.output_dir/'inputs.json.gz',dict(config=cfg,plan=plan,
            implementation={n:(Path(__file__).parent/n).read_text()
                for n in ('critic_input_audit.py','audit_critic_inputs.py')}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            report['jax_devices']=[str(d) for d in jax.devices()]
            run(cfg,args,guard,report,plan,source)
        except Exception as exc:
            report['error']=repr(exc); raise
        finally:
            report['resources']=guard.report()
            write_gzip_json(args.output_dir/'result.json.gz',report,guard.budget)


if __name__=='__main__':
    main()
