"""Verify frozen-policy parity and finite full-state learning before formal V225."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import re
import jax
import numpy as np
from fixed_plane_v225_near_side import PROFILE, PARENT_SHA256
from preflight_fixed_plane_v225 import core_hashes


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_nominal_isolation(before, after):
    """Require no event and numerically identical complete saved nominal metrics."""
    required = {'terminated', 'truncated', 'hit_event_count', 'hit_count',
                'ball_racket_anchor_xy_dist_m', 'v225/group', 'v225/momentum_event'}
    if not required.issubset(before) or set(before) != set(after):
        raise ValueError('V225 nominal isolation metrics missing or different')
    for metrics in [before, after]:
        if np.any(metrics['v225/group']) or np.any(metrics['v225/momentum_event']):
            raise ValueError('V225 nominal control contains a non-nominal event')
    errors = {}
    for name in before:
        a, b = np.asarray(before[name]), np.asarray(after[name])
        if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('V225 nominal control has invalid shape or values')
        errors[name] = float(np.max(np.abs(a-b)))
        if errors[name] > 2e-5:
            raise ValueError(f'V225 nominal control changed {name}')
    return errors


def assess(design: Path, probe_run: Path, screens: Path, learned_screen: Path,
           isolation_screen: Path, momentum_screen: Path):
    artifacts = []
    parity = []
    for seed in [20266501, 20266502]:
        old_path = screens/'legacy_parent_cuda/case_parent'/f'metrics_{seed}.npz'
        current_path = screens/'current_parent_cuda/case_parent'/f'metrics_{seed}.npz'
        old, current = np.load(old_path), np.load(current_path)
        errors = {}
        for key in old.files:
            if old[key].shape != current[key].shape or not np.isfinite(current[key]).all():
                raise ValueError('V225 legacy replay shape/finite mismatch')
            error = float(np.max(abs(old[key]-current[key])))
            if error > 2e-5:
                raise ValueError(f'V225 changed historical {key}: max error {error}')
            errors[key] = error
        parity.append(dict(seed=seed, max_errors=errors))
        artifacts += [old_path, current_path]
    workspace_manifest = screens/'workspace_cuda/manifest.json'
    workspace = json.loads(workspace_manifest.read_text())
    parent_manifest = screens/'full_parent_cuda/manifest.json'
    parent = json.loads(parent_manifest.read_text())
    for manifest in [workspace, parent]:
        if (manifest['source_sha256'] != PARENT_SHA256 or manifest['n_envs'] != 288
                or manifest['steps'] != 1200 or len(manifest['seeds']) != 2):
            raise ValueError('V225 paired screen identity differs')
    paired = []
    initial_regressions = []
    for old, new in zip(parent['cases']['parent'], workspace['cases']['1'], strict=True):
        if old['seed'] != new['seed']:
            raise ValueError('Paired V172/workspace seeds differ')
        ob, nb = old['block'], new['block']
        old_failed = {key for key, value in ob.items() if key.endswith('_ok') and value == 0.}
        new_failed = {key for key, value in nb.items() if key.endswith('_ok') and value == 0.}
        if not new_failed.issubset(old_failed):
            initial_regressions.append([new['seed'], 'quality_flags', sorted(new_failed-old_failed)])
        if (nb['advance_eval/mean_hits'] < ob['advance_eval/mean_hits']-.10
                or nb['advance_eval/mean_len_frac'] < ob['advance_eval/mean_len_frac']-.02):
            initial_regressions.append([new['seed'], 'hit_or_length'])
        for key, tolerance in [('late_physical_in_band_rate', .001),
                               ('ball_view_in_bounds', .01)]:
            name = 'advance_eval/'+key
            if name in ob and np.isfinite(ob[name]) and nb[name] < ob[name]-tolerance:
                initial_regressions.append([new['seed'], key])
        for key, tolerance in [('mean_hit_contact_center_dist', .0005),
                               ('rapid_steady_racket_vxy_rms_m_s', .005),
                               ('rapid_steady_racket_omega_rms_rad_s', .02)]:
            name = 'advance_eval/'+key
            if name in ob and np.isfinite(ob[name]) and nb[name] > ob[name]+tolerance:
                initial_regressions.append([new['seed'], key])
        paired.append(dict(seed=new['seed'], parent_hits=ob['advance_eval/mean_hits'],
                           workspace_hits=nb['advance_eval/mean_hits'],
                           parent_len=ob['advance_eval/mean_len_frac'],
                           workspace_len=nb['advance_eval/mean_len_frac'],
                           parent_passed=ob['advance_eval/passed'],
                           workspace_passed=nb['advance_eval/passed'],
                           source_failed_gates=sorted(old_failed),
                           workspace_failed_gates=sorted(new_failed)))
    artifacts += [workspace_manifest, parent_manifest]
    # Hard-domain failure is a baseline, not a source-policy admission failure.
    near_path = screens/'near_side_cuda/manifest.json'
    near = json.loads(near_path.read_text())
    if near['source_sha256'] != PARENT_SHA256 or set(near['cases']) != {'8', '11'}:
        raise ValueError('V225 near-side exposure screen missing')
    for case in near['cases'].values():
        for block in case:
            if not np.isfinite(block['block']['advance_eval/mean_hits']):
                raise ValueError('V225 hard-domain screen nonfinite')
    artifacts.append(near_path)
    for name in ['unit_tests.log', 'legacy_contract_tests.log',
                 'final_gate_contract_tests_r4.log', 'isolation_admission_test.log']:
        path = design/name
        log = path.read_text()
        if not re.search(r'\d+ passed', log) or re.search(r'\d+ failed|ERROR collecting', log):
            raise ValueError(f'V225 required CPU tests did not pass: {name}')
        artifacts.append(path)
    checkpoint = probe_run/'mjx_curriculum_last.pkl'
    payload = pickle.loads(checkpoint.read_bytes())
    args = payload['args']
    if (args['curriculum_profile'] != PROFILE or args['max_stage_updates'] != 16
            or payload['stage_index'] != 1 or payload['stage_update'] != 16
            or payload['step'] != 1334837248+16*131072
            or int(payload['opt'].t) <= 335771
            or (payload['obs_dim'], payload['critic_obs_dim'], payload['act_dim']) != (72, 393, 7)
            or args['n_envs'] != 1024 or args['n_steps'] != 128
            or digest(Path(args['resume_from'])) != PARENT_SHA256):
        raise ValueError('V225 bounded learning state/optimizer restoration mismatch')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((payload['params'], payload['opt']))):
        raise ValueError('V225 probe produced nonfinite state')
    progress = probe_run/'curriculum_progress.csv'
    with progress.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 16:
        raise ValueError('V225 probe did not complete16 updates')
    finite_columns = ['approx_kl', 'explained_var', 'policy_loss', 'value_loss']
    for key in finite_columns:
        values = [float(row[key]) for row in rows]
        if not values or not np.isfinite(values).all():
            raise ValueError(f'V225 learning telemetry invalid: {key}')
    if max(float(row['approx_kl']) for row in rows) > .03:
        raise ValueError('V225 bounded probe exceeded KL admission limit')
    completed = [row for row in rows if float(row.get('episodes', 0)) > 0]
    count = sum(float(row['episodes']) for row in completed)
    mean_hits = sum(float(row['episodes'])*float(row['mean_hits']) for row in completed)/max(count, 1)
    mean_len = sum(float(row['episodes'])*float(row['mean_len']) for row in completed)/max(count, 1)
    if count < 512 or mean_hits < 12.5 or mean_len/1200 < .95:
        raise ValueError(f'V225 bounded source learning regressed: {count}/{mean_hits}/{mean_len}')
    learned = json.loads(learned_screen.read_text())
    if (learned['source_sha256'] != digest(checkpoint) or learned['n_envs'] != 288
            or learned['steps'] != 1200 or learned['seeds'] != [20266501, 20266502]):
        raise ValueError('V225 learned-policy two-seed screen identity differs')
    learned_results = learned['cases']['1']
    if len(learned_results) != 2 or any(x['block']['advance_eval/passed'] != 1. for x in learned_results):
        raise ValueError('V225 learned policy has not restored complete original graduation gates')
    for old, candidate in zip(parent['cases']['parent'], learned_results, strict=True):
        if (candidate['seed'] != old['seed']
                or candidate['block']['advance_eval/mean_hits'] < old['block']['advance_eval/mean_hits']-.10
                or candidate['block']['advance_eval/mean_len_frac'] < old['block']['advance_eval/mean_len_frac']-.02):
            raise ValueError('V225 learned-policy paired source retention failed')
    artifacts.append(learned_screen)
    isolation = json.loads(isolation_screen.read_text())
    if (isolation['source_sha256'] != PARENT_SHA256 or isolation['n_envs'] != 32
            or isolation['steps'] != 128 or isolation['seeds'] != [20266504]
            or set(isolation['cases']) != {'8', '11'}):
        raise ValueError('V225 nominal isolation identity differs')
    isolation_paths = [isolation_screen.parent/f'case_{case}/metrics_20266504.npz'
                       for case in [8, 11]]
    with np.load(isolation_paths[0]) as before, np.load(isolation_paths[1]) as after:
        isolation_errors = validate_nominal_isolation(
            dict(before.items()), dict(after.items()))
    momentum = json.loads(momentum_screen.read_text())
    if (momentum['source_sha256'] != PARENT_SHA256 or momentum['n_envs'] != 288
            or momentum['steps'] != 1200 or momentum['seeds'] != [20266503]
            or set(momentum['cases']) != {'11'}):
        raise ValueError('V225 corrected momentum screen identity differs')
    momentum_path = momentum_screen.parent/'case_11/metrics_20266503.npz'
    with np.load(momentum_path) as metrics:
        if not all(np.isfinite(metrics[name]).all() for name in metrics.files):
            raise ValueError('V225 corrected momentum screen has nonfinite metrics')
        event = metrics['v225/momentum_event'] > .5
        delta = metrics['v225/momentum_delta_m_s']
        done = metrics['terminated']+metrics['truncated'] > .5
        active = np.cumsum(done, axis=0)-done == 0
        initial_events = event & active
        if (np.any(initial_events.sum(axis=0) > 1)
                or np.any(event & (metrics['v225/group'] == 0))
                or np.any(event & (metrics['hit_count'] < 3))
                or np.max(np.abs(delta)) > .080001
                or np.any(delta[~event] != 0.)):
            raise ValueError('V225 one-shot momentum or control-lane contract failed')
        exposure = {str(group): int(np.sum(initial_events & (metrics['v225/group'] == group)))
                    for group in [0, 1, 2]}
        if exposure['1'] < 8 or exposure['2'] < 8:
            raise ValueError('V225 corrected momentum CUDA test did not exercise both groups')
        momentum_summary = dict(first_episode_events=exposure,
            max_abs_delta_m_s=float(np.max(np.abs(delta))),
            max_first_episode_events_per_lane=int(initial_events.sum(axis=0).max()),
            recovery=momentum['cases']['11'][0]['recovery'])
    artifacts += [isolation_screen, momentum_screen, *isolation_paths, momentum_path]
    artifacts += [checkpoint, progress, probe_run/'stability_preflight.json', probe_run/'command.sh']
    for manifest_path in [workspace_manifest, parent_manifest, near_path,
                          screens/'legacy_parent_cuda/manifest.json',
                          screens/'current_parent_cuda/manifest.json', learned_screen,
                          isolation_screen, momentum_screen]:
        manifest = json.loads(manifest_path.read_text())
        for path, sha in manifest.get('artifact_hashes', {}).items():
            if digest(Path(path)) != sha:
                raise ValueError('V225 experiment artifact hash changed')
            artifacts.append(Path(path))
    return dict(accepted=True, profile=PROFILE, source_sha256=PARENT_SHA256,
        dimensions=[72, 393, 7], core_hashes=core_hashes(),
        parent_restarted_not_probe_promoted=True, legacy_parity=parity,
        source_workspace_paired=paired, frozen_workspace_negative_results=initial_regressions,
        learned_workspace_restored_gates=learned_results, hard_domain_baseline=near['cases'],
        late_nominal_isolation_max_errors=isolation_errors,
        corrected_momentum_screen=momentum_summary,
        probe=dict(run=str(probe_run.resolve()), updates=16,
                   adam_before=335771, adam_after=int(payload['opt'].t),
                   episodes=count, population_mean_hits=mean_hits,
                   population_len_frac=mean_len/1200, finite_telemetry_columns=finite_columns),
        artifact_hashes={str(path.resolve()): digest(path) for path in artifacts})


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--design', type=Path, required=True)
    cli.add_argument('--probe-run', type=Path, required=True)
    cli.add_argument('--screens', type=Path, required=True)
    cli.add_argument('--learned-screen', type=Path, required=True)
    cli.add_argument('--isolation-screen', type=Path, required=True)
    cli.add_argument('--momentum-screen', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError('Refusing existing V225 admission')
    result = assess(args.design, args.probe_run, args.screens, args.learned_screen,
                    args.isolation_screen, args.momentum_screen)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print('V225 formal admission ACCEPTED', json.dumps(result['probe']), flush=True)


if __name__ == '__main__':
    main()
