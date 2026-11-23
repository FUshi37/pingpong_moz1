"""Reject recovery candidates that regress paired stability or task success."""
import argparse
import hashlib
import json
import math
from pathlib import Path

LOWER_IS_BETTER = (
    'advance_eval/mean_hit_vxy',
    'advance_eval/rms_hit_vxy',
    'advance_eval/mean_hit_racket_vxy',
    'advance_eval/rapid_steady_racket_vxy_rms_m_s',
    'advance_eval/rapid_steady_racket_omega_rms_rad_s',
)
HIGHER_IS_BETTER = (
    'advance_eval/hit1_rate',
    'advance_eval/hit3_rate',
    'advance_eval/hit12_rate',
    'advance_eval/episode_truncation_rate',
    'advance_eval/ball_view_in_bounds',
)


def compare_blocks(candidate: dict, reference: dict) -> dict:
    """Use no positive degradation margin; missing/nonfinite metrics fail."""
    differences = {}
    regressions = []
    for names, sign in [(LOWER_IS_BETTER, 1.0), (HIGHER_IS_BETTER, -1.0)]:
        for name in names:
            value = float(candidate.get(name, float('nan')))
            old = float(reference.get(name, float('nan')))
            valid = math.isfinite(value) and math.isfinite(old)
            differences[name] = value - old if valid else None
            if not valid or sign * (value - old) > 0.0:
                regressions.append(name)
    return dict(noninferior=not regressions, differences=differences,
                regressions=regressions)


def digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def authenticate_manifest(path: Path) -> dict:
    evidence = json.loads(path.read_text())
    if not evidence.get('artifact_hashes'):
        raise ValueError('Missing hashed evidence')
    for name, expected in evidence['artifact_hashes'].items():
        if digest(Path(name)) != expected:
            raise ValueError(f'Changed evidence: {name}')
    return evidence


def main() -> None:
    from preflight_fixed_plane_v225 import authenticate
    from prepare_fixed_plane_v226 import core_hashes
    from fixed_plane_v226_stability_recovery import PROFILE, PARENT_SHA256
    import jax
    import numpy as np

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--baseline', type=Path, required=True)
    cli.add_argument('--candidate', type=Path, required=True)
    cli.add_argument('--legacy-parity', type=Path, required=True)
    cli.add_argument('--probe-run', type=Path, required=True)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to replace an assessment')
    baseline = authenticate_manifest(args.baseline)
    candidate = authenticate_manifest(args.candidate)
    parity = authenticate_manifest(args.legacy_parity)
    preflight_path = args.probe_run / 'stability_preflight.json'
    preflight = json.loads(preflight_path.read_text())
    if (candidate.get('profile') != PROFILE
            or candidate.get('n_envs') != 288
            or candidate.get('steps') != 1200
            or candidate.get('seeds') != [20266801, 20266802]
            or preflight.get('source_sha256') != PARENT_SHA256
            or preflight.get('core_hashes') != core_hashes()
            or preflight.get('kind') != 'probe'):
        raise ValueError('Probe identity or paired sampling contract differs')
    payload, _ = authenticate(Path(candidate['checkpoint']), candidate['source_sha256'])
    if (Path(candidate['checkpoint']).resolve().parent != args.probe_run.resolve()
            or payload['args']['curriculum_profile'] != PROFILE
            or tuple(payload[k] for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or not 0 < payload['args']['max_stage_updates'] <= 64
            or int(payload['opt'].t) <= preflight['source_adam_t']
            or not all(np.all(np.isfinite(np.asarray(x)))
                       for x in jax.tree.leaves(payload['params']))):
        raise ValueError('Invalid bounded full-state optimization probe')
    old = {(item['case'], item['seed']): item['block']
           for item in baseline['results'] if item['policy'] == 'v225_s5'}
    cases = {(item['case'], item['seed']): item for item in candidate['results']}
    if set(cases) != {(case, seed) for case in [1, 10]
                      for seed in [20266801, 20266802]}:
        raise ValueError('Incomplete candidate domains/seeds')
    comparisons = []
    for (case, seed), item in sorted(cases.items()):
        result = compare_blocks(item['block'], old[(1 if case == 1 else 6, seed)])
        result.update(case=case, seed=seed,
                      complete_gate_passed=item['block'].get('advance_eval/passed') == 1.0,
                      recovery=item['recovery'])
        comparisons.append(result)
    accepted = (parity.get('passed') is True
                and all(item['noninferior'] and item['complete_gate_passed']
                        for item in comparisons))
    artifacts = {str(path.resolve()): digest(path)
                 for path in [args.baseline, args.candidate, args.legacy_parity,
                              preflight_path, Path(candidate['checkpoint'])]}
    for evidence in [baseline, candidate, parity]:
        artifacts.update(evidence['artifact_hashes'])
    result = dict(accepted=accepted, profile=PROFILE, source_sha256=PARENT_SHA256,
                  core_hashes=core_hashes(), artifact_hashes=artifacts,
                  comparisons=comparisons, probe_checkpoint=candidate['checkpoint'],
                  probe_weights_promoted=False,
                  formal_source='passed V225 S5; independent seed; start S1')
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print('V226_ADMISSION', accepted, flush=True)


if __name__ == '__main__':
    main()
