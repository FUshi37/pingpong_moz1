"""Authenticate the minimal V225 continuation and immutable simulation contract."""
import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import pickle
import jax
import numpy as np
from fixed_plane_v225_near_side import (PROFILE, PARENT_PROFILE, PARENT_SHA256,
    START_STAGE, TEMPLATE_PATH, TEMPLATE_SHA256, validate_resume_payload)
from fixed_plane_v223_attitude import validate_saved_config
from train_juggle_mjx_curriculum import build_curriculum


def course(profile=PROFILE):
    return build_curriculum(curriculum_profile=profile,
        measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile='recovered_rmp_rmpmd_v2')


def authenticate(path, expected_sha=PARENT_SHA256):
    blob = path.read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected_sha:
        raise ValueError('V225 checkpoint SHA mismatch')
    payload = pickle.loads(blob)
    if payload['args']['curriculum_profile'] == PARENT_PROFILE and digest != PARENT_SHA256:
        raise ValueError('V225 parent is not the submitted V172')
    if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves((payload['params'], payload['opt']))):
        raise ValueError('V225 full training state contains nonfinite values')
    return payload, digest


def core_hashes():
    root = Path(__file__).resolve().parent
    names = ['fixed_plane_v225_near_side.py', 'preflight_fixed_plane_v225.py',
             'evaluate_fixed_plane_v225.py', 'assess_fixed_plane_v225.py',
             'mjx_juggle_env.py', 'train_juggle_mjx_curriculum.py',
             'train_juggle_mjx_ppo.py', 'validate_juggle_mjx_ppo.py',
             'run_gpu1_fixed_plane_v225.sh', 'references/v225_mocap_near_side_templates.json']
    return {str(root/name): hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint', type=Path, required=True)
    cli.add_argument('--checkpoint-sha', default=PARENT_SHA256)
    cli.add_argument('--start-stage', default=START_STAGE)
    cli.add_argument('--resume-curriculum-state', action='store_true')
    cli.add_argument('--admission', type=Path)
    cli.add_argument('--report', type=Path, required=True)
    args = cli.parse_args()
    if args.report.exists():
        raise ValueError('Refusing existing V225 preflight')
    source, sha = authenticate(args.checkpoint, args.checkpoint_sha)
    validate_resume_payload(source, requested_start_stage=args.start_stage,
        resume_curriculum_state=args.resume_curriculum_state,
        reset_optimizer=False, reset_critic=False, allow_obs_dim_migration=False)
    original = course(source['args']['curriculum_profile'])
    expected = next(s.cfg for s in original if s.name == source['stage_name'])
    saved = {k: asdict(v) if is_dataclass(v) else v for k, v in source['env_cfg'].items()}
    historical_defaults = validate_saved_config(saved, asdict(expected))
    stages = course()
    if hashlib.sha256(TEMPLATE_PATH.read_bytes()).hexdigest() != TEMPLATE_SHA256:
        raise ValueError('V225 empirical template hash changed')
    if args.admission:
        admission = json.loads(args.admission.read_text())
        if (admission.get('accepted') is not True or admission.get('profile') != PROFILE
                or admission.get('source_sha256') != PARENT_SHA256
                or admission.get('dimensions') != [72, 393, 7]
                or admission.get('core_hashes') != core_hashes()):
            raise ValueError('V225 admission or current-source hash rejected')
        for path, digest in admission['artifact_hashes'].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise ValueError('V225 admission artifact hash mismatch')
    report = dict(profile=PROFILE, parent_sha256=sha,
        source_profile=source['args']['curriculum_profile'], source_stage=source['stage_name'],
        source_update=source['stage_update'], true_step=source['step'], adam_t=int(source['opt'].t),
        dimensions=[72, 393, 7], finite_full_state=True, input_migration=False,
        historical_default_off_fields=historical_defaults,
        source_checkpoint=str(args.checkpoint.resolve()),
        start_stage=args.start_stage, stages=len(stages), core_hashes=core_hashes(),
        parent_restarted_not_probe_promoted=True, physical_dr_coverage_proven=False,
        lessons=[dict(index=i+1, name=s.name, min_updates=s.min_updates,
                      initial_xy_jitter_m=s.cfg.racket_launch_xy_jitter,
                      near_y_max_m=s.cfg.v225_near_y_max_m,
                      nominal_fraction=s.cfg.v225_nominal_fraction,
                      mirror_fraction=s.cfg.v225_mirror_fraction,
                      momentum_delta_max_m_s=s.cfg.v225_momentum_delta_limit_m_s,
                      contraction_ratio=s.cfg.v225_contraction_ratio) for i, s in enumerate(stages)])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
