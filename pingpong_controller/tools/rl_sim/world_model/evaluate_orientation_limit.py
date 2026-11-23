"""Fixed native128 deterministic/sampled evaluation of a bounded/free-racket-attitude frozen comparison."""

import argparse
from pathlib import Path
import subprocess

import jax

from .native_entry_collision import write_compact_json as write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .free_orientation import audit, phase_plan
from .paired_precision import load_actor
from .runtime_guard import RunGuard
from .train_pose_trial import phase_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--mode', choices=('bounded_2deg', 'free_orientation'), required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config(); plan = phase_plan(cfg)
    args.initialization_dir = authenticate(repository_root(),
        cfg['frozen_evaluation_artifacts']['static_initialization']).parent
    args.stage = 1; args.seed = cfg['frozen_evaluation']['seed']; args.smoke = False
    reference = plan['actor']; authenticate(repository_root(), reference)
    reserve = 3 * 2**20
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600., startup_reserve_bytes=reserve) as guard:
        phase_budget(guard, plan)
        guard.budget.check(additional_bytes=reserve)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_orientation_limit_native_evaluation', variant=args.mode, status='INCOMPLETE',
            config_sha256=config_hash(cfg), training_version='WMJ-V1', gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sources},
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            actor_updates=0, critic_updates=0, world_updates=0, evaluated_actor_updates=30,
            promoted=False, plan=cfg['orientation_limit_trial']['plan'])
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            sources={n:(Path(__file__).parent/n).read_text() for n in ('evaluate_orientation_limit.py','free_orientation.py')}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            actor,metadata=load_actor(cfg,reference)
            report['actor_training_metadata']=metadata
            audit(cfg,args,guard,report,actor=actor,actor_reference=reference)
        except Exception as exc:
            report['error']=repr(exc)
            raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
