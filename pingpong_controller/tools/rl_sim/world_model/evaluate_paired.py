"""Fixed native128 deterministic/sampled evaluation of a paired u30 export."""

import argparse
from pathlib import Path
import subprocess

import jax

from .artifacts import write_json_unique
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .evaluate_frozen import audit
from .paired_precision import load_actor
from .runtime_guard import RunGuard
from .train_paired import phase_plan, phase_budget


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--actor', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    args = parser.parse_args()
    cfg = load_config(); plan = phase_plan(cfg)
    args.initialization_dir = authenticate(repository_root(),
        cfg['frozen_evaluation_artifacts']['static_initialization']).parent
    args.stage = 1; args.seed = cfg['frozen_evaluation']['seed']; args.smoke = False
    reference = dict(path=str(args.actor.resolve()), sha256=file_hash(args.actor))
    reserve = cfg['bounded_imagination_pilot']['native_evaluation_startup_reserve_bytes']
    with RunGuard(cfg, args.gpu_uuid, args.output_dir, 3600., startup_reserve_bytes=reserve) as guard:
        phase_budget(guard, plan)
        guard.budget.check(additional_bytes=reserve)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_paired_precision_native_evaluation', status='INCOMPLETE',
            config_sha256=config_hash(cfg), training_version='WMJ-V1', gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sources},
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            actor_updates=0, critic_updates=0, world_updates=0, evaluated_actor_updates=30,
            promoted=False, plan=cfg['paired_precision_trial']['plan'])
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            sources={'evaluate_paired.py':Path(__file__).read_text()}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            actor,metadata=load_actor(cfg,reference)
            report['actor_training_metadata']=metadata
            audit(cfg,args,guard,report,actor_override=actor,actor_reference=reference)
        except Exception as exc:
            report['error']=repr(exc)
            raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
