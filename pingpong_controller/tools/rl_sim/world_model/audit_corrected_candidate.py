"""Frozen u8192/u12288 model and action checks under corrected projection."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import numpy as np

from .contracts import authenticate,config_hash,file_hash,load_config,repository_root
from .corrected_candidate import make_capture
from .corrected_action_audit import run_actions
from .expanded_auxiliary import restore_candidate
from .expanded_model_audit import paired_windows
from .native_entry_collision import write_compact_json
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget


def audit(cfg,args,guard,report,plan):
    started=time.monotonic();data=make_capture()(cfg,args,guard,report,plan)
    report['capture_wall_s']=time.monotonic()-started
    report['capture_parameters_sha256']=data['frozen']
    state,metadata=restore_candidate(cfg,plan['candidate_auxiliary'])
    models=dict(parent_u8192=data['auxiliary'],candidate_u12288={k:state[k] for k in ('parameters','scale')})
    frozen=numeric_tree_digest((data['actor'],data['robot'],models))
    report['candidate_checkpoint']=dict(plan['candidate_auxiliary'],updates=metadata['updates'])
    paired_windows(data,models,plan,guard,report,lanes=np.arange(32))
    progress=args.output_dir/'model_progress.json'
    digest=write_compact_json(progress,dict(status='FROZEN_MODEL_CHECK_COMPLETE',
        paired_world=report['paired_world'],window_population=report['window_population']),guard.budget)
    report['model_progress']=dict(path=str(progress),sha256=digest)
    report['paired_world']={name:dict(nonfinite_controls=value['nonfinite_controls'],wall_s=value['wall_s'],
        horizons={h:{k:v for k,v in row.items() if k not in ('by_region','per_window')} for h,row in value['horizons'].items()},
        releases={h:{k:v for k,v in row.items() if k!='cases'} for h,row in value['releases'].items()})
        for name,value in report['paired_world'].items()}
    run_actions(data,models,cfg,args,guard,report,plan)
    after=numeric_tree_digest((data['actor'],data['robot'],models))
    if frozen!=after:raise ValueError('Frozen models changed')
    for name,digest in plan['preserved_sources'].items():
        if file_hash(Path(__file__).parent/name)!=digest:raise ValueError('Preserved source changed:'+name)
    report.update(status='FROZEN_CORRECTED_CANDIDATE_AUDIT_COMPLETE',parameters_unchanged=True,
        frozen_parameters_before_sha256=frozen,frozen_parameters_after_sha256=after,
        actor_updates=0,critic_updates=0,world_updates=0,default_replaced=False,world_qualified=False,
        sources_preserved=len(plan['preserved_sources']),diagnostic_sources_excluded_from_u12288_refit=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args();cfg=load_config()
    setting=cfg['corrected_candidate_audit'];plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if (not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20
            or any(plan[k] for k in ('actor_updates','critic_updates','world_updates','default_replacement'))):
        raise ValueError('Only the authorized bounded frozen audit may run')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_corrected_candidate_audit',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
        names=('corrected_candidate.py','corrected_action_audit.py','audit_corrected_candidate.py')
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text() for n in names},protocol=authenticate(repository_root(),plan['protocol']).read_text()),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
