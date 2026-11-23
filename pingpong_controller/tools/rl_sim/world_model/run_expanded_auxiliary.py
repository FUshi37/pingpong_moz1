"""Explicit bounded stages for the user-approved expanded auxiliary refresh."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive,write_array_archive_unique
from .contracts import authenticate,config_hash,file_hash,load_config,repository_root
from .expanded_auxiliary import make_source_capture,retained_label_mask,restore_candidate
from .native_entry_collision import write_compact_json
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget


def collect(cfg,args,guard,report,plan):
    data=make_source_capture()(cfg,args,guard,report,plan)
    raw={k[4:]:v.reshape((-1,)+v.shape[2:]) for k,v in data['raw'].items() if k.startswith('aux_')}
    prefix=np.repeat(data['prefix'],5,axis=0)
    mask=retained_label_mask(raw['overlap'],prefix,np.random.default_rng(plan['label_retention']['seed']))
    times,lanes=np.nonzero(mask);rows={k:v[mask] for k,v in raw.items()}
    split=np.array([0 if r['split']=='train' else 1 for r in plan['source_rows']],np.int8)
    regions=np.array([r['region_id'] for r in plan['source_rows']],np.int32)
    rows.update(split=split[lanes],region_id=regions[lanes],family_id=lanes.astype(np.int32),substep_id=times.astype(np.int32))
    if np.any(rows['active_pair_count'][rows['overlap']]>1):raise ValueError('Ambiguous multiple native contact pairs')
    report['labels']=write_array_archive_unique(args.output_dir/'labels.npz',rows,guard.budget)
    report['actions']=write_array_archive_unique(args.output_dir/'actions.npz',dict(actions=data['actions'],
        native_prefix=data['prefix'],controlled_release=data['observed_labels']['controlled_release']),guard.budget)
    report['source_rows']=plan['source_rows']
    report['counts']={s:dict(sources=int((split==i).sum()),rows=int((rows['split']==i).sum()),
        contact_rows=int(((rows['split']==i)&rows['overlap']).sum()),
        retained_families=int(len(np.unique(rows['family_id'][rows['split']==i])))) for i,s in enumerate(('train','validation'))}
    report['retention']=dict(plan['label_retention'],total_native_substeps=128*5*64,
        task_prefix_substeps=int(prefix.sum()),retained_rows=len(times),labels_bytes=sum(v.nbytes for v in rows.values()))
    if numeric_tree_digest((data['actor'],data['robot'],data['auxiliary']))!=data['frozen']:
        raise ValueError('Collection changed frozen parameters')
    report.update(status='EXPANDED_AUXILIARY_LABELS_COMPLETE',actor_updates=0,critic_updates=0,world_updates=0,
        parameters_unchanged=True,default_replaced=False)


def audit_new(cfg,args,guard,report,plan):
    from .expanded_model_audit import paired_windows
    from .auxiliary_refresh import restore_auxiliary
    source=json.loads(authenticate(repository_root(),cfg['expanded_auxiliary_refresh']['collection']).read_text())
    folder=Path(cfg['expanded_auxiliary_refresh']['collection']['path']).parent
    saved=read_array_archive(folder/'actions.npz',source['actions'])
    # Preserve the collection's actor-in-graph numerical path. Inlining archived
    # actions changes XLA arithmetic even though the physical contract is equal.
    data=make_source_capture()(cfg,args,guard,report,plan)
    action_difference=np.abs(data['actions']-saved['actions'])
    state,_=restore_candidate(cfg,cfg['expanded_auxiliary_refresh']['candidate_checkpoint'])
    models=dict(parent=data['auxiliary'],candidate={k:state[k] for k in ('parameters','scale')})
    paired_windows(data,models,plan,guard,report,lanes=np.arange(48,64))
    report.update(status='EXPANDED_AUXILIARY_NEW_VALIDATION_COMPLETE',
        collection_actions_exact=bool(np.array_equal(data['actions'],saved['actions'])),
        collection_action_max_abs=float(action_difference.max()),
        paired_models_use_same_native_actions=True,
        collection_initial_hash_equal=report['initial_native_sha256']==source['initial_native_sha256'],
        original_failure_clocks=source['native_population']['first_failure_control'],
        source_split='original_validation_only',actor_updates=0,critic_updates=0,world_updates=0,default_replaced=False)


def audit_diagnostic(cfg,args,guard,report,plan):
    from .expanded_model_audit import paired_windows
    from .error_attribution import make_capture
    previous=json.loads(authenticate(repository_root(),plan['diagnostic_plan']).read_text())
    data=make_capture()(cfg,args,guard,report,previous)
    state,_=restore_candidate(cfg,cfg['expanded_auxiliary_refresh']['candidate_checkpoint'])
    models=dict(parent=data['auxiliary'],candidate={k:state[k] for k in ('parameters','scale')})
    paired_windows(data,models,previous,guard,report,lanes=np.arange(32))
    report.update(status='EXPANDED_AUXILIARY_PRIOR32_DIAGNOSTIC_COMPLETE',source_split='diagnostic_only_excluded_from_refit',
        actor_updates=0,critic_updates=0,world_updates=0,default_replaced=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['collect','fit','new-audit','diagnostic','retention'],required=True)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config();setting=cfg['expanded_auxiliary_refresh']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or plan['max_phase_bytes']!=12*2**20 or plan['maximum_auxiliary_updates']!=4096:
        raise ValueError('User-approved bounded auxiliary phase required')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_expanded_auxiliary_'+args.phase,training_version='WMJ-V1',status='INCOMPLETE',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in paths},actor_updates=0,critic_updates=0,default_replaced=False)
        names=('expanded_auxiliary.py','run_expanded_auxiliary.py','fit_expanded_auxiliary.py','expanded_model_audit.py','audit_expanded_retention.py')
        write_compact_json(args.output_dir/'implementation.json',dict(report=report,
            sources={p.name:p.read_text() for p in paths if p.name in names}),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            if args.phase=='fit':
                from .fit_expanded_auxiliary import fit
                fit(cfg,args,guard,report,plan)
            elif args.phase=='retention':
                from .audit_expanded_retention import audit
                audit(cfg,args,guard,report,plan)
            else:{'collect':collect,'new-audit':audit_new,'diagnostic':audit_diagnostic}[args.phase](cfg,args,guard,report,plan)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
