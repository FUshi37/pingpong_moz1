"""Compare private FK implementations on the fixed integration control population."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_full_prior_world import AuditReader, release_cpu_heap
from .audit_support_replay import load_support_chain
from .contracts import config_hash, file_hash, load_config, repository_root
from .control_kinematics import extract_control_chain, forward_control_kinematics
from .control_projection import project_reference
from .control_seed import read_control_seed
from .fit_robot import summarize_errors
from .kinematics import forward_kinematics
from .native_stack import build_native_environment
from .runtime_guard import RunGuard
from .world_control import ControlContract


def audit(cfg,args,guard,report):
    root=repository_root()/cfg['resources']['output_root']
    evidence=root/'full_prior_integration_a05/result.json'
    if file_hash(evidence)!='ed458c606de8ec47d801b8676366bb5df72d292af04832a0efac2433ea732d95':
        raise ValueError('Frozen integration population changed')
    population=json.loads(evidence.read_text())
    reader=AuditReader(root/'support_replay_full_a01',guard.budget,max_bytes=0)
    bundle=build_native_environment(cfg,n_envs=1)
    contract=ControlContract.from_native(bundle.environment)
    model,matrix=load_support_chain(cfg,reader.report['geometry'])
    quaternion=extract_control_chain(model,reader.report['geometry']['reset_fk']['qpos'],
        joint_ids=matrix.joint_ids,site_id=reader.report['geometry']['racket']['site_id'],
        geom_id=reader.report['geometry']['racket']['geom_id'])

    def make_predictor(chain,fk):
        @jax.jit
        def predict(context,actions,measured,references):
            def tick(unused,values):
                action,q,reference=values
                scale=contract.projection.velocity_limit*contract.action_velocity_scale*context.action_scale_mult[:,None]
                nominal=jnp.clip(jnp.clip(action,-1.,1.)*scale,
                    -contract.projection.velocity_limit,contract.projection.velocity_limit)
                result=project_reference(chain,contract.projection,q,reference,nominal,kinematics=fk)
                accepted=jnp.clip(result.accepted_qvel/jnp.maximum(scale,1e-6),-1.,1.)
                return unused,(result,accepted)
            return jax.lax.scan(tick,None,(actions,measured,references))[1]
        return predict

    predictors=dict(matrix=make_predictor(matrix,forward_kinematics),
                    quaternion=make_predictor(quaternion,forward_control_kinematics))
    all_errors={k:dict(reference=[],accepted=[]) for k in predictors}
    counts={k:0 for k in predictors};rows=[];names=list(reader.records)
    for selection in population['trials']:
        guard.check();trial=selection['trial'];lanes=np.array(selection['lanes'])
        keys=names[2*trial:2*trial+2]
        _,context=read_control_seed(reader.records[keys[0]]['first_seed_ref'],
            context_sha256=reader.report['context_sha256'],lanes=lanes)
        streams={'state__q_rad','state__support_q_rad','state__q_ref_rad',
                 'action__raw_normalized','action__accepted_normalized'}
        pieces=[]
        for key in keys:
            full=reader.read(key,names=streams)
            pieces.append({k:v[:,lanes].copy() for k,v in full.items()})
            del full
        actions=np.concatenate([x['action__raw_normalized'] for x in pieces])
        accepted=np.concatenate([x['action__accepted_normalized'] for x in pieces])
        before=np.concatenate([x['state__q_ref_rad'][:-1] for x in pieces])
        after=np.concatenate([x['state__q_ref_rad'][1:] for x in pieces])
        measured=np.concatenate([np.concatenate((x['state__q_rad'][:-1],x['state__support_q_rad'][:-1]),axis=-1)
                                 for x in pieces])
        native_hold=np.all(after==before,axis=-1)
        comparisons={}
        for mode,predict in predictors.items():
            result,action=jax.device_get(predict(context,jnp.asarray(actions),jnp.asarray(measured),jnp.asarray(before)))
            ref_error=np.abs(result.reference_q-after)
            action_error=np.abs(action-accepted)
            mismatch=(native_hold!=result.fallback_to_hold)&(ref_error.max(axis=-1)>1e-4)
            counts[mode]+=int(mismatch.sum())
            all_errors[mode]['reference'].append(ref_error.ravel())
            all_errors[mode]['accepted'].append(action_error.ravel())
            comparisons[mode]=dict(reference=summarize_errors([ref_error.ravel()]),
                accepted=summarize_errors([action_error.ravel()]),
                hold_mismatches=[dict(control=int(t),lane=int(lanes[i]),
                    native_hold=bool(native_hold[t,i]),world_hold=bool(result.fallback_to_hold[t,i]),
                    reference_error_max_rad=float(ref_error[t,i].max()),
                    position_violation=float(result.position_violation[t,i]),
                    orientation_violation=float(result.orientation_violation[t,i]),
                    linear_violation=float(result.linear_max_violation[t,i])) for t,i in np.argwhere(mismatch)])
        rows.append(dict(trial=trial,lanes=lanes.tolist(),comparisons=comparisons))
        write_json_unique(args.output_dir/f'trial_{trial:02d}.json',rows[-1],guard.budget)
        release_cpu_heap()
        print(f'[PROJECTION_VARIANTS] trial={trial+1} cumulative_hold_mismatch={counts}',flush=True)
    report.update(status='PROJECTION_VARIANT_DIAGNOSTIC_COMPLETE',trials=rows,
        population_sha256=file_hash(evidence),dataset_sha256=reader.identity,hold_mismatch_counts=counts,
        metrics={m:{k:summarize_errors(v) for k,v in e.items()} for m,e in all_errors.items()},
        production_control_changed=False,physical_fitting_fk_changed=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,900.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_projection_variant_audit_v1',training_version='WMJ-V1',status='INCOMPLETE',
            config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,
            source_sha256={p.name:file_hash(p) for p in paths},actor_updates=0,critic_updates=0,world_qualified=False)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,
            source={p.name:p.read_text() for p in paths}),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
