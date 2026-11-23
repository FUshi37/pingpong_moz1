"""Frozen native-versus-imagination projection branch diagnosis, no updates."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .audit_support_replay import load_support_chain
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .control_kinematics import extract_control_chain
from .control_projection import ProjectionResult
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .projection_consistency import make_capture, make_quaternion_projector
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget


def summarize(prediction, actual, reference, prefix, indices):
    error = np.max(np.abs(prediction.reference_q-reference), axis=-1)
    mismatch = prediction.fallback_to_hold != actual['fallback'].astype(bool)
    rows = []
    for t, lane in zip(*np.nonzero(prefix & (mismatch | (error > 1e-5)))):
        rows.append(dict(source_index=indices[lane], control=int(t)+1,
            reference_error_rad=float(error[t,lane]), native={k:float(v[t,lane]) for k,v in actual.items()},
            predicted={key:np.asarray(getattr(prediction,key)[t,lane]).tolist() for key in prediction._fields}))
    return dict(reference_error_rad=dict(all=distribution(error), task_prefix=distribution(error[prefix])),
        fallback_disagreements=dict(all=int(mismatch.sum()), task_prefix=int((mismatch & prefix).sum())),
        differing_task_prefix_cases=rows[:64], differing_task_prefix_count=len(rows),
        nonfinite=sum(int((~np.isfinite(v)).sum()) for v in prediction))


def audit(cfg,args,guard,report,plan):
    from mjx_topology_execution import make_collect
    started=time.monotonic(); data=make_capture()(cfg,args,guard,report,plan)
    report['capture_wall_s']=time.monotonic()-started
    env=data['environment']; wb=data['world_bank']; c=data['control']; context=data['context']
    q=jnp.asarray(wb.physics.q_rad[:-1]); reference=jnp.asarray(wb.control.reference_q_rad[:-1]); actions=jnp.asarray(data['actions'])
    denominator=c.projection.velocity_limit[None]*c.action_velocity_scale*context.action_scale_mult[:,None]
    velocity=jnp.clip(jnp.clip(actions,-1.,1.)*denominator,-c.projection.velocity_limit[None],c.projection.velocity_limit[None])
    model,chain=load_support_chain(cfg,data['geometry'])
    quaternion=extract_control_chain(model,data['geometry']['reset_fk']['qpos'],joint_ids=chain.joint_ids,
        site_id=data['geometry']['racket']['site_id'],geom_id=data['geometry']['racket']['geom_id'])
    addresses=model.jnt_qposadr[np.asarray(chain.joint_ids)]
    template=data['native_tail']
    native_project,_=make_collect(lambda state,v:env._project_cartesian_pose_reference(
        state,state.data.qpos[:,env.arm_qadr],v),mode='project')
    def native(q,ref,v):
        state=template._replace(arm_cmd_q=ref,data=template.data.replace(qpos=template.data.qpos.at[:,addresses].set(q)))
        return ProjectionResult(*native_project(state,v))
    def replay(fn):
        @jax.jit
        def run(q,ref,v):
            return jax.lax.scan(lambda carry,x:(carry,fn(*x)),None,(q,ref,v))[1]
        guard.check();start=time.monotonic();result=jax.device_get(run(q,reference,velocity))
        elapsed=time.monotonic()-start;run.clear_cache()
        return result,elapsed
    actual={k:np.stack([row[k] for row in data['native_bounds']]) for k in ('fallback','nonlinear_backstop','current_safe','position_violation','linear_violation')}
    buffer=io.BytesIO()
    np.savez_compressed(buffer,q=np.asarray(q),reference=np.asarray(reference),velocity=np.asarray(velocity),
        expected_reference=wb.control.reference_q_rad[1:],prefix=data['prefix'],**actual)
    payload=buffer.getvalue();guard.budget.check(additional_bytes=2*len(payload)+8192)
    path=args.output_dir/'projection_inputs.npz'
    with path.open('xb') as stream:stream.write(payload)
    report['replay_inputs']=dict(path=str(path),sha256=file_hash(path),bytes=len(payload),
        geometry=data['geometry'],projection_contract={k:np.asarray(v).tolist() if hasattr(v,'shape') else v for k,v in c.projection.__dict__.items()})
    predicted={}; timings={}
    variants={'native_standalone':native,
        'matrix_baseline':lambda q,ref,v:data['runtime'].projector(chain,c.projection,q,ref,v),
        'quaternion_candidate':lambda q,ref,v:make_quaternion_projector(quaternion)(chain,c.projection,q,ref,v)}
    for name,fn in variants.items():
        predicted[name],timings[name]=replay(fn)
        report.setdefault('comparisons',{})[name]=summarize(predicted[name],actual,wb.control.reference_q_rad[1:],data['prefix'],plan['source_indices'])
        print('[PROJECTION] '+name+' '+str({k:v for k,v in report['comparisons'][name].items() if k!='differing_task_prefix_cases'}),flush=True)
    lane=plan['source_indices'].index(38);tick=11
    report['source38_control12']=dict(measured_q_rad=np.asarray(q[tick,lane]).tolist(),reference_q_rad=np.asarray(reference[tick,lane]).tolist(),
        raw_action=np.asarray(actions[tick,lane]).tolist(),nominal_qvel_rad_s=np.asarray(velocity[tick,lane]).tolist(),
        actual={k:float(v[tick,lane]) for k,v in actual.items()},
        predictions={name:{k:np.asarray(getattr(p,k)[tick,lane]).tolist() for k in p._fields} for name,p in predicted.items()})
    after=numeric_tree_digest((data['actor'],data['robot'],data['auxiliary']))
    if after!=data['frozen']:raise ValueError('Frozen weights changed')
    report.update(status='FROZEN_PROJECTION_DIAGNOSIS_COMPLETE',parameters_unchanged=True,default_changed=False,
        parameter_updates=0,timings_s=timings,old_sources_preserved=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args();cfg=load_config()
    setting=cfg['projection_consistency_candidate'];plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or setting['default_enabled'] or plan['max_phase_bytes']!=2*2**20:
        raise ValueError('Only the authorized independent frozen probe may run')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_projection_consistency',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))},parameter_updates=0)
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text()
            for n in ('projection_consistency.py','audit_projection_consistency.py')}),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
