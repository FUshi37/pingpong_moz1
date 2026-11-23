"""Frozen full native/controller pairing of the local coordinate precision fix."""
from __future__ import annotations
import argparse
import inspect
import json
from pathlib import Path
from types import MethodType
import time

import jax
import jax.numpy as jnp
import numpy as np

from .audit_error_attribution import control_diagnostics
from .audit_control_components import make_component_replays
from .audit_free_imagination import audit as original_capture
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .error_attribution import instrument_with_solver
from .free_orientation import install_position_only_reference
from .native_entry_collision import write_compact_json
from .projection_consistency import capture_source, NATIVE_METRICS
from .projection_precision import native_projector, make_runtime
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_pose_trial import phase_budget
from .velocity_reward import clone_globals


def install_candidate(env,stack):
    info=install_position_only_reference(env,stack)
    env._project_cartesian_pose_reference=MethodType(native_projector(env._project_cartesian_pose_reference.__func__),env)
    return dict(info,coordinate_matmul_precision='highest',bounds_and_inequalities_unchanged=True,
        native_halfspace_EXEC_unchanged=True,default_native_class_unchanged=True)


def audit(cfg,args,guard,report,plan):
    namespace=dict(original_capture.__globals__,instrument_environment=instrument_with_solver,
        install_position_only_reference=install_candidate,make_runtime=make_runtime,NATIVE_METRICS=NATIVE_METRICS)
    exec(compile(capture_source(inspect.getsource(original_capture)),'<WMJ-coordinate-fix-native-capture>','exec'),namespace)
    start=time.monotonic();data=namespace[original_capture.__name__](cfg,args,guard,report,plan)
    report['capture_wall_s']=time.monotonic()-start
    for name in ('q_before','dq_before','q_after','dq_after'):
        data['raw'][name]=np.concatenate((data['raw'][name],data['raw']['support_'+name]),axis=-1)
    report['control_components']=control_diagnostics(data,guard,plan['source_indices'])
    rt=data['runtime'];wb=data['world_bank'];prefix=data['prefix']
    replay=clone_globals(make_component_replays,project_reference=rt.projector)(rt.options['chain'],data['control'])['projection']
    p=jax.device_get(replay(data['context'],jnp.asarray(data['actions']),jnp.asarray(wb.physics.q_rad[:-1]),
        jnp.asarray(wb.control.reference_q_rad[:-1])))
    replay.clear_cache()
    native=np.stack([row['fallback'] for row in data['native_bounds']]).astype(bool)
    mismatch=p['fallback']!=native
    report['branch_pair']=dict(all_controls=int(mismatch.size),native_prefix_controls=int(prefix.sum()),
        all_disagreements=int(mismatch.sum()),native_prefix_disagreements=int((mismatch&prefix).sum()),
        mismatch_locations=[dict(source_index=plan['source_indices'][l],control=int(t)+1) for t,l in zip(*np.nonzero(mismatch&prefix))])
    lane=plan['source_indices'].index(38);tick=11
    report['source38_control12']=dict(native_reference=wb.control.reference_q_rad[tick+1,lane].tolist(),
        world_reference=p['reference'][tick,lane].tolist(),native_fallback=bool(native[tick,lane]),world_fallback=bool(p['fallback'][tick,lane]),
        native_position_violation_m=float(data['native_bounds'][tick]['position_violation'][lane]))
    after=numeric_tree_digest((data['actor'],data['robot'],data['auxiliary']))
    if after!=data['frozen']:raise ValueError('Frozen weights changed')
    report.update(status='FROZEN_COORDINATE_PRECISION_INTEGRATION_COMPLETE',parameters_unchanged=True,
        parameter_updates=0,default_changed=False,full_model_or_policy_qualification=False,
        preserved_sources=all(file_hash(Path(__file__).parent/n)==h for n,h in plan['preserved_sources'].items()))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config();setting=cfg['projection_consistency_candidate'];plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or setting['default_enabled']:raise ValueError('Independent frozen candidate only')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_projection_precision_integrated',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_sha256={n:file_hash(Path(__file__).parent/n) for n in ('projection_precision.py','audit_projection_precision_integrated.py')})
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text() for n in report['source_sha256']}),guard.budget)
        try:
            if jax.default_backend()!='gpu':raise ValueError('CUDA required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
