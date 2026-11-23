"""Replay saved projection inputs with local FP32 coordinate transforms only."""
from __future__ import annotations
import __future__
import argparse
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import time

import jax
import jax.numpy as jnp
from mujoco import mjx
from mujoco.mjx._src import smooth
import numpy as np

from cartesian_pose_action_projection_jax import matrix_to_euler_xyz_jax
from .audit_support_replay import load_support_chain
from .contracts import authenticate, config_hash, file_hash, load_config, repository_root
from .control_projection import ProjectionContract, ProjectionResult
from .free_imagination import make_projector as baseline_projector
from .free_orientation import remove_reference_attitude_backstops, position_only_projection
from .kinematics import forward_kinematics
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .projection_precision import make_projector, native_projector, coordinate_precision
from .runtime_guard import RunGuard
from .train_pose_trial import phase_budget


def native_methods(cfg, geometry, contract):
    """Execute the frozen native method with only its needed FK dependency.

    MJX forward's site/axis/anchor outputs come from smooth.kinematics;
    dynamics and contacts cannot affect these values. No trajectory is run.
    """
    model, chain=load_support_chain(cfg,geometry)
    arrays={k:getattr(contract,k) for k in ('q_lower','q_upper','velocity_limit','reference_horizon_s',
        'world_to_policy_rotation','policy_origin_world','reference_rotation','position_lower','position_upper',
        'outer_position_lower','outer_position_upper','roll_pitch_limit','outer_roll_pitch_limit')}
    names=('arm_lo','arm_hi','arm_vel_limit_rad_s','recovered_rmp_qvel_reference_error_horizon_s_per_joint',
        'cartesian_world_to_policy_rotation','cartesian_policy_origin_world','fixed_contact_racket_reference_rotation',
        'cartesian_pose_position_lower','cartesian_pose_position_upper','cartesian_pose_outer_position_lower',
        'cartesian_pose_outer_position_upper','cartesian_pose_roll_pitch_limit','cartesian_pose_outer_roll_pitch_limit')
    ids=np.asarray(chain.joint_ids);addresses=model.jnt_qposadr[ids]
    env=SimpleNamespace(**dict(zip(names,arrays.values())),dt=contract.dt_s,
        arm_qadr=addresses[:7],arm_vadr=model.jnt_dofadr[ids[:7]],arm_jids=ids[:7],racket_site_id=geometry['racket']['site_id'],
        cfg=SimpleNamespace(cartesian_pose_position_cbf_gain_s=contract.position_gain_s,
            cartesian_pose_orientation_cbf_gain_s=contract.orientation_gain_s,cartesian_pose_projection_sweeps=contract.sweeps,
            cartesian_pose_projection_tolerance=contract.tolerance))
    m=mjx.put_model(model);base=mjx.make_data(m)
    env.batched_forward=jax.vmap(lambda unused,d:smooth.kinematics(m,d),in_axes=(None,0))
    source=authenticate(repository_root(),cfg['authenticated_inputs']['xml']).parent/'mjx_juggle_env.py'
    if file_hash(source)!='39ba2da83f498cb84b041e016d6f1031130007dad9efafc21ead388f214e3b09':
        raise ValueError('Native source changed')
    method=next(n for n in ast.walk(ast.parse(source.read_text())) if isinstance(n,ast.FunctionDef) and n.name=='_project_cartesian_pose_reference')
    code=remove_reference_attitude_backstops(ast.unparse(method))
    namespace=dict(jnp=jnp,matrix_to_euler_xyz_jax=matrix_to_euler_xyz_jax,
        relative_xyz_euler_rate_jacobian_jax=lambda a,r,j:jnp.zeros_like(j),project_cartesian_pose_qvel_jax=position_only_projection)
    exec(compile(code,'<WMJ-native-free-projection-FK-replay>','exec',flags=__future__.annotations.compiler_flag),namespace)
    original=namespace[method.name];fixed=native_projector(original)
    def bind(fn):
        from mjx_topology_execution import make_collect
        def call(q,ref,v):
            n=q.shape[0];d=jax.tree.map(lambda x:jnp.broadcast_to(x,(n,)+x.shape),base)
            positions=jnp.broadcast_to(jnp.asarray(geometry['reset_fk']['qpos'],jnp.float32),(n,model.nq)).at[:,addresses].set(q)
            state=SimpleNamespace(model=None,arm_cmd_q=ref,data=d.replace(qpos=positions))
            return ProjectionResult(*fn(env,state,q[:,:7],v))
        return make_collect(call,mode='project')[0]
    return chain,bind(original),bind(fixed)


def delta(a,b,prefix):
    ref=np.max(np.abs(a.reference_q-b.reference_q),axis=-1)
    vel=np.max(np.abs(a.accepted_qvel-b.accepted_qvel),axis=-1)
    branch=a.fallback_to_hold!=b.fallback_to_hold
    return dict(reference_rad=distribution(ref[prefix]),accepted_qvel_rad_s=distribution(vel[prefix]),
        fallback_disagreements=int((prefix&branch).sum()),all_fallback_disagreements=int(branch.sum()),
        disagreement_locations=[dict(control=int(t)+1,lane=int(l)) for t,l in zip(*np.nonzero(prefix&branch))])


def audit(cfg,args,guard,report,plan):
    path=repository_root()/cfg['resources']['output_root']/'projection_consistency_audit_20261110_a02/result.json'
    source=json.loads(path.read_text());ref=source['replay_inputs'];archive=authenticate(repository_root(),ref)
    with np.load(archive,allow_pickle=False) as f:raw={k:f[k] for k in f.files}
    c=ProjectionContract(**{k:jnp.asarray(v,jnp.float32) if isinstance(v,list) else v for k,v in ref['projection_contract'].items()})
    chain,native_old,native_fixed=native_methods(cfg,ref['geometry'],c)
    q,reference,velocity=[jnp.asarray(raw[k]) for k in ('q','reference','velocity')];prefix=raw['prefix']
    old=baseline_projector();fixed=make_projector();results={};timings={}
    variants=dict(native_old=native_old,native_fixed=native_fixed,
        world_old=lambda q,r,v:old(chain,c,q,r,v),world_fixed=lambda q,r,v:fixed(chain,c,q,r,v))
    for name,fn in variants.items():
        @jax.jit
        def run(q,r,v):return jax.lax.scan(lambda carry,x:(carry,fn(*x)),None,(q,r,v))[1]
        guard.check();start=time.monotonic();results[name]=jax.device_get(run(q,reference,velocity))
        timings[name]=dict(compile_and_first_s=time.monotonic()-start)
        clocks=[]
        for _ in range(5):
            start=time.monotonic();jax.block_until_ready(run(q,reference,velocity));clocks.append(time.monotonic()-start)
        timings[name]['median_replay_s']=float(np.median(clocks));run.clear_cache()
        @jax.jit
        def slim(q,r,v):
            def tick(carry,x):
                p=fn(*x);return carry,(p.reference_q,p.fallback_to_hold)
            return jax.lax.scan(tick,None,(q,r,v))[1]
        sr,sb=jax.device_get(slim(q,reference,velocity));slim.clear_cache()
        p=results[name]
        report.setdefault('output_pruning',{})[name]=dict(reference_rad=distribution(np.max(np.abs(sr-p.reference_q),axis=-1)[prefix]),
            fallback_disagreements=int(((sb!=p.fallback_to_hold)&prefix).sum()))
    report['comparisons']={name:delta(results[name],results['native_fixed'],prefix) for name in results}
    @jax.jit
    def positions(q,ref):
        def tick(carry,x):
            full=x[0].at[:,:7].set(x[1])
            with jax.default_matmul_precision('highest'):
                fk=jax.vmap(lambda z:forward_kinematics(chain,z))(full)
            before=fk.site_position-c.policy_origin_world[None]
            regular=jnp.einsum('ij,nj->ni',c.world_to_policy_rotation,before)
            exact=coordinate_precision.einsum('ij,nj->ni',c.world_to_policy_rotation,before)
            return carry,(fk.site_position,regular,exact)
        return jax.lax.scan(tick,None,(q,ref))[1]
    p,low,high=jax.device_get(positions(q,reference));positions.clear_cache()
    lane=plan['source_indices'].index(38);t=11
    report['source38_control12']=dict(world_site_xyz_m=p[t,lane].tolist(),default_transform_xyz_m=low[t,lane].tolist(),
        fp32_transform_xyz_m=high[t,lane].tolist(),outer_z_m=float(c.outer_position_upper[2]),tolerance_m=c.tolerance,
        captured_native_fallback=bool(raw['fallback'][t,lane]),
        candidates={name:{k:np.asarray(getattr(v,k)[t,lane]).tolist() for k in v._fields} for name,v in results.items()})
    direction=jnp.linspace(-.7,.7,32*7,dtype=jnp.float32).reshape((32,7))
    def f(v):return fixed(chain,c,q[t],reference[t],v).reference_q
    derivative=jax.jit(lambda v,d:jax.jvp(f,(v,),(d,))[1])(velocity[t],direction)
    fd=(jax.jit(f)(velocity[t]+.01*direction)-jax.jit(f)(velocity[t]-.01*direction))/.02
    report['action_gradient']=dict(finite=bool(np.isfinite(derivative).all()),
        central_difference_absolute_error=distribution(np.abs(np.asarray(fd-derivative))),epsilon_qvel_rad_s=.01)
    report.update(status='FROZEN_COORDINATE_PRECISION_REPLAY_COMPLETE',input_report=dict(path=str(path),sha256=file_hash(path)),
        replay_inputs=dict(path=str(archive),sha256=file_hash(archive)),timings=timings,
        position_transform_error_m=distribution(np.max(np.abs(low-high),axis=-1)[prefix]),
        parameter_updates=0,default_changed=False,native_fk_only=True,
        sources_unchanged=all(file_hash(Path(__file__).parent/n)==h for n,h in plan['preserved_sources'].items()),
        scope='same archived4096 projection inputs; prefix1862; no new trajectories or network updates')
    print('[PRECISION] '+str(report['comparisons']),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gpu-uuid',required=True)
    args=p.parse_args();cfg=load_config();setting=cfg['projection_consistency_candidate']
    plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if not setting['authorized'] or setting['default_enabled']:raise ValueError('Frozen candidate only')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        report=dict(schema='wmj_v1_projection_coordinate_precision',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),source_sha256={n:file_hash(Path(__file__).parent/n)
                for n in ('projection_precision.py','audit_projection_precision.py')})
        write_compact_json(args.output_dir/'implementation.json',dict(sources={n:(Path(__file__).parent/n).read_text()
            for n in ('projection_precision.py','audit_projection_precision.py')}),guard.budget)
        try:
            if jax.default_backend()!='gpu':raise ValueError('CUDA required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:report['error']=repr(exc);raise
        finally:report['resources']=guard.report();write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
