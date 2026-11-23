"""Physical finite differences along frozen world-return bias directions."""

import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from . import audit_attribution_native as native
from .artifacts import write_json_unique
from .attribution_common import population,actors,plan_and_budget,worker_check,source_report
from .contracts import file_hash,load_config
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .native_evaluation import build_evaluation_step
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def native_reward_step(p,lanes,cfg):
    """Reuse the audited reward observer, preserving the full native loop."""
    node=next(n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(native.audit))))
        if isinstance(n,ast.FunctionDef) and n.name=='tick')
    source=ast.unparse(node)
    factory='def factory(p,env,adapter,goals,lanes,native_step,reward,episode):\n'+textwrap.indent(source,'    ')+'\n    return tick\n'
    namespace=dict(native.__dict__);exec(compile(factory,'<frozen-native-reward-probe>','exec'),namespace)
    step,hook=build_evaluation_step(p['env'],p['bundle'].stack,perturb=False,trace_observer=native.contact_observer)
    reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients'])
    fn=namespace['factory'](p,p['env'],p['adapter'],p['goals'],lanes,step,reward,EpisodeContract.from_config(cfg))
    return fn,dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(),hook=hook)


def audit(cfg,args,guard,report):
    from mjx_topology_execution import make_collect
    plan=plan_and_budget(cfg,guard);p=population(cfg);parameters=actors(cfg)
    lanes=np.asarray(plan['selected_lane_ids']);frozen=numeric_tree_digest(parameters)
    world_path=args.world_dir/'result.json';world=json.loads(world_path.read_text())
    if world['status']!='WORLD_FROZEN_ATTRIBUTION_COMPLETE' or world['selected_case_ids']!=plan['selected_case_ids']:
        raise ValueError('A complete matching frozen world gradient report is required')
    fn,identity=native_reward_step(p,lanes,cfg)
    scoped,execution=make_collect(fn,mode='project');step=jax.jit(scoped)
    keys=jax.random.split(jax.random.PRNGKey(plan['seed']+2),128)
    gamma=cfg['imagination']['gamma_5ms'];report.update(world_reference=dict(path=str(world_path),sha256=file_hash(world_path)),
        selected_case_ids=plan['selected_case_ids'],horizon_controls=64,exploration_std=0.,
        source=identity,execution=execution,probes={},scope='No optimizer;temporary output-bias offsets only. Native direct reward compared with model direct-reward derivative;critic bootstrap excluded from comparison.')
    for name,actor in parameters.items():
        g=world['gradients'][name];jacobian=np.asarray(g['output_bias_jacobian']);terms=g['term_names'][:-1]
        direction=jacobian.sum(axis=(0,1));direction/=max(float(np.linalg.norm(direction)),1e-12)
        expected=np.einsum('ncj,j->nc',jacobian[:,:-1],direction)
        data={}
        for offset in (0.,.0003,-.0003,.001,-.001):
            # These are immutable diagnostic arguments, never checkpoint writes.
            probe_actor=dict(actor,out=dict(actor['out'],b=actor['out']['b']+jnp.float32(offset)*jnp.asarray(direction,jnp.float32)))
            state=(p['initial'],p['history'],p['task'],p['events'],keys)
            returns=np.zeros((len(lanes),len(terms)));event_modes=[]
            for tick in range(64):
                worker_check(guard)
                result=step(*state,probe_actor,jnp.float32(0));state=result[:5]
                values=jax.device_get(result[6]);trace=jax.device_get(result[5])
                for column,term in enumerate(terms):returns[:,column]+=gamma**tick*values[term][lanes]
                event_modes.append(np.stack((trace['valid_release'][:,lanes].sum(axis=0),
                    trace['valid_apex'][:,lanes].sum(axis=0),np.asarray(state[2].failure_code)[lanes]),axis=-1))
            data[offset]=(returns,np.asarray(event_modes))
        probes=[]
        for eps in (.0003,.001):
            plus,m1=data[eps];minus,m2=data[-eps];fd=(plus-minus)/(2*eps)
            probes.append(dict(epsilon=eps,native_central_total=float(fd.sum()),
                model_automatic_direct_total=float(expected.sum()),
                same_200hz_event_modes=bool(np.array_equal(m1,m2)),
                native_per_case_terms=fd.tolist(),model_per_case_terms=expected.tolist(),
                native_plus_return_terms=plus.tolist(),native_minus_return_terms=minus.tolist(),
                warning='Matching200Hz modes does not certify all1ms contact/solver modes;small finite differences may be noisy.'))
        report['probes'][name]=dict(direction=direction.tolist(),term_names=terms,
            native_baseline_return_terms=data[0.][0].tolist(),rows=probes)
        print(f'[NATIVE-FD] {name} '+str([{k:x[k] for k in ('epsilon','native_central_total','model_automatic_direct_total','same_200hz_event_modes')} for x in probes]),flush=True)
    report.update(status='NATIVE_FROZEN_GRADIENT_ATTRIBUTION_COMPLETE',parameters_unchanged=numeric_tree_digest(parameters)==frozen)
    if not report['parameters_unchanged']:raise ValueError('Checkpoint actor parameters changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--world-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        plan_and_budget(cfg,guard);args.output_dir.mkdir(parents=True,exist_ok=False);report=source_report(cfg,args)
        write_json_unique(args.output_dir/'implementation.json',dict(report=report,source=Path(__file__).read_text()),guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('CUDA required')
            audit(cfg,args,guard,report)
        except Exception as exc:report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report();write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
