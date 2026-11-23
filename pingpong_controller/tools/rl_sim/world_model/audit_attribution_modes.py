"""Frozen own-policy world probes with actual1ms contact-mode comparisons."""

import argparse
import ast
import inspect
import json
from pathlib import Path
import textwrap

import jax
import jax.numpy as jnp
import numpy as np

from . import audit_attribution_world as source
from .artifacts import write_json_unique
from .attribution_common import population,actors,frozen_models,plan_and_budget,worker_check,source_report
from .audit_action_effects import initial_world
from .contracts import file_hash,load_config
from .frozen_task import init_evaluation
from .full_prior_world import world_actor_observation
from .imagined_env import EpisodeContract
from .imagined_rewards import RewardContract
from .migration import actor_mean
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest


def build_mode_step(bindings):
    """Clone only the diagnostic evaluator; expose already-produced1ms labels."""
    node=next(n for n in ast.walk(ast.parse(textwrap.dedent(inspect.getsource(source.audit))))
        if isinstance(n,ast.FunctionDef) and n.name=='step')
    class Trace(ast.NodeTransformer):
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='output' for t in node.targets):
                return [node,*ast.parse("output=dict(output,contact_active=labels['contact_active'],contact_entry=labels['contact_entry'])").body]
            return self.generic_visit(node)
    node=ast.fix_missing_locations(Trace().visit(node));code=ast.unparse(node)
    names=tuple(bindings)
    factory='def factory('+','.join(names)+'):\n'+textwrap.indent(code,'    ')+'\n    return step\n'
    namespace=dict(source.__dict__);exec(compile(factory,'<frozen-world1ms-modes>','exec'),namespace)
    return namespace['factory'](**bindings)


def audit(cfg,args,guard,report):
    plan=plan_and_budget(cfg,guard);p=population(cfg);lanes=np.asarray(plan['selected_lane_ids'])
    robot,auxiliary,physics,refs=frozen_models(cfg,guard.budget)
    world,context,events=initial_world(dict(native=p['initial'],events=p['events']),p['env'],physics,lanes)
    goals=p['goals'][lanes];n=len(lanes);parameters=actors(cfg);frozen=numeric_tree_digest((parameters,robot,auxiliary))
    task=init_evaluation(goals,jax.random.PRNGKey(plan['seed']+1),jnp.asarray(lanes));gamma=cfg['imagination']['gamma_5ms']
    prior_path=args.world_dir/'result.json';prior=json.loads(prior_path.read_text())
    if prior['status']!='WORLD_FROZEN_ATTRIBUTION_COMPLETE':raise ValueError('Missing prior world gradients')
    step=build_mode_step(dict(robot=robot,auxiliary=auxiliary,context=context,control=p['control'],goals=goals,
        physics=physics,n=n,reward=RewardContract(**cfg['imagination_implementation']['reward']['coefficients']),
        episode=EpisodeContract.from_config(cfg)))
    terms=source.TERM_NAMES
    @jax.jit
    def run(actor,bias):
        actor=dict(actor,out=dict(actor['out'],b=actor['out']['b']+bias))
        def tick(carry,_):
            state,total=carry
            raw=actor_mean(actor,world_actor_observation(state[0],goals[:,0]))
            after,output=step(state,raw,jnp.ones(n,bool));labels,rewards,_,_,_=output
            rewards=jnp.stack([rewards[k] for k in terms],axis=-1)
            modes=jnp.stack([labels[k] for k in ('contact_active','contact_entry','valid_release','valid_apex')],axis=-1)
            return (after,total+1),(rewards,modes,after[1].failure_code)
        _,(rewards,modes,failures)=jax.lax.scan(tick,((world,task,events),jnp.int32(0)),None,length=64)
        discounted=(rewards*jnp.asarray(gamma)**jnp.arange(64)[:,None,None]).sum(axis=0)
        return discounted,modes,failures
    report.update(prior=dict(path=str(prior_path),sha256=file_hash(prior_path)),selected_case_ids=plan['selected_case_ids'],
        model_references=refs,horizon_controls=64,exploration_std=0.,term_names=list(terms),runs={})
    for name,actor in parameters.items():
        gradient=np.asarray(prior['gradients'][name]['output_bias_jacobian'])
        direction=gradient.sum(axis=(0,1));direction/=max(np.linalg.norm(direction),1e-12)
        expected=np.einsum('ncj,j->nc',gradient[:,:-1],direction)
        zero=jax.device_get(run(actor,jnp.zeros(7)));old=np.asarray(prior['gradients'][name]['per_case_discounted_return_terms'])[:,:-1]
        rows=[]
        for eps in (1e-5,3e-5,1e-4,3e-4,.001,.003,.01):
            worker_check(guard)
            plus=jax.device_get(run(actor,jnp.asarray(eps*direction,jnp.float32)))
            minus=jax.device_get(run(actor,jnp.asarray(-eps*direction,jnp.float32)))
            fd=(plus[0]-minus[0])/(2*eps);different=plus[1]!=minus[1]
            changed=np.argwhere(np.any(different,axis=-1))
            rows.append(dict(epsilon=eps,automatic_direct=float(expected.sum()),central_direct=float(fd.sum()),
                per_case_central=fd.sum(axis=-1).tolist(),per_case_automatic=expected.sum(axis=-1).tolist(),
                mode_changes_1ms=int(np.any(different,axis=-1).sum()),failure_clock_changed=bool(np.any(plus[2]!=minus[2])),
                changes=[dict(control=int(t),substep=int(s),case_id=plan['selected_case_ids'][i],
                    plus=plus[1][t,s,i].tolist(),minus=minus[1][t,s,i].tolist()) for t,s,i in changed],
                changed_mode_order=['contact_active','contact_entry','valid_release','valid_apex']))
        report['runs'][name]=dict(direction=direction.tolist(),zero_bias_return_terms=zero[0].tolist(),
            source_direct_return_max_abs_delta=float(np.max(np.abs(zero[0]-old))),rows=rows)
        print(f'[MODE-FD] {name} '+str([{k:r[k] for k in ('epsilon','automatic_direct','central_direct','mode_changes_1ms','failure_clock_changed')} for r in rows]),flush=True)
    report.update(status='FROZEN_1MS_MODE_ATTRIBUTION_COMPLETE',parameters_unchanged=numeric_tree_digest((parameters,robot,auxiliary))==frozen)
    if not report['parameters_unchanged']:raise ValueError('Frozen model changed')


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
