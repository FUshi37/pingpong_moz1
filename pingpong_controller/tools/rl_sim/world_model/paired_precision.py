"""Private paired-training bindings; preserved production and pilot paths stay intact."""
from __future__ import annotations
import ast
import inspect
import json
import textwrap
from pathlib import Path
from typing import Any, Callable
import jax
import jax.numpy as jnp
import numpy as np
from .actor_critic import actor_return, frozen_value, lambda_returns
from .branch_bank import BranchSeedBank
from .contracts import authenticate, config_hash, file_hash, repository_root
from .course_learner import CompactCourseWindow
from .course_rollout import validate_course_window
from .imagination_checkpoint import restore_checkpoint
from .imagined_rollout import window_batch
from .mixed_precision import make_runtime
from .module_gradient_tools import private_clone
from .pilot_checkpoint import pilot_signature
from .pilot_runtime import setup_pilot, prepare_pilot_session
from .state_adapter import numeric_tree_digest


def random32_clone(function: Callable, overrides: dict | None = None) -> Callable:
    """Keep native random normals/indices independent of the ambient x64 flag."""
    tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
    class FixRandom(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            name=ast.unparse(node.func)
            if name in ('jax.random.normal','jax.random.randint') and not any(k.arg=='dtype' for k in node.keywords):
                dtype='float32' if name.endswith('normal') else 'int32'
                node.keywords.append(ast.keyword(arg='dtype',value=ast.Attribute(
                    value=ast.Name(id='jnp',ctx=ast.Load()),attr=dtype,ctx=ast.Load())))
            return node
    tree=FixRandom().visit(tree);namespace=dict(function.__globals__);namespace.update(overrides or {})
    exec(compile(ast.fix_missing_locations(tree),'<WMJ-paired-private-RNG>','exec'),namespace)
    return namespace[function.__name__]


def learning_state(session: Any) -> dict:
    """Every learning/RNG counter is inherited; only the physical pool is rebuilt."""
    return {name:getattr(session,name) for name in session._fields if name!='pool'}


def mixed_lambda_targets(batch: Any, values: Any, *, lambda_value: float) -> Any:
    """Use reward precision for the recurrence, then the FP32 critic boundary."""
    return lambda_returns(batch,values.astype(batch.reward.dtype),lambda_value=lambda_value).astype(jnp.float32)


def compact_window(window: Any, target: Any, *, lambda_value: float = .99) -> Any:
    validate_course_window(window)
    valid=np.asarray(window.transitions.valid,bool)
    values=frozen_value(target,window.transitions.next_critic_features)
    targets=np.asarray(mixed_lambda_targets(window_batch(window),values,lambda_value=lambda_value))[valid]
    inputs=np.asarray(window.critic_inputs,dtype=np.float32)[valid]
    transitions=window.transitions._replace(next_critic_features=jnp.zeros(valid.shape+(0,),jnp.float32))
    return CompactCourseWindow(*jax.device_get((window.environment,window.exploration_rng,
        transitions,inputs,targets,window.captures)))


def make_gradient(runtime: Any, table: Any, options: dict) -> Callable:
    from .course_rollout import course_policy_window
    from .course_env import course_critic_features
    def features(state):
        return course_critic_features(state).astype(jnp.float32)
    def step(*args,**kwargs):
        state,transition=runtime.course_step(*args,**kwargs)
        return state,transition._replace(next_critic_features=transition.next_critic_features.astype(jnp.float32))
    window_fn=random32_clone(course_policy_window,dict(course_step=step,course_critic_features=features))
    def objective(actor,target,environment,keys,shape_template):
        window=window_fn(actor,runtime.robot,runtime.auxiliary,environment,keys,table,
            horizon=shape_template.shape[0],**options)
        batch=window_batch(window)
        values=frozen_value(target,window.transitions.next_critic_features).astype(batch.reward.dtype)
        return -jnp.mean(actor_return(batch,values)),window
    return jax.jit(jax.value_and_grad(objective,argnums=0,has_aux=True))


prepare_session=random32_clone(prepare_pilot_session)


def setup(cfg: dict, guard: Any, mode: str, plan: dict) -> tuple:
    """Restore pinned u20 learning state, then attach the same new native pool."""
    with jax.enable_x64(False):
        fresh,bank,table,robot,auxiliary,options,evidence=setup_pilot(cfg,guard)
        signature=pilot_signature(cfg,evidence)
        path=authenticate(repository_root(),plan['parent_learner']);manifest=json.loads(path.read_text())
        if manifest['metadata']['kind']!='bounded_S01_pilot_learner' or manifest['metadata']['actor_updates']!=20:
            raise ValueError('Paired trial requires the exact u20 full learner')
        parent=restore_checkpoint(plan['parent_learner'],dict(learner=fresh),signature=signature)['learner']
        if (int(parent.actor_updates)!=20 or int(parent.critic_updates)!=80
                or int(parent.actor_optimizer.step)!=20 or int(parent.critic_optimizer.step)!=80):
            raise ValueError('Parent optimizer counters changed')
        inherited_hash=numeric_tree_digest(learning_state(parent))
        session=parent._replace(pool=fresh.pool._replace(
            imagined_transition_words=parent.pool.imagined_transition_words,windows=parent.pool.windows))
    physics={k:v for k,v in options.items() if k not in
             ('control_contract','episode_contract','reward_contract','stage_id','exploration_std')}
    with jax.enable_x64(mode=='mixed'):
        runtime=make_runtime(mode,robot,auxiliary,physics,options['control_contract'])
        session=session._replace(pool=session.pool._replace(environment=runtime.prepare_course(session.pool.environment)))
        tree,metadata=bank.snapshot()
        identity=config_hash(dict(models=evidence['model_references'],precision=mode,plan=cfg['paired_precision_trial']['plan']['sha256']))
        metadata=dict(metadata,model_identity=identity,
                      rows=[dict(r,model_identity=identity) for r in metadata['rows']])
        tree=dict(tree,environment=jax.device_get(runtime.prepare_course(tree['environment'])))
        bank=BranchSeedBank.restore(tree,metadata,model_identity=identity,stage_id='S01')
        options=dict(options,**runtime.options,control_contract=runtime.control)
        if numeric_tree_digest(learning_state(session))!=inherited_hash:
            raise ValueError('Precision conversion changed parent learning state')
    evidence=dict(evidence,parent_learner=plan['parent_learner'],parent_signature=signature,
        inherited_learning_sha256=inherited_hash,initial_session_sha256=numeric_tree_digest(session),
        numeric_mode=mode,seed_bank_model_identity=identity,
        inherited_actor_updates=20,inherited_critic_updates=80,
        old_imagined_states_reused=0,common_initial_seed_order=True)
    return session,bank,table,runtime,options,evidence


def signature(cfg: dict, mode: str, evidence: dict, microbatch: int) -> str:
    names=('paired_precision.py','train_paired.py','mixed_precision.py','course_rollout.py',
           'course_env.py','course_control.py','pilot_learning.py','imagination_checkpoint.py')
    return config_hash(dict(schema='wmj_v1_paired_precision_runtime',mode=mode,
        parent=evidence['parent_signature'],plan=cfg['paired_precision_trial']['plan'],
        microbatch=microbatch,model_references=evidence['model_references'],
        sources={n:file_hash(Path(__file__).parent/n) for n in names}))


def load_actor(cfg: dict, reference: dict) -> tuple:
    from .migration import expand_actor_history,load_v321_actor,_validate_actor
    path=authenticate(repository_root(),reference);payload=json.loads(path.read_text());meta=payload['metadata']
    if (meta['kind']!='paired_precision_actor' or meta['actor_updates']!=30
            or meta['new_actor_updates']!=10 or meta['teacher_transitions_used_for_learning']!=0
            or meta['plan']!=cfg['paired_precision_trial']['plan']):
        raise ValueError('Unexpected paired actor provenance')
    template=dict(actor=expand_actor_history(load_v321_actor(repository_root(),cfg['lineage']['actor_source'])))
    actor=restore_checkpoint(reference,template,signature=payload['runtime_signature'])['actor']
    _validate_actor(actor,969)
    return actor,meta
