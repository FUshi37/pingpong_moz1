"""Private V324 task hook: observe native contacts and apply approved forces."""

import ast
from functools import partial
import hashlib
import importlib
import inspect
from pathlib import Path
import textwrap
import types

import jax
import jax.numpy as jnp

from .contracts import file_hash
from .frozen_task import consume_events, fail_evaluation
from .physical_events import physical_event_substep
from .teacher_trace import contact_pair_masks, point_twist


class _EvaluationHook(ast.NodeTransformer):
    def __init__(self):
        self.loops=0
        self.returns=0

    def visit_FunctionDef(self,node):
        if node.name=='_step_impl':
            node.args.kwonlyargs.extend([ast.arg(arg='wmj_payload'),ast.arg(arg='wmj_goals')])
            node.args.kw_defaults.extend([None,None])
        return self.generic_visit(node)

    def visit_Assign(self,node):
        call=node.value
        if (isinstance(call,ast.Call) and ast.unparse(call.func)=='jax.lax.fori_loop'
                and len(call.args)==4 and ast.unparse(call.args[2])=='one_substep'):
            self.loops+=1
            hook=ast.Call(func=ast.Name('_wmj_evaluation_loop',ast.Load()),
                args=[ast.Name('self',ast.Load()),*call.args,
                      ast.Name('wmj_payload',ast.Load()),ast.Name('wmj_goals',ast.Load())],keywords=[])
            assignment=ast.Assign(targets=[ast.Tuple(elts=[ast.Name(n,ast.Store())
                for n in ('_wmj_result','_wmj_payload','_wmj_trace')],ctx=ast.Store())],value=hook)
            node.value=ast.Name('_wmj_result',ast.Load())
            return [assignment,node]
        return self.generic_visit(node)

    def visit_Return(self,node):
        if (isinstance(node.value,ast.Tuple) and len(node.value.elts)==5
                and ast.unparse(node.value.elts[-1])=='metrics'):
            self.returns+=1
            return ast.parse("metrics['wmj_payload']=_wmj_payload\nmetrics['wmj_trace']=_wmj_trace").body+[node]
        return self.generic_visit(node)


def augment_source(source):
    hook=_EvaluationHook()
    tree=hook.visit(ast.parse(textwrap.dedent(source)))
    if hook.loops!=1 or hook.returns!=1:
        raise ValueError('Expected exactly one native physics loop and one native metrics return')
    return ast.unparse(ast.fix_missing_locations(tree))


def native_event_trace(env,before,after):
    pair,other=contact_pair_masks(after.contact.geom,after.contact.dist,
        ball_geom_id=env.ball_geom_id,racket_geom_id=env.racket_contact_geom_id)
    active=pair&(after.contact.dist<=0.)
    addresses=after.contact.efc_address
    force=jnp.take(after.efc_force,jnp.maximum(addresses,0),axis=-1)
    force=jnp.sum(jnp.where(active&(addresses>=0),force,0.),axis=-1)
    support=env._wmj_eval_twist
    position=after.geom_xpos[:,env.racket_contact_geom_id]
    linear,angular=point_twist(position,after.xaxis[:,support['jids']],
        after.xanchor[:,support['jids']],before.qvel[:,support['vadr']],support['hinge_mask'])
    return dict(contact_overlap=jnp.any(active,axis=-1),contact_normal_force_n=force,
        racket_linear_velocity_m_s=linear,racket_angular_velocity_rad_s=angular,
        racket_position_m=position,
        other_ball_collision=jnp.any(other,axis=-1),
        ball_qpos_before=before.qpos[:,env.ball_qadr:env.ball_qadr+7],
        ball_qpos_after=after.qpos[:,env.ball_qadr:env.ball_qadr+7],
        ball_qvel_before=before.qvel[:,env.ball_vadr:env.ball_vadr+6],
        ball_qvel_after=after.qvel[:,env.ball_vadr:env.ball_vadr+6])


def merge_observer_trace(output,observer,env,before,after,labels):
    """Opt-in numerical observations cannot replace existing physical/task labels."""
    if observer is None:
        return output
    extra=observer(env,before,after,labels)
    if set(extra)&set(output):
        raise ValueError('Diagnostic observer cannot overwrite evaluation labels')
    return dict(output,**extra)


def evaluation_loop(env,lower,upper,substep,initial,payload,goals,*,observer=None):
    if lower!=0 or upper!=5 or float(env.timestep)!=.001:
        raise ValueError('Expected the fixed V324 five-by-one-ms path')

    def tick(carry,index):
        native,(task,events)=carry
        previous=native[0]
        active=~task.done
        force=jnp.where(active[:,None],task.schedule.force_world_n,0.)
        forced=previous.replace(xfrc_applied=previous.xfrc_applied.at[:,env.ball_body_id,:3].add(force))
        after=substep(index,(forced,*native[1:]))
        current=after[0].replace(xfrc_applied=previous.xfrc_applied)
        # Completed lanes are padding, not further physical samples. Preserve
        # their data while shared native controller queue phases still advance.
        current=jax.tree.map(lambda a,b:jnp.where(active.reshape(active.shape+(1,)*(a.ndim-1)),a,b)
            if hasattr(a,'ndim') and a.ndim and a.shape[0]==active.shape[0] else a,current,previous)
        after=(current,*after[1:])
        trace=native_event_trace(env,previous,current)
        finite=(jnp.all(jnp.isfinite(current.qpos),axis=-1)&jnp.all(jnp.isfinite(current.qvel),axis=-1)
                &jnp.isfinite(trace['contact_normal_force_n']))
        failure=jnp.where(~finite,1,jnp.where(trace['other_ball_collision'],2,0))
        task=task._replace(elapsed_ms=task.elapsed_ms+(active&(failure!=0)).astype(jnp.int32))
        task=fail_evaluation(task,failure)
        usable=~task.done
        events,labels=physical_event_substep(events,trace,valid=usable,
            initialization_active=jnp.zeros_like(usable))
        task,scoring=consume_events(task,labels,goals,perturb=env._wmj_eval_perturb)
        output=dict(scoring,contact_entry=labels['contact_entry'],contact_release=labels['contact_release'],
            impact_id=labels['impact_id'],actual_apex=labels['actual_apex'],
            apex_position_m=labels['apex_position_m'],release_velocity_m_s=labels['release_velocity_m_s'],
            racket_release_velocity_m_s=trace['racket_linear_velocity_m_s'],
            racket_release_angular_velocity_rad_s=trace['racket_angular_velocity_rad_s'],
            racket_position_m=trace['racket_position_m'],
            pulse_applied=(jnp.linalg.norm(force,axis=-1)>0)&active,force_world_n=force,
            pulse_onset_delay_s=jnp.where(scoring['pulse_start'],(1-labels['apex_fraction'])*.001,0.),
            failure_code=jnp.where(active&~usable,failure,0),elapsed_ms=task.elapsed_ms)
        output=merge_observer_trace(output,observer,env,previous,current,labels)
        return (after,(task,events)),output

    (result,payload),trace=jax.lax.scan(tick,(initial,payload),jnp.arange(5))
    return result,payload,trace


def build_evaluation_step(environment,stack,*,perturb,trace_observer=None):
    """Return an explicit callable, leaving environment.step and native files intact."""
    native=importlib.import_module('mjx_juggle_env')
    if file_hash(Path(native.__file__))!=stack.spec.native_env_sha256:
        raise ValueError('Native source differs from authenticated V324')
    original=inspect.getsource(native.MjxJuggleEnv._step_impl)
    source=augment_source(original)
    namespace=dict(environment._step_impl.__globals__)
    namespace['_wmj_evaluation_loop']=(evaluation_loop if trace_observer is None else
        partial(evaluation_loop,observer=trace_observer))
    exec(compile(source,'<WMJ-V1-frozen-native-task>','exec'),namespace)
    environment._wmj_eval_perturb=bool(perturb)
    import mujoco
    import numpy as np
    model=environment.mj_model
    ancestors=[];body=int(model.geom_bodyid[environment.racket_contact_geom_id])
    while body:
        start=int(model.body_jntadr[body])
        ancestors.extend(range(start,start+int(model.body_jntnum[body])))
        body=int(model.body_parentid[body])
    ids=np.asarray(ancestors,np.int32)
    if not np.isin(model.jnt_type[ids],[mujoco.mjtJoint.mjJNT_HINGE,mujoco.mjtJoint.mjJNT_SLIDE]).all():
        raise ValueError('Expected scalar-joint fixed-base racket ancestry')
    environment._wmj_eval_twist=dict(jids=ids,vadr=model.jnt_dofadr[ids].copy(),
        hinge_mask=jnp.asarray(model.jnt_type[ids]==mujoco.mjtJoint.mjJNT_HINGE))
    augmented=types.MethodType(namespace['_step_impl'],environment)

    def step(state,action,*,wmj_payload,wmj_goals):
        # Preserve every outer V324 wrapper, including its persistent metadata.
        # This scoped binding executes only when the caller traces a JIT graph.
        original_step=environment._step_impl
        def bound(instance,native,commands,*args,**kwargs):
            return augmented(native,commands,*args,wmj_payload=wmj_payload,
                wmj_goals=wmj_goals,**kwargs)
        environment._step_impl=types.MethodType(bound,environment)
        try:
            return environment.step(state,action)
        finally:
            environment._step_impl=original_step
    return step,dict(native_sha256=stack.spec.native_env_sha256,
        hook_sha256=file_hash(Path(__file__)),augmented_source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        physics_changes='approved_external_ball_force_only',source_files_modified=False,
        zero_force_pairing='REQUIRED',force_onset='next_1ms_boundary_after_actual_apex')
