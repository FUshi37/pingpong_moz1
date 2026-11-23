"""Read-only native-input substitutions for frozen error attribution only."""
import ast
import inspect
from types import MethodType

import jax
import jax.numpy as jnp
import numpy as np

from . import teacher_trace
from .collect_solver_auxiliary import solver_labels
from .normal_coupling import acceleration_terms
from .tangent_contact import tangent_contact_step
from .velocity_reward import clone_globals
from .world_physics import PhysicsState


def capture_source(source: str) -> str:
    """Retain the authenticated a39 collection, stopping before its model audit."""
    tree=ast.parse(source);function=tree.body[0]
    anchors=[i for i,node in enumerate(function.body) if ast.unparse(node)=='step.clear_cache()']
    if len(anchors)!=1:
        raise ValueError('Expected unique native-capture boundary')
    end=anchors[0]
    expected=['step.clear_cache()','plain.clear_cache()','jax.clear_caches()','release_cpu_heap()']
    if [ast.unparse(n) for n in function.body[end:end+4]]!=expected:
        raise ValueError('Native-capture cleanup changed')
    function.body=function.body[:end+4]+ast.parse('''return dict(world_bank=world_bank,events_bank=events_bank,
        raw=raw,observed_labels=observed_labels,actions=actions,prefix=prefix,native_bounds=native_bounds,
        runtime=runtime,initial=initial,context=context,goals=goals,actor=actor,robot=robot,
        auxiliary=auxiliary,options=options,control=control,episode=episode,frozen=frozen)''').body
    return ast.unparse(ast.fix_missing_locations(tree))


def augmented_recorded_loop(environment, *, labels=solver_labels):
    """Add flat diagnostic arrays; leave every original carry and sample intact."""
    def loop(substep, initial, count, sample):
        def observe(index,before,after):
            trace=sample(index,before,after)
            extra=labels(environment,after[0])
            return dict(trace,solver_target=extra['target'],solver_closure_error=extra['closure_error'],
                solver_pair_count=extra['active_pair_count'],solver_overlap=extra['overlap'])
        return teacher_trace.recorded_loop(substep,initial,count,observe)
    return loop


def instrument_with_solver(environment, stack, *, support_trace=False):
    """Only private method namespaces gain observer dependencies; no module patch."""
    identity=teacher_trace.instrument_environment(environment,stack,support_trace=support_trace)
    observe=clone_globals(teacher_trace._observe_native_loop,
                          recorded_loop=augmented_recorded_loop(environment))
    original=environment._step_impl.__func__
    environment._step_impl=MethodType(clone_globals(original,_wmj_recorded_loop=observe),environment)
    return dict(identity,additional_labels='native geom-origin J*qacc linear3/angular3;diagnostic only')


def make_capture():
    from .audit_free_imagination import audit
    namespace=dict(audit.__globals__,instrument_environment=instrument_with_solver)
    exec(compile(capture_source(inspect.getsource(audit)),'<WMJ-frozen-error-capture>','exec'),namespace)
    return namespace[audit.__name__]


def window_arrays(values, starts, lanes, *, length):
    times=np.arange(length)[:,None]+np.asarray(starts)[None]
    valid=times<len(values)
    return values[np.minimum(times,len(values)-1),np.asarray(lanes)],valid


def largest_prefix_errors(errors, prefix, source_indices, *, limit=5):
    """Locate existing task-prefix errors without selecting failed-time samples."""
    ticks,lanes=np.nonzero(prefix)
    values=np.asarray(errors)[ticks,lanes]
    order=np.argsort(-values,kind='stable')[:limit]
    return [dict(source_index=int(source_indices[lanes[i]]),lane=int(lanes[i]),
                 control=int(ticks[i])+1,error=float(values[i])) for i in order]


def native_contact_step(ball, raw, terms, options):
    """Oracle robot/solver inputs are confined to this diagnostic function."""
    pose=dict(geom_position=raw['racket_geom_position_before_m'],rotation=raw['racket_geom_rotation_before'],
        velocity=raw['racket_linear_velocity_before_m_s'],omega=raw['racket_angular_velocity_before_rad_s'])
    racket=dict(geom_position_m=pose['geom_position'],geom_rotation_world=pose['rotation'],
        racket_linear_velocity_m_s=pose['velocity'],racket_angular_velocity_rad_s=pose['omega'],
        joint_q_rad=raw['q_before'],delayed_q_command_rad=raw['delayed_q_command'])
    after,contact=tangent_contact_step(None,options['surface'],ball,racket,terms,
        ball_radius_m=options['ball_radius_m'],position_scale=options['position_scale'],
        contract=options['contact_contract'],learned=False)
    return after,contact,pose


def conditional_step(runtime, mode, native_interval):
    """Construct an explicit diagnostic substitution, never a corrected runtime."""
    if mode=='own_control_robot_aux':return runtime.world_step
    if mode not in ('native_commands_own_robot_aux','native_robot_pred_aux','native_robot_true_solver'):
        raise ValueError('Unknown diagnostic substitution')
    def controller(*args,**kwargs):
        after,trace=runtime.controller(*args,**kwargs)
        commands={key:native_interval['delayed_'+key+'_command'] for key in ('q','dq','ddq')}
        if mode!='native_commands_own_robot_aux':commands['diagnostic']=native_interval
        return after,dict(trace,delayed_commands=commands)
    overrides=dict(control_step=controller)
    if mode!='native_commands_own_robot_aux':
        def physics(robot,auxiliary,state,command,**options):
            raw=command['diagnostic'];cmd={key:command[key] for key in ('q','dq','ddq')}
            with jax.default_matmul_precision('highest'):
                terms=(raw['solver_target'] if mode=='native_robot_true_solver' else
                    acceleration_terms(auxiliary,raw['q_before'],raw['dq_before'],cmd))
                ball,contact,pose=native_contact_step(state.ball,raw,terms,options)
            after=PhysicsState(raw['q_after'],raw['dq_after'],ball)
            return after,dict(q_before_rad=raw['q_before'],dq_before_rad_s=raw['dq_before'],
                q_after_rad=raw['q_after'],dq_after_rad_s=raw['dq_after'],racket_before=pose,
                solver_terms=terms,contact=contact)
        overrides['full_prior_physics_step']=physics
    return clone_globals(runtime.world_step,**overrides)
