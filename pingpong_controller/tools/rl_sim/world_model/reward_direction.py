"""Private preparation, physical-event metrics and frozen reward comparisons."""
from __future__ import annotations
import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np

from .ball_flight import ballistic_apex
from .velocity_reward import diagnostic_course_step


METRIC_KEYS=('metric_release_position_m','metric_release_velocity_m_s',
             'metric_predicted_apex_m','metric_actual_apex_position_m','metric_actual_apex_vxy_m_s')


def prepare_source(source: str) -> str:
    """Reuse authenticated native/bank preparation, stopping before collection."""
    tree=ast.parse(source);fn=tree.body[0]
    ends=[i for i,node in enumerate(fn.body) if ast.unparse(node)=='original = env._step_impl']
    if len(ends)!=1:raise ValueError('Native preparation boundary changed')
    fn.body=fn.body[:ends[0]]+ast.parse('''return dict(environment=env,bundle=bundle,native=native,
        history=history,events=events,initial=initial_course,adapter=adapter,goals=goals,
        runtime=runtime,actor=actor,robot=robot,auxiliary=auxiliary,options=options,
        control=control,episode=episode,reward=reward)''').body
    return ast.unparse(ast.fix_missing_locations(tree))


def make_prepare():
    from .audit_free_imagination import audit
    from .audit_projection_precision_integrated import install_candidate
    from .projection_precision import make_runtime
    from .expanded_auxiliary import restore_candidate
    scope=dict(audit.__globals__,install_position_only_reference=install_candidate,
               make_runtime=make_runtime,restore_auxiliary=restore_candidate)
    exec(compile(prepare_source(inspect.getsource(audit)),'<WMJ-reward-native-preparation>','exec'),scope)
    return scope['audit']


def metric_task(original):
    """Expose actual1ms event positions without changing state or reward values."""
    def task(task,labels,next_goal,*,contract):
        after,terms,metrics=original(task,labels,next_goal,contract=contract)
        release=metrics['valid_release'];apex=metrics['valid_apex']
        predicted,_,_=ballistic_apex(labels['release_position_m'],labels['release_velocity_m_s'])
        def mask(value,enabled):return jnp.where(enabled[...,None],value,0.)
        return after,terms,dict(metrics,
            metric_release_position_m=mask(labels['release_position_m'],release),
            metric_release_velocity_m_s=mask(labels['release_velocity_m_s'],release),
            metric_predicted_apex_m=mask(predicted,release),
            metric_actual_apex_position_m=mask(labels['apex_position_m'],apex),
            metric_actual_apex_vxy_m_s=jnp.where(apex,jnp.linalg.norm(labels['apex_velocity_m_s'][...,:2],axis=-1),0.))
    return task


def observed_course(runtime):
    """Only add float outputs; leave all source control, physics and counters intact."""
    from .course_env import course_step
    original=diagnostic_course_step(runtime)
    tree=ast.parse(inspect.getsource(course_step));fn=tree.body[0];body=[];found=[]
    for node in fn.body:
        text=ast.unparse(node)
        if (text.startswith('counts = {') and isinstance(node.value, ast.DictComp)
                and ast.unparse(node.value.generators[0].iter) == 'task_events.items()'):
            node.value.generators[0].ifs.append(ast.parse('key not in metric_keys',mode='eval').body)
            found.append('counts')
        body.append(node)
        if text.startswith('diagnostics = dict('):
            body.extend(ast.parse("diagnostics.update({key:jnp.where(trusted.reshape(trusted.shape+(1,)*(task_events[key].ndim-2)),jnp.sum(task_events[key],axis=0),0.) for key in metric_keys})").body)
            found.append('diagnostics')
    if found!=['counts','diagnostics']:raise ValueError('Course output instrumentation anchors changed')
    fn.body=body
    namespace=dict(original.__globals__,metric_keys=METRIC_KEYS,task_substep=metric_task(original.__globals__['task_substep']))
    exec(compile(ast.fix_missing_locations(tree),'<WMJ-reward-event-observer>','exec'),namespace)
    return namespace['course_step']


def score_terms(terms,weights):
    old=sum(value for key,value in terms.items() if key not in ('velocity_xy','velocity_z'))
    candidate=old-terms['apex_xy']-terms['apex_z']+weights['xy']*terms['velocity_xy']+weights['z']*terms['velocity_z']
    return old,candidate


def summarize_stream(stream,source_indices,goals,gamma,weights):
    """Preserve failed sources, absent apices and both observation horizons."""
    result={};goals=np.asarray(goals)
    for horizon in (64,128):
        valid=np.asarray(stream['valid'][:horizon],bool)
        discounted={k:np.sum(np.asarray(v[:horizon])*gamma**np.arange(horizon)[:,None],axis=0)
                    for k,v in stream['terms'].items()}
        old,candidate=score_terms(discounted,weights)
        counts=stream['events'];rows=[]
        for lane,source in enumerate(source_indices):
            records=[]
            for tick in range(horizon):
                for kind,key,point_key in [('release','valid_release','metric_predicted_apex_m'),
                                          ('actual_apex','valid_apex','metric_actual_apex_position_m')]:
                    count=int(counts[key][tick,lane])
                    if not count:continue
                    record=dict(kind=kind,control=tick+1,count=count)
                    # Multiple events in one control cannot be disaggregated from
                    # these sums; retain the count and explicitly mark unknown.
                    if count==1:
                        point=np.asarray(counts[point_key][tick,lane]);delta=point-goals[lane]
                        record.update(position_m=point.tolist(),xy_error_m=float(np.linalg.norm(delta[:2])),
                                      z_error_m=float(abs(delta[2])))
                        if kind=='release':
                            record.update(release_position_m=counts['metric_release_position_m'][tick,lane].tolist(),
                                outgoing_velocity_m_s=counts['metric_release_velocity_m_s'][tick,lane].tolist())
                        else:record['vxy_m_s']=float(counts['metric_actual_apex_vxy_m_s'][tick,lane])
                    else:record['metrics']='UNKNOWN_multiple_events_in_one_control'
                    records.append(record)
            fail=np.flatnonzero(stream['failure'][:horizon,lane]);end=int(fail[0]+1) if fail.size else None
            rows.append(dict(source_index=source,valid_controls=int(valid[:,lane].sum()),
                first_failure_control=end,failure_bits=int(stream['failure_code'][end-1,lane]) if end else 0,events=records))
        result[str(horizon)]=dict(old=np.asarray(old).tolist(),candidate=np.asarray(candidate).tolist(),
            terms={k:v.tolist() for k,v in discounted.items()},hits=np.asarray(counts['valid_release'][:horizon]).sum(axis=0).astype(int).tolist(),
            actual_apices=np.asarray(counts['valid_apex'][:horizon]).sum(axis=0).astype(int).tolist(),
            failed=np.any(stream['failure'][:horizon],axis=0).tolist(),sources=rows)
    return result


def compare_returns(native,world,criteria):
    """Compare all32 realized rewards; changed failure/event modes stay included."""
    report={}
    for term in ('old','candidate'):
        nv={k:np.asarray(v[term],float) for k,v in native.items()};wv={k:np.asarray(v[term],float) for k,v in world.items()}
        nd=nv['plus']-nv['minus'];wd=wv['plus']-wv['minus'];nr=np.abs(nv['base']-nv['repeat']);wr=np.abs(wv['base']-wv['repeat'])
        resolved=(np.abs(nd)>np.maximum(criteria['return_difference_floor'],criteria['repeat_multiplier']*nr)) & (np.abs(wd)>np.maximum(criteria['return_difference_floor'],criteria['repeat_multiplier']*wr))
        count=int(resolved.sum());agreement=float(np.mean(nd[resolved]*wd[resolved]>0)) if count else None
        report[term]=dict(source_count=len(nd),resolved_indices=np.flatnonzero(resolved).tolist(),unresolved_indices=np.flatnonzero(~resolved).tolist(),
            resolved_count=count,direction_agreement_fraction=agreement,native_difference=nd.tolist(),world_difference=wd.tolist(),
            native_repeat_abs=nr.tolist(),world_repeat_abs=wr.tolist(),
            mean_native_plus_minus=float(nd.mean()),mean_world_plus_minus=float(wd.mean()),
            mean_native_plus_base=float((nv['plus']-nv['base']).mean()),mean_world_plus_base=float((wv['plus']-wv['base']).mean()),
            mean_native_repeat_abs=float(nr.mean()),mean_world_repeat_abs=float(wr.mean()),
            native_failure_delta=int(np.sum(native['plus']['failed'])-np.sum(native['base']['failed'])),
            world_failure_delta=int(np.sum(world['plus']['failed'])-np.sum(world['base']['failed'])),
            native_hit_delta=int(np.sum(native['plus']['hits'])-np.sum(native['base']['hits'])),
            world_hit_delta=int(np.sum(world['plus']['hits'])-np.sum(world['base']['hits'])),
            diagnostic_pass=bool(count>=criteria['minimum_resolved'] and agreement>=criteria['minimum_direction_fraction']))
    return report
