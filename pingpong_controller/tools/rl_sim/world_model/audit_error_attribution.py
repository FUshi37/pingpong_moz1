"""Frozen attribution of expanded-attitude control, robot and contact errors."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import time

import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .audit_control_components import make_component_replays
from .audit_full_prior_world import release_cpu_heap
from .contact_dynamics import ContactState
from .contact_model import world_spin
from .contracts import authenticate,config_hash,file_hash,load_config,repository_root
from .error_attribution import make_capture,window_arrays,conditional_step,native_contact_step,largest_prefix_errors
from .imagined_env import finite_physics
from .module_gradient_tools import distribution
from .native_entry_collision import write_compact_json
from .normal_coupling import acceleration_terms
from .physical_events import scan_physical_events
from .release_alignment import compare_releases
from .robot_support_model import support_robot_step
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .support_losses import racket_kinematics
from .train_pose_trial import phase_budget
from .velocity_reward import clone_globals


MODES=('own_control_robot_aux','native_commands_own_robot_aux','native_robot_pred_aux','native_robot_true_solver')


def scopes(values, prefix, contact=None):
    result=dict(all=distribution(values),native_task_prefix=distribution(np.asarray(values)[prefix]))
    if contact is not None:result['contact_prefix']=distribution(np.asarray(values)[prefix & contact])
    return result


def command_errors(predicted, raw, prefix):
    return {key:scopes(np.max(np.abs(np.asarray(predicted[key])-raw['delayed_'+key+'_command']),axis=(1,3)),prefix)
            for key in ('q','dq','ddq')}


def control_diagnostics(data,guard,source_indices):
    rt=data['runtime'];control=data['control'];context=data['context'];wb=data['world_bank'];raw=data['raw'];prefix=data['prefix']
    component=clone_globals(make_component_replays,project_reference=rt.projector)(rt.options['chain'],control)
    values={}
    for name,fn in component.items():
        guard.check()
        if name=='projection':
            prediction=jax.device_get(fn(context,jnp.asarray(data['actions']),jnp.asarray(wb.physics.q_rad[:-1]),
                jnp.asarray(wb.control.reference_q_rad[:-1])))
            values[name]=dict(reference=scopes(np.max(np.abs(prediction['reference']-wb.control.reference_q_rad[1:]),axis=-1),prefix),
                accepted=scopes(np.max(np.abs(prediction['accepted']-wb.control.previous_accepted_action[1:]),axis=-1),prefix),
                fallback_controls=int(prediction['fallback'].sum()))
            errors=np.max(np.abs(prediction['reference']-wb.control.reference_q_rad[1:]),axis=-1)
            locations=largest_prefix_errors(errors,prefix,source_indices)
            for row in locations:
                t,lane=row['control']-1,row['lane']
                row.update(predicted_reference_rad=prediction['reference'][t,lane].tolist(),
                    native_reference_rad=wb.control.reference_q_rad[t+1,lane].tolist(),
                    before_reference_rad=wb.control.reference_q_rad[t,lane].tolist(),
                    predicted_accepted=prediction['accepted'][t,lane].tolist(),
                    native_accepted=wb.control.previous_accepted_action[t+1,lane].tolist(),
                    predicted_fallback=bool(prediction['fallback'][t,lane]),
                    raw_action=data['actions'][t,lane].tolist(),
                    measured_q_rad=wb.physics.q_rad[t,lane].tolist())
            values[name]['largest_prefix_locations']=locations
        else:
            prediction=jax.device_get(fn(data['initial'].control,context,jnp.asarray(wb.control.reference_q_rad[1:])))
            values[name]=command_errors(prediction,raw,prefix)
        fn.clear_cache()
    @jax.jit
    def own():
        def tick(state,x):
            after,tr=rt.controller(state,x[0],x[1],context,chain=rt.options['chain'],contract=control)
            return after,(after,tr['delayed_commands'])
        return jax.lax.scan(tick,data['initial'].control,(jnp.asarray(data['actions']),jnp.asarray(wb.physics.q_rad[:-1])))
    _,(states,cmd)=jax.device_get(own());own.clear_cache()
    values['persistent_own_reference']=command_errors(cmd,raw,prefix)
    values['persistent_own_reference']['largest_command_locations']={key:largest_prefix_errors(
        np.max(np.abs(cmd[key]-raw['delayed_'+key+'_command']),axis=(1,3)),prefix,source_indices)
        for key in ('q','dq','ddq')}
    leaves=jax.tree_util.tree_flatten_with_path(states)[0]
    actual=jax.tree.leaves(jax.tree.map(lambda x:x[1:],wb.control))
    fields=[]
    for (path,a),b in zip(leaves,actual):
        error=np.abs(np.asarray(a).astype(float)-np.asarray(b).astype(float))
        axes=tuple(range(2,error.ndim))
        per=error.max(axis=axes) if axes else error
        fields.append(dict(path=jax.tree_util.keystr(path),dtype=str(a.dtype),**scopes(per,prefix),
            largest_prefix_locations=largest_prefix_errors(per,prefix,source_indices,limit=2)))
    values['persistent_state_fields']=fields
    return values


def local_diagnostics(data,guard):
    raw={k:v.reshape((-1,)+v.shape[2:]) for k,v in data['raw'].items()}
    prefix=np.repeat(data['prefix'],5,axis=0);contact=raw['contact_overlap']
    robot=data['robot'];aux=data['auxiliary'];opt=data['options']
    q=raw['q_before'];dq=raw['dq_before'];commands={k:raw['delayed_'+k+'_command'] for k in ('q','dq','ddq')}
    @jax.jit
    def predict(q,dq,commands,raw):
        with jax.default_matmul_precision('highest'):
            nq,ndq=support_robot_step(robot,opt['robot_scale'],q,dq,commands)
            pose=racket_kinematics(opt['chain'],q,dq)
            terms=acceleration_terms(aux,q,dq,commands)
            before=ContactState(raw['ball_qpos_before'][...,:3],raw['ball_qvel_before'][...,:3],
                world_spin(raw['ball_qpos_before'][...,3:],raw['ball_qvel_before'][...,3:]),
                raw['contact_overlap'],jnp.zeros(q.shape[:-1],jnp.float32))
            predicted,_,_=native_contact_step(before,raw,terms,opt)
            oracle,_,_=native_contact_step(before,raw,raw['solver_target'],opt)
        return nq,ndq,pose,terms,predicted,oracle
    guard.check();nq,ndq,pose,terms,predicted,oracle=jax.device_get(predict(jnp.asarray(q),jnp.asarray(dq),
        jax.tree.map(jnp.asarray,commands),jax.tree.map(jnp.asarray,raw)))
    metrics=dict(robot_q_one_ms_rad=np.max(np.abs(nq-raw['q_after']),axis=-1),
        robot_dq_one_ms_rad_s=np.max(np.abs(ndq-raw['dq_after']),axis=-1),
        native_q_fk_geom_position_m=np.linalg.norm(pose['geom_position']-raw['racket_geom_position_before_m'],axis=-1),
        native_q_fk_rotation_frobenius=np.linalg.norm(pose['rotation']-raw['racket_geom_rotation_before'],axis=(-1,-2)),
        native_q_fk_linear_velocity_m_s=np.linalg.norm(pose['velocity']-raw['racket_linear_velocity_before_m_s'],axis=-1),
        auxiliary_linear_m_s2=np.linalg.norm(terms[...,:3]-raw['solver_target'][...,:3],axis=-1),
        auxiliary_angular_rad_s2=np.linalg.norm(terms[...,3:]-raw['solver_target'][...,3:],axis=-1),
        contact_pred_aux_one_ms_velocity_m_s=np.linalg.norm(predicted.velocity_m_s-raw['ball_qvel_after'][...,:3],axis=-1),
        contact_true_solver_one_ms_velocity_m_s=np.linalg.norm(oracle.velocity_m_s-raw['ball_qvel_after'][...,:3],axis=-1),
        solver_normal_closure_abs=np.abs(raw['solver_closure_error']))
    result={k:scopes(v,prefix,contact) for k,v in metrics.items()}
    result['contact_prefix_count']=int((prefix&contact).sum())
    result['multi_pair_contact_count']=int((prefix & (raw['solver_pair_count']>1)).sum())
    predict.clear_cache();release_cpu_heap()
    return result


def model_windows(data,plan,guard,report):
    rt=data['runtime'];raw=data['raw'];wb=data['world_bank'];n=len(plan['source_indices'])
    contacts=raw['contact_overlap'].reshape(640,n)
    first_contact=np.where(contacts.any(axis=0),contacts.argmax(axis=0)//5,-1)
    starts=np.stack((np.zeros(n,int),np.maximum(first_contact-2,0),np.full(n,32),np.full(n,64))).ravel()
    lanes=np.tile(np.arange(n),4);enabled=np.ones(4*n,bool);enabled[n:2*n]=first_contact>=0
    selected,available=window_arrays(data['actions'],starts,lanes,length=128);available &= enabled[None]
    actions=jnp.asarray(np.where(available[...,None],selected,0.))
    selected_raw={k:window_arrays(np.swapaxes(v,1,2),starts,lanes,length=128)[0].swapaxes(1,2) for k,v in raw.items()}
    selected_raw=jax.tree.map(jnp.asarray,selected_raw)
    initial=jax.tree.map(lambda x:jnp.asarray(x[starts,lanes]),wb)
    events=jax.tree.map(lambda x:jnp.asarray(x[starts,lanes]),data['events_bank'])
    context=jax.tree.map(lambda x:x[lanes],data['context']);goal=data['goals'][lanes]
    actual=jax.tree.map(lambda x:window_arrays(np.swapaxes(x,1,2),starts,lanes,length=128)[0].swapaxes(1,2),data['observed_labels'])
    native_prefix=window_arrays(data['prefix'],starts,lanes,length=128)[0]&available
    truth_p=window_arrays(wb.physics.ball.position_m[1:],starts,lanes,length=128)[0]
    truth_v=window_arrays(wb.physics.ball.velocity_m_s[1:],starts,lanes,length=128)[0]
    truth_pose=window_arrays(wb.cached_racket_pose[1:],starts,lanes,length=128)[0]
    ids=[f'{plan["source_indices"][lane]}/start{start}' for lane,start in zip(lanes,starts)]
    reports={}
    for mode in MODES:
        @jax.jit
        def predict():
            def tick(carry,inputs):
                world,event=carry;action,record=inputs
                step=conditional_step(rt,mode,record)
                after,tr=step(data['robot'],data['auxiliary'],world,action,context=context,
                    control_contract=data['control'],task_goal_xyz=goal,ar1_enabled=False,**data['options'])
                with jax.default_matmul_precision('highest'):
                    event_trace=model_event_trace(tr['physics'],surface=data['options']['surface'],
                        ball_radius_m=data['options']['ball_radius_m'],contact_contract=data['options']['contact_contract'])
                valid=jnp.broadcast_to(finite_physics(after.physics),(5,len(lanes)))
                event,labels=scan_physical_events(event,event_trace,valid=valid,initialization_active=jnp.zeros_like(valid))
                return (after,event),dict(position=after.physics.ball.position_m,velocity=after.physics.ball.velocity_m_s,
                    pose=after.cached_racket_pose,finite=finite_physics(after.physics),labels=labels)
            return jax.lax.scan(tick,(initial,events),(actions,selected_raw))
        guard.check();started=time.monotonic();_,prediction=jax.device_get(predict())
        metrics=dict(ball_position_m=np.linalg.norm(prediction['position']-truth_p,axis=-1),
            ball_velocity_m_s=np.linalg.norm(prediction['velocity']-truth_v,axis=-1),
            racket_position_m=np.linalg.norm(prediction['pose'][...,:3]-truth_pose[...,:3],axis=-1))
        horizons={};releases={}
        for h in plan['horizons']:
            mask=available[h-1]
            horizons[str(h)]={k:dict(available=distribution(v[h-1][mask]),
                native_task_prefix=distribution(v[h-1][native_prefix[h-1]])) for k,v in metrics.items()}
            valid=np.broadcast_to(native_prefix[:h,None],(h,5,len(lanes))).reshape(h*5,-1)
            a={k:v[:h].reshape((-1,)+v.shape[2:]) for k,v in actual.items()}
            m={k:v[:h].reshape((-1,)+v.shape[2:]) for k,v in prediction['labels'].items()}
            full=available[h-1]
            valid = valid & full[None]
            releases[str(h)]=compare_releases(a['controlled_release']&valid,m['controlled_release']&valid,
                a['release_position_m'],a['release_velocity_m_s'],m['release_position_m'],m['release_velocity_m_s'],ids)
        reports[mode]=dict(horizons=horizons,releases=releases,wall_s=time.monotonic()-started,
            nonfinite_controls=int((available & ~prediction['finite']).sum()))
        print('[ATTRIBUTION] '+mode+' H8='+str({k:v for k,v in releases['8'].items() if k!='cases'}),flush=True)
        predict.clear_cache();jax.clear_caches();release_cpu_heap()
    report['conditional_windows']=reports
    report['windows']=dict(starts=starts.tolist(),source_indices=[plan['source_indices'][i] for i in lanes],
        available_windows={str(h):int(available[h-1].sum()) for h in plan['horizons']},
        native_prefix_windows={str(h):int(native_prefix[h-1].sum()) for h in plan['horizons']})


def audit(cfg,args,guard,report,plan):
    before=time.monotonic();data=make_capture()(cfg,args,guard,report,plan)
    report['capture_wall_s']=time.monotonic()-before
    # Keep original native7-joint streams intact until all collection checks finish.
    for name in ('q_before','dq_before','q_after','dq_after'):
        data['raw'][name]=np.concatenate((data['raw'][name],data['raw']['support_'+name]),axis=-1)
    prior=json.loads(authenticate(repository_root(),plan['prior_native_audit']).read_text())
    report['prior_replay_comparison']=dict(initial_native_hash_equal=report['initial_native_sha256']==prior['initial_native_sha256'],
        prior_valid_prefix_controls=prior['native_population']['valid_prefix_controls'],
        current_valid_prefix_controls=report['native_population']['valid_prefix_controls'],
        prior_first_failures=prior['native_population']['first_failure_control'],
        current_first_failures=report['native_population']['first_failure_control'])
    report['control_components']=control_diagnostics(data,guard,plan['source_indices'])
    if not args.control_detail_only:
        report['one_ms_components']=local_diagnostics(data,guard)
        model_windows(data,plan,guard,report)
    after=numeric_tree_digest((data['actor'],data['robot'],data['auxiliary']))
    if after!=data['frozen']:raise ValueError('Frozen parameters changed')
    report.update(status=('FROZEN_CONTROL_DETAIL_COMPLETE' if args.control_detail_only else
        'FROZEN_ERROR_ATTRIBUTION_COMPLETE'),parameters_unchanged=True,
        frozen_parameters_after_sha256=after,parameter_updates=0,default_changed=False,world_qualified=False,
        limitations=['Conditional replacements are diagnostic oracle inputs, never executable imagination contracts',
            'Nonlinear component interactions prevent unique additive percentages',
            'Primary releases occur before first native physical failure; post-failure horizon errors are separate',
            'Only the existing32 independent training sources; no full-domain qualification'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--control-detail-only',action='store_true',
        help='Repeat the identical native sources only to locate control-error tails')
    parser.add_argument('--gpu-uuid',required=True);args=parser.parse_args();cfg=load_config()
    setting=cfg['expanded_error_attribution'];plan=json.loads(authenticate(repository_root(),setting['plan']).read_text())
    if (not setting['authorized'] or plan['max_phase_bytes']!=2*2**20
            or any(plan[k] for k in ('actor_updates','critic_updates','world_updates','default_replacement'))):
        raise ValueError('Only bounded frozen attribution is authorized')
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.,startup_reserve_bytes=2**20) as guard:
        phase_budget(guard,plan);args.output_dir.mkdir(parents=True,exist_ok=False)
        paths=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_expanded_error_attribution',status='INCOMPLETE',training_version='WMJ-V1',
            plan=setting['plan'],gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in paths},actor_updates=0,critic_updates=0,world_updates=0)
        write_compact_json(args.output_dir/'implementation.json',dict(report=report,sources={n:(Path(__file__).parent/n).read_text()
            for n in ('error_attribution.py','audit_error_attribution.py')}),guard.budget)
        write_compact_json(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:raise ValueError('One CUDA device required')
            audit(cfg,args,guard,report,plan)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_compact_json(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':main()
