"""Paired free-attitude own-state predictions on explicitly retained sources."""
from __future__ import annotations

import time
import jax
import jax.numpy as jnp
import numpy as np

from .action_effects import model_event_trace
from .audit_full_prior_world import release_cpu_heap
from .error_attribution import window_arrays
from .imagined_env import finite_physics
from .module_gradient_tools import distribution
from .physical_events import scan_physical_events
from .release_alignment import compare_releases
from .state_adapter import numeric_tree_digest


def paired_windows(data,models,plan,guard,report,*,lanes):
    frozen=numeric_tree_digest((data['actor'],data['robot'],models))
    rt=data['runtime'];wb=data['world_bank'];n=len(plan['source_rows']);lanes=np.asarray(lanes)
    contact=data['raw']['contact_overlap'].reshape(640,n)[:,lanes]
    first=np.where(contact.any(axis=0),contact.argmax(axis=0)//5,-1)
    starts=np.stack((np.zeros(len(lanes),int),np.maximum(first-2,0),np.full(len(lanes),32),np.full(len(lanes),64))).ravel()
    selected_lanes=np.tile(lanes,4);count=len(starts)
    enabled=np.ones(count,bool);enabled[len(lanes):2*len(lanes)]=first>=0
    selected,available=window_arrays(data['actions'],starts,selected_lanes,length=128);available &= enabled[None]
    actions=jnp.asarray(np.where(available[...,None],selected,0.))
    initial=jax.tree.map(lambda x:jnp.asarray(x[starts,selected_lanes]),wb)
    events=jax.tree.map(lambda x:jnp.asarray(x[starts,selected_lanes]),data['events_bank'])
    context=jax.tree.map(lambda x:x[selected_lanes],data['context']);goal=data['goals'][selected_lanes]
    actual=jax.tree.map(lambda x:window_arrays(np.swapaxes(x,1,2),starts,selected_lanes,length=128)[0].swapaxes(1,2),data['observed_labels'])
    native_prefix=window_arrays(data['prefix'],starts,selected_lanes,length=128)[0]&available
    truth={key:window_arrays(value[1:],starts,selected_lanes,length=128)[0] for key,value in
        [('ball_position_m',wb.physics.ball.position_m),('ball_velocity_m_s',wb.physics.ball.velocity_m_s),('racket_position_m',wb.cached_racket_pose[...,:3])]}
    ids=[f'{plan["source_indices"][lane]}/start{start}' for lane,start in zip(selected_lanes,starts)]
    regions=np.array([plan['source_rows'][lane]['region_id'] for lane in selected_lanes])
    @jax.jit
    def predict(auxiliary):
        def tick(carry,action):
            world,event=carry
            after,tr=rt.world_step(data['robot'],auxiliary,world,action,context=context,
                control_contract=data['control'],task_goal_xyz=goal,ar1_enabled=False,**data['options'])
            with jax.default_matmul_precision('highest'):
                trace=model_event_trace(tr['physics'],surface=data['options']['surface'],
                    ball_radius_m=data['options']['ball_radius_m'],contact_contract=data['options']['contact_contract'])
            finite=finite_physics(after.physics);valid=jnp.broadcast_to(finite,(5,count))
            event,labels=scan_physical_events(event,trace,valid=valid,initialization_active=jnp.zeros_like(valid))
            return (after,event),dict(ball_position_m=after.physics.ball.position_m,ball_velocity_m_s=after.physics.ball.velocity_m_s,
                racket_position_m=after.cached_racket_pose[...,:3],finite=finite,labels=labels)
        return jax.lax.scan(tick,(initial,events),actions)
    results={}
    for name,auxiliary in models.items():
        guard.check();started=time.monotonic();_,pred=jax.device_get(predict(auxiliary))
        errors={k:np.linalg.norm(pred[k]-v,axis=-1) for k,v in truth.items()};horizons={};releases={}
        for h in plan['horizons']:
            scopes=dict(available=available[h-1],native_task_prefix=native_prefix[h-1])
            horizons[str(h)]={k:{scope:distribution(v[h-1][mask]) for scope,mask in scopes.items()} for k,v in errors.items()}
            horizons[str(h)]['by_region']={str(int(region)):{k:distribution(v[h-1][native_prefix[h-1]&(regions==region)])
                for k,v in errors.items()} for region in np.unique(regions)}
            horizons[str(h)]['per_window']={k:[float(x) if np.isfinite(x) else None for x in v[h-1]]
                for k,v in errors.items()}
            valid=np.broadcast_to(native_prefix[:h,None],(h,5,count)).reshape(h*5,count)&available[h-1][None]
            a={k:v[:h].reshape((-1,)+v.shape[2:]) for k,v in actual.items()}
            m={k:v[:h].reshape((-1,)+v.shape[2:]) for k,v in pred['labels'].items()}
            releases[str(h)]=compare_releases(a['controlled_release']&valid,m['controlled_release']&valid,
                a['release_position_m'],a['release_velocity_m_s'],m['release_position_m'],m['release_velocity_m_s'],ids)
        results[name]=dict(horizons=horizons,releases=releases,wall_s=time.monotonic()-started,
            nonfinite_controls=int((available&~pred['finite']).sum()))
        print('[EXPANDED_WORLD] '+name+' H8='+str({k:v for k,v in releases['8'].items() if k!='cases'}),flush=True)
    report['paired_world']=results
    report['window_population']=dict(lanes=selected_lanes.tolist(),source_indices=[plan['source_indices'][i] for i in selected_lanes],
        starts=starts.tolist(),regions=regions.tolist(),available_end_counts={str(h):int(available[h-1].sum()) for h in plan['horizons']},
        native_prefix_end_counts={str(h):int(native_prefix[h-1].sum()) for h in plan['horizons']})
    if frozen!=numeric_tree_digest((data['actor'],data['robot'],models)):raise ValueError('Frozen audit changed parameters')
    report.update(frozen_parameters_before_sha256=frozen,frozen_parameters_after_sha256=frozen,
        parameters_unchanged=True,world_qualified=False,comparison='same native actions and complete seeds;own robot/control/ball/history;true labels only for metrics')
    predict.clear_cache();jax.clear_caches();release_cpu_heap()
