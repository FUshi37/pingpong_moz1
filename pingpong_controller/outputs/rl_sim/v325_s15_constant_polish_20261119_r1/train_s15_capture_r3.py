"""Explicit fixed-S15 continuation; never enters S16 or changes native dynamics."""
import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

AUDIT = Path(__file__).resolve().parent
ROOT = AUDIT.parents[3]
PLAN = json.loads((AUDIT / 'PLAN.json').read_text())
SOURCE = Path(PLAN['source_snapshot'])
sys.path[:0] = [str(AUDIT), str(SOURCE)]
from direction_speed_cohorts import DirectionEnv, cohort_labels, wind_sector
from capture_margin import CaptureEnv, CAPTURE_FIELDS
from bounded_validation_memory import release_caches
from train_fixed_plane_wind import (load_stack, restore_training_state, course_config,
    make_train_functions, strong_arrays, numeric_digest, digest_file)
from fixed_plane_wind_bootstrap_restore import learner_settings_for_stack
from fixed_plane_v321_long_episode import QUALITY_FIELDS
from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS, reference_quality_summary

FIELDS = tuple(dict.fromkeys(QUALITY_FIELDS + HEIGHT_FIELDS + REFERENCE_FIELDS + (
    'v321/finite', 'wind/mode', 'wind/positive_y_violation', 'wind/speed_m_s',
    'terminated', 'truncated', 'new_hit', 'hit_count', 'polish/wind_sector',
    'polish/peak_assigned', 'polish/physical_contact_frames', 'polish/edge_contact_frames',
    'polish/contact_face_deviation_max_deg', 'done/racket_z_out_of_bounds',
    'done/ball_too_high', 'done/ball_view_y_too_low', 'done/ball_view_z_too_low') + CAPTURE_FIELDS))
STOP = False


def save(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)


def sums(metrics, mask):
    return {k: float(np.sum(np.asarray(v)*mask)) for k, v in metrics.items() if k in FIELDS}


def combine(rows):
    return {k: sum(row.get(k, 0.) for row in rows) for k in set().union(*rows)}


def make_functions(stack, env, args):
    original, captured = stack.ppo.make_train_fns, {}

    def capture(**kwargs):
        captured.update(kwargs)
        return None, None

    stack.ppo.make_train_fns = capture
    try:
        make_train_functions(stack, env, args)
    finally:
        stack.ppo.make_train_fns = original
    captured['full_rollout_metric_names'] = tuple(dict.fromkeys(captured['full_rollout_metric_names'] + FIELDS))
    for k in ('reference_params', 'actor_anchor_replay_obs', 'counterfactual_replay_obs', 'counterfactual_replay_actions'):
        if captured.get(k) is not None:
            captured[k] = jax.device_put(captured[k])
    assert captured['counterfactual_supervision_coef'] == 0.
    collect, update = original(**captured)
    return collect, update, {k: v for k, v in captured.items() if isinstance(v, (str, int, float, bool, tuple))}


def evaluation(env, ppo, collect, learner, label, folder):
    n = env.n_envs
    assert n == 256
    keys = jax.random.split(jax.random.PRNGKey(2026111922), n)
    state, obs = jax.jit(env.inner.reset)(keys)
    centers = -np.pi + (np.arange(16)+.5)*np.pi/16
    angles = np.r_[np.repeat(centers, 7), [-np.pi]*5, [-np.pi/2]*6, [0.]*5]
    assert len(angles) == 128
    background = np.zeros((n,3), np.float32)
    background[128:, 0] = 1.6*np.cos(angles)
    background[128:, 1] = np.minimum(1.6*np.sin(angles),0.)
    modes = np.r_[np.zeros(128, np.int32), np.ones(128, np.int32)]
    state = state._replace(wind_mode=jnp.asarray(modes), wind_background_m_s=jnp.asarray(background),
        wind_ou_m_s=jnp.zeros((n,3)), wind_gust_vector_m_s=jnp.zeros((n,3)))
    state, obs = strong_arrays((state, obs))
    runner = ppo.RunnerState(state, obs, env.get_critic_obs(state, obs),
        jax.random.PRNGKey(2026111923), jnp.zeros(n), jnp.zeros(n,jnp.int32))
    alive = np.ones(n,bool)
    lifetime = np.zeros(n)
    hits = np.zeros(n)
    accum = [[],[]]
    for block in range(40):
        runner, transitions = collect(learner.params,runner)
        packet = jax.device_get(transitions)
        done = np.asarray(packet.done,bool)
        active = alive[None,:] & np.concatenate((np.ones((1,n),bool),np.cumprod(~done[:-1],axis=0).astype(bool)))
        for group in (0,1):
            accum[group].append(sums(packet.metrics,active & (modes==group)[None,:]))
        lifetime += active.sum(axis=0)*env.dt
        hits = np.maximum(hits,np.max(np.where(active,np.asarray(packet.hit_count),0.),axis=0))
        alive &= ~done.any(axis=0)
        assert np.all(np.asarray(packet.metrics['v321/finite'])==1.)
        assert not np.any(np.asarray(packet.metrics['wind/positive_y_violation']))
        if (block+1)%10==0:
            print('FROZEN',label,(block+1)*.64,'s',flush=True)
        del packet,transitions
    groups={}
    for group,name in ((0,'clean'),(1,'wind160')):
        mask=modes==group
        total=combine(accum[group])
        groups[name]=dict(n=128,survival=float(alive[mask].mean()),mean_seconds=float(lifetime[mask].mean()),
            mean_hits=float(hits[mask].mean()),hit12=float(np.mean(hits[mask]>=12)),
            quality=reference_quality_summary(total),sums=total)
    sector=np.asarray(wind_sector(jnp.asarray(background)))
    by_sector={str(i):dict(n=int(np.sum((modes==1)&(sector==i))),
        survival=float(alive[(modes==1)&(sector==i)].mean()),
        mean_seconds=float(lifetime[(modes==1)&(sector==i)].mean())) for i in range(8)}
    result=dict(label=label,seconds=25.6,groups=groups,wind_sectors=by_sector,
        alive=alive.tolist(),lifetime_s=lifetime.tolist(),hits=hits.tolist(),
        wind_vectors_m_s=background.tolist(),optimizer_updates=0)
    save(folder/('EVAL_'+label+'.json'),result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--arm',choices=('original_recipe','capture_margin'),required=True)
    parser.add_argument('--updates',type=int,choices=(3,128),default=128)
    opts=parser.parse_args()
    assert os.environ.get('PP_MEMORY_SCOPE_ACTIVE')=='YES'
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    manifest=json.loads(Path(PLAN['source_manifest']).read_text())
    assert all(digest_file(SOURCE/name)==sha for name,sha in manifest['source_files'].items())
    with jax.default_device(jax.devices('cpu')[0]):
        stack=load_stack(ROOT,SOURCE,325)
        parent,step,stage,source_update=restore_training_state(stack,Path(PLAN['source']))
        assert digest_file(Path(PLAN['source']))==PLAN['source_sha256']
        assert (stage,source_update,int(parent.opt.t))==(15,14400,963375)
        parent_digest=numeric_digest(parent)
        assert stack.bootstrap_anchor_stage == 14
        assert numeric_digest(stack.bootstrap_stage_anchor['pi']) != numeric_digest(parent.params['pi'])
        source_anchor_digest = numeric_digest(stack.bootstrap_stage_anchor)
        stack.cfg=course_config(stack,stack.spec.course[14])
        args=argparse.Namespace(**stack.parent['args'])
        vars(args).update(learner_settings_for_stack(stack))
    args.wind_training_mix=stack.wind_training_mix
    args.n_envs=256
    args.n_steps=128
    args.minibatch_size=4096
    assert stack.cfg.wind_config.max_speed_m_s==1.6 and stack.cfg.wind_config.wind_y_nonpositive
    assert stack.cfg.wind_config.mode_weights==(.1,.9,0.,0.)
    folder=AUDIT/(('smoke_r3_' if opts.updates==3 else 'pilot_r3_')+opts.arm)
    folder.mkdir(exist_ok=False)
    cloud=None
    if opts.updates==128:
        import wandb
        cloud=wandb.init(entity='fushi37',project='pingpong-mjx',id=os.environ['S15_RUN_ID'],
            name=os.environ['S15_RUN_ID'],resume='never',mode='online',dir=os.environ['WANDB_DIR'],
            settings=wandb.Settings(x_disable_stats=True),config=dict(task='fixedS15constantwindpolish',
            max_wind_m_s=1.6,stage=15,arm=opts.arm,updates=128,n_envs=256,BC=0,
            parent_sha256=PLAN['source_sha256'],gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
    release_caches()
    print('S15_CONSTRUCT_NATIVE',datetime.now().astimezone().isoformat(),flush=True)
    native=stack.env_class(stack.xml,n_envs=256,cfg=stack.cfg)
    env=CaptureEnv(DirectionEnv(native,balanced=False),enabled=opts.arm=='capture_margin')
    learner=jax.device_put(parent)
    release_caches()
    keys=jax.random.split(jax.random.PRNGKey(2026102807),256)
    state,obs=jax.jit(env.reset)(keys)
    state,obs=strong_arrays((state,obs))
    # Logger parity: identical native state/obs/reward/done for four fixed steps.
    @jax.jit
    def pair(s):
        action=jnp.full((256,7),.01)
        old=env.inner.step(s,action)
        new=env.step(s,action)
        return old,new
    print('S15_NATIVE_PARITY_BEGIN',flush=True)
    probe=state
    for _ in range(4):
        old,new=pair(probe)
        assert numeric_digest((old[0],old[1],old[3]))==numeric_digest((new[0],new[1],new[3]))
        cost=new[4]['reward/capture_margin_cost']
        np.testing.assert_array_equal(jax.device_get(new[2]),jax.device_get(jnp.where(cost<0.,old[2]+cost,old[2])))
        assert not np.any(np.asarray(cost)[np.asarray(probe.wind_mode)==0])
        probe=old[0]
    print('S15_NATIVE_PARITY_PASSED',flush=True)
    del pair,probe,old,new
    release_caches()
    runner=stack.ppo.RunnerState(state,obs,env.get_critic_obs(state,obs),jax.random.PRNGKey(2026102808),
        jnp.zeros(256),jnp.zeros(256,jnp.int32))
    collect,update,settings=make_functions(stack,env,args)
    save(folder/'PREFLIGHT.json',dict(stage=15,fixed_stage=True,source_update=source_update,
        true_source_step=step,source_adam_t=int(parent.opt.t),source_sha256=PLAN['source_sha256'],
        source_learner_digest=parent_digest,physical_cfg=asdict(stack.cfg),xml_sha256=digest_file(stack.xml),
        sampler=opts.arm,observation_dimensions=[env.obs_dim,env.critic_obs_dim,env.act_dim],
        ppo=settings,anchor_source='original_saved_S15_stage_entry',source_anchor_digest=source_anchor_digest,capture_margin=opts.arm=='capture_margin',native_logger_parity_4steps_exact=True,seed=2026102807,gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES']))
    if opts.updates==128:
        original=json.loads((AUDIT/'pilot_r2_original_sampling/EVAL_source.json').read_text())
        save(folder/'EVAL_source_reused.json',original)
        cloud.summary['source_frozen']=original['groups']
    window=[deque(maxlen=32) for _ in range(9)]
    print('S15_LEARNING_BEGIN',opts.arm,opts.updates,flush=True)
    began=time.monotonic()
    with (folder/'updates.jsonl').open('x') as stream:
        for iteration in range(1,opts.updates+1):
            runner,transitions=collect(learner.params,runner)
            new,loss=update(learner,runner,transitions)
            loss={k:float(v) for k,v in jax.device_get(loss).items()}
            assert int(new.opt.t)==int(learner.opt.t)+32 and loss['ppo_minibatches_applied']==32.
            assert loss['counterfactual_supervision_mse']==0.
            assert numeric_digest(new.params['teacher_pi'])==numeric_digest(parent.params['teacher_pi'])
            packet=jax.device_get(transitions.metrics)
            modes=np.asarray(packet['wind/mode']);sector=np.asarray(packet['polish/wind_sector'])
            assert np.all(np.asarray(packet['v321/finite'])==1.)
            assert not np.any(packet['wind/positive_y_violation']) and np.max(packet['wind/speed_m_s'])<=1.60001
            if env.balanced:
                clean,expected=cohort_labels(256)
                np.testing.assert_array_equal(modes==0,np.broadcast_to(clean,modes.shape))
                np.testing.assert_array_equal(sector[:,~np.asarray(clean)],np.broadcast_to(np.asarray(expected)[~np.asarray(clean)],sector[:,~np.asarray(clean)].shape))
            labels=np.where(modes==0,0,sector+1).astype(int)
            quality={}
            fractions={}
            for group in range(9):
                mask=labels==group
                window[group].append(sums(packet,mask))
                quality[str(group)]=reference_quality_summary(combine(window[group]))
                fractions[str(group)]=float(mask.mean())
            row=dict(actual_update=iteration,source_update=source_update,stage=15,**loss,
                clean_step_fraction=fractions['0'],peak_step_fraction=float(np.mean(packet['polish/peak_assigned'])),step_fractions=fractions,quality=quality,
                capture={k:float(np.mean(packet[k])) for k in CAPTURE_FIELDS},
                wall_seconds=time.monotonic()-began)
            stream.write(json.dumps(row)+'\n');stream.flush()
            if cloud:
                flat={k:v for k,v in row.items() if isinstance(v,(int,float))}
                flat.update({f'coverage/{k}':v for k,v in fractions.items()})
                flat.update({f'quality/{g}/{k}':v for g,q in quality.items() for k,v in q.items()})
                flat.update(row['capture'])
                cloud.log(flat)
            learner=new
            if iteration==1 or iteration%16==0: print('S15_UPDATE',iteration,'clean',fractions['0'],'quality',quality,flush=True)
            del transitions,packet
    checkpoint=folder/'full_learner.pkl'
    with checkpoint.open('xb') as stream:
        pickle.dump(dict(schema='v325_fixed_s15_polish_optin_v1',params=jax.device_get(learner.params),
            opt=jax.device_get(learner.opt),step=step+opts.updates*256*128,
            source_update=source_update,actual_updates=opts.updates,source_sha256=PLAN['source_sha256'],
            sampler=opts.arm,stage=15,source_anchor=jax.device_get(stack.bootstrap_stage_anchor),
            anchor_source='original_saved_S15_stage_entry',capture_margin=opts.arm=='capture_margin'),stream)
    final=evaluation(env,stack.ppo,collect,learner,'final',folder) if opts.updates==128 else None
    result=dict(complete=True,stage=15,maximum_wind_m_s=1.6,updates=opts.updates,
        true_adam_t=int(learner.opt.t),checkpoint=str(checkpoint),checkpoint_sha256=digest_file(checkpoint),
        parent_unchanged=numeric_digest(parent)==parent_digest,formal_curriculum_advance=False,final=final)
    save(folder/'RESULT.json',result)
    if cloud:
        cloud.summary['final_frozen']=final['groups'];cloud.finish()
    print('S15_PILOT_COMPLETE',opts.arm,flush=True)


if __name__=='__main__':
    main()
