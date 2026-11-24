"""Opt-in V325 S15 polish, fixed1.6m/s, with full learner and direction gates."""
import argparse
import ast
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from train_s15_capture_r3 import (AUDIT,ROOT,SOURCE,PLAN,FIELDS,DirectionEnv,make_functions,
    load_stack,restore_training_state,course_config,numeric_digest,digest_file,
    learner_settings_for_stack,strong_arrays,release_caches,save)
from capture_margin import CaptureEnv, CAPTURE_FIELDS
from s15_tracking import subgroup_gate
from dynamic_tracking import DynamicS15Progress as S15Progress
from pilot_decision import decide,compare
from fixed_plane_v321_long_episode import CheckpointRetention,QUALITY_FIELDS
from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS,make_reference_gate,gate_quality
from fixed_plane_wind_strong import quality_gate
from fixed_plane_wind_objectives import quality_violation_score
from direction_speed_cohorts import cohort_labels
from contact_metrics import contact_normal_errors
from critic_diagnostics import make_critic_diagnostics
from legacy_wind_console import LegacyWindConsole
from frozen_calm_fresh_velocity import (components_factory,adam_factory,new_trainable_state,TAG,prepare_teacher_optimizer)
from ball_velocity_teacher import VelocityTeacherObservationEnv
from strict_1200_acceptance import FullEpisode99Progress,gate99
from stratified_wind_sampling import StratifiedWindEnv,layout
from pose_startup_ramp import degrees as pose_startup_degrees
from training_snapshot import save_snapshot,load_snapshot,prune_owned_snapshots,state_digest
import gc
import ctypes
from correct_contact_logger import CorrectContactEnv
from native_contact_force import contact_normal_forces

CONTACT_FIELDS=('actual_contact/available_impact_contacts','actual_contact/normal_error_sum_deg',
    'actual_contact/normal_error_sq_sum_deg2','impact/count','impact/normal_error_deg_sq_sum','polish/background_speed_m_s')


class ActualContactEnv(CorrectContactEnv):
    def step(self,state,action):
        after,obs,reward,done,metrics=super().step(state,action)
        contact=after.data._impl.contact
        force=contact_normal_forces(after.data._impl.efc_force,
            np.asarray(contact.efc_address),np.asarray(contact.dim),self.force_cone,xp=jnp)
        outgoing=jnp.stack([metrics['dynamics/target_v'+axis+'_m_s'] for axis in 'xyz'],axis=-1)
        incoming=state.data.qvel[:,self.ball_vadr:self.ball_vadr+3]
        values=contact_normal_errors(contact.frame,contact.geom,contact.dist,force,incoming,outgoing,
                                    self.ball_geom_id,self.racket_geom_id)
        metrics['polish/background_speed_m_s']=jnp.linalg.norm(state.wind_background_m_s,axis=-1)
        edge=metrics['raw_physical_contact_edge']
        metrics.update({'actual_contact/available_impact_contacts':edge*values['available_contacts'],
            'actual_contact/normal_error_sum_deg':edge*values['error_deg_sum'],
            'actual_contact/normal_error_sq_sum_deg2':edge*values['error_deg_sq_sum']})
        return after,obs,reward,done,metrics


def continuation_functions(stack,env,args):
    original=stack.ppo.make_train_fns
    def with_contact_fields(**kwargs):
        kwargs['full_rollout_metric_names']=tuple(dict.fromkeys(kwargs['full_rollout_metric_names']+CONTACT_FIELDS))
        kwargs['reference_params']=stack.frozen_s15_actor_params
        stack.teacher_legacy_history_adapter_removed=prepare_teacher_optimizer(stack.ppo)
        return original(**kwargs)
    stack.ppo.make_train_fns=with_contact_fields
    try:return make_functions(stack,env,args)
    finally:stack.ppo.make_train_fns=original


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-full',type=Path)
    parser.add_argument('--experimental-candidate',action='store_true')
    parser.add_argument('--pose-startup-ramp-s',type=float,choices=(1.,3.),default=1.)
    parser.add_argument('--stratified-wind',action='store_true')
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--max-updates',type=int,default=0)
    opts=parser.parse_args()
    assert opts.runtime.resolve().parent==Path('/dev/shm')
    assert os.environ.get('PP_MEMORY_SCOPE_ACTIVE')=='YES'
    assert opts.experimental_candidate and opts.pose_startup_ramp_s==3. and opts.stratified_wind
    assert os.environ['CUDA_VISIBLE_DEVICES']==opts.gpu_uuid
    assert opts.gpu_uuid in ('GPU-91f9b105-f5c8-b00e-de70-39d3ee1ce7b4','GPU-e74458ec-002a-1bac-e0be-d0a5713b661e')
    assert jax.default_backend()=='gpu' and len(jax.devices())==1
    manifest=json.loads(Path(PLAN['source_manifest']).read_text())
    assert all(digest_file(SOURCE/name)==sha for name,sha in manifest['source_files'].items())
    ramp_evidence=json.loads((AUDIT/'pose_ramp_domains/RESULT.json').read_text())
    assert ramp_evidence['complete'] and ramp_evidence['optimizer_updates']==0
    r9=json.loads((AUDIT/'r9_pose_before_wind/RESULT.json').read_text())
    assert r9['complete'] and r9['optimizer_updates']==0
    assert r9['results']['source_pose3s']['peak_clean']['groups']['clean']['survival']>=r9['results']['source_native']['peak_clean']['groups']['clean']['survival']
    # This is an explicitly isolated learning experiment, not a promotion:
    # three-domain means improved, but remaining speed/sector regressions remain.
    sampler='stratified_speed_direction_cells_v1' if opts.stratified_wind else 'original_fullbatch'
    pilot=Path(PLAN['source'])
    pilot_sha=PLAN['source_sha256']
    opts.resume=opts.resume or pilot
    verdict=dict(experimental_unqualified=True,source_sha256=pilot_sha,
        reason='Isolated source-centered highest-precision privileged warm-start wind clone with fresh conditional-actor Adam with immutable full S15 base; BC0; clean correction identicallyzero; strict1200s99target; notfinalaccepted',
        new_goal_domains=['clean','original0--1.6mixture','fixed1.6allY<=0directions'],
        new_candidate_adopted=False,teacher_actor_privileged=True,frozen_calm_actor=True)
    with jax.default_device(jax.devices('cpu')[0]):
        stack=load_stack(ROOT,SOURCE,325)
        parent,source_step,source_stage,source_update=restore_training_state(stack,pilot)
        assert digest_file(pilot)==pilot_sha
        assert (source_stage,source_update,int(parent.opt.t))==(15,14400,963375)
        assert stack.bootstrap_anchor_stage==14
        stack.cfg=course_config(stack,stack.spec.course[14])
        args=argparse.Namespace(**stack.parent['args'])
        vars(args).update(learner_settings_for_stack(stack))
        if opts.resume.resolve()==pilot.resolve():
            learner=new_trainable_state(stack.ppo,parent,jax.random.PRNGKey(2026112037))
            total_updates=0
            true_step=source_step
        else:
            with opts.resume.open('rb') as stream:checkpoint=pickle.load(stream)
            assert checkpoint['schema']=='v325_s15_fresh_velocity_optin_v1'
            assert checkpoint['pose_startup_ramp_s']==opts.pose_startup_ramp_s and checkpoint['stratified_wind']==opts.stratified_wind
            assert checkpoint['source_sha256']==pilot_sha and checkpoint['sampler']==sampler
            assert checkpoint['stage']==15 and checkpoint['validation_root_sha256']==pilot_sha
            assert numeric_digest(checkpoint['source_anchor'])==numeric_digest(parent.params)
            assert numeric_digest(checkpoint['frozen_base_params'])==numeric_digest(parent.params)
            assert checkpoint['capture_margin'] is False
            total_updates=checkpoint['actual_updates']
            assert int(checkpoint['opt'].t)==963375+32*total_updates
            learner=stack.ppo.TrainState(checkpoint['params'],checkpoint['opt'])
            true_step=checkpoint['step']
    frozen_base=jax.device_put(parent.params)
    native_components=stack.ppo.policy_components
    stack.ppo.policy_components=components_factory(native_components,frozen_base)
    stack.ppo.adam_step=adam_factory(stack.ppo.adam_step,963375)
    stack.frozen_s15_actor_params=frozen_base
    stack.bootstrap_stage_anchor=frozen_base
    assert stack.cfg.wind_config.max_speed_m_s==1.6 and stack.cfg.wind_config.wind_y_nonpositive
    args.wind_training_mix=stack.wind_training_mix
    args.n_envs,args.n_steps,args.minibatch_size=1024,128,16384
    args.seed=2026111937
    opts.output.mkdir(exist_ok=False)
    import fixed_plane_wind_pose as pose_module
    if opts.pose_startup_ramp_s!=1.:
        def effective(state,dt_s,cfg,*,xp):
            return pose_startup_degrees(state,dt_s,cfg,opts.pose_startup_ramp_s,xp=xp)
        pose_module.effective_degrees=effective
    native=stack.env_class(stack.xml,n_envs=1024,cfg=stack.cfg)
    env=VelocityTeacherObservationEnv(StratifiedWindEnv(CaptureEnv(ActualContactEnv(native,balanced=False),enabled=False),enabled=True),enabled=True)
    assert (env.obs_dim,env.critic_obs_dim,env.act_dim,env.max_steps)==(121,453,7,240000)
    learner=jax.device_put(learner)
    initial_rng,reset_key=jax.random.split(jax.random.PRNGKey(2026111937))
    keys=jax.random.split(reset_key,1024)
    state,obs=jax.jit(env.reset)(keys)
    state,obs=strong_arrays((state,obs))
    runner=stack.ppo.RunnerState(state,obs,env.get_critic_obs(state,obs),
        initial_rng,jnp.zeros(1024),jnp.zeros(1024,jnp.int32))
    runner,learner=strong_arrays((runner,learner))
    initialization_template=(total_updates==0)
    forced_gain_obs=obs.at[:,117].set(1.)
    if not initialization_template:forced_gain_obs=forced_gain_obs.at[:,114:117].set(0.)
    base_mean=native_components(frozen_base,forced_gain_obs[:,:114])[0]
    neutral_mean=stack.ppo.policy_mean(learner.params,forced_gain_obs)
    expected=base_mean
    np.testing.assert_allclose(np.asarray(expected),np.asarray(neutral_mean),rtol=0.,atol=2.e-6)
    if initialization_template:
        assert numeric_digest(learner.params['v'])==numeric_digest(parent.params['v'])
        assert numeric_digest(learner.opt.m['v'])==numeric_digest(parent.opt.m['v'])
        assert numeric_digest(learner.opt.v['v'])==numeric_digest(parent.opt.v['v'])
    save(opts.output/'NEUTRAL_TEACHER_INITIALIZATION.json',dict(source_neutral_centered_migration_atol_2e6=initialization_template,calm_mean_exact=True,
        initialization_template_before_full_resume=bool(opts.resume_full),
        frozen_base_sha256=PLAN['source_sha256'],critic_params_and_moments_restored=True,
        source_global_Adam_t=963375,new_wind_input_rows_local_Adam_t=0,actor_critic_action=[121,453,7],
        wind_actor_full_S15_clone=True,wind_input_rows_initially_zero=True,critic_moments_restored=True,new_wind_actor_moments_fresh=True,source_actor_weights_preserved=True,BC=0))
    @jax.jit
    def calm_error(params,observations):
        mean=stack.ppo.policy_mean(params,observations)
        baseline=native_components(frozen_base,observations[:,:114])[0]
        return jnp.max(jnp.abs(mean-baseline))
    print('[V325] privileged source-centered wind clone121; full S15 calm actor frozen; pose startup',opts.pose_startup_ramp_s,
          's; wind startup remains5hits+1s; actor121; strict1200s99%',flush=True)
    release_caches()
    collect,update,settings=continuation_functions(stack,env,args)
    assert getattr(stack.ppo.adam_step,'_frozen_calm_teacher_adam',False)
    assert stack.teacher_legacy_history_adapter_removed
    critic_diagnostics=make_critic_diagnostics(stack.ppo,settings)
    fields=tuple(dict.fromkeys(QUALITY_FIELDS+HEIGHT_FIELDS+REFERENCE_FIELDS))
    progress=FullEpisode99Progress(1024,fields)
    console=LegacyWindConsole(1024,fields,180.)
    gate=make_reference_gate(quality_gate(325,stack.spec.course[14],stack.helpers.quality_passed),325)
    save(opts.output/'PREFLIGHT.json',dict(timestamp=datetime.now().astimezone().isoformat(),
        source_sha256=PLAN['source_sha256'],resume_sha256=digest_file(opts.resume),
        validation_root_sha256=pilot_sha,acceptance=verdict,stage=15,max_wind_m_s=1.6,
        actor_critic_action=[121,453,7],n_envs=1024,physical_cfg=asdict(stack.cfg),
        source_adam_t=int(learner.opt.t),source_true_step=true_step,ppo=settings,sampler=sampler,
        seed=2026111937,minibatch_size=16384,n_steps=128,native_pair4exact=False,
        native_parity_provenance="Opt-in startup timing and sampling change; native snapshot/PPO roundtrip required before first accepted update",
        pose_startup_ramp_s=opts.pose_startup_ramp_s,stratified_wind=opts.stratified_wind,
        remaining_frozen_regressions=ramp_evidence["decision"],strict_episode_success_rate=.99,
        console_format="original V325 detailed per-update",resume_learner_digest=numeric_digest(learner),
        xml_sha256=digest_file(stack.xml),source_learner_digest=numeric_digest(parent),
        original_reward_no_mature_offset=True,original_anchor=False,frozen_full_s15_anchor=True,teacher_actor_privileged=True,
        frozen_calm_actor=True,wind_actor_math_precision='highest_centered_at_source',teacher_true_ball_velocity_suffix=True,reference_control_precision='legacy',new_wind_input_Adam_origin=963375,
        legacy114_history_row_adapter_removed_for_new121head=True,
        no_curriculum_advance=True,scope='constantwindonly; allY<=0angles; no2m/s',
        checkpoint_schema='dedicatedoptinfulllearner; do notuselegacycurriculumresumer'))
    import wandb
    run=wandb.init(entity='fushi37',project='pingpong-mjx',id=opts.run_id,name=opts.run_id,
        resume='never',mode='online',dir=os.environ['WANDB_DIR'],
        settings=wandb.Settings(x_disable_stats=True),config=dict(stage=15,max_wind_m_s=1.6,
        n_envs=1024,sampler=sampler,BC=0,teacher_actor_privileged=True,frozen_calm_actor=True,actor_obs_dim=121,
        source_sha256=PLAN['source_sha256'],resume_sha256=digest_file(opts.resume),
        validation_root_sha256=pilot_sha,physical_episode_seconds=1200.,
        stage_evidence_seconds=180.,final_evidence_seconds=1200.,seed=2026111937,no_S16=True,
        update_cap=opts.max_updates,accepted_checkpoint=str(pilot),actor_guard=False,
        experimental_candidate=True,final1200s99accepted=False,pose_startup_ramp_s=opts.pose_startup_ramp_s,stratified_wind=opts.stratified_wind))
    stopped=[False]
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    retained=CheckpointRetention(opts.output,cleanup_every=5)
    hold=0
    iteration=0
    status={}
    recent_score=(0.,-1.e6,-1.e6,-1.e6)
    started=time.monotonic()

    state_contract=dict(schema='v325_fresh_velocity_episode99_v1',stage=15,n_envs=1024,n_steps=128,
        source_sha256=PLAN['source_sha256'],pose_startup_ramp_s=opts.pose_startup_ramp_s,
        stratified_wind=opts.stratified_wind,actor_critic_action=[121,453,7],
        teacher_kind=TAG,frozen_base_sha256=PLAN['source_sha256'],wind_input_Adam_origin=963375,wind_actor_authority='original_full_S15_actor',wind_actor_math_precision='highest_centered_at_source',teacher_true_ball_velocity_suffix=True,calm_actor_math_precision='legacy',wind_actor_optimizer='fresh_local_clock')
    if opts.resume_full:
        learner,runner,aux,meta=load_snapshot(opts.resume_full,learner,runner,state_contract)
        progress,console,hold=aux['progress'],aux['console'],aux['hold']
        recent_score,status=aux['recent_score'],aux['status']
        total_updates,true_step=meta['actual_updates'],meta['true_step']
        assert int(learner.opt.t)==963375+32*total_updates
        assert true_step==3497132032+131072*total_updates
        print(f'[V325] complete environment/RNG/episode state restored at u={14400+total_updates}',flush=True)

    def save_recovery():
        path=opts.output/f'trainer_state_u{14400+total_updates:08d}.zip'
        if path.exists():return
        proof=save_snapshot(path,learner,runner,dict(progress=progress,console=console,hold=hold,
            recent_score=recent_score,status=status),state_contract,
            dict(actual_updates=total_updates,true_step=true_step,run_id=opts.run_id))
        prune_owned_snapshots(opts.output,path)
        run.summary['latest_complete_training_state']=str(path)
        print(f'[V325 state] u={14400+total_updates} bytes={proof["bytes"]}',flush=True)

    def checkpoint_now(kind='periodic'):
        path=opts.output/f'{kind}_u{14400+total_updates:08d}.pkl'
        if path.exists():return
        payload=dict(schema='v325_s15_fresh_velocity_optin_v1',pose_startup_ramp_s=opts.pose_startup_ramp_s,stratified_wind=opts.stratified_wind,params=jax.device_get(learner.params),
            opt=jax.device_get(learner.opt),step=true_step,source_update=14400,
            actual_updates=total_updates,source_sha256=PLAN['source_sha256'],
            sampler=sampler,stage=15,anchor_source='frozen_full_S15_u14400',
            capture_margin=sampler=='capture_margin',
            source_anchor=jax.device_get(frozen_base),frozen_base_params=jax.device_get(frozen_base),
            teacher_kind=TAG,actor_obs_dim=121,new_wind_input_Adam_origin=963375,validation_root_sha256=pilot_sha,
            last_summary=status,all32_courses_completed=False,legacy_actor_and_critic_optimizer_reset=False,new_wind_actor_optimizer='fresh_local_clock',frozen_base_source_optimizer=jax.device_get(parent.opt),
            training_episode99_ready=kind=='converged',s15_polish_completed=False)
        with path.open('xb') as stream:pickle.dump(payload,stream)
        if kind=='periodic':retained.register(path,recent_score)
        run.summary['best_checkpoint']=str(retained.best) if retained.best else str(path)
        run.summary['best_checkpoint_training_only']=True
        run.summary['accepted_checkpoint']=str(pilot)
        print(f'[V325 save] {path.name}; periodic_kept={len(retained.saved)}',flush=True)

    iteration_committed=True
    try:
        with (opts.output/'updates.jsonl').open('x') as stream:
            while not stopped[0]:
                iteration_committed=False
                if iteration==0:
                    smoke=opts.output/'snapshot_smoke.zip'
                    save_snapshot(smoke,learner,runner,{},state_contract,dict(actual_updates=total_updates,true_step=true_step))
                    clone,clone_runner,_,_=load_snapshot(smoke,learner,runner,state_contract)
                    original_input_digest=state_digest((learner,runner))
                    assert state_digest((clone,clone_runner))==original_input_digest
                runner,transitions=collect(learner.params,runner)
                explained,target_variance=critic_diagnostics(learner.params,runner,transitions)
                new,loss=update(learner,runner,transitions)
                if iteration==0:
                    clone_runner,clone_transitions=collect(clone.params,clone_runner)
                    clone,clone_loss=update(clone,clone_runner,clone_transitions)
                    assert state_digest((new,runner,transitions,loss))==state_digest((clone,clone_runner,clone_transitions,clone_loss))
                    save(opts.output/'NATIVE_FULL_STATE_ROUNDTRIP.json',dict(input_digest=original_input_digest,
                        output_digest=state_digest((new,runner,transitions,loss)),complete_state_roundtrip_exact=True,
                        accepted_optimizer_updates=1,duplicate_verification_branch_discarded=True))
                    del clone,clone_runner,clone_transitions,clone_loss
                    smoke.unlink();smoke.with_suffix(smoke.suffix+'.json').unlink()
                    gc.collect();ctypes.CDLL(None).malloc_trim(0)
                loss={k:float(v) for k,v in jax.device_get(loss).items()}
                loss['teacher/wind_actor_Adam_t']=int(new.opt.actor_t)
                if float(target_variance)>1.e-12:
                    loss['critic/explained_variance']=float(explained)
                loss['critic/return_target_variance']=float(target_variance)
                assert all(np.isfinite(value) for value in loss.values()),'NonfinitePPOupdate'
                assert loss['ppo_minibatches_applied']==32. and loss['counterfactual_supervision_mse']==0.
                assert int(new.opt.t)==int(learner.opt.t)+32
                assert TAG in new.params
                clean_observations=transitions.obs[-1,:round(.1*1024)]
                np.testing.assert_array_equal(np.asarray(clean_observations[:,114:117]),0.)
                loss['teacher/frozen_calm_mean_max_abs_error']=float(calm_error(new.params,clean_observations))
                assert loss['teacher/frozen_calm_mean_max_abs_error']==0.,'FrozenCalmActorChanged'
                metrics=jax.device_get(transitions.metrics)
                assert np.all(np.asarray(metrics['v321/finite'])==1.)
                assert not np.any(metrics['wind/positive_y_violation']) and np.max(metrics['wind/speed_m_s'])<=1.60001
                if env.balanced:
                    clean,sectors=cohort_labels(1024)
                    np.testing.assert_array_equal(np.asarray(metrics['wind/mode'])==0,
                        np.broadcast_to(clean,np.asarray(metrics['wind/mode']).shape))
                    wind=~np.asarray(clean)
                    np.testing.assert_array_equal(np.asarray(metrics['polish/wind_sector'])[:,wind],
                        np.broadcast_to(np.asarray(sectors)[wind],np.asarray(metrics['polish/wind_sector'])[:,wind].shape))
                progress.add(metrics,jax.device_get(transitions.done))
                console.add(metrics,jax.device_get(transitions.done))
                summaries=progress.summaries()
                stage_gate=subgroup_gate(summaries,gate)
                final_gate=gate99(summaries,gate,minimum_n=16)
                hold=hold+1 if final_gate['passed'] else 0
                learner=new
                total_updates+=1;iteration+=1;true_step+=1024*128
                iteration_committed=True
                qpenalty=[quality_violation_score(gate_quality(s['quality']),325,0. if name=='clean' else 1.6)
                          for name,s in summaries.items()]
                mature=all(s['short30'].get('n',0)>=16 for s in summaries.values())
                worst_survival=min(s['short30'].get('survival',0.) for s in summaries.values())
                recent_score=(float(stage_gate['passed']),float(mature)*worst_survival,
                    -max(v[0] for v in qpenalty),-sum(v[1] for v in qpenalty))
                status=dict(update=14400+total_updates,actual_update=total_updates,session_update=iteration,stage=15,
                    true_step=true_step,**loss,stage_gate=stage_gate,final_gate=final_gate,hold=hold,
                    groups=summaries,wall_seconds=time.monotonic()-started)
                stream.write(json.dumps(status)+'\n');stream.flush()
                flat={k:v for k,v in status.items() if isinstance(v,(int,float))}
                flat.update({'polish/hold':hold,'polish/stage180_passed':int(stage_gate['passed']),
                             'polish/full1200_passed':int(final_gate['passed'])})
                masks=progress.masks(metrics)
                for name,summary in summaries.items():
                    for horizon in ('short30','stage','full1200'):
                        observed=summary[horizon]
                        flat[f'{horizon}/{name}/n']=observed.get('n',0)
                        flat[f'{horizon}/{name}/available']=int(observed.get('n',0)>0)
                        if observed.get('n',0)>0:
                            flat.update({f'{horizon}/{name}/{k}':v for k,v in observed.items()})
                    flat.update({f'quality/{name}/{k}':v for k,v in summary['quality'].items()})
                    mask=masks[name]
                    available=float(np.sum(metrics['actual_contact/available_impact_contacts']*mask))
                    contact_count=float(np.sum(metrics['polish/physical_contact_frames']*mask))
                    flat[f'contact/{name}/available_actual_normals']=available
                    flat[f'contact/{name}/edge_contact_frames']=float(np.sum(metrics['polish/edge_contact_frames']*mask))
                    if contact_count>0:
                        flat[f'contact/{name}/edge_contact_fraction']=flat[f'contact/{name}/edge_contact_frames']/contact_count
                    if available>0:
                        flat[f'contact/{name}/actual_normal_rms_deg']=float(np.sqrt(
                            np.sum(metrics['actual_contact/normal_error_sq_sum_deg2']*mask)/available))
                flat.update({name:float(np.mean(metrics[name])) for name in CAPTURE_FIELDS})
                run.log(flat)
                print(console.line(status['update'],hold,final_gate),flush=True)
                if iteration==3:save(opts.output/'FIRST3_ACTUAL_UPDATES.json',dict(finite=True,
                    Adam_steps=96,stage=15,max_wind_m_s=1.6,no_positive_y=True))
                if iteration%10==0:
                    temperature=float(subprocess.check_output(['nvidia-smi','-i',os.environ['CUDA_VISIBLE_DEVICES'],
                        '--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
                    if temperature>=78.:
                        run.summary['temperature_stop_c']=temperature
                        stopped[0]=True
                if total_updates%64==0:
                    checkpoint_now();save_recovery()
                if hold>=16:
                    checkpoint_now('converged');run.summary['training_episode99_ready']=True;run.summary['final1200s99accepted']=False;break
                if opts.max_updates and iteration>=opts.max_updates:break
                del transitions,metrics
    finally:
        if iteration_committed:
            save_recovery()
        else:
            run.summary['incomplete_update_no_state_commit']=True
        checkpoint_now('bounded_final')
        checkpoint_now()
        retained.prune()
        run.summary['all32_courses_completed']=False
        run.finish()
    print(f'[V325] stopped at u={14400+total_updates}; updates_this_process={iteration}',flush=True)


if __name__=='__main__':main()
