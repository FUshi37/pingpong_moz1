"""Paired frozen parent/candidate own-state rollouts on all old/new validation."""

import argparse
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_array_archive_unique, write_json_unique
from .audit_full_prior_world import AuditReader, make_bank_builder, read_selected_trial, release_cpu_heap, teacher_seeds
from .auxiliary_refresh import restore_auxiliary, validation_batches
from .contracts import config_hash, file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .full_prior_seeds import select_teacher_seeds
from .full_prior_world import full_prior_world_rollout
from .native_stack import build_native_environment
from .new_pose_audit import diagnostic_starts, prefix_validity, select_windows
from .normal_checkpoint import load_components
from .physics_metrics import ball_metrics, robot_metrics
from .position_contact_data import prepare_position_contact_batch
from .release_alignment import compare_releases
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .supplement_data import SupplementReader
from .world_control import ControlContract


def audit(cfg, args, guard, report):
    root=repository_root()/cfg['resources']['output_root']
    old=AuditReader(root/'support_replay_full_a01',guard.budget,max_bytes=0)
    new=SupplementReader(cfg,cfg['targeted_supplement']['collection_reference'],guard.budget)
    parameters,options,refs,parent,parent_ref=load_components(cfg,old,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    candidate_state,_=restore_auxiliary(cfg,cfg['auxiliary_refresh']['candidate_checkpoint'])
    candidate={k:candidate_state[k] for k in ('parameters','scale')}
    if numeric_tree_digest(parent['scale'])!=numeric_tree_digest(candidate['scale']):
        raise ValueError('Paired comparison requires identical inherited normalization')
    models=dict(parent=parent,candidate=candidate)
    frozen=numeric_tree_digest(dict(robot=parameters['robot'],**models))
    bundle=build_native_environment(cfg,n_envs=1)
    contract=ControlContract.from_native(bundle.environment);builder=make_bank_builder(options,contract)
    report.update(parent_reference=parent_ref,candidate_reference=cfg['auxiliary_refresh']['candidate_checkpoint'],
        fixed_model_references=refs,source_dataset_sha256=dict(old=old.identity,new=new.identity),
        final_test_used=False,training_sources_used=False,learned_contact_residual=False,
        sampling='all_validation_lanes_in_each_source_trial;first_controlled_release_centered;misses_at_reset',
        comparison='identical_complete_seeds_raw_actions_and_windows;own_robot_ball_control_history_recursion',
        scopes=['task_prefix','finite_physics'],windows=[],reconstructions=[],event_diagnostics=[])

    @jax.jit
    def predict(auxiliary,seed,context,actions,goal,raw):
        _,trace=full_prior_world_rollout(parameters['robot'],auxiliary,seed,actions,
            context=context,control_contract=contract,task_goal_xyz=goal,ar1_enabled=False,**options)
        physical=jax.tree.map(lambda x:x.reshape((-1,)+x.shape[2:]),trace['physics'])
        batch=prepare_position_contact_batch(None,raw)
        metrics,_=ball_metrics(physical['contact'],batch);metrics.update(robot_metrics(physical,raw,batch))
        finite=jnp.all(jnp.isfinite(trace['actor_observation']),axis=-1)
        for value in (physical['q_after_rad'],physical['dq_after_rad_s'],
                      physical['contact']['after'].position_m,physical['contact']['after'].velocity_m_s):
            finite &= jnp.all(jnp.isfinite(value).reshape(actions.shape[0],5,-1,value.shape[-1]),axis=(1,3))
        return dict(metrics=metrics,finite=finite,release=physical['contact']['release'],
            position=physical['contact']['before'].position_m,velocity=physical['contact']['before'].velocity_m_s)

    pieces,regions,nonfinite,prefix_counts={},{},{},{}
    source_totals={};batch_number=0
    for source,reader in [('old',old),('new',new)]:
        names=list(reader.records);source_totals[source]=0
        for trial in range(len(names)//2):
            keys=names[2*trial:2*trial+2];record=reader.records[keys[0]];groups=record['record']['groups']
            for selection in validation_batches(groups):
                guard.check();lanes=selection['lanes'];useful=selection['useful'];source_totals[source]+=useful
                raw,values,_=read_selected_trial(reader,keys,lanes)
                bank,context,checks=teacher_seeds(reader,record,lanes,raw,values,options,builder)
                report['reconstructions'].append(dict(source=source,trial=trial,batch=batch_number,
                    lanes=lanes[:useful].tolist(),computational_padding=len(lanes)-useful,checks=checks))
                _,prefix=prefix_validity(raw['ball_model_valid'],np.zeros_like(raw['ball_model_valid']),
                    ~values['state__task_within_task_bounds'][1:],values['state__task_within_task_bounds'][0])
                region=np.array([groups[i]['region_id'] for i in lanes[:useful]])
                for horizon in (1,8,32,64,128):
                    guard.check()
                    starts=diagnostic_starts(raw['event_controlled_release'] & raw['ball_model_valid'],horizon)
                    selected={k:select_windows(v,starts*5,horizon*5) for k,v in raw.items()}
                    actions=select_windows(values['action__raw_normalized'],starts,horizon)
                    seed=select_teacher_seeds(bank,starts)
                    goal=jnp.asarray(values['state__goal_xyz_m'][starts,np.arange(len(lanes))])
                    primary=select_windows(prefix,starts*5,horizon*5)[:,:useful]
                    prefix_key=f'{source}/{horizon}';prefix_counts[prefix_key]=prefix_counts.get(prefix_key,0)+int(primary.all(axis=0).sum())
                    arrays=dict(lanes=lanes[:useful],starts=starts[:useful],task_prefix_valid=primary,
                                finite_physics=selected['ball_model_valid'][:,:useful])
                    event_results={}
                    for model,auxiliary in models.items():
                        result=jax.device_get(predict(auxiliary,seed,context,jnp.asarray(actions),goal,
                                                     jax.tree.map(jnp.asarray,selected)))
                        finite_key=f'{source}/{model}/{horizon}'
                        nonfinite[finite_key]=nonfinite.get(finite_key,0)+int((~result['finite'][:,:useful].all(axis=0)).sum())
                        arrays[model+'__finite']=result['finite'][:,:useful]
                        for metric,(value,mask) in result['metrics'].items():
                            value,mask=value[...,:useful],mask[...,:useful]
                            arrays[model+'__'+metric]=np.where(np.isfinite(value),value,0.)
                            arrays[model+'__'+metric+'__finite']=np.isfinite(value)
                            arrays[model+'__'+metric+'__valid']=mask
                            for scope,valid in [('task_prefix',primary),('finite_physics',selected['ball_model_valid'][:,:useful])]:
                                use=mask & (valid[-1] if value.ndim==1 else valid)
                                key=f'{source}/{model}/{horizon}/{scope}/{metric}'
                                pieces.setdefault(key,[]).append(value[use])
                                regions.setdefault(key,[]).append(np.broadcast_to(region,use.shape)[use])
                        if horizon in (8,128):
                            predicted=result['release'][:,:useful] & (result['velocity'][:,:useful,2]>0.)
                            actual=selected['event_controlled_release'][:,:useful]
                            event_results[model]={scope:compare_releases(actual & valid,predicted & valid,
                                selected['ball_qpos_before'][:,:useful,:3],selected['ball_qvel_before'][:,:useful,:3],
                                result['position'][:,:useful],result['velocity'][:,:useful],
                                [f'{source}/{trial}/{int(lane)}' for lane in lanes[:useful]])
                                for scope,valid in [('task_prefix',primary),('finite_physics',selected['ball_model_valid'][:,:useful])]}
                        del result
                    filename=f'errors_b{batch_number:02d}_h{horizon}.npz'
                    archive=write_array_archive_unique(args.output_dir/filename,arrays,guard.budget)
                    window=dict(source=source,trial=trial,batch=batch_number,horizon=horizon,
                        lanes=lanes[:useful].tolist(),starts=starts[:useful].tolist(),archive_file=filename,archive=archive,
                        event_definition='native_controlled_vs_model_upward_geometric;unequal_counts_ambiguous',
                        ordinal_release_diagnostic=event_results)
                    path=args.output_dir/f'window_b{batch_number:02d}_h{horizon}.json'
                    digest=write_json_unique(path,window,guard.budget)
                    report['windows'].append(dict(path=str(path),sha256=digest,source=source,trial=trial,horizon=horizon))
                print(f'[PAIRED] {source} trial={trial} batch={batch_number} cases={useful} all_horizons_complete',flush=True)
                batch_number+=1
                del raw,values,bank,selected,seed,arrays
                release_cpu_heap()
    summary,strata={},{}
    for key,parts in pieces.items():
        value=np.concatenate(parts);ids=np.concatenate(regions[key])
        summary[key]=summarize_errors([value])
        strata[key]={str(int(region)):summarize_errors([value[ids==region]]) for region in np.unique(ids)}
    if numeric_tree_digest(dict(robot=parameters['robot'],**models))!=frozen:
        raise ValueError('Paired audit changed frozen parameters')
    report.update(status='AUXILIARY_REFRESH_PAIRED_AUDIT_COMPLETE',metrics=summary,by_region=strata,
        validation_sources=source_totals,model_nonfinite_windows=nonfinite,
        complete_task_prefix_windows=prefix_counts,frozen_before_sha256=frozen,frozen_after_sha256=frozen,
        world_qualified=False,promoted=False,policy_or_training_bank_modified=False,
        limitations=['Same-clock and own-release errors are both retained',
                     'Different horizons have different complete-prefix survivor populations',
                     'No Actor/Critic or robot-proxy update;no task-training qualification'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--gpu-uuid',required=True)
    args=parser.parse_args();cfg=load_config()
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,3600.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        sources=sorted(Path(__file__).parent.glob('*.py'))
        report=dict(schema='wmj_v1_auxiliary_refresh_paired_audit',training_version='WMJ-V1',status='INCOMPLETE',
            gpu_uuid=args.gpu_uuid,config_sha256=config_hash(cfg),
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sources},actor_updates=0,critic_updates=0,model_updates=0)
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report,sources={p.name:p.read_text() for p in sources}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc);raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
