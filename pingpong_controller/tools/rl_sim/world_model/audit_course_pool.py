"""Guarded 1024-lane course/pool/gradient validation, with zero updates."""

import argparse
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import init_critic
from .artifacts import write_json_unique
from .audit_full_prior_world import (AuditReader, make_bank_builder,
    read_selected_trial, teacher_seeds, release_cpu_heap)
from .branch_bank import BranchSeedBank
from .contracts import config_hash, file_hash, load_config, repository_root
from .course_env import init_course, course_critic_layout
from .course_goals import build_goal_table
from .course_learner import (make_course_gradient, course_gradient_window,
    commit_course_windows, course_runtime_signature, validate_course_config,
    prepare_course_session, compact_course_window, compact_critic_arrays)
from .course_rollout import (course_policy_window, record_captured_seeds,
                             validate_course_window, course_actor_objective)
from .full_prior_seeds import select_teacher_seeds
from .imagined_env import EpisodeContract, init_imagined
from .imagination_checkpoint import save_checkpoint, restore_checkpoint
from .imagination_learner import (init_learner, slice_environment,
    critic_training_arrays, critic_gradient_batch)
from .migration import expand_actor_history, load_v321_actor
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .persistent_pool import init_pool, count_as_int
from .runtime_guard import RunGuard
from .state_adapter import numeric_tree_digest
from .train_imagination import (validate_no_update_contract, gradient_summary,
                                tree_difference, timed)
from .world_control import ControlContract


def select_initial_seeds(reader, names):
    """Select by native reset validity and native contact lead before rollout."""
    arrays = reader.read(names[0], names={'state__task_within_task_bounds',
                                        'substep__event_contact_entry'})
    entries = arrays['substep__event_contact_entry'].reshape(-1, 1024)
    first = np.where(entries.any(axis=0), entries.argmax(axis=0)+1, -1)*.001
    train = np.array([g['split']=='train' for g in reader.records[names[0]]['record']['groups']])
    valid = train & arrays['state__task_within_task_bounds'][0] & (first>=.080)
    lanes = np.flatnonzero(valid)[:32]
    if len(lanes)<4:
        raise ValueError('Insufficient complete training-partition precontact seeds')
    return lanes, first[lanes], dict(native_eligible_count=int(valid.sum()),
        native_training_count=int(train.sum()), selected_lanes=lanes.tolist(),
        independent_native_seeds=len(lanes), selection='native admission and >=80ms actual contact lead',
        all_imagined_outcomes_retained=True, coverage='first acquisition only; numerical diagnostic')


def summarize_windows(windows):
    keys = ('valid', 'terminated', 'truncated', 'model_truncated')
    result = {key:int(sum(np.asarray(getattr(w.transitions,key)).sum() for w in windows))
              for key in keys}
    events = ('valid_release', 'valid_apex', 'goal_completed', 'goal_success',
              'goal_switched', 'disturbance_started', 'disturbance_applied')
    result['events'] = {key:int(sum(np.asarray(w.transitions.task_events[key]).sum()
                                   for w in windows)) for key in events}
    result['scheduled_transitions'] = sum(w.transitions.valid.size for w in windows)
    return result


def audit(cfg, args, guard, report):
    reward = validate_no_update_contract(cfg)
    root = repository_root()/cfg['resources']['output_root']
    reader = AuditReader(root/'support_replay_full_a01', guard.budget, max_bytes=0)
    parameters, physics, references, auxiliary, aux_reference = load_components(cfg, reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    native = build_native_environment(cfg, n_envs=1)
    control = ControlContract.from_native(native.environment)
    names = list(reader.records)[:2]
    lanes, lead, selection = select_initial_seeds(reader, names)
    raw, values, _ = read_selected_trial(reader, names, lanes)
    bank, context, checks = teacher_seeds(reader, reader.records[names[0]], lanes, raw, values,
        physics, make_bank_builder(physics, control))
    initial = select_teacher_seeds(bank, jnp.zeros(len(lanes), jnp.int32))
    # Local holding starts over its actual racket, with a registered apex height.
    goals = jnp.concatenate((initial.cached_racket_pose[:, :2], jnp.full((len(lanes),1),1.4)),axis=-1)
    base = init_imagined(initial, context, goals)
    unique = init_course(base, jax.random.PRNGKey(args.seed))
    # The catalog is numerical-only; it does not certify dynamic goal transitions.
    table = build_goal_table(np.unique(np.asarray(goals),axis=0),
        static_admitted=np.ones(len(np.unique(np.asarray(goals),axis=0)),bool))
    layout = course_critic_layout(unique)
    if layout['dimension'] != cfg['imagination_implementation']['critic']['input_dimension']:
        raise ValueError('Course critic layout differs from the registered JSON')
    write_json_unique(args.output_dir/'critic_layout.json', layout, guard.budget)
    actor = expand_actor_history(load_v321_actor(repository_root(),cfg['lineage']['actor_source']))
    critic = init_critic(jax.random.PRNGKey(args.seed+1),layout['dimension'])
    model_id = config_hash(dict(references=references, auxiliary=aux_reference))
    cache = BranchSeedBank(model_identity=model_id, stage_id='S01')
    unique_keys = jax.random.split(jax.random.PRNGKey(args.seed+2),len(lanes))
    for i,lane in enumerate(lanes):
        cache.add_teacher_precontact(slice_environment(unique,i,i+1), unique_keys[i:i+1],
            split='train',source_id=f'{names[0]}/lane{lane}/control0',time_to_contact_s=lead[i])
    indices = np.arange(1024)%len(lanes)
    duplicated = jax.tree.map(lambda x:x[indices],base)
    state = init_course(duplicated,jax.random.PRNGKey(args.seed+3),jnp.arange(1024))
    keys = jax.random.split(jax.random.PRNGKey(args.seed+4),1024)
    session = init_learner(actor,critic,init_pool(state,keys),seed=args.seed+5)
    session, draw = prepare_course_session(session,slice_environment(state,0,768),cache,
        minimum_precontact_lead_s=.080)
    options = dict(physics,control_contract=control,episode_contract=EpisodeContract.from_config(cfg),
        reward_contract=reward,stage_id='S01',exploration_std=cfg['imagination']['exploration_std'])
    robot = parameters['robot']
    frozen = dict(actor=session.actor,critic=session.critic,target=session.target_critic,
        actor_optimizer=session.actor_optimizer,critic_optimizer=session.critic_optimizer,
        robot=robot,auxiliary=auxiliary)
    report.update(dataset_sha256=reader.identity,model_references=references,
        auxiliary_reference=aux_reference,selection=selection,seed_checks=checks,
        critic_dimension=layout['dimension'],pool_sampling=[draw],
        frozen_before_sha256=numeric_tree_digest(frozen))
    del bank,raw,values,native
    jax.clear_caches()
    release_cpu_heap()
    guard.check()
    print('1024 logical lanes prepared; frozen256-lane gradient microbatches',flush=True)
    if args.diagnose_gradients:
        from jax.experimental import checkify
        def objective(actor, target, environment, keys):
            return course_actor_objective(actor,target,robot,auxiliary,environment,keys,
                table,horizon=64,**options)
        checked = jax.jit(checkify.checkify(jax.value_and_grad(objective,has_aux=True),
                                          errors=checkify.nan_checks))
        error, result = checked(session.actor,session.target_critic,
            slice_environment(session.pool.environment,0,256),session.pool.exploration_rng[:256])
        jax.block_until_ready(result)
        report['nan_diagnostic'] = dict(error=error.get(),forward=summarize_windows([result[0][1]]))
        print(report['nan_diagnostic'],flush=True)
        error.throw()
        report['status'] = 'NAN_DIAGNOSTIC_FINITE'
        return
    gradient_fn = make_course_gradient(robot,auxiliary,table,**options)
    (gradient,windows,metrics),elapsed = timed(course_gradient_window,
        session,table,gradient_fn,stage_id='S01',microbatch=256,horizon=64,compact=True)
    report['actor_gradient'] = dict(metrics,**gradient_summary(gradient),first_including_compile_s=elapsed)
    report['first_window'] = summarize_windows(windows)
    report['captured_seeds'] = [record_captured_seeds(cache,w,split='train',source_id=f'window0/micro{i}')
                                for i,w in enumerate(windows)]
    write_json_unique(args.output_dir/'gradient_diagnostic.json',report,guard.budget)
    guard.check()
    del windows
    release_cpu_heap()
    (warm_gradient,windows,warm_metrics),warm_s = timed(course_gradient_window,
        session,table,gradient_fn,stage_id='S01',microbatch=256,horizon=64,compact=True)
    report['actor_gradient_warmed'] = dict(warm_metrics,wall_s=warm_s,
        gradient_repeat_max_abs=tree_difference(gradient,warm_gradient),
        valid_transitions_per_s=report['first_window']['valid']/warm_s,
        scheduled_transitions_per_s=65536/warm_s,optimizer_steps=0)
    guard.check()
    print('Full frozen actor gradient complete; checking critic batches',flush=True)
    inputs,targets = compact_critic_arrays(windows)
    rng = np.random.default_rng(args.seed+7)
    critic_reports = []
    for index in range(4):
        picked = rng.integers(len(inputs),size=8192)
        (grad,loss),elapsed = timed(critic_gradient_batch,session.critic,
            inputs[picked],targets[picked],microbatch=2048)
        critic_reports.append(dict(batch=index,batch_size=8192,microbatch=2048,
            loss=loss,**gradient_summary(grad),wall_s=elapsed,updates=0,
            distinct_source_rows=len(np.unique(picked))))
        guard.check()
    report['critic_gradients'] = critic_reports
    session = commit_course_windows(session,windows)
    del windows,inputs,targets
    release_cpu_heap()
    # Persistent lanes continue. Only failed/timed-out lanes and short branches reset.
    next_session,draw = prepare_course_session(session,slice_environment(state,0,768),cache,
        minimum_precontact_lead_s=.080)
    prepared = next_session.pool
    ended = np.asarray(session.pool.environment.terminated|session.pool.environment.truncated
                       |session.pool.environment.model_truncated)
    continuing = np.flatnonzero(~ended[:768])
    for left,right in zip(jax.tree.leaves(session.pool.environment),jax.tree.leaves(prepared.environment)):
        np.testing.assert_array_equal(np.asarray(left)[continuing],np.asarray(right)[continuing])
    session = next_session
    report['pool_sampling'].append(draw)
    report['continuing_persistent_lanes'] = len(continuing)
    run = jax.jit(lambda s,k:course_policy_window(actor,robot,auxiliary,s,k,table,horizon=64,**options))
    second = [];times = []
    for start in range(0,1024,256):
        window,elapsed = timed(run,slice_environment(prepared.environment,start,start+256),
                              prepared.exploration_rng[start:start+256])
        validate_course_window(window)
        second.append(compact_course_window(window,session.target_critic));times.append(elapsed)
        record_captured_seeds(cache,window,split='train',source_id=f'window1/micro{start//256}')
        guard.check()
    report['second_window'] = dict(summarize_windows(second),microbatch_wall_s=times)
    session = commit_course_windows(session,second)
    del second
    bank_tree,bank_metadata = cache.snapshot()
    snapshot = dict(learner=session,branch_bank=bank_tree,robot=robot,auxiliary=auxiliary,goals=table)
    signature = course_runtime_signature(cfg,model_references=dict(references,auxiliary=aux_reference),
        geometry_sha256=config_hash(reader.report['geometry']),critic_layout=layout)
    checkpoint = save_checkpoint(args.output_dir/'course_pool.json',snapshot,signature=signature,
        budget=guard.budget,metadata=dict(branch_bank=bank_metadata,actor_updates=0,critic_updates=0,
            teacher_transitions_used_for_learning=0,source_commit=report['source_commit'],
            config_sha256=config_hash(cfg),model_references=dict(references,auxiliary=aux_reference)))
    restored = restore_checkpoint(checkpoint,snapshot,signature=signature)
    restored_bank = BranchSeedBank.restore(restored['branch_bank'],bank_metadata,
        model_identity=model_id,stage_id='S01')
    if numeric_tree_digest(restored)!=numeric_tree_digest(snapshot):
        raise ValueError('Full course/learner/bank restore differs')
    draw_a, log_a = prepare_course_session(session,slice_environment(state,0,768),cache,
        minimum_precontact_lead_s=.080)
    draw_b, log_b = prepare_course_session(restored['learner'],slice_environment(state,0,768),
        restored_bank,minimum_precontact_lead_s=.080)
    if numeric_tree_digest(draw_a)!=numeric_tree_digest(draw_b) or log_a!=log_b:
        raise ValueError('Checkpoint resampling/RNG continuation differs')
    a,_ = timed(run,slice_environment(session.pool.environment,0,256),session.pool.exploration_rng[:256])
    b,_ = timed(run,slice_environment(restored['learner'].pool.environment,0,256),
                restored['learner'].pool.exploration_rng[:256])
    difference = tree_difference(a,b)
    if difference:
        raise ValueError('Same course state/RNG/checkpoint continuation differs')
    report['resume'] = dict(checkpoint=checkpoint,runtime_signature=signature,
        max_abs=difference,bank_rows=len(restored_bank.rows),resampling_replay_exact=True)
    report['frozen_after_sha256'] = numeric_tree_digest(dict(actor=session.actor,critic=session.critic,
        target=session.target_critic,actor_optimizer=session.actor_optimizer,
        critic_optimizer=session.critic_optimizer,robot=robot,auxiliary=auxiliary))
    if report['frozen_after_sha256']!=report['frozen_before_sha256']:
        raise ValueError('Frozen parameters or optimizers changed')
    report.update(status='COURSE_POOL_NO_UPDATE_COMPLETE',
        imagined_valid_transitions=count_as_int(session.pool.imagined_transition_words),
        task_qualification=False,optimizer_update_throughput_measured=False,
        limitations=['32 or fewer independent initial physical seeds',
            'S01 GPU pool; S04/S05 timing and active-force restore covered by numerical unit tests',
            'no learned-policy improvement or global task qualification'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True)
    parser.add_argument('--seed',type=int,default=2026110924)
    parser.add_argument('--diagnose-gradients',action='store_true')
    args = parser.parse_args()
    cfg = load_config()
    validate_no_update_contract(cfg)
    validate_course_config(cfg)
    with RunGuard(cfg,args.gpu_uuid,args.output_dir,1800.) as guard:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        sources = sorted(Path(__file__).parent.glob('*.py'))
        report = dict(schema='wmj_v1_course_pool_no_update',status='INCOMPLETE',
            training_version='WMJ-V1',config_sha256=config_hash(cfg),gpu_uuid=args.gpu_uuid,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            source_sha256={p.name:file_hash(p) for p in sources},
            arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            actual_actor_updates=0,actual_critic_updates=0,actual_world_updates=0)
        write_json_unique(args.output_dir/'implementation.json',
            dict(report=report,source={p.name:p.read_text() for p in sources}),guard.budget)
        write_json_unique(args.output_dir/'config_snapshot.json',cfg,guard.budget)
        try:
            if jax.default_backend()!='gpu' or len(jax.devices())!=1:
                raise ValueError('Exactly one CUDA device required')
            audit(cfg,args,guard,report)
        except Exception as exc:
            report['error']=repr(exc)
            raise
        finally:
            report['resources']=guard.report()
            write_json_unique(args.output_dir/'result.json',report,guard.budget)


if __name__=='__main__':
    main()
