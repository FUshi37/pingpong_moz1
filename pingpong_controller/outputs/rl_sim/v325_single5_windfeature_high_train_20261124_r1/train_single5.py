"""Single ordinary 118-D MLP PPO with causal observed-force inputs; no residual."""
import argparse
from dataclasses import asdict
from datetime import datetime
import ctypes
import gc
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time

A = Path(__file__).resolve().parent
BASE = A.parent / 'v325_s15_constant_polish_20261119_r1'
sys.path.insert(0,str(A.parent/'v325_single_actor_ablation_20261120_r1'))
sys.path.insert(0, str(BASE))
from train_s15_capture_r3 import (ROOT, SOURCE, PLAN, load_stack, restore_training_state,
    course_config, learner_settings_for_stack, strong_arrays, numeric_digest,
    digest_file, save, make_functions, release_caches)
import jax
import jax.numpy as jnp
import numpy as np
from single5_runtime import initialize,adam_factory,MODES,configure,install_pose_mode,Single5Env,REVISION
from single5_runtime import assert_single_actor,apply_factory,actor_parameter_count
from calm_trust_region import guard_update, load_bank
from learning_stability import critic_warmup_adam,cap_actor_update
from train_s15_fresh_velocity_r31 import ActualContactEnv, CONTACT_FIELDS
from capture_margin import CaptureEnv
from xwind_sampling import XWindEnv
from randomized_onset import TIMING_FIELDS,PREVIEW_FIELDS
from targeted_reward import FIELDS as TARGETED_FIELDS
from pose_startup_ramp import degrees as pose_degrees
from fixed_plane_v321_long_episode import QUALITY_FIELDS, CheckpointRetention
from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS, make_reference_gate
from fixed_plane_wind_strong import quality_gate
from adaptive_progress import AdaptiveProgress as FullEpisode99Progress,gate99,foundation_gate99
from legacy_wind_console import LegacyWindConsole
from critic_diagnostics import make_critic_diagnostics
from training_snapshot import save_snapshot, load_snapshot, prune_owned_snapshots, state_digest
from termination_diagnostics import TERMINATION_FIELDS
from precision_maturity import PRECISION_FIELDS,verify_mature_batch,verify_parent_maturity
from startup_rehearsal import wrap_rehearsal,StartupProgress,long_rollout,FIELDS as REHEARSAL_FIELDS,PROGRESS_FIELDS
from actor_phase import phase_update
from rehearsal_stagger import warm_rehearsals
from actor_mask import output_only_adam
from weak_release_floor import FIELDS as RELEASE_FLOOR_FIELDS
from cached_initialization import runner_template,restore_cached
from qualified_branch import verify_qualified_parent,compatible_parent_contract
from phase_transition import verify_phase_parent, frozen_backbone_equal
from calm_projection import load_projection, project_wind_optimizer
from feature_training import (migrate_learner, frozen_base_equal as frozen_backbone_equal,
    verify_feature_parent as verify_phase_parent, feature_apply_factory,
    assert_supported_actor as assert_single_actor)
from wind_features import WindFeatureEnv


def continuation_functions(stack, env, args):
    original = stack.ppo.make_train_fns
    effective = {}
    trust_plan=json.loads((A/'TRAIN_PLAN.json').read_text())['calm_trust_region']
    stability_plan=json.loads((A/'TRAIN_PLAN.json').read_text())['learning_stability']
    bank=load_bank(A) if trust_plan['enabled'] else None
    def build(**kwargs):
        kwargs['full_rollout_metric_names'] += TARGETED_FIELDS+TIMING_FIELDS+PREVIEW_FIELDS+TERMINATION_FIELDS+('reward/total',)
        kwargs['full_rollout_metric_names'] += PRECISION_FIELDS+RELEASE_FLOOR_FIELDS
        kwargs['full_rollout_metric_names'] += REHEARSAL_FIELDS
        kwargs['time_limit_bootstrap']=True
        goal_fields = tuple('reward/goal_'+name for name in
            ('alive','hit','outgoing','normal','height_margin','braking','apex','failure'))
        kwargs['full_rollout_metric_names'] += goal_fields
        kwargs['full_rollout_metric_names'] = tuple(dict.fromkeys(kwargs['full_rollout_metric_names'] + CONTACT_FIELDS + ('observed_force/valid','observed_force/age_s','observed_force/error_sq_m2_s2','single5/wind_angle_rad','pose/outer_limit_deg','reward/dynamics_outgoing','reward/single5_outgoing_replace','reward/single5_apex_precision','reward/single5_center')))
        kwargs['reference_params'] = stack.single_actor_anchor_params
        kwargs['apply_every_minibatch'] = False
        kwargs['residual_l2_coef'] = 0.
        private_plan=json.loads((A/'TRAIN_PLAN.json').read_text())
        kwargs['learning_rate']=private_plan['ppo']['learning_rate']
        kwargs['actor_anchor_kl_coef']=private_plan['ppo']['actor_anchor_kl_coef']
        phase_mass=private_plan['actor_phase']['minimum_startup_mass']
        if phase_mass:
            kwargs['reset_family_min_actor_mass']=phase_mass
        if not trust_plan['enabled']:
            kwargs['actor_anchor_replay_kl_coef']=0.
        if trust_plan['enabled']:
            kwargs['actor_anchor_replay_obs']=None
            kwargs['actor_anchor_replay_kl_coef']=trust_plan['replay_kl_coef']
            kwargs['actor_anchor_replay_max_kl']=0.
        effective.update({key: value for key, value in kwargs.items()
                          if isinstance(value, (str, int, float, bool, tuple))})
        current = stack.ppo.adam_step
        # The saved history wrapper explicitly demands teacher_pi. Reproduce
        # its row conditioning in the shared fresh-actor adapter for both arms.
        epsilon = getattr(current, '_wind_history_xy_epsilon', 1.e-5)
        multiplier = getattr(current, '_wind_history_xy_lr_multiplier', 1.)
        base = getattr(current, '_wind_history_optimizer_base', current)
        effective['history_xy_epsilon'] = epsilon
        effective['history_xy_lr_multiplier'] = multiplier
        optimizer_plan=json.loads((A/'TRAIN_PLAN.json').read_text())['feature_optimizer']
        stack.ppo.adam_step = adam_factory(base, history_xy_epsilon=epsilon,
            history_xy_lr_multiplier=multiplier,
            independent_actor_critic_clipping=optimizer_plan['independent_actor_critic_clipping'],
            permanent_wind_rows_only=optimizer_plan.get('permanent_wind_rows_only',False))
        effective['permanent_wind_rows_only']=stack.ppo.adam_step._permanent_wind_rows_only
        effective['independent_actor_critic_clipping']=optimizer_plan['independent_actor_critic_clipping']
        effective['feature_warmup_minibatches'] = stack.ppo.adam_step._feature_warmup_accepted_minibatches
        effective['feature_lr_multiplier'] = stack.ppo.adam_step._feature_lr_multiplier
        projection=private_plan.get('calm_parameter_projection',{})
        if projection.get('enabled',False):
            basis,origin=load_projection(A,projection)
            stack.ppo.adam_step=project_wind_optimizer(stack.ppo.adam_step,basis,origin)
            effective['calm_projection_rank']=basis.shape[1]
        stack.ppo.adam_step=critic_warmup_adam(stack.ppo.adam_step,
            start_t=stability_plan['critic_warmup_start_t'],minibatches=stability_plan['critic_only_minibatches'])
        stack.ppo.adam_step=output_only_adam(stack.ppo.adam_step,
            enabled=private_plan['actor_training']['output_layer_only'])
        native_collect,native_update=original(**kwargs)
        return native_collect,phase_update(native_update,minimum_mass=phase_mass)
    stack.ppo.make_train_fns = build
    try:
        collect, update, _ = make_functions(stack, env, args)
        update=guard_update(update,stack.ppo.policy_mean,getattr(stack,'calm_guard_anchor_params',stack.single_actor_anchor_params),bank,
            enabled=trust_plan['enabled'],maximum_kl=trust_plan['maximum_initial_policy_kl'],
            holdout=jnp.asarray(np.load(A/'calm_holdout_observations.npy')),
            maximum_sampled_cohort_step_kl=stability_plan['maximum_sampled_cohort_step_kl'],
            sample_size=stability_plan['step_observations'],
            project_from_anchor=trust_plan.get('projection_origin')=='qualified_anchor',
            phase_grouping=trust_plan.get('phase_grouping',False),
            maximum_absolute_action_change=trust_plan.get('maximum_absolute_action_change'),
            held_maximum_absolute_action_change=trust_plan.get('held_maximum_absolute_action_change'))
        if not trust_plan['enabled']:
            update=cap_actor_update(update,stack.ppo.policy_mean,
                maximum_kl=stability_plan['maximum_sampled_cohort_step_kl'],
                sample_size=stability_plan['step_observations'])
        return collect, update, effective
    finally:
        stack.ppo.make_train_fns = original


def log_failure_events(stream,update,metrics,done):
    tags=TERMINATION_FIELDS
    selected=(np.asarray(done,bool)|(np.asarray(metrics['timing/long_confirmed_gap_event'])>.5))
    fields=TIMING_FIELDS+PREVIEW_FIELDS+tags+('terminated','truncated','single5/wind_angle_rad','polish/background_speed_m_s','wind/speed_m_s','wind/mode')
    for t,lane in zip(*np.nonzero(selected)):
        record={name:float(metrics[name][t,lane]) for name in fields}
        censor=float(metrics['sampling/rehearsal_cutoff'][t,lane])>.5
        record.update(update=update,lane=int(lane),
            event='rehearsal_censor' if censor else ('episode_end' if done[t,lane] else 'long_confirmation_gap'),
            rehearsal_lane=bool(metrics['sampling/rehearsal_lane'][t,lane]>.5))
        record['recognized_termination']=any(record[tag]>.5 for tag in tags)
        stream.write(json.dumps(record)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--mode', choices=MODES, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--update-limit', type=int)
    parser.add_argument('--resume-model', type=Path)
    parser.add_argument('--resume-full', type=Path)
    opts = parser.parse_args()
    plan = json.loads((A / 'TRAIN_PLAN.json').read_text())
    plan_sha = digest_file(A / 'TRAIN_PLAN.json')
    resume_plan_sha=plan_sha
    parent_contract=None
    qualified_receipt={}
    wind_adaptation=plan.get('training_phase')=='wind_adaptation'
    branch=plan.get('phase_transition',plan.get('qualified_branch',{}))
    phase_transition=bool(plan.get('phase_transition')) and str(opts.resume_model)==branch['model']
    if phase_transition:
        assert str(opts.resume_model)==branch['model'] and str(opts.resume_full)==branch['full_state']
    if opts.resume_model and str(opts.resume_model)==branch.get('model'):
        assert str(opts.resume_full)==branch['full_state']
        parent_contract,qualified_receipt=(verify_phase_parent(plan) if phase_transition else verify_qualified_parent(plan))
        resume_plan_sha=branch['parent_plan_sha256']
    assert os.environ.get('PP_MEMORY_SCOPE_ACTIVE') == 'YES'
    assert opts.runtime.parent == Path('/dev/shm')
    assert os.environ['CUDA_VISIBLE_DEVICES'] == opts.gpu_uuid and opts.gpu_uuid in plan['allowed_gpu_uuids']
    assert jax.default_backend() == 'gpu' and len(jax.devices()) == 1
    assert bool(opts.resume_model) == bool(opts.resume_full)
    assert digest_file(Path(plan['single_parent'])) == plan['single_parent_sha256']
    maturity_receipt=verify_parent_maturity(plan)
    assert opts.update_limit is None or opts.update_limit>0
    assert opts.update_limit is not None or plan['continuous_experiment'] or opts.resume_model
    for name, sha in json.loads((A / 'CODE_MANIFEST.json').read_text()).items():
        assert digest_file(Path(name)) == sha, name
    for name, sha in json.loads(Path(PLAN['source_manifest']).read_text())['source_files'].items():
        assert digest_file(SOURCE / name) == sha, name
    assert digest_file(Path(plan['source_model'])) == plan['source_sha256']
    with jax.default_device(jax.devices('cpu')[0]):
        stack = load_stack(ROOT, SOURCE, 325)
        parent, true_step, stage, update_id = restore_training_state(stack, Path(plan['source_model']))
        assert (stage, update_id, int(parent.opt.t)) == (15, 14400, 963375)
        stack.cfg = configure(course_config(stack, stack.spec.course[14]))
        stack.ppo.apply_mlp=feature_apply_factory(apply_factory(stack.ppo.apply_mlp,enabled=True))
        learner, anchor = initialize(stack.ppo, parent, mode=opts.mode)
        with Path(plan['single_parent']).open('rb') as stream: selected_parent=pickle.load(stream)
        with Path(plan['continuation_parent']).open('rb') as stream:continuation_parent=pickle.load(stream)
        true_step=int(continuation_parent['step'])
        completed = 0
        if opts.resume_model:
            with opts.resume_model.open('rb') as stream:
                saved = pickle.load(stream)
            assert saved['schema'] == 'v325_single5_training_v1'
            assert saved['mode'] == opts.mode and saved['plan_sha256'] == resume_plan_sha
            assert saved['initial_single_actor_digest'] == numeric_digest(anchor['pi'])
            learner = stack.ppo.TrainState(saved['params'], saved['opt'])
            completed, true_step = saved['actual_updates'], saved['step']
        args = argparse.Namespace(**stack.parent['args'])
        vars(args).update(learner_settings_for_stack(stack))
        args.n_envs, args.n_steps, args.minibatch_size = 1024, 128, 16384
        args.seed = plan['seed']
        args.wind_training_mix = stack.wind_training_mix
    initial_learner_params=jax.tree.map(jnp.array,learner.params)
    anchor = jax.device_put(anchor)
    native_components = stack.ppo.policy_components
    direct_trace_calls = [0]
    if opts.mode == 'direct':
        def direct_components(params, obs):
            assert_single_actor(params)
            direct_trace_calls[0] += 1
            return native_components(params, obs)
        stack.ppo.policy_components = direct_components
    stack.single_actor_anchor_params = anchor
    stack.bootstrap_stage_anchor = anchor
    if plan['calm_trust_region']['enabled']:
        from checkpoint_numpy import NumpyCheckpoint
        anchor_path=Path(plan['calm_trust_region']['anchor_model'])
        assert digest_file(anchor_path)==plan['calm_trust_region']['anchor_model_sha256']
        with anchor_path.open('rb') as stream:guard_source=NumpyCheckpoint(stream).load()
        assert guard_source['actual_updates']==(branch['actual_updates'] if wind_adaptation else 32)
        stack.calm_guard_anchor_params=jax.device_put(guard_source['params'])
    install_pose_mode()
    native = stack.env_class(stack.xml, n_envs=1024, cfg=stack.cfg)
    env = XWindEnv(CaptureEnv(ActualContactEnv(native, balanced=False), enabled=False),half_angle_deg=plan['wind_domain']['x_axis_half_angle_deg'], enabled=True,
        calm_only=plan['training_phase']=='calm_foundation')
    env = Single5Env(env, reward_enabled=False, goal_reward_enabled=plan['reward_revision']['goal_reward_enabled'],random_onset_enabled=True,targeted_reward_enabled=plan['reward_revision'].get('targeted_reward_enabled',False),weak_release_floor_enabled=plan['reward_revision']['weak_calm_release_floor']['enabled'],weak_release_floor_weight=plan['reward_revision']['weak_calm_release_floor']['weight'])
    env=WindFeatureEnv(env)
    rehearsal=plan['rehearsal']
    env=wrap_rehearsal(env,enabled=rehearsal['enabled'],fraction=rehearsal['fraction'],seconds=rehearsal['seconds'],allow_empty=wind_adaptation)
    rehearsal_n=env.n_rehearsal
    assert rehearsal_n==rehearsal['n_envs']==(0 if wind_adaptation else 102)
    assert not env.reward_enabled and env.goal_reward_enabled==plan['reward_revision']['goal_reward_enabled']
    assert (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) == (120, 453, 7, 240000)
    assert env.cfg.wind_config.max_speed_m_s == 1.6 and env.cfg.wind_config.wind_y_nonpositive
    assert env.cfg.wind_config.startup_no_wind_hits == 5 and env.cfg.wind_config.startup_ramp_s == 1.
    initial_rng, key = jax.random.split(jax.random.PRNGKey(plan['seed']))
    keys=jax.random.split(key,1024)
    cached=plan['cached_initialization']['enabled']
    stagger_receipt={'skipped_on_full_state_resume':bool(opts.resume_full)}
    if opts.resume_full or cached:
        runner=runner_template(env,stack.ppo,keys,initial_rng)
        if not opts.resume_full:
            learner,runner,cache_metadata,stagger_receipt=restore_cached(plan,env,learner,runner)
            assert cache_metadata['true_step']==true_step
    else:
        state,obs=jax.jit(env.reset)(keys)
        runner=stack.ppo.RunnerState(state,obs,env.get_critic_obs(state,obs),initial_rng,
                                    jnp.zeros(1024),jnp.zeros(1024,jnp.int32))
    if not opts.resume_full:
        learner,runner=strong_arrays((learner,runner))
    release_caches()
    collect, update, settings = continuation_functions(stack, env, args)
    if not opts.resume_full and not cached:
        runner,stagger_receipt=warm_rehearsals(learner,env,runner,collect,
            plan['rehearsal_stagger'],numeric_digest)
    assert settings['feature_warmup_minibatches'] == plan['feature_optimizer']['warmup_accepted_minibatches']
    assert settings['feature_lr_multiplier'] == plan['feature_optimizer']['feature_row_learning_rate_multiplier']
    assert settings['counterfactual_supervision_coef'] == 0. and settings['residual_l2_coef'] == 0.
    assert settings.get('teacher_distill_coef',0.) == 0.
    assert not settings['apply_every_minibatch'] and settings['target_kl'] == .006
    fields = tuple(dict.fromkeys(QUALITY_FIELDS + HEIGHT_FIELDS + REFERENCE_FIELDS))
    progress = FullEpisode99Progress(1024-rehearsal_n, fields)
    startup_tracker=StartupProgress(seconds=rehearsal['seconds'])
    restored_hold=0
    progress.half_angle_deg=plan['wind_domain']['x_axis_half_angle_deg']
    console = LegacyWindConsole(1024, fields, 180.)
    contract = dict(schema='v325_single5_native_carry_v1', mode=opts.mode,
                    source_sha256=plan['source_sha256'], initial_single_actor_digest=numeric_digest(anchor['pi']),
                    actor_critic_action=[120, 453, 7],single_MLP_input=120, seed=plan['seed'], n_envs=1024,
                    n_steps=128, stage=15, plan_sha256=plan_sha,
                    source_actor='Retained288withTwoZeroInitializedCausalWindInputRows', pose_ramp_s=0., pose_limit_deg=5., revision=REVISION,
                    feature_optimizer=plan['feature_optimizer'],
                    learning_stability=plan['learning_stability'],
                    reward_maturity=maturity_receipt,
                    weak_calm_release_floor=plan['reward_revision'].get('weak_calm_release_floor',{}),
                    actor_training=plan['actor_training'],cached_initialization=plan['cached_initialization'],
                    rehearsal_stagger=plan['rehearsal_stagger'],actor_phase=plan['actor_phase'],
                    rehearsal=rehearsal,long_statistics_n_envs=1024-rehearsal_n,
                    full_episode_outcome_capacity=plan['full_episode_outcome_capacity'],calm_trust_region=plan['calm_trust_region'],onset=plan['onset'],wind_domain=plan['wind_domain'],timing=plan['timing'])
    if opts.resume_full:
        load_contract=(parent_contract if phase_transition else (contract if parent_contract is None else compatible_parent_contract(
            parent_contract,contract,resume_plan_sha,calm_guard=plan['calm_trust_region'])))
        before_restore_digest=numeric_digest(learner)
        load_runner=runner._replace(obs=runner.obs[:,:118]) if phase_transition else runner
        learner, runner, aux, meta = load_snapshot(opts.resume_full, learner, load_runner, load_contract)
        assert numeric_digest(learner)==before_restore_digest,'Model and full native learner disagree'
        if parent_contract is not None:
            assert meta['actual_updates']==branch['actual_updates']==completed
            qualified_receipt['native_full_learner_matches_model_exactly']=True
        if phase_transition:
            del runner, aux
            state,obs=jax.jit(env.reset)(keys)
            runner=stack.ppo.RunnerState(state,obs,env.get_critic_obs(state,obs),initial_rng,
                jnp.zeros(1024),jnp.zeros(1024,jnp.int32))
            learner,runner=strong_arrays((learner,runner))
            assert numeric_digest(learner)==before_restore_digest
            qualified_receipt['new_phase_native_reset_completed']=True
            learner=migrate_learner(learner)
            assert bool(frozen_backbone_equal(learner.params,initial_learner_params))
            qualified_receipt['old118_values_and_moments_kept_new2_rows_zero']=True
        else:
            progress, console = aux['progress'], aux['console']
            startup_tracker=aux['startup_tracker']
            restored_hold=int(aux['hold'])
        completed, true_step = meta['actual_updates'], meta['true_step']
    if opts.mode == 'direct':
        assert_single_actor(learner.params)
        assert learner.params['pi']['l1']['w'].shape==(120,282)
    opts.output.mkdir(exist_ok=False)
    save(opts.output / 'PREFLIGHT.json', dict(timestamp=datetime.now().astimezone().isoformat(),
        **contract, physical_cfg=asdict(stack.cfg), ppo=settings, gpu_uuid=opts.gpu_uuid,
        rehearsal_stagger_receipt=stagger_receipt,qualified_best_resume=qualified_receipt,
        xml_sha256=digest_file(stack.xml), initial_native_runner_digest=state_digest(runner),
        source_parent_unchanged_sha256=plan['source_sha256'],
        actor_parameter_keys=list(learner.params), actor_total_parameters=actor_parameter_count(learner.params['pi']),
        trainable_actor_parameters=plan['direct_trainable_actor_parameters'],
        source_weights_retained=True, optimizer_reset_for_new_observation_reward=False, newly_trainable_actor_local_Adam_t=int(learner.opt.actor_t),
        BC=0, one_time_initialization_migration=True, PPO_supervision=False, direct_policy_has_no_residual=opts.mode == 'direct',
        normal_path_promoted=False, final1200s99accepted=False))
    @jax.jit
    def entry_error(params, reference, observations):
        return jnp.max(jnp.abs(stack.ppo.policy_mean(params, observations) - stack.ppo.policy_mean(reference, observations)))
    if not opts.resume_model:
        assert numeric_digest(learner.params)==numeric_digest(initial_learner_params)
        error = float(entry_error(learner.params, initial_learner_params, runner.obs))
        assert error <= 2.e-6
        save(opts.output / 'INITIAL_POLICY_PARITY.json', dict(mean_max_abs=error, atol=2.e-6, actual_updates=0,reference='Owninitialactoranchor;notalosslessS15claim'))
    release_caches()
    diagnostics = make_critic_diagnostics(stack.ppo, settings)
    import wandb
    run = wandb.init(entity='fushi37', project='pingpong-mjx', id=opts.run_id, name=opts.run_id,
                     resume='never', mode='online', dir=os.environ['WANDB_DIR'],
                     settings=wandb.Settings(x_disable_stats=True),
                     config=dict(contract, BC=0, update_budget=opts.update_limit, direct_actor=opts.mode == 'direct'))
    stopped = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
    retention = CheckpointRetention(opts.output, cleanup_every=5)
    status = {}; score = (0., -1.e6, -1.e6, -1.e6); hold=restored_hold
    started = time.monotonic(); session = 0; committed = True; normal_completion = False
    gate = make_reference_gate(quality_gate(325, stack.spec.course[14], stack.helpers.quality_passed), 325)
    def checkpoint(kind):
        path = opts.output / f'{kind}_u{completed:08d}.pkl'
        if path.exists():
            return path
        if opts.mode == 'direct':
            assert_single_actor(learner.params)
        payload = dict(schema='v325_single5_training_v1', mode=opts.mode,
                       params=jax.device_get(learner.params), opt=jax.device_get(learner.opt),
                       initial_single_actor_digest=numeric_digest(anchor['pi']),
                       source_sha256=plan['source_sha256'], plan_sha256=plan_sha,
                       step=true_step, actual_updates=completed, stage=15,
                       initial_actor='CompleteS15migration' if opts.mode=='direct' else 'OriginalcompleteS15', BC=0, last_summary=status,
                       requires_actor_runtime=plan['observation']['runtime'],
                       native_actor_obs_dim=120,single_actor_input_dim=120,simulation_only=True,
                       normal_path_promoted=False, final1200s99accepted=False)
        with path.open('xb') as stream:
            pickle.dump(payload, stream)
        if kind == 'periodic':
            retention.register(path, score)
        return path
    def recovery():
        path = opts.output / f'trainer_state_u{completed:08d}.zip'
        if not path.exists():
            save_snapshot(path, learner, runner, dict(progress=progress, console=console,startup_tracker=startup_tracker,hold=hold), contract,
                          dict(actual_updates=completed, true_step=true_step, run_id=opts.run_id))
            if completed in (32,64) or completed in plan.get("phase_milestones",[]):
                import os
                best=opts.output/f'milestone_state_u{completed:08d}.zip'
                os.link(path,best)
                os.link(path.with_suffix('.zip.json'),best.with_suffix('.zip.json'))
            prune_owned_snapshots(opts.output, path)
    try:
        with (opts.output / 'updates.jsonl').open('x') as stream:
            event_stream=(opts.output/'failure_events.jsonl').open('x')
            while (opts.update_limit is None or completed < opts.update_limit) and not stopped[0]:
                committed = False
                input_runner = runner
                if session == 0:
                    proof = opts.output / 'snapshot_smoke.zip'
                    save_snapshot(proof, learner, runner, {}, contract, dict(actual_updates=completed, true_step=true_step))
                    clone, clone_runner, _, _ = load_snapshot(proof, learner, runner, contract)
                    input_digest = state_digest((learner, runner))
                    assert state_digest((clone, clone_runner)) == input_digest
                runner, transitions = collect(learner.params, runner)
                verify_mature_batch(transitions.metrics)
                if session == 0 and not opts.resume_model:
                    assert numeric_digest(learner.params)==numeric_digest(initial_learner_params)
                    error = float(entry_error(learner.params, initial_learner_params, transitions.obs[-1])); assert error <= 2.e-6
                    save(opts.output / 'FILLED_ROLLOUT_POLICY_PARITY.json', dict(mean_max_abs=error, atol=2.e-6, actual_updates=0,reference='Owninitialactoranchor;notalosslessS15claim'))
                if session == 0:
                    gc.collect(); ctypes.CDLL(None).malloc_trim(0)
                explained, variance = diagnostics(learner.params, runner, transitions)
                new, loss = update(learner, runner, transitions)
                if session == 0:
                    clone_runner, clone_transitions = collect(clone.params, clone_runner)
                    clone, clone_loss = update(clone, clone_runner, clone_transitions)
                    assert state_digest((new, runner, transitions, loss)) == state_digest((clone, clone_runner, clone_transitions, clone_loss))
                    save(opts.output / 'NATIVE_FULL_STATE_ROUNDTRIP.json', dict(complete_state_roundtrip_exact=True,
                        input_digest=input_digest, output_digest=state_digest((new, runner, transitions, loss)),
                        accepted_optimizer_updates=1, duplicate_branch_discarded=True))
                    del clone, clone_runner, clone_transitions, clone_loss

                if wind_adaptation:
                    assert bool(frozen_backbone_equal(new.params,initial_learner_params)), 'Frozen backbone changed'
                loss = {key: float(value) for key, value in jax.device_get(loss).items()}
                if plan['calm_trust_region']['enabled']:
                    assert loss['calm_trust/post_kl']<=plan['calm_trust_region']['maximum_initial_policy_kl']+1.e-7
                assert loss['learning_stability/sampled_max_step_kl']<=plan['learning_stability']['maximum_sampled_cohort_step_kl']+1.e-7
                assert all(np.isfinite(list(loss.values()))) and loss['counterfactual_supervision_mse'] == 0.
                assert loss['teacher_distill_mse'] == 0.
                if opts.mode=='direct':assert loss['residual_rms']==0. and loss['residual_abs_max']==0.
                minis = int(loss['ppo_minibatches_applied']); assert 0 <= minis <= 32
                assert int(new.opt.t) == int(learner.opt.t) + minis
                if opts.mode == 'direct':
                    assert_single_actor(new.params)
                else:
                    assert numeric_digest(new.params['teacher_pi']) == numeric_digest(anchor['teacher_pi'])
                metrics = jax.device_get(transitions.metrics)
                assert np.all(metrics['v321/finite'] == 1.) and not np.any(metrics['wind/positive_y_violation'])
                assert np.max(metrics['wind/speed_m_s']) <= 1.60001
                if plan['training_phase']=='calm_foundation':
                    assert np.all(metrics['wind/mode']==0.) and np.all(metrics['wind/speed_m_s']==0.)
                log_failure_events(event_stream,completed+1,metrics,jax.device_get(transitions.done))
                dones=jax.device_get(transitions.done)
                long_metrics,long_done=long_rollout(metrics,dones,rehearsal_n)
                progress.add(long_metrics,long_done)
                rehearsal_metrics={k:metrics[k][:,:rehearsal_n] for k in PROGRESS_FIELDS}
                startup_tracker.add(rehearsal_metrics,dones[:,:rehearsal_n])
                console.add(metrics,dones)
                old_optimizer_count = int(learner.opt.t)
                learner = new; session += 1; completed += 1; true_step += 131072; committed = True
                if session==1:
                    proof.unlink(missing_ok=True); proof.with_suffix('.zip.json').unlink(missing_ok=True)
                summaries = progress.summaries()
                selected_gate=foundation_gate99 if plan['training_phase']=='calm_foundation' else gate99
                final_gate = selected_gate(summaries, gate, minimum_n=100)
                startup_summary=startup_tracker.summary()
                if startup_summary['n']<rehearsal['minimum_episodes']:
                    final_gate['failures'].append('startup20:samples')
                elif startup_summary['survival']<rehearsal['minimum_success']:
                    final_gate['failures'].append('startup20:survival99')
                final_gate['passed']=not final_gate['failures']
                np.testing.assert_allclose(metrics['pose/outer_limit_deg'],5.,atol=1.e-5)
                hold=hold+1 if final_gate['passed'] else 0
                status = dict(update=completed, actual_update=completed, stage=15, mode=opts.mode,
                              true_step=true_step, actor_local_Adam_t=int(learner.opt.actor_t),
                              applied_minibatches_total=int(learner.opt.t), **loss,
                              **{'critic/explained_variance': float(explained), 'critic/return_target_variance': float(variance)},
                              groups=summaries, hold=hold, final_gate=final_gate, wall_seconds=time.monotonic()-started)
                for precision_field in PRECISION_FIELDS:
                    status[precision_field]=float(np.mean(metrics[precision_field]))
                for release_field in RELEASE_FLOOR_FIELDS:
                    status[release_field]=float(np.mean(metrics[release_field]))
                status['release_floor/weak_events_total']=int(np.sum(metrics['release_floor/weak_event']))
                floor_cost=metrics['reward/weak_calm_release_floor']
                assert np.isfinite(floor_cost).all() and np.all(floor_cost<=0.)
                assert np.all(floor_cost[metrics['wind/mode']>.5]==0.)
                status['feature_only_warmup'] = int(int(learner.opt.t) < 512)
                status['startup_rehearsal']=startup_summary
                cutoff=metrics['sampling/rehearsal_cutoff']>.5
                assert np.all(np.asarray(transitions.truncated)[cutoff])
                assert not np.any(np.asarray(transitions.terminated)[cutoff])
                assert np.isfinite(np.asarray(transitions.timeout_value)[cutoff]).all()
                status['sampling/rehearsal_censors']=int(cutoff.sum())
                status['sampling/long_n_envs']=1024-rehearsal_n
                status['frozen_backbone_verified']=int(wind_adaptation)
                status['actor_wind_rows_only'] = int(plan['feature_optimizer']['permanent_wind_rows_only'])
                status['actor_clock_restarted_this_update'] = int(old_optimizer_count <= 512 < int(learner.opt.t))
                stability=plan['learning_stability']
                status['learning_stability/critic_only_warmup']=int(old_optimizer_count<stability['critic_warmup_start_t']+stability['critic_only_minibatches'])
                for population,mask in [('clean',metrics['wind/mode']<.5),('wind',metrics['wind/mode']>.5)]:
                    live=mask&(metrics['terminated']<.5)
                    if np.any(live):
                        status[f'reward_audit/{population}/live_total_per_s']=float(np.mean(metrics['reward/total'][live])/.005)
                        status[f'reward_audit/{population}/kinetic_brake_per_s']=float(np.mean(metrics['reward/targeted_brake'][live])/.005)
                        if status['learning_stability/critic_only_warmup']:
                            assert status[f'reward_audit/{population}/kinetic_brake_per_s']>=-5.,'Kinetic brake excessively penalizes the unchanged parent'
                stream.write(json.dumps(status)+'\n'); stream.flush()
                flat = {key: value for key, value in status.items() if isinstance(value, (int, float))}
                for key,value in startup_summary.items():
                    if isinstance(value,(int,float)):
                        flat['sampling/startup20/'+key]=value
                for population, summary in summaries.items():
                    for horizon in ('short30', 'stage', 'full1200'):
                        available=summary[horizon].get('n',0)>0
                        flat[f'population/{population}/{horizon}/available']=int(available)
                        for key, value in summary[horizon].items():
                            if isinstance(value, (int, float)) and (available or key!='survival'):
                                flat[f'population/{population}/{horizon}/{key}'] = value
                for key in ('reward/single5_outgoing_replace','reward/single5_apex_precision','reward/single5_center'):
                    flat[key]=float(np.mean(metrics[key]))
                for field in TARGETED_FIELDS:flat[field]=float(np.mean(metrics[field]))
                flat['reward/single5_outgoing_applied']=float(np.mean(metrics['reward/goal_outgoing'] if env.goal_reward_enabled else metrics['reward/dynamics_outgoing']))
                for field in ('alive','hit','outgoing','normal','height_margin','braking','apex','failure'):
                    flat['reward/goal_'+field]=float(np.mean(metrics['reward/goal_'+field]))
                for field in ('observed_force/valid','observed_force/age_s','observed_force/error_sq_m2_s2'):
                    flat[field]=float(np.mean(metrics[field]))
                for population,summary in summaries.items():
                    for field in ('center_rms_m','apex_error_rms_m','in_band','adjacent_jump_m','racket_vxy_excess_rms_m_s'):
                        if field in summary.get('quality',{}):
                            flat[f'population/{population}/quality/{field}']=summary['quality'][field]
                run.log(flat, step=completed)
                stream.flush();event_stream.flush()
                print(console.line(completed, hold, final_gate).replace('Adam+32',f'Adam+{minis}')+f' architecture={opts.mode} actorAdam={int(learner.opt.actor_t)}', flush=True)
                scores=[row['short30'].get('survival',0.) for row in summaries.values() if row['short30'].get('n',0)>=4]
                score=(float(final_gate['passed']),min(scores,default=0.),
                       float(summaries['clean']['short30'].get('survival',0.)),
                       -float(summaries['clean']['quality'].get('apex_error_rms_m',1.)))
                if completed % 8 == 0: checkpoint('periodic')
                if completed in (32,64) or completed in plan.get("phase_milestones",[]): checkpoint('milestone')
                if completed % 32 == 0: recovery()
                gc.collect(); ctypes.CDLL(None).malloc_trim(0)
                if hold>=16:
                    readiness_name='READY_FOR_CALM_FIXED_POLICY_1200S.json' if plan['training_phase']=='calm_foundation' else 'READY_FOR_FIXED_POLICY_1200S.json'
                    save(opts.output/readiness_name,dict(update=completed,gate=final_gate,
                        training_readiness_only=True,final1200s99accepted=False))
                    break
        checkpoint('interrupted'  if stopped[0] else 'bounded_final'); recovery()
        if opts.mode == 'direct':
            assert direct_trace_calls[0] > 0
        save(opts.output / 'POLICY_STRUCTURE_PROOF.json', dict(mode=opts.mode,
             parameter_keys=list(learner.params), direct_trace_calls_rejecting_all_residual_params=direct_trace_calls[0],
             direct_actor_MLP_count=1 if opts.mode == 'direct' else None,
             actor_total_parameters=actor_parameter_count(learner.params['pi']),
             trainable_actor_parameters=plan['direct_trainable_actor_parameters'],
             no_embedded_source_or_residual_in_direct_checkpoint=opts.mode == 'direct', actual_updates=completed))
        run.summary['actual_updates'] = completed; run.summary['normal_path_promoted'] = False
        run.summary['final1200s99accepted'] = False; normal_completion = True
    finally:
        if not committed and not normal_completion and session==0 and (opts.output/'snapshot_smoke.zip').exists():
            save(opts.output/'PRECOMMIT_FAILURE_RECOVERY.json',dict(optimizer_update_committed=False,fixture='snapshot_smoke.zip',input_state_preserved=True))
        if committed and not normal_completion:
            checkpoint('interrupted' if stopped[0] else 'last'); recovery()
        run.finish()
    print('[V325 actual single actor] completed', opts.mode, completed, flush=True)


if __name__ == '__main__':
    main()
