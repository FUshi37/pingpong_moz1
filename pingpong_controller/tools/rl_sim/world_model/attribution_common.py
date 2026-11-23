"""Shared immutable population and budgets for frozen pilot attribution only."""

import json
from pathlib import Path
import subprocess

import jax
import jax.numpy as jnp
import numpy as np

from .contracts import authenticate, config_hash, file_hash, repository_root
from .frozen_seed import load_warmed_seed
from .frozen_task import init_evaluation
from .migration import expand_actor_history, load_v321_actor
from .native_observation import NativeObservationAdapter
from .native_stack import build_native_environment
from .physical_events import init_physical_events
from .pilot_checkpoint import load_pilot_actor
from .resources import StorageBudget
from .teacher_initialization import begin_ball_trial
from .world_control import ControlContract


def apex_replay_deltas(current, previous):
    """Match the physical impact, not the goal-arrival counter (often zero)."""
    prior={(r['case_id'],r['impact_id']):r['apex_position_m']
           for r in previous if r['valid_apex']}
    return [float(np.max(np.abs(np.asarray(r['apex_position_m'])-prior[key])))
            for r in current if r['valid_apex']
            and (key:=(r['case_id'],r['impact_id'])) in prior]


def plan_and_budget(cfg, guard):
    plan = json.loads(authenticate(repository_root(), cfg['frozen_attribution_diagnostic']['plan']).read_text())
    if (not plan['authorized'] or plan['parameter_updates'] or plan['reward_changes']
            or plan['constraint_changes'] or plan['new_scenarios']
            or plan['max_new_output_bytes'] != 8*2**20):
        raise ValueError('Frozen attribution authorization changed')
    guard.budget = StorageBudget(guard.budget.root,
        min(guard.budget.max_bytes, plan['phase_start_used_bytes'] + plan['max_new_output_bytes']),
        guard.budget.min_free_bytes)
    guard.budget.check(additional_bytes=2**20)
    return plan


def worker_check(guard):
    guard.check()
    # Leave room for asynchronously written scope/resource logs and final JSON.
    guard.budget.check(additional_bytes=256*1024)


def source_report(cfg, args):
    sources = sorted(Path(__file__).parent.glob('*.py'))
    return dict(schema='wmj_v1_frozen_attribution', status='INCOMPLETE', training_version='WMJ-V1',
        config_sha256=config_hash(cfg), gpu_uuid=args.gpu_uuid, stage='S01', wandb=None,
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={p.name: file_hash(p) for p in sources},
        actor_updates=0, critic_updates=0, world_updates=0, promoted=False,
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})


def population(cfg):
    """Restore the exact fixed128 warmed population and stationary release."""
    refs = cfg['frozen_evaluation_artifacts']
    manifest = json.loads(authenticate(repository_root(), refs['cases']).read_text())
    cases = manifest['stages']['S01']
    seed = cfg['frozen_evaluation']['seed']
    bundle = build_native_environment(cfg, n_envs=128, stage_index=0)
    env = bundle.environment
    fresh, _ = jax.jit(env.reset)(jax.random.split(jax.random.PRNGKey(seed), 128))
    context = dict(schema='wmj_fixed_warmed_population', cases_sha256=manifest['cases_sha256'],
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        domain_sha256=config_hash(cfg['domain']), ids=list(range(128)), reset_seed=seed,
        warmup_controls=128, native_source_sha256=bundle.stack.spec.native_env_sha256,
        initialization_sha256=refs['static_initialization']['sha256'], warmup_ar1_enabled=False)
    warmed, rows = load_warmed_seed(refs['warmed_seed'], fresh, context)
    known = np.asarray([r['status']=='ADMITTED' for r in rows])
    goals = jnp.asarray([c['goals_xyz_m'] for c in cases], jnp.float32)
    initial = jax.jit(lambda f,w,p: begin_ball_trial(env,f,w,p,jnp.zeros((128,3)),
        ar1_enabled=False))(fresh,warmed,goals[:,0])
    adapter = NativeObservationAdapter(env, initial, full_chain=True)
    history, _ = adapter.initialize(initial)
    task = init_evaluation(goals, jax.random.PRNGKey(seed+1), jnp.arange(128))
    task = task._replace(done=~jnp.asarray(known))
    return dict(bundle=bundle, env=env, initial=initial, adapter=adapter, history=history,
        task=task, events=init_physical_events(128), goals=goals, cases=cases, known=known,
        control=ControlContract.from_native(env), context=context)


def actors(cfg):
    baseline = expand_actor_history(load_v321_actor(repository_root(), cfg['lineage']['actor_source']))
    result = {'baseline': baseline}
    for step in (10,20):
        result[f'u{step}'] = load_pilot_actor(cfg, cfg['bounded_imagination_pilot']['actor_exports'][str(step)])[0]
    return result


def initial_interface(p,lanes):
    """Compare the actual model inputs separately from native solver scratch."""
    from .world_control import control_from_native
    from .native_observation import camera_from_native
    from .state_adapter import numeric_tree_digest
    native=p['initial'];env=p['env']
    return dict(control_sha256=numeric_tree_digest(control_from_native(native)),
        camera_history_sha256=numeric_tree_digest((camera_from_native(native),p['history'])),
        qpos=np.asarray(native.data.qpos)[lanes].tolist(),qvel=np.asarray(native.data.qvel)[lanes].tolist(),
        racket_position_m=np.asarray(native.data.site_xpos)[lanes,env.racket_site_id].tolist(),
        racket_rotation=np.asarray(native.data.site_xmat)[lanes,env.racket_site_id].tolist())


def frozen_models(cfg, budget):
    from .support_data import SupportReplayReader
    from .normal_checkpoint import load_components
    from .auxiliary_refresh import restore_auxiliary
    root = repository_root()/cfg['resources']['output_root']
    reader = SupportReplayReader(root/'support_replay_full_a01', budget, max_bytes=0)
    params, options, refs, _, _ = load_components(cfg, reader,
        robot_checkpoint=root/'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root/'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root/'solver_aux_fit_a01/auxiliary_u004096.json')
    ref = cfg['bounded_imagination_pilot']['candidate_checkpoint']
    restored, metadata = restore_auxiliary(cfg, ref)
    return params['robot'], dict(parameters=restored['parameters'], scale=restored['scale']), options, dict(refs, auxiliary=ref)
