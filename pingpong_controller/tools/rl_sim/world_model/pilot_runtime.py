"""Explicit candidate-only runtime; preserved native seeds create a fresh pool."""

import json

import jax
import jax.numpy as jnp
import numpy as np

from .actor_critic import init_critic
from .audit_full_prior_world import AuditReader
from .auxiliary_refresh import restore_auxiliary
from .branch_bank import BranchSeedBank
from .build_training_seed_bank import load_training_bank
from .contracts import authenticate, config_hash, repository_root
from .course_env import course_critic_layout
from .course_goals import build_goal_table
from .course_learner import prepare_course_session, validate_course_config
from .imagined_env import EpisodeContract
from .imagination_learner import init_learner
from .migration import expand_actor_history, load_v321_actor
from .native_stack import build_native_environment
from .normal_checkpoint import load_components
from .persistent_pool import init_pool
from .train_imagination import validate_no_update_contract
from .world_control import ControlContract


def validate_pilot(cfg):
    """Keep the new training authorization separate from historical default-off CLIs."""
    settings = cfg['bounded_imagination_pilot']
    expected = dict(authorized=True, stage='S01', maximum_actor_updates=20,
        critic_updates_per_actor=4, evaluate_every_actor_updates=10,
        logical_envs=1024, persistent_envs=768, branch_envs=256, horizon=64,
        actor_lr=1e-5, critic_lr=1e-4, world_model_frozen=True,
        contact_residual_enabled=False, default_model_replacement=False,
        automatic_stage_advance=False, source_merge=False)
    if any(settings.get(key) != value for key, value in expected.items()):
        raise ValueError('Pilot exceeds its bounded user authorization')
    if settings['candidate_checkpoint'] != cfg['auxiliary_refresh']['candidate_checkpoint']:
        raise ValueError('Pilot candidate checkpoint changed')
    if settings['training_seed_bank'] != cfg['training_seed_bank']['bank_reference']:
        raise ValueError('Unregistered training sources')
    validate_course_config(cfg)
    return validate_no_update_contract(cfg)


def require_action_gate(cfg):
    reference = cfg['bounded_imagination_pilot']['action_check']
    report = json.loads(authenticate(repository_root(), reference).read_text())
    if (report['status'] != 'CANDIDATE_ACTION_RECHECK_COMPLETE'
            or not report['pilot_gate']['passed']
            or report['candidate_checkpoint'] != cfg['bounded_imagination_pilot']['candidate_checkpoint']):
        raise ValueError('Candidate action-effect admission did not pass')
    return reference


def setup_pilot(cfg, guard):
    """Use authenticated native state only; discard every old imagined pool state."""
    reward = validate_pilot(cfg)
    root = repository_root() / cfg['resources']['output_root']
    reader = AuditReader(root / 'support_replay_full_a01', guard.budget, max_bytes=0)
    params, physics, refs, parent, parent_ref = load_components(cfg, reader,
        robot_checkpoint=root / 'support_robot_fit_a02/robot_support_u006144.json',
        contact_checkpoint=root / 'contact_activation_control_a01/contact_coupled_u014336.json',
        auxiliary_checkpoint=root / 'solver_aux_fit_a01/auxiliary_u004096.json')
    native = build_native_environment(cfg, n_envs=1)
    control = ControlContract.from_native(native.environment)
    original, metadata = load_training_bank(cfg, cfg['training_seed_bank']['bank_reference'],
        reader=reader, options=physics, control_contract=control,
        model_references=dict(refs, auxiliary=parent_ref))
    settings = cfg['bounded_imagination_pilot']
    restored, aux_metadata = restore_auxiliary(cfg, settings['candidate_checkpoint'])
    auxiliary = dict(parameters=restored['parameters'], scale=restored['scale'])
    refs = dict(refs, auxiliary=dict(settings['candidate_checkpoint'],
        model_signature=aux_metadata['model_signature'], optimizer_updates=aux_metadata['updates']))
    bank = BranchSeedBank(model_identity=config_hash(refs), stage_id='S01')
    for index, row in enumerate(original.rows):
        if row['origin'] != 'teacher_full_state_only' or row['split'] != 'train':
            raise ValueError('A model refresh cannot reuse old imagined branch states')
        state, keys = original.gather(np.array([index]))
        if np.any(np.asarray(state.episode_steps)) or np.any(np.asarray(state.task.hit_count)):
            raise ValueError('Expected actual native reset states with fresh task counters')
        bank.add_teacher_precontact(state, keys, split='train', source_id=row['source_id'],
            time_to_contact_s=row['time_to_contact_s'])
    seed = settings['seed']
    # Whole-bank initial permutation gives each native source one logical lane.
    order = np.random.default_rng(seed).permutation(len(bank.rows))
    state, _ = bank.gather(order)
    if len(order) != 1024:
        raise ValueError('The pilot requires the complete existing1024-source bank')
    goals = np.unique(np.asarray(state.task.goal_xyz), axis=0)
    table = build_goal_table(goals, static_admitted=np.ones(len(goals), bool))
    layout = course_critic_layout(state)
    actor = expand_actor_history(load_v321_actor(repository_root(), cfg['lineage']['actor_source']))
    critic = init_critic(jax.random.PRNGKey(seed + 1), layout['dimension'])
    keys = jax.random.split(jax.random.PRNGKey(seed + 2), 1024)
    session = init_learner(actor, critic, init_pool(state, keys), seed=seed + 3)
    options = dict(physics, control_contract=control,
        episode_contract=EpisodeContract.from_config(cfg), reward_contract=reward,
        stage_id='S01', exploration_std=cfg['imagination']['exploration_std'])
    evidence = dict(model_references=refs, training_bank=settings['training_seed_bank'],
        geometry_sha256=config_hash(reader.report['geometry']), critic_layout=layout,
        initial_order=order.tolist(), source_rows=metadata['source_rows'],
        old_imagined_states_reused=0, online_geometry_queries=0,
        teacher_transitions_used_for_learning=0, reinitialized_pool=True)
    return session, bank, table, params['robot'], auxiliary, options, evidence


def prepare_pilot_session(session, bank):
    """Reset failed persistent lanes from training sources; preserve saved RNG."""
    next_key, draw = jax.random.split(session.sampling_rng)
    indices = np.asarray(jax.random.randint(draw, (768,), 0, len(bank.rows)))
    fresh, _ = bank.gather(indices)
    session, report = prepare_course_session(session._replace(sampling_rng=next_key),
        fresh, bank, minimum_precontact_lead_s=.080)
    report['persistent_candidate_indices'] = indices.tolist()
    report['persistent_candidate_probability'] = 1. / len(bank.rows)
    return session, report
