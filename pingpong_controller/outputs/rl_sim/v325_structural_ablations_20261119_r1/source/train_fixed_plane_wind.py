#!/usr/bin/env python3
"""Private V325/V326 wind continuation; manual launch only, native execution."""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, replace
from functools import partial
import importlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from fixed_plane_v323_ball_obs_noise import canonical, digest_file
from fixed_plane_wind_calm_release import MODE as CALM_RELEASE_MIX
from fixed_plane_wind_center_tail import MODE as CENTER_TAIL_MIX
from fixed_plane_wind_early_recovery import MODE as EARLY_RECOVERY_MIX
from fixed_plane_wind_resume_precision_clock import MODE as MATURE_PRECISION_MIX
from fixed_plane_wind_v2_group_balance import MODE as V2_GROUP_BALANCE_MIX
from fixed_plane_wind_calm_return import MODE as CALM_RETURN_MIX
from fixed_plane_wind_impact_normal_precision import MODE as IMPACT_NORMAL_MIX
from fixed_plane_wind_cubic_recovery import MODE as CUBIC_RECOVERY_MIX
from fixed_plane_wind_clean_joint_quality import MODE as CLEAN_JOINT_QUALITY_MIX
from fixed_plane_v325_wind import (
    append_wind_critic, make_config, make_profile_config, spec_for,
    STARTUP_NO_WIND_HITS, STARTUP_RAMP_S,
)
from train_fixed_plane_ball_obs_noise import (
    SOURCE_FILES as PARENT_SOURCES, learner_settings, numeric_digest,
    load_stack as load_parent_stack, make_train_functions as make_parent_functions,
)

SOURCE_FILES = (*PARENT_SOURCES, 'fixed_plane_v325_wind.py', 'fixed_plane_v326_wind.py',
    'fixed_plane_wind_env.py', 'fixed_plane_wind_validation.py', 'train_fixed_plane_wind.py',
    'train_fixed_plane_v325.py', 'train_fixed_plane_v326.py',
    'fixed_plane_v327_wind_reward.py', 'train_fixed_plane_v327.py',
    'fixed_plane_wind_fresh_validation.py', 'fixed_plane_wind_training_gates.py',
    'fixed_plane_wind_reward.py', 'fixed_plane_wind_pose.py', 'fixed_plane_wind_objectives.py',
    'fixed_plane_wind_strong.py', 'fixed_plane_wind_history.py', 'fixed_plane_wind_progression.py',
    'fixed_plane_wind_precision.py', 'fixed_plane_wind_actor.py',
    'fixed_plane_wind_bootstrap.py', 'fixed_plane_wind_bootstrap_restore.py',
    'fixed_plane_wind_online_recipe.py', 'fixed_plane_wind_bounded_recovery_reward.py',
    'fixed_plane_wind_history_optimizer.py', 'fixed_plane_wind_clean_rehearsal.py',
    'fixed_plane_wind_dynamics_guidance.py', 'probe_fixed_plane_wind_guidance.py',
    'fixed_plane_wind_flight_outcomes.py', 'fixed_plane_wind_tail_precision.py',
    'fixed_plane_wind_recovery_safety.py', 'fixed_plane_wind_lower_guard.py',
    'fixed_plane_wind_optimizer_conditioning.py', 'fixed_plane_wind_survival_precision.py', 'fixed_plane_wind_direction_domain.py',
    'fixed_plane_wind_gate_precision.py', 'fixed_plane_wind_reference_constraints.py',
    'fixed_plane_wind_center_precision_reward.py', 'fixed_plane_wind_clean_group_priority.py',
    'fixed_plane_wind_normalized_priority.py', 'fixed_plane_wind_precontact_center.py', 'fixed_plane_wind_balanced_precontact.py', 'fixed_plane_wind_release_precision.py', 'fixed_plane_wind_cycle_release.py',
    'fixed_plane_wind_measured_precision.py', 'fixed_plane_wind_steady_braking.py',
    'fixed_plane_wind_confirmed_launch_braking.py', 'fixed_plane_wind_center_tail.py',
    'fixed_plane_wind_calm_release.py', 'fixed_plane_wind_early_recovery.py',
    'fixed_plane_wind_resume_precision_clock.py', 'fixed_plane_wind_v2_group_balance.py',
    'fixed_plane_wind_calm_return.py', 'fixed_plane_wind_impact_normal_precision.py',
    'fixed_plane_wind_cubic_recovery.py', 'fixed_plane_wind_clean_joint_quality.py')


def registered_spec(version):
    if version == 327:
        from fixed_plane_v327_wind_reward import SPEC
        return SPEC
    return spec_for(version)


def base_version(spec):
    return getattr(spec, 'base_version', spec.version)


def metric_prefix(spec):
    return f'v{base_version(spec)-4}'


def course_config(stack, lesson, *, enabled=True):
    from fixed_plane_wind_clean_rehearsal import effective_lesson
    lesson = effective_lesson(lesson, getattr(stack, 'wind_training_mix', 'legacy'))
    cfg = make_profile_config(stack.native_cfg, lesson, stack.spec.version, enabled=enabled)
    from fixed_plane_wind_bounded_recovery_reward import with_reward_config
    recipe = getattr(stack, 'bootstrap_online_recipe', None) or {}
    cfg = with_reward_config(cfg, bool(recipe.get('reward')),
                             aggregation=recipe.get('quality_aggregation', 'arithmetic'),
                             score_mode=recipe.get('quality_score_mode', 'gaussian'))
    from fixed_plane_wind_clean_rehearsal import DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX
    from fixed_plane_wind_dynamics_guidance import with_guidance_config
    mix = getattr(stack, 'wind_training_mix', 'legacy')
    from fixed_plane_wind_clean_rehearsal import GATE_PRECISION_MIX, POSITIVE_PRECISION_MIX, REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX, CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX, PRECONTACT_CENTER_MIX, BALANCED_PRECONTACT_MIX, RELEASE_PRECISION_MIX, CYCLE_RELEASE_MIX, MEASURED_PRECISION_MIX, STEADY_BRAKING_MIX, CONFIRMED_LAUNCH_BRAKING_MIX
    clean_apex_enabled = mix in (CENTER_APEX_MIX, CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX, PRECONTACT_CENTER_MIX, BALANCED_PRECONTACT_MIX, RELEASE_PRECISION_MIX, CYCLE_RELEASE_MIX, MEASURED_PRECISION_MIX, STEADY_BRAKING_MIX, CONFIRMED_LAUNCH_BRAKING_MIX, CENTER_TAIL_MIX, CALM_RELEASE_MIX,
                                 EARLY_RECOVERY_MIX, MATURE_PRECISION_MIX, V2_GROUP_BALANCE_MIX, CALM_RETURN_MIX, IMPACT_NORMAL_MIX, CUBIC_RECOVERY_MIX, CLEAN_JOINT_QUALITY_MIX)
    center_reward_enabled = mix == CENTER_PRECISION_MIX or clean_apex_enabled
    clean_apex_priority = mix in (CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX)
    clean_group_priority = mix == CLEAN_GROUP_PRIORITY_MIX
    normalized_group_priority = mix in (NORMALIZED_GROUP_PRIORITY_MIX, V2_GROUP_BALANCE_MIX)
    precontact_center = mix in (PRECONTACT_CENTER_MIX,BALANCED_PRECONTACT_MIX,RELEASE_PRECISION_MIX,CYCLE_RELEASE_MIX)
    balanced_precontact = mix in (BALANCED_PRECONTACT_MIX,RELEASE_PRECISION_MIX,CYCLE_RELEASE_MIX)
    release_precision = mix == RELEASE_PRECISION_MIX
    cycle_release = mix == CYCLE_RELEASE_MIX
    measured_precision = mix == MEASURED_PRECISION_MIX
    steady_braking = mix == STEADY_BRAKING_MIX
    confirmed_launch_braking = mix == CONFIRMED_LAUNCH_BRAKING_MIX
    center_tail = mix == CENTER_TAIL_MIX
    calm_release = mix == CALM_RELEASE_MIX
    early_recovery = mix == EARLY_RECOVERY_MIX
    mature_precision = mix == MATURE_PRECISION_MIX
    calm_return = mix == CALM_RETURN_MIX
    impact_normal = mix == IMPACT_NORMAL_MIX
    cubic_recovery = mix == CUBIC_RECOVERY_MIX
    clean_joint_quality = mix == CLEAN_JOINT_QUALITY_MIX
    mix = REFERENCE_CONSTRAINT_MIX if center_reward_enabled else mix
    selected_mix = mix
    mix = NEGATIVE_Y_MIX if mix in (GATE_PRECISION_MIX,POSITIVE_PRECISION_MIX,REFERENCE_CONSTRAINT_MIX) else mix
    cfg = with_guidance_config(cfg, mix in (DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX)
                                and getattr(lesson, 'strong_wind_reward', False),
                                quality_guard=mix in (QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
                                safety_guard=mix in (SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX))
    from fixed_plane_wind_flight_outcomes import with_flight_config
    cfg = with_flight_config(cfg, mix in (FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
        tail_precision=mix in (TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
        recovery_safety=mix in (RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX), full_flight=mix in (FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
        lower_guard=mix in (LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX))
    from fixed_plane_wind_survival_precision import with_survival_precision_config
    cfg=with_survival_precision_config(cfg, mix in (SURVIVAL_MIX,NEGATIVE_Y_MIX)
        and getattr(cfg, 'wind_dynamics_guidance', False))
    if mix == NEGATIVE_Y_MIX:
        from fixed_plane_wind_direction_domain import with_negative_y_config
        cfg=replace(cfg,wind_config=with_negative_y_config(cfg.wind_config))
    from fixed_plane_wind_gate_precision import with_precision_config
    cfg = with_precision_config(cfg, selected_mix in (GATE_PRECISION_MIX,POSITIVE_PRECISION_MIX,REFERENCE_CONSTRAINT_MIX)
        and getattr(cfg, 'wind_dynamics_guidance', False), stack.spec.version,
        positive=selected_mix in (POSITIVE_PRECISION_MIX,REFERENCE_CONSTRAINT_MIX))
    from fixed_plane_wind_reference_constraints import with_reference_config
    cfg = with_reference_config(cfg,selected_mix==REFERENCE_CONSTRAINT_MIX,stack.spec.version)
    from fixed_plane_wind_center_precision_reward import with_center_reward_config, with_clean_apex_config, with_clean_apex_priority_config
    cfg = with_center_reward_config(cfg,center_reward_enabled,stack.spec.version)
    cfg = with_clean_apex_config(cfg,clean_apex_enabled)
    cfg = with_clean_apex_priority_config(cfg,clean_apex_priority)
    from fixed_plane_wind_clean_group_priority import with_group_priority_config
    cfg = with_group_priority_config(cfg,clean_group_priority)
    from fixed_plane_wind_normalized_priority import with_normalized_config
    cfg = with_normalized_config(cfg,normalized_group_priority)
    from fixed_plane_wind_precontact_center import with_precontact_config
    cfg = with_precontact_config(cfg,precontact_center)
    from fixed_plane_wind_balanced_precontact import with_balance_config
    cfg = with_balance_config(cfg,balanced_precontact)
    from fixed_plane_wind_release_precision import with_release_config
    cfg = with_release_config(cfg,release_precision)
    from fixed_plane_wind_cycle_release import with_cycle_config
    cfg = with_cycle_config(cfg,cycle_release)
    from fixed_plane_wind_measured_precision import with_measured_precision_config
    from fixed_plane_wind_steady_braking import with_steady_braking_config
    cfg = with_measured_precision_config(cfg, measured_precision)
    from fixed_plane_wind_confirmed_launch_braking import with_confirmed_launch_braking_config
    cfg = with_steady_braking_config(cfg, steady_braking)
    cfg = with_confirmed_launch_braking_config(cfg, confirmed_launch_braking)
    from fixed_plane_wind_center_tail import with_center_tail_config
    cfg = with_center_tail_config(cfg, center_tail)
    from fixed_plane_wind_calm_release import with_calm_release_config
    from fixed_plane_wind_early_recovery import with_early_recovery_config
    from fixed_plane_wind_resume_precision_clock import with_mature_precision_config
    cfg = with_calm_release_config(cfg, calm_release)
    cfg = with_early_recovery_config(cfg, early_recovery)
    cfg = with_mature_precision_config(cfg, mature_precision)
    from fixed_plane_wind_calm_return import with_calm_return_config
    from fixed_plane_wind_impact_normal_precision import with_impact_normal_config
    cfg = with_calm_return_config(cfg, calm_return)
    cfg = with_impact_normal_config(cfg, impact_normal)
    from fixed_plane_wind_cubic_recovery import with_cubic_recovery_config
    cfg = with_cubic_recovery_config(cfg, cubic_recovery)
    from fixed_plane_wind_clean_joint_quality import with_joint_quality_config
    return with_joint_quality_config(cfg, clean_joint_quality)


def save_wind_checkpoint(stack: SimpleNamespace, path: Path, train_state: Any,
                         args: argparse.Namespace, env: Any, step: int, *,
                         extra: dict[str, Any]) -> None:
    """Serialize nested dynamic configuration classes as independent values."""
    stack.ppo.save_checkpoint(path, train_state, args, env, step,
                              extra={**extra, 'env_cfg': asdict(env.cfg)})


def strong_arrays(tree: Any) -> Any:
    """Keep values/dtypes while avoiding a weak-scalar second rollout compile."""
    import jax
    import jax.numpy as jnp
    return jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.asarray(x).dtype), tree)


def load_stack(repo: Path, source: Path, version: int):
    """Reuse authentication; select the native class without sensing changes."""
    spec = registered_spec(version)
    original_version = base_version(spec)
    stack = load_parent_stack(repo, source, original_version - 2)
    native = importlib.import_module('mjx_juggle_env')
    from fixed_plane_wind_env import make_wind_base
    base = make_wind_base(native.MjxJuggleEnv, spec)
    stack.env_class = (stack.module.environment_class(base) if original_version == 325 else
        stack.module.environment_class(base, reward_preset='late_center_precision',
                                      grouped_conditions=True))
    if version in (325, 326):
        from fixed_plane_wind_reward import reward_environment
        stack.env_class = reward_environment(stack.env_class)
        from fixed_plane_wind_history import history_environment
        from fixed_plane_wind_flight_outcomes import flight_environment
        stack.env_class = flight_environment(stack.env_class)
        from fixed_plane_wind_reference_constraints import reference_environment
        stack.env_class = reference_environment(stack.env_class)
        stack.env_class = history_environment(stack.env_class)
    stack.spec = spec
    stack.default_spec = spec
    stack.repo_root = repo
    if version == 327:
        from fixed_plane_v327_wind_reward import reward_config
        stack.legacy_native_cfg = stack.native_cfg
        stack.native_cfg = reward_config(stack.native_cfg)
    stack.cfg = course_config(stack, spec.course[0])
    return stack


def restore_training_state(stack, resume: Path | None):
    """Keep all parent leaves; reject cross-course or changed-domain recovery."""
    import jax
    import jax.numpy as jnp
    from validate_juggle_mjx_ppo import load_checkpoint
    spec = stack.spec
    if spec.version == 327:
        from fixed_plane_v327_wind_reward import validate_source
        if resume is None:
            raise ValueError('V327 is continuation-only; provide V325/u506 or V327')
        source_payload = load_checkpoint(resume)
        if source_payload.get('args', {}).get('curriculum_profile') == spec_for(325).profile:
            validate_source(digest_file(resume), source_payload)
            legacy = SimpleNamespace(**{**vars(stack), 'spec': spec_for(325),
                                        'default_spec': spec_for(325),
                                        'native_cfg': stack.legacy_native_cfg})
            state, step, stage, updates = restore_training_state(legacy, resume)
            stack.cfg = course_config(stack, spec.course[stage])
            return state, step, stage, updates
    payload = stack.parent if resume is None else load_checkpoint(resume)
    from fixed_plane_wind_bootstrap_restore import normalize_bootstrap_checkpoint, learner_settings_for_stack
    payload = normalize_bootstrap_checkpoint(stack, payload, resume)
    spec = stack.spec
    from fixed_plane_wind_clean_rehearsal import checkpoint_mix
    stack.wind_training_mix = checkpoint_mix(payload, spec.version)
    stage, updates = 0, 0
    if resume is not None:
        contract = payload.get('wind_contract', {})
        stage, updates = payload.get('stage_index'), payload.get('global_update')
        config_ok = False
        legacy_revision = False
        pose_migration = False
        fast_migration = False
        expected_lesson = None
        course_ok = canonical(contract.get('course')) == canonical([asdict(s) for s in spec.course])
        if spec.version in (325, 326):
            from fixed_plane_v325_wind import legacy_course, reward_bridge_course
            from fixed_plane_wind_pose import pose_contract
            from fixed_plane_wind_history import history_contract
            from fixed_plane_wind_objectives import objective_contract
            from fixed_plane_wind_reward import repair_contract
            from fixed_plane_wind_progression import (FAST_SOURCE_SHA256, EXPOSURE_SOURCE_SHA256,
                                                       PRECISION_SOURCE_SHA256)
            fast_migration = (spec.version == 325 and stage == 9 and updates == 3515
                and digest_file(resume) == FAST_SOURCE_SHA256)
            exposure_migration = (spec.version == 325 and stage == 9 and updates == 3583
                and digest_file(resume) == EXPOSURE_SOURCE_SHA256)
            precision_migration = (spec.version == 325 and stage == 9 and updates == 2683
                and digest_file(resume) == PRECISION_SOURCE_SHA256)
            old_course = legacy_course(spec.version)
            archived = reward_bridge_course(spec.version)
            prefix_migration = (isinstance(stage, int) and 0 <= stage < 7
                and spec.course[:7] == old_course[:7]
                and canonical(contract.get('course')) == canonical([asdict(s) for s in old_course])
                and 'reward_repair' not in contract)
            archived_course = (canonical(contract.get('course')) == canonical(
                [asdict(s) for s in archived]) and contract.get('reward_repair') == repair_contract()
                and 'pose_curriculum' not in contract)
            pose_prefix = archived_course and isinstance(stage, int) and 0 <= stage < 9
            s10_migration = (spec.version==325 and stage==9 and resume.is_file()
                and digest_file(resume)=='8237041c76bd4299724acbac4ced0b2d61a2368f2af7dd6b11156116905332be')
            if s10_migration:
                from fixed_plane_wind_pose import archived_pose_first_course
                expected_lesson=archived_pose_first_course(spec.version,archived)[9]
            pose_migration = pose_prefix or s10_migration
            if pose_prefix:
                expected_lesson = archived[stage]
            course_ok = ((course_ok and contract.get('reward_repair') == repair_contract()
                          and contract.get('pose_curriculum') == pose_contract()
                          and contract.get('observed_history') == history_contract()
                          and contract.get('balanced_objectives') == objective_contract(spec.version))
                         or prefix_migration or pose_migration or fast_migration or exposure_migration or precision_migration)
            stack.reward_bridge_migration = prefix_migration
            stack.pose_checkpoint_migration = pose_migration
            stack.fast_pose_checkpoint_migration = fast_migration
            stack.exposure_checkpoint_migration = exposure_migration
            stack.precision_checkpoint_migration = precision_migration
        if isinstance(stage, int) and 0 <= stage < len(spec.course):
            current_dict = asdict(course_config(stack, expected_lesson or spec.course[stage]))
            current = canonical(current_dict)
            stored = canonical(payload.get('env_cfg'))
            if spec.version in (325, 326):
                from fixed_plane_wind_fresh_validation import PROTOCOL as FROZEN_PROTOCOL
                from fixed_plane_wind_training_gates import PROTOCOL, gate_contract
                sheltered = contract.get('startup_shelter') == dict(
                    confirmed_hits=STARTUP_NO_WIND_HITS, ramp_s=STARTUP_RAMP_S)
                from fixed_plane_wind_pose import pose_config_matches
                config_ok = (pose_config_matches(payload.get('env_cfg', {}), current_dict) and sheltered
                    and ((contract.get('validation_protocol') == PROTOCOL
                          and contract.get('training_gates') == gate_contract(spec.version))
                         or contract.get('validation_protocol') == FROZEN_PROTOCOL))
                stack.gate_protocol_migration = contract.get('validation_protocol') != PROTOCOL
                legacy_revision = (contract.get('validation_protocol') is None
                    and stored == canonical(asdict(make_config(stack.native_cfg, spec.course[stage]))))
                config_ok |= legacy_revision or s10_migration
                if fast_migration or exposure_migration:
                    normalized = json.loads(canonical(payload.get('env_cfg', {})))
                    normalized['wind_pose']['ramp_updates'] = current_dict['wind_pose']['ramp_updates']
                    normalized['wind_pose']['quality_gated'] = current_dict['wind_pose']['quality_gated']
                    config_ok |= pose_config_matches(normalized, current_dict) and sheltered
            else:
                config_ok = stored == current
        saved_cfg = json.loads(canonical(payload.get('env_cfg', {})))
        if (payload.get('args', {}).get('curriculum_profile') != spec.profile
                or contract.get('parent_sha256') != spec.origin.parent_sha256
                or not course_ok
                or not isinstance(stage, int) or not 0 <= stage < len(spec.course)
                or payload.get('stage_name') != (expected_lesson or spec.course[stage]).name
                or not isinstance(updates, int) or updates < 0
                or payload.get('step') != spec.origin.parent_step + updates * 131072
                or payload.get('opt') is None
                or int(payload['opt'].t) != spec.origin.parent_adam_t + updates * 32
                    + stack.actor_bootstrap_offset
                or tuple(payload.get(k) for k in ('obs_dim', 'critic_obs_dim', 'act_dim'))
                    != ((114, 453, 7) if saved_cfg.get(
                        'wind_history', {}).get('enabled', False) else (72, 411, 7))
                or not config_ok):
            raise ValueError('Wind recovery requires the exact same full course and stage')
        expected_learner = learner_settings_for_stack(stack)
        if stack.wind_training_mix != 'legacy':
            expected_learner['counterfactual_supervision_coef'] = 0.
        for name, value in expected_learner.items():
            if payload['args'].get(name) != value:
                raise ValueError(f'Wind recovery learner mismatch: {name}')
        anchor = stack.parent.get('stage_actor_anchor_reference_params')
        if payload.get('obs_dim') == 114 and anchor is not None:
            from fixed_plane_wind_history import expand_reference_params
            anchor = expand_reference_params(anchor, actor_only=True)
        if base_version(spec) == 326 and numeric_digest(payload.get(
                'stage_actor_anchor_reference_params')) != numeric_digest(
                    anchor):
            raise ValueError('Wind recovery must retain the exact M13 actor anchor')
        stack.legacy_wind_resume = legacy_revision
        advance = bool(payload.get('stage_converged', False) and not legacy_revision)
        if spec.version in (325, 326):
            saved_progress = payload.get('pose_stage_updates', 0)
            if (type(saved_progress) is not int or not 0 <= saved_progress <= updates
                    or (not pose_migration and not stack.reward_bridge_migration
                        and contract.get('pose_curriculum') is not None
                        and saved_progress != payload.get('stage_update'))):
                raise ValueError('Invalid pose-stage progress clock')
            stack.pose_resume_updates = 0 if advance or pose_migration else saved_progress
            ramp_steps=payload.get('pose_ramp_progress_steps',0)
            if (type(ramp_steps) is not int or ramp_steps<0
                    or (not advance and not pose_migration and ramp_steps>saved_progress*128)):
                raise ValueError('Invalid quality-qualified pose clock')
            stack.pose_resume_progress_steps=0 if advance or pose_migration else ramp_steps
            if fast_migration:
                from fixed_plane_wind_progression import rebase_pose_steps
                old_cfg = json.loads(canonical(payload['env_cfg']))['wind_pose']
                new_updates = spec.course[stage].pose_ramp_updates
                stack.pose_resume_progress_steps = rebase_pose_steps(ramp_steps,
                    old_updates=old_cfg['ramp_updates'], new_updates=new_updates)
                stack.pose_progress_rebase = dict(source_sha256=digest_file(resume),
                    old_steps=ramp_steps, old_ramp_updates=old_cfg['ramp_updates'],
                    new_steps=stack.pose_resume_progress_steps, new_ramp_updates=new_updates)
        stage += int(advance)
        if stage == len(spec.course):
            raise ValueError('Wind course already completed')
    state = stack.ppo.TrainState(jax.tree.map(jnp.asarray, payload['params']),
                                jax.tree.map(jnp.asarray, payload['opt']))
    if resume is not None and (state.params['pi']['l1']['w'].shape[0],
            state.params['v']['l1']['w'].shape[0]) != (payload['obs_dim'], payload['critic_obs_dim']):
        raise ValueError('Checkpoint network dimensions disagree with saved metadata')
    if resume is None:
        state = append_wind_critic(state)
    stack.cfg = course_config(stack, spec.course[stage])
    from fixed_plane_wind_history import dimensions_for_config, expand_history_learner
    expected_dims = dimensions_for_config(stack.cfg)
    if expected_dims[0] == 114:
        state = expand_history_learner(state)
    if (state.params['pi']['l1']['w'].shape[0] != expected_dims[0]
            or state.params['v']['l1']['w'].shape[0] != expected_dims[1]
            or state.params['log_std'].shape != (7,)):
        raise ValueError('Wind continuation network layout does not match its stage')
    for moments in (state.opt.m, state.opt.v):
        if jax.tree.structure(moments) != jax.tree.structure(state.params):
            raise ValueError('Incomplete wind Adam trees')
        for a, b in zip(jax.tree.leaves(state.params), jax.tree.leaves(moments)):
            if a.shape != b.shape or a.dtype != b.dtype:
                raise ValueError('Wind Adam layout mismatch')
    numeric_digest(state)
    if spec.version in (325, 326):
        from fixed_plane_wind_pose import with_pose_clock
        stack.cfg = with_pose_clock(stack.cfg,
            elapsed_steps=getattr(stack, 'pose_resume_progress_steps', 0))
    return state, int(payload['step']), stage, updates


def make_train_functions(stack, env, args):
    """Reuse the exact parent objective and all32 minibatches, with native MJX."""
    from fixed_plane_wind_optimizer_conditioning import configure_optimizer
    mix = getattr(args, 'wind_training_mix', getattr(stack, 'wind_training_mix', 'legacy'))
    from fixed_plane_wind_clean_rehearsal import REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX, CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX, PRECONTACT_CENTER_MIX, BALANCED_PRECONTACT_MIX, RELEASE_PRECISION_MIX, CYCLE_RELEASE_MIX, MEASURED_PRECISION_MIX, STEADY_BRAKING_MIX, CONFIRMED_LAUNCH_BRAKING_MIX
    optimizer_mix = REFERENCE_CONSTRAINT_MIX if mix in (CENTER_PRECISION_MIX,CENTER_APEX_MIX,CLEAN_APEX_PRIORITY_MIX,CLEAN_GROUP_PRIORITY_MIX,NORMALIZED_GROUP_PRIORITY_MIX,PRECONTACT_CENTER_MIX,BALANCED_PRECONTACT_MIX,RELEASE_PRECISION_MIX,CYCLE_RELEASE_MIX,MEASURED_PRECISION_MIX,STEADY_BRAKING_MIX,CONFIRMED_LAUNCH_BRAKING_MIX,CENTER_TAIL_MIX,CALM_RELEASE_MIX,EARLY_RECOVERY_MIX,MATURE_PRECISION_MIX,V2_GROUP_BALANCE_MIX,CALM_RETURN_MIX,IMPACT_NORMAL_MIX,CUBIC_RECOVERY_MIX,CLEAN_JOINT_QUALITY_MIX) else mix
    configure_optimizer(stack.ppo, getattr(stack, 'bootstrap_online_recipe', None), optimizer_mix)
    ppo = SimpleNamespace(**vars(stack.ppo))
    from fixed_plane_wind_clean_rehearsal import mix_contract
    if mix_contract(mix) is not None and stack.spec.version not in (325, 326):
        raise ValueError('No-BC clean rehearsal is scoped to V325/V326')

    def make_fns(**kwargs):
        if mix != 'legacy':
            kwargs.update(counterfactual_replay_obs=None, counterfactual_replay_actions=None,
                          counterfactual_supervision_coef=0.)
        if env.obs_dim == 114:
            from fixed_plane_wind_history import pad_replay_observations, expand_reference_params
            for key in ('counterfactual_replay_obs', 'actor_anchor_replay_obs'):
                if kwargs.get(key) is not None:
                    kwargs[key] = pad_replay_observations(kwargs[key])
            if kwargs.get('reference_params') is not None:
                kwargs['reference_params'] = expand_reference_params(kwargs['reference_params'], actor_only=True)
        kwargs['full_rollout_metric_names'] += ('wind/mode',)
        if stack.spec.version in (325, 326):
            from fixed_plane_v321_long_episode import QUALITY_FIELDS
            kwargs['full_rollout_metric_names'] += QUALITY_FIELDS
            if getattr(getattr(env, 'cfg', None), 'wind_flight_alignment', False):
                from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS
                kwargs['full_rollout_metric_names'] += HEIGHT_FIELDS
            if getattr(getattr(env, 'cfg', None), 'wind_reference_constraints', False):
                from fixed_plane_wind_reference_constraints import REFERENCE_FIELDS
                kwargs['full_rollout_metric_names'] += REFERENCE_FIELDS
        if getattr(stack, 'bootstrap_online_recipe', None):
            kwargs.update(reference_params=stack.bootstrap_stage_anchor,
                          actor_anchor_kl_coef=args.actor_anchor_kl_coef)
        return stack.ppo.make_train_fns(**kwargs)

    ppo.make_train_fns = make_fns
    proxy = SimpleNamespace(**{**vars(stack), 'spec': stack.spec.origin, 'ppo': ppo})
    return make_parent_functions(proxy, env, args, 'legacy')


def storage_contract(mode: str, wandb_upload: str = 'source_artifact') -> dict:
    """Keep the original storage path or explicitly select M12's retention."""
    if mode not in ('keep_all', 'm12_best'):
        raise ValueError('Unknown wind checkpoint retention policy')
    if wandb_upload not in ('source_artifact', 'metrics_only'):
        raise ValueError('Unknown W&B upload policy')
    return dict(mode=mode, save_every_updates=100,
        cleanup_every_saves=5 if mode == 'm12_best' else None,
        preserve_converged=True, wandb_mode='online', wandb_spool='RAM only',
        wandb_upload=wandb_upload)


def checkpoint_selection_score(stage: int, quality_ok: bool, endpoint: dict,
                               ongoing_s_mean: float, *, quality: dict | None = None,
                               version: int = 325, wind_m_s: float = 0.) -> tuple:
    """Retain M12's stage/quality/survival/length/ongoing-duration ordering."""
    tail=(endpoint.get('survival',0.),endpoint.get('len_fraction',0.),ongoing_s_mean)
    if quality is not None:
        from fixed_plane_wind_objectives import quality_violation_score
        worst,total=quality_violation_score(quality,version,wind_m_s)
        return (stage,float(quality_ok),-worst,-total,*tail)
    return (stage,float(quality_ok),*tail)


def transition_domain(stack, runner, stage):
    """Resample wind alone; retain physical state, policy history and native RNG."""
    from fixed_plane_v325_wind import WIND_FIELDS, initialize_wind
    import jax.numpy as jnp
    previous_cfg=getattr(stack,'cfg',None)
    stack.cfg = course_config(stack, stack.spec.course[stage])
    if stack.spec.version in (325, 326) and hasattr(stack.cfg, 'wind_pose'):
        from fixed_plane_wind_pose import with_pose_clock
        clock = np.asarray(runner.env_state.total_env_steps)
        if clock.shape != (1024,) or np.any(clock != clock[0]):
            raise ValueError('Pose curriculum requires synchronized persistent policy-step clocks')
        stack.cfg = with_pose_clock(stack.cfg, origin_steps=int(clock[0]))
    env = stack.env_class(stack.xml, n_envs=1024, cfg=stack.cfg)
    if env.obs_dim == 114 and runner.obs.shape[-1] == 72:
        state, obs = env.attach(runner.env_state, runner.obs)
        runner = runner._replace(env_state=state, obs=obs)
    if hasattr(runner.env_state,'wind_pose_ramp_steps'):
        runner=runner._replace(env_state=runner.env_state._replace(
            wind_pose_ramp_steps=jnp.zeros_like(runner.env_state.wind_pose_ramp_steps),
            wind_pose_ramp_enabled=jnp.zeros_like(runner.env_state.wind_pose_ramp_enabled)))
    if (stack.spec.version in (325,326) and hasattr(stack.cfg,'wind_pose')
            and previous_cfg is not None and canonical({k:v for k,v in asdict(previous_cfg.wind_config).items() if k!='force_limit_n'})
                ==canonical({k:v for k,v in asdict(stack.cfg.wind_config).items() if k!='force_limit_n'})):
        return env,runner._replace(critic_obs=env.get_critic_obs(runner.env_state,runner.obs))
    process = initialize_wind(runner.env_state.rng, stack.cfg.wind_config,
                              runner.env_state.step_count*env.dt)
    if stack.cfg.wind_config.startup_no_wind_hits:
        hits = runner.env_state.wind_startup_hit_count
        start = jnp.where(hits >= stack.cfg.wind_config.startup_no_wind_hits,
            jnp.where(runner.env_state.wind_startup_started_s >= 0.,
                      runner.env_state.wind_startup_started_s, runner.env_state.step_count*env.dt), -1.)
        process = process._replace(wind_startup_hit_count=hits, wind_startup_started_s=start)
    state = runner.env_state._replace(**dict(zip(WIND_FIELDS, process)))
    return env, runner._replace(env_state=state,
        critic_obs=env.get_critic_obs(state, runner.obs))


def execution_binding(stack, source, checkpoint, seed, retention='keep_all',
                      wandb_upload='source_artifact'):
    import jax
    import mujoco
    from fixed_plane_wind_history import dimensions_for_config
    binding = dict(profile=stack.spec.profile, parent_sha256=stack.spec.origin.parent_sha256,
        checkpoint_sha256=digest_file(checkpoint), seed=seed,
        gpu_uuid=stack.spec.origin.gpu_uuid, jax_version=jax.__version__,
        mujoco_version=mujoco.__version__, xml_sha256=digest_file(stack.xml),
        native_sources=stack.saved_contract['frozen_parent_source_sha256'],
        replay_sha256=stack.saved_contract['replay_sha256'],
        source_sha256={name: digest_file(source/name) for name in SOURCE_FILES},
        course=[asdict(s) for s in stack.spec.course], execution='native',
        storage=storage_contract(retention, wandb_upload),
        actor_critic_action=list(dimensions_for_config(stack.cfg)),
        training_episode_s=1200.)
    if getattr(stack, 'actor_bootstrap', None):
        from fixed_plane_wind_bootstrap_restore import learner_settings_for_stack
        binding.update(actor_bootstrap=stack.actor_bootstrap, learner=learner_settings_for_stack(stack))
    if getattr(stack, 'bootstrap_online_recipe', None):
        binding['bootstrap_online_recipe'] = stack.bootstrap_online_recipe
    if getattr(stack, 'online_recipe_migration', None):
        binding['online_recipe_migration'] = stack.online_recipe_migration
    if hasattr(stack,'reward_resume_policy_steps'):
        binding['normalized_reward_resume'] = dict(
            policy_steps=stack.reward_resume_policy_steps, physical_clock_unchanged=True)
    if hasattr(stack, 'calm_return_reward_resume_steps'):
        binding['calm_return_reward_resume'] = dict(
            policy_steps=stack.calm_return_reward_resume_steps, physical_clock_unchanged=True)
    if hasattr(stack, 'cubic_recovery_reward_resume_steps'):
        binding['cubic_recovery_reward_resume'] = dict(
            policy_steps=stack.cubic_recovery_reward_resume_steps, physical_clock_unchanged=True)
    if hasattr(stack, 'clean_joint_quality_reward_resume_steps'):
        binding['clean_joint_quality_reward_resume'] = dict(
            policy_steps=stack.clean_joint_quality_reward_resume_steps, physical_clock_unchanged=True)
    if hasattr(stack, 'impact_normal_reward_resume_steps'):
        binding['impact_normal_reward_resume'] = dict(
            policy_steps=stack.impact_normal_reward_resume_steps, physical_clock_unchanged=True)
    if hasattr(stack,"precontact_reward_resume_steps"):
        binding["precontact_reward_resume"] = dict(
            base_policy_steps=8192, new_policy_steps=stack.precontact_reward_resume_steps,
            birth_update=stack.precontact_reward_birth_update,
            physical_clock_unchanged=True)
    if stack.spec.version in (325, 326):
        from fixed_plane_wind_training_gates import PROTOCOL, gate_contract
        from fixed_plane_wind_reward import repair_contract
        from fixed_plane_wind_pose import pose_contract
        from fixed_plane_wind_objectives import objective_contract
        from fixed_plane_wind_history import history_contract
        binding.update(validation_protocol=PROTOCOL, startup_shelter=dict(
            confirmed_hits=STARTUP_NO_WIND_HITS, ramp_s=STARTUP_RAMP_S),
            training_gates=gate_contract(stack.spec.version), reward_repair=repair_contract(),
            pose_curriculum=pose_contract(),balanced_objectives=objective_contract(stack.spec.version),
            observed_history=history_contract())
    return binding


def validate_qualification(report, binding):
    """Check actual numeric audit records, independent of any declared pass flag."""
    if binding.get('profile') == registered_spec(327).profile:
        from fixed_plane_v327_wind_reward import validate_reward_pair
        validate_reward_pair(report, binding)
        return
    parity = report.get('zero_wind_replay', {})
    digest = parity.get('reference_digest', '')
    probes = report.get('probes', [])
    if (canonical(report.get('binding')) != canonical(binding)
            or report.get('backend') != 'gpu'
            or parity.get('steps') != 4 or not re.fullmatch('[a-f0-9]{64}', digest)
            or digest != parity.get('candidate_digest')
            or digest != parity.get('disabled_digest')
            or [p.get('stage') for p in probes] != [0, 2, 4, 6, 7]):
        raise ValueError('A matching frozen CUDA wind qualification is required')
    for probe in probes:
        rows = np.asarray(probe.get('samples', []), float)
        limit = binding['course'][probe['stage']]['max_speed_m_s']
        if (rows.shape != (10, 4) or not np.isfinite(rows).all()
                or not np.all(rows[:, 0] == 64)
                or np.any(rows[:, 1] < 0.) or np.any(rows[:, 1] > limit+1e-6)
                or np.any(rows[:, 2] < 0.) or np.any(rows[:, 2] > .002001)
                or np.any(rows[:, 3] < 0.) or np.any(rows[:, 3] > 64)):
            raise ValueError('Incomplete/nonfinite or out-of-bound frozen wind probe')


def qualify(stack, args, train_state, binding, runtime, stop):
    """Frozen-policy execution audit and candidate sweep; never update PPO."""
    if stack.spec.version == 327:
        from fixed_plane_v327_wind_reward import qualify_reward_pair
        qualify_reward_pair(stack, args, train_state, binding, runtime, stop)
        return
    import jax
    import jax.numpy as jnp
    from bounded_validation_memory import release_caches
    native = importlib.import_module('mjx_juggle_env')
    old_class = (stack.module.environment_class(native.MjxJuggleEnv)
        if base_version(stack.spec) == 325 else stack.module.environment_class(
            native.MjxJuggleEnv, reward_preset='late_center_precision', grouped_conditions=True))
    original = old_class(stack.xml, n_envs=8, cfg=stack.native_cfg)
    quiet = stack.env_class(stack.xml, n_envs=8,
        cfg=course_config(stack, stack.spec.course[0], enabled=False))
    zero_cfg = course_config(stack, stack.spec.course[0])
    zero = stack.env_class(stack.xml, n_envs=8, cfg=replace(zero_cfg,
        wind_config=replace(zero_cfg.wind_config, max_speed_m_s=0.)))
    keys = jax.random.split(jax.random.PRNGKey(args.seed), 8)
    a, ao = jax.jit(original.reset)(keys)
    b, bo = jax.jit(quiet.reset)(keys)
    c, co = jax.jit(zero.reset)(keys)
    reference, candidate, disabled = [], [], []
    step_a, step_b, step_c = jax.jit(original.step), jax.jit(quiet.step), jax.jit(zero.step)
    for _ in range(4):
        if stop[0]:
            raise InterruptedError('Wind qualification cancelled')
        action = stack.ppo.policy_mean(train_state.params, ao)
        a, ao, ar, ad, am = step_a(a, action)
        b, bo, br, bd, bm = step_b(b, action)
        c, co, cr, cd, cm = step_c(c, action)
        common = type(a)(*(getattr(b, name) for name in a._fields))
        common_zero = type(a)(*(getattr(c, name) for name in a._fields))
        reference.append(numeric_digest((a, ao, ar, ad, am)))
        disabled.append(numeric_digest((common, bo, br, bd, {k: bm[k] for k in am})))
        candidate.append(numeric_digest((common_zero, co, cr, cd, {k: cm[k] for k in am})))
    import hashlib
    report = dict(binding=binding, backend=jax.default_backend(),
        scope='frozen numerical execution; candidate ranges are not effectiveness or hardware evidence',
        zero_wind_replay=dict(steps=4,
            reference_digest=hashlib.sha256(''.join(reference).encode()).hexdigest(),
            candidate_digest=hashlib.sha256(''.join(candidate).encode()).hexdigest(),
            disabled_digest=hashlib.sha256(''.join(disabled).encode()).hexdigest()), probes=[])
    del a, b, c, original, quiet, zero, step_a, step_b, step_c
    release_caches()
    for stage in (0, 2, 4, 6, 7):
        env = stack.env_class(stack.xml, n_envs=64,
            cfg=course_config(stack, stack.spec.course[stage]))
        rng, reset = jax.random.split(jax.random.PRNGKey(args.seed))
        state, obs = jax.jit(env.reset)(jax.random.split(reset, 64))
        from fixed_plane_wind_validation import build_evaluator
        evaluator = build_evaluator(env, stack.ppo, args, metric_prefix(stack.spec), False)
        active = jnp.ones(64, bool)
        samples, totals = [], {}
        for second in range(10):
            if stop[0]:
                raise InterruptedError('Wind qualification cancelled')
            (state, obs, active, rng), packet = evaluator(train_state.params, state, obs, active, rng)
            from fixed_plane_v325_wind import incremental_force, wind_velocity
            velocity = wind_velocity(state, state.step_count*env.dt, env.cfg.wind_config)
            force, _ = incremental_force(state.data.qvel[:, env.ball_vadr:env.ball_vadr+3],
                                         velocity, env.cfg.wind_config)
            finite = np.isfinite(np.asarray(obs)).all(axis=-1)
            finite &= np.isfinite(np.asarray(state.data.qpos)).all(axis=-1)
            finite &= np.isfinite(np.asarray(state.data.qvel)).all(axis=-1)
            samples.append([int(finite.sum()), float(jnp.linalg.norm(velocity, axis=-1).max()),
                float(jnp.linalg.norm(force, axis=-1).max()), int(np.asarray(active).sum())])
            for name, value in jax.device_get(packet).items():
                totals[name] = totals.get(name, 0.) + float(np.asarray(value).sum())
            print(f'[V{stack.spec.version} frozen sweep] S{stage+1} t={second+1}/10s', flush=True)
        report['probes'].append(dict(stage=stage, samples=samples,
            hits_mean=float(np.asarray(state.hit_count).mean()),
            survival=float(np.asarray(active).mean()), metrics=totals))
        temperature = float(subprocess.check_output(['nvidia-smi', '-i',
            stack.spec.origin.gpu_uuid, '--query-gpu=temperature.gpu',
            '--format=csv,noheader,nounits'], text=True))
        if temperature >= 78.:
            raise RuntimeError('Frozen wind qualification reached78C')
        del state, obs, evaluator, env
        release_caches()
    (runtime/'wind_qualification_attempt.json').write_text(json.dumps(report, indent=2))
    validate_qualification(report, binding)
    (runtime/'wind_qualification.json').write_text(json.dumps(report, indent=2))


def main(version):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--runtime', type=Path)
    parser.add_argument('--save-dir', type=Path)
    parser.add_argument('--run-id')
    parser.add_argument('--seed', type=int, default=2026111100+version-320)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--qualification', type=Path)
    parser.add_argument('--probe-updates', type=int, default=0,
                        help='Explicit bounded diagnostic PPO run; zero is normal training')
    parser.add_argument('--actor-bootstrap-online-recipe', choices=(
        'recovery_full20_stage_anchor_002_v1',
        'recovery_full20_bounded_quality_anchor002_v1',
        'recovery_full20_joint_quality_anchor002_v1',
        'recovery_full20_joint_quality_sigma01_anchor005_v1',
        'recovery_full20_joint_precision_low_lr_v1',
        'recovery_full20_gate_margin_quality_v1',
        'recovery_full20_gate_margin_gaussian_apex_v1',
        'recovery_full20_height_priority_quality_v1',
        'recovery_full20_height_sigma005_anchor00125_v1',
        'recovery_full20_height_z_margin_v1',
        'recovery_full20_height_motion_z_margin_v1',
        'recovery_full20_height_motion_z_anticipation_v1',
        'recovery_full20_height_motion_z_anticipation_apex008_v1',
        'recovery_full20_apex008_native_z_failure_floor_v1',
        'recovery_full20_native_z_floor_apex_support_v1',
        'recovery_full20_apex_support_history_xy_eps000002_v1'))
    parser.add_argument('--checkpoint-retention', choices=('keep_all', 'm12_best'),
                        default='keep_all')
    parser.add_argument('--wandb-upload', choices=('source_artifact', 'metrics_only'),
                        default='source_artifact')
    from fixed_plane_wind_clean_rehearsal import MODES
    parser.add_argument('--training-mix', choices=MODES, default=None,
                        help='Explicit online clean-rehearsal/objective override; inherit on resume')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--preflight', action='store_true')
    modes.add_argument('--qualify', action='store_true')
    opts = parser.parse_args()
    if not 0 <= opts.probe_updates <= 2000:
        raise ValueError('Probe update budget must lie in0--2000')
    spec = registered_spec(version)
    if opts.preflight:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    elif (opts.runtime is None or opts.save_dir is None or not opts.run_id
            or not re.fullmatch('[a-zA-Z0-9_-]+', opts.run_id)
            or opts.runtime.resolve().parent != Path('/dev/shm')
            or os.environ.get('PP_MEMORY_SCOPE_ACTIVE') != 'YES'
            or os.environ.get('CUDA_VISIBLE_DEVICES') != spec.origin.gpu_uuid):
        raise ValueError('Use the isolated version-specific wind launcher')
    source = Path(__file__).resolve().parent
    stack = load_stack(opts.repo_root.resolve(), source, version)
    stack.requested_online_recipe = opts.actor_bootstrap_online_recipe
    train_state, global_step, stage, updates = restore_training_state(stack, opts.resume)
    spec = stack.spec
    stack.wind_training_mix = opts.training_mix or stack.wind_training_mix
    if stack.wind_training_mix != 'legacy' and spec.version not in (325, 326):
        raise ValueError('No-BC clean rehearsal is scoped to V325/V326')
    stack.cfg = course_config(stack, spec.course[stage])
    from fixed_plane_wind_clean_rehearsal import NORMALIZED_GROUP_PRIORITY_MIX
    if stack.wind_training_mix == NORMALIZED_GROUP_PRIORITY_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Normalized reward requires its reviewed V325 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_normalized_priority import source_resume_steps, bind_reward_resume
        stack.reward_resume_policy_steps = source_resume_steps(
            load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = bind_reward_resume(stack.env_class, stack.reward_resume_policy_steps)
    from fixed_plane_wind_clean_rehearsal import PRECONTACT_CENTER_MIX
    if stack.wind_training_mix == PRECONTACT_CENTER_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError("Precontact reward requires its reviewed V325 source")
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_precontact_center import source_resume_context, bind_reward_resume
        context = source_resume_context(
            load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.precontact_reward_resume_steps = context['policy_steps']
        stack.precontact_reward_birth_update = context['birth_update']
        stack.env_class = bind_reward_resume(stack.env_class, stack.precontact_reward_resume_steps)
    from fixed_plane_wind_clean_rehearsal import BALANCED_PRECONTACT_MIX
    if stack.wind_training_mix == BALANCED_PRECONTACT_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Balanced preview requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_balanced_precontact import source_resume_context
        from fixed_plane_wind_precontact_center import bind_reward_resume
        context=source_resume_context(load_checkpoint(opts.resume),digest_file(opts.resume))
        stack.precontact_reward_resume_steps=context['policy_steps']
        stack.precontact_reward_birth_update=context['birth_update']
        stack.env_class=bind_reward_resume(stack.env_class,context['policy_steps'])
    from fixed_plane_wind_clean_rehearsal import RELEASE_PRECISION_MIX
    if stack.wind_training_mix == RELEASE_PRECISION_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Release precision requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_release_precision import source_resume_context
        from fixed_plane_wind_precontact_center import bind_reward_resume
        context=source_resume_context(load_checkpoint(opts.resume),digest_file(opts.resume))
        stack.precontact_reward_resume_steps=context['policy_steps']
        stack.precontact_reward_birth_update=context['birth_update']
        stack.env_class=bind_reward_resume(stack.env_class,context['policy_steps'])
    from fixed_plane_wind_clean_rehearsal import CYCLE_RELEASE_MIX
    if stack.wind_training_mix == CYCLE_RELEASE_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Cycle release requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_cycle_release import source_resume_context
        from fixed_plane_wind_precontact_center import bind_reward_resume
        context=source_resume_context(load_checkpoint(opts.resume),digest_file(opts.resume))
        stack.precontact_reward_resume_steps=context['policy_steps']
        stack.precontact_reward_birth_update=context['birth_update']
        stack.env_class=bind_reward_resume(stack.env_class,context['policy_steps'])
    from fixed_plane_wind_clean_rehearsal import MEASURED_PRECISION_MIX
    if stack.wind_training_mix == MEASURED_PRECISION_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Measured precision requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_measured_precision import verify_source
        verify_source(load_checkpoint(opts.resume), digest_file(opts.resume))
    from fixed_plane_wind_clean_rehearsal import STEADY_BRAKING_MIX
    if stack.wind_training_mix == STEADY_BRAKING_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Steady braking requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_steady_braking import verify_source
        from fixed_plane_wind_normalized_priority import bind_reward_resume
        phase = verify_source(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = bind_reward_resume(stack.env_class, phase)
    if stack.wind_training_mix == CENTER_TAIL_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Contact-offset cost requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_center_tail import center_tail_phase
        phase = center_tail_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = partial(stack.env_class, precontact_resume_policy_steps=phase)
    if stack.wind_training_mix == CALM_RELEASE_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Calm vertical release credit requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_calm_release import calm_release_phase
        phase = calm_release_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == EARLY_RECOVERY_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Early racket recovery requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_early_recovery import reward_phase
        phase = reward_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == CALM_RETURN_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Calm midpoint return requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_calm_return import return_reward_phase
        phase = return_reward_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.calm_return_reward_resume_steps = phase
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == IMPACT_NORMAL_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Impact normal precision requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_impact_normal_precision import normal_reward_phase
        phase = normal_reward_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.impact_normal_reward_resume_steps = phase
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == CUBIC_RECOVERY_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Cubic recovery requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_cubic_recovery import cubic_reward_phase
        phase = cubic_reward_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.cubic_recovery_reward_resume_steps = phase
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == CLEAN_JOINT_QUALITY_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Clean joint-quality priority requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_clean_joint_quality import joint_reward_phase
        phase = joint_reward_phase(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.clean_joint_quality_reward_resume_steps = phase
        stack.env_class = partial(stack.env_class, reward_resume_policy_steps=phase)
    if stack.wind_training_mix == MATURE_PRECISION_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Mature precision context requires its reviewed V325 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_resume_precision_clock import verify_mature_source
        verify_mature_source(load_checkpoint(opts.resume), digest_file(opts.resume))
    if stack.wind_training_mix == V2_GROUP_BALANCE_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('V2 group balance requires its reviewed V325 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_v2_group_balance import resume_policy_steps
        from fixed_plane_wind_normalized_priority import bind_reward_resume
        phase = resume_policy_steps(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.reward_resume_policy_steps = phase
        stack.env_class = bind_reward_resume(stack.env_class, phase)
    from fixed_plane_wind_clean_rehearsal import CONFIRMED_LAUNCH_BRAKING_MIX
    if stack.wind_training_mix == CONFIRMED_LAUNCH_BRAKING_MIX:
        if spec.version != 325 or opts.resume is None:
            raise ValueError('Confirmed-launch braking requires its reviewed V325 S16 source')
        from validate_juggle_mjx_ppo import load_checkpoint
        from fixed_plane_wind_confirmed_launch_braking import verify_source
        from fixed_plane_wind_precontact_center import bind_reward_resume
        phase = verify_source(load_checkpoint(opts.resume), digest_file(opts.resume))
        stack.env_class = bind_reward_resume(stack.env_class, phase)
    from fixed_plane_wind_online_recipe import enter_stage
    enter_stage(stack, train_state.params, stage)
    from fixed_plane_wind_bootstrap_restore import learner_settings_for_stack
    if (getattr(stack, 'actor_bootstrap_requires_probe', False)
            and not opts.preflight and opts.probe_updates == 0):
        raise ValueError('New actor source requires a bounded PPO probe until S10 graduation')
    binding = execution_binding(stack, source, opts.resume or stack.parent_path, opts.seed,
                                opts.checkpoint_retention, opts.wandb_upload)
    from fixed_plane_wind_clean_rehearsal import mix_contract
    binding['training_mix'] = mix_contract(stack.wind_training_mix)
    effective_learner = learner_settings_for_stack(stack)
    if stack.wind_training_mix != 'legacy':
        effective_learner['counterfactual_supervision_coef'] = 0.
    if stack.wind_training_mix == 'clean10_dynamics_flight_v1':
        binding['source_learner'] = binding.get('learner', {})
        binding['learner'] = dict(effective_learner)
    if opts.preflight:
        print(json.dumps(dict(binding=binding, parent_model=spec.parent_model,
            parent_checkpoint=str(stack.parent_path), true_step=global_step,
            Adam_t=int(train_state.opt.t), current_stage=stage+1,
            learner=effective_learner,
            wind_physics=asdict(stack.cfg.wind_config), candidate_bounds=True,
            no_camera_noise_added=True, training_started=False,
            frozen_qualification_required=spec.version not in (325, 326),
            promotion_evidence=False), indent=2))
        return
    import jax
    import jax.numpy as jnp
    from bounded_validation_memory import release_caches
    if jax.default_backend() != 'gpu' or len(jax.devices()) != 1:
        raise ValueError('The selected physical CUDA GPU is required')
    if not opts.qualify:
        if opts.qualification is None and spec.version not in (325, 326):
            raise ValueError('Frozen CUDA wind qualification must precede formal training')
        if opts.qualification is not None:
            validate_qualification(json.loads(opts.qualification.read_text()), binding)
    stop = [False]
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.__setitem__(0, True))
    args = argparse.Namespace(**stack.parent['args'])
    vars(args).update(effective_learner)
    args.wind_training_mix = stack.wind_training_mix
    args.curriculum_profile, args.seed, args.xml = spec.profile, opts.seed, stack.xml
    args.save_dir, args.wandb_name, args.wandb_id = opts.save_dir, opts.run_id, opts.run_id
    args.wandb_mode, args.wandb_resume = 'online', 'never'
    args.resume_from, args.resume_curriculum_state = opts.resume or stack.parent_path, False
    args.reset_optimizer_on_resume = args.reset_critic_on_resume = False
    args.apply_every_minibatch, args.conditional_reset_done = True, True
    args.total_update_limit, args.max_stage_updates = None, -1
    args.max_stages, args.mjx_execution_mode = len(spec.course), 'native'
    args.save_every_updates = 100
    args.archive_every_updates = 0
    args.checkpoint_retention = opts.checkpoint_retention
    args.wandb_upload = opts.wandb_upload
    if opts.qualify:
        qualify(stack, args, train_state, binding, opts.runtime, stop)
        return
    env = stack.env_class(stack.xml, n_envs=1024, cfg=stack.cfg)
    from fixed_plane_wind_history import dimensions_for_config
    if (env.obs_dim, env.critic_obs_dim, env.act_dim, env.max_steps) != (*dimensions_for_config(stack.cfg),240000):
        raise ValueError('Wind layout/horizon mismatch')
    rng, reset_key = jax.random.split(jax.random.PRNGKey(opts.seed))
    state, obs = jax.jit(env.reset)(jax.random.split(reset_key, 1024))
    if spec.version in (325, 326):
        state, obs = strong_arrays((state, obs))
    runner = stack.ppo.RunnerState(state, obs, env.get_critic_obs(state, obs), rng,
        jnp.zeros(1024), jnp.zeros(1024, jnp.int32))
    jax.block_until_ready((runner, train_state))
    release_caches()
    run_training(stack, env, args, opts, train_state, runner, global_step,
                 stage, updates, binding, stop)


def run_training(stack, env, args, opts, train_state, runner, global_step,
                 stage, updates, binding, stop):
    """V325/V326 advance directly from all sampled-training gates and hold."""
    import jax
    import jax.numpy as jnp
    import wandb
    from bounded_validation_memory import release_caches
    from fixed_plane_v321_long_episode import (
        CheckpointRetention, EndpointWindow, QUALITY_FIELDS, quality_summary,
    )
    from fixed_plane_v322_condition_gates import (
        ConditionEndpointWindow, condition_gate_settings, condition_gates_passed,
    )
    from fixed_plane_wind_validation import (
        WindEndpointWindow, current_domain_mask, endpoints_passed, evaluate_policy,
    )
    from fixed_plane_wind_training_gates import (
        TrainingSegments, advance_hold, completion_metadata, gate_contract,
        progression_decision, training_gate_decision,
    )
    from fixed_plane_wind_clean_rehearsal import effective_lesson, group_limits, mix_contract, FLIGHT_MODES
    mix = getattr(args, 'wind_training_mix', 'legacy')
    quality_fields = QUALITY_FIELDS
    if mix in FLIGHT_MODES:
        from fixed_plane_wind_flight_outcomes import HEIGHT_FIELDS, flight_quality_summary
        quality_fields += HEIGHT_FIELDS
        quality_summary = flight_quality_summary
    from fixed_plane_wind_clean_rehearsal import REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX, CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX, PRECONTACT_CENTER_MIX, BALANCED_PRECONTACT_MIX, RELEASE_PRECISION_MIX, CYCLE_RELEASE_MIX, MEASURED_PRECISION_MIX, STEADY_BRAKING_MIX, CONFIRMED_LAUNCH_BRAKING_MIX
    reference_mode = mix in (REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX, CLEAN_APEX_PRIORITY_MIX, CLEAN_GROUP_PRIORITY_MIX, NORMALIZED_GROUP_PRIORITY_MIX, PRECONTACT_CENTER_MIX, BALANCED_PRECONTACT_MIX, RELEASE_PRECISION_MIX, CYCLE_RELEASE_MIX, MEASURED_PRECISION_MIX, STEADY_BRAKING_MIX, CONFIRMED_LAUNCH_BRAKING_MIX, CENTER_TAIL_MIX, CALM_RELEASE_MIX,
                             EARLY_RECOVERY_MIX, MATURE_PRECISION_MIX, V2_GROUP_BALANCE_MIX, CALM_RETURN_MIX, IMPACT_NORMAL_MIX, CUBIC_RECOVERY_MIX, CLEAN_JOINT_QUALITY_MIX)
    if reference_mode:
        from fixed_plane_wind_reference_constraints import (
            REFERENCE_FIELDS, reference_quality_summary, gate_quality,
            make_reference_gate, reference_checkpoint_score,
        )
        quality_fields += REFERENCE_FIELDS
        quality_summary = reference_quality_summary
    def project_quality(q):
        return (gate_quality(q) if reference_mode and getattr(stack.spec.course[stage],'strong_wind_reward',False)
                else q)
    spec = stack.spec
    training_only = spec.version in (325, 326)
    grouped = base_version(spec) == 326
    prefix = metric_prefix(spec)
    opts.save_dir.mkdir(parents=True, exist_ok=False)
    contract = dict(profile=spec.profile, parent_model=spec.parent_model,
        parent_sha256=spec.origin.parent_sha256, parent_step=spec.origin.parent_step,
        course=[asdict(s) for s in spec.course], binding=binding,
        execution='native', environment_resume='fresh native episodes and wind; fresh gate windows',
        transition='resample only wind; preserve robot/ball/q_ref and await current-domain episodes',
        storage=storage_contract(opts.checkpoint_retention, opts.wandb_upload),
        candidate_bounds=True, hardware_coverage=False)
    if spec.version in (325, 326):
        from fixed_plane_wind_training_gates import PROTOCOL
        from fixed_plane_wind_reward import repair_contract
        from fixed_plane_wind_pose import pose_contract
        from fixed_plane_wind_objectives import objective_contract
        contract['training_mix'] = mix_contract(mix)
        contract['effective_course'] = [asdict(effective_lesson(s, mix)) for s in spec.course]
        contract['validation_protocol'] = PROTOCOL
        contract['training_gates'] = gate_contract(spec.version)
        contract['startup_shelter'] = dict(confirmed_hits=STARTUP_NO_WIND_HITS, ramp_s=STARTUP_RAMP_S)
        contract['legacy_checkpoint_migration'] = bool(getattr(stack, 'legacy_wind_resume', False))
        contract['gate_protocol_migration'] = bool(getattr(stack, 'gate_protocol_migration', False))
        contract['training_endpoints'] = 'refreshing full-duration sampled-training segments'
        contract['transition'] = 'resample wind; retain physical state; start new-domain training segments immediately'
        contract['independent_validation_required'] = False
        contract['reward_repair'] = repair_contract()
        contract['reward_bridge_migration'] = bool(getattr(stack, 'reward_bridge_migration', False))
        contract['pose_curriculum'] = pose_contract()
        from fixed_plane_wind_history import history_contract
        contract['observed_history'] = history_contract()
        contract['balanced_objectives']=objective_contract(spec.version)
        contract['pose_checkpoint_migration'] = bool(getattr(stack, 'pose_checkpoint_migration', False))
        contract['fast_pose_checkpoint_migration'] = bool(getattr(stack, 'fast_pose_checkpoint_migration', False))
        contract['exposure_checkpoint_migration'] = bool(getattr(stack, 'exposure_checkpoint_migration', False))
        contract['precision_checkpoint_migration'] = bool(getattr(stack, 'precision_checkpoint_migration', False))
        contract['pose_clock_rebase'] = getattr(stack, 'pose_progress_rebase', None)
        contract['bounded_probe_updates'] = getattr(opts, 'probe_updates', 0)
        if getattr(stack, 'actor_bootstrap', None):
            contract['actor_bootstrap'] = stack.actor_bootstrap
        if getattr(stack, 'bootstrap_online_recipe', None):
            contract['bootstrap_online_recipe'] = stack.bootstrap_online_recipe
        if getattr(stack, 'online_recipe_migration', None):
            contract['online_recipe_migration'] = stack.online_recipe_migration
    (opts.runtime/'contract.json').write_text(json.dumps(contract, indent=2))
    run = wandb.init(project='pingpong-mjx', name=opts.run_id, id=opts.run_id,
        resume='never', mode='online', dir=str(opts.runtime),
        settings=wandb.Settings(x_disable_stats=True),
        config={**vars(args), 'wind_contract': contract},
        tags=[f'v{spec.version}', spec.parent_model, 'horizontal-wind-increment', 'native-mjx',
              'training-only-advancement' if training_only else 'frozen-acceptance'])
    run.define_metric('update')
    run.define_metric('*', step_metric='update')
    if opts.wandb_upload == 'source_artifact':
        artifact = wandb.Artifact(opts.run_id+'-source', type='source')
        artifact.add_dir(str(opts.runtime/'source'))
        paths = [opts.runtime/'contract.json', opts.runtime/'command.sh']
        if opts.qualification is not None:
            paths.append(opts.qualification)
        for path in paths:
            artifact.add_file(str(path))
        run.log_artifact(artifact)
    run.summary['source_artifact_upload'] = opts.wandb_upload == 'source_artifact'
    if training_only:
        run.summary['independent_validation_required'] = False
        run.summary['independent_acceptance_completed'] = False
        run.summary['training_course_completed'] = False
    print(f'[V{spec.version}] W&B {run.url}', flush=True)
    retention = (CheckpointRetention(opts.save_dir, cleanup_every=5)
                 if opts.checkpoint_retention == 'm12_best' else None)
    quality_window = deque(maxlen=32)
    entry_update, hold, last_eval = updates, 0, updates-100
    pose_entry_update = updates
    pose_resume_updates = getattr(stack, 'pose_resume_updates', 0)
    from fixed_plane_wind_pose import ramp_complete,exposure_elapsed_steps
    pose_ready = ramp_complete(getattr(stack,'pose_resume_progress_steps',0)/128.,spec.course[stage])
    from fixed_plane_wind_objectives import RegressionGuard
    health_guard=RegressionGuard(spec.version,wind_m_s=spec.course[stage].max_speed_m_s) if training_only else None
    from fixed_plane_wind_strong import ExpansionGate,quality_gate as make_quality_gate
    if reference_mode:
        original_quality_gate = make_quality_gate
        def make_quality_gate(version,lesson,native):
            base = original_quality_gate(version,lesson,native)
            return make_reference_gate(base,version) if getattr(lesson,'strong_wind_reward',False) else base
    from fixed_plane_wind_progression import pose_growth_status,growth_hold_updates
    expansion=ExpansionGate(required_hold=growth_hold_updates(spec.course[stage]))
    run_start_update = updates
    domain_valid = np.ones(1024, bool)
    row, evaluation = {}, None

    def windows():
        lesson = spec.course[stage]
        if training_only:
            return TrainingSegments(lesson.seconds, env.dt, 1024, grouped=grouped,
                                    quality_fields=quality_fields), None, None
        return (EndpointWindow(lesson.seconds, env.dt),
                WindEndpointWindow(lesson.seconds, env.dt),
                ConditionEndpointWindow(lesson.seconds, env.dt) if grouped else None)

    endpoint, wind_endpoints, conditions = windows()
    collect, update = make_train_functions(stack, env, args)

    def save(kind):
        lesson = spec.course[stage]
        args.resume_start_stage = lesson.name
        path = opts.save_dir/(f'periodic_u{updates:08d}.pkl'
            if kind == 'periodic' and retention is not None else
            f'{kind}_s{stage+1:02d}_u{updates:08d}.pkl')
        if path.exists():
            raise FileExistsError(path)
        score = checkpoint_selection_score(stage,
            make_quality_gate(spec.version,lesson,stack.helpers.quality_passed)(row.get('quality', {})), row.get('endpoint', {}),
            float(np.asarray(runner.running_length).mean())*env.dt,
            quality=project_quality(row.get('quality',{})) if hasattr(getattr(env,'cfg',None),'wind_pose') else None,
            version=spec.version,wind_m_s=lesson.max_speed_m_s if hasattr(getattr(env,'cfg',None),'wind_pose') else 0.)
        if reference_mode and getattr(lesson,'strong_wind_reward',False):
            score = reference_checkpoint_score(score,row.get('quality',{}),spec.version)
        extra = dict(stage_index=stage, stage_name=lesson.name,
            stage_update=pose_resume_updates+updates-pose_entry_update, global_update=updates,
            stage_converged=kind == 'converged', wind_contract=contract,
            last_summary=row, independent_evaluation=evaluation,
            final_acceptance_passed=bool(kind == 'converged' and stage == len(spec.course)-1),
            checkpoint_kind=kind, selection_score=score,
            selection_kind='mixed_policy_training_rank')
        if training_only:
            extra.update(completion_metadata(stage, len(spec.course), converged=kind == 'converged'))
            extra['pose_stage_updates'] = pose_resume_updates+updates-pose_entry_update
            extra['pose_ramp_progress_steps']=(int(np.asarray(exposure_elapsed_steps(
                runner.env_state,env.cfg.wind_pose,xp=jnp))[0])
                if hasattr(getattr(env,'cfg',None),'wind_pose') else 0)
        if grouped:
            anchor = stack.parent['stage_actor_anchor_reference_params']
            if env.obs_dim == 114:
                from fixed_plane_wind_history import expand_reference_params
                anchor = expand_reference_params(anchor, actor_only=True)
            extra['stage_actor_anchor_reference_params'] = anchor
        from fixed_plane_wind_online_recipe import checkpoint_fields
        extra.update(checkpoint_fields(stack))
        if hasattr(stack, 'precontact_reward_birth_update'):
            extra['precontact_reward_birth_update'] = stack.precontact_reward_birth_update
        save_wind_checkpoint(stack, path, train_state, args, env, global_step, extra=extra)
        if kind == 'periodic' and retention is not None:
            retention.register(path, score)
            run.summary['best_periodic_checkpoint'] = str(retention.best)
        print(f'[V{spec.version} save] {path.name}; periodic_kept='
              f'{len(retention.saved) if retention is not None else "all"}', flush=True)

    try:
        if training_only:
            jax.block_until_ready((runner, train_state))
            release_caches()
        while not stop[0]:
            started, previous_t = time.monotonic(), int(train_state.opt.t)
            runner, transitions = collect(train_state.params, runner)
            train_state, losses = update(train_state, runner, transitions)
            loss = {k: float(v) for k, v in jax.device_get(losses).items()}
            if int(train_state.opt.t)-previous_t != 32 or not all(np.isfinite(v) for v in loss.values()):
                raise FloatingPointError('Wind PPO requires all32 finite Adam minibatches')
            updates += 1
            global_step += 131072
            pose_stage_updates = pose_resume_updates+updates-pose_entry_update
            gated_pose=hasattr(getattr(env,'cfg',None),'wind_pose') and env.cfg.wind_pose.quality_gated
            credited_steps=(int(np.asarray(exposure_elapsed_steps(runner.env_state,
                env.cfg.wind_pose,xp=jnp))[0]) if hasattr(getattr(env,'cfg',None),'wind_pose')
                else pose_stage_updates*128)
            reached_pose = ramp_complete(credited_steps/128., spec.course[stage])
            pose_just_completed = reached_pose and not pose_ready
            pose_ready = reached_pose
            done, terminated = (np.asarray(v, bool) for v in (transitions.done, transitions.terminated))
            length, hits = map(np.asarray, (transitions.episode_length, transitions.hit_count))
            recent = np.asarray(transitions.metrics[prefix+'/recent_hit']) > .5
            mode_labels = np.asarray(transitions.metrics['wind/mode'])
            if training_only:
                from fixed_plane_v322_condition_gates import condition_ids
                labels = (condition_ids(np.asarray(transitions.metrics['v322/initial_group']),
                    np.asarray(transitions.metrics['v322/initial_y_m'])) if grouped else None)
                endpoint.add(hits=np.asarray(transitions.metrics['hit_event_count']),
                    done=done, terminated=terminated, recent=recent, modes=mode_labels,
                    conditions=labels, quality_metrics={name: np.asarray(transitions.metrics[name])
                                                        for name in quality_fields})
            else:
                valid, domain_valid = current_domain_mask(done, domain_valid)
                lengths = np.where(valid, length, -1)
                ended = done & valid
                endpoint.add(lengths, hits, ended, recent, terminated=terminated)
                nominal = (np.asarray(transitions.metrics['v322/initial_group']) == 0
                           if grouped else np.ones_like(valid))
                wind_endpoints.add(np.where(nominal, lengths, -1), hits, ended & nominal,
                                   recent, mode_labels, terminated=terminated)
                if grouped:
                    conditions.add(lengths, hits, ended, recent,
                        np.asarray(transitions.metrics['v322/initial_group']),
                        np.asarray(transitions.metrics['v322/initial_y_m']), terminated=terminated)
            means = {k.removeprefix(stack.ppo.COMPACT_METRIC_MEAN_PREFIX): float(v)
                for k, v in jax.device_get(jax.tree.map(jnp.mean, transitions.metrics)).items()
                if k.startswith(stack.ppo.COMPACT_METRIC_MEAN_PREFIX)}
            if means.get(prefix+'/finite') != 1. or not all(np.isfinite(v) for v in means.values()):
                raise FloatingPointError('Nonfinite wind-domain simulation metrics')
            quality_window.append(means)
            if pose_just_completed:
                # This rollout still contains the final ramp ticks: discard it
                # as well, so every admitted sample uses the target bound.
                quality_window.clear()
                endpoint, wind_endpoints, conditions = windows()
                entry_update = updates
                print(f'[V{spec.version} pose ramp] complete at u={updates}; '
                      'fresh final-angle training segments and hold window', flush=True)
            quality = quality_summary({k: sum(m[k] for m in quality_window)*131072
                                       for k in quality_fields})
            outcome = endpoint.summary()
            wind_summary = endpoint.wind_summary() if training_only else wind_endpoints.summary()
            group_summary = (endpoint.condition_summary() if training_only
                             else conditions.summary() if grouped else None)
            lesson = spec.course[stage]
            current_quality_gate=make_quality_gate(spec.version,lesson,stack.helpers.quality_passed)
            eligible = (len(quality_window) == 32 and outcome['n'] >= 1024
                and (updates-entry_update)*.64 >= lesson.seconds
                and current_quality_gate(quality)
                and endpoints_passed(wind_summary, lesson, spec.origin)
                and (not grouped or condition_gates_passed(group_summary, condition_gate_settings())))
            if training_only:
                wind_quality = endpoint.wind_quality_summary()
                decision = training_gate_decision(window_size=len(quality_window), quality=quality,
                    stage_age_s=(updates-entry_update)*.64, lesson=lesson, origin=spec.origin,
                    outcome=outcome, wind=wind_summary, wind_quality=wind_quality,
                    quality_gate=current_quality_gate, conditions=group_summary,
                    group_limits=group_limits(mix))
                eligible = decision['passed']
                if not pose_ready:
                    decision['failures'].append('pose_ramp_incomplete')
                    decision['passed'] = eligible = False
                health=(health_guard.add(project_quality(quality),stage_updates=pose_stage_updates,
                    window_size=len(quality_window),per_wind={k:project_quality(v) for k,v in wind_quality.items()})
                    if hasattr(getattr(env,'cfg',None),'wind_pose') else
                    dict(stop=False,streak=0,severe_metrics=[]))
                if health['stop']:
                    keep_learning=getattr(lesson,'continue_on_quality_regression',False)
                    health={**health,'would_stop':True,'stop':not keep_learning,'course_hold':keep_learning}
                    if not keep_learning:
                        stop[0]=True
                    decision['failures'].append('persistent_quality_regression')
                    decision['passed']=eligible=False
                    run.summary['quality_regression_hold' if keep_learning else 'quality_regression_stop']=health
                    if not keep_learning or health['streak']==16 or updates%100==0:
                        print(f'[V{spec.version} quality protection] {health}',flush=True)
                if hasattr(getattr(env,'cfg',None),'wind_pose'):
                    if means.get('wind/entry_force_clipped',0.)>.0001:
                        decision['failures'].append('wind_force_clipping')
                        decision['passed']=eligible=False
                    block_quality = quality_summary({k: means[k]*131072 for k in quality_fields})
                    block_wind = {name: quality_summary(values)
                                  for name,values in (endpoint.qualities[-1] if endpoint.qualities else {}).items()}
                    growth_status = pose_growth_status(version=spec.version,lesson=lesson,
                        gate=current_quality_gate,rolling=project_quality(quality),
                        per_wind={k:project_quality(v) for k,v in wind_quality.items()},
                        latest=project_quality(block_quality),
                        latest_per_wind={k:project_quality(v) for k,v in block_wind.items()})
                    quality_ok = growth_status['passed']
                    can_grow=expansion.add(quality_ok=quality_ok,window_size=len(quality_window),
                                           physical_failure=bool(terminated.any()) or means.get('wind/entry_force_clipped',0.)>.0001)
                    if gated_pose:
                        runner=runner._replace(env_state=runner.env_state._replace(
                            wind_pose_ramp_enabled=jnp.full(1024,can_grow and not pose_ready and not stop[0],dtype=jnp.bool_)))
                hold, ready = advance_hold(hold, decision, stopping=stop[0])
            else:
                hold = hold+1 if eligible else 0
            row = dict(update=updates, stage=stage+1, quality=quality, endpoint=outcome,
                       wind=wind_summary, conditions=group_summary)
            if training_only:
                row.update(training_gate_status=decision, wind_quality=wind_quality,
                           quality_regression_guard=health,
                           completed_training_segments=endpoint.completed_segments,
                           censored_training_segments=endpoint.censored_segments)
                if hasattr(getattr(env,'cfg',None),'wind_pose'):
                    row['pose_growth_quality'] = dict(passed=quality_ok,
                        latest_block=block_quality, latest_per_wind=block_wind,
                        status=growth_status, required_hold=expansion.required_hold)
            flat = dict(loss, update=updates, global_step=global_step, stage=stage+1,
                **{'convergence/hold': hold, 'convergence/eligible': int(eligible),
                   'sps': 131072/(time.monotonic()-started)})
            if training_only:
                for index, name in enumerate(('clean', 'constant', 'slow', 'gust')):
                    flat[f'sampling/{name}_step_fraction'] = float(np.mean(mode_labels == index))
                for name, status in decision.get('groups', {}).items():
                    flat[f'convergence/{name}_passed'] = int(status['passed'])
                flat['pose/stage_updates'] = pose_stage_updates
                flat['pose/qualified_updates'] = credited_steps/128.
                flat['pose/growth_hold'] = expansion.streak
                flat['pose/growth_hold_required'] = expansion.required_hold
                flat['pose/ramp_complete'] = int(pose_ready)
                flat['pose/time_driven'] = int(hasattr(getattr(env,'cfg',None),'wind_pose')
                    and bool(env.cfg.wind_pose.ramp_updates) and not gated_pose)
                flat['quality_guard/streak']=health['streak']
                flat['quality_guard/stop']=int(health['stop'])
                flat['quality_guard/course_hold']=int(health.get('course_hold',False))
                flat['quality_guard/would_stop']=int(health.get('would_stop',False))
                flat['pose/exposure_updates']=credited_steps/128.
                if hasattr(getattr(env,'cfg',None),'wind_pose'):
                    flat['pose/growth_quality_passed'] = int(quality_ok)
                    flat.update({f'pose_block_quality/{k}': v for k,v in block_quality.items()})
            for namespace, values in (('quality', quality), ('endpoint', outcome), ('metric', means)):
                flat.update({f'{namespace}/{k}': v for k, v in values.items()})
            for name, values in wind_summary.items():
                flat.update({f'wind_endpoint/{name}/{k}': v for k, v in values.items()})
            if training_only:
                flat['convergence/failed_gate_count'] = len(decision['failures'])
                flat['convergence/validation_required'] = 0
                flat.update({f'wind_quality/{name}/{k}': v for name, values in wind_quality.items()
                             for k, v in values.items()})
            if grouped:
                flat.update({f'condition/{name}/{k}': v
                    for name, values in group_summary.items() for k, v in values.items()})
            if done.any():
                flat.update({'completed/mean_hits': float(hits[done].mean()),
                             'completed/mean_s': float(length[done].mean()*env.dt)})
                flat.update({f'completed/hit{k}': float((hits[done] >= k).mean()) for k in (1,3,12)})
            run.log(flat)
            blockers = (' blockers='+','.join(decision['failures'])
                        if training_only and not eligible else '')
            print(f'[V{spec.version}] S{stage+1}/{len(spec.course)} u={updates} '
                  f'hold={hold}/16 wind={wind_summary} quality={quality} Adam+32 '
                  f'gates_passed={int(eligible)}{blockers}'
                  +(f' pose_credit={credited_steps/128.:.2f}/{getattr(lesson,"pose_ramp_updates",0)}'
                    f' growth_hold={expansion.streak}/{expansion.required_hold}' if training_only and hasattr(getattr(env,'cfg',None),'wind_pose') else ''), flush=True)
            del transitions
            if updates % 100 == 0:
                save('periodic')
            accepted = (ready and progression_decision(spec.version, hold, eligible, None)
                        if training_only else False)
            if not training_only and hold >= 16 and updates-last_eval >= 100:
                jax.block_until_ready((runner, train_state))
                release_caches()
                evaluation = {}
                reset_seed = opts.seed+100000+stage*100
                domains = ((False, 'wind'), (True, 'clean'))
                for quiet, name in domains:
                    evaluation[name] = evaluate_policy(stack, args, train_state.params, stage,
                        reset_seed, stop, quiet=quiet)
                    run.log({'update': updates, f'validation/{name}/passed': int(evaluation[name]['passed'])})
                    if not evaluation[name]['passed']:
                        break
                (opts.runtime/f'validation_u{updates:08d}.json').write_text(json.dumps(evaluation, indent=2))
                last_eval = updates
                release_caches()
                accepted = (set(evaluation) == {'wind','clean'}
                            and all(item['passed'] for item in evaluation.values()))
            if accepted:
                save('converged')
                if training_only:
                    print(f'[V{spec.version} training advance] S{stage+1}/{len(spec.course)} '
                          f'u={updates}; all gates and hold passed; no independent validation', flush=True)
                stage += 1
                if stage == len(spec.course):
                    if training_only:
                        run.summary['training_course_completed'] = True
                    break
                if training_only:
                    jax.block_until_ready((runner, train_state))
                    del collect, update, env
                    release_caches()
                env, runner = transition_domain(stack, runner, stage)
                if env.obs_dim == 114:
                    from fixed_plane_wind_history import expand_history_learner
                    train_state = expand_history_learner(train_state)
                from fixed_plane_wind_online_recipe import enter_stage
                enter_stage(stack, train_state.params, stage)
                collect, update = make_train_functions(stack, env, args)
                quality_window.clear()
                domain_valid[:] = False
                endpoint, wind_endpoints, conditions = windows()
                entry_update, hold, last_eval = updates, 0, updates-100
                pose_entry_update, pose_resume_updates = updates, 0
                pose_ready = ramp_complete(0, spec.course[stage])
                health_guard=RegressionGuard(spec.version,wind_m_s=spec.course[stage].max_speed_m_s) if training_only else None
                expansion=ExpansionGate(required_hold=growth_hold_updates(spec.course[stage]))
                evaluation = None
            temperature = float(subprocess.check_output(['nvidia-smi', '-i', spec.origin.gpu_uuid,
                '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'], text=True))
            if temperature >= 78:
                stop[0] = True
            if (getattr(opts, 'probe_updates', 0)
                    and updates-run_start_update >= opts.probe_updates):
                stop[0] = True
        if stop[0] and stage < len(spec.course):
            if retention is None:
                save('interrupted')
            elif updates % 100 or not retention.saved:
                save('periodic')
    finally:
        try:
            if retention is not None:
                retention.prune()
            if training_only:
                release_caches()
        finally:
            run.finish()
