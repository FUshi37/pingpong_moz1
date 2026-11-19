"""Explicit V325/V326 online clean rehearsal without legacy action imitation."""
from __future__ import annotations

from dataclasses import replace
from typing import TypeVar


MIX = 'clean10_no_bc_v1'
DYNAMICS_MIX = 'clean10_dynamics_v1'
QUALITY_MIX = 'clean10_dynamics_quality_v1'
SAFE_MIX = 'clean10_dynamics_safe_v1'
FLIGHT_MIX = 'clean10_dynamics_flight_v1'
TAIL_MIX = 'clean10_dynamics_tail_v1'
RECOVERY_MIX = 'clean10_dynamics_recovery_v1'
FULL_FLIGHT_MIX = 'clean10_dynamics_recovery_v2'
LOWER_GUARD_MIX = 'clean10_dynamics_lower_guard_v1'
OPTIMIZER_MIX = 'clean10_dynamics_lower_history_v1'
FAST_OPTIMIZER_MIX = 'clean10_dynamics_lower_history_fast_v1'
SURVIVAL_MIX = 'clean10_dynamics_survival_precision_v1'
NEGATIVE_Y_MIX = 'clean10_dynamics_survival_precision_negy_v1'
GATE_PRECISION_MIX = 'clean10_dynamics_gate_precision_negy_v1'
POSITIVE_PRECISION_MIX = 'clean10_dynamics_gate_precision_negy_v2'
REFERENCE_CONSTRAINT_MIX = 'clean10_dynamics_reference_constraints_negy_v1'
CENTER_PRECISION_MIX = 'clean10_dynamics_reference_center_negy_v1'
CENTER_APEX_MIX = 'clean10_dynamics_reference_center_negy_v2'
FLIGHT_MODES = (FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX)
MODES = ('legacy', MIX, DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX)
FLIGHT_MODES += (GATE_PRECISION_MIX, POSITIVE_PRECISION_MIX, REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX)
MODES += (GATE_PRECISION_MIX, POSITIVE_PRECISION_MIX, REFERENCE_CONSTRAINT_MIX, CENTER_PRECISION_MIX, CENTER_APEX_MIX)
Lesson = TypeVar('Lesson')


def mix_contract(mode: str) -> dict | None:
    """Keep the historical objective selectable and record effective overrides."""
    if mode == 'legacy':
        return None
    if mode == CENTER_APEX_MIX:
        from fixed_plane_wind_center_precision_reward import clean_apex_contract
        return {**mix_contract(CENTER_PRECISION_MIX), 'revision':mode,
                'clean_apex_preservation':clean_apex_contract()}
    if mode == CENTER_PRECISION_MIX:
        from fixed_plane_wind_center_precision_reward import center_reward_contract
        return {**mix_contract(REFERENCE_CONSTRAINT_MIX), 'revision':mode,
                'center_precision_reward':center_reward_contract()}
    if mode == REFERENCE_CONSTRAINT_MIX:
        from fixed_plane_wind_reference_constraints import reference_contract
        return {**mix_contract(POSITIVE_PRECISION_MIX), 'revision':mode,
                'reference_constraints':reference_contract(),
                'quality':'wind excess over cached physical reference; clean raw; independent height/center RMS'}
    if mode == POSITIVE_PRECISION_MIX:
        from fixed_plane_wind_gate_precision import positive_contract
        return {**mix_contract(NEGATIVE_Y_MIX), 'revision': mode,
                'gate_precision': positive_contract()}
    if mode == GATE_PRECISION_MIX:
        from fixed_plane_wind_gate_precision import precision_contract
        return {**mix_contract(NEGATIVE_Y_MIX), 'revision': mode,
                'gate_precision': precision_contract()}
    if mode not in MODES:
        raise ValueError('Unknown wind training mixture')
    result = dict(revision=mode, legacy_bc_coefficient=0., clean_episode_fraction=.1,
        rehearsal='fresh online clean trajectories in the same PPO updates',
        wind_probabilities='rescale existing non-clean weights proportionally to0.9',
        sample_fraction='episode reset probability; log actual rollout step fractions',
        group_limits=dict(clean=dict(min_survival=.99, min_length_fraction=.99),
                          wind=dict(min_survival=.95, min_length_fraction=.99)),
        quality='original per-wind physical-quality limits; clean uses zero wind allowance',
        progression='all active populations plus original rolling32 and hold16')
    if mode in (DYNAMICS_MIX, QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX):
        from fixed_plane_wind_dynamics_guidance import guidance_contract
        result['dynamics_guidance'] = guidance_contract(
            quality_guard=mode in (QUALITY_MIX, SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
            safety_guard=mode in (SAFE_MIX, FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
            flight_alignment=mode in (FLIGHT_MIX, TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
            tail_precision=mode in (TAIL_MIX, RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX),
            recovery_safety=mode in (RECOVERY_MIX, FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX), full_flight=mode in (FULL_FLIGHT_MIX, LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX))
        if mode in (LOWER_GUARD_MIX, OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX):
            from fixed_plane_wind_lower_guard import lower_guard_contract
            result['dynamics_guidance']['lower_stop_guard'] = lower_guard_contract()
    if mode in (OPTIMIZER_MIX, FAST_OPTIMIZER_MIX, SURVIVAL_MIX, NEGATIVE_Y_MIX):
        from fixed_plane_wind_optimizer_conditioning import optimizer_contract
        result['optimizer_conditioning'] = optimizer_contract(fast=mode == FAST_OPTIMIZER_MIX)
    if mode in (SURVIVAL_MIX,NEGATIVE_Y_MIX):
        from fixed_plane_wind_survival_precision import survival_precision_contract
        result['survival_precision'] = survival_precision_contract()
    if mode == NEGATIVE_Y_MIX:
        from fixed_plane_wind_direction_domain import direction_domain_contract
        result['wind_direction_domain']=direction_domain_contract()
    return result


def group_limits(mode: str) -> dict | None:
    contract = mix_contract(mode)
    return None if contract is None else contract['group_limits']


def effective_lesson(lesson: Lesson, mode: str) -> Lesson:
    contract = mix_contract(mode)
    if contract is None:
        return lesson
    windy = sum(lesson.mode_weights[1:])
    if windy <= 0.:
        raise ValueError('Clean rehearsal requires a wind training population')
    weights = (.1, *(weight / windy * .9 for weight in lesson.mode_weights[1:]))
    return replace(lesson, mode_weights=weights)


def checkpoint_mix(payload: dict, version: int) -> str:
    """Reject a coefficient change without its explicit saved training contract."""
    args = payload.get('args', {})
    mode = args.get('wind_training_mix', 'legacy')
    contract = mix_contract(mode)
    stored = payload.get('wind_contract', {}).get('training_mix')
    if stored != contract:
        raise ValueError('Wind checkpoint training-mixture contract changed')
    if mode != 'legacy' and (version not in (325, 326)
                            or args.get('counterfactual_supervision_coef') != 0.):
        raise ValueError('No-BC rehearsal requires explicit V325/V326 zero-BC settings')
    return mode
