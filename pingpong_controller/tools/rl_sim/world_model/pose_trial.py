"""Isolated FP32 pose reward trial; historical runtimes stay unchanged."""

from dataclasses import replace
import json
from pathlib import Path

from .contracts import authenticate, config_hash, file_hash, repository_root
from .paired_precision import signature as paired_signature
from .pose_margin import PoseMarginContract, penalty, roll_pitch_rad


def with_pose_reward(runtime, contract: PoseMarginContract):
    """Change only reward after a trusted 5 ms transition, including terminal rows."""
    if not contract.enabled or contract.weight_per_s == 0.:
        return runtime

    def step(*args, **kwargs):
        after, transition = runtime.course_step(*args, **kwargs)
        angles = roll_pitch_rad(after.world.cached_racket_pose,
                                runtime.control.projection.reference_rotation)
        extra = penalty(angles, transition.valid, transition.model_truncated,
                        .005, contract).astype(transition.reward.dtype)
        return after, transition._replace(reward=transition.reward + extra,
            reward_terms=dict(transition.reward_terms, pose_margin=extra))

    return replace(runtime, course_step=step)


def phase_plan(cfg: dict) -> dict:
    trial = cfg['pose_margin_trial']
    plan = json.loads(authenticate(repository_root(), trial['plan']).read_text())
    if (not trial['authorized'] or not plan['authorized'] or trial['default_enabled']
            or plan['maximum_actor_updates'] != 10 or plan['critic_updates_per_actor'] != 4
            or plan['mode'] != 'fp32' or plan['contract'] != trial['contract']
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or cfg['pose_margin_candidate']['contract']['enabled']
            or config_hash(cfg['imagination_implementation']['reward']['coefficients'])
            != plan['default_reward_sha256']):
        raise ValueError('Pose trial differs from its authorized contract')
    return plan


def signature(cfg: dict, evidence: dict, microbatch: int) -> str:
    names = ('pose_trial.py', 'train_pose_trial.py', 'pose_margin.py')
    return config_hash(dict(schema='wmj_v1_pose_margin_trial_runtime',
        paired_parent=paired_signature(cfg, 'fp32', evidence, microbatch),
        plan=cfg['pose_margin_trial']['plan'], contract=cfg['pose_margin_trial']['contract'],
        sources={n: file_hash(Path(__file__).parent / n) for n in names}))


def verify_gate(gate: dict, plan_reference: dict, evidence: dict, sig: str,
                microbatch: int, contract: dict) -> None:
    """Reject a gate from another reward, initial pool, implementation or budget."""
    if (gate['status'] != 'POSE_TRIAL_NO_UPDATE_PASS' or gate['plan'] != plan_reference
            or gate['mode'] != 'fp32' or gate['selected_microbatch'] != microbatch
            or gate['runtime_signature'] != sig or gate['candidate_contract'] != contract
            or any(gate[name] != 0 for name in ('actor_updates', 'critic_updates', 'world_updates'))
            or not gate['initial_learning_unchanged']
            or any(gate['runtime'][name] != evidence[name] for name in
                   ('inherited_learning_sha256', 'initial_session_sha256'))):
        raise ValueError('Invalid or incomparable pose trial no-update gate')


def load_actor(cfg: dict, reference: dict) -> tuple:
    from .imagination_checkpoint import restore_checkpoint
    from .migration import expand_actor_history, load_v321_actor, _validate_actor
    path = authenticate(repository_root(), reference)
    payload = json.loads(path.read_text()); meta = payload['metadata']
    if (meta['kind'] != 'pose_margin_trial_actor' or meta['actor_updates'] != 30
            or meta['new_actor_updates'] != 10 or meta['new_critic_updates'] != 40
            or meta['teacher_transitions_used_for_learning'] != 0 or meta['interrupted']
            or meta['plan'] != cfg['pose_margin_trial']['plan']
            or meta['candidate_contract'] != cfg['pose_margin_trial']['contract']):
        raise ValueError('Unexpected pose trial actor provenance')
    template = dict(actor=expand_actor_history(load_v321_actor(
        repository_root(), cfg['lineage']['actor_source'])))
    actor = restore_checkpoint(reference, template, signature=payload['runtime_signature'])['actor']
    _validate_actor(actor, 969)
    return actor, meta
