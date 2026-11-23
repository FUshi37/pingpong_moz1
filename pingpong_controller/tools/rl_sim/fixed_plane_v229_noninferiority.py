"""V229 uses original racket bounds plus matched frozen-reference stability."""
from dataclasses import replace
from typing import Any, Mapping, Sequence

from fixed_plane_v228_motion_alignment import (
    PARENT_PROFILE, PARENT_SHA256,
)

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v229'
START_STAGE = 'fplc_v229_s01_workspace_alignment'
# Finalized from the preregistered source-only CUDA population before training.
REPLAY_SHA256 = '2741d1898620b62d227b0ee45a4f73aae6691503e2bb1e41054ed0761f079cf7'
ORIGINAL_RACKET_LIMITS = {
    'max_recent_mean_hit_racket_vxy': .065,
    'max_recent_rapid_steady_racket_vxy_rms_m_s': .053,
    'max_recent_rapid_steady_racket_omega_rms_rad_s': .28,
}


def validate_anchor_contract(settings: Any, *, source_digest: str,
                             replay_digest: str) -> None:
    """Freeze demonstrated S5 skill, preserving old corrective BC separately."""
    from fixed_plane_v227_nominal_reference import validate_anchor_contract as validate
    if len(REPLAY_SHA256) != 64:
        raise ValueError('Successful S5 replay collection has not passed')
    validate(settings, source_digest=source_digest, replay_digest=replay_digest,
             expected_replay=REPLAY_SHA256)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Retain every lesson/reward; remove only unrequested extra improvements."""
    if len(source) != 17 or source[0].name != 'fplc_v228_s01_workspace_alignment':
        raise ValueError('V229 requires the complete preserved V228 course')
    return [replace(stage, **ORIGINAL_RACKET_LIMITS,
        name=stage.name.replace('fplc_v228_', 'fplc_v229_', 1),
        notes='V229 retains all17 recovery lessons,72/393/7, V228 rewards, '
        'control, physics and DR. Keep original V225 absolute racket bounds '
        '0.065/0.053m/s and0.28rad/s. Require mandatory same-domain/same-RNG '
        'frozenS5 nominal retention on all five ball/racket speed metrics, '
        'with no positive margin. Ball mean/RMS caps0.0224277973/0.0259883013, '
        'hit/height/survival/view/recovery gates stay unchanged. Reachable '
        'early center cost, ball XY weight2, late XY/Omega weights '
        '2.5948680958444186/2.593570971629849 and whole-course source replay '
        'KL1/bound0.006 remain. Preservation replay is independently sampled '
        'successful on-policy S5; original failed-state corrective BC remains. '
        'No extra historical-window racket improvement '
        'is mistaken for a non-degradation requirement.') for stage in source]


def validate_paired_stability_contract(settings: Any) -> None:
    """Original bounds are valid only with mandatory blocking paired retention."""
    if (settings.advance_mode != 'converged'
            or settings.advance_validation_mode != 'block'
            or settings.advance_retention_stage_name != START_STAGE
            or settings.advance_retention_final_stage_name not in (None, '')
            or settings.advance_eval_deterministic
            or settings.advance_eval_n_envs != 288
            or settings.advance_eval_steps != 1200):
        raise ValueError('V229 requires blocking 288x1200 matched nominal S5 retention')


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool,
                            allow_bounded_probe_start: bool = False) -> None:
    """Start from passed S5; neither reinterpret nor promote bounded trials."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        from fixed_plane_v228_motion_alignment import validate_resume_payload as validate_s5
        if requested_start_stage != START_STAGE or allow_bounded_probe_start:
            raise ValueError('V229 initial continuation must start at S1')
        validate_s5(payload,
            requested_start_stage='fplc_v228_s01_workspace_alignment',
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic,
            allow_obs_dim_migration=allow_obs_dim_migration)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V229 rejects capped/incompatible/reset continuation')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V229 requires exact stage and causal history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
