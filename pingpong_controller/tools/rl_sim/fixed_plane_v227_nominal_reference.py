"""V227 keeps V226 domains/rewards and bounds cumulative nominal-policy drift."""
from dataclasses import replace
from typing import Any, Callable, Mapping, Sequence

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v227'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v225'
PARENT_SHA256 = 'b102831e018585b7127668fcc40a8de258325e2a7a58551f02ddc3b6ddce2608'
REPLAY_SHA256 = '793575f1da785c295ed110a29e1d18052f4f9fa5edc14a332079e534fb210bc3'
START_STAGE = 'fplc_v227_s01_workspace_alignment'


def select_course_reference(profile: str, legacy_reference: Any,
                            source_loader: Callable[[], Any]) -> Any:
    """V227 alone retains the immutable S5 origin across stage advances."""
    return source_loader() if profile == PROFILE else legacy_reference


def validate_anchor_contract(settings: Any, *, source_digest: str,
                             replay_digest: str,
                             expected_source: str = PARENT_SHA256,
                             expected_replay: str = REPLAY_SHA256) -> None:
    """Fail closed on changed replay, source or reviewed optimizer bounds."""
    if (source_digest != expected_source or replay_digest != expected_replay
            or settings.actor_anchor_reference_checkpoint is None
            or settings.actor_anchor_replay_obs is None
            or settings.actor_anchor_kl_coef != 0.0
            or settings.actor_anchor_replay_kl_coef != 1.0
            or settings.actor_anchor_replay_max_kl != 0.006):
        raise ValueError('V227 pins S5, complete nominal replay, KL weight1/bound0.006')


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all 17 V226 lessons and numerical environment/gate settings."""
    if len(source) != 17 or source[0].name != 'fplc_v226_s01_workspace_alignment':
        raise ValueError('V227 requires the unchanged complete V226 course')
    return [replace(stage, name=stage.name.replace('fplc_v226_', 'fplc_v227_', 1),
                    notes=stage.notes + ' V227 additionally retains frozen S5 as '
                    'the whole-course nominal-replay KL origin; coefficient1, '
                    'hard cumulative bound0.006 using existing PPO backtracking. '
                    'The source origin never resets at stage transitions.')
            for stage in source]


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, allow_obs_dim_migration: bool,
                            allow_bounded_probe_start: bool = False) -> None:
    """Restore passed S5, never the regressed V225S6 or bounded V226/V227 actor."""
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        from fixed_plane_v226_stability_recovery import validate_resume_payload as validate_s5
        if not requested_start_stage.startswith('fplc_v227_'):
            raise ValueError('V227 entry must name a V227 lesson')
        validate_s5(payload,
            requested_start_stage=requested_start_stage.replace('fplc_v227_', 'fplc_v226_', 1),
            resume_curriculum_state=resume_curriculum_state,
            reset_optimizer=reset_optimizer, reset_critic=reset_critic,
            allow_obs_dim_migration=allow_obs_dim_migration,
            allow_bounded_probe_start=allow_bounded_probe_start)
        return
    if (source != PROFILE or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))
            or tuple(payload.get(k) for k in ['obs_dim', 'critic_obs_dim', 'act_dim']) != (72, 393, 7)
            or payload.get('args', {}).get('max_stage_updates') != -1):
        raise ValueError('V227 own continuation requires unreset uncapped full72/393/7')
    if resume_curriculum_state:
        if requested_start_stage != payload.get('stage_name'):
            raise ValueError('V227 requires exact stage/history on own resume')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload, next_stage=requested_start_stage)
