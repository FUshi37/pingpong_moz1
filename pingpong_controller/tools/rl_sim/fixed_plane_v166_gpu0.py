"""Reward-only centering continuation from the stopped GPU0 noise lesson."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v166'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v165'
PARENT_STAGE = 'fplc_v165_gpu0_s8_obs_bridge_2mm'
PARENT_STEP = 2431385600
PARENT_UPDATE = 156
PARENT_SHA256 = '80ef145d0998b18ab11d52c1a703f03a877c521085f9279e47d7b2c116a5fc78'
WEIGHTS = {
    'fixed_plane_v159_mean_center_penalty_weight': 14.0,
    'fixed_plane_v166_physical_ball_xy_weight': 12.0,
    'fixed_plane_v92_signed_height_reward_scale': 15.309336569170775,
}


def build_stages(source):
    """Preserve all fourteen lessons and gates; reweight only noise and DR."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace('v165_', 'v166_', 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 7:
            from mjx_juggle_env import MjxFixedPlaneV166RewardConfig
            cfg = MjxFixedPlaneV166RewardConfig(**vars(cfg))
            cfg = replace(cfg, **WEIGHTS)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V166 strengthens center guidance and directly prices small physical ball XY drift '
            'while retaining the complete V165 motion and height protections. '
            'Every V165 gate, launch/survival, control, domain and PPO remains.')))
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Restore pinned full state and preserve exact own continuation history."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (156, 477, 7)
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('V166 requires full 156/477/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE
                or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or resume_curriculum_state
                or requested_start_stage != 'fplc_v166_gpu0_s8_obs_bridge_2mm'):
            raise ValueError('V166 pins V165 noise-stage u156 and fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V166 own continuation requires exact stage/history')
    else:
        raise ValueError('V166 rejects another profile')
