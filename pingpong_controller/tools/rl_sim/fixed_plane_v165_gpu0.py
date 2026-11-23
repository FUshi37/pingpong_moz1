"""Reward-only centering continuation from the stopped GPU0 noise lesson."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v165'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v161'
PARENT_STAGE = 'fplc_v161_gpu0_s8_obs_bridge_2mm'
PARENT_STEP = 2410938368
PARENT_UPDATE = 868
PARENT_SHA256 = 'ba917159fc46261a3e8058c0accf53ad9a93e2a89938558e4601dbbdbef08f2a'
WEIGHTS = {
    'fixed_plane_v159_mean_center_penalty_weight': 6.0,
    'fixed_plane_v158_late_squared_vxy_weight': 36.0,
    'fixed_plane_v158_late_squared_omega_weight': 2.0,
    'fixed_plane_v156_adjacent_height_penalty_multiplier': 1.5,
    'fixed_plane_v92_signed_height_reward_scale': 11.8733440561136,
    'fixed_plane_v164_dense_ball_xy_anchor_weight': 3.5,
    'fixed_plane_v164_dense_racket_xy_anchor_weight': 3.5,
    'fixed_plane_v165_steady_band_violation_weight': 500.0,
}


def build_stages(source):
    """Preserve all fourteen lessons and gates; reweight only noise and DR."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace('v161_', 'v165_', 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 7:
            from mjx_juggle_env import MjxFixedPlaneV165RewardConfig
            cfg = MjxFixedPlaneV165RewardConfig(**vars(cfg))
            cfg = replace(cfg, **WEIGHTS)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V165 rejects observed motion/centering trade-offs by complete reward repricing '
            'and explicitly prices late physical height-band violations. '
            'Every V161 gate, launch/survival, control, domain and PPO remains.')))
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Restore pinned full state and preserve exact own continuation history."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (156, 477, 7)
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('V165 requires full 156/477/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE
                or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or resume_curriculum_state
                or requested_start_stage != 'fplc_v165_gpu0_s8_obs_bridge_2mm'):
            raise ValueError('V165 pins V161 noise-stage u868 and fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V165 own continuation requires exact stage/history')
    else:
        raise ValueError('V165 rejects another profile')
