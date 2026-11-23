"""Reward-only centering continuation from the stopped GPU0 noise lesson."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v167'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v166'
PARENT_STAGE = 'fplc_v166_gpu0_s8_obs_bridge_2mm'
PARENT_STEP = 2439905280
PARENT_UPDATE = 65
PARENT_SHA256 = '096742cc6224f18d43bb67c1e6eddd77ac494531c51cebbefeb8f0d9fa4a4a54'
WEIGHTS = {
    'racket_flatness_penalty_weight': 36.0,
    'racket_flatness_angle_deadband_rad': .0005,
    'fixed_plane_v167_global_reward_multiplier': 1.1173893422045234,
}


def build_stages(source):
    """Preserve all fourteen lessons and gates; reweight only noise and DR."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace('v166_', 'v167_', 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 7:
            from mjx_juggle_env import MjxFixedPlaneV167RewardConfig
            cfg = MjxFixedPlaneV167RewardConfig(**vars(cfg))
            cfg = replace(cfg, **WEIGHTS)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V167 prices small racket tilt continuously and rescales all reward terms uniformly '
            'without changing relative legacy motion, height, center or survival prices. '
            'Every V166 gate, launch/survival, control, domain and PPO remains.')))
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Restore pinned full state and preserve exact own continuation history."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (156, 477, 7)
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('V167 requires full 156/477/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE
                or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or resume_curriculum_state
                or requested_start_stage != 'fplc_v167_gpu0_s8_obs_bridge_2mm'):
            raise ValueError('V167 pins V166 noise-stage u65 and fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V167 own continuation requires exact stage/history')
    else:
        raise ValueError('V167 rejects another profile')
