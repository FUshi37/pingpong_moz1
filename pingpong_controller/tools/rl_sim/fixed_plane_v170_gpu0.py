"""Reward-only centering continuation from the stopped GPU0 noise lesson."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v170'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v167'
PARENT_STAGE = 'fplc_v167_gpu0_s8_obs_bridge_2mm'
PARENT_STEP = 2447638528
PARENT_UPDATE = 59
PARENT_SHA256 = 'eb1813335eead4377c8e52f3b2970c19321c2014fca81898809850e74048aca5'
WEIGHTS = {
    'fixed_plane_v168_quality_weights': (30.0, 100.0, 10.0, 20.0, 0.0),
    'hit_next_contact_anchor_penalty_weight': 24.0,
    'hit_next_contact_anchor_sigma_m': .010,
    'fixed_plane_v165_steady_band_violation_weight': 2000.0,
}


def build_stages(source):
    """Preserve all fourteen lessons and gates; reweight only noise and DR."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace('v169_', 'v170_', 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 7:
            cfg = replace(cfg, **WEIGHTS)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V170 rewards the existing predicted return location while allowing small corrective ball velocity; '
            'existing center, late motion, adjacent apex and outgoing ball statistics. '
            'All V167 gates, launch/survival, control, domain and PPO remain.')))

    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Restore pinned full state and preserve exact own continuation history."""
    if (reset_optimizer or reset_critic or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (156, 477, 7)
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('V170 requires full 156/477/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE
                or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or resume_curriculum_state
                or requested_start_stage != 'fplc_v170_gpu0_s8_obs_bridge_2mm'):
            raise ValueError('V170 pins V167 noise-stage u59 and fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V170 own continuation requires exact stage/history')
    else:
        raise ValueError('V170 rejects another profile')
