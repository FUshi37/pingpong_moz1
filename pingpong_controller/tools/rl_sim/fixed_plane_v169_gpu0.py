"""Reward-only centering continuation from the stopped GPU0 noise lesson."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v169'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v167'
PARENT_STAGE = 'fplc_v167_gpu0_s8_obs_bridge_2mm'
PARENT_STEP = 2447638528
PARENT_UPDATE = 59
PARENT_SHA256 = 'eb1813335eead4377c8e52f3b2970c19321c2014fca81898809850e74048aca5'
WEIGHTS = {
    'fixed_plane_v168_quality_weights': (30.0, 100.0, 5.0, 20.0, 70.0),
}


def build_stages(source):
    """Preserve all fourteen lessons and gates; reweight only noise and DR."""
    result = []
    for index, stage in enumerate(source):
        name = stage.name.replace('v168_', 'v169_', 1)
        cfg = replace(stage.cfg, fixed_contact_training_phase=name)
        if index >= 7:
            cfg = replace(cfg, **WEIGHTS)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            'V169 balances existing fixed-reference physical quality prices using the exact '
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
        raise ValueError('V169 requires full 156/477/7 actor/critic/Adam')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (payload.get('stage_name') != PARENT_STAGE
                or payload.get('step') != PARENT_STEP
                or payload.get('stage_update') != PARENT_UPDATE
                or resume_curriculum_state
                or requested_start_stage != 'fplc_v169_gpu0_s8_obs_bridge_2mm'):
            raise ValueError('V169 pins V167 noise-stage u59 and fresh reward history')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V169 own continuation requires exact stage/history')
    else:
        raise ValueError('V169 rejects another profile')
