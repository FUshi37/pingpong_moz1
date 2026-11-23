"""Preserve the verified actor/rewards and calibrate advancement to measured quality."""
from dataclasses import replace

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v171'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_gpu0_v167'
PARENT_SHA256 = 'eb1813335eead4377c8e52f3b2970c19321c2014fca81898809850e74048aca5'
GATE_FIELDS = (
    'min_updates', 'max_recent_mean_hit_contact_center_dist_m',
    'max_recent_rapid_steady_racket_vxy_rms_m_s',
    'max_recent_rapid_steady_racket_omega_rms_rad_s',
    'max_recent_late_adjacent_jump_m', 'min_recent_late_physical_in_band_rate',
)


def build_stages(source):
    """Retain all stage identities and every environment/reward value exactly."""
    result = []
    for index, stage in enumerate(source):
        gates = {}
        if index == 6:
            gates = dict(max_recent_mean_hit_contact_center_dist_m=.00305,
                max_recent_rapid_steady_racket_vxy_rms_m_s=.0305,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.205,
                max_recent_late_adjacent_jump_m=.0070,
                min_recent_late_physical_in_band_rate=.998)
        elif index == 7:
            gates = dict(min_updates=30,
                max_recent_mean_hit_contact_center_dist_m=.0045,
                max_recent_rapid_steady_racket_vxy_rms_m_s=.033,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.210,
                max_recent_late_adjacent_jump_m=.0073,
                min_recent_late_physical_in_band_rate=.998)
        elif index >= 8:
            gates = dict(max_recent_mean_hit_contact_center_dist_m=.005,
                max_recent_rapid_steady_racket_vxy_rms_m_s=.037,
                max_recent_rapid_steady_racket_omega_rms_rad_s=.225)
        result.append(replace(stage, **gates, notes=(
            'V171 preserves the verified V167 full actor/rewards and stable stage identities. '
            'Only gates are calibrated under explicit user authorization; independent clean '
            'retention and physical height/survival validation remain required.')))
    return result


def validate_resume_payload(payload, *, requested_start_stage,
                            resume_curriculum_state, reset_optimizer,
                            reset_critic):
    """Preserve full actor/critic/Adam and causal metrics for unchanged rewards."""
    if (reset_optimizer or reset_critic
            or payload.get('opt') is None
            or (payload.get('obs_dim'), payload.get('critic_obs_dim'),
                payload.get('act_dim')) != (156, 477, 7)
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('V171 requires full156/477/7 state and exact causal history')
    source = payload.get('args', {}).get('curriculum_profile')
    if source == PARENT_PROFILE:
        if (not resume_curriculum_state
                or payload.get('stage_name') != requested_start_stage
                or payload.get('stage_name') != 'fplc_v167_gpu0_s8_obs_bridge_2mm'
                or payload.get('stage_update') != 59
                or payload.get('step') != 2447638528):
            raise ValueError('V171 pins the verified V167u59 actor without reinterpretation')
    elif source == PROFILE:
        if not resume_curriculum_state:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
        elif payload.get('stage_name') != requested_start_stage:
            raise ValueError('V171 continuation requires exact stage/history')
    else:
        raise ValueError('V171 rejects another source profile')
