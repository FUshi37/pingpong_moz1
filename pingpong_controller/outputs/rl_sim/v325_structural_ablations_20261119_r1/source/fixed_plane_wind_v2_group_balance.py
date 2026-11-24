"""Default-off group weighting directly over the original v2 rewards."""

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)
from fixed_plane_wind_normalized_priority import normalized_priority_contract

MODE = 'clean10_dynamics_reference_center_negy_v17'


def v2_balance_contract() -> dict:
    """Retain v2 component prices while testing normalized group importance."""
    return {
        **normalized_priority_contract(),
        'revision': 'original_v2_normalized_group_priority_v1',
        'base': BASE_MODE,
        'source': dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                       true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
        'clean_apex_rate_per_s': 40.,
        'independence': 'No tripled clean-height credit or mature-precision override',
    }


def resume_policy_steps(payload: dict, sha256: str) -> int:
    """Admit the pinned original source and complete same-recipe continuations."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_normalized_group_priority')
    cfg = payload.get('env_cfg', {})
    if cfg.get('wind_clean_apex_priority', False):
        raise ValueError('V2 group balance does not admit extra clean height priority')
    if cfg.get('wind_mature_precision_reward', False):
        raise ValueError('V2 group balance does not stack a different precision recipe')
    return min(8192, (payload['global_update'] - SOURCE_UPDATE) * 128)
