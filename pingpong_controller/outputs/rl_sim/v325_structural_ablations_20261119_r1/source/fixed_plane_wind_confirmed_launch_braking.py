"""Restrict added steady-recovery braking to confirmed launches."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import verify_source as verify_s16_source
from fixed_plane_wind_steady_braking import steady_braking_contract


MODE = 'clean10_dynamics_reference_center_negy_v12'
SOURCE_UPDATE = 14900
SOURCE_STEP = 3562668032
SOURCE_ADAM_T = 979375
SOURCE_SHA256 = 'f3fd48a5eb644bd2ed4bf80adaf6bac7cec681636d6ea57a96884444fe00a605'


def confirmed_launch_braking_contract() -> dict:
    return {
        **steady_braking_contract(),
        'scope': 'steady recovery after native launch confirmation, with neither edge pending',
        'preserved_original_reward_phase_steps': 8192,
        'source': dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                       true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
    }


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindConfirmedLaunchBrakingConfig', [
        ('wind_confirmed_launch_braking', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_confirmed_launch_braking_config(cfg: Any, enabled: bool) -> Any:
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def after_launch_confirmation(active: Any, pending_before: Any, pending_after: Any) -> Any:
    """Leave the entire pending-launch transition to the original reward."""
    return active & ~pending_before & ~pending_after


def verify_source(payload: dict, sha256: str) -> int:
    """Authenticate the original mature learner, or this recipe's continuation."""
    verify_s16_source(
        payload, sha256, mode=MODE, config_flag='wind_confirmed_launch_braking',
        source=(SOURCE_UPDATE, SOURCE_STEP, SOURCE_ADAM_T, SOURCE_SHA256),
    )
    return min(8192, (payload['global_update'] - SOURCE_UPDATE) * 128)
