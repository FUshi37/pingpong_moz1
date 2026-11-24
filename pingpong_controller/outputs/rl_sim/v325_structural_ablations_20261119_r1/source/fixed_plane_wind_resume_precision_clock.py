"""Reward-only context preserving a verified v2 checkpoint's mature feedback."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)

MODE = 'clean10_dynamics_reference_center_negy_v16'
REWARD_POLICY_STEPS_OFFSET = 8192


def mature_precision_contract() -> dict:
    """Describe the reward clock independently of physical elapsed time."""
    return dict(base=BASE_MODE, reward_policy_steps_offset=REWARD_POLICY_STEPS_OFFSET,
        scope='original center and clean apex precision credits',
        source_reward_phase='verified fully mature at the pinned source checkpoint',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
        physical_time='retain the native reset clock',
        coefficients='retain original v2 amplitudes, scores and event masks')


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindMaturePrecisionConfig', [
        ('wind_mature_precision_reward', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_mature_precision_config(cfg: Any, enabled: bool) -> Any:
    """Keep all historical configuration values and types when disabled."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def reward_steps(physical_steps: Any, *, enabled: bool) -> Any:
    """Supply mature precision context without writing any environment state."""
    return physical_steps + REWARD_POLICY_STEPS_OFFSET if enabled else physical_steps


def verify_mature_source(payload: dict, sha256: str) -> None:
    """Accept the pinned original learner and this recipe's full continuations."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_mature_precision_reward')
