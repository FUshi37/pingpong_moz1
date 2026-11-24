"""Extend the existing vertical release credit to the calm population."""

from dataclasses import field, make_dataclass
from functools import lru_cache
from typing import Any

from fixed_plane_wind_gate_precision import (
    VERTICAL_RELEASE_INTERVAL_CAP_S, VERTICAL_RELEASE_RATE_PER_S,
    VERTICAL_RELEASE_SIGMA_M_S, vertical_release_credit,
)
from fixed_plane_wind_measured_precision import (
    BASE_MODE, SOURCE_ADAM_T, SOURCE_SHA256, SOURCE_STEP, SOURCE_UPDATE,
    verify_source,
)


MODE = 'clean10_dynamics_reference_center_negy_v14'
WARMUP_STEPS = 8192


def calm_release_contract() -> dict:
    """Record the independent reward-only extension of the original recipe."""
    return dict(
        base=BASE_MODE, sigma_m_s=VERTICAL_RELEASE_SIGMA_M_S,
        rate_per_s=VERTICAL_RELEASE_RATE_PER_S,
        release_interval_cap_s=VERTICAL_RELEASE_INTERVAL_CAP_S,
        target='reuse theoretical vertical velocity from the physical release height for 1.425 m apex',
        timing='live valid physical releases in clean mode with the original shelter and elapsed release clock',
        warmup_policy_steps=WARMUP_STEPS, original_reward_clock='unchanged',
        source=dict(sha256=SOURCE_SHA256, update=SOURCE_UPDATE,
                    true_step=SOURCE_STEP, Adam_t=SOURCE_ADAM_T, stage_index=15),
    )


@lru_cache(maxsize=16)
def _config_type(parent: type) -> type:
    return make_dataclass('WindCalmReleaseConfig', [
        ('wind_calm_release', bool, field(default=True)),
    ], bases=(parent,), frozen=True)


def with_calm_release_config(cfg: Any, enabled: bool) -> Any:
    """Preserve the original configuration object on every other recipe."""
    return _config_type(type(cfg))(**vars(cfg)) if enabled else cfg


def calm_release_credit(metrics: dict, *, calm: Any, shelter_gain: Any,
                        release_elapsed_s: Any, phase_steps: Any, xp: Any):
    """Use the existing vertical reward at clean releases with a separate ramp."""
    progress = xp.clip(phase_steps / WARMUP_STEPS, 0., 1.)
    warmup = progress ** 2 * (3. - 2. * progress)
    return vertical_release_credit(metrics, gain=calm * shelter_gain * warmup,
                                   release_elapsed_s=release_elapsed_s, xp=xp)


def calm_release_phase(payload: dict, sha256: str) -> int:
    """Authenticate the original full learner or a compatible continuation."""
    verify_source(payload, sha256, mode=MODE, config_flag='wind_calm_release')
    return min(WARMUP_STEPS, (payload['global_update'] - SOURCE_UPDATE) * 128)
