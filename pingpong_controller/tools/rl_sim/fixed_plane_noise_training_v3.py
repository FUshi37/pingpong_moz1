"""Explicit training-only completion; physical training episodes stay1200s."""
from __future__ import annotations

import math

from fixed_plane_noise_validation_v2 import QUALITY_MAX
from fixed_plane_v321_long_episode import quality_passed
from fixed_plane_v323_ball_obs_noise import spec_for

PROTOCOL = 'training_quality_v3'


def protocol_contract(version: int) -> dict:
    """Retain current numeric quality limits without a frozen-test prerequisite."""
    spec_for(version)
    return dict(protocol=PROTOCOL, training_episode_s=1200.,
        window_updates=32, hold_updates=16, min_steady_events=100,
        quality_max=dict(QUALITY_MAX), apex_band_m=[1.4, 1.45], min_in_band=.99,
        independent_evaluation_required=False, restart1200s_clock=False,
        endpoint_and_condition_statistics='diagnostic mixed-policy telemetry',
        scope='training_only', deterministic_certified=False,
        final_acceptance_passed=False, hardware_coverage=False)


def validate_episode(horizon_s: float, dt_s: float, max_steps: int) -> None:
    """Reject a shortened physical training horizon, including launcher mistakes."""
    if horizon_s != 1200. or dt_s != .005 or max_steps != 240000:
        raise ValueError('Training-quality mode preserves1200s/240000-step episodes')


def training_quality_passed(quality: dict) -> bool:
    """Require every original field and a finite, sufficiently populated window."""
    fields = set(QUALITY_MAX) | {'apex_m', 'in_band', 'steady_events'}
    return (fields.issubset(quality)
        and all(math.isfinite(quality[name]) for name in fields)
        and quality_passed(quality))


class TrainingQualityWindow:
    """Gate event-weighted rolling summaries after32 real updates, hold16."""

    def __init__(self) -> None:
        self.updates = 0
        self.hold = 0

    def add(self, quality: dict) -> dict:
        self.updates += 1
        eligible = self.updates >= 32 and training_quality_passed(quality)
        self.hold = self.hold + 1 if eligible else 0
        return dict(qualified=self.hold >= 16, hold=self.hold,
            window_size=min(self.updates, 32), eligible=bool(eligible))


def completion_metadata(gate: dict, update: int) -> dict:
    """Training quality never rewrites a failed or absent frozen certification."""
    if (not gate.get('qualified') or gate.get('hold', 0) < 16
            or gate.get('window_size') != 32):
        raise ValueError('A complete rolling32/hold16 training pass is required')
    return dict(qualified=True, scope='training_only', protocol=PROTOCOL,
        update=update, window_updates=32, hold=gate['hold'],
        final_acceptance_passed=False, deterministic_certified=False,
        hardware_coverage=False)


def protected_checkpoint_name(update: int, *, qualified: bool) -> str:
    """Keep qualified and interruption snapshots outside periodic pruning."""
    prefix = 'training_qualified' if qualified else 'interrupted'
    return f'{prefix}_u{update:08d}.pkl'
