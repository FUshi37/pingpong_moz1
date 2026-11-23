"""V254: submitted clean V172 with measured v1.5 camera-position AR(1) noise."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from mjx_juggle_env import MjxFixedPlaneV159RewardConfig

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_v254"
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v172"
PARENT_SHA256 = "96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a"
PARENT_STAGE = "fplc_v159_gpu1_s10_clean_commit"
START_STAGE = "fplc_v254_s11_camera_ar1_1x"
MEAN_M = (.000048, .000072, -.000227)
STD_M = (.001451, .001418, .002970)
RHO_AT_90HZ = (.935, .907, .790)
NOISE_FIELDS = (
    "ball_position_noise_ar1_enabled", "ball_position_noise_mean_m",
    "ball_position_noise_std_m", "ball_position_noise_rho_at_90hz",
)
STAGE_LABELS = (
    "camera_ar1_1x", "camera_ar1_1x_consolidation", "plant_quarter",
    "plant_half", "mass_small", "target_hold_small", "small_dr_consolidation",
)
STAGE_NAMES = tuple(f"fplc_v254_s{number}_{label}"
                    for number, label in enumerate(STAGE_LABELS, 11))
# Optional state/critic extension; legacy registries and 393-D layout stay exact.
DOMAIN_RANDOMIZED_STATE_FIELDS = frozenset({"ball_position_noise_epsilon_m"})
CRITIC_STATE_FIELDS = DOMAIN_RANDOMIZED_STATE_FIELDS


@dataclass(frozen=True)
class CameraPositionNoiseConfig(MjxFixedPlaneV159RewardConfig):
    """Opt-in stationary 1x camera noise; no new fields on legacy configs."""

    ball_position_noise_ar1_enabled: bool = True
    ball_position_noise_mean_m: tuple[float, float, float] = MEAN_M
    ball_position_noise_std_m: tuple[float, float, float] = STD_M
    ball_position_noise_rho_at_90hz: tuple[float, float, float] = RHO_AT_90HZ


def validate_noise_config(cfg: CameraPositionNoiseConfig) -> None:
    """Fail closed on changed calibration or stacked observation errors."""
    for name, expected in zip(NOISE_FIELDS[1:], (MEAN_M, STD_M, RHO_AT_90HZ), strict=True):
        values = np.asarray(getattr(cfg, name), dtype=float)
        if values.shape != (3,) or not np.array_equal(values, expected):
            raise ValueError("V254 pins the measured v1.5 camera noise at exactly 1x")
    if (not cfg.fixed_plane_v149_simple_obs_enabled
            or not cfg.fixed_plane_v99_contiguous_history_obs_enabled
            or not cfg.asymmetric_critic or not cfg.critic_complete_dr_privileged
            or cfg.critic_fixed_plane_task_context
            or cfg.ball_obs_rate_hz not in (90., 120., 200.)
            or not cfg.ball_obs_fractional_rate
            or cfg.ball_obs_pos_noise_std != 0. or cfg.ball_obs_vel_noise_std != 0.
            or cfg.ball_obs_vel_xy_noise_std != 0.
            or cfg.ball_obs_posthit_vel_xy_noise_std != 0.
            or cfg.dr_randomize_ball_obs_frame
            or cfg.dr_randomize_latency or cfg.dr_post_hit_ball_obs_latency
            or cfg.ball_obs_dropout_prob != 0. or cfg.ball_obs_dropout_burst_prob != 0.
            or cfg.ball_obs_require_camera_visible or cfg.ball_obs_require_view_bounds
            or cfg.ball_obs_require_view_z_high or cfg.ball_obs_reset_respects_camera_visibility
            or tuple(cfg.ball_obs_nominal_pos_bias_base) != (0., 0., 0.)
            or tuple(cfg.ball_obs_nominal_vel_bias_base) != (0., 0., 0.)
            or bool(getattr(cfg, "plant_aware_fixed_plane_v1_enabled", False))
            or cfg.fixed_contact_high_ball_predictor_enabled):
        raise ValueError("V254 requires72-D history, complete critic and unstacked fresh position noise")


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep ten clean lessons, replace sensing DR, retain five physical lessons."""
    if len(source) != 17 or source[9].name != PARENT_STAGE:
        raise ValueError("V254 requires the complete preserved V172 course")
    result = list(source[:10])
    clean = source[9].cfg
    # Replace the entire old sensing ladder, including its dormant range fields.
    sensing_fields = (
        "ball_obs_pos_noise_std", "ball_obs_vel_noise_std", "dr_randomize_ball_obs_frame",
        "dr_ball_obs_pos_bias_base_m", "dr_ball_obs_rot_bias_deg",
        "dr_ball_obs_vel_bias_base_m_s", "dr_ball_obs_scale_range",
    )
    for stage, name in zip(source[10:], STAGE_NAMES, strict=True):
        cfg = CameraPositionNoiseConfig(**vars(stage.cfg))
        cfg = replace(cfg, **{field: getattr(clean, field) for field in sensing_fields},
                      fixed_contact_training_phase=name)
        validate_noise_config(cfg)
        result.append(replace(stage, name=name, cfg=cfg, notes=(
            "V254 restores submitted clean V172S10u314; stationary 1x v1.5 XYZ "
            "position AR(1) noise is applied before causal history caching. "
            "S11/S12 adapt and consolidate at 90Hz; S13--S17 retain original "
            "physical DR order and all gates. No legacy sensing DR is stacked. "
            "Actor72 retained; critic393 prefix plus3 noise latents. Truth, "
            "reward, collision and control unchanged. No training evidence yet.")))
    return result


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
    allow_obs_dim_migration: bool,
) -> None:
    """Require pinned V172 entry or exact complete-state V254 continuation."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))):
        raise ValueError("V254 preserves actor, critic and complete Adam state")
    source = payload.get("args", {}).get("curriculum_profile")
    dims = tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
    if source == PARENT_PROFILE:
        if (dims != (72, 393, 7) or not allow_obs_dim_migration
                or resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_index") != 10 or payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != 314 or payload.get("step") != 1334837248
                or int(payload["opt"].t) != 335771):
            raise ValueError("V254 pins clean V172S10u314, critic-only migration and fresh history")
        for key in ("convergence/stage_converged", "advance_eval/passed", "retention_eval/passed"):
            if float(payload.get("last_row", {}).get(key, 0.)) != 1.:
                raise ValueError("V254 parent must have passed complete V172 graduation")
    elif source == PROFILE:
        if (dims != (72, 396, 7) or allow_obs_dim_migration
                or payload.get("args", {}).get("max_stage_updates") != -1):
            raise ValueError("V254 own resume requires full72/396/7 uncapped state without migration")
        index = payload.get("stage_index", 0)
        if (not isinstance(index, int) or not 11 <= index <= 17
                or payload.get("stage_name") != STAGE_NAMES[index - 11]):
            raise ValueError("V254 requires its exact registered noise stage")
        if resume_curriculum_state:
            if requested_start_stage != payload.get("stage_name"):
                raise ValueError("V254 own resume requires exact stage and causal history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V254 rejects another source profile")
