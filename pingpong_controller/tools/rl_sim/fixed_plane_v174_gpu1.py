"""Opt-in reward-only centering and physical-height-tail repair."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v174"
CENTER_ONLY_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v175"
BASELINE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v176"
PROTECTED_CENTER_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v177"
MINIMUM_ONLY_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v178"
PROFILES = (
    PROFILE, CENTER_ONLY_PROFILE, BASELINE_PROFILE, PROTECTED_CENTER_PROFILE,
    MINIMUM_ONLY_PROFILE,
)
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v173"
PARENT_STAGE = "fplc_v159_gpu1_s12_obs_small_4mm"
PARENT_STEP = 1517813760
PARENT_UPDATE = 594
PARENT_SHA256 = "bff8261b139257e7f8a9d4b8cb19968b01e597ef1ba377c7f35ae148bc4d92f3"
REWARD_FIELDS = (
    "fixed_plane_v168_quality_weights",
    "fixed_plane_v168_quality_references",
    "fixed_plane_v165_steady_band_violation_weight",
)
# Measured, frozen source center; other references are inactive in these trials.
QUALITY_REFERENCES = (.006131369263223591, .053**2, .25**2, .007, .025**2)
PROTECTED_REFERENCES = (
    .006131369263223591, .04882409793958216**2,
    .22447837844736654**2, .00872005693098845, .026974840834010198**2,
)


def build_stages(source: Sequence[Any], *, profile: str = PROFILE) -> list[Any]:
    """Preserve all 17 lessons and gates, repricing only S12 and its tail."""
    if profile not in PROFILES:
        raise ValueError("Unregistered reward trial")
    from mjx_juggle_env import MjxFixedPlaneV168RewardConfig
    result = []
    for index, stage in enumerate(source):
        cfg = stage.cfg
        if index >= 11 and profile not in (BASELINE_PROFILE, MINIMUM_ONLY_PROFILE):
            cfg = MjxFixedPlaneV168RewardConfig(**vars(cfg))
            cfg = replace(cfg,
                fixed_plane_v168_quality_weights=(
                    (30., 5., 5., 5., 5.) if profile == PROTECTED_CENTER_PROFILE
                    else (3., 0., 0., 0., 0.)),
                fixed_plane_v168_quality_references=(
                    PROTECTED_REFERENCES if profile == PROTECTED_CENTER_PROFILE
                    else QUALITY_REFERENCES),
                fixed_plane_v165_steady_band_violation_weight=(
                    200. if profile in (PROFILE, PROTECTED_CENTER_PROFILE) else 0.),
            )
        result.append(replace(stage, cfg=cfg,
            min_updates=min(stage.min_updates, 30),
            notes=("Minimum-update-only continuation; exact parent rewards, "
                  "physical domain, quality gates and native history. "
                  "Minimum30 never bypasses rolling/block/retention checks."
                  if profile == MINIMUM_ONLY_PROFILE else
                  "Reward-only repair: fixed-reference physical center credit; "
                  "V174 also prices existing late out-of-band events. All "
                  "observation/control/domain/quality gates and hold are preserved. "
                  "Minimum updates30 never bypasses rolling/block/retention checks.")))
    return result


def validate_resume_payload(
    payload: Mapping[str, Any], *, profile: str, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
) -> None:
    """Keep actor, critic and Adam; reset stale reward history at transition."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or (payload.get("obs_dim"), payload.get("critic_obs_dim"),
                payload.get("act_dim")) != (72, 477, 7)):
        raise ValueError("Reward repair requires full72/477/7 actor/critic/Adam")
    source = payload.get("args", {}).get("curriculum_profile")
    if source == PARENT_PROFILE:
        preserve_history = profile == MINIMUM_ONLY_PROFILE
        if (resume_curriculum_state != preserve_history
                or requested_start_stage != PARENT_STAGE
                or payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != PARENT_UPDATE
                or payload.get("step") != PARENT_STEP):
            raise ValueError("V173 S12u594 requires history matching the reward transition")
    elif source == profile:
        if resume_curriculum_state:
            if payload.get("stage_name") != requested_start_stage:
                raise ValueError("Own continuation requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("Unrelated source or cross-trial reward history")
