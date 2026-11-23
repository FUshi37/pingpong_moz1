"""Gate-only retention calibration for the complete V219 DR continuation."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v220"
BASE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v219"
START_STAGE = "fplc_v220_gpu1_s14_plant_half_stability_reward"
RETENTION_STAGE = "fplc_v159_gpu1_s9_motion_bridge"
RETENTION_GATES = {
    "target_mean_hits": 12.8,
    "min_recent_late_physical_in_band_rate": .975,
    "max_recent_late_physical_below_floor_rate": .0125,
    "max_recent_late_physical_above_ceiling_rate": .02,
    "max_recent_late_adjacent_jump_m": .0105,
    "max_recent_mean_hit_contact_center_dist_m": .0056,
}


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve all training gates and domains, including the original S9."""
    if len(source) != 17 or source[13].name != START_STAGE.replace("v220", "v219"):
        raise ValueError("V220 requires the complete immutable V219 course")
    return list(source[:13]) + [replace(stage,
        name=stage.name.replace("fplc_v219_", "fplc_v220_"),
        notes=stage.notes + " V220 alone calibrates S9 retention during S14--S17; "
              "actual S9 training and every main-domain gate stay unchanged.")
        for stage in source[13:]]


def retention_stage(stage: Any) -> Any:
    """Copy S9 for validation only; never mutate its environment or course."""
    if stage.name != RETENTION_STAGE:
        raise ValueError("V220 retention calibration requires the exact S9 domain")
    return replace(stage, **RETENTION_GATES)


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Retain actor, critic and Adam; changed retention earns fresh gates."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 477, 7)
            or payload.get("args", {}).get("max_stage_updates") != -1):
        raise ValueError("V220 requires full uncapped72/477/7 actor/critic/Adam")
    source = payload["args"]["curriculum_profile"]
    if source == BASE_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_name") != START_STAGE.replace("v220", "v219")
                or int(payload.get("stage_update", 0)) <= 0
                or int(payload.get("step", 0)) <= 2521169920):
            raise ValueError("V220 requires current formal V219 S14 and fresh gate history")
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get("stage_name"):
                raise ValueError("V220 continuation requires exact saved stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V220 rejects unrelated or bounded-probe sources")
