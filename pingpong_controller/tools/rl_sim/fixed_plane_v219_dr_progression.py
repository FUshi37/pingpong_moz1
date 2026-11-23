"""Opt-in DR graduation calibrated to stable physical juggling evidence."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

from fixed_plane_v200_gpu1 import (
    PARENT_PROFILE, PARENT_STAGE, PARENT_UPDATE, PARENT_STEP, PARENT_ADAM_T,
)

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v219"
BASE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v215"
START_STAGE = "fplc_v219_gpu1_s14_plant_half_stability_reward"
GATE_FIELDS = (
    "target_mean_len_frac",
    "min_recent_late_physical_in_band_rate",
    "max_recent_late_physical_below_floor_rate",
    "max_recent_late_physical_above_ceiling_rate",
    "max_recent_late_adjacent_jump_m",
    "max_recent_mean_hit_contact_center_dist_m",
)
BRIDGE_GATES = dict(zip(GATE_FIELDS, (.98, .88, .05, .10, .0115, .0075), strict=True))
FINAL_GATES = dict(zip(GATE_FIELDS, (.99, .90, .04, .08, .011, .007), strict=True))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep V215 dynamics/rewards and all seventeen domains in order."""
    if len(source) != 17 or source[13].name != START_STAGE.replace("v219", "v215"):
        raise ValueError("V219 requires the complete immutable V215 course")
    result = list(source[:13])
    for index, stage in enumerate(source[13:], start=13):
        result.append(replace(stage,
            name=stage.name.replace("fplc_v215_", "fplc_v219_"),
            **(FINAL_GATES if index == 16 else BRIDGE_GATES),
            notes="V219 calibrates only six late DR graduation fields. "
                  "Physical apex target1.425m and band1.400--1.450m, "
                  "99.5% full episodes, hits11--15, all safety/view/motion "
                  "limits, strict independent gates,32-update hold, complete "
                  "stochastic block validation and S9 retention remain. "
                  "S14--S16 use98% length/88% physical height-band rate; "
                  "S17 consolidates at99%/90%. V215 rewards, DR support, "
                  "physics/control/reset/observation/PPO/BC stay exact. "
                  "Experimental simulation-only; no forced advancement."))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Restart the formal parent; bounded probes remain evidence-only."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 477, 7)
            or payload.get("args", {}).get("max_stage_updates") != -1):
        raise ValueError("V219 requires full formal72/477/7 actor/critic/Adam")
    source = payload["args"]["curriculum_profile"]
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != PARENT_UPDATE
                or payload.get("step") != PARENT_STEP
                or int(payload["opt"].t) != PARENT_ADAM_T):
            raise ValueError("V219 pins original V199 S14/u1500 with fresh gate history")
    elif source == PROFILE:
        if resume_curriculum_state:
            if requested_start_stage != payload.get("stage_name"):
                raise ValueError("Own continuation requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V219 rejects unrelated or bounded reward-probe sources")
