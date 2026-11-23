"""Height/survival graduation and isolated upper-tail reward continuations."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v183"
PARENT_STAGE = "fplc_v159_gpu1_s13_plant_quarter"
PARENT_UPDATE = 1736
PARENT_STEP = 1832910848
PARENT_SHA = "8fadaa3fd6dce75c730b393ce965b4eed122fb2b2f062ea1b7057b58abf62958"
BASELINE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v184"
EVENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v185"
SHOULDER_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v186"
BALANCED_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v187"
HIT_CREDIT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v188"
STRONG_EVENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v189"
SYMMETRIC_BAND_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v190"
PROFILES = (BASELINE_PROFILE, EVENT_PROFILE, SHOULDER_PROFILE, BALANCED_PROFILE,
            HIT_CREDIT_PROFILE, STRONG_EVENT_PROFILE, SYMMETRIC_BAND_PROFILE)
REWARD_FIELDS = ("fixed_plane_v184_upper_event_weight",
                 "fixed_plane_v184_upper_shoulder_weight",
                 "fixed_plane_v92_signed_height_reward_scale",
                 "hit_count_floor_reward_weight",
                 "fixed_plane_v165_steady_band_violation_weight")
GATE_FIELDS = ("target_mean_hits", "target_hit12_rate", "target_mean_len_frac")


def start_stage(profile: str) -> str:
    """Return the explicit versioned S13 identity; never guess saved progress."""
    if profile not in PROFILES:
        raise ValueError("Unregistered height/survival continuation")
    version = profile.rsplit("v", 1)[1]
    return f"fplc_v{version}_gpu1_s13_plant_quarter_height_survival"


def build_stages(source: Sequence[Any], *, profile: str) -> list[Any]:
    """Preserve the first twelve stages and every physical/observation field."""
    if profile not in PROFILES or len(source) != 17:
        raise ValueError("Height/survival continuation requires the complete V183 course")
    from mjx_juggle_env import MjxFixedPlaneV184RewardConfig, MjxFixedPlaneV190RewardConfig
    version = profile.rsplit("v", 1)[1]
    result = list(source[:12])
    for stage in source[12:]:
        cfg = stage.cfg
        if profile == SYMMETRIC_BAND_PROFILE:
            cfg = replace(MjxFixedPlaneV190RewardConfig(**vars(cfg)),
                fixed_plane_v165_steady_band_violation_weight=150.)
        elif profile == HIT_CREDIT_PROFILE:
            cfg = replace(cfg, hit_count_floor_reward_weight=15.)
        elif profile == BALANCED_PROFILE:
            cfg = replace(cfg, fixed_plane_v92_signed_height_reward_scale=9.)
        elif profile != BASELINE_PROFILE:
            cfg = replace(MjxFixedPlaneV184RewardConfig(**vars(cfg)),
                fixed_plane_v184_upper_event_weight=(
                    150. if profile == STRONG_EVENT_PROFILE else 40.),
                fixed_plane_v184_upper_shoulder_weight=(
                    8. if profile == SHOULDER_PROFILE else 0.))
        identity = stage.name.replace("fplc_v159_gpu1_", f"fplc_v{version}_gpu1_")
        result.append(replace(stage, cfg=cfg,
            name=identity + "_height_survival",
            target_mean_hits=0., target_hit12_rate=None, target_mean_len_frac=1.,
            notes="V184 family prioritizes full survival and the unchanged 1.400--1.450 m "
                  "fourth-plus physical-height gates. Mean hits/P12 remain diagnostics. "
                  "Basic hit1/hit3, physical-event sample counts, center, adjacent-height, "
                  "motion, view, safety, 32 consecutive/block and S9 checks remain. "
                  "V185 adds only upper-event cost40; V186 adds quadratic shoulder8; "
                  "V187 alone scales the existing symmetric 1.425m height reward6->9. "
                  "V188 alone halves uniform confirmed-hit credit30->15. "
                  "V189 alone increases V185's upper event cost40->150. "
                  "V190 isolates symmetric out-of-band cost150 with original center rewards. "
                  "Actor72/critic477, RMP/PD, physics, DR and PPO remain V183-exact. "
                  "Experimental simulation-only; not hardware-coverage evidence."))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, profile: str,
                            requested_start_stage: str, resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool) -> None:
    """Restore full state with fresh history after reward/gate contract changes."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 477, 7)):
        raise ValueError("Height repair requires full72/477/7 and unchanged Adam")
    source = payload.get("args", {}).get("curriculum_profile")
    if source == PARENT_PROFILE:
        if (payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != PARENT_UPDATE
                or payload.get("step") != PARENT_STEP
                or payload.get("stop_reason") != "received SIGINT"
                or resume_curriculum_state
                or requested_start_stage != start_stage(profile)):
            raise ValueError("Pinned interrupted V183 S13 and fresh gate history required")
    elif source == profile:
        if payload.get("args", {}).get("max_stage_updates") != -1:
            raise ValueError("Do not promote a bounded probe actor as an own formal continuation")
        if resume_curriculum_state:
            if payload.get("stage_name") != requested_start_stage:
                raise ValueError("Own continuation requires exact saved stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("Reject unrelated or rejected reward-candidate checkpoint")
