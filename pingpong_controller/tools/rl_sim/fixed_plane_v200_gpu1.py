"""Reward-only workspace stability continuations from pre-regression S14."""

from dataclasses import replace
from typing import Any, Mapping, Sequence

PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v199"
PARENT_STAGE = "fplc_v191_gpu1_s14_plant_half_height_cadence"
PARENT_SHA = "1ffe39bd716794ce21d039fdd8fccb723e0f3a47b49e77ee8ca373fb690d1b17"
PARENT_UPDATE = 1500
PARENT_STEP = 2521169920
PARENT_ADAM_T = 614684
ADJACENT_WEAK_PROFILES = tuple(
    f"fixed_plane_limit_cycle_qvel_rmp_gpu1_v{v}" for v in (213, 214, 215, 216))
ADJACENT_WEAK_MULTIPLIERS = dict(zip(
    ADJACENT_WEAK_PROFILES, (6.0, 7.5, 3.0, 2.25), strict=True))
WEAK_OUTCOME_PROFILES = tuple(
    f"fixed_plane_limit_cycle_qvel_rmp_gpu1_v{v}" for v in (211, 212)) + ADJACENT_WEAK_PROFILES
PROFILES = tuple(f"fixed_plane_limit_cycle_qvel_rmp_gpu1_v{v}" for v in (200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210)) + WEAK_OUTCOME_PROFILES
REWARD_FIELDS = (
    "fixed_contact_workspace_z_barrier_weight",
    "fixed_contact_workspace_z_barrier_downward_lookahead_s",
    "fixed_contact_survival_reward_weight",
    "full_episode_completion_reward",
    "full_episode_completion_min_hits",
    "fixed_plane_v156_adjacent_height_penalty_multiplier",
    "fixed_plane_v191_absolute_height_multiplier",
    "fixed_plane_v191_margin_weight",
    "fixed_plane_v211_late_weak_penalty_weight",
)
REWARD_DEFAULTS = {"fixed_plane_v191_absolute_height_multiplier": 1.,
                   "fixed_plane_v191_margin_weight": 0.,
                   "fixed_plane_v211_late_weak_penalty_weight": 0.}


def validate_reward_only_config(before: Mapping[str, Any], after: Mapping[str, Any]) -> None:
    """Check added and removed fields as well as changed existing fields."""
    absent = object()
    changed = {key for key in before.keys() | after.keys()
               if before.get(key, absent) != after.get(key, absent)}
    if changed - set(REWARD_FIELDS):
        raise ValueError("Only explicitly registered reward fields may change")


def start_stage(profile: str) -> str:
    """Return an explicit versioned S14; never infer checkpoint stage identity."""
    if profile not in PROFILES:
        raise ValueError("Unregistered stability profile")
    return f"fplc_v{profile.rsplit('v', 1)[1]}_gpu1_s14_plant_half_stability_reward"


def build_stages(source: Sequence[Any], *, profile: str) -> list[Any]:
    """Preserve every physical/gate field; change only late reward settings."""
    if profile not in PROFILES or len(source) != 17 or source[13].name != PARENT_STAGE:
        raise ValueError("Complete V199 course required")
    if profile in ADJACENT_WEAK_PROFILES:
        base = build_stages(source,
            profile="fixed_plane_limit_cycle_qvel_rmp_gpu1_v212")
        multiplier = ADJACENT_WEAK_MULTIPLIERS[profile]
        version = profile.rsplit("v", 1)[1]
        result = list(base[:13])
        for stage in base[13:]:
            cfg = replace(stage.cfg,
                fixed_plane_v156_adjacent_height_penalty_multiplier=multiplier)
            result.append(replace(stage, cfg=cfg,
                name=stage.name.replace("fplc_v212_", f"fplc_v{version}_"),
                notes=f"V{version} retains all V212 rewards and changes only "
                    f"the adjacent-height penalty multiplier4.5->{multiplier:g}. "
                    "The bounded60-point late weak-resolution cost, absolute "
                    "height1.5 and workspace320/60ms rewards remain exact. "
                    "All17-stage gates, first13 stages, physics/control/RMP/PD/"
                    "DR/reset/PPO/BC and72/477/7 full-state restore are preserved. "
                    "Both complete physical height-pair quality and weak counts "
                    "must pass the preregistered paired non-regression checks. "
                    "This opt-in reward hypothesis is unproven."))
        return result
    if profile in WEAK_OUTCOME_PROFILES:
        from mjx_juggle_env import MjxFixedPlaneV211RewardConfig
        base_version, weight = (207, 30.) if profile == WEAK_OUTCOME_PROFILES[0] else (210, 60.)
        base = build_stages(source,
            profile=f"fixed_plane_limit_cycle_qvel_rmp_gpu1_v{base_version}")
        version = profile.rsplit("v", 1)[1]
        result = list(base[:13])
        for stage in base[13:]:
            cfg = replace(MjxFixedPlaneV211RewardConfig(**vars(stage.cfg)),
                fixed_plane_v211_late_weak_penalty_weight=weight)
            result.append(replace(stage, cfg=cfg,
                name=stage.name.replace(f"fplc_v{base_version}_", f"fplc_v{version}_"),
                notes=f"V{version} retains V{base_version}'s reward combination and "
                    f"adds only a bounded{weight:g}-point late weak-resolution cost. "
                    "It uses existing low-survival/subfloor flags after3 confirmed "
                    "contacts. Early acquisition, physical event classification, "
                    "hit accounting, all other rewards, all17-stage gates, "
                    "physics/control/RMP/PD/DR/reset/PPO/BC and72/477/7 full-state "
                    "restore remain exact. This opt-in hypothesis is unproven."))
        return result
    result = list(source[:13])
    version = profile.rsplit("v", 1)[1]
    for stage in source[13:]:
        cfg = stage.cfg
        if profile in PROFILES[9:11]:
            from mjx_juggle_env import MjxFixedPlaneV191RewardConfig
            cfg = replace(MjxFixedPlaneV191RewardConfig(**vars(cfg)),
                fixed_contact_workspace_z_barrier_weight=(240. if profile == PROFILES[9] else 320.),
                fixed_contact_workspace_z_barrier_downward_lookahead_s=(.030 if profile == PROFILES[9] else .060),
                fixed_plane_v191_absolute_height_multiplier=1.5,
                fixed_plane_v156_adjacent_height_penalty_multiplier=4.5)
        elif profile in PROFILES[7:9]:
            from mjx_juggle_env import MjxFixedPlaneV191RewardConfig
            cfg = replace(MjxFixedPlaneV191RewardConfig(**vars(cfg)),
                fixed_contact_workspace_z_barrier_weight=(240. if profile == PROFILES[7] else 320.),
                fixed_contact_workspace_z_barrier_downward_lookahead_s=(.030 if profile == PROFILES[7] else .060),
                fixed_plane_v191_absolute_height_multiplier=1.25)
        elif profile in PROFILES[5:7]:
            cfg = replace(cfg,
                fixed_contact_workspace_z_barrier_weight=(160. if profile == PROFILES[5] else 240.),
                fixed_contact_workspace_z_barrier_downward_lookahead_s=.030)
        elif profile == PROFILES[3]:
            cfg = replace(cfg, fixed_contact_workspace_z_barrier_weight=240.,
                fixed_contact_workspace_z_barrier_downward_lookahead_s=.040,
                full_episode_completion_reward=100., full_episode_completion_min_hits=11)
        elif profile != PROFILES[0]:
            cfg = replace(cfg, fixed_contact_workspace_z_barrier_weight=320.,
                fixed_contact_workspace_z_barrier_downward_lookahead_s=.060)
            if profile == PROFILES[2]:
                cfg = replace(cfg, fixed_contact_survival_reward_weight=60.)
            elif profile == PROFILES[4]:
                cfg = replace(cfg, fixed_plane_v156_adjacent_height_penalty_multiplier=4.5)
        result.append(replace(stage, cfg=cfg,
            name=stage.name.replace("fplc_v191_gpu1_", f"fplc_v{version}_gpu1_")
                .replace("_height_cadence", "_stability_reward"),
            notes="V200 is an unchanged-reward matched control. V201 raises "
                  "only site-height workspace barrier160->320 and reward-only "
                  "downstroke prediction20->60ms. V202 additionally raises "
                  "post-first-hit survival40->60 per second. V203 uses "
                  "barrier240/40ms and adds completion100 for6s with>=11hits, "
                  "without any maximum-hit cap. V204 retains V201 and "
                  "raises only adjacent-apex penalty multiplier3->4.5. V205 uses "
                  "the original160 weight with30ms forecast, V206 uses240/30ms; "
                  "neither changes height/survival/completion rewards. V207 retains "
                  "V206, V208 retains V201; each reweights only absolute signed-height "
                  "credit1->1.25, preserving adjacent3 and unresolved losses. The "
                  "added V191 margin field stays0/off. First13 stages, "
                  "all gates, physics/control/RMP/PD, DR/reset/PPO/BC and "
                  "actor72/critic477 remain exact. No action/state filtering, "
                  "hard-bound changes or critic/Adam reset; no forced advance."))
        if profile in PROFILES[9:11]:
            result[-1] = replace(result[-1], notes=
                "V209 retains V206's240/30ms workspace reward; V210 retains "
                "V201's320/60ms. Both multiply absolute height1->1.5 and "
                "linear adjacent-height penalty3->4.5 together, preserving "
                "their original relative balance. Unresolved loss, center, "
                "survival and completion rewards remain unchanged. The V191 "
                "margin field stays0/off. First13 stages, the full17-stage "
                "course, all gates, physical/control/RMP/PD/DR/reset/PPO/BC "
                "and72/477/7 full actor/critic/Adam remain exact. No forced "
                "advance or source-state reset; this is an unproven opt-in probe.")
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, profile: str,
                            requested_start_stage: str, resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool) -> None:
    """Pin pre-regression full state; prohibit promotion of bounded actors."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or payload.get("opt") is None
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 477, 7)
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or payload.get("args", {}).get("max_stage_updates") != -1):
        raise ValueError("Stability repair requires full formal72/477/7 and Adam")
    source = payload["args"]["curriculum_profile"]
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != start_stage(profile)
                or payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != PARENT_UPDATE
                or payload.get("step") != PARENT_STEP
                or int(payload["opt"].t) != PARENT_ADAM_T):
            raise ValueError("Stability repair pins pre-regression V199 S14/u1500")
    elif source == profile:
        if resume_curriculum_state:
            if payload.get("stage_name") != requested_start_stage:
                raise ValueError("Own continuation requires exact saved stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("Reject unrelated or already-regressed source")
