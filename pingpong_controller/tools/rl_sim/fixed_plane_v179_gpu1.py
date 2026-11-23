"""Reward-only balanced precision probes and evidence-calibrated continuation."""

from dataclasses import replace

BASELINE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v179"
BALANCED_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v180"
ANCHOR_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v181"
GATE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v182"
RETENTION_GATE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v183"
PROFILES = (BASELINE_PROFILE, BALANCED_PROFILE, ANCHOR_PROFILE, GATE_PROFILE,
            RETENTION_GATE_PROFILE)
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v178"
PARENT_STAGE = "fplc_v159_gpu1_s12_obs_small_4mm"
PARENT_SOURCES = {
    (1168, 1593049088): "cabfa787bec93cd62acf1afc0d59eb3a1e69c6f2b11d2095f47ee35b0b1f59e8",
    (719, 1534197760): "b8cc8bdac284abb15ea0d9e47fd3f8f771d9b24773619ab2484cad22f9b13716",
}
REWARD_FIELDS = (
    "fixed_plane_v168_quality_weights", "fixed_plane_v168_quality_references",
    "fixed_plane_v165_steady_band_violation_weight",
    "fixed_plane_v134_late_xy_anchor_penalty_weight",
)
GATE_FIELDS = (
    "max_recent_mean_hit_contact_center_dist_m",
    "max_recent_late_physical_below_floor_rate",
    "max_recent_late_physical_above_ceiling_rate",
    "min_recent_late_physical_in_band_rate",
)


def build_stages(source, *, profile):
    """Keep all 17 domains, observability and survival/motion/safety gates."""
    if profile not in PROFILES:
        raise ValueError("Unregistered precision/tail experiment")
    from mjx_juggle_env import MjxFixedPlaneV168RewardConfig
    result = []
    for index, stage in enumerate(source):
        cfg, gates = stage.cfg, {}
        if index >= 11 and profile in (BALANCED_PROFILE, ANCHOR_PROFILE):
            cfg = replace(MjxFixedPlaneV168RewardConfig(**vars(cfg)),
                fixed_plane_v168_quality_weights=(12., 0., 0., 0., 0.),
                fixed_plane_v168_quality_references=(.0064, .053**2, .28**2, .009, .025**2),
                fixed_plane_v165_steady_band_violation_weight=150.)
            if profile == ANCHOR_PROFILE:
                cfg = replace(cfg, fixed_plane_v134_late_xy_anchor_penalty_weight=
                    stage.cfg.fixed_plane_v134_late_xy_anchor_penalty_weight * .5)
        if index >= 11 and profile in (GATE_PROFILE, RETENTION_GATE_PROFILE):
            gates = dict(max_recent_mean_hit_contact_center_dist_m=.0065,
                max_recent_late_physical_below_floor_rate=.008,
                max_recent_late_physical_above_ceiling_rate=.008,
                min_recent_late_physical_in_band_rate=.988)
        if index == 8 and profile == RETENTION_GATE_PROFILE:
            gates = dict(max_recent_late_physical_above_ceiling_rate=.007,
                         min_recent_late_physical_in_band_rate=.992)
        notes = (
            "V179 matched original-reward control; V180 symmetric physical-tail "
            "cost and mean-contact precision; V181 releases half of the late "
            "world-anchor cost. V182 preserves original rewards and calibrates "
            "only four quality gates after negative reward evidence. All physical, "
            "survival/hit/view/motion/safety contracts and consecutive/block/S9 "
            "checks remain. Not formal hardware-coverage evidence.")
        if profile == RETENTION_GATE_PROFILE:
            notes = (
                "V183 retains V178 rewards and all 17 physical domains. "
                "Calibrated S12-S17 center/tails/band: 6.5mm/0.8%/0.8%/98.8%; "
                "S9 high tail/band: 0.7%/99.2%, retaining its 5.3mm center gate. "
                "Longevity, hit, view, motion, safety, consecutive, independent "
                "block and S9 retention checks remain. Admission requires "
                "authenticated full blocks and negative reward experiments. "
                "Gate calibration is not improved precision or hardware evidence.")
        result.append(replace(stage, cfg=cfg, **gates, notes=notes))
    return result


def validate_resume_payload(payload, *, profile, requested_start_stage,
                            resume_curriculum_state, reset_optimizer, reset_critic):
    """Restore full actor/critic/Adam and clear history only at a new contract."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 477, 7)):
        raise ValueError("Precision/tail repair requires full72/477/7 and Adam")
    source = payload.get("args", {}).get("curriculum_profile")
    if source == PARENT_PROFILE:
        key = (payload.get("stage_update"), payload.get("step"))
        if (key not in PARENT_SOURCES or resume_curriculum_state
                or requested_start_stage != PARENT_STAGE
                or payload.get("stage_name") != PARENT_STAGE
                or (profile not in (GATE_PROFILE, RETENTION_GATE_PROFILE) and key[0] != 1168)):
            raise ValueError("Pinned parent and fresh contract history required")
    elif source == profile:
        if resume_curriculum_state:
            if payload.get("stage_name") != requested_start_stage:
                raise ValueError("Own continuation requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("Reject unrelated or failed-candidate parent")
