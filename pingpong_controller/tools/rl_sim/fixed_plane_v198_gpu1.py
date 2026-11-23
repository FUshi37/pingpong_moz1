"""Opt-in stage-local height calibration after negative reward experiments."""

from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v198"
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v191"
PARENT_STAGE = "fplc_v191_gpu1_s13_plant_quarter_height_cadence"
START_STAGE = "fplc_v198_gpu1_s13_plant_quarter_height_cadence"
PARENT_SHA = "f2a9b9c7e6ad585161a36cb455292cb664ba60965a99eec30ff34bef124af208"
PARENT_UPDATE = 1521
PARENT_STEP = 2312372224
PARENT_ADAM_T = 564929
GATE_FIELDS = (
    "min_recent_late_physical_in_band_rate",
    "max_recent_late_physical_below_floor_rate",
    "max_recent_late_physical_above_ceiling_rate",
)
SEEDS = (20265661, 20265662)


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Calibrate S13 alone; keep every other complete stage numerically exact."""
    if len(source) != 17 or source[12].name != PARENT_STAGE:
        raise ValueError("Complete original V191 course required")
    result = list(source)
    result[12] = replace(source[12], name=START_STAGE,
        min_recent_late_physical_in_band_rate=.97,
        max_recent_late_physical_below_floor_rate=.02,
        max_recent_late_physical_above_ceiling_rate=.02,
        notes="V198 changes only S13 physical fourth-plus height gates to "
              "97% in 1.400--1.450 m and at most 2% in either tail. "
              "All rewards, full survival, hits11--15, center6.5mm, jump9mm, "
              "motion/view/safety, rolling32/hold32, stochastic full block "
              "and unchanged S9 retention remain. S14--S17 DR and gates "
              "remain V191-exact; no policy improvement is claimed.")
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Require full actor/critic/Adam; never reuse history across changed gates."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or tuple(payload.get(k) for k in
                     ("obs_dim", "critic_obs_dim", "act_dim")) != (72, 477, 7)
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or payload.get("args", {}).get("max_stage_updates") != -1):
        raise ValueError("V198 requires full formal 72/477/7 state and Adam")
    source = payload["args"]["curriculum_profile"]
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_name") != PARENT_STAGE
                or payload.get("stage_update") != PARENT_UPDATE
                or payload.get("step") != PARENT_STEP
                or int(payload["opt"].t) != PARENT_ADAM_T
                or payload.get("stop_reason") != "received SIGINT"):
            raise ValueError("V198 pins interrupted V191/u1521 and fresh gate history")
    elif source == PROFILE:
        if resume_curriculum_state:
            if payload.get("stage_name") != requested_start_stage:
                raise ValueError("V198 own resume requires exact saved stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V198 rejects another source profile")


def require_gate_evidence(path: str | Path) -> dict[str, Any]:
    """Recompute complete stochastic CUDA blocks rather than trust pass flags."""
    from assess_fixed_plane_v179 import read_population
    from train_juggle_mjx_curriculum import build_curriculum, rapid_steady_quality_gate

    evidence = json.loads(Path(path).read_text())
    if (evidence.get("kind") != "gate_only_s13_height_calibration"
            or evidence.get("profile") != PROFILE
            or evidence.get("parent_sha256") != PARENT_SHA):
        raise ValueError("Pinned stage-local gate evidence required")
    for prefix in ("preregister", "negative_reward_report"):
        source = Path(evidence[f"{prefix}_file"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != evidence[f"{prefix}_sha256"]:
            raise ValueError("Changed calibration provenance")
    kwargs = dict(measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile="recovered_rmp_rmpmd_v2")
    parent = build_curriculum(curriculum_profile=PARENT_PROFILE, **kwargs)
    stages = build_stages(parent)
    audited = {}
    for label, index in (("s13", 12), ("s9", 8)):
        directory = Path(evidence[f"{label}_directory"])
        manifest, populations = read_population(directory, prefix="", checkpoint_sha=PARENT_SHA)
        if (manifest.get("deterministic", False)
                or manifest["profile"] != PARENT_PROFILE
                or manifest["stage"] != parent[index].name
                or manifest["cfg_sha256"] != hashlib.sha256(
                    repr(vars(parent[index].cfg)).encode()).hexdigest()):
            raise ValueError("Wrong stochastic domain or unchanged configuration")
        checks = []
        for population in populations:
            block, quality = population["block"], population["quality"]
            failed = [key for key, value in block.items()
                if key.endswith("_ok") and (not math.isfinite(value) or value < .5)
                and key != "advance_eval/rapid_steady_quality_ok"]
            if (failed or not rapid_steady_quality_gate(stages[index], quality)
                    or not 11. <= block["advance_eval/mean_hits"] <= 15.
                    or block["advance_eval/mean_len_frac"] < 1.
                    or block["advance_eval/episode_truncation_rate"] < .99
                    or (label == "s9" and not block["advance_eval/passed"])):
                raise ValueError(f"{label} failed seed {population['seed']}: {failed}; {quality}")
            checks.append(dict(seed=population["seed"], quality=quality,
                mean_hits=block["advance_eval/mean_hits"],
                len_frac=block["advance_eval/mean_len_frac"],
                original_passed=bool(block["advance_eval/passed"])))
        audited[label] = dict(checks=checks, manifest_sha256=hashlib.sha256(
            (directory / "manifest.json").read_bytes()).hexdigest())
    return dict(profile=PROFILE, parent_sha256=PARENT_SHA, authenticated_domains=audited,
                gate_changes={key: getattr(stages[12], key) for key in GATE_FIELDS})
