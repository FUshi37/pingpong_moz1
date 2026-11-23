"""Authenticate paired upper-height experiments before a formal continuation."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from assess_fixed_plane_v179 import QUALITY_KEYS
from fixed_plane_v184_gpu1 import (
    PARENT_SHA, BASELINE_PROFILE, EVENT_PROFILE, SHOULDER_PROFILE, BALANCED_PROFILE,
    HIT_CREDIT_PROFILE, STRONG_EVENT_PROFILE, SYMMETRIC_BAND_PROFILE,
)


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_blocks(directory: str | Path, *, prefix: str, checkpoint_sha: str,
                seeds: Sequence[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify hashes and complete CUDA populations rather than trust pass flags."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    key = {"": "checkpoint_sha256", "reference_": "reference_sha256",
           "original_": "original_sha256"}[prefix]
    if (manifest[key] != checkpoint_sha or manifest["seeds"] != list(seeds)
            or manifest["n_envs"] != 288 or manifest["steps"] != 1200
            or not manifest["devices"] or not all("cuda" in d for d in manifest["devices"])):
        raise ValueError("Mismatched full GPU block evidence")
    for name, digest in manifest["artifacts"].items():
        if Path(name).name != name or sha256(directory / name) != digest:
            raise ValueError("Evaluation artifact hash mismatch")
    populations = []
    for seed in seeds:
        block = json.loads((directory / f"{prefix}block_{seed}.json").read_text())
        with np.load(directory / f"{prefix}metrics_{seed}.npz") as archive:
            arrays = {k: archive[k] for k in archive.files}
        if any(v.shape != (288,) or not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError("Invalid per-lane physical statistics")
        if block["advance_eval/episodes"] < 144:
            raise ValueError("Incomplete full-episode block")
        populations.append(dict(seed=seed, block=block, arrays=arrays))
    return manifest, populations


def combine(populations: Sequence[dict[str, Any]]) -> dict[str, float]:
    result = {}
    for key, (num, den, rms) in QUALITY_KEYS.items():
        denominator = sum(p["arrays"][den].sum() for p in populations)
        if denominator <= 0.:
            raise ValueError("Missing physical-event evidence")
        value = sum(p["arrays"][num].sum() for p in populations) / denominator
        result[key] = math.sqrt(max(value, 0.)) if rms else float(value)
    return result


def paired_interval(candidate: Sequence[dict[str, Any]],
                    reference: Sequence[dict[str, Any]], key: str) -> list[float]:
    num, den, rms = QUALITY_KEYS[key]
    rng = np.random.default_rng(20261011)
    estimates = []
    for _ in range(1000):
        cn = cd = rn = rd = 0.
        for cp, rp in zip(candidate, reference, strict=True):
            indices = rng.integers(0, 288, 288)
            cn += cp["arrays"][num][indices].sum()
            cd += cp["arrays"][den][indices].sum()
            rn += rp["arrays"][num][indices].sum()
            rd += rp["arrays"][den][indices].sum()
        if cd <= 0. or rd <= 0.:
            raise ValueError("Invalid bootstrap denominator")
        cv, rv = cn / cd, rn / rd
        estimates.append(math.sqrt(cv) - math.sqrt(rv) if rms else cv - rv)
    return np.quantile(estimates, [.05, .95]).tolist()


def assess_trial(directory: str | Path, *,
                 seeds: Sequence[int] = (20265661, 20265662)) -> dict[str, Any]:
    """Compare candidate against its matched PPO control and the stopped parent."""
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    _, candidate = read_blocks(directory, prefix="", seeds=seeds,
        checkpoint_sha=manifest["checkpoint_sha256"])
    references = {}
    for label, prefix, key in (("matched", "reference_", "reference_sha256"),
                               ("parent", "original_", "original_sha256")):
        _, references[label] = read_blocks(directory, prefix=prefix, seeds=seeds,
            checkpoint_sha=manifest[key])
    if manifest["original_sha256"] != PARENT_SHA:
        raise ValueError("Wrong protected original parent")
    metrics = {"candidate": combine(candidate), **{
        label: combine(values) for label, values in references.items()}}
    c = metrics["candidate"]
    checks = {}
    uncertainty = {}
    for label, populations in references.items():
        r = metrics[label]
        checks[f"upper_tail_gain_vs_{label}"] = (
            c["late_above_ceiling_rate"] <= .8 * r["late_above_ceiling_rate"]
            and c["late_above_ceiling_rate"] <= r["late_above_ceiling_rate"] - .003)
        checks[f"band_gain_vs_{label}"] = c["late_in_band_rate"] >= r["late_in_band_rate"] + .002
        upper_ci = paired_interval(candidate, populations, "late_above_ceiling_rate")
        center_ci = paired_interval(candidate, populations, "mean_hit_contact_center_dist_m")
        uncertainty[label] = dict(upper_tail_delta_90pct_ci=upper_ci,
                                  center_delta_90pct_ci_m=center_ci)
        checks[f"upper_tail_ci_vs_{label}"] = upper_ci[1] < 0.
        checks[f"center_stable_vs_{label}"] = (
            c["mean_hit_contact_center_dist_m"] <= r["mean_hit_contact_center_dist_m"] + .0001
            and center_ci[1] <= .0002)
        checks[f"low_tail_stable_vs_{label}"] = (
            c["late_below_floor_rate"] <= r["late_below_floor_rate"] + .001)
        checks[f"motion_stable_vs_{label}"] = all(c[k] <= 1.03 * r[k] for k in (
            "steady_racket_vxy_rms_m_s", "steady_racket_omega_rms_rad_s"))
        checks[f"adjacent_stable_vs_{label}"] = (
            c["late_adjacent_jump_m"] <= r["late_adjacent_jump_m"] + .0002)
        checks[f"view_safety_stable_vs_{label}"] = all(
            cp["block"][k] >= rp["block"][k] - .002
            for cp, rp in zip(candidate, populations, strict=True)
            for k in ("advance_eval/ball_view_in_bounds", "advance_eval/ball_view_z_ideal"))
        checks[f"joint_safety_stable_vs_{label}"] = all(
            cp["arrays"][k].mean() <= rp["arrays"][k].mean() + .0012
            for cp, rp in zip(candidate, populations, strict=True)
            for k in ("arm_qvel_limit_exceed_fraction", "arm_qacc_limit_exceed_fraction"))
        checks[f"acquisition_stable_vs_{label}"] = all(
            cp["block"][k] >= rp["block"][k] - .002
            for cp, rp in zip(candidate, populations, strict=True)
            for k in ("advance_eval/hit1_rate", "advance_eval/hit3_rate"))
    checks["full_survival"] = all(
        p["block"]["advance_eval/mean_len_frac"] >= 1.
        and p["block"]["advance_eval/episode_truncation_rate"] >= .995
        for p in candidate)
    checks["preserved_center_motion_gates"] = (
        c["mean_hit_contact_center_dist_m"] <= .0065
        and c["steady_racket_vxy_rms_m_s"] <= .053
        and c["steady_racket_omega_rms_rad_s"] <= .28
        and c["late_adjacent_jump_m"] <= .009)
    checks = {k: bool(v) for k, v in checks.items()}
    return dict(accepted=all(checks.values()), profile=manifest["profile"],
        directory=str(directory), candidate_sha256=manifest["checkpoint_sha256"],
        reference_sha256=manifest["reference_sha256"], parent_sha256=PARENT_SHA,
        checks=checks, failed_checks=[k for k, v in checks.items() if not v],
        metrics=metrics, paired_uncertainty=uncertainty)


def require_admission(path: str | Path, *, profile: str, source_sha: str) -> dict[str, Any]:
    """Recompute both independent PPO trials and unchanged-domain S9 retention."""
    if source_sha != PARENT_SHA:
        raise ValueError("Initial formal reward admission restores the protected parent")
    evidence = json.loads(Path(path).read_text())
    if evidence.get("profile") != profile or evidence.get("parent_sha256") != PARENT_SHA:
        raise ValueError("Wrong reward admission identity")
    contract_path = Path(evidence["preregister_file"])
    if sha256(contract_path) != evidence["preregister_sha256"]:
        raise ValueError("Preregistered acceptance contract hash mismatch")
    contract = json.loads(contract_path.read_text())
    if (contract["ppo_seeds"] != [20263711, 20263712]
            or contract.get("probe_updates") != 64
            or contract.get("screen_eval_seeds") != [20265661, 20265662]
            or contract.get("confirmation_eval_seeds") != [20265711, 20265712]):
        raise ValueError("Wrong independent PPO seed contract")
    audited = []
    if profile in (EVENT_PROFILE, SHOULDER_PROFILE, BALANCED_PROFILE, HIT_CREDIT_PROFILE,
                   STRONG_EVENT_PROFILE, SYMMETRIC_BAND_PROFILE):
        if profile == BALANCED_PROFILE and contract.get("arms", {}).get("v187") != {
                "fixed_plane_v92_signed_height_reward_scale": 9.0}:
            raise ValueError("Balanced height reward requires its preregistered follow-up contract")
        if profile == HIT_CREDIT_PROFILE and contract.get("arms", {}).get("v188") != {
                "hit_count_floor_reward_weight": 15.0}:
            raise ValueError("Hit credit reward requires its preregistered follow-up contract")
        if profile == STRONG_EVENT_PROFILE and contract.get("arms", {}).get("v189") != {
                "fixed_plane_v184_upper_event_weight": 150.0}:
            raise ValueError("Stronger upper reward requires its preregistered follow-up contract")
        if profile == SYMMETRIC_BAND_PROFILE and contract.get("arms", {}).get("v190") != {
                "fixed_plane_v165_steady_band_violation_weight": 150.0}:
            raise ValueError("Symmetric band reward requires its preregistered follow-up contract")
        trials = evidence.get("trials", [])
        if {t["ppo_seed"] for t in trials} != set(contract["ppo_seeds"]) or len(trials) != 2:
            raise ValueError("Two distinct matched PPO seeds are required")
        if not any(t.get("eval_seeds") == contract["confirmation_eval_seeds"] for t in trials):
            raise ValueError("Independent confirmation evaluation seeds are required")
        import pickle
        for trial in trials:
            if trial.get("eval_seeds") not in (
                    contract["screen_eval_seeds"], contract["confirmation_eval_seeds"]):
                raise ValueError("Unregistered evaluation seed population")
            directory = Path(trial["evaluation_directory"])
            manifest = json.loads((directory / "manifest.json").read_text())
            checkpoint = Path(manifest["checkpoint"])
            if sha256(checkpoint) != manifest["checkpoint_sha256"]:
                raise ValueError("Candidate checkpoint hash mismatch")
            p = pickle.loads(checkpoint.read_bytes())
            if (p["args"]["seed"] != trial["ppo_seed"]
                    or p["args"]["curriculum_profile"] != profile
                    or p["stage_update"] != contract["probe_updates"]
                    or sha256(p["args"]["resume_from"]) != PARENT_SHA):
                raise ValueError("Candidate does not reproduce the preregistered full-state trial")
            control = Path(manifest["reference_checkpoint"])
            q = pickle.loads(control.read_bytes())
            if (sha256(control) != manifest["reference_sha256"]
                    or q["args"]["seed"] != trial["ppo_seed"]
                    or q["args"]["curriculum_profile"] != BASELINE_PROFILE
                    or q["stage_update"] != contract["probe_updates"]
                    or sha256(q["args"]["resume_from"]) != PARENT_SHA):
                raise ValueError("Matched control has another seed/source/budget")
            result = assess_trial(directory, seeds=trial["eval_seeds"])
            if not result["accepted"]:
                raise ValueError(f"Rejected reward trial: {result['failed_checks']}")
            audited.append(result)
    elif profile == BASELINE_PROFILE:
        negatives = evidence.get("negative_trials", [])
        decisions = [assess_trial(Path(p)) for p in negatives]
        if ({d["profile"] for d in decisions} != {EVENT_PROFILE, SHOULDER_PROFILE}
                or any(d["accepted"] for d in decisions)):
            raise ValueError("Fallback requires both recorded negative reward hypotheses")
        audited = decisions
    else:
        raise ValueError("Unregistered formal reward profile")
    from train_juggle_mjx_curriculum import build_curriculum, rapid_steady_quality_gate
    stage = build_curriculum(curriculum_profile=profile,
        measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile="recovered_rmp_rmpmd_v2")[8]
    directory = Path(evidence["retention_directory"])
    manifest, populations = read_blocks(directory, prefix="",
        checkpoint_sha=evidence["retention_checkpoint_sha256"], seeds=evidence["retention_eval_seeds"])
    if manifest["stage"] != stage.name:
        raise ValueError("Retention evaluated a different physical stage")
    for population in populations:
        block = population["block"]
        arrays = population["arrays"]
        quality = combine([population])
        for key, numerator, denominator in (
            ("hit4_apex_z_m", "rapid_steady_hit4_apex_sum_z_m", "rapid_steady_hit4_events"),
            ("steady_band_rate", "rapid_steady_steady_band_events", "rapid_steady_steady_events"),
            ("steady_surface_z_abs_error_m", "rapid_steady_steady_surface_z_abs_error_sum_m", "rapid_steady_steady_events")):
            quality[key] = float(arrays[numerator].sum() / arrays[denominator].sum())
        quality.update(late_physical_events=float(arrays["v80_late_event"].sum()),
            late_pair_events=float(arrays["v80_late_pair_event"].sum()),
            steady_events=float(arrays["rapid_steady_steady_events"].sum()))
        failed = [k for k, v in block.items() if k.endswith("_ok") and v < .5
                  and k != "advance_eval/rapid_steady_quality_ok"]
        if (not rapid_steady_quality_gate(stage, quality) or failed
                or block["advance_eval/mean_len_frac"] < 1.
                or block["advance_eval/episode_truncation_rate"] < .995):
            raise ValueError(f"S9 retention fails: {failed}, {quality}")
    return dict(parent_sha256=PARENT_SHA, trials=audited,
        retention_manifest_sha256=sha256(directory / "manifest.json"))
