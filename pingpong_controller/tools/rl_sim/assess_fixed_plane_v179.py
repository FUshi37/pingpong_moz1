"""Recompute precision/tail quality from authenticated complete GPU blocks."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from fixed_plane_v179_gpu1 import (
    GATE_PROFILE, RETENTION_GATE_PROFILE, GATE_FIELDS,
    BALANCED_PROFILE, ANCHOR_PROFILE,
)

SEEDS = (20265661, 20265662)
QUALITY_KEYS = {
    "mean_hit_contact_center_dist_m": ("hit_contact_center_dist_sum", "hit_event_count", False),
    "late_below_floor_rate": ("v80_late_below_floor_event", "v80_late_event", False),
    "late_above_ceiling_rate": ("v80_late_above_ceiling_event", "v80_late_event", False),
    "late_in_band_rate": ("v80_late_in_band_event", "v80_late_event", False),
    "late_adjacent_jump_m": ("v80_late_pair_jump_sum_m", "v80_late_pair_event", False),
    "steady_racket_vxy_rms_m_s": ("rapid_steady_steady_racket_vxy_sq_sum", "rapid_steady_steady_events", True),
    "steady_racket_omega_rms_rad_s": ("rapid_steady_steady_racket_omega_sq_sum", "rapid_steady_steady_events", True),
}


def read_population(directory: Path, *, prefix: str, checkpoint_sha: str):
    """Verify hashes, seed/length/device contracts, and measured event ratios."""
    if prefix not in ("", "reference_", "original_"):
        raise ValueError("Invalid cached population prefix")
    manifest = json.loads((directory / "manifest.json").read_text())
    sha_key = {"": "checkpoint_sha256", "reference_": "reference_sha256",
               "original_": "original_sha256"}[prefix]
    if (manifest[sha_key] != checkpoint_sha or tuple(manifest["seeds"]) != SEEDS
            or manifest["n_envs"] != 288 or manifest["steps"] != 1200
            or not all("cuda" in d for d in manifest["devices"])):
        raise ValueError("Incomplete or mismatched GPU population")
    for filename, expected in manifest["artifacts"].items():
        path = directory / filename
        if path.parent != directory or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen evaluation artifact hash mismatch")
    result = []
    for seed in SEEDS:
        block = json.loads((directory / f"{prefix}block_{seed}.json").read_text())
        with np.load(directory / f"{prefix}metrics_{seed}.npz") as values:
            arrays = {key: values[key] for key in values.files}
        quality = {}
        for key, (numerator, denominator, rms) in QUALITY_KEYS.items():
            num, den = arrays[numerator], arrays[denominator]
            if num.shape != (288,) or den.shape != (288,) or den.sum() <= 0:
                raise ValueError("Invalid full-lane physical event statistics")
            value = float(num.sum() / den.sum())
            quality[key] = math.sqrt(max(value, 0.)) if rms else value
        quality.update(
            late_physical_events=float(arrays["v80_late_event"].sum()),
            late_pair_events=float(arrays["v80_late_pair_event"].sum()),
            steady_events=float(arrays["rapid_steady_steady_events"].sum()))
        for key, numerator, denominator in (
            ("hit4_apex_z_m", "rapid_steady_hit4_apex_sum_z_m", "rapid_steady_hit4_events"),
            ("steady_band_rate", "rapid_steady_steady_band_events", "rapid_steady_steady_events"),
            ("steady_surface_z_abs_error_m", "rapid_steady_steady_surface_z_abs_error_sum_m", "rapid_steady_steady_events")):
            den = arrays[denominator].sum()
            quality[key] = float(arrays[numerator].sum() / den) if den > 0 else float("nan")
        if not all(math.isfinite(v) for v in quality.values()):
            raise ValueError("Nonfinite physical quality")
        result.append(dict(seed=seed, block=block, quality=quality, arrays=arrays))
    return manifest, result


def assess_reward_trial(candidate_directory: Path, matched_directory: Path,
                        original_directory: Path):
    """Compare paired physical means and retain negative bounded trials."""
    manifest = json.loads((candidate_directory / "manifest.json").read_text())
    matched_manifest = json.loads((matched_directory / "manifest.json").read_text())
    original_manifest = json.loads((original_directory / "manifest.json").read_text())
    _, candidate = read_population(candidate_directory, prefix="",
        checkpoint_sha=manifest["checkpoint_sha256"])
    _, matched = read_population(matched_directory, prefix="",
        checkpoint_sha=matched_manifest["checkpoint_sha256"])
    _, original = read_population(original_directory, prefix="",
        checkpoint_sha=original_manifest["checkpoint_sha256"])

    def combine(populations):
        result = {}
        for key, (numerator, denominator, rms) in QUALITY_KEYS.items():
            value = sum(p["arrays"][numerator].sum() for p in populations) / sum(
                p["arrays"][denominator].sum() for p in populations)
            result[key] = math.sqrt(max(value, 0.)) if rms else float(value)
        return result

    metrics = {"candidate": combine(candidate), "matched": combine(matched),
               "original": combine(original)}
    c = metrics["candidate"]
    checks = {}
    deltas = {}
    for label, populations in (("matched", matched), ("original", original)):
        r = metrics[label]
        checks[f"center_gain_2pct_vs_{label}"] = (
            c["mean_hit_contact_center_dist_m"] <= .98 * r["mean_hit_contact_center_dist_m"])
        checks[f"height_band_nondecreasing_vs_{label}"] = c["late_in_band_rate"] >= r["late_in_band_rate"]
        checks[f"individual_tails_stable_vs_{label}"] = all(
            c[key] <= r[key] + .001 for key in ("late_below_floor_rate", "late_above_ceiling_rate"))
        checks[f"motion_stable_vs_{label}"] = all(c[key] <= 1.03 * r[key] for key in (
            "steady_racket_vxy_rms_m_s", "steady_racket_omega_rms_rad_s", "late_adjacent_jump_m"))
        # Matched initial random lanes are bootstrapped within each eval seed.
        rng = np.random.default_rng(20261010)
        estimates = []
        for _ in range(1000):
            cn = cd = rn = rd = 0.
            for cp, rp in zip(candidate, populations, strict=True):
                indices = rng.integers(0, 288, 288)
                cn += cp["arrays"]["hit_contact_center_dist_sum"][indices].sum()
                cd += cp["arrays"]["hit_event_count"][indices].sum()
                rn += rp["arrays"]["hit_contact_center_dist_sum"][indices].sum()
                rd += rp["arrays"]["hit_event_count"][indices].sum()
            estimates.append(cn / cd - rn / rd)
        interval = np.quantile(estimates, [.05, .95]).tolist()
        deltas[label] = dict(center_delta_90pct_ci_m=interval)
        checks[f"center_improvement_ci_vs_{label}"] = interval[1] < 0.
    checks["motion_gates_retained"] = (
        c["steady_racket_vxy_rms_m_s"] <= .053
        and c["steady_racket_omega_rms_rad_s"] <= .28
        and c["late_adjacent_jump_m"] <= .009)
    checks["survival_hit_gates_retained"] = all(
        p["block"]["advance_eval/mean_len_frac"] >= .995
        and p["block"]["advance_eval/episode_truncation_rate"] >= .995
        and p["block"]["advance_eval/mean_hits"] >= 12.9
        and p["block"]["advance_eval/hit12_rate"] >= .99 for p in candidate)
    checks["view_and_safety_stable"] = True
    for cp, rp in zip(candidate, original, strict=True):
        for key in ("advance_eval/ball_view_in_bounds", "advance_eval/ball_view_z_ideal"):
            checks["view_and_safety_stable"] &= cp["block"][key] >= rp["block"][key] - .002
        for key in ("arm_qvel_limit_exceed_fraction", "arm_qacc_limit_exceed_fraction"):
            checks["view_and_safety_stable"] &= (
                cp["arrays"][key].mean() <= rp["arrays"][key].mean() + .0012)
    checks = {key: bool(value) for key, value in checks.items()}
    return dict(accepted=bool(all(checks.values())), profile=manifest["profile"],
        candidate_sha256=manifest["checkpoint_sha256"],
        candidate_directory=str(candidate_directory), matched_directory=str(matched_directory),
        original_directory=str(original_directory), checks=checks, metrics=metrics,
        paired_uncertainty=deltas, failed_checks=[key for key, value in checks.items() if not value],
        interpretation="Two evaluation seeds; positive screen still requires another PPO seed and original S9 retention")


def require_gate_evidence(path: Path, checkpoint_sha: str, *, profile=GATE_PROFILE):
    """Authenticate quality calibration and retain longevity/view/motion/safety."""
    from train_juggle_mjx_curriculum import build_curriculum, rapid_steady_quality_gate
    evidence = json.loads(path.read_text())
    if (evidence["checkpoint_sha256"] != checkpoint_sha
            or evidence.get("profile") != profile
            or profile not in (GATE_PROFILE, RETENTION_GATE_PROFILE)):
        raise ValueError("Gate evidence belongs to another policy")
    negative_paths = [Path(p).resolve() for p in evidence.get("negative_reward_decisions", [])]
    if len(set(negative_paths)) < 2:
        raise ValueError("Calibrated fallback requires distinct negative reward trials")
    stages = build_curriculum(curriculum_profile=profile,
        measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile="recovered_rmp_rmpmd_v2")
    audited = {}
    for label, index in (("s12", 11), ("s9", 8)):
        directory = Path(evidence[f"{label}_directory"])
        manifest, populations = read_population(directory,
            prefix=evidence.get(f"{label}_prefix", ""), checkpoint_sha=checkpoint_sha)
        if manifest["stage"] != stages[index].name:
            raise ValueError("Wrong evaluated physical domain")
        checks = []
        for population in populations:
            block, quality = population["block"], population["quality"]
            if not rapid_steady_quality_gate(stages[index], quality):
                raise ValueError(f"{label} quality fails seed{population['seed']}: {quality}")
            failed = [k for k, v in block.items() if k.endswith("_ok") and (not math.isfinite(v) or v < .5)
                      and k != "advance_eval/rapid_steady_quality_ok"]
            if (failed or block["advance_eval/episodes"] < 144
                    or block["advance_eval/mean_len_frac"] < .995
                    or block["advance_eval/episode_truncation_rate"] < .995
                    or block["advance_eval/mean_hits"] < 12.9
                    or block["advance_eval/hit12_rate"] < .99):
                raise ValueError(f"{label} preserved longevity/view/safety gates fail: {failed}")
            checks.append(dict(seed=population["seed"], quality=quality,
                full=block["advance_eval/episode_truncation_rate"],
                hits=block["advance_eval/mean_hits"], hit12=block["advance_eval/hit12_rate"]))
        audited[label] = dict(manifest_sha256=hashlib.sha256(
            (directory / "manifest.json").read_bytes()).hexdigest(), checks=checks)
    negative_profiles = set()
    for decision_path in negative_paths:
        decision = json.loads(decision_path.read_text())
        recomputed = assess_reward_trial(Path(decision["candidate_directory"]),
            Path(decision["matched_directory"]), Path(decision["original_directory"]))
        if recomputed["accepted"] is not False:
            raise ValueError("Fallback cannot discard a passing reward candidate")
        negative_profiles.add(recomputed["profile"])
    if not {BALANCED_PROFILE, ANCHOR_PROFILE}.issubset(negative_profiles):
        raise ValueError("Both registered reward hypotheses require distinct negative evidence")
    return dict(checkpoint_sha256=checkpoint_sha, authenticated_domains=audited,
        quality_gates={label: {key: getattr(stages[index], key) for key in GATE_FIELDS}
                       for label, index in (("s12", 11), ("s9", 8))},
        negative_reward_evidence=evidence["negative_reward_decisions"])
