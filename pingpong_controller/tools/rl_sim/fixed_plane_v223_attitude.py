"""V223: resume clean V172 and learn only local attitude observation bias."""

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from mjx_juggle_env import MjxFixedPlaneV159RewardConfig

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v223"
PARENT_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v172"
PARENT_SHA256 = "96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a"
START_STAGE = "fplc_v223_gpu1_s11_attitude_bias_025deg"
ANGLES_DEG = (.25, .5, .75, 1., 1.)


def validate_saved_config(saved: Mapping[str, Any], expected: Mapping[str, Any]) -> dict:
    """Allow only the historical absent/default-off V173 critic-context field."""
    additions = {"critic_fixed_plane_task_context": False}
    normalized = dict(saved)
    used = {}
    for key, value in additions.items():
        if key not in normalized and expected.get(key) is value:
            normalized[key] = value
            used[key] = value
    if normalized != dict(expected):
        raise ValueError("V223 saved source configuration differs from its course")
    return used


@dataclass(frozen=True)
class RacketAttitudeObservationConfig(MjxFixedPlaneV159RewardConfig):
    """V223-only configuration; every legacy config keeps its original fields."""

    racket_attitude_observation_dr_enabled: bool = True
    racket_attitude_observation_max_angle_deg: float = .25
    racket_attitude_observation_nominal_fraction: float = .5


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Preserve completed S1--S10, then append four biases and consolidation."""
    if len(source) != 17 or source[9].name != "fplc_v159_gpu1_s10_clean_commit":
        raise ValueError("V223 requires the complete immutable V172 course")
    baseline = source[9]
    result = list(source[:10])
    labels = ("025deg", "050deg", "075deg", "100deg", "100deg_consolidation")
    for number, (angle, label) in enumerate(zip(ANGLES_DEG, labels, strict=True), 11):
        cfg = RacketAttitudeObservationConfig(**vars(baseline.cfg))
        cfg = replace(cfg, racket_attitude_observation_max_angle_deg=angle)
        result.append(replace(
            baseline, name=f"fplc_v223_gpu1_s{number}_attitude_bias_{label}", cfg=cfg,
            min_updates=80 if number == 15 else 60,
            notes="V223 resumes clean V172S10u314 full72/393/7. Only current "
                  "racket attitude observation55:61 is right-composed with a "
                  f"local XY rotation vector of norm<={angle}degree, sampled "
                  "once per episode;50% nominal probability. No physical mount, "
                  "ball, actuator, PD, RMP, contact, dropout or reward changes. "
                  "V172S10 physical gates and independent cleanS10 retention "
                  "remain; no update cap or forced advance. Observation "
                  "robustness alone is not physical mounting-error coverage."))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool) -> None:
    """Reject another source, partial optimizer restoration or stage skipping."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
            != (72, 393, 7)):
        raise ValueError("V223 requires full72/393/7 actor,critic and Adam")
    source = payload.get("args", {}).get("curriculum_profile")
    if source == PARENT_PROFILE:
        if (resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_index") != 10
                or payload.get("stage_name") != "fplc_v159_gpu1_s10_clean_commit"
                or payload.get("stage_update") != 314 or payload.get("step") != 1334837248
                or int(payload["opt"].t) != 335771):
            raise ValueError("V223 pins passed clean V172S10u314 and fresh gate history")
        for key in ("convergence/stage_converged", "advance_eval/passed",
                    "retention_eval/passed"):
            if float(payload.get("last_row", {}).get(key, 0.)) != 1.:
                raise ValueError("V223 parent must have passed every graduation gate")
    elif source == PROFILE:
        if payload.get("args", {}).get("max_stage_updates") != -1:
            raise ValueError("V223 rejects promotion of a bounded-probe actor")
        if resume_curriculum_state:
            if requested_start_stage != payload.get("stage_name"):
                raise ValueError("V223 own resume requires exact stage and history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V223 rejects another source profile")
