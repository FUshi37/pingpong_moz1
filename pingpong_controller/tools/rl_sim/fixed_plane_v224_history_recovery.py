"""V224: real-space intersection, causal history and gradual recovery support."""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

from fixed_plane_v223_attitude import RacketAttitudeObservationConfig
from fixed_plane_v224_history_context import SCHEMA

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_v224"
START_STAGE = "fplc_v224_s01_workspace_masked_history"
RETENTION_STAGE = "fplc_v224_s02_causal_history"
V223_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v223"
V172_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v172"
PARENTS = {
    V223_PROFILE: dict(sha="ec046d9c5f6502f0be86faa07749ed23f8e7a61d4bf03ac2b82b011846611b76",
        stage="fplc_v223_gpu1_s15_attitude_bias_100deg_consolidation",
        index=15, update=270, step=1535639552, adam=384072),
    V172_PROFILE: dict(sha="96c9b8349fa700402a62d1c34e5f8ed2b968bab47cf2b9d3957a1db6cfccba4a",
        stage="fplc_v159_gpu1_s10_clean_commit", index=10, update=314,
        step=1334837248, adam=335771),
}


def low_disk_stop_reason(free_bytes: int) -> str | None:
    """Reserve 1.5 GiB for orderly checkpointing on the nearly full host."""
    if free_bytes < int(1.5 * 1024**3):
        return f"V224 disk reserve reached: free={free_bytes / 1024**3:.3f} GiB"
    return None


def intersect_workspace(
    sim_lower: Sequence[float], sim_upper: Sequence[float],
    real_lower: Sequence[float], real_upper: Sequence[float], *, inset_m: float,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Inset only real-limited faces; preserve already tighter simulator faces."""
    values = (*sim_lower, *sim_upper, *real_lower, *real_upper, inset_m)
    if not all(math.isfinite(float(v)) for v in values) or inset_m < 0.:
        raise ValueError("Workspace endpoints and inset must be finite")
    low = tuple(max(s, r + inset_m) if s < r else s
                for s, r in zip(sim_lower, real_lower, strict=True))
    high = tuple(min(s, r - inset_m) if s > r else s
                 for s, r in zip(sim_upper, real_upper, strict=True))
    if any(a >= b for a, b in zip(low, high, strict=True)):
        raise ValueError("The intersected workspace is empty")
    return low, high


@dataclass(frozen=True)
class HistoryRecoveryConfig(RacketAttitudeObservationConfig):
    """V224-only fields; no legacy configuration/state layout changes."""

    v224_history_enabled: bool = True
    v224_actor_schema: str = SCHEMA
    v224_history_masked: bool = True
    v224_episode_weighted_convergence: bool = True
    v224_flight_impulse_xy_max_m_s: float = 0.


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Create a separate 14-lesson course from the final V223 domain."""
    if len(source) != 15 or source[-1].name != PARENTS[V223_PROFILE]["stage"]:
        raise ValueError("V224 requires the immutable completed V223 course")
    base = source[-1]
    cfg = HistoryRecoveryConfig(**vars(base.cfg))
    low, high = intersect_workspace(cfg.cartesian_pose_position_lower_m,
        cfg.cartesian_pose_position_upper_m, (-.15, -.60, .95), (.15, -.35, 1.20),
        inset_m=.005)
    cfg = replace(cfg, cartesian_pose_position_lower_m=low,
        cartesian_pose_position_upper_m=high,
        fixed_contact_racket_x_bounds_m=(low[0], high[0]),
        fixed_contact_racket_y_bounds_m=(low[1], high[1]),
        fixed_contact_racket_z_bounds_m=(low[2], high[2]))
    lessons = [("workspace_masked_history", {}),
               ("causal_history", dict(v224_history_masked=False))]
    lessons += [(f"initial_xy_{int(v*1000):02d}mm", dict(racket_launch_xy_jitter=v))
                for v in (.001, .002, .004, .008, .012)]
    lessons += [(f"initial_vxy_{int(v*1000):02d}mms", dict(racket_launch_vxy_max=v))
                for v in (.005, .01, .02)]
    lessons += [(f"flight_xy_{int(v*1000):02d}mms",
                 dict(v224_flight_impulse_xy_max_m_s=v)) for v in (.01, .02, .04)]
    lessons += [("unchanged_consolidation", {})]
    result = []
    for number, (label, change) in enumerate(lessons, 1):
        cfg = replace(cfg, **change)
        result.append(replace(base, name=f"fplc_v224_s{number:02d}_{label}", cfg=cfg,
            min_updates=120 if number == 14 else 60,
            notes="V224 117/438/7 preserves the 72-D prefix, physics, reward "
                  "and 2 degree projection. Workspace is the real/sim intersection "
                  "with 5 mm inset only on real-limited faces. Context at 50/100/200 ms "
                  "uses post-DR observed orientations and deployment-only values; "
                  "S1 masks it for ablation; central retention uses live S2 history. Expand placed-ball XY then "
                  "initial velocity, then a single <=40 mm/s XY flight impulse after "
                  "3 hits at first eligible flight after sampled 2--4 s. No contact, "
                  "mount, plant, delay or mass DR change. All original numerical "
                  "graduation gates retained; uncapped and simulation-only."))
    return result


def validate_resume_payload(
    payload: Mapping[str, Any], *, requested_start_stage: str,
    resume_curriculum_state: bool, reset_optimizer: bool,
    reset_critic: bool, allow_obs_dim_migration: bool,
) -> None:
    """Restore authenticated full state; grow only zero-initialized input rows."""
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi", "v"}.issubset(payload.get("params", {}))
            or payload.get("args", {}).get("max_stage_updates") != -1):
        raise ValueError("V224 requires an uncapped full actor/critic/Adam source")
    source = payload["args"]["curriculum_profile"]
    dims = tuple(payload.get(k) for k in ("obs_dim", "critic_obs_dim", "act_dim"))
    if source in PARENTS:
        pin = PARENTS[source]
        if (dims != (72,393,7) or not allow_obs_dim_migration
                or resume_curriculum_state or requested_start_stage != START_STAGE
                or payload.get("stage_name") != pin["stage"]
                or payload.get("stage_index") != pin["index"]
                or payload.get("stage_update") != pin["update"]
                or payload.get("step") != pin["step"]
                or int(payload["opt"].t) != pin["adam"]):
            raise ValueError("V224 pins V223 final or submitted V172 with117/438 migration")
        for key in ("convergence/stage_converged", "advance_eval/passed",
                    "retention_eval/passed"):
            if float(payload.get("last_row", {}).get(key, 0.)) != 1.:
                raise ValueError("V224 parent has not passed full graduation")
    elif source == PROFILE:
        if dims != (117,438,7):
            raise ValueError("V224 continuation requires117/438/7")
        if resume_curriculum_state:
            if requested_start_stage != payload.get("stage_name"):
                raise ValueError("V224 continuation requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload, next_stage=requested_start_stage)
    else:
        raise ValueError("V224 rejects another source")
