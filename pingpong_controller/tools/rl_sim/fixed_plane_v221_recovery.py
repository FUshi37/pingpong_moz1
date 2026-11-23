"""Explicit experimental small physical/contact/camera DR after S17."""

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

from mjx_juggle_env import MjxFixedPlaneV211RewardConfig

PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v221"
CALIBRATED_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v222"
PROFILES = (PROFILE,CALIBRATED_PROFILE)
BASE_PROFILE = "fixed_plane_limit_cycle_qvel_rmp_gpu1_v220"
START_STAGE = "fplc_v221_gpu1_s17_small_dr_consolidation"
FIRST_TAIL_STAGE = "fplc_v221_gpu1_s18_flight_xy_bridge"


@dataclass(frozen=True)
class RecoveryDRConfig(MjxFixedPlaneV211RewardConfig):
    """New fields are absent from every legacy configuration and checkpoint."""

    fplc_v221_episode_weighted_convergence: bool = True
    fplc_v221_flight_impulse_xy_max_m_s: float = 0.
    fplc_v221_flight_impulse_step: int = 480
    fplc_v221_flight_impulse_min_hits: int = 3
    fplc_v221_flight_impulse_clearance_m: float = .05
    fplc_v221_contact_dr_min_hits: int = 3
    fplc_v221_camera_dropout_frames: bool = False
    fplc_v221_camera_dropout_min_hits: int = 3
    fplc_v221_camera_burst_min_frames: int = 2
    fplc_v221_camera_burst_max_frames: int = 3
    fplc_v221_short_gap_prediction_max_s: float = .04


@dataclass(frozen=True)
class CameraClockDRConfig(RecoveryDRConfig):
    """V222 separates periodic camera refreshes from accepted source timestamps."""

    fplc_v222_independent_camera_clock: bool = True


def _contact_cfg(cfg: RecoveryDRConfig, fraction: float) -> RecoveryDRConfig:
    """Candidate bounds around the exact V98 pair; launch stays nominal."""
    time_s, damping = cfg.ball_racket_pair_solref
    friction = cfg.ball_racket_pair_friction[0]
    return replace(cfg, dr_randomize_contact=True,
        dr_ball_friction_range=(.20,.20),
        dr_racket_friction_range=(friction*(1-.02*fraction), friction*(1+.02*fraction)),
        dr_ball_solref_time_range=(time_s*(1-.01*fraction), time_s*(1+.01*fraction)),
        dr_ball_solref_damping_range=(damping*(1-.005*fraction), damping*(1+.005*fraction)))


def build_stages(source: Sequence[Any]) -> list[Any]:
    """Keep S1--S16 exact and append seven gradual small-DR lessons."""
    if len(source) != 17 or not source[-1].name.startswith("fplc_v220_gpu1_s17_"):
        raise ValueError("V221 requires the complete immutable V220 course")
    base = source[-1]
    cfg = RecoveryDRConfig(**vars(base.cfg))
    common = dict(max_recent_late_adjacent_jump_m=.0115,
                  max_recent_mean_hit_contact_center_dist_m=.0075)
    result = list(source[:16]) + [replace(base, name=START_STAGE, cfg=cfg, **common,
        notes="V221 pools completed episodes by count rather than equally weighting "
              "small and synchronized update populations. Only S17 adjacent11->11.5mm "
              "and center7->7.5mm auxiliary gates change; 90% physical band,99.5% "
              "full episodes,99% length,32 hold, all view/motion/safety remain.")]
    labels = ("flight_xy_bridge", "flight_xy_commit", "contact_bridge", "contact_commit",
              "camera_single_frame", "camera_short_burst", "three_dr_consolidation")
    for number, label in enumerate(labels, 18):
        name = f"fplc_v221_gpu1_s{number}_{label}"
        if number != 24:
            cfg = replace(cfg, fixed_contact_training_phase=name)
        if number in (18,19):
            cfg = replace(cfg, fplc_v221_flight_impulse_xy_max_m_s=.01 if number==18 else .02)
        elif number in (20,21):
            cfg = _contact_cfg(cfg,.5 if number==20 else 1.)
        elif number in (22,23):
            cfg = replace(cfg, fplc_v221_camera_dropout_frames=True,
                ball_obs_dropout_on_refresh_only=True,
                ball_obs_dropout_prob=.005 if number==22 else .01,
                ball_obs_dropout_max_steps=1,
                ball_obs_dropout_burst_prob=0. if number==22 else .001,
                ball_obs_dropout_burst_max_steps=3)
        gates = common if number==24 else dict(
            max_recent_late_adjacent_jump_m=.012,
            max_recent_mean_hit_contact_center_dist_m=.008,
            min_recent_late_physical_in_band_rate=.88,
            max_recent_late_physical_below_floor_rate=.05,
            max_recent_late_physical_above_ceiling_rate=.10,
            target_episode_truncation_rate=.99)
        result.append(replace(base, name=name, cfg=cfg, **gates,
            target_mean_len_frac=.99,
            notes="V221 experimental small recovery DR; no hardware-coverage claim. "
                  "S18/19 one free-flight XY impulse +/-0.01/0.02m/s at2.4s after3 hits, "
                  "no position jump or vertical impulse. S20/21 episode-coherent pair "
                  "time +/-0.5/1%,damping +/-0.25/0.5%,friction +/-1/2% after3 hits, "
                  "static launch nominal. S22/23 camera single0.5/1% and0/0.1% "
                  "2--3-frame bursts at90Hz; observed-state-only ballistic prediction "
                  "<=40ms with distinct source age and represented age, invalid termination. "
                  "S24 keeps S23 domain unchanged and consolidates at99.5% full/90% band. "
                  "All rewards, PPO, actor72/critic477/action7 and safety limits retained."))
    return result


def _completed_final(payload: Mapping[str, Any]) -> None:
    row = payload.get("last_row", {})
    for key, value in (("stage_name",payload.get("stage_name")),
                       ("stage_index",17), ("stage_update",payload.get("stage_update")),
                       ("global_step",payload.get("step"))):
        if row.get(key) != value:
            raise ValueError("V221 source final-stage validation disagrees with checkpoint")
    for key in ("convergence/stage_converged","advance_eval/ran","advance_eval/passed",
                "retention_eval/ran","retention_eval/passed"):
        if float(row.get(key,0.)) != 1.:
            raise ValueError("V221 cannot skip an unvalidated source S17")


def build_motion_calibrated_stages(source: Sequence[Any]) -> list[Any]:
    """Keep V221 S1--S17 and physical DR; correct only tail motion/clock contracts."""
    if len(source)!=24 or source[16].name!=START_STAGE:
        raise ValueError("V222 requires the complete V221 prototype course")
    result=list(source[:17])
    for index,stage in enumerate(source[17:],17):
        cfg=CameraClockDRConfig(**vars(stage.cfg)) if index>=21 else stage.cfg
        result.append(replace(stage,name=stage.name.replace('fplc_v221_','fplc_v222_'),
            cfg=cfg,max_recent_rapid_steady_racket_vxy_rms_m_s=.055,
            notes=stage.notes+' V222 raises only added-DR XY RMS53->55mm/s, keeping '
                  'S17/S9 and hard safety exact. Camera refresh uses successive200Hz '
                  'clock ticks independently of the last accepted source timestamp; '
                  'a dropped90Hz frame never enables early200Hz retry.'))
    return result


def validate_resume_payload(payload: Mapping[str, Any], *, requested_start_stage: str,
                            resume_curriculum_state: bool, reset_optimizer: bool,
                            reset_critic: bool, profile: str = PROFILE) -> None:
    if profile not in PROFILES:raise ValueError('Unknown recovery profile')
    if (reset_optimizer or reset_critic or payload.get("opt") is None
            or not {"pi","v"}.issubset(payload.get("params",{}))
            or tuple(payload.get(k) for k in ("obs_dim","critic_obs_dim","act_dim"))!=(72,477,7)
            or payload.get("args",{}).get("max_stage_updates") != -1):
        raise ValueError("V221 requires the full uncapped72/477/7 actor/critic/Adam")
    source=payload["args"]["curriculum_profile"]
    if source==BASE_PROFILE:
        if (resume_curriculum_state or payload.get("stage_index")!=17
                or not str(payload.get("stage_name","")).startswith('fplc_v220_gpu1_s17_')
                or int(payload.get("stage_update",0))<1):
            raise ValueError("V221 requires formal V220 S17 and fresh gate history")
        first_tail=FIRST_TAIL_STAGE.replace('v221','v'+profile.rsplit('v',1)[1])
        if requested_start_stage==first_tail:
            _completed_final(payload)
        elif requested_start_stage!=START_STAGE:
            raise ValueError("V221 transition starts only at S17 or its validated successor")
    elif source==profile:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get("stage_name"):
                raise ValueError("V221 own resume requires exact stage/history")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
    else:
        raise ValueError("V221 rejects another source profile")
