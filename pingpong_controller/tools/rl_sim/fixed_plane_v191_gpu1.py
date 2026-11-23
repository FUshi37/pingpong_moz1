"""Pinned full-state height-repair continuations with physical hit bounds."""

from dataclasses import replace
import math
from typing import Any, Mapping, Sequence

PARENT_PROFILE="fixed_plane_limit_cycle_qvel_rmp_gpu1_v184"
PARENT_SHA="cc689908731540d77df1a28025f3faa1dcc4e249ba47c8916fa650dd5bfda431"
PARENT_STAGE="fplc_v184_gpu1_s13_plant_quarter_height_survival"
PARENT_UPDATE=2137
PARENT_STEP=2113011712
PROFILES=tuple(f"fixed_plane_limit_cycle_qvel_rmp_gpu1_v{v}" for v in (191,192,193,194,195,196,197))
REWARD_FIELDS=("fixed_plane_v191_absolute_height_multiplier","fixed_plane_v191_margin_weight",
               "fixed_plane_v194_damping_focus_margin_weight",
               "fixed_plane_v195_late_vxy_excess_weight",
               "fixed_plane_v196_late_action_variation_weight")
GATE_FIELDS=("target_mean_hits","max_recent_mean_hits")


def physical_hit_envelope(*, episode_s: float, third_hit_time_s: tuple[float,float],
                         late_contact_z_m: tuple[float,float],
                         apex_z_m: tuple[float,float],
                         gravity_m_s2: float) -> tuple[float,float]:
    """Bound a six-second episode using three launch hits plus late flights.

    The upper edge includes one endpoint/contact-resolution phase allowance.
    It is a mean-hit graduation range, never an event or episode hit cap.
    """
    values=(episode_s,gravity_m_s2,*third_hit_time_s,*late_contact_z_m,*apex_z_m)
    if (not all(math.isfinite(v) for v in values) or gravity_m_s2<=0.
            or not 0.<=third_hit_time_s[0]<=third_hit_time_s[1]<episode_s
            or not late_contact_z_m[0]<=late_contact_z_m[1]<apex_z_m[0]<=apex_z_m[1]):
        raise ValueError("Invalid physical hit-envelope inputs")
    shortest=2.*math.sqrt(2.*(apex_z_m[0]-late_contact_z_m[1])/gravity_m_s2)
    longest=2.*math.sqrt(2.*(apex_z_m[1]-late_contact_z_m[0])/gravity_m_s2)
    low=3+math.floor((episode_s-third_hit_time_s[1])/longest)
    high=4+math.floor((episode_s-third_hit_time_s[0])/shortest)
    return float(low),float(high)


def start_stage(profile: str) -> str:
    if profile not in PROFILES: raise ValueError("Unregistered height/cadence profile")
    return f"fplc_v{profile.rsplit('v',1)[1]}_gpu1_s13_plant_quarter_height_cadence"


def build_stages(source: Sequence[Any], *, profile: str) -> list[Any]:
    """Change only late reward fields and the physically computed hit range."""
    from mjx_juggle_env import MjxFixedPlaneV191RewardConfig, MjxFixedPlaneV194RewardConfig
    from mjx_juggle_env import MjxFixedPlaneV195RewardConfig
    from mjx_juggle_env import MjxFixedPlaneV196RewardConfig
    if profile not in PROFILES or len(source)!=17: raise ValueError("Complete V184 course required")
    result=list(source[:12]);v=profile.rsplit('v',1)[1]
    lo,hi=physical_hit_envelope(episode_s=6.,third_hit_time_s=(.74,1.24),
        late_contact_z_m=(1.07,1.12),apex_z_m=(1.4,1.45),gravity_m_s2=9.81)
    for s in source[12:]:
        cfg=s.cfg
        if profile==PROFILES[6]:
            cfg=replace(MjxFixedPlaneV191RewardConfig(**vars(cfg)),
                fixed_plane_v191_absolute_height_multiplier=1.25)
        elif profile==PROFILES[5]:
            cfg=replace(MjxFixedPlaneV196RewardConfig(**vars(cfg)),
                fixed_plane_v196_late_action_variation_weight=1.)
        elif profile==PROFILES[4]:
            cfg=replace(MjxFixedPlaneV195RewardConfig(**vars(cfg)),
                fixed_plane_v191_absolute_height_multiplier=1.5,
                fixed_plane_v194_damping_focus_margin_weight=24.,
                fixed_plane_v195_late_vxy_excess_weight=6000.)
        elif profile==PROFILES[3]:
            cfg=replace(MjxFixedPlaneV194RewardConfig(**vars(cfg)),
                fixed_plane_v194_damping_focus_margin_weight=24.)
        elif profile!=PROFILES[0]:
            cfg=replace(MjxFixedPlaneV191RewardConfig(**vars(cfg)),
                fixed_plane_v191_absolute_height_multiplier=1.5 if profile==PROFILES[1] else 1.,
                fixed_plane_v191_margin_weight=12. if profile==PROFILES[2] else 0.)
        result.append(replace(s,cfg=cfg,
            name=s.name.replace('fplc_v184_gpu1_',f'fplc_v{v}_gpu1_').replace('_height_survival','_height_cadence'),
            target_mean_hits=lo,max_recent_mean_hits=hi,
            notes="V191 family retains all V184 non-hit gates and physical contracts. "
                  "Mean confirmed hits must remain in the physical11--15 range; "
                  "there is no per-episode hit cap or change to counting. "
                  "V192 multiplies only the absolute signed-height component1.5x, "
                  "keeping adjacent and unresolved costs exact. V193 adds only a "
                  "symmetric bounded12-point margin cost outside1.415--1.435m. "
                  "First12 stages, actor72/critic477, RMP/PD, DR, PPO and resets remain exact."))
        if profile == PROFILES[3]:
            result[-1] = replace(result[-1], notes=(
                "V194 retains the full V184 course and every non-hit gate. "
                "Only resolved fourth-plus height margin cost is added: weight24 "
                "times damping-conditioned focus for low-damping overshoot or "
                "high-damping undershoot, zero at nominal damping and within "
                "1.415--1.435m. Privileged damping changes reward only. "
                "Physical/control/observation/reset/DR/PPO/BC remain exact; "
                "mean hits11--15, full survival1.0, no event cap."))
        if profile == PROFILES[4]:
            result[-1] = replace(result[-1], notes=(
                "V195 retains V192 absolute-height1.5x with exact adjacent and "
                "unresolved costs, adds the V194 damping-focus margin24 and "
                "late physical-contact XY excess cost6000*(max(vxy-.040,0))^2. "
                "First3 launch hits are excluded from both new penalties. "
                "All physical/control/observation/reset/DR/PPO/BC/non-hit gates "
                "remain exact, with mean hits11--15 and full survival1.0."))
        if profile == PROFILES[5]:
            result[-1] = replace(result[-1], notes=(
                "V196 adds only -sum(normalized action delta squared) per5ms "
                "tick after3 confirmed launch hits; zero for unchanged action. "
                "There is no action filtering, noise/entropy change or quiet bonus. "
                "All original V184 reward components and physical/control/obs/DR/"
                "reset/PPO/BC/non-hit gates remain exact; mean hits11--15."))
        if profile == PROFILES[6]:
            result[-1] = replace(result[-1], notes=(
                "V197 changes only absolute signed-height credit to1.25x. "
                "Adjacent/unresolved and all original V184 reward components "
                "remain exact; no damping/XY/action-variation additions. "
                "All physics/control/obs/DR/reset/PPO/BC/non-hit gates exact; "
                "mean hits11--15 and full survival1.0."))
    return result


def validate_resume_payload(payload: Mapping[str,Any], *, profile: str,
                            requested_start_stage: str, resume_curriculum_state: bool,
                            reset_optimizer: bool, reset_critic: bool) -> None:
    """Reject resets, unrelated parents, stale history and bounded probe promotion."""
    if (profile not in PROFILES or reset_optimizer or reset_critic
            or payload.get('opt') is None
            or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,477,7)):
        raise ValueError("Pinned full actor/critic/Adam72/477/7 required")
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        if (payload.get('stage_name')!=PARENT_STAGE or payload.get('stage_update')!=PARENT_UPDATE
                or payload.get('step')!=PARENT_STEP or payload.get('stop_reason')!='received SIGINT'
                or resume_curriculum_state or requested_start_stage!=start_stage(profile)):
            raise ValueError("Pinned interrupted V184 and fresh contract history required")
    elif source==profile:
        if payload.get('args',{}).get('max_stage_updates')!=-1:
            raise ValueError("Bounded probe actors cannot be promoted as formal resumes")
        if resume_curriculum_state:
            if payload.get('stage_name')!=requested_start_stage:
                raise ValueError("Own resume requires exact saved history/stage")
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(payload,next_stage=requested_start_stage)
    else:
        raise ValueError("Unrelated parent or rejected reward checkpoint")
