"""Cohort survival admission with preserved primary convergence and history."""
from dataclasses import replace
from typing import Any, Mapping, Sequence
import numpy as np

PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v282_survival_gate'
PARENT_PROFILE = 'fixed_plane_limit_cycle_qvel_rmp_v281_stable_completion'
# Physical lessons, rewards and primary gates are identical: retain their IDs
# so the existing causal rolling history can be restored without translation.
START_STAGE = 'fplc_v281_s11_near_y_080mm'
RETENTION_STAGE = 'fplc_v281_s01_workspace_alignment'
COHORT_MIN_LEN = .90
COHORT_MIN_FULL = .85
COHORT_MIN_HIT3 = .85


def recovery_gate(metrics: Mapping[str, Any], cfg: Any, *, max_steps: int = 1200) -> dict[str, float]:
    """Evaluate every initial lane; keep cadence/three-hit centering diagnostic.

    Arrays are (time, lanes). A full episode, real physical hits, exposure and
    post-kick continuation are distinct requirements. No failure or sample is
    excluded; first-episode accounting matches the preserved legacy gate.
    """
    from fixed_plane_v225_near_side import recovery_gate as legacy
    result = legacy(metrics, cfg)
    if cfg.v225_near_y_max_m < .08:
        return result
    a = {k:np.asarray(v) for k,v in metrics.items()}
    done = (a['terminated']+a['truncated']) > .5
    active = (np.cumsum(done,axis=0)-done)==0
    hits = (a['hit_event_count']>.5)&active
    kicks = (a['v225/momentum_event']>.5)&active
    groups = a['v225/group'][0].astype(int)
    offsets = a['v225/reset_y_offset_m'][0]
    n = len(groups)
    ended = done.any(axis=0)
    last = np.where(ended,done.argmax(axis=0),done.shape[0]-1)
    lengths = np.minimum(last+1,max_steps)/float(max_steps)
    full = ended & (a['truncated'][last,np.arange(n)]>.5)
    p3 = hits.sum(axis=0)>=3
    exposed = kicks.any(axis=0)
    post3 = np.zeros(n,dtype=bool)
    if cfg.v225_momentum_delta_limit_m_s>0:
        first_kick = kicks.argmax(axis=0)
        post3 = exposed & ((hits & (np.arange(len(hits))[:,None]>first_kick[None,:])).sum(axis=0)>=3)
    result['legacy_passed'] = result['passed']
    result.update(cohort_min_len_frac=COHORT_MIN_LEN,cohort_min_full_rate=COHORT_MIN_FULL,
                  cohort_min_hit3_rate=COHORT_MIN_HIT3)
    passed = True
    def assess(label: str, lanes: np.ndarray, *, recovery: bool, group_level: bool) -> bool:
        count = len(lanes)
        length = float(lengths[lanes].mean()) if count else 0.
        finish = float(full[lanes].mean()) if count else 0.
        launch = float(p3[lanes].mean()) if count else 0.
        ok = (count>=cfg.v225_min_group_episodes and length>=COHORT_MIN_LEN
              and finish>=COHORT_MIN_FULL and launch>=COHORT_MIN_HIT3)
        result.update({label+'_mean_len_frac':length,label+'_full_rate':finish,
                       label+'_physical_hit3_rate':launch})
        if recovery and cfg.v225_momentum_delta_limit_m_s>0:
            exposure = float(exposed[lanes].mean()) if count else 0.
            continuation = float(post3[lanes].mean()) if count else 0.
            result[label+'_actual_exposure_rate'] = exposure
            result[label+'_actual_postkick_hit3_rate'] = continuation
            ok = ok and continuation>=COHORT_MIN_HIT3
            if group_level:
                ok = ok and exposure>=.90
        result[label+'_survival_ok'] = float(ok)
        return bool(ok)
    for group in (0,1,2):
        passed = assess(f'group{group}',np.flatnonzero(groups==group),recovery=group>0,group_level=True) and passed
    for index,(low,high) in enumerate(((.02,.05),(.05,.08),(.08,.10001)),1):
        if cfg.v225_near_y_max_m<=low:
            continue
        lanes = np.flatnonzero((groups==1)&(offsets>=low)&(offsets<high))
        passed = assess(f'near_bin{index}',lanes,recovery=True,group_level=False) and passed
    result['passed'] = float(passed)
    return result


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source)!=17 or source[10].name!=START_STAGE:
        raise ValueError('V282 requires all17 identical V281 physical lessons')
    return [replace(s,notes=('V282 preserves V281 stage IDs/rewards/domains/primary gates and genuine '
        'rolling history. S11+ subgroup admission uses actual mean duration>=.90/full>=.85/'
        'physical hit3>=.85, minimum8initial lanes, recovery exposure>=.90 and postkick3>=.85. '
        'P12 and three-hit contraction remain diagnostics. Overall length.96/full.94, apex1.37--1.41 '
        'and XY.026/.030 and all safety gates remain. Fresh independent and nominal validation '
        'required; latest full actor/critic/Adam/history,1024x128,all minibatches,no cap.')) for s in source]


def validate_training_contract(settings: Any) -> None:
    from fixed_plane_v281_stable_completion import validate_training_contract as previous
    previous(settings)


def validate_resume_payload(payload: Mapping[str,Any], *, requested_start_stage: str,
        resume_curriculum_state: bool, reset_optimizer: bool, reset_critic: bool,
        allow_obs_dim_migration: bool) -> None:
    if (reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or not {'pi','v'}.issubset(payload.get('params',{}))
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)
            or payload.get('args',{}).get('max_stage_updates')!=-1):
        raise ValueError('V282 requires complete uncapped72/393/7 state')
    source = payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        u=int(payload.get('stage_update',0))
        tail=payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage!=START_STAGE
                or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=11
                or u<106 or payload.get('step')!=3153723392+u*131072
                or int(payload['opt'].t)!=779422+u*32 or not isinstance(tail,list) or not tail
                or int(tail[-1]['stage_update'])!=u or tail[-1]['stage_name']!=START_STAGE):
            raise ValueError('V282 requires latest V281 full causal primary history; no fresh-window restart')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V282 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V282 rejects another source or from-scratch launch')
