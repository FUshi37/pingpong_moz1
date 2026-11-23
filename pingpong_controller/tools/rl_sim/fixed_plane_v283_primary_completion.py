"""User-prioritized overall stability completion, with real DR coverage."""
from dataclasses import replace
from typing import Any, Mapping, Sequence
import numpy as np
PROFILE='fixed_plane_limit_cycle_qvel_rmp_v283_primary_completion'
PARENT_PROFILE='fixed_plane_limit_cycle_qvel_rmp_v282_survival_gate'
START_STAGE='fplc_v281_s11_near_y_080mm'
RETENTION_STAGE='fplc_v281_s01_workspace_alignment'


def recovery_gate(metrics: Mapping[str,Any], cfg: Any, *, max_steps: int=1200) -> dict[str,float]:
    """Keep all initial lanes and failures; gate domain coverage, not cadence.

    The native validator separately ANDs this with unchanged overall physical
    survival, apex, XY, motion, view and safety gates. Conditional survival is
    reported honestly and is not redefined as success.
    """
    from fixed_plane_v282_survival_gate import recovery_gate as previous
    original=previous(metrics,cfg,max_steps=max_steps)
    if cfg.v225_near_y_max_m<.08:
        return original
    result={k.removesuffix('_survival_ok')+'_diagnostic_survival_ok' if k.endswith('_survival_ok') else k:v
            for k,v in original.items()}
    result['previous_survival_passed']=result['passed']
    a={k:np.asarray(v) for k,v in metrics.items()}
    done=(a['terminated']+a['truncated'])>.5
    active=(np.cumsum(done,axis=0)-done)==0
    hits=(a['hit_event_count']>.5)&active
    kicks=(a['v225/momentum_event']>.5)&active
    groups=a['v225/group'][0].astype(int);offsets=a['v225/reset_y_offset_m'][0]
    started=hits.sum(axis=0)>=3
    exposed=kicks.any(axis=0)
    first=kicks.argmax(axis=0)
    post3=exposed&((hits&(np.arange(len(hits))[:,None]>first[None,:])).sum(axis=0)>=3)
    passed=True
    def coverage(label,lanes,recovery):
        count=len(lanes);launches=int(started[lanes].sum())
        ok=count>=cfg.v225_min_group_episodes and launches>=cfg.v225_min_group_episodes
        result[label+'_initial_case_count']=float(count)
        result[label+'_real_hit3_count']=float(launches)
        if recovery and cfg.v225_momentum_delta_limit_m_s>0:
            actual=int(exposed[lanes].sum());following=int(post3[lanes].sum())
            result[label+'_real_exposure_count']=float(actual)
            result[label+'_real_postkick_hit3_count']=float(following)
            result[label+'_exposure_among_started']=actual/max(launches,1)
            ok=(ok and actual>=cfg.v225_min_group_episodes
                and following>=cfg.v225_min_group_episodes and actual>=.90*launches)
        result[label+'_coverage_ok']=float(ok)
        return bool(ok)
    for group in (0,1,2):
        passed=coverage(f'group{group}',np.flatnonzero(groups==group),group>0) and passed
    for index,(low,high) in enumerate(((.02,.05),(.05,.08),(.08,.10001)),1):
        if cfg.v225_near_y_max_m<=low:continue
        lanes=np.flatnonzero((groups==1)&(offsets>=low)&(offsets<high))
        passed=coverage(f'near_bin{index}',lanes,True) and passed
    result['passed']=float(passed)
    return result


def build_stages(source: Sequence[Any]) -> list[Any]:
    if len(source)!=17 or source[10].name!=START_STAGE:
        raise ValueError('V283 requires all17 identical V281 physical lessons')
    return [replace(s,notes=('V283 uses overall absolute primary stability as completion: '
        'length.96/full.94,apex1.37--1.41,XY.026/.030,unchanged safety/quality gates. '
        'Keep every difficult-cohort failure visible; conditional survival/P12/three-hit '
        'centering are diagnostics. Require all cohort/severity populations and>=8 real '
        'launches; for recovery require actual kick exposure>=90%of started cases and '
        '>=8 exposed/postkick-three-hit cases. All17 physical lessons/rewards/primary '
        'gates/V281 IDs and full causal history,72/393/7,numeric1024x128 PPO remain. '
        'No claim that difficult initialization is solved. Fresh independent/nominal '
        'validation; every minibatch,no update cap or forced advancement.')) for s in source]



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
        raise ValueError('V283 requires complete uncapped72/393/7 state')
    source = payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        u=int(payload.get('stage_update',0))
        tail=payload.get('stage_history_tail')
        if (not resume_curriculum_state or requested_start_stage!=START_STAGE
                or payload.get('stage_name')!=START_STAGE or payload.get('stage_index')!=11
                or u<126 or payload.get('step')!=3153723392+u*131072
                or int(payload['opt'].t)!=779422+u*32 or not isinstance(tail,list) or not tail
                or int(tail[-1]['stage_update'])!=u or tail[-1]['stage_name']!=START_STAGE):
            raise ValueError('V283 requires latest V281 full causal primary history; no fresh-window restart')
    elif source==PROFILE:
        if resume_curriculum_state:
            if requested_start_stage!=payload.get('stage_name'):
                raise ValueError('V283 requires exact own history')
        else:
            from fixed_plane_completed_resume import validate_completed_stage
            validate_completed_stage(dict(payload),next_stage=requested_start_stage)
    else:
        raise ValueError('V283 rejects another source or from-scratch launch')
