"""V233 accepts PPO transactions only after direct frozen-source nominal checks."""
from dataclasses import replace
import math

from assess_fixed_plane_v226 import LOWER_IS_BETTER,HIGHER_IS_BETTER,compare_blocks
from fixed_plane_v232_full_survival import (
    PARENT_PROFILE,PARENT_SHA256,REPLAY_SHA256,validate_anchor_contract,
    validate_training_contract as validate_base_training,
    full_episode_gate,retention_stage,relaxed_hit_admission,
)

PROFILE='fixed_plane_limit_cycle_qvel_rmp_v233'
START_STAGE='fplc_v233_s01_workspace_alignment'
PROBE_STAGE='fplc_v233_s08_near_y_020mm_bridge'
RETENTION_STAGE=START_STAGE
GUARD_SEEDS=(20266840,20266841)
SAFETY_KEYS=('advance_eval/arm_qvel_limit_exceed_fraction',
             'advance_eval/arm_qacc_limit_exceed_fraction')
GUARD_KEYS=LOWER_IS_BETTER+HIGHER_IS_BETTER+SAFETY_KEYS


def build_stages(source):
    if len(source)!=17 or source[0].name!='fplc_v232_s01_workspace_alignment':
        raise ValueError('V233 preserves all17 V232 full-survival lessons')
    return [replace(s,name=s.name.replace('fplc_v232_','fplc_v233_',1),
        notes=s.notes+' V233 additionally evaluates every atomic PPO proposal '
        'on unchanged nominal S1 stochastic lanes against immutable passedS5; '
        'each seed must preserve five motion metrics, hit/survival/view and '
        'joint exceedance fractions. Rejected actor/critic/Adam proposals '
        'are rolled back or fractionally backtracked. No policy-input change.')
        for s in source]


def attach_nominal(summary,blocks,full_passes):
    if len(blocks)!=2 or len(full_passes)!=2:
        raise ValueError('V233 requires both fixed nominal guard populations')
    result=dict(summary)
    for index,(block,full) in enumerate(zip(blocks,full_passes,strict=True)):
        for key in GUARD_KEYS:
            result[f'v233_nominal_seed{index}/{key}']=float(block.get(key,float('nan')))
        result[f'v233_nominal_seed{index}/full_episode_passed']=float(full)
    return result


def nominal_guard(candidate,reference):
    passed=True;regressions=0;audit={}
    for index in range(2):
        prefix=f'v233_nominal_seed{index}/'
        current={k:candidate.get(prefix+k,float('nan')) for k in GUARD_KEYS}
        source={k:reference.get(prefix+k,float('nan')) for k in GUARD_KEYS}
        pair=compare_blocks(current,source)
        ok=pair['noninferior']
        regressions+=len(pair['regressions'])
        for key in SAFETY_KEYS:
            a,b=float(current[key]),float(source[key])
            if not(math.isfinite(a) and math.isfinite(b) and a<=b):
                ok=False;regressions+=1
            pair['differences'][key]=a-b if math.isfinite(a) and math.isfinite(b) else None
        full=float(candidate.get(prefix+'full_episode_passed',0.))==1.
        if not full:regressions+=1
        passed=passed and ok and full
        for key,delta in pair['differences'].items():
            metric=key.removeprefix('advance_eval/')
            audit[f'transaction/v233_seed{index}_delta_{metric}']=float('nan') if delta is None else delta
        audit[f'transaction/v233_seed{index}_full_episode_passed']=float(full)
    audit['transaction/v233_nominal_guard_passed']=float(passed)
    audit['transaction/v233_nominal_regressions']=float(regressions)
    return bool(passed),audit


def combine_decision(base_passed,base_audit,candidate,reference):
    passed,audit=nominal_guard(candidate,reference)
    combined=bool(base_passed and passed)
    return combined,{**base_audit,**audit,'transaction/accepted':float(combined)}


def validate_paired_stability_contract(args):
    from fixed_plane_v232_full_survival import validate_paired_stability_contract as validate
    from types import SimpleNamespace
    validate(SimpleNamespace(**{**vars(args),'advance_retention_stage_name':
        'fplc_v232_s01_workspace_alignment'}))
    if args.advance_retention_stage_name!=RETENTION_STAGE:
        raise ValueError('V233 requires its original nominal S1 retention')


def validate_training_contract(args):
    validate_base_training(args)
    if (not args.transactional_main_metric_guard
            or args.transaction_interval_updates!=8 or args.transaction_eval_n_envs!=96
            or tuple(args.transaction_eval_seeds or ())!=GUARD_SEEDS
            or not args.transaction_backtrack_rejected_candidate
            or not args.transaction_revalidate_incumbent_on_rejected_boundary
            or not args.transaction_primary_value_nonregression
            or args.transaction_secondary_gap_tolerance!=0.
            or args.transaction_primary_full_rate_tolerance!=0.
            or args.same_stage_reexec_every_updates!=0):
        raise ValueError('V233 pins atomic8-update stochastic96x2 nominal source guards')


def validate_resume_payload(payload,*,requested_start_stage,resume_curriculum_state,
                            reset_optimizer,reset_critic,allow_obs_dim_migration,probe=False):
    source=payload.get('args',{}).get('curriculum_profile')
    if source==PARENT_PROFILE:
        from fixed_plane_v232_full_survival import validate_resume_payload as validate
        if not requested_start_stage.startswith('fplc_v233_'):
            raise ValueError('V233 requires an explicit new full-survival lesson')
        return validate(payload,requested_start_stage=requested_start_stage.replace('fplc_v233_','fplc_v232_',1),
            resume_curriculum_state=resume_curriculum_state,reset_optimizer=reset_optimizer,
            reset_critic=reset_critic,allow_obs_dim_migration=allow_obs_dim_migration,probe=probe)
    if (source!=PROFILE or probe or reset_optimizer or reset_critic or allow_obs_dim_migration
            or payload.get('opt') is None or payload.get('args',{}).get('max_stage_updates')!=-1
            or tuple(payload.get(k) for k in ('obs_dim','critic_obs_dim','act_dim'))!=(72,393,7)):
        raise ValueError('V233 rejects negative/capped/predecessor actors as formal parents')
    if resume_curriculum_state:
        if requested_start_stage!=payload.get('stage_name'):
            raise ValueError('V233 requires causal exact-stage history')
    else:
        from fixed_plane_completed_resume import validate_completed_stage
        validate_completed_stage(payload,next_stage=requested_start_stage)
