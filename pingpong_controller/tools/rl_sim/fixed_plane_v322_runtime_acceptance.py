"""Opt-in V322 central-longevity and native in-flight recovery admission.

This module never changes training or the historical five-condition gates.
The user-approved admission has its own scope and preserves the old status.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from fixed_plane_v322_long_episode import QUALITY_FIELDS, quality_passed, quality_summary
from fixed_plane_v322_condition_gates import CONDITION_NAMES

ACCEPTANCE_SCOPE = 'center_runtime_recovery'
LONG_SECONDS = 1200.
CONTROL_SECONDS = 60.
RUNTIME_SCHEDULE_STEP = 1000
NATIVE_RECOVERY_SETTINGS = dict(
    v225_enabled=True,
    v225_momentum_delta_limit_m_s=.08,
    v225_momentum_target_cap_m_s=.11,
    v225_min_hit3_rate=.95,
    v225_contraction_ratio=.8,
    v225_min_contraction_rate=.75,
)


def primary_gate_report(record: dict, *, source_update: int, source_step: int,
                        updates_seen: int) -> dict:
    """Recompute admission from a real logged update, retaining every group.

    The 1200s clock and true step must agree with the unchanged1024x128
    learner. This is separate from, and never overwrites, original S4 gates.
    """
    def at_least(name, floor):
        value = record.get(name, float('nan'))
        return math.isfinite(value) and value >= floor

    update = record.get('update', -1)
    elapsed_s = (update-source_update)*128*.005
    quality = {key.removeprefix('quality/'):value
               for key, value in record.items() if key.startswith('quality/')}
    conditions = {
        name:{key.removeprefix(f'condition/{name}/'):value
              for key, value in record.items()
              if key.startswith(f'condition/{name}/')}
        for name in CONDITION_NAMES
    }
    checks = dict(
        stage4=record.get('stage') == 4,
        horizon_1200s=record.get('episode_limit_s') == LONG_SECONDS,
        clock_complete=elapsed_s >= LONG_SECONDS,
        true_step_continuous=record.get('global_step') == (
            source_step+(update-source_update)*131072),
        full_quality_window=updates_seen >= 32,
        original_quality=quality_passed(quality),
        nominal_samples=at_least('condition/nominal/n',64),
        overall_samples=at_least('endpoint/n',1024),
        nominal_survival=at_least('condition/nominal/survival',.93),
        nominal_length=at_least('condition/nominal/len_fraction',.95),
        nominal_hit_density=at_least('condition/nominal/hits_mean',1.2*LONG_SECONDS),
        every_minibatch_applied=record.get('ppo_minibatches_applied') == 32,
        no_rejected_update=record.get('ppo_candidates_rejected') == 0,
        no_rollback=record.get('ppo_update_rolled_back') == 0,
        finite_simulation=record.get('metric/v322/finite') == 1.,
    )
    return dict(scope=ACCEPTANCE_SCOPE,passed=all(checks.values()),checks=checks,
        update=update,elapsed_s=elapsed_s,quality=quality,
        nominal=conditions['nominal'],conditions=conditions,
        extreme_initial_conditions={name:conditions[name]
            for name in ('near_50_80','near_80_100')},
        original_all_five_passed=record.get('condition/all_gates_passed') == 1,
        frozen_runtime_acceptance_pending=True)


@dataclass
class ConsecutivePrimaryAdmission:
    """Count actual consecutive updates, never repeated observer snapshots."""
    source_update: int
    source_step: int
    last_update: int | None = None
    updates_seen: int = 0
    hold: int = 0

    def add(self, record: dict) -> dict:
        expected = self.source_update+1 if self.last_update is None else self.last_update+1
        if record.get('update') != expected:
            raise ValueError('Admission requires consecutive actual updates')
        self.last_update = expected
        self.updates_seen += 1
        report = primary_gate_report(record,source_update=self.source_update,
            source_step=self.source_step,updates_seen=self.updates_seen)
        self.hold = self.hold+1 if report['passed'] else 0
        return {**report,'hold':self.hold,'required_hold':16,
                'ready_for_frozen_evaluation':self.hold >= 16}


def validate_native_recovery_config(cfg: dict) -> None:
    """Refuse a changed inherited stress/recovery contract."""
    if any(cfg.get(key) != value for key,value in NATIVE_RECOVERY_SETTINGS.items()):
        raise ValueError('V322 runtime evaluation requires the pinned native recovery config')


def native_evaluation_metrics(after,obs,native_reward,metrics,*,xp):
    """Read native state/Q metrics without compiling unused training shaping.

    Frozen actions depend only on unchanged observations, never the reward.
    Match V322's recurrent velocity accounting; additionally require finite
    physical positions/velocities as well as observations/native reward.
    No input/state/observation or native termination is changed.
    """
    packet = dict(metrics)
    recurrent = after.hit_count>=2
    packet['recurrent_hit_racket_vxy_sum'] = xp.where(
        recurrent,metrics['hit_racket_vxy_sum'],0.)
    packet['recurrent_hit_racket_vxy_sq_sum'] = xp.where(
        recurrent,metrics['hit_racket_vxy_sq_sum'],0.)
    packet['v322/finite'] = (xp.all(xp.isfinite(obs),axis=-1)
        & xp.all(xp.isfinite(after.data.qpos),axis=-1)
        & xp.all(xp.isfinite(after.data.qvel),axis=-1)
        & xp.isfinite(native_reward)).astype(xp.float32)
    return packet


def runtime_momentum_intervention(native,velocity,target_vy,*,group,
        already_applied,step,scheduled_step,hit_count,in_contact,clearance_m,
        maximum_delta_m_s: float,kick_mask=None,xp):
    """Explicit evaluation-only arguments at the native intervention edge.

    Default off calls the original function with identical arguments. Opt-in
    does not overwrite any group, target, pose, history or observation state.
    Native eligibility, single-event accounting and clipping stay authoritative.
    Inputs are velocity(N,3) and matching(N,) fields.
    """
    if kick_mask is not None:
        if maximum_delta_m_s != NATIVE_RECOVERY_SETTINGS['v225_momentum_delta_limit_m_s']:
            raise ValueError('Runtime stress must retain the native velocity clip')
        if kick_mask.shape != group.shape or velocity.shape != (len(group),3):
            raise ValueError('Runtime stress requires matching lane shapes')
        group = xp.where(kick_mask,xp.ones_like(group),group)
        target_vy = xp.where(kick_mask,
            NATIVE_RECOVERY_SETTINGS['v225_momentum_target_cap_m_s'],target_vy)
        scheduled_step = xp.where(kick_mask,
            xp.full_like(step,RUNTIME_SCHEDULE_STEP),scheduled_step)
    return native(velocity,target_vy,group=group,already_applied=already_applied,
        step=step,scheduled_step=scheduled_step,hit_count=hit_count,
        in_contact=in_contact,clearance_m=clearance_m,
        maximum_delta_m_s=maximum_delta_m_s)


def frozen_runtime_report(packet: dict) -> dict:
    """Recompute four eight-case groups from unfiltered physical evidence.

    Quality uses original event-sum definitions. Every failed or unexposed
    case stays in its eight-case denominator. Declared pass flags are ignored.
    """
    import numpy as np

    fields = QUALITY_FIELDS + (
        'alive','recent','duration_s','hits','numerical_error',
        'kick_event_count','kick_delta_m_s','kick_time_s','kick_previous_hits',
        'kick_in_contact','kick_clearance_m','kick_post_vy_m_s','postkick_hits',
        'first_postkick_abs_ey_m','third_postkick_abs_ey_m','initial_group',
    )
    arrays = {key:np.asarray(packet[key]) for key in fields}
    if any(value.shape != (32,) for value in arrays.values()):
        raise ValueError('Frozen runtime admission requires32 physical cases')
    error = np.asarray(packet['pre_event_pair_max_error'])
    frames = np.asarray(packet['pre_event_pair_frames'])
    if error.shape != (16,) or frames.shape != (16,):
        raise ValueError('Frozen runtime admission requires16 paired controls')
    complete = packet.get('simulated_ticks') == 12000
    checks = dict(
        finite=all(np.isfinite(value).all() for value in arrays.values())
            and bool(np.isfinite(error).all()) and bool(np.isfinite(frames).all()),
        no_numerical_failures=bool((arrays['numerical_error']==0.).all()),
        central_initial_conditions=bool((arrays['initial_group']==0.).all()),
        paired_pre_event_identity=bool((error==0.).all()),
        paired_pre_event_coverage=bool((frames>=RUNTIME_SCHEDULE_STEP-1).all()),
        valid_duration=bool(((arrays['duration_s']>=0.)
            & (arrays['duration_s']<=CONTROL_SECONDS+.005)).all()),
        controls_no_intervention=bool((arrays['kick_event_count'][:16]==0.).all()
            and (arrays['kick_delta_m_s'][:16]==0.).all()),
    )
    results = {}
    for arm in range(2):
        for mode,name in enumerate(('deterministic','stochastic')):
            selection = slice(arm*16+mode*8,arm*16+(mode+1)*8)
            q = quality_summary({key:float(arrays[key][selection].sum())
                                 for key in QUALITY_FIELDS})
            survival = float(((arrays['alive'][selection]>.5)
                & (arrays['recent'][selection]>.5)).mean())
            fraction = float(np.minimum(arrays['duration_s'][selection]/CONTROL_SECONDS,1.).mean())
            density = float(arrays['hits'][selection].mean()/CONTROL_SECONDS)
            group_checks = dict(quality=quality_passed(q),survival=survival>=.875,
                length=fraction>=.95,hit_density=density>=1.2)
            recovery = {}
            if arm:
                exposed = ((arrays['kick_event_count'][selection]==1.)
                    & (arrays['kick_delta_m_s'][selection]>=.04-1e-6)
                    & (arrays['kick_delta_m_s'][selection]<=.08+1e-6)
                    & (arrays['kick_time_s'][selection]>=5.-1e-6)
                    & (arrays['kick_previous_hits'][selection]>=3.)
                    & (arrays['kick_in_contact'][selection]==0.)
                    & (arrays['kick_clearance_m'][selection]>=.05-2e-5)
                    & (arrays['kick_post_vy_m_s'][selection]<=.11+1e-6))
                post3 = arrays['postkick_hits'][selection]>=3.
                contraction = post3 & (arrays['third_postkick_abs_ey_m'][selection]
                    <= np.maximum(.015,.8*arrays['first_postkick_abs_ey_m'][selection]))
                group_checks.update(positive_native_exposure=bool(exposed.all()),
                    postkick_hit3=float(post3.mean())>=.95)
                recovery = dict(exposure_rate=float(exposed.mean()),
                    postkick_hit3_rate=float(post3.mean()),
                    contraction_rate=float(contraction.mean()),
                    native_three_hit_contraction_passed=float(contraction.mean())>=.75,
                    contraction_is_diagnostic=True)
            results[f'{"recovery" if arm else "control"}/{name}'] = dict(
                n=8,quality=q,survival=survival,len_fraction=fraction,
                hits_mean=float(arrays['hits'][selection].mean()),
                hit_density_per_s=density,checks=group_checks,
                passed=all(group_checks.values()),**recovery)
    return dict(scope=ACCEPTANCE_SCOPE,complete=bool(complete),
        observed_seconds=packet.get('simulated_ticks',0)*.005,
        checks=checks,groups=results,
        passed=bool(complete and all(checks.values())
                    and all(row['passed'] for row in results.values())),
        cases=32,paired_controls=16,all_failures_retained=True,
        original_all_five_admission_claim=False)
