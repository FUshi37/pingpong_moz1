"""User-authorized quality repair with separately attributed long evidence."""
from dataclasses import dataclass
import math

from fixed_plane_v322_long_episode import quality_passed

LONG_SOURCE_SHA = '187739cdc9125de13afceae386788671df6010d6cec729df7351096e4fba0777'


def validate_long_qualification(source: dict, audit: dict, source_sha: str) -> dict:
    """Authenticate the completed u31201 qualification; never rewrite it."""
    p = audit.get('primary', {})
    nominal = p.get('nominal', {})
    if not (source_sha == audit.get('sha256') == LONG_SOURCE_SHA
            and audit.get('passed') is True and p.get('passed') is True
            and audit.get('update') == source.get('global_update') == 31201
            and audit.get('step') == source.get('step') == 8217296896
            and audit.get('Adam_t') == 2015646
            and source.get('stage_index') == 3
            and source.get('stage_converged') is False
            and p.get('hold', 0) >= 16
            and p.get('ready_for_frozen_evaluation') is True
            and nominal.get('n', 0) >= 64
            and nominal.get('survival', 0.) >= .93
            and nominal.get('len_fraction', 0.) >= .95):
        raise ValueError('Quality repair requires authenticated u31201 long qualification')
    return dict(inherited=True, source_sha256=source_sha,
        source_update=31201, source_step=8217296896,
        source_Adam_t=2015646, primary=p,
        fresh_1200s_endpoints_required=False, episode_limit_s=1200.,
        numerical_quality_thresholds_unchanged=True,
        qualification_kind='historical mixed-policy training, not frozen1200s')


def validate_quality_repair_resume(source: dict, audit: dict, source_sha: str) -> dict:
    """Continue the latest full repair state, preserving canonical ancestry."""
    from copy import deepcopy
    from fixed_plane_v322_long_episode import PROFILE, loss_settings
    if source_sha == LONG_SOURCE_SHA:
        result = validate_long_qualification(source,audit,source_sha)
    else:
        contract = source.get('v322_contract',{})
        result = deepcopy(contract.get('quality_repair',{}))
        u = source.get('global_update',0)
        p = audit.get('primary',{});nominal=p.get('nominal',{})
        if not (audit.get('sha256') == LONG_SOURCE_SHA
                and audit.get('passed') is True and p.get('passed') is True
                and audit.get('update') == 31201 and audit.get('step') == 8217296896
                and audit.get('Adam_t') == 2015646 and p.get('hold',0) >= 16
                and nominal.get('n',0) >= 64 and nominal.get('survival',0) >= .93
                and nominal.get('len_fraction',0) >= .95
                and result.get('inherited') is True
                and result.get('source_sha256') == LONG_SOURCE_SHA
                and result.get('source_update') == 31201
                and result.get('source_step') == 8217296896
                and result.get('source_Adam_t') == 2015646
                and contract.get('profile') == PROFILE
                and contract.get('reward_preset') in ('late_cycle_repair','late_center_precision')
                and source.get('stage_index') == 3 and source.get('stage_converged') is False
                and tuple(source.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) == (72,393,7)
                and u > 31201 and source.get('step') == 8217296896+(u-31201)*131072
                and int(source['opt'].t) == 2015646+(u-31201)*32
                and source.get('last_summary',{}).get('update') == u
                and all(source.get('args',{}).get(k)==v for k,v in loss_settings('anchor1').items())):
            raise ValueError('Repair continuation lost the full-state/PPO/long ancestry contract')
        result['primary'] = deepcopy(p)
    result['resume_origin'] = dict(update=source['global_update'],step=source['step'],
        Adam_t=int(source['opt'].t),sha256=source_sha)
    return result


@dataclass
class QualityRepairAdmission:
    """Require real consecutive PPO updates and the original quality window."""
    source_update: int
    source_step: int
    updates_seen: int = 0
    hold: int = 0

    def add(self, *, update: int, step: int, quality: dict,
            quality_window: int, loss: dict, finite: bool) -> dict:
        expected = self.source_update+self.updates_seen+1
        if update != expected or step != self.source_step+(update-self.source_update)*131072:
            raise ValueError('Quality repair requires consecutive real learner updates/steps')
        self.updates_seen += 1
        checks = dict(original_quality=quality_passed(quality),
            full_quality_window=quality_window == 32,
            all_minibatches=loss.get('ppo_minibatches_applied') == 32,
            no_rejection=loss.get('ppo_candidates_rejected') == 0,
            no_rollback=loss.get('ppo_update_rolled_back') == 0,
            finite=finite and all(math.isfinite(v) for v in loss.values()))
        passed = all(checks.values())
        self.hold = self.hold+1 if passed else 0
        return dict(scope='center_runtime_quality_repair', passed=passed,
            checks=checks, hold=self.hold, required_hold=16,
            ready_for_frozen_evaluation=self.hold >= 16,
            update=update, step=step, quality=quality,
            new_exposure_s=self.updates_seen*.64,
            long_qualification='inherited u31201; no new1200s clock')


def validate_quality_source_history(source_bytes: bytes, history_records) -> tuple[dict,dict]:
    """Authenticate current full state and every real training-quality update."""
    import pickle
    from fixed_plane_v322_long_episode import PROFILE, loss_settings

    source = pickle.loads(source_bytes)
    contract = source.get('v322_contract', {})
    inherited = contract.get('quality_repair', {})
    if (contract.get('profile') != PROFILE or source.get('stage_index') != 3
            or source.get('stage_converged') is not False
            or contract.get('reward_preset') not in ('late_cycle_repair','late_center_precision')
            or inherited.get('inherited') is not True
            or inherited.get('source_sha256') != LONG_SOURCE_SHA
            or inherited.get('source_update') != 31201
            or inherited.get('source_step') != 8217296896
            or inherited.get('source_Adam_t') != 2015646
            or 'v322_acceptance' in source
            or tuple(source.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)):
        raise ValueError('Expected exact inherited V322 late-cycle quality repair')
    if any(source.get('args',{}).get(k) != v for k,v in loss_settings('anchor1').items()):
        raise ValueError('Quality repair changed the authorized PPO settings')
    origin = inherited.get('resume_origin',dict(update=31201,step=8217296896,
                                              Adam_t=2015646,sha256=LONG_SOURCE_SHA))
    if (origin.get('update',0) < 31201
            or origin.get('step') != 8217296896+(origin['update']-31201)*131072
            or origin.get('Adam_t') != 2015646+(origin['update']-31201)*32
            or len(origin.get('sha256','')) != 64):
        raise ValueError('Invalid full-state repair continuation origin')
    admission = QualityRepairAdmission(origin['update'],origin['step'])
    last = None
    for row in history_records:
        loss = {k:row[k] for k in ('ppo_minibatches_applied','ppo_candidates_rejected',
                                   'ppo_update_rolled_back') if k in row}
        if (loss.get('ppo_minibatches_applied') != 32
                or loss.get('ppo_candidates_rejected') != 0
                or loss.get('ppo_update_rolled_back') != 0
                or row.get('metric/v322/finite') != 1.):
            raise ValueError('Repair history contains skipped/rejected/nonfinite updates')
        last = admission.add(update=row['update'],step=row['global_step'],
            quality={k[8:]:v for k,v in row.items() if k.startswith('quality/')},
            quality_window=min(admission.updates_seen+1,32),loss=loss,finite=True)
    if (not last or not last['ready_for_frozen_evaluation']
            or last['update'] != source['global_update'] or last['step'] != source['step']
            or int(source['opt'].t) != 2015646+(last['update']-31201)*32
            or source.get('last_summary',{}).get('quality') != last['quality']):
        raise ValueError('Current full state lacks matched fresh quality/Adam evidence')
    return source,last


def build_quality_accepted_payload(source_bytes: bytes, history_records,
        frozen_metadata: dict, frozen_packet: dict) -> dict:
    """Require actual unfiltered updates and matched original CUDA gates.

    Append only acceptance metadata; keep the actor, critic, optimizer,
    original course scope and graduation flags numerically unchanged.
    """
    import copy
    import hashlib
    from fixed_plane_v322_accepted_checkpoint import numeric_state_digest
    from fixed_plane_v322_long_episode import GPU_UUID, XML_SHA256
    from fixed_plane_v322_runtime_acceptance import frozen_runtime_report
    source,last = validate_quality_source_history(source_bytes,history_records)
    inherited = source['v322_contract']['quality_repair']
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    if (frozen_metadata.get('purpose') != 'final'
            or frozen_metadata.get('eligible_as_final_evidence') is not True
            or frozen_metadata.get('source_sha256') != source_sha
            or frozen_metadata.get('source_update') != source['global_update']
            or frozen_metadata.get('source_step') != source['step']
            or frozen_metadata.get('source_Adam_t') != int(source['opt'].t)
            or frozen_metadata.get('gpu_uuid') != GPU_UUID
            or frozen_metadata.get('XML_sha256') != XML_SHA256
            or not frozen_metadata.get('devices')
            or any(not d.lower().startswith('cuda') for d in frozen_metadata['devices'])):
        raise ValueError('Frozen quality evidence belongs to a different source or stack')
    frozen = frozen_runtime_report(frozen_packet)
    if not frozen['passed']:
        raise ValueError('Original frozen control/runtime quality gates failed')
    keys = ('params','opt','stage_actor_anchor_reference_params')
    digests = {k:numeric_state_digest(source.get(k)) for k in keys}
    result = copy.deepcopy(source)
    result['v322_acceptance'] = dict(accepted=True,scope='center_runtime_quality_repair',
        source_checkpoint_sha256=source_sha,source_update=source['global_update'],
        source_step=source['step'],source_Adam_t=int(source['opt'].t),
        historical_long_qualification=copy.deepcopy(inherited),fresh_quality=last,
        fresh_1200s_clock_required=False,episode_limit_s=1200.,
        frozen_control_and_runtime_recovery=frozen,
        frozen_provenance=copy.deepcopy(frozen_metadata),numeric_state_sha256=digests,
        original_stage_converged_preserved=False,original_course_not_relabelled=True,
        frozen1200s_policy_certification=False,
        extreme_initial_condition_challenge='Separate nonblocking evaluation')
    if any(numeric_state_digest(result.get(k)) != digests[k] for k in keys):
        raise ValueError('Acceptance packaging modified numerical training state')
    return result


def build_training_qualified_payload(source_bytes: bytes, history_records) -> dict:
    """Honor the user's training-only stop without claiming frozen acceptance."""
    import copy
    import hashlib
    from fixed_plane_v322_accepted_checkpoint import numeric_state_digest
    source,last = validate_quality_source_history(source_bytes,history_records)
    if 'v322_training_acceptance' in source:
        raise ValueError('Training qualification metadata is already present')
    keys = ('params','opt','stage_actor_anchor_reference_params')
    digests = {k:numeric_state_digest(source.get(k)) for k in keys}
    result = copy.deepcopy(source)
    result['v322_training_acceptance'] = dict(qualified=True,
        scope='training_quality_with_inherited_1200s',
        user_stop_instruction='新的训练只要在训练中所有指标满足要求就可以停止训练',
        source_checkpoint_sha256=hashlib.sha256(source_bytes).hexdigest(),
        source_update=source['global_update'],source_step=source['step'],
        source_Adam_t=int(source['opt'].t),fresh_training_quality=last,
        historical_1200s_qualification=copy.deepcopy(source['v322_contract']['quality_repair']),
        fresh_1200s_clock_required=False,episode_limit_s=1200.,
        all_original_numeric_training_gates_retained=True,
        current_policy_frozen_acceptance_claim=False,
        current_policy_frozen_status='not assessed; no inherited frozen pass',
        original_stage_converged_preserved=source.get('stage_converged'),
        original_course_not_relabelled=True,numeric_state_sha256=digests)
    if any(numeric_state_digest(result.get(k)) != digests[k] for k in keys):
        raise ValueError('Training qualification changed numerical state')
    return result


def close_then_quality_acceptance(source_bytes: bytes, frozen_metadata: dict,
        frozen_packet: dict, *, close_learner, read_closed_history) -> dict:
    """Close only a genuinely passed paused actor, then audit the SDK tail.

    A live SDK can retain its last record until normal cloud close. That
    record must be flushed and read, never synthesized from a saved summary.
    """
    import hashlib
    import pickle
    from fixed_plane_v322_runtime_acceptance import frozen_runtime_report
    source = pickle.loads(source_bytes)
    if (frozen_metadata.get('source_sha256') != hashlib.sha256(source_bytes).hexdigest()
            or frozen_metadata.get('source_update') != source.get('global_update')
            or frozen_metadata.get('source_step') != source.get('step')
            or frozen_metadata.get('source_Adam_t') != int(source['opt'].t)
            or frozen_metadata.get('purpose') != 'final'
            or frozen_metadata.get('eligible_as_final_evidence') is not True
            or not quality_passed(source.get('last_summary',{}).get('quality',{}))
            or not frozen_runtime_report(frozen_packet)['passed']):
        raise ValueError('Unmatched or failed quality evidence cannot request a stop')
    close_learner()
    return build_quality_accepted_payload(source_bytes,read_closed_history(),
                                          frozen_metadata,frozen_packet)
