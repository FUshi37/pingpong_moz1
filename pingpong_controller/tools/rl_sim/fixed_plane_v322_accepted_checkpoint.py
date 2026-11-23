"""Package explicit new-task acceptance without rewriting legacy graduation."""
from __future__ import annotations

import copy
import hashlib
import pickle

import numpy as np

from fixed_plane_v322_long_episode import PROFILE, GPU_UUID, XML_SHA256, loss_settings
from fixed_plane_v322_runtime_acceptance import (
    ACCEPTANCE_SCOPE, ConsecutivePrimaryAdmission, frozen_runtime_report,
)


def numeric_state_digest(value) -> str:
    """Hash finite numerical trees canonically, independent of pickle memo IDs."""
    digest = hashlib.sha256()

    def visit(node):
        if node is None:
            digest.update(b'none')
        elif isinstance(node,dict):
            digest.update(b'dict')
            for key in sorted(node):
                digest.update(str(key).encode())
                visit(node[key])
        elif isinstance(node,(list,tuple)):
            digest.update(b'sequence')
            for item in node:
                visit(item)
        elif hasattr(node,'__dict__'):
            visit(vars(node))
        else:
            array = np.asarray(node)
            if array.dtype.kind not in 'biufc' or not np.isfinite(array).all():
                raise ValueError('Accepted source contains invalid numerical state')
            digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes(order='C'))

    visit(value)
    return digest.hexdigest()


def validate_primary_source(source_bytes: bytes,source_origin: dict,
        history_records) -> tuple[dict,dict]:
    """Validate the exact latest full state and every actual learner update."""
    source = pickle.loads(source_bytes)
    if (source.get('v322_contract',{}).get('profile') != PROFILE
            or source.get('stage_index') != 3
            or tuple(source.get(k) for k in ('obs_dim','critic_obs_dim','act_dim')) != (72,393,7)
            or 'v322_acceptance' in source):
        raise ValueError('Expected an original complete V322 S4 training checkpoint')
    if any(source.get('args',{}).get(k) != value
           for k,value in loss_settings('anchor1').items()):
        raise ValueError('Accepted source does not preserve the authorized PPO contract')
    admission = ConsecutivePrimaryAdmission(source_origin['update'],source_origin['step'])
    last = None
    for row in history_records:
        if (row.get('ppo_minibatches_applied') != 32
                or row.get('ppo_candidates_rejected') != 0
                or row.get('ppo_update_rolled_back') != 0
                or row.get('metric/v322/finite') != 1.):
            raise ValueError('Actual learner history violates the unfiltered-update contract')
        last = admission.add(row)
    if (last is None or not last['ready_for_frozen_evaluation']
            or last['update'] != source['global_update']
            or source['step'] != source_origin['step']+
                (source['global_update']-source_origin['update'])*131072
            or int(source['opt'].t) != source_origin['Adam_t']+
                (source['global_update']-source_origin['update'])*32):
        raise ValueError('Latest complete source lacks matched central1200s/hold evidence')
    saved = source.get('last_summary',{})
    if (saved.get('update') != last['update'] or saved.get('quality') != last['quality']
            or saved.get('conditions') != last['conditions']):
        raise ValueError('Source checkpoint summary and actual update evidence disagree')
    return source,last


def build_accepted_payload(source_bytes: bytes,source_origin: dict,
        history_records,frozen_metadata: dict,frozen_packet: dict) -> dict:
    """Require matched actual updates, complete primary hold and frozen data.

    Source bytes must be the task's trusted, normally saved local checkpoint.
    No actor, critic, Adam, old contract or old graduation flag is modified.
    The only new field is explicit `v322_acceptance` under the approved scope.
    """
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    source,last = validate_primary_source(source_bytes,source_origin,history_records)
    if (frozen_metadata.get('purpose') != 'final'
            or frozen_metadata.get('eligible_as_final_evidence') is not True
            or frozen_metadata.get('source_sha256') != source_sha
            or frozen_metadata.get('source_update') != source['global_update']
            or frozen_metadata.get('source_step') != source['step']
            or frozen_metadata.get('source_Adam_t') != int(source['opt'].t)
            or frozen_metadata.get('gpu_uuid') != GPU_UUID
            or not frozen_metadata.get('devices')
            or any(not device.lower().startswith('cuda') for device in frozen_metadata['devices'])
            or frozen_metadata.get('XML_sha256') != XML_SHA256):
        raise ValueError('Final CUDA evidence is incomplete or belongs to another source')
    recomputed = frozen_runtime_report(frozen_packet)
    if not recomputed['passed']:
        raise ValueError('Raw frozen control/recovery evidence does not pass')
    state_keys = ('params','opt','stage_actor_anchor_reference_params')
    before = {key:numeric_state_digest(source.get(key)) for key in state_keys}
    result = copy.deepcopy(source)
    result['v322_acceptance'] = dict(accepted=True,scope=ACCEPTANCE_SCOPE,
        source_checkpoint_sha256=source_sha,source_update=source['global_update'],
        source_step=source['step'],source_Adam_t=int(source['opt'].t),
        primary=last,frozen_control_and_runtime_recovery=recomputed,
        frozen_provenance=copy.deepcopy(frozen_metadata),numeric_state_sha256=before,
        original_stage_converged_preserved=source.get('stage_converged'),
        original_all_five_passed=last['original_all_five_passed'],
        original_course_not_relabelled=True,
        frozen1200s_policy_certification=False,
        extreme_initial_condition_challenge='Separate nonblocking evaluation; retain all outcomes')
    if any(numeric_state_digest(result.get(key)) != before[key] for key in state_keys):
        raise ValueError('Packaging modified numerical training state')
    return result
