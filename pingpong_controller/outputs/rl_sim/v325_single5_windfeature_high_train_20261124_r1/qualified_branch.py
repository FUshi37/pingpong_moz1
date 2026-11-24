"""Pinned rollback with exact physical/learner carry and only a smaller PPO LR."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile


def compatible_parent_contract(old: dict, new: dict, expected_sha: str, *, calm_guard=None) -> dict:
    if old.get('plan_sha256')!=expected_sha:
        raise ValueError('Wrong qualified parent plan')
    expected={**old,'plan_sha256':new.get('plan_sha256')}
    if calm_guard is not None:
        expected['calm_trust_region']=calm_guard
    if expected!=new:
        raise ValueError('Qualified rollback changed a non-plan carry contract')
    return old


def verify_qualified_parent(plan: dict) -> tuple[dict, dict]:
    branch=plan['qualified_branch']
    for field in ('model','full_state','parent_plan','decision','analysis'):
        path=Path(branch[field])
        if path.is_symlink() or not path.is_file():
            raise ValueError('Qualified parent evidence is not a regular file')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream,'sha256').hexdigest()!=branch[field+'_sha256']:
                raise ValueError('Qualified parent evidence hash mismatch: '+field)
    decision=json.loads(Path(branch['decision']).read_text())['candidate_decisions'][str(branch['actual_updates'])]
    if not decision['accepted'] or decision['failures']:
        raise ValueError('The selected earlier checkpoint did not pass its original metrics')
    old_plan=json.loads(Path(branch['parent_plan']).read_text())
    expected_guard={**old_plan['calm_trust_region'],'enabled':True,
        'maximum_initial_policy_kl':.00025,
        'projection_origin':'qualified_anchor','phase_grouping':True,
        'anchor_model':branch['model'],'anchor_model_sha256':branch['model_sha256']}
    if plan['ppo']['learning_rate']!=old_plan['ppo']['learning_rate']:
        raise ValueError('Qualified projection retains the original learning rate')
    if plan['calm_trust_region']!=expected_guard:
        raise ValueError('Unreviewed cumulative policy constraint')
    left=json.loads(json.dumps(old_plan));right=json.loads(json.dumps(plan))
    for key in ('revision','qualified_branch','motivation','probe_update_budget','probe_total_update_limit'):
        left.pop(key,None);right.pop(key,None)
    right['ppo']['learning_rate']=left['ppo']['learning_rate']
    right['calm_trust_region']=left['calm_trust_region']
    right['arms']['direct']['run_id']=left['arms']['direct']['run_id']
    if left!=right:
        raise ValueError('A setting besides the learning rate or run identity changed')
    with ZipFile(branch['full_state']) as archive:
        manifest=json.loads(archive.read('manifest.json'))
    if manifest['metadata']['actual_updates']!=branch['actual_updates']:
        raise ValueError('Wrong qualified update in native carry')
    return manifest['contract'],dict(qualified_update=branch['actual_updates'],
        source_model_sha256=branch['model_sha256'],source_full_sha256=branch['full_state_sha256'],
        same_actor_critic_all_Adam_physics_RNG_and_statistics=True,
        learning_rate_before=old_plan['ppo']['learning_rate'],learning_rate_after=plan['ppo']['learning_rate'],
        training_projection=expected_guard,
        selection='Earliercheckpointpassedalloriginalmetrics;latestfailedandisnotused')
