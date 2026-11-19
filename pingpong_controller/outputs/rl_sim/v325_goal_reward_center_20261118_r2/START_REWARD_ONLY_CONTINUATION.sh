#!/usr/bin/env bash
set -euo pipefail
reference_root=/home/yangzhe/Project/pingpong_controller
reference_audit=$reference_root/pingpong_controller/outputs/rl_sim/v325_goal_reward_center_20261118_r2
reference_python=/home/yangzhe/miniconda3/envs/pingpong/bin/python
"$reference_python" -B - "$reference_audit" <<'PY_CHECK'
from pathlib import Path
import hashlib,json,re,sys
p=Path(sys.argv[1])
for name in ('COMPONENT_TESTS.txt','REGRESSION_GREEN.txt'):
    value=(p/name).read_text()
    assert re.search(r'\d+ passed',value) and not re.search(r'\d+ failed',value),name
pair=json.loads((p/'CENTER_NATIVE_PAIR.json').read_text())
assert pair['checkpoint_full_learner_roundtrip'] and len(pair['cases'])==12
assert pair['reference_reset_exact']
assert all(c['physical_state_actor_critic_done_exact'] for c in pair['cases'].values())
for version in (325,326):
    courses=json.loads((p/f'ALL_COURSES_V{version}.json').read_text())
    assert len(courses)==32 and max(s['max_wind'] for s in courses)==4.
source=json.loads((p/'CONTINUATION_SOURCE.json').read_text())
assert source['all32_nonreward_configs_exact'] and source['complete_learner_roundtrip_exact']
assert hashlib.sha256((p/'continuation_source.pkl').read_bytes()).hexdigest()==source['sha256']
assert (source['global_update'],source['stage_index'])>=(12800,14)
pre=json.loads((p/'CONTINUATION_PREFLIGHT.json').read_text())
assert (pre['true_step'],pre['Adam_t'])==(source['true_step'],source['Adam_t'])
assert pre['wind_physics']['wind_y_nonpositive'] and pre['wind_physics']['mode_weights'][0]==.1
assert pre['learner']['n_envs']==1024 and pre['learner']['counterfactual_supervision_coef']==0
assert 'reference_constraints' in pre['binding']['training_mix']
assert 'clean_apex_preservation' in pre['binding']['training_mix']
plan=json.loads((p/'PLAN.json').read_text())
protected=plan['protected']
import ast
sim=Path('/home/yangzhe/Project/pingpong_controller/pingpong_controller/tools/rl_sim')
assert all(hashlib.sha256((sim/name).read_bytes()).hexdigest()==sha for name,sha in plan['frozen_files_sha256'].items())
tree=ast.parse((sim/'fixed_plane_wind_reference_constraints.py').read_text())
nodes={n.name:ast.dump(n,include_attributes=False) for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
assert all(nodes[k]==v for k,v in plan['frozen_reference_functions'].items())
assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest()==sha for name,sha in protected.items())
print('PASS unified motion references, independent height/center precision, complete learner, all64 courses')
PY_CHECK
cd "$reference_root"
export CONFIRM_GPU0_READY=YES ACKNOWLEDGE_CANDIDATE_WIND=YES
export V325_CHECKPOINT_RETENTION=m12_best V325_WANDB_UPLOAD=metrics_only
export V325_TRAINING_MIX=clean10_dynamics_reference_center_negy_v2 V325_SEED=2026102807
export V325_PROBE_UPDATES=0 V325_GUIDANCE_PROBE=NO
unset V325_ACTOR_BOOTSTRAP_ONLINE_RECIPE V325_QUALIFICATION V325_GUIDANCE_REFERENCE_JSON
exec bash pingpong_controller/tools/rl_sim/run_gpu0_fixed_plane_v325_wind.sh \
    --formal fplc_v325_gpu0_reference_center_20261118_r2 "$reference_audit/continuation_source.pkl"
