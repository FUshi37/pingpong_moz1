"""Authenticate and freeze an isolated V226 experiment from passed V225S5."""
import argparse
from dataclasses import asdict,is_dataclass
import hashlib
import json
from pathlib import Path
import shlex
import shutil
from fixed_plane_v226_stability_recovery import PROFILE,PARENT_PROFILE,PARENT_SHA256,START_STAGE,validate_resume_payload
from preflight_fixed_plane_v225 import authenticate,course
from freeze_training_source import freeze_source

ROOT=Path('/home/yangzhe/Project/pingpong_controller')
OUTPUT=ROOT/'pingpong_controller/outputs/rl_sim'
OLD=OUTPUT/'fplc_v225_gpu1_near_side_1024x128_seed20266510_20261016_r1'
DESIGN=OUTPUT/'v226_stability_recovery_design_20261018_r1'


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_hashes():
    sim=Path(__file__).resolve().parent
    names=['fixed_plane_v226_stability_recovery.py','prepare_fixed_plane_v226.py',
        'evaluate_fixed_plane_v226.py','run_gpu1_fixed_plane_v226.sh',
        'assess_fixed_plane_v226.py',
        'mjx_juggle_env.py','train_juggle_mjx_curriculum.py','validate_juggle_mjx_ppo.py',
        'train_juggle_mjx_ppo.py','fixed_plane_v225_near_side.py',
        'fixed_plane_v166_ball_xy_reward.py','references/v225_mocap_near_side_templates.json']
    return {str(sim/name):digest(sim/name) for name in names}


def prepare(run,launcher,kind,admission,seed):
    run=run.resolve()
    if run.exists() or run.parent!=OUTPUT:raise ValueError('V226 needs a new output directory')
    source=OLD/'05_fplc_v225_s05_placed_xy_08mm.pkl'
    payload,sha=authenticate(source,PARENT_SHA256)
    start=START_STAGE if kind=='formal' else 'fplc_v226_s10_near_y_050mm'
    validate_resume_payload(payload,requested_start_stage=start,
        resume_curriculum_state=False,reset_optimizer=False,reset_critic=False,
        allow_obs_dim_migration=False,allow_bounded_probe_start=kind=='probe')
    if int(payload['opt'].t)!=362007:raise ValueError('V226 source Adam differs')
    saved={k:asdict(v) if is_dataclass(v) else v for k,v in payload['env_cfg'].items()}
    if saved!=asdict(course(PARENT_PROFILE)[4].cfg):raise ValueError('Original S5 configuration differs')
    baseline_path=DESIGN/'paired_baseline_cuda/manifest.json'
    baseline=json.loads(baseline_path.read_text())
    if len(baseline['results'])!=12:raise ValueError('Complete paired baseline required')
    for name,expected in baseline['artifact_hashes'].items():
        if digest(name)!=expected:raise ValueError('Paired baseline changed')
    if kind=='formal':
        if admission is None:raise ValueError('V226 formal training needs probe admission')
        accepted=json.loads(admission.read_text())
        if (not accepted.get('accepted') or accepted.get('profile')!=PROFILE
                or accepted.get('source_sha256')!=PARENT_SHA256
                or accepted.get('core_hashes')!=core_hashes()):
            raise ValueError('V226 current-source admission rejected')
        for name,expected in accepted['artifact_hashes'].items():
            if digest(name)!=expected:raise ValueError('V226 admission evidence changed')
    stages=course(PROFILE)
    run.mkdir()
    for name in ['matplotlib','wandb','wandb_cache','wandb_config']:(run/name).mkdir()
    freeze_source(repo_root=ROOT,run_dir=run,launcher=launcher)
    frozen=run/'source_snapshot/pingpong_controller/tools/rl_sim'
    command=shlex.split((OLD/'command.sh').read_text())
    script=next(i for i,x in enumerate(command) if Path(x).name=='train_juggle_mjx_curriculum.py')
    command[script]=str(frozen/'train_juggle_mjx_curriculum.py')
    replacements={'--save-dir':str(run),'--xml':str(frozen/'moz1_pd_1ms_v98.xml'),
        '--curriculum-profile':PROFILE,'--resume-from':str(source),'--resume-start-stage':start,
        '--max-stages':'17','--max-stage-updates':'32' if kind=='probe' else '-1',
        '--advance-retention-stage-name':START_STAGE,'--wandb-id':run.name,
        '--wandb-name':run.name,'--seed':str(seed)}
    for key,value in replacements.items():command[command.index(key)+1]=value
    tag_index=command.index('--wandb-tags')
    command[tag_index+1:]=['gpu1','fixed-plane-v226','protected-stability',
        'near-side-recovery','unchanged72d','passed-v225-s5','preserved-physics',kind]
    assert '--resume-curriculum-state' not in command
    (run/'command.sh').write_text('exec '+shlex.join(command)+'\n')
    if admission:shutil.copy2(admission,run/'formal_admission.json')
    report=dict(profile=PROFILE,kind=kind,source_checkpoint=str(source),source_sha256=sha,
        source_step=payload['step'],source_adam_t=int(payload['opt'].t),dimensions=[72,393,7],
        start_stage=start,stages=17,n_envs=1024,n_steps=128,seed=seed,
        restore_actor_critic_adam=True,reset_curriculum_history=True,
        bounded_probe_promoted=False,core_hashes=core_hashes(),
        baseline_manifest=str(baseline_path),baseline_sha256=digest(baseline_path),
        memory_high_gib=11,memory_max_gib=12,memory_stop_gib=10.5,
        lessons=[dict(index=i+1,name=s.name,min_updates=s.min_updates,
            near_y_cap_m=s.cfg.v225_near_y_max_m,nominal_fraction=s.cfg.v225_nominal_fraction,
            momentum_delta_m_s=s.cfg.v225_momentum_delta_limit_m_s,
            ball_mean_limit_m_s=s.max_recent_mean_hit_vxy,
            ball_rms_limit_m_s=s.max_recent_rms_hit_vxy) for i,s in enumerate(stages)])
    (run/'stability_preflight.json').write_text(json.dumps(report,indent=2)+'\n')
    print('V226_PREFLIGHT',kind,start,'source=S5u129','dims=72/393/7','envs=1024x128',flush=True)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--run-dir',type=Path,required=True)
    cli.add_argument('--launcher',type=Path,required=True)
    cli.add_argument('--kind',choices=['probe','formal'],required=True)
    cli.add_argument('--admission',type=Path)
    cli.add_argument('--seed',type=int,default=20266810)
    args=cli.parse_args();prepare(args.run_dir,args.launcher,args.kind,args.admission,args.seed)


if __name__=='__main__':main()
