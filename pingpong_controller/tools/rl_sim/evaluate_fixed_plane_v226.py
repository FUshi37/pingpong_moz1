"""Paired unchanged-layout stability/recovery evaluation with raw event evidence."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import jax
import numpy as np
import train_juggle_mjx_curriculum as trainer
from preflight_fixed_plane_v225 import authenticate,course
from fixed_plane_v225_near_side import recovery_gate
from fixed_plane_v226_stability_recovery import PROFILE


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--checkpoint',type=Path,required=True)
    cli.add_argument('--sha256',required=True)
    cli.add_argument('--out',type=Path,required=True)
    cli.add_argument('--profile',default=PROFILE)
    cli.add_argument('--cases',type=int,nargs='+',default=[1,10])
    cli.add_argument('--seeds',type=int,nargs='+',default=[20266801,20266802])
    cli.add_argument('--n-envs',type=int,default=288)
    cli.add_argument('--steps',type=int,default=1200)
    cli.add_argument('--save-full-episode-audit', action='store_true',
                     help='Save lossless initial-lane episode evidence; legacy default off.')
    args=cli.parse_args()
    if jax.default_backend()!='gpu':raise ValueError('CUDA required')
    payload,sha=authenticate(args.checkpoint,args.sha256)
    params=jax.tree.map(jax.numpy.asarray,payload['params'])
    settings=argparse.Namespace(**payload['args'])
    settings.curriculum_profile=args.profile
    settings.xml=Path(__file__).resolve().parent/'moz1_pd_1ms_v98.xml'
    settings.advance_eval_n_envs=args.n_envs;settings.advance_eval_steps=args.steps
    settings.advance_eval_min_episodes=args.n_envs//2
    settings.advance_eval_stochastic=True;settings.advance_eval_deterministic=False
    settings.advance_validation_mode='block';settings.bounded_validation_memory=False
    stages=course(args.profile)
    args.out.mkdir(parents=True,exist_ok=False)
    original=trainer._get_advance_validation_runtime;captured={}
    def capture(*a,**k):
        env,rollout=original(*a,**k)
        def run(params,key):
            result=rollout(params,key);captured['output']=result;return result
        return env,run
    trainer._get_advance_validation_runtime=capture
    manifest=dict(profile=args.profile,checkpoint=str(args.checkpoint.resolve()),
        source_sha256=sha,n_envs=args.n_envs,steps=args.steps,seeds=args.seeds,
        devices=[str(d) for d in jax.devices()],results=[])
    for index in args.cases:
        stage=stages[index-1];cache={}
        folder=args.out/f's{index:02d}';folder.mkdir()
        (folder/'config.json').write_text(json.dumps(asdict(stage.cfg),indent=2))
        for seed in args.seeds:
            print('EVAL_START',index,seed,flush=True)
            block=trainer.run_advance_validation(settings,1,[stage],params,
                jax.random.PRNGKey(seed),runtime_cache=cache)
            output=captured.pop('output')
            names=[name for name in output['metrics'] if name in
                {'terminated','truncated','hit_event_count','hit_count','ball_racket_anchor_xy_dist_m'}
                or any(t in name for t in ['v225/','v226/','hit_vxy','racket_vxy',
                    'racket_omega','apex','late_','recurrent_hit_event','contact_center',
                    'next_contact_anchor','early_racket_xy_anchor'])]
            arrays=jax.device_get({name:output['metrics'][name] for name in names})
            if args.save_full_episode_audit:
                from fixed_plane_v232_full_survival import episode_audit_arrays
                arrays.update(episode_audit_arrays(jax.device_get({
                    name: output[name] for name in
                    ('first_episode_done', 'episode_length', 'done', 'evaluation_active')})))
            np.savez_compressed(folder/f'metrics_{seed}.npz',**arrays)
            result=dict(case=index,seed=seed,block=block,
                        recovery=recovery_gate(arrays,stage.cfg))
            (folder/f'block_{seed}.json').write_text(json.dumps(result,indent=2))
            manifest['results'].append(result)
            (args.out/'progress.json').write_text(json.dumps(manifest,indent=2))
            print('EVAL_RESULT',index,seed,{k:block.get('advance_eval/'+k) for k in
                ['passed','mean_hits','mean_len_frac','mean_hit_vxy','rms_hit_vxy',
                 'late_physical_in_band_rate']},flush=True)
            del arrays,output
        cache.clear();jax.clear_caches();gc.collect()
    manifest['artifact_hashes']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.out.rglob('*') if p.is_file()}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
