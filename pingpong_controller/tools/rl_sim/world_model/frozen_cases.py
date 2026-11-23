"""Policy-independent fixed128-case course manifests; no admission filtering."""

from __future__ import annotations

import numpy as np

from .contracts import config_hash


GROUPS=('nominal','near_side','mirror','boundary')


def _stratified(rng,n,lower,upper):
    unit=np.stack([(rng.permutation(n)+rng.random(n))/n for _ in range(3)],axis=1)
    return lower+unit*(upper-lower)


def _next_goal(rng,current,lower,upper,distance_range):
    low,high=distance_range
    for _ in range(4096):
        if low==0.:
            proposed=rng.uniform(lower,upper)
        else:
            direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
            proposed=current+direction*rng.uniform(low,high)
        distance=np.linalg.norm(proposed-current)
        if np.all(proposed>=lower) and np.all(proposed<=upper) and low<=distance<=high and distance>1e-8:
            return proposed
    raise RuntimeError('Cannot fill a fixed goal sequence; never silently shorten it')


def generate_cases(cfg,*,seed):
    """Freeze proposals before geometry checks or any policy evaluation.

    Each axis of each independent group is stratified. Boundary cases balance
    the six faces6/6/5/5/5/5, with positions within2cm on the assigned face.
    All stages share starts; S04/S05 share their complete physical task list.
    """
    rng=np.random.default_rng(seed)
    lower=np.array(cfg['domain']['task_goal_xyz_lower_m']);upper=np.array(cfg['domain']['task_goal_xyz_upper_m'])
    starts={'nominal':_stratified(rng,32,lower,upper),
        'near_side':_stratified(rng,32,np.maximum(lower,[-np.inf,-.38,-np.inf]),upper),
        'boundary':_stratified(rng,32,lower,upper)}
    faces=np.arange(32)%6;rng.shuffle(faces)
    for i,face in enumerate(faces):
        axis,side=divmod(int(face),2)
        starts['boundary'][i,axis]=(lower[axis]+rng.uniform(0.,.02) if side==0
                                    else upper[axis]-rng.uniform(0.,.02))
    stages={}
    for stage,length,bounds in (('S01',1,(0.,0.)),('S02',2,(.02,.08)),
                                ('S03',2,(.08,.20)),('S04',4,(0.,.20))):
        grouped={}
        for group in ('nominal','near_side','boundary'):
            grouped[group]=[]
            for index,point in enumerate(starts[group]):
                goals=[point.copy()]
                for _ in range(length-1): goals.append(_next_goal(rng,goals[-1],lower,upper,bounds))
                grouped[group].append(dict(case_id=f'{group}_{index:02d}',group=group,
                    goals_xyz_m=np.array(goals).tolist(),mirror_of=None,
                    boundary_face=int(faces[index]) if group=='boundary' else None,
                    admission='NOT_AUDITED',transition_admission='UNKNOWN_dynamic_not_policy_selected'))
        grouped['mirror']=[dict(row,case_id=f'mirror_{i:02d}',group='mirror',
            mirror_of=row['case_id'],goals_xyz_m=(np.array(row['goals_xyz_m'])*[-1.,1.,1.]).tolist())
            for i,row in enumerate(grouped['nominal'])]
        stages[stage]=[row for group in GROUPS for row in grouped[group]]
    import copy
    stages['S05']=copy.deepcopy(stages['S04'])
    payload=dict(schema='wmj_v1_frozen_cases',training_version='WMJ-V1',seed=int(seed),
        cases_per_stage=128,stages=stages,distribution=cfg['frozen_evaluation'],
        physics_sha256=config_hash(cfg['physics']),control_sha256=config_hash(cfg['control']),
        final_test_used=False,policy_outputs_used=False)
    payload['cases_sha256']=config_hash(stages)
    return payload


def validate_case_manifest(manifest,cfg):
    """Recompute the entire deterministic manifest; reject altered cases/order."""
    if (manifest['seed']!=cfg['frozen_evaluation']['seed']
            or manifest!=generate_cases(cfg,seed=manifest['seed'])):
        raise ValueError('Frozen case manifest or task/distribution identity changed')
