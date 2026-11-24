"""Fixed-stage, per-direction/per-speed evidence; no controller or reward edits."""
from pathlib import Path
import sys

import numpy as np

SOURCE=Path(__file__).resolve().parent.parent/'v325_structural_ablations_20261119_r1/source'
sys.path.insert(0,str(SOURCE))
from fixed_plane_wind_training_gates import TrainingSegments


def group_indices(n_envs):
    lane=np.arange(n_envs)
    clean=lane<round(.1*n_envs)
    slot=np.maximum(lane-round(.1*n_envs),0)
    sector=slot%8
    peak=(slot//8+sector)%4==0
    labels=np.where(clean,0,1+2*sector+peak)
    names=['clean']+[f'sector{i:02d}_{kind}' for i in range(8) for kind in ('continuous','peak160')]
    return {name:np.flatnonzero(labels==i) for i,name in enumerate(names)}


class S15Progress:
    def __init__(self,n_envs,quality_fields,seconds=180.):
        self.indices=group_indices(n_envs)
        self.quality_fields=tuple(quality_fields)
        self.seconds=seconds
        self.groups={name:{
            'stage':TrainingSegments(seconds,.005,len(indices),quality_fields=self.quality_fields),
            'short30':TrainingSegments(30.,.005,len(indices),quality_fields=()),
            'full1200':TrainingSegments(1200.,.005,len(indices),quality_fields=())}
            for name,indices in self.indices.items()}

    def add(self,metrics,done):
        for name,indices in self.indices.items():
            modes=np.asarray(metrics['wind/mode'])[:,indices]
            assert np.all(modes==(0 if name=='clean' else 1)),name
            arguments=dict(hits=np.asarray(metrics['new_hit'])[:,indices],done=np.asarray(done)[:,indices],
                terminated=np.asarray(metrics['terminated'])[:,indices],
                recent=np.asarray(metrics['v321/recent_hit'])[:,indices],modes=modes)
            self.groups[name]['stage'].add(**arguments,quality_metrics={
                field:np.asarray(metrics[field])[:,indices] for field in self.quality_fields})
            self.groups[name]['short30'].add(**arguments)
            self.groups[name]['full1200'].add(**arguments)

    def summaries(self):
        result={}
        for name,trackers in self.groups.items():
            mode='clean' if name=='clean' else 'constant'
            result[name]={key:value.summary() for key,value in trackers.items()}
            result[name]['quality']=trackers['stage'].wind_quality_summary()[mode]
            result[name]['quality_updates']=trackers['stage'].quality_updates
        return result


def subgroup_gate(summaries,quality_gate,*,horizon='stage',minimum_n=64,seconds=180.):
    failures=[]
    for name,row in summaries.items():
        measured=row[horizon]
        if measured.get('n',0)<minimum_n:
            failures.append(name+':samples')
            continue
        floor=.99 if name=='clean' else .95
        if measured.get('survival',0.)<floor:failures.append(name+':survival')
        if measured.get('len_fraction',0.)<.99:failures.append(name+':length')
        if measured.get('hits_mean',0.)<1.2*seconds:failures.append(name+':hits')
        gate=quality_gate.for_wind('clean' if name=='clean' else 'constant')
        if row.get('quality_updates',0)!=32 or not gate(row['quality']):
            failures.append(name+':quality')
    return dict(passed=not failures,failures=failures)
