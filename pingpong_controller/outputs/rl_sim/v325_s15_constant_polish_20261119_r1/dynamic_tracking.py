"""Track actual episode labels, including unchanged native random sampling."""
from collections import deque

import numpy as np

from s15_tracking import TrainingSegments
from fixed_plane_v325_wind import MODE_NAMES


class DynamicS15Progress:
    def __init__(self,n_envs,quality_fields,seconds=180.,*,short_seconds=30.,full_seconds=1200.):
        self.n_envs=n_envs
        self.quality_fields=tuple(quality_fields)
        self.names=['clean']+[f'sector{i:02d}_{kind}' for i in range(8) for kind in ('continuous','peak160')]
        self.horizons={'stage':seconds,'short30':short_seconds,'full1200':full_seconds}
        self.groups={name:{key:TrainingSegments(value,.005,1,
            quality_fields=self.quality_fields if key=='stage' else ())
            for key,value in self.horizons.items()} for name in self.names}
        self.age=np.zeros((3,n_envs),np.int64)
        self.hits=np.zeros((3,n_envs))
        self.labels=np.full(n_envs,-1,int)

    def label_array(self,metrics):
        mode=np.asarray(metrics['wind/mode']).astype(int)
        sector=np.asarray(metrics['polish/wind_sector']).astype(int)
        peak=np.asarray(metrics['polish/peak_assigned'])>.5
        if not np.isin(mode,[0,1]).all():raise ValueError('FixedS15onlyadmitsclean/constantwind')
        labels=np.where(mode==0,0,1+2*sector+peak).astype(int)
        if not ((labels>=0)&(labels<17)).all():raise ValueError('Invalidwindsector')
        return labels

    def masks(self,metrics):
        labels=self.label_array(metrics)
        return {name:labels==i for i,name in enumerate(self.names)}

    def add(self,metrics,done):
        labels=self.label_array(metrics)
        modes=np.asarray(metrics['wind/mode']).astype(int)
        done=np.asarray(done,bool)
        terminated=np.asarray(metrics['terminated'],bool)
        recent=np.asarray(metrics['v321/recent_hit'],bool)
        hits=np.asarray(metrics['new_hit'])
        for i,name in enumerate(self.names):
            mask=labels==i
            values={mode:{field:0. for field in self.quality_fields} for mode in MODE_NAMES}
            selected='clean' if i==0 else 'constant'
            values[selected]={field:float(np.asarray(metrics[field])[mask].sum()) for field in self.quality_fields}
            self.groups[name]['stage'].qualities.append(values)
        for t in range(len(labels)):
            continuing=np.any(self.age>0,axis=0)
            if np.any(continuing&(self.labels!=labels[t])):
                raise ValueError('Episode group changed without a physical reset')
            self.labels[:]=labels[t]
            self.age+=1
            self.hits+=hits[t]
            for h,(horizon,seconds) in enumerate(self.horizons.items()):
                full=self.age[h]>=round(seconds/.005)
                selected=full|terminated[t]
                for lane in np.flatnonzero(selected):
                    tracker=self.groups[self.names[labels[t,lane]]][horizon]
                    tracker.outcomes.append((min(1.,self.age[h,lane]/tracker.steps),float(self.hits[h,lane]),
                        bool(full[lane] and not terminated[t,lane] and recent[t,lane]),int(modes[t,lane]),0))
                    tracker.completed_segments+=1
                for lane in np.flatnonzero(done[t]&~terminated[t]&~full):
                    self.groups[self.names[labels[t,lane]]][horizon].censored_segments+=1
                reset=full|done[t]
                self.age[h,reset]=0
                self.hits[h,reset]=0.

    def summaries(self):
        result={}
        for name,trackers in self.groups.items():
            mode='clean' if name=='clean' else 'constant'
            result[name]={key:tracker.summary() for key,tracker in trackers.items()}
            result[name]['quality']=trackers['stage'].wind_quality_summary()[mode]
            result[name]['quality_updates']=trackers['stage'].quality_updates
        return result
