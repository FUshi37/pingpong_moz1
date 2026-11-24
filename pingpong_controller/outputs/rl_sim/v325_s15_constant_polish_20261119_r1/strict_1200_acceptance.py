"""Opt-in full-episode success, including continuous hits and wind-cell gates."""
from collections import deque
import numpy as np

from dynamic_tracking import DynamicS15Progress
from s15_tracking import TrainingSegments
from fixed_plane_v325_wind import MODE_NAMES


def complete_episode_success(steps, hits, max_gap_steps, terminated):
    """One real1200s simulated episode; an endpoint hit alone is insufficient."""
    return ((np.asarray(steps) >= 240000) & (np.asarray(hits) >= 1440)
            & (np.asarray(max_gap_steps) <= 240) & ~np.asarray(terminated,bool))


def gate99(summaries: dict, quality_gate, minimum_n: int = 16) -> dict:
    """Training readiness uses actual complete episodes in every joint wind cell."""
    failures=[]
    for name,row in summaries.items():
        measured=row['full1200']
        if measured.get('n',0)<minimum_n:
            failures.append(name+':samples');continue
        if measured.get('survival',0.)<.99-1.e-12:failures.append(name+':survival99')
        if measured.get('len_fraction',0.)<.99:failures.append(name+':length')
        if measured.get('hits_mean',0.)<1440.:failures.append(name+':hits')
        qgate=quality_gate.for_wind('clean' if name=='clean' else 'constant')
        if row.get('quality_updates',0)!=32 or not qgate(row['quality']):
            failures.append(name+':quality')
    return dict(passed=not failures,failures=failures,seconds=1200.,minimum_success=.99,
                minimum_n=minimum_n,continuous_gap_limit_s=1.2,minimum_hits_per_episode=1440)


class FullEpisode99Progress(DynamicS15Progress):
    """Keep all legacy diagnostic horizons; harden only complete-episode evidence."""

    def __init__(self,n_envs:int,quality_fields:tuple[str,...]):
        super().__init__(n_envs,quality_fields,180.)
        self.names=['clean']+[f'sector{i:02d}_'+('peak160' if j==4 else f'bin{j}')
                              for i in range(8) for j in range(5)]
        self.groups={name:{h:TrainingSegments(seconds,.005,1,
            quality_fields=self.quality_fields if h=='stage' else ())
            for h,seconds in self.horizons.items()} for name in self.names}
        self.since_hit=np.zeros(n_envs,np.int64)
        self.max_gap=np.zeros(n_envs,np.int64)
        self.has_hit=np.zeros(n_envs,bool)
        self.episode_notes=deque(maxlen=4096)

    def label_array(self,metrics):
        modes=np.asarray(metrics['wind/mode']).astype(int)
        sector=np.asarray(metrics['polish/wind_sector']).astype(int)
        speed=np.asarray(metrics['polish/background_speed_m_s'])
        if not np.isin(modes,[0,1]).all() or not np.isfinite(speed).all():
            raise ValueError('StrictS15 accepts only clean/constant finite background wind')
        if (speed<0.).any() or (speed>1.60001).any():raise ValueError('Invalidwindrange')
        bins=np.where(speed>=1.6-1.e-5,4,np.minimum((speed/.4).astype(int),3))
        labels=np.where(modes==0,0,1+5*sector+bins).astype(int)
        if not ((labels>=0)&(labels<41)).all():raise ValueError('Invalidwindcell')
        return labels

    def add(self,metrics,done):
        labels=self.label_array(metrics)
        modes=np.asarray(metrics['wind/mode']).astype(int)
        done=np.asarray(done,bool);terminated=np.asarray(metrics['terminated'],bool)
        recent=np.asarray(metrics['v321/recent_hit'],bool);hits=np.asarray(metrics['new_hit'])
        for i,name in enumerate(self.names):
            mask=labels==i;selected='clean' if i==0 else 'constant'
            values={mode:{field:0. for field in self.quality_fields} for mode in MODE_NAMES}
            values[selected]={field:float(np.asarray(metrics[field])[mask].sum()) for field in self.quality_fields}
            self.groups[name]['stage'].qualities.append(values)
        for t in range(len(labels)):
            continuing=np.any(self.age>0,axis=0)
            if np.any(continuing&(self.labels!=labels[t])):
                raise ValueError('Episode wind cell changed without a physical reset')
            self.labels[:]=labels[t]
            self.age+=1;self.hits+=hits[t]
            self.since_hit=np.where(hits[t]>0.,0,self.since_hit+1)
            self.has_hit|=hits[t]>0.
            self.max_gap=np.where(self.has_hit,np.maximum(self.max_gap,self.since_hit),0)
            for h,(horizon,seconds) in enumerate(self.horizons.items()):
                full=self.age[h]>=round(seconds/.005)
                selected=(full|done[t]) if horizon=='full1200' else (full|terminated[t])
                for lane in np.flatnonzero(selected):
                    tracker=self.groups[self.names[labels[t,lane]]][horizon]
                    successful=bool(full[lane] and not terminated[t,lane] and recent[t,lane])
                    if horizon=='full1200':
                        successful=successful and bool(complete_episode_success(
                            self.age[h,lane],self.hits[h,lane],self.max_gap[lane],terminated[t,lane]))
                        self.episode_notes.append((self.names[labels[t,lane]],int(self.age[h,lane]),
                            float(self.hits[h,lane]),int(self.max_gap[lane]),successful))
                    tracker.outcomes.append((min(1.,self.age[h,lane]/tracker.steps),
                        float(self.hits[h,lane]),successful,int(modes[t,lane]),0))
                    tracker.completed_segments+=1
                if horizon!='full1200':
                    for lane in np.flatnonzero(done[t]&~terminated[t]&~full):
                        self.groups[self.names[labels[t,lane]]][horizon].censored_segments+=1
                reset=full|done[t];self.age[h,reset]=0;self.hits[h,reset]=0.
            self.since_hit[done[t]]=0;self.max_gap[done[t]]=0;self.has_hit[done[t]]=False
