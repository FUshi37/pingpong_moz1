"""Actual1200s continuity with separately audited physical/confirmation gaps."""
from collections import deque
import numpy as np
from pathlib import Path
import sys

A=Path(__file__).resolve().parent
sys.path[:0]=[str(A.parent/'v325_s15_constant_polish_20261119_r1'),
             str(A.parent/'v325_structural_ablations_20261119_r1/source')]
from strict_1200_acceptance import FullEpisode99Progress,gate99 as _legacy_gate99
from s15_tracking import TrainingSegments
from fixed_plane_v325_wind import MODE_NAMES
from xwind_sampling import labels_for_wind


def gate99(summaries,quality_gate,minimum_n=16):
    result=_legacy_gate99(summaries,quality_gate,minimum_n)
    result.update(calm_confirmation_gap_limit_s=1.2,
        wind_confirmation_gap_cap_s=1.8,wind_confirmation_period_multiplier=3.,
        physical_gap_limit_s=1.2)
    return result


def foundation_gate99(summaries,quality_gate,minimum_n=100):
    """Calm-phase readiness only; preserve the full1200s success contract."""
    result=gate99({'clean':summaries['clean']},quality_gate,minimum_n)
    result.update(phase='calm_foundation',three_population_final_pass=False)
    return result


def complete_success(steps,hits,confirmed_ratio,physical_ratio,terminated):
    return ((np.asarray(steps)>=240000)&(np.asarray(hits)>=1440)&
            (np.asarray(confirmed_ratio)<=1.+1.e-6)&(np.asarray(physical_ratio)<=1.+1.e-6)&
            ~np.asarray(terminated,bool))


class AdaptiveProgress(FullEpisode99Progress):
    def __init__(self,n_envs,quality_fields):
        super().__init__(n_envs,quality_fields)
        self.names=['clean']+[f'xsector{i:02d}_bin{j}' for i in range(4) for j in range(4)]+[
            f'xpeak_dir{i:02d}' for i in range(10)]
        self.groups={name:{h:TrainingSegments(seconds,.005,1,
            capacity=100 if h=='full1200' else 4096,
            quality_fields=self.quality_fields if h=='stage' else ())
            for h,seconds in self.horizons.items()} for name in self.names}
        self.episode_notes=deque(maxlen=4096)
        self.half_angle_deg=45.

    def label_array(self,m):
        return labels_for_wind(m['wind/mode'],m['polish/background_speed_m_s'],
                               m['single5/wind_angle_rad'],self.half_angle_deg)

    def add(self,m,done):
        labels=self.label_array(m);modes=np.asarray(m['wind/mode'],int)
        done=np.asarray(done,bool);terminated=np.asarray(m['terminated'],bool)
        hits=np.asarray(m['new_hit']);recent=np.asarray(m['timing/physical_recent'],bool)
        for index,name in enumerate(self.names):
            mask=labels==index;selected='clean' if index==0 else 'constant'
            values={mode:{field:0. for field in self.quality_fields} for mode in MODE_NAMES}
            values[selected]={field:float(np.asarray(m[field])[mask].sum()) for field in self.quality_fields}
            self.groups[name]['stage'].qualities.append(values)
        for t in range(len(labels)):
            continuing=np.any(self.age>0,axis=0)
            if np.any(continuing&(self.labels!=labels[t])):raise ValueError('Wind cell changed within episode')
            self.labels[:]=labels[t];self.age+=1;self.hits+=hits[t]
            for h,(horizon,seconds) in enumerate(self.horizons.items()):
                full=self.age[h]>=round(seconds/.005)
                selected=(full|done[t]) if horizon=='full1200' else (full|terminated[t])
                for lane in np.flatnonzero(selected):
                    tracker=self.groups[self.names[labels[t,lane]]][horizon]
                    success=bool(full[lane] and not terminated[t,lane] and recent[t,lane])
                    if horizon=='full1200':
                        success=success and bool(complete_success(self.age[h,lane],self.hits[h,lane],
                            m['timing/confirmed_max_ratio'][t,lane],m['timing/physical_max_ratio'][t,lane],
                            terminated[t,lane]))
                        self.episode_notes.append(dict(group=self.names[labels[t,lane]],
                            seconds=float(self.age[h,lane]*.005),hits=float(self.hits[h,lane]),success=success,
                            physical_gap_s=float(m['timing/physical_max_gap_s'][t,lane]),
                            confirmed_gap_s=float(m['timing/confirmed_max_gap_s'][t,lane]),
                            confirmed_max_ratio=float(m['timing/confirmed_max_ratio'][t,lane])))
                    tracker.outcomes.append((min(1.,self.age[h,lane]/tracker.steps),
                        float(self.hits[h,lane]),success,int(modes[t,lane]),0))
                    tracker.completed_segments+=1
                if horizon!='full1200':
                    for lane in np.flatnonzero(done[t]&~terminated[t]&~full):
                        self.groups[self.names[labels[t,lane]]][horizon].censored_segments+=1
                reset=full|done[t];self.age[h,reset]=0;self.hits[h,reset]=0.
