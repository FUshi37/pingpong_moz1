"""Opt-in censored startup sampling; keep the original long-task MDP target."""
from collections import deque
import math

import jax.numpy as jnp
import numpy as np

FIELDS=('sampling/rehearsal_lane','sampling/rehearsal_cutoff')
PROGRESS_FIELDS=('terminated','sampling/rehearsal_cutoff','diagnostic/episode_seconds',
                 'hit_count','timing/confirmed_max_ratio','timing/physical_max_ratio')


class StartupRehearsalEnv:
    def __init__(self,inner,*,fraction,seconds,allow_empty=False):
        self.inner=inner
        self.n_rehearsal=round(inner.n_envs*fraction)
        if not (0<self.n_rehearsal<inner.n_envs or (allow_empty and self.n_rehearsal==0)):
            raise ValueError('Both rehearsal and long lanes must be present')
        self.cutoff_steps=round(seconds/inner.dt)
        if not 0<self.cutoff_steps<inner.max_steps or not math.isclose(
                self.cutoff_steps*inner.dt,seconds,abs_tol=1.e-9):
            raise ValueError('Invalid rehearsal duration')
        self.role=jnp.arange(inner.n_envs)<self.n_rehearsal

    def __getattr__(self,name):return getattr(self.inner,name)

    def step(self,state,action):
        after,obs,reward,done,metrics=self.inner.step(state,action)
        cutoff=self.role&(after.step_count>=self.cutoff_steps)&~done
        metrics={**metrics,
            'truncated':jnp.where(cutoff,jnp.ones_like(metrics['truncated']),metrics['truncated']),
            'sampling/rehearsal_lane':self.role.astype(jnp.float32),
            'sampling/rehearsal_cutoff':cutoff.astype(jnp.float32)}
        return after,obs,reward,done|cutoff,metrics


def wrap_rehearsal(inner,*,enabled=False,fraction=.1,seconds=20.,allow_empty=False):
    if not enabled:return inner
    return StartupRehearsalEnv(inner,fraction=fraction,seconds=seconds,allow_empty=allow_empty)


def long_rollout(metrics,done,n_rehearsal):
    """Exclude deliberately censored lanes from every long-episode statistic."""
    if not n_rehearsal:return metrics,done
    count=np.asarray(done).shape[1]
    selected={k:(v[:,n_rehearsal:] if np.ndim(v)>=2 and np.shape(v)[1]==count else v)
              for k,v in metrics.items()}
    return selected,np.asarray(done)[:,n_rehearsal:]


class StartupProgress:
    def __init__(self,*,seconds=20.,capacity=1024):
        self.seconds=seconds
        self.outcomes=deque(maxlen=capacity)
        self.completed_episodes=0

    def add(self,metrics,done):
        done=np.asarray(done,bool)
        for t,lane in zip(*np.nonzero(done)):
            def value(name):return float(metrics[name][t,lane])
            duration=value('diagnostic/episode_seconds')
            hits=value('hit_count')
            terminated=value('terminated')>.5
            success=(value('sampling/rehearsal_cutoff')>.5 and not terminated
                and duration>=self.seconds-1.e-5
                and hits>=math.ceil(1.2*self.seconds)
                and value('timing/confirmed_max_ratio')<=1.+1.e-6
                and value('timing/physical_max_ratio')<=1.+1.e-6)
            self.outcomes.append((duration,hits,success,terminated))
            self.completed_episodes+=1

    def summary(self):
        n=len(self.outcomes)
        return dict(n=n,survival=float(np.mean([x[2] for x in self.outcomes])) if n else 0.,
            mean_seconds=float(np.mean([x[0] for x in self.outcomes])) if n else 0.,
            mean_hits=float(np.mean([x[1] for x in self.outcomes])) if n else 0.,
            completed_episodes=self.completed_episodes,seconds=self.seconds,
            counts_as_full1200_success=False)
