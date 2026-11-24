"""Opt-in fixed rehearsal cells; no privileged wind is added to actor inputs."""
import math
import numpy as np


def layout(n_envs: int):
    clean_n=round(.1*n_envs);peak_n=round(.25*n_envs)
    mode=np.ones(n_envs,np.int32);mode[:clean_n]=0
    sector=np.zeros(n_envs,np.int32);bins=np.zeros(n_envs,np.int32)
    axis=np.full(n_envs,-1,np.int32)
    peak_start=clean_n;peak_end=clean_n+peak_n
    uniform=peak_n*5//8
    sector[peak_start:peak_end]=np.arange(peak_n)%8
    bins[peak_start:peak_end]=4
    axis[peak_start+uniform:peak_end]=np.arange(peak_n-uniform)%3
    slot=np.arange(n_envs-peak_end)
    sector[peak_end:]=slot%8;bins[peak_end:]=(slot//8)%4
    return mode,sector,bins,axis


def vectors_from_uniforms(uniforms,assignments,*,xp):
    mode,sector,bins,axis=[xp.asarray(x) for x in assignments]
    low=xp.minimum(bins,3)*.4;high=low+.4
    radius=xp.where(bins==4,1.6,xp.sqrt(low*low+uniforms[:,0]*(high*high-low*low)))
    angle=-math.pi+(sector+uniforms[:,1])*(math.pi/8.)
    axis_angle=xp.where(axis==0,-math.pi,xp.where(axis==1,-.5*math.pi,0.))
    angle=xp.where(axis>=0,axis_angle,angle)
    vector=xp.stack([radius*xp.cos(angle),xp.minimum(radius*xp.sin(angle),0.),xp.zeros_like(radius)],axis=-1)
    return xp.where((mode==0)[:,None],0.,vector),mode


class StratifiedWindEnv:
    """Keep native physics/reset tensors; select only the sampled wind assignment."""

    def __init__(self,inner,enabled=False):
        self.inner=inner;self.enabled=enabled
        self.assignment=layout(inner.n_envs)

    def __getattr__(self,name):return getattr(self.inner,name)

    def _assign(self,state,keys,selected):
        import jax
        import jax.numpy as jnp
        uniforms=jax.vmap(lambda key:jax.random.uniform(
            jax.random.fold_in(key,2026111957),(2,),dtype=jnp.float32))(keys)
        vectors,modes=vectors_from_uniforms(uniforms,self.assignment,xp=jnp)
        return state._replace(wind_mode=jnp.where(selected,modes,state.wind_mode),
            wind_background_m_s=jnp.where(selected[:,None],vectors,state.wind_background_m_s))

    def reset(self,keys):
        if not self.enabled:return self.inner.reset(keys)
        import jax.numpy as jnp
        state,obs=self.inner.reset(keys)
        return self._assign(state,keys,jnp.ones(self.n_envs,bool)),obs

    def reset_done(self,state,obs,done,keys):
        if not self.enabled:return self.inner.reset_done(state,obs,done,keys)
        state,obs=self.inner.reset_done(state,obs,done,keys)
        return self._assign(state,keys,done),obs

    def step(self,state,action):return self.inner.step(state,action)
