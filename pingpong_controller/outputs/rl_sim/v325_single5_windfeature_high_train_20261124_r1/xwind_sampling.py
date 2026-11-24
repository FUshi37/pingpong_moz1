"""Two admitted X-side cones with fixed representative and hard-case lanes."""
import math
import numpy as np


def layout(n_envs):
    clean_n=round(.1*n_envs);peak_n=round(.25*n_envs)
    mode=np.ones(n_envs,np.int32);mode[:clean_n]=0
    sector=np.zeros(n_envs,np.int32);bins=np.zeros(n_envs,np.int32)
    axis=np.full(n_envs,-1,np.int32);hard=np.zeros(n_envs,bool)
    start=clean_n;end=start+peak_n;uniform=peak_n*5//8
    sector[start:end]=np.arange(peak_n)%4;bins[start:end]=4
    axis[start+uniform:end]=np.arange(peak_n-uniform)%2
    count=n_envs-end;representative=(count*3//4)//16*16
    slots=np.arange(representative);sector[end:end+representative]=slots%4
    bins[end:end+representative]=(slots//4)%4
    sector[end+representative:]=np.where(np.arange(count-representative)%2==0,0,3)
    bins[end+representative:]=3;hard[end+representative:]=True
    return mode,sector,bins,axis,hard


def vectors_from_uniforms(uniforms,assignments,half_angle_deg,*,xp):
    if not 0.<half_angle_deg<=45.:raise ValueError('X-side half angle must be in(0,45]')
    mode,sector,bins,axis,hard=[xp.asarray(x) for x in assignments]
    width=half_angle_deg*math.pi/180.
    low=xp.minimum(bins,3)*.4;high=low+.4
    radius=xp.where(bins==4,1.6,xp.sqrt(low*low+uniforms[:,0]*(high*high-low*low)))
    offset=(sector%2+uniforms[:,1])*(width/2.)
    angle=xp.where(sector<2,-math.pi+offset,-width+offset)
    hard_angle=xp.where(sector<2,-math.pi+uniforms[:,1]*width/4.,-uniforms[:,1]*width/4.)
    angle=xp.where(hard,hard_angle,angle)
    angle=xp.where(axis==0,-math.pi,xp.where(axis==1,0.,angle))
    vector=xp.stack([radius*xp.cos(angle),xp.minimum(radius*xp.sin(angle),0.),xp.zeros_like(radius)],axis=-1)
    return xp.where((mode==0)[:,None],0.,vector),mode


def labels_for_wind(mode,speed,angle,half_angle_deg=45.):
    mode,speed,angle=np.asarray(mode,int),np.asarray(speed),np.asarray(angle)
    angle=np.where(np.abs(angle-np.pi)<1.e-6,-np.pi,angle)
    width=np.deg2rad(half_angle_deg)
    allowed=(angle>=-np.pi-1.e-6)&(angle<=1.e-6)&((angle<=-np.pi+width+1.e-6)|(angle>=-width-1.e-6))
    if (not np.isin(mode,[0,1]).all() or not np.isfinite(speed).all() or
        np.any((speed<0.)|(speed>1.60001)) or np.any((mode>0)&~allowed)):
        raise ValueError('Wind outside registered X-side Y<=0 domain')
    offset=np.where(angle<-np.pi/2,angle+np.pi,angle+width)
    side=(angle>=-np.pi/2).astype(int)
    sector=2*side+np.clip(np.floor(offset/(width/2)),0,1).astype(int)
    direction=4*side+np.clip(np.floor(offset/(width/4)),0,3).astype(int)
    direction=np.where(np.abs(angle+np.pi)<1.e-6,8,direction)
    direction=np.where(np.abs(angle)<1.e-6,9,direction)
    bins=np.clip(np.floor(speed/.4),0,3).astype(int)
    return np.where(mode==0,0,np.where(speed>=1.6-1.e-5,17+direction,1+4*sector+bins)).astype(int)


class XWindEnv:
    def __init__(self,inner,half_angle_deg,enabled=False,calm_only=False):
        self.inner=inner;self.enabled=enabled;self.half_angle_deg=half_angle_deg
        self.calm_only=calm_only
        self.assignment=layout(inner.n_envs)
    def __getattr__(self,name):return getattr(self.inner,name)
    def _assign(self,state,keys,selected):
        import jax
        import jax.numpy as jnp
        uniforms=jax.vmap(lambda key:jax.random.uniform(jax.random.fold_in(key,2026112312),(2,)))(keys)
        vectors,modes=vectors_from_uniforms(uniforms,self.assignment,self.half_angle_deg,xp=jnp)
        if self.calm_only:
            vectors=jnp.zeros_like(vectors)
            modes=jnp.zeros_like(modes)
        return state._replace(wind_mode=jnp.where(selected,modes,state.wind_mode),
            wind_background_m_s=jnp.where(selected[:,None],vectors,state.wind_background_m_s))
    def reset(self,keys):
        state,obs=self.inner.reset(keys)
        if self.enabled:
            import jax.numpy as jnp
            state=self._assign(state,keys,jnp.ones(self.n_envs,bool))
        return state,obs
    def reset_done(self,state,obs,done,keys):
        state,obs=self.inner.reset_done(state,obs,done,keys)
        return (self._assign(state,keys,done),obs) if self.enabled else (state,obs)
    def step(self,state,action):return self.inner.step(state,action)
