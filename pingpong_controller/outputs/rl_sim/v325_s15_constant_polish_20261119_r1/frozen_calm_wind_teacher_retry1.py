"""Opt-in simulation teacher: immutable full S15 actor plus wind-only correction."""
import jax
import jax.numpy as jnp

BASE_OBS_DIM=114
TEACHER_OBS_DIM=118
TAG='frozen_calm_wind_teacher_v1'


def components_factory(native_components,apply_mlp,frozen_params,correction_limit=.25):
    """Every original actor parameter is outside the new trainable tree."""
    if not 0.<correction_limit<=.25:raise ValueError('Unregistered correction authority')
    frozen=jax.tree_util.tree_map(jax.lax.stop_gradient,frozen_params)
    def components(params,obs):
        if TAG not in params:
            return native_components(params,obs[...,:BASE_OBS_DIM] if obs.shape[-1]==TEACHER_OBS_DIM else obs)
        if obs.ndim!=2 or obs.shape[-1]!=TEACHER_OBS_DIM:
            raise ValueError('Privileged wind teacher requires explicit118-D actor observation')
        base=jax.lax.stop_gradient(native_components(frozen,obs[:,:BASE_OBS_DIM])[0])
        nonzero=jnp.linalg.norm(obs[:,BASE_OBS_DIM:BASE_OBS_DIM+3],axis=-1)>0.
        gain=jnp.clip(obs[:,-1],0.,1.)*jnp.clip(jnp.linalg.norm(obs[:,BASE_OBS_DIM:BASE_OBS_DIM+3],axis=-1),0.,1.)
        residual=correction_limit*jnp.tanh(apply_mlp(params['pi'],obs))
        delta=jnp.where(nonzero[:,None],gain[:,None]*residual,jnp.zeros_like(residual))
        return base+delta,base,delta
    return components


def new_trainable_state(ppo,parent,key,hidden_dim=128):
    """Only the new wind head starts fresh; old critic/moments and Adam t persist."""
    head=ppo.init_mlp(key,TEACHER_OBS_DIM,hidden_dim,7,out_scale=0.)
    params={'pi':head,'v':parent.params['v'],'log_std':parent.params['log_std'],TAG:jnp.asarray(1.,jnp.float32)}
    opt=ppo.adam_init(params)
    m=dict(opt.m);v=dict(opt.v)
    for name in ('v','log_std'):
        m[name]=parent.opt.m[name];v[name]=parent.opt.v[name]
    return ppo.TrainState(params,ppo.OptimState(m,v,parent.opt.t))


class TeacherObservationEnv:
    """Append background wind3 and native startup gain1; physical state unchanged."""

    def __init__(self,inner,enabled=False,startup_gain_fn=None):
        self.inner=inner;self.enabled=enabled;self.startup_gain_fn=startup_gain_fn
        if enabled and inner.obs_dim!=BASE_OBS_DIM:
            raise ValueError('Frozen S15 prefix must remain114-D')
        self.obs_dim=TEACHER_OBS_DIM if enabled else inner.obs_dim

    def __getattr__(self,name):return getattr(self.inner,name)

    def observe(self,state,obs):
        if not self.enabled:return obs
        if self.startup_gain_fn is None:
            from fixed_plane_v325_wind import startup_gain
            gain=startup_gain(state,state.step_count*self.dt,self.cfg.wind_config)
        else:gain=self.startup_gain_fn(state)
        wind=state.wind_background_m_s/1.6
        return jnp.concatenate([obs,wind,gain[:,None]],axis=-1)

    def reset(self,keys):
        state,obs=self.inner.reset(keys)
        return state,self.observe(state,obs)

    def reset_done(self,state,obs,done,keys):
        if not self.enabled:return self.inner.reset_done(state,obs,done,keys)
        state,obs=self.inner.reset_done(state,obs[:,:BASE_OBS_DIM],done,keys)
        return state,self.observe(state,obs)

    def get_critic_obs(self,state,obs):
        return self.inner.get_critic_obs(state,obs[:,:BASE_OBS_DIM] if self.enabled else obs)

    def step(self,state,action):
        state,obs,reward,done,metrics=self.inner.step(state,action)
        return state,self.observe(state,obs),reward,done,metrics


def adam_factory(native_adam,origin_t:int):
    """Give only the newly initialized head its own Adam bias-correction age."""
    if origin_t<0:raise ValueError('Invalid source Adam age')
    def step(params,grads,opt,learning_rate,max_grad_norm,beta1=.9,beta2=.999,eps=1.e-5):
        updated,after,norm=native_adam(params,grads,opt,learning_rate,max_grad_norm,beta1,beta2,eps)
        if TAG not in params:return updated,after,norm
        local_t=after.t-origin_t
        m=jax.tree_util.tree_map(lambda x:x/(1.-beta1**local_t),after.m['pi'])
        v=jax.tree_util.tree_map(lambda x:x/(1.-beta2**local_t),after.v['pi'])
        updated=dict(updated)
        updated['pi']=jax.tree_util.tree_map(lambda p,mh,vh:
            p-float(learning_rate)*mh/(jnp.sqrt(vh)+eps),params['pi'],m,v)
        return updated,after,norm
    step._frozen_calm_teacher_adam=True
    return step


def prepare_teacher_optimizer(ppo):
    """Remove the legacy114-D row adapter only from this tagged teacher optimizer."""
    current=ppo.adam_step
    base=getattr(current,'_wind_history_optimizer_base',current)
    if not getattr(base,'_frozen_calm_teacher_adam',False):
        raise ValueError('Teacher requires its explicit local-age Adam adapter')
    ppo.adam_step=base
    return current is not base
