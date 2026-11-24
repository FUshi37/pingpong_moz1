"""Default-off PPO stabilization; one actor, no teacher/action additions."""
import jax
import jax.numpy as jnp


def cohort_step_measure(previous_params,transitions,policy_mean,*,sample_size=2048,sigma=.005):
    flat=transitions.obs.reshape((-1,transitions.obs.shape[-1]))
    size=min(sample_size,len(flat));indices=(jnp.arange(size)*65)%len(flat)
    obs=flat[indices];shape=transitions.obs.shape[:-1];metrics=getattr(transitions,'metrics',{})
    mode=metrics.get('wind/mode',jnp.ones(shape)).reshape(-1)[indices]
    speed=metrics.get('wind/speed_m_s',jnp.ones(shape)).reshape(-1)[indices]
    angle=metrics.get('single5/wind_angle_rad',jnp.zeros(shape)).reshape(-1)[indices]
    group=jnp.where(mode<.5,0,jnp.where(speed<.2,1,jnp.where(angle<-jnp.pi/2.,2,3)))
    mask=group[:,None]==jnp.arange(4)[None,:];counts=jnp.sum(mask,axis=0)
    old=jax.lax.stop_gradient(policy_mean(previous_params,obs))
    def measure(params):
        values=jnp.sum((policy_mean(params,obs)-old)**2,axis=-1)/(2.*sigma**2)
        return jnp.max(jnp.sum(values[:,None]*mask,axis=0)/jnp.maximum(counts,1))
    return measure


def critic_warmup_adam(native,*,start_t,minibatches=0):
    """Temporarily preserve actor values, moments and clock without resetting."""
    if not minibatches:return native
    def step(params,grads,opt,*args,**kwargs):
        warm=(opt.t>=start_t)&(opt.t<start_t+minibatches)
        masked={**grads,'pi':jax.tree.map(lambda g:jnp.where(warm,0.,g),grads['pi']),
            'log_std':jnp.where(warm,0.,grads['log_std'])}
        updated,after,norm=native(params,masked,opt,*args,**kwargs)
        def preserve(new,old):return jax.tree.map(lambda a,b:jnp.where(warm,b,a),new,old)
        updated={**updated,'pi':preserve(updated['pi'],params['pi']),
            'log_std':preserve(updated['log_std'],params['log_std'])}
        def moments(new,old):
            return {**new,'pi':preserve(new['pi'],old['pi']),
                'log_std':preserve(new['log_std'],old['log_std'])}
        after=after._replace(m=moments(after.m,opt.m),v=moments(after.v,opt.v),
            actor_t=jnp.where(warm,opt.actor_t,after.actor_t))
        return updated,after,norm
    step.__dict__.update(native.__dict__)
    return step


def cap_actor_update(native,policy_mean,*,maximum_kl=None,sample_size=2048,
                     sigma=.005,compile_update=True):
    """Bound each sampled live cohort's actor step; keep critic and Adam values."""
    if maximum_kl is None:return native
    if maximum_kl<=0. or sample_size<1:raise ValueError('Invalid actor-step bound')
    scales=2.**(-jnp.arange(13,dtype=jnp.float32))
    def update(previous,runner,transitions):
        candidate,metrics=native(previous,runner,transitions)
        flat=transitions.obs.reshape((-1,transitions.obs.shape[-1]))
        size=min(sample_size,len(flat));indices=(jnp.arange(size)*65)%len(flat)
        obs=flat[indices];shape=transitions.obs.shape[:-1]
        mode=transitions.metrics.get('wind/mode',jnp.ones(shape)).reshape(-1)[indices]
        speed=transitions.metrics.get('wind/speed_m_s',jnp.ones(shape)).reshape(-1)[indices]
        angle=transitions.metrics.get('single5/wind_angle_rad',jnp.zeros(shape)).reshape(-1)[indices]
        group=jnp.where(mode<.5,0,jnp.where(speed<.2,1,jnp.where(angle<-jnp.pi/2.,2,3)))
        mask=group[:,None]==jnp.arange(4)[None,:]
        counts=jnp.sum(mask,axis=0)
        old_mean=jax.lax.stop_gradient(policy_mean(previous.params,obs))
        def measured(pi):
            mean=policy_mean({**candidate.params,'pi':pi},obs)
            values=jnp.sum((mean-old_mean)**2,axis=-1)/(2.*sigma**2)
            return jnp.max(jnp.sum(values[:,None]*mask,axis=0)/jnp.maximum(counts,1))
        old_pi,new_pi=previous.params['pi'],candidate.params['pi']
        def attempt(carry,scale):
            selected,value,selected_scale,found=carry
            pi=jax.tree.map(lambda a,b:a+scale*(b-a),old_pi,new_pi)
            kl=measured(pi);choose=(~found)&jnp.isfinite(kl)&(kl<=maximum_kl)
            selected=jax.tree.map(lambda a,b:jnp.where(choose,b,a),selected,pi)
            return (selected,jnp.where(choose,kl,value),jnp.where(choose,scale,selected_scale),found|choose),None
        (pi,post,scale,found),_=jax.lax.scan(attempt,(old_pi,jnp.array(0.),jnp.array(0.),jnp.array(False)),scales)
        after=candidate._replace(params={**candidate.params,'pi':pi})
        metrics={**metrics,'learning_stability/candidate_sampled_max_kl':measured(new_pi),
            'learning_stability/sampled_max_step_kl':post,'learning_stability/step_scale':scale,
            'learning_stability/observations':jnp.asarray(size,jnp.float32)}
        return after,metrics
    return jax.jit(update) if compile_update else update
