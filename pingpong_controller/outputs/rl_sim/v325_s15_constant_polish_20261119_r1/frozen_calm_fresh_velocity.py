"""Opt-in privileged wind clone; immutable complete S15 policy protects calm."""
from typing import NamedTuple,Any
import jax
import jax.numpy as jnp

BASE_OBS_DIM=114
TEACHER_OBS_DIM=121
TAG='frozen_calm_fresh_velocity_v1'
HISTORY_ROWS=tuple(72+3*lag+axis for lag in range(1,8) for axis in (0,1))


class FreshActorOptimState(NamedTuple):
    m: Any
    v: Any
    t: Any
    actor_t: Any


def components_factory(native_components,frozen_params):
    """Warm-start wind inference with the existing actor, including its decoder."""
    frozen=jax.tree.map(jax.lax.stop_gradient,frozen_params)
    def components(params,obs):
        if TAG not in params:
            return native_components(params,obs[:,:114] if obs.shape[-1]==121 else obs)
        if obs.ndim!=2 or obs.shape[-1]!=121:
            raise ValueError('Wind-clone teacher requires explicit121-D observation')
        from fixed_plane_wind_actor import apply_bound_gate
        prefix=obs[:,:114]
        base,_,_=native_components(frozen,prefix)
        # Subtract the same high precision warm-start mean, keeping the original
        # legacy mean as the physical baseline. This removes migration drift.
        with jax.default_matmul_precision('highest'):
            _,teacher,_=native_components(frozen,prefix)
            def wind_mean(pi):
                hidden=jnp.tanh(prefix@pi['l1']['w'][:114]+pi['l1']['b']
                                +obs[:,114:]@pi['l1']['w'][114:])
                hidden=jnp.tanh(hidden@pi['l2']['w']+pi['l2']['b'])
                raw=hidden@pi['out']['w']+pi['out']['b']
                correction=jax.lax.stop_gradient(frozen['residual_action_scale'])*jnp.tanh(raw)
                baseline,correction=apply_bound_gate(frozen,prefix,teacher,correction)
                return baseline+correction
            initial_pi={**frozen['pi'],'l1':{**frozen['pi']['l1'],
                'w':jnp.concatenate([frozen['pi']['l1']['w'],
                    jnp.zeros_like(params['pi']['l1']['w'][114:])],axis=0)}}
            learned=wind_mean(params['pi'])
            warm=wind_mean(initial_pi)
        nonzero=jnp.linalg.norm(obs[:,114:117],axis=-1)>0.
        gain=jnp.clip(obs[:,117],0.,1.)*jnp.clip(jnp.linalg.norm(obs[:,114:117],axis=-1),0.,1.)
        delta=jnp.where(nonzero[:,None],gain[:,None]*(learned-warm),jnp.zeros_like(base))
        return base+delta,jax.lax.stop_gradient(base),delta
    return components


def _extend_actor_tree(tree):
    result=dict(tree)
    pi={k:dict(v) for k,v in tree['pi'].items()}
    weights=pi['l1']['w']
    if weights.shape[0]!=114:raise ValueError('Expected complete114-D S15 actor')
    pi['l1']['w']=jnp.concatenate([weights,jnp.zeros((7,weights.shape[1]),weights.dtype)],axis=0)
    result['pi']=pi
    return result


def new_trainable_state(ppo,parent,key=None):
    """Copy source weights and critic moments; initialize only the new wind actor Adam."""
    del key
    params=_extend_actor_tree(parent.params);params[TAG]=jnp.asarray(1.,jnp.float32)
    m=_extend_actor_tree(parent.opt.m);m[TAG]=jnp.asarray(0.,jnp.float32)
    v=_extend_actor_tree(parent.opt.v);v[TAG]=jnp.asarray(0.,jnp.float32)
    m['pi']=jax.tree.map(jnp.zeros_like,params['pi'])
    v['pi']=jax.tree.map(jnp.zeros_like,params['pi'])
    return ppo.TrainState(params,FreshActorOptimState(m,v,parent.opt.t,jnp.asarray(0,jnp.int32)))


def adam_factory(native_adam,origin_t:int):
    """Keep critic Adam/global clock; a newly allocated wind actor has local Adam."""
    def step(params,grads,opt,learning_rate,max_grad_norm,beta1=.9,beta2=.999,eps=1.e-5):
        updated,after,norm=native_adam(params,grads,opt,learning_rate,max_grad_norm,beta1,beta2,eps)
        if TAG not in params:return updated,after,norm
        if not isinstance(opt,FreshActorOptimState):raise ValueError('Explicit new-actor Adam state required')
        active=sum(jnp.sum(jnp.square(x)) for x in jax.tree.leaves(grads['pi']))>0.
        actor_t=opt.actor_t+active.astype(jnp.int32)
        age=jnp.maximum(actor_t,1)
        mhat=jax.tree.map(lambda x:x/(1.-beta1**age),after.m['pi'])
        vhat=jax.tree.map(lambda x:x/(1.-beta2**age),after.v['pi'])
        candidate=jax.tree.map(lambda p,m,v:p-float(learning_rate)*m/(jnp.sqrt(v)+float(eps)),params['pi'],mhat,vhat)
        rows=jnp.asarray(HISTORY_ROWS)
        weights=params['pi']['l1']['w']
        mh=mhat['l1']['w'];vh=vhat['l1']['w']
        values=candidate['l1']['w'].at[rows].set(weights[rows]-float(learning_rate)*mh[rows]/(jnp.sqrt(vh[rows])+2.e-6))
        values=values.at[114:].set(weights[114:]-float(learning_rate)*mh[114:]/(jnp.sqrt(vh[114:])+2.e-6))
        candidate={**candidate,'l1':{**candidate['l1'],'w':values}}
        pi=jax.tree.map(lambda new,old:jnp.where(active,new,old),candidate,params['pi'])
        m={**after.m,'pi':jax.tree.map(lambda new,old:jnp.where(active,new,old),after.m['pi'],opt.m['pi'])}
        v={**after.v,'pi':jax.tree.map(lambda new,old:jnp.where(active,new,old),after.v['pi'],opt.v['pi'])}
        return {**updated,'pi':pi},FreshActorOptimState(m,v,after.t,actor_t),norm
    step._frozen_calm_teacher_adam=True
    step._wind_clone_optimizer=True
    return step


def prepare_teacher_optimizer(ppo):
    """Use the explicit121-D clone adapter; preserve the legacy114-D implementation."""
    current=ppo.adam_step;base=getattr(current,'_wind_history_optimizer_base',current)
    if not getattr(base,'_wind_clone_optimizer',False):raise ValueError('Explicit clone optimizer required')
    if current is not base:
        if (getattr(current,'_wind_history_xy_epsilon',None)!=2.e-6
                or getattr(current,'_wind_history_xy_lr_multiplier',1.)!=1.):
            raise ValueError('Source history conditioning recipe changed')
    ppo.adam_step=base
    return current is not base
