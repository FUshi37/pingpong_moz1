"""Default-off output-layer fine-tuning of one ordinary actor MLP."""
import jax
import jax.numpy as jnp


def output_only_adam(native, *, enabled=False):
    if not enabled:
        return native
    def step(params, grads, opt, *args, **kwargs):
        pi_grad={name:(value if name=='out' else jax.tree.map(jnp.zeros_like,value))
                 for name,value in grads['pi'].items()}
        masked={**grads,'pi':pi_grad,'log_std':jnp.zeros_like(grads['log_std'])}
        updated,after,norm=native(params,masked,opt,*args,**kwargs)
        def actor(new,old):
            return {name:(value if name=='out' else old[name]) for name,value in new.items()}
        updated={**updated,'pi':actor(updated['pi'],params['pi']),
                 'log_std':params['log_std']}
        def moments(new,old):
            return {**new,'pi':actor(new['pi'],old['pi']),'log_std':old['log_std']}
        after=after._replace(m=moments(after.m,opt.m),v=moments(after.v,opt.v))
        return updated,after,norm
    step.__dict__.update(native.__dict__)
    step._output_layer_only=True
    return step
