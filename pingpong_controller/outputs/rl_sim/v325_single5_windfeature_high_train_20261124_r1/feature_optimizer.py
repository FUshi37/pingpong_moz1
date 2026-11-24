"""Optimizer groups within one ordinary actor; no extra policy or action sum."""
import jax
import jax.numpy as jnp
from single_actor import ActorOptim, adam_factory as ordinary_adam_factory


def feature_adam_factory(previous_adam, *, history_xy_epsilon=1.e-5,
                         history_xy_lr_multiplier=1., feature_lr_multiplier=32.,
                         warmup_minibatches=512, independent_actor_critic_clipping=False,
                         permanent_wind_rows_only=False):
    """Learn two wind input rows first, then restart actor Adam for joint PPO.

    The global accepted-minibatch clock and critic moments never reset. The
    one-time actor reset occurs before the first accepted post-warmup step,
    making rollback/retry and checkpoint restoration deterministic.
    """
    ordinary = ordinary_adam_factory(previous_adam,
        history_xy_epsilon=history_xy_epsilon,
        history_xy_lr_multiplier=history_xy_lr_multiplier)

    def step(params, grads, opt, learning_rate, max_grad_norm,
             beta1=.9, beta2=.999, eps=1.e-5):
        width=params['pi']['l1']['w'].shape[0]
        if not isinstance(opt, ActorOptim) or width not in (118,120):
            raise ValueError('Feature warmup requires the explicit118-D single actor')
        feature_rows=slice(118,120) if width==120 else slice(114,116)
        feature_only = jnp.asarray(permanent_wind_rows_only) | (opt.t < warmup_minibatches)
        boundary = (opt.t == warmup_minibatches) & (not permanent_wind_rows_only)
        def reset_actor_moments(tree):
            actor = jax.tree.map(lambda value: jnp.where(boundary, jnp.zeros_like(value), value), tree['pi'])
            return {**tree, 'pi': actor}
        opt = opt._replace(m=reset_actor_moments(opt.m), v=reset_actor_moments(opt.v),
                           actor_t=jnp.where(boundary, 0, opt.actor_t))
        masked = jax.tree.map(jnp.zeros_like, grads['pi'])
        masked = {**masked, 'l1': {**masked['l1'], 'w': masked['l1']['w'].at[feature_rows].set(
            grads['pi']['l1']['w'][feature_rows])}}
        actor_grad = jax.tree.map(lambda mask, original: jnp.where(feature_only, mask, original),
                                  masked, grads['pi'])
        effective = {**grads, 'pi': actor_grad,
                     'log_std': jnp.where(feature_only, jnp.zeros_like(grads['log_std']), grads['log_std'])}
        raw_norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree.leaves(effective)))
        clip_limit = max_grad_norm
        if independent_actor_critic_clipping:
            actor_norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree.leaves(
                (effective['pi'], effective['log_std']))))
            critic_norm = jnp.sqrt(sum(jnp.sum(v*v) for v in jax.tree.leaves(effective['v'])))
            actor_scale = jnp.minimum(1., max_grad_norm/jnp.maximum(actor_norm, 1.e-12))
            critic_scale = jnp.minimum(1., max_grad_norm/jnp.maximum(critic_norm, 1.e-12))
            effective = dict(pi=jax.tree.map(lambda x: x*actor_scale, effective['pi']),
                v=jax.tree.map(lambda x: x*critic_scale, effective['v']),
                log_std=effective['log_std']*actor_scale)
            # Both groups are bounded by max_grad_norm; avoid clipping twice.
            clip_limit = 2.*max_grad_norm
        updated, after, norm = ordinary(params, effective, opt, learning_rate,
                                        clip_limit, beta1, beta2, eps)
        if independent_actor_critic_clipping:
            norm = raw_norm
        # These are parameter-group learning rates, all inside the same MLP.
        wind_rows = params['pi']['l1']['w'][feature_rows] + feature_lr_multiplier * (
            updated['pi']['l1']['w'][feature_rows]-params['pi']['l1']['w'][feature_rows])
        actor = jax.tree.map(lambda original, new: jnp.where(feature_only, original, new),
                             params['pi'], updated['pi'])
        actor = {**actor, 'l1': {**actor['l1'], 'w': actor['l1']['w'].at[feature_rows].set(wind_rows)}}
        result = {**updated, 'pi': actor,
                  'log_std': jnp.where(feature_only, params['log_std'], updated['log_std'])}
        if permanent_wind_rows_only:
            # A zero gradient alone still decays old moments and moves weights.
            # Preserve frozen values AND both moments, including log sigma.
            def retain_frozen_moments(changed, before):
                pi = {**before['pi'], 'l1': {**before['pi']['l1'],
                    'w': before['pi']['l1']['w'].at[feature_rows].set(
                        changed['pi']['l1']['w'][feature_rows])}}
                return {**changed, 'pi': pi, 'log_std': before['log_std']}
            after = after._replace(m=retain_frozen_moments(after.m, opt.m),
                                   v=retain_frozen_moments(after.v, opt.v))
        return result, after, norm
    step._single_actor_ablation = True
    step._feature_warmup_accepted_minibatches = warmup_minibatches
    step._feature_lr_multiplier = feature_lr_multiplier
    step._independent_actor_critic_clipping = independent_actor_critic_clipping
    step._permanent_wind_rows_only = permanent_wind_rows_only
    return step
