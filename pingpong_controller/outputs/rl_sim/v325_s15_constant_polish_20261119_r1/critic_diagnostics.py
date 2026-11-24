"""Read-only critic diagnostics using exactly the native PPO return targets."""


def make_critic_diagnostics(ppo, settings):
    import jax
    import jax.numpy as jnp

    @jax.jit
    def evaluate(params, runner, transitions):
        last_value = ppo.value_fn(params, runner.critic_obs)
        _, returns = ppo.compute_gae(transitions.reward, transitions.done,
            transitions.value, last_value, settings['gamma'], settings['gae_lambda'],
            terminated=transitions.terminated, truncated=transitions.truncated,
            timeout_values=transitions.timeout_value,
            time_limit_bootstrap=settings['time_limit_bootstrap'])
        variance = jnp.var(returns)
        explained = 1. - jnp.var(returns-transitions.value) / jnp.maximum(variance, 1.e-12)
        return explained, variance

    return evaluate
