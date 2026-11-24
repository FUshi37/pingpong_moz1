"""Stagger training-start ages using genuine fixed-parent short trajectories."""
from collections import namedtuple
import json
import numpy as np


def balanced_targets(n_short, *, maximum_blocks, seed):
    if n_short < 1 or maximum_blocks < 1:
        raise ValueError('Invalid rehearsal staggering')
    target = np.arange(n_short, dtype=np.int32) % (maximum_blocks + 1)
    np.random.default_rng(seed).shuffle(target)
    return target


def freeze_lanes(new, old, active, *, xp):
    import jax
    def select(a, b):
        if hasattr(a, 'shape') and a.ndim and a.shape[0] == len(active):
            return xp.where(active.reshape((len(active),) + (1,) * (a.ndim - 1)), a, b)
        return b
    return jax.tree.map(select, new, old)


def merge_population(full, short, *, n_full, n_short, xp=np):
    import jax
    def merge(a, b):
        if hasattr(a, 'shape') and a.ndim and a.shape[0] == n_full:
            if b.shape != (n_short,) + a.shape[1:]:
                raise ValueError('Incompatible warm population shape')
            return xp.concatenate((b, a[n_short:]), axis=0)
        if not np.array_equal(np.asarray(a), np.asarray(b)):
            raise ValueError('Warm population changed nonpopulation constants')
        return a
    return jax.tree.map(merge, full, short)


def warm_rehearsals(learner, env, runner, collect, config, digest):
    import ctypes
    import gc
    import jax
    import jax.numpy as jnp
    if not config['enabled']:
        return runner, {'enabled': False}
    assert config['steps_per_block'] == 128 and config['dt_s'] == .005
    assert env.dt == .005 and env.n_envs == 1024
    n_short = config['warmup_n_envs']
    assert n_short == env.n_rehearsal == 102
    targets = balanced_targets(n_short, maximum_blocks=config['warmup_blocks_max'],
                               seed=config['seed'])
    assert int(targets.max()) * 128 * .005 < 20.
    def long_part(value):
        relevant=(value.env_state,value.obs,value.critic_obs,
                  value.running_return,value.running_length)
        return jax.tree.map(lambda x: x[n_short:] if hasattr(x,'shape') and x.ndim
                            and x.shape[0] == 1024 else x,relevant)
    jax.block_until_ready(runner)
    before_trim=digest(runner)
    jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
    assert digest(runner)==before_trim
    long_before=digest(long_part(runner))
    parent_before=digest(learner)
    @jax.jit
    def blend(proposed,previous,active):
        return freeze_lanes(proposed,previous,active,xp=jnp)._replace(rng=proposed.rng)
    failures=0
    for index in range(config['warmup_blocks_max']):
        active=jnp.concatenate((jnp.asarray(targets>index),jnp.zeros(1024-n_short,bool)))
        proposed,transitions=collect(learner.params,runner)
        jax.block_until_ready(proposed.obs)
        failures+=int(jnp.sum(transitions.terminated*active[None,:]))
        assert float(jnp.min(transitions.metrics['center_precision/warmup']))==1.
        # Entire inactive trajectories are padding. Retain all their original
        # physical/history/counter fields, while advancing the independent RNG.
        runner=blend(proposed,runner,active)
        if index in (0,15,30):
            print(json.dumps(dict(phase='rehearsal_stagger_fixed_parent',block=index+1,
                                  optimizer_updates=0,native_batch=1024)),flush=True)
        del transitions,proposed
        if (index+1)%8==0:
            gc.collect();ctypes.CDLL(None).malloc_trim(0)
    assert digest(long_part(runner))==long_before
    assert digest(learner)==parent_before
    age=np.asarray(runner.env_state.step_count[:n_short])*.005
    assert (age>=0.).all() and (age<20.).all()
    assert (np.asarray(runner.env_state.wind_mode)==0).all()
    receipt=dict(enabled=True,scheduled_blocks=targets.tolist(),
        scheduled_valid_short_steps=int(targets.sum())*128,fixed_parent_unchanged=True,
        optimizer_updates=0,long922_initial_state_bitwise_unchanged=True,
        initialization_cache_release_states_unchanged=True,
        source_short_batch=1024,native_full_batch=1024,additional_model_constructors=0,
        all_ages_from_native_full_batch_rollouts=True,
        warmup_physical_failures=failures,short_age_min_s=float(age.min()),
        short_age_max_s=float(age.max()),short_age_mean_s=float(age.mean()),
        episode_seconds_unchanged=[20.,1200.],warmup_never_counts_as1200_success=True)
    del blend
    jax.clear_caches();gc.collect();ctypes.CDLL(None).malloc_trim(0)
    return runner,receipt
