"""Training-only cumulative policy constraint; the deployed actor stays one MLP."""
from pathlib import Path
import hashlib
import json
import jax
import jax.numpy as jnp
import numpy as np


def load_bank(folder: Path):
    plan=json.loads((folder/'TRAIN_PLAN.json').read_text())
    path=folder/'calm_observations.npy'
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==plan['calm_trust_region']['bank_sha256']
    bank=np.load(path,allow_pickle=False)
    if bank.shape!=(plan['calm_trust_region']['bank_rows'],118) or not np.isfinite(bank).all():
        raise ValueError('Invalid fixed calm/startup observation bank')
    return jnp.asarray(bank)


def guard_update(native_update, policy_mean, anchor, bank, *, enabled=True,
                 maximum_kl=.001, target_kl=.006, sigma=.005, compile_update=True,
                 holdout=None,maximum_sampled_cohort_step_kl=None,sample_size=2048,
                 project_from_anchor=False,phase_grouping=False,
                 maximum_absolute_action_change=None,
                 held_maximum_absolute_action_change=None):
    """Backtrack only actor parameters; retain genuine critic optimization.

    This compares Gaussian action distributions, without action labels, BC,
    teacher outputs or a second deployed actor. Frozen bank predictions come
    from the same initial ordinary single MLP used by PPO's existing anchor.
    """
    if not enabled:return native_update
    reference=jax.lax.stop_gradient(policy_mean(anchor,bank))
    held_reference=None if holdout is None else jax.lax.stop_gradient(policy_mean(anchor,holdout))
    if held_maximum_absolute_action_change is not None and holdout is None:
        raise ValueError('A holdout action bound requires a holdout bank')
    scales=jnp.asarray([2.**(-i) for i in range(13)],jnp.float32)
    if phase_grouping and bank.shape[-1]<113:
        raise ValueError('Phase protection requires the causal118-D observation layout')
    startup=None if not phase_grouping else bank[:,112]<.999
    anchor_pi=jax.tree.map(jax.lax.stop_gradient,anchor['pi']) if project_from_anchor else None

    def kl(before,after):return jnp.mean(jnp.sum((before-after)**2,axis=-1))/(2.*sigma**2)

    def bank_kl(before,after):
        if not phase_grouping:return kl(before,after)
        values=jnp.sum((before-after)**2,axis=-1)/(2.*sigma**2)
        return jnp.maximum(jnp.sum(jnp.where(startup,values,0.))/jnp.maximum(startup.sum(),1),
            jnp.sum(jnp.where(~startup,values,0.))/jnp.maximum((~startup).sum(),1))

    def update(previous,runner,transitions):
        candidate,metrics=native_update(previous,runner,transitions)
        sampled=None
        if maximum_sampled_cohort_step_kl is not None:
            from learning_stability import cohort_step_measure
            sampled=cohort_step_measure(previous.params,transitions,policy_mean,sample_size=sample_size,sigma=sigma)
        old_pi,new_pi=previous.params['pi'],candidate.params['pi']
        initial_kl=bank_kl(reference,policy_mean(candidate.params,bank))
        def try_scale(carry,item):
            selected,selected_kl,selected_scale,found,selected_origin=carry
            origin_flag,scale=item
            origin=(jax.tree.map(lambda a,b:jnp.where(origin_flag>.5,a,b),anchor_pi,old_pi)
                    if project_from_anchor else old_pi)
            pi=jax.tree.map(lambda a,b:a+scale*(b-a),origin,new_pi)
            predicted=policy_mean({**candidate.params,'pi':pi},bank)
            value=bank_kl(reference,predicted)
            choose=(~found)&jnp.isfinite(value)&(value<=maximum_kl)
            if maximum_absolute_action_change is not None:
                error=jnp.max(jnp.abs(predicted-reference))
                choose=choose&jnp.isfinite(error)&(error<=maximum_absolute_action_change)
            if held_maximum_absolute_action_change is not None:
                error=jnp.max(jnp.abs(policy_mean({**candidate.params,'pi':pi},holdout)-held_reference))
                choose=choose&jnp.isfinite(error)&(error<=held_maximum_absolute_action_change)
            if sampled is not None:
                sampled_kl=sampled({**candidate.params,'pi':pi})
                choose=choose&jnp.isfinite(sampled_kl)&(sampled_kl<=maximum_sampled_cohort_step_kl)
            selected=jax.tree.map(lambda a,b:jnp.where(choose,b,a),selected,pi)
            return (selected,jnp.where(choose,value,selected_kl),
                    jnp.where(choose,scale,selected_scale),found|choose,
                    jnp.where(choose,origin_flag,selected_origin)),None
        starting=(old_pi,bank_kl(reference,policy_mean(previous.params,bank)),
                  jnp.asarray(0.),jnp.asarray(False),jnp.asarray(0.))
        origins=(jnp.r_[jnp.ones_like(scales),jnp.zeros_like(scales)]
                 if project_from_anchor else jnp.zeros_like(scales))
        attempts=jnp.tile(scales,2) if project_from_anchor else scales
        (pi,post,scale,found,origin),_=jax.lax.scan(try_scale,starting,(origins,attempts))
        observations=transitions.obs.reshape((-1,transitions.obs.shape[-1]))
        per_update_kl=kl(policy_mean(previous.params,observations),
                         policy_mean({**candidate.params,'pi':pi},observations))
        accept=found&jnp.isfinite(per_update_kl)&(per_update_kl<=target_kl)
        pi=jax.tree.map(lambda new,old:jnp.where(accept,new,old),pi,old_pi)
        params={**candidate.params,'pi':pi}
        boundary=(previous.opt.t==512)&(candidate.opt.t>previous.opt.t)
        def moments(current,old):
            fallback=jax.tree.map(lambda x:jnp.where(boundary,jnp.zeros_like(x),x),old['pi'])
            return {**current,'pi':jax.tree.map(lambda a,b:jnp.where(accept,a,b),current['pi'],fallback)}
        optimizer=candidate.opt._replace(m=moments(candidate.opt.m,previous.opt.m),
            v=moments(candidate.opt.v,previous.opt.v),
            actor_t=jnp.where(accept,candidate.opt.actor_t,jnp.where(boundary,0,previous.opt.actor_t)))
        final=candidate._replace(params=params,opt=optimizer)
        metrics={**metrics,'calm_trust/candidate_kl':initial_kl,
            'ppo_exact_kl':jnp.where(accept,per_update_kl,0.),
            'calm_trust/post_kl':bank_kl(reference,policy_mean(params,bank)),
            'calm_trust/projected':(initial_kl>maximum_kl).astype(jnp.float32),
            'calm_trust/step_scale':jnp.where(accept,scale,0.),
            'calm_trust/actor_rollback':(~accept).astype(jnp.float32),
            'calm_trust/actual_step_kl':jnp.where(accept,per_update_kl,0.),
            'calm_trust/anchor_projection':jnp.where(accept,origin,0.),
            'calm_trust/phase_grouping':jnp.asarray(phase_grouping,jnp.float32)}
        if holdout is not None:
            metrics['calm_trust/holdout_kl']=kl(held_reference,policy_mean(params,holdout))
        if maximum_absolute_action_change is not None:
            metrics['calm_trust/max_action_change']=jnp.max(jnp.abs(policy_mean(params,bank)-reference))
        if held_maximum_absolute_action_change is not None:
            metrics['calm_trust/holdout_max_action_change']=jnp.max(jnp.abs(policy_mean(params,holdout)-held_reference))
        if sampled is not None:
            metrics.update({'learning_stability/candidate_sampled_max_kl':sampled(candidate.params),
                'learning_stability/sampled_max_step_kl':sampled(params),
                'learning_stability/step_scale':jnp.where(accept,scale,0.)})
        return final,metrics
    return jax.jit(update) if compile_update else update
