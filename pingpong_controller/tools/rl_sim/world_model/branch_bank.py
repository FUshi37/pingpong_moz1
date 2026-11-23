"""Bounded CPU full-state seed cache, populated by actual causal events only."""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from .course_sampling import CATEGORIES, COURSE_COUNTS, sample_course_starts
from .persistent_pool import prepare_window


class BranchSeedBank:
    """RAM cache; source data and experiment evidence are never deleted.

    Every stored state has one explicit lane and contains the complete
    environment, including its camera RNG. Exploration RNG is retained too.
    Old imagined drift is unusable under a different model identity or course.
    """

    def __init__(self, *, model_identity, stage_id, capacity_per_category=1024,
                 max_bytes=128*2**20, diagnostic=False):
        if not model_identity or stage_id not in COURSE_COUNTS or capacity_per_category<1 or max_bytes<1:
            raise ValueError('Invalid seed bank contract')
        self.model_identity=model_identity
        self.stage_id=stage_id
        self.capacity_per_category=capacity_per_category
        self.max_bytes=max_bytes
        self.diagnostic=diagnostic
        self.rows=[]
        self._states=[]
        self._keys=[]
        self.evicted=dict.fromkeys(CATEGORIES,0)
        self.serial=0
        self._bytes=0

    def _append(self,state,key,row):
        expected='validation' if self.diagnostic else 'train'
        if row['split']!=expected:
            raise ValueError('Seed partition does not match the bank')
        if state.episode_steps.shape!=(1,) or np.shape(key)!=(1,2):
            raise ValueError('Store one complete lane per seed')
        if any(np.any(np.asarray(x)) for x in (state.terminated,state.truncated,state.model_truncated)):
            raise ValueError('An ended lane cannot become a branch seed')
        state=jax.tree.map(lambda x:np.asarray(x).copy(),jax.device_get(state))
        if any(not np.isfinite(x).all() for x in jax.tree.leaves(state)):
            raise ValueError('Nonfinite complete seed')
        if self._states:
            if jax.tree.structure(state)!=jax.tree.structure(self._states[0]):
                raise ValueError('Seed layout changed within one bank')
            for left,right in zip(jax.tree.leaves(state),jax.tree.leaves(self._states[0])):
                if left.shape!=right.shape or left.dtype!=right.dtype:
                    raise ValueError('Seed numerical layout changed')
        category=row['category']
        indices=[i for i,x in enumerate(self.rows) if x['category']==category]
        removed=indices[0] if len(indices)>=self.capacity_per_category else None
        byte_count=self._bytes+sum(x.nbytes for x in jax.tree.leaves(state))+8
        if removed is not None:
            byte_count-=sum(x.nbytes for x in jax.tree.leaves(self._states[removed]))+8
        if byte_count>self.max_bytes:
            raise MemoryError('Bounded branch seed cache is full')
        if removed is not None:
            for sequence in (self.rows,self._states,self._keys):
                sequence.pop(removed)
            self.evicted[category]+=1
        self.rows.append(dict(row,seed_id=self.serial,model_identity=self.model_identity,
            stage_id=self.stage_id,ar1_enabled=self.stage_id=='S05'))
        self._states.append(state)
        self._keys.append(np.asarray(key,dtype=np.uint32).copy())
        self._bytes=byte_count
        self.serial+=1

    def add_teacher_precontact(self,state,key,*,split,source_id,time_to_contact_s,ar1_enabled=False):
        """Admit a native full seed; future contact time is selection metadata only."""
        if bool(ar1_enabled)!=(self.stage_id=='S05'):
            raise ValueError('Source observation noise contract differs from the course')
        if (np.any(np.asarray(state.events.contact_active))
                or np.any(np.asarray(state.world.physics.ball.previous_contact))
                or not np.isfinite(time_to_contact_s) or time_to_contact_s<=0.):
            raise ValueError('Expected a noncontact seed with measured positive lead')
        self._append(state,key,dict(category='precontact',split=split,source_id=source_id,
            origin='teacher_full_state_only',time_to_contact_s=float(time_to_contact_s)))

    def observe(self,before,after,exploration_keys,task_events,*,split,source_id):
        """Capture complete post-event states; never manufacture successful hits.

        Recovery requires the course scheduler's actual pulse-applied marker.
        It remains a candidate recovery state, not a claim of policy success.
        """
        n=after.episode_steps.shape[0]
        valid=~np.asarray(after.terminated|after.truncated|after.model_truncated)
        switched=np.asarray(task_events.get('goal_switched',np.zeros(n)))>0
        switched &= np.asarray(after.task.goal_id)==np.asarray(before.task.goal_id)+1
        switched &= np.asarray(after.completed_goals)==np.asarray(before.completed_goals)+1
        recovery=np.asarray(task_events.get('disturbance_applied',np.zeros(n)))>0
        enabled=COURSE_COUNTS[self.stage_id]
        added=0
        for category,mask,allowed in (('goal_switch',switched,enabled[1]),
                ('recoverable_disturbance',recovery,enabled[2])):
            if not allowed:
                continue
            for lane in np.flatnonzero(valid & mask):
                state=jax.tree.map(lambda x:x[lane:lane+1],after)
                self._append(state,exploration_keys[lane:lane+1],dict(category=category,
                    split=split,source_id=source_id,origin='causal_imagined_event',
                    parent_lane=int(lane),time_to_contact_s=None,
                    recoverability='unverified_candidate' if category=='recoverable_disturbance' else None))
                added+=1
        return added

    def gather(self,indices):
        indices=np.asarray(indices)
        if (indices.ndim!=1 or not len(indices) or indices.dtype.kind not in 'iu'
                or np.any(indices<0) or np.any(indices>=len(self.rows))):
            raise ValueError('Invalid full-seed indices')
        state=jax.tree.map(lambda *x:jnp.asarray(np.concatenate(x,axis=0)),
            *(self._states[i] for i in indices))
        keys=jnp.asarray(np.concatenate([self._keys[i] for i in indices],axis=0))
        return state,keys

    def snapshot(self):
        """Return numeric tree plus JSON metadata for a single learner checkpoint."""
        if not self.rows:
            raise ValueError('Cannot serialize an empty seed bank')
        states=jax.tree.map(lambda *x:np.concatenate(x,axis=0),*self._states)
        keys=np.concatenate(self._keys,axis=0)
        metadata=dict(schema='wmj_v1_causal_branch_bank',model_identity=self.model_identity,
            stage_id=self.stage_id,capacity_per_category=self.capacity_per_category,
            max_bytes=self.max_bytes,diagnostic=self.diagnostic,rows=copy.deepcopy(self.rows),
            evicted=dict(self.evicted),serial=self.serial)
        return dict(environment=states,exploration_rng=keys),metadata

    @classmethod
    def restore(cls,tree,metadata,*,model_identity,stage_id):
        if (metadata['schema']!='wmj_v1_causal_branch_bank'
                or metadata['model_identity']!=model_identity or metadata['stage_id']!=stage_id):
            raise ValueError('Seed bank model/course identity changed')
        if len(metadata['rows'])!=tree['environment'].episode_steps.shape[0]:
            raise ValueError('Seed bank metadata/numerical count differs')
        bank=cls(model_identity=model_identity,stage_id=stage_id,
            capacity_per_category=metadata['capacity_per_category'],
            max_bytes=metadata['max_bytes'],diagnostic=metadata['diagnostic'])
        for lane,row in enumerate(metadata['rows']):
            if (row['model_identity']!=model_identity or row['stage_id']!=stage_id
                    or row['ar1_enabled']!=(stage_id=='S05')):
                raise ValueError('Seed row identity differs from its snapshot')
            bank._append(jax.tree.map(lambda x:x[lane:lane+1],tree['environment']),
                tree['exploration_rng'][lane:lane+1],row)
            bank.rows[-1]=dict(row)
        bank.evicted=dict(metadata['evicted'])
        bank.serial=metadata['serial']
        return bank


def prepare_training_window(pool,fresh_persistent,bank,rng,fresh_keys,*,minimum_precontact_lead_s):
    """Prepare exactly768 persistent and256 short lanes without online geometry."""
    if (pool.environment.episode_steps.shape!=(1024,)
            or fresh_persistent.episode_steps.shape!=(768,) or fresh_keys.shape!=(1024,2)):
        raise ValueError('The registered pool is exactly768+256 lanes')
    draw=sample_course_starts(rng,bank.rows,stage_id=bank.stage_id,
        model_identity=bank.model_identity,minimum_precontact_lead_s=minimum_precontact_lead_s,
        diagnostic=bank.diagnostic)
    branches,_=bank.gather(draw.indices)
    fresh=jax.tree.map(lambda p,b:jnp.concatenate((p,b),axis=0),fresh_persistent,branches)
    # Future noise is independent per branch; retain realized AR1 noise and
    # all historical samples. Continuing lanes are not rekeyed byprepare_window.
    camera=fresh.world.observation.camera._replace(rng=jax.vmap(
        lambda key:jax.random.fold_in(key,1))(fresh_keys))
    fresh=fresh._replace(world=fresh.world._replace(observation=
        fresh.world.observation._replace(camera=camera)))
    prepared,mask=prepare_window(pool,fresh,fresh_keys,persistent_envs=768)
    return prepared,dict(draw.report,persistent_resets=int(np.asarray(mask[:768]).sum()),
        branch_resets=int(np.asarray(mask[768:]).sum()),seed_indices=draw.indices.tolist(),
        seed_probability=draw.probability.tolist(),runtime_geometry_queries=0)
