"""Finite cached goal catalog and conditional50/30/20 task sampling.

Static admission is performed offline by the caller. This table does not
certify dynamic reachability; unknown transitions stay explicit in task
evaluation. Farther-than20cm edges require supplied feasibility evidence.
"""

from typing import Any, NamedTuple
import jax
import jax.numpy as jnp
import numpy as np

from .sampling import SamplingShortage


class GoalTable(NamedTuple):
    xyz_m: Any
    success_rate: Any
    improvement: Any
    observations: Any
    far_edges: Any


def build_goal_table(xyz_m,*,static_admitted,far_edges=None):
    xyz=np.asarray(xyz_m,np.float32)
    if (xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<1 or not np.isfinite(xyz).all()
            or np.any(xyz<np.array([-.38,-.75,1.30],np.float32))
            or np.any(xyz>np.array([.38,-.28,1.45],np.float32))):
        raise ValueError('Goal candidates must lie inside the registeredXYZ bounds')
    if np.shape(static_admitted)!=(len(xyz),) or not np.all(static_admitted):
        raise ValueError('Unknown static admission must be retained outside the training table')
    if len(np.unique(xyz,axis=0))!=len(xyz):
        raise ValueError('Duplicate goals distort uniform sampling probabilities')
    far=np.zeros((len(xyz),len(xyz)),bool) if far_edges is None else np.asarray(far_edges,bool)
    if far.shape!=(len(xyz),len(xyz)):
        raise ValueError('Far-edge evidence must identify ordered catalog pairs')
    return GoalTable(jnp.asarray(xyz),jnp.zeros(len(xyz)),jnp.zeros(len(xyz)),
                     jnp.zeros(len(xyz)),jnp.asarray(far))


def allowed_goals(table,current,stage_id):
    distance=jnp.linalg.norm(current[:,None]-table.xyz_m[None],axis=-1)
    if stage_id=='S01':
        return distance<1e-6
    if stage_id=='S02':
        return (distance>=.02-1e-7)&(distance<=.08+1e-7)
    closest=jnp.argmin(distance,axis=1)
    found=jnp.min(distance,axis=1)<1e-6
    far=found[:,None] & table.far_edges[closest]
    if stage_id=='S03':
        return ((distance>=.08-1e-7)&(distance<=.20+1e-7)) | ((distance>.20)&far)
    if stage_id in ('S04','S05'):
        return ((distance>1e-6)&(distance<=.20+1e-7)) | ((distance>.20)&far)
    raise ValueError('Unknown course')


def validate_start_goals(table,current,*,stage_id):
    """Host preflight; only cached distances, never IK or robot geometry."""
    if stage_id=='S01':
        return
    # Every reachable catalog node must have a next choice too, preventing a
    # sampled node from becoming a future dead end inside a compiled window.
    current=jnp.concatenate((jnp.asarray(current),table.xyz_m),axis=0)
    available=np.asarray(allowed_goals(table,current,stage_id)).any(axis=1)
    if not available.all():
        raise SamplingShortage(f'{stage_id} goal graph has{int((~available).sum())} missing successors')


def draw_next_goals(table,current,keys,*,stage_id):
    """Draw a proposal; the caller commits RNG only when the goal switches."""
    n=current.shape[0]
    if stage_id=='S01':
        return current,keys,jnp.ones(n,bool),dict(mixture_weights=jnp.tile(jnp.array([1.,0.,0.]),(n,1)),
            probability=jnp.zeros((n,table.xyz_m.shape[0])))
    eligible=allowed_goals(table,current,stage_id)
    frontier=eligible & (table.observations>0) & (table.success_rate>=.2) & (table.success_rate<=.8) & (table.improvement>0)
    review=eligible & (table.observations>0) & (table.success_rate>=.9)
    has_frontier=jnp.any(frontier,axis=1)
    has_review=jnp.any(review,axis=1)
    weights=jnp.stack((.5+.3*~has_frontier+.2*~has_review,.3*has_frontier,.2*has_review),axis=-1)
    def normalized(mask):
        return mask/jnp.maximum(mask.sum(axis=1,keepdims=True),1)
    probability=(weights[:,0,None]*normalized(eligible)+weights[:,1,None]*normalized(frontier)
                 +weights[:,2,None]*normalized(review))
    split=jax.vmap(lambda k:jax.random.split(k,2))(keys)
    chosen=jax.vmap(jax.random.categorical)(split[:,1],jnp.where(probability>0,jnp.log(jnp.maximum(probability,1e-30)),-jnp.inf))
    return table.xyz_m[chosen],split[:,0],jnp.any(eligible,axis=1),dict(
        mixture_weights=weights,probability=probability)
