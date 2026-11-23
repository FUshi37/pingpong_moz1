"""Task-scaled pairing of the private native evaluation graph."""

import jax
import numpy as np


DEFAULT_LIMITS=dict(qpos=1e-5,qvel=1e-3,ball_linear_velocity=1e-5,
                    reference_q=1e-5,previous_accepted_action=1e-5)


def assess_native_pair(left,right,admitted,environment,*,limits=None):
    """Keep minor compiler motion differences; reject ball/control/clock changes.

    This is a numerical integration check, not task or model qualification.
    The0.001rad/s generalized-velocity ceiling is under0.06deg/s; ball linear
    velocity has its own1e-5m/s check. Keep the old1e-4 smoke result visible.
    The V324 execution implementation and its previous evidence stay unchanged.
    """
    mask=np.asarray(admitted,bool)
    limits=DEFAULT_LIMITS if limits is None else limits
    if set(limits)!=set(DEFAULT_LIMITS) or any(not np.isfinite(x) or x<=0 for x in limits.values()):
        raise ValueError('Expected positive explicit native comparison limits')
    if not mask.any():
        return dict(pass_=False,reason='no_admitted_pair_lanes')
    ball=slice(environment.ball_vadr,environment.ball_vadr+3)
    pairs=dict(qpos=(left.data.qpos,right.data.qpos),qvel=(left.data.qvel,right.data.qvel),
        ball_linear_velocity=(left.data.qvel[:,ball],right.data.qvel[:,ball]),
        reference_q=(left.arm_cmd_q,right.arm_cmd_q),previous_accepted_action=(left.prev_action,right.prev_action))
    motion={}
    for name,(a,b) in pairs.items():
        tolerance=limits[name]
        difference=np.asarray(a)[mask]-np.asarray(b)[mask]
        error=float(np.max(np.abs(difference)))
        motion[name]=dict(max_abs=error if np.isfinite(error) else None,tolerance=tolerance,
            pass_=bool(np.isfinite(error) and error<=tolerance))
    leaves_a,structure_a=jax.tree_util.tree_flatten_with_path(left)
    leaves_b,structure_b=jax.tree_util.tree_flatten_with_path(right)
    if structure_a!=structure_b: raise ValueError('Native state layout changed')
    discrete=[]
    for (path,a),(_,b) in zip(leaves_a,leaves_b):
        a,b=np.asarray(a),np.asarray(b)
        name=jax.tree_util.keystr(path)
        if a.dtype.kind not in 'biu' or '.data.' in name:
            continue
        if a.ndim and a.shape[0]==len(mask): a,b=a[mask],b[mask]
        if not np.array_equal(a,b): discrete.append(name)
    return dict(pass_=all(x['pass_'] for x in motion.values()) and not discrete,
        observables=motion,discrete_changes=discrete,
        strict_velocity_smoke_pass=(motion['qvel']['max_abs'] is not None
                                   and motion['qvel']['max_abs']<=1e-4),
        bitwise_float_equality_required=False,
        scope='native_evaluation_graph_only; physical_parameters_and_EXEC_unchanged')
