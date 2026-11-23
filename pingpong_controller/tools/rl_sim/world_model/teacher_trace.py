"""Authenticated, private 1 kHz observation of the unchanged native physics loop.

This adds outputs, never actuator/contact arithmetic. The native source stays
untouched. The observer requires an actual paired replay before collection.
Contact arrays describe the solver interval beginning at the pre-step qpos;
post-step site_xpos is deliberately not used as a post-integration pose.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap

import jax
import jax.numpy as jnp

from .contracts import file_hash


def recorded_loop(substep, initial, count: int, sample):
    """Carry the original state unchanged alongside a separately written trace."""
    example = jax.eval_shape(lambda: sample(0, initial, initial))
    trace = jax.tree.map(lambda a: jnp.zeros((count,)+a.shape, a.dtype), example)

    def tick(index, carry):
        before, trace = carry
        after = substep(index, before)
        observed = sample(index, before, after)
        trace = jax.tree.map(lambda buffer, value: buffer.at[index].set(value), trace, observed)
        return after, trace

    return jax.lax.fori_loop(0, count, tick, (initial, trace))


def contact_pair_masks(geoms, distance, *, ball_geom_id, racket_geom_id):
    """Identify native pairs without treating contact-slot allocation as impact."""
    pair = (((geoms[..., 0] == ball_geom_id) & (geoms[..., 1] == racket_geom_id))
            | ((geoms[..., 1] == ball_geom_id) & (geoms[..., 0] == racket_geom_id)))
    ball = (geoms[..., 0] == ball_geom_id) | (geoms[..., 1] == ball_geom_id)
    other = ball & ~pair & (distance <= 0.)
    return pair, other


def point_twist(position, axes, anchors, dq_rad_s, hinge_mask):
    """World twist from native preintegration axes and velocities, including base."""
    linear_jacobian = jnp.where(hinge_mask[None, :, None],
        jnp.cross(axes, position[:, None, :]-anchors), axes)
    angular_jacobian = jnp.where(hinge_mask[None, :, None], axes, 0.)
    return (jnp.sum(linear_jacobian*dq_rad_s[:, :, None], axis=1),
            jnp.sum(angular_jacobian*dq_rad_s[:, :, None], axis=1))


class _ObservePhysics(ast.NodeTransformer):
    def __init__(self):
        self.loops = 0
        self.returns = 0

    def visit_Assign(self, node):
        call = node.value
        if (isinstance(call, ast.Call) and ast.unparse(call.func) == 'jax.lax.fori_loop'
                and len(call.args) == 4 and ast.unparse(call.args[2]) == 'one_substep'):
            self.loops += 1
            observer = ast.Call(func=ast.Name('_wmj_recorded_loop', ast.Load()),
                args=[ast.Name('self', ast.Load()), ast.Name('state', ast.Load()), *call.args,
                      ast.Tuple(elts=[ast.Name(n, ast.Load()) for n in (
                          'low_level_rmp_q_history', 'low_level_rmp_qvel_history',
                          'low_level_rmp_qdd_history')], ctx=ast.Load())], keywords=[])
            assignment = ast.Assign(targets=[ast.Tuple(elts=[
                ast.Name('_wmj_loop_result', ast.Store()), ast.Name('_wmj_trace', ast.Store())],
                ctx=ast.Store())], value=observer)
            node.value = ast.Name('_wmj_loop_result', ast.Load())
            return [assignment, node]
        return self.generic_visit(node)

    def visit_Return(self, node):
        if (isinstance(node.value, ast.Tuple) and len(node.value.elts) == 5
                and ast.unparse(node.value.elts[-1]) == 'metrics'):
            self.returns += 1
            assignment = ast.parse("metrics['wmj_trace'] = _wmj_trace").body[0]
            return [assignment, node]
        return self.generic_visit(node)


def augment_source(source: str) -> str:
    """Require unique known anchors; never silently instrument a changed loop."""
    tree = ast.parse(textwrap.dedent(source))
    observer = _ObservePhysics()
    tree = observer.visit(tree)
    if observer.loops != 1:
        raise ValueError('Expected exactly one native one_substep loop')
    if observer.returns != 1:
        raise ValueError('Expected exactly one native five-result return')
    return ast.unparse(ast.fix_missing_locations(tree))


def _observe_native_loop(env, state, lower, upper, substep, initial, commands):
    if lower != 0 or upper != 5 or float(env.timestep) != .001:
        raise ValueError('Only the fixed V324 five-by-one-ms physics path is supported')

    def sample(index, before, after):
        previous, current = before[0], after[0]
        pair_geoms = current.contact.geom
        pair_mask, other_mask = contact_pair_masks(pair_geoms, current.contact.dist,
            ball_geom_id=env.ball_geom_id, racket_geom_id=env.racket_contact_geom_id)
        pair_present = jnp.any(pair_mask, axis=-1)
        distance = jnp.min(jnp.where(pair_mask, current.contact.dist, 1e6), axis=-1)
        active = pair_mask & (current.contact.dist <= 0.)
        # The force is stored separately from geometric overlap; a soft
        # contact with zero reaction is not fabricated into an impulse label.
        addresses = current.contact.efc_address
        force = jnp.take(current.efc_force, jnp.maximum(addresses, 0), axis=-1)
        force = jnp.sum(jnp.where(active & (addresses >= 0), force, 0.), axis=-1)
        interval = slice(env.ball_vadr, env.ball_vadr+6)
        result = dict(
            q_before=previous.qpos[:, env.arm_qadr],
            dq_before=previous.qvel[:, env.arm_vadr],
            q_after=current.qpos[:, env.arm_qadr],
            dq_after=current.qvel[:, env.arm_vadr],
            ball_qpos_before=previous.qpos[:, env.ball_qadr:env.ball_qadr+7],
            ball_qpos_after=current.qpos[:, env.ball_qadr:env.ball_qadr+7],
            ball_qvel_before=previous.qvel[:, interval],
            ball_qvel_after=current.qvel[:, interval],
            delayed_q_command=commands[0][index],
            delayed_dq_command=commands[1][index],
            delayed_ddq_command=commands[2][index],
            actuator_ctrl=current.ctrl[:, env.arm_aids_j],
            contact_present=pair_present, contact_distance_m=distance,
            contact_overlap=jnp.any(active, axis=-1),
            other_ball_collision=jnp.any(other_mask, axis=-1),
            contact_normal_force_n=force,
        )
        if hasattr(env, '_wmj_support_trace'):
            support = env._wmj_support_trace
            ancestors = support['ancestor_ids']
            position = current.geom_xpos[:, env.racket_contact_geom_id]
            velocity, omega = point_twist(position, current.xaxis[:, ancestors],
                current.xanchor[:, ancestors], previous.qvel[:, support['ancestor_dof']],
                support['hinge_mask'])
            result.update(
                support_q_before=previous.qpos[:, support['qadr']],
                support_q_after=current.qpos[:, support['qadr']],
                support_dq_before=previous.qvel[:, support['vadr']],
                support_dq_after=current.qvel[:, support['vadr']],
                racket_geom_position_before_m=position,
                racket_geom_rotation_before=current.geom_xmat[:, env.racket_contact_geom_id].reshape(-1, 3, 3),
                racket_site_position_before_m=current.site_xpos[:, env.racket_site_id],
                racket_linear_velocity_before_m_s=velocity,
                racket_angular_velocity_before_rad_s=omega)
        return result

    return recorded_loop(substep, initial, upper, sample)


def instrument_environment(environment, stack, *, support_trace=False) -> dict:
    """Bind one instance only, retaining V324's combined state constructor."""
    from pathlib import Path
    import importlib
    import types

    native = importlib.import_module('mjx_juggle_env')
    if file_hash(Path(native.__file__)) != stack.spec.native_env_sha256:
        raise ValueError('Native teacher source changed')
    if not environment.recovered_rmp_motion_mode:
        raise ValueError('WMJ observer requires the registered RMP control path')
    if hasattr(environment, '_wmj_trace_identity'):
        raise ValueError('Teacher is already instrumented')
    if support_trace:
        import mujoco
        import numpy as np
        model = environment.mj_model
        ids = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT,
                                        f'LegWaist-{i}') for i in range(6)])
        if np.any(ids < 0) or np.any(model.jnt_type[ids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Expected six native LegWaist hinge joints')
        ancestors = []
        body = int(model.geom_bodyid[environment.racket_contact_geom_id])
        while body:
            start = int(model.body_jntadr[body])
            ancestors.extend(range(start, start+int(model.body_jntnum[body])))
            body = int(model.body_parentid[body])
        ancestors = np.array(ancestors, dtype=np.int32)
        joint_types = model.jnt_type[ancestors]
        if not np.isin(joint_types, [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]).all():
            raise ValueError('Native twist observer requires scalar ancestor joints')
        environment._wmj_support_trace = dict(qadr=model.jnt_qposadr[ids].copy(),
            vadr=model.jnt_dofadr[ids].copy(), ancestor_ids=ancestors,
            ancestor_dof=model.jnt_dofadr[ancestors].copy(),
            hinge_mask=jnp.asarray(joint_types == mujoco.mjtJoint.mjJNT_HINGE))
    original = inspect.getsource(native.MjxJuggleEnv._step_impl)
    augmented = augment_source(original)
    # The V324 wrapper privately supplies its combined camera/cohort state
    # constructor here. Using native globals would lose those persistent fields.
    namespace = dict(environment._step_impl.__globals__)
    namespace['_wmj_recorded_loop'] = _observe_native_loop
    exec(compile(augmented, '<WMJ-V1-native-observer>', 'exec'), namespace)
    environment._step_impl = types.MethodType(namespace['_step_impl'], environment)
    identity = dict(schema='wmj_native_substep_observer_v1',
        native_sha256=stack.spec.native_env_sha256,
        augmented_source_sha256=hashlib.sha256(augmented.encode()).hexdigest(),
        observer_sha256=file_hash(Path(__file__)), qualification='PAIRING_REQUIRED',
        contact_time='pre-integration solver interval',
        ball_qvel_frame='linear_world_angular_ball_local',
        support_trace=support_trace,
        racket_twist_time='preintegration_qvel_and_native_kinematics',
        physical_modifications=False)
    environment._wmj_trace_identity = identity
    return identity
