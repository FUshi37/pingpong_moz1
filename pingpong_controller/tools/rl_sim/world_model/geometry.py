"""Extract contact geometry from the instantiated model and audit native FK."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np


def audit_differentiable_fk(environment, state) -> dict:
    """Check actual reset plus 16 joint-limit samples against CPU MuJoCo."""
    import jax
    import jax.numpy as jnp
    from .kinematics import extract_chain, forward_kinematics

    model = environment.mj_model
    baseline = np.asarray(state.data.qpos[0], dtype=np.float64)
    chain = extract_chain(model, baseline, joint_ids=environment.arm_jids,
                          site_id=environment.racket_site_id,
                          geom_id=environment.racket_geom_id)
    rng = np.random.default_rng(2026110502)
    q = rng.uniform(np.asarray(environment.arm_lo), np.asarray(environment.arm_hi), (16, 7))
    q = np.concatenate((baseline[np.asarray(environment.arm_qadr)][None, :], q))
    prediction = jax.jit(jax.vmap(lambda a: forward_kinematics(chain, a)))(jnp.asarray(q, jnp.float32))
    data = mujoco.MjData(model)
    maxima = dict(site_position_m=0., geom_position_m=0., rotation=0., jacobian=0., autodiff=0.)
    for index, joint_q in enumerate(q):
        data.qpos[:] = baseline
        data.qpos[np.asarray(environment.arm_qadr)] = joint_q
        mujoco.mj_forward(model, data)
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jp, jr, environment.racket_site_id)
        expected = dict(site_position_m=data.site_xpos[environment.racket_site_id],
            geom_position_m=data.geom_xpos[environment.racket_geom_id],
            rotation=data.site_xmat[environment.racket_site_id].reshape(3, 3),
            jacobian=np.concatenate((jp[:, environment.arm_vadr], jr[:, environment.arm_vadr])))
        actual = dict(site_position_m=prediction.site_position[index],
            geom_position_m=prediction.geom_position[index], rotation=prediction.site_rotation[index],
            jacobian=jnp.concatenate((prediction.site_jacobian[index], prediction.angular_jacobian[index])))
        for name in expected:
            maxima[name] = max(maxima[name], float(np.max(np.abs(np.asarray(actual[name])-expected[name]))))
    derivative = jax.jacfwd(lambda a: forward_kinematics(chain, a).site_position)(jnp.asarray(q[0], jnp.float32))
    maxima['autodiff'] = float(np.max(np.abs(np.asarray(derivative-prediction.site_jacobian[0]))))
    if any(value > 3e-6 for value in maxima.values()):
        raise ValueError(f'Compiled arm FK/Jacobian mismatch: {maxima}')
    return dict(status='PASS', configurations=len(q), seed=2026110502,
                max_absolute_errors=maxima, tolerance=3e-6,
                configuration_admissibility='not_a_reachability_or_collision_certificate')


def compiled_geometry(environment, state) -> dict:
    """Record compiled surfaces, mass/inertia and site transforms in SI units."""
    model = environment.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = np.asarray(state.data.qpos[0], dtype=np.float64)
    data.qvel[:] = np.asarray(state.data.qvel[0], dtype=np.float64)
    mujoco.mj_forward(model, data)
    racket = environment.racket_geom_id
    site = environment.racket_site_id
    ball_body = environment.ball_body_id
    ball_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'ball_geom')
    if ball_geom < 0:
        candidates = np.flatnonzero((model.geom_bodyid == ball_body)
                                    & (model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE))
        if len(candidates) != 1:
            raise ValueError('Compiled ball sphere identity is ambiguous')
        ball_geom = int(candidates[0])
    mesh_id = int(model.geom_dataid[racket])
    if model.geom_type[racket] != mujoco.mjtGeom.mjGEOM_MESH or mesh_id < 0:
        raise ValueError('V324 requires the actual finite racket mesh')
    start, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    vertices = np.asarray(model.mesh_vert[start:start+count])
    face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    faces = np.asarray(model.mesh_face[face_start:face_start+face_count])
    # qpos0 alone omits the reset's calibrated arm configuration. Compare FK
    # using the actual reset qpos, with the physical base untouched.
    site_rotation = data.site_xmat[site].reshape(3, 3)
    geom_rotation = data.geom_xmat[racket].reshape(3, 3)
    offset_site = site_rotation.T @ (data.geom_xpos[racket] - data.site_xpos[site])
    relative_rotation = site_rotation.T @ geom_rotation
    mjx_pos = np.asarray(state.data.site_xpos[0, site])
    mjx_rotation = np.asarray(state.data.site_xmat[0, site]).reshape(3, 3)
    pos_error = float(np.max(np.abs(mjx_pos-data.site_xpos[site])))
    rotation_error = float(np.max(np.abs(mjx_rotation-site_rotation)))
    if pos_error > 2e-6 or rotation_error > 2e-6:
        raise ValueError(f'CPU/MJX reset FK mismatch: {pos_error}, {rotation_error}')
    digest = hashlib.sha256()
    for name in ('body_parentid', 'body_pos', 'body_quat', 'body_mass', 'body_inertia',
                 'jnt_type', 'jnt_pos', 'jnt_axis', 'jnt_range', 'jnt_qposadr',
                 'site_bodyid', 'site_pos', 'site_quat', 'geom_type', 'geom_size',
                 'geom_bodyid', 'geom_pos', 'geom_quat', 'geom_dataid', 'mesh_vert',
                 'mesh_face', 'pair_geom1', 'pair_geom2', 'pair_friction',
                 'pair_solref', 'pair_solimp', 'pair_solreffriction', 'pair_margin',
                 'pair_gap', 'pair_dim'):
        value = np.asarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(str((value.shape, value.dtype.str)).encode())
        digest.update(value.tobytes())
    pair_ids = np.asarray(environment.ball_racket_pair_ids, dtype=int)
    return dict(
        schema='wmj_compiled_geometry_v1', geometry_sha256=digest.hexdigest(),
        source='actual_V324_constructor_with_WMJ_domain_overrides',
        timestep_s=float(model.opt.timestep), gravity_m_s2=model.opt.gravity.tolist(),
        integrator=int(model.opt.integrator), solver_iterations=int(model.opt.iterations),
        line_search_iterations=int(model.opt.ls_iterations),
        ball=dict(radius_m=float(model.geom_size[ball_geom, 0]),
                  compiled_template_mass_kg=float(model.body_mass[ball_body]),
                  episode_mass_kg=float(state.model.body_mass[0, ball_body]),
                  episode_inertia_kg_m2=np.asarray(state.model.body_inertia[0, ball_body]).tolist()),
        racket=dict(geom_id=int(racket), body_id=int(environment.racket_body_id),
                    site_id=int(site), mesh_id=mesh_id, vertex_count=count,
                    face_count=face_count, compiled_vertices_local_m=vertices.tolist(),
                    compiled_faces=faces.tolist(), geom_size_m=model.geom_size[racket].tolist(),
                    site_to_geom_translation_m=offset_site.tolist(),
                    site_to_geom_rotation=relative_rotation.tolist()),
        pairs={name: np.asarray(getattr(model, 'pair_'+name))[pair_ids].tolist()
               for name in ('friction', 'solref', 'solimp', 'solreffriction', 'margin', 'gap', 'dim')},
        reset_fk=dict(status='PASS', position_max_abs_error_m=pos_error,
                      rotation_max_abs_error=rotation_error, tolerance=2e-6,
                      joint_order=list(environment.arm_jids),
                      qpos=np.asarray(state.data.qpos[0]).tolist(),
                      racket_site_world_m=data.site_xpos[site].tolist()),
    )
