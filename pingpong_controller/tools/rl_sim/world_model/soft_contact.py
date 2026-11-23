"""Finite 1 ms contact prior using native softness and a ball-side mass model.

This is a reduced approximation, not a replacement for V324's teacher solver.
It neglects racket acceleration and dynamic recoil in the effective mass.
The normal row is solved first; tangential forces are capped in the existing
anisotropic friction ellipse. The compiled reference inverse mass in the
constraint regularizer is kept separate from the episode's physical mass.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from mujoco.mjx._src.math import make_frame
import numpy as np

from .artifacts import read_array_archive
from .contracts import authenticate, file_hash, repository_root


@dataclass(frozen=True)
class SoftContactContract:
    mass_kg: float
    inertia_kg_m2: float
    dt_s: float
    solref: tuple[float, float]
    solimp: tuple[float, ...]
    friction: tuple[float, float]
    reference_inverse_mass_kg_inv: float
    impratio: float
    reference_safe: bool

    def __post_init__(self):
        values = [self.mass_kg, self.inertia_kg_m2, self.dt_s,
                  self.reference_inverse_mass_kg_inv, self.impratio,
                  *self.solref, *self.solimp, *self.friction]
        if (not np.isfinite(values).all() or self.mass_kg <= 0 or self.inertia_kg_m2 <= 0 or self.impratio <= 0
                or self.reference_inverse_mass_kg_inv <= 0 or self.dt_s != .001
                or len(self.solref) != 2 or min(self.solref) <= 0
                or len(self.solimp) != 5 or len(self.friction) != 2 or min(self.friction) <= 0):
            raise ValueError('Require positive native 1 ms, condim3, isotropic-ball parameters')


def contract_from_collection(cfg: dict, collection: dict) -> tuple[SoftContactContract, dict]:
    """Read actual uniform episode context; no substitution of template mass."""
    root = repository_root()
    model_path = authenticate(root, cfg['collection']['compiled_model_cache'])
    model = mujoco.MjModel.from_binary_path(str(model_path))
    context_ref = collection['context']['reference']
    manifest_path = Path(context_ref['manifest'])
    if file_hash(manifest_path) != context_ref['sha256']:
        raise ValueError('Recorded contact context manifest changed')
    manifest = json.loads(manifest_path.read_text())
    if manifest['context_sha256'] != collection['context']['sha256']:
        raise ValueError('Recorded contact context identity mismatch')
    fields = ('.opt.impratio', '.body_mass', '.body_inertia', '.body_invweight0')
    keys = {leaf['path']: f'leaf_{i:06d}' for i, leaf in enumerate(manifest['leaves']) if leaf['path'] in fields}
    if set(keys) != set(fields):
        raise ValueError('Recorded contact context omitted mass or regularization')
    arrays = read_array_archive(manifest_path.parent/manifest['archive_file'], manifest['archive'], names=keys.values())
    ball = int(model.geom_bodyid[model.geom('ball').id])
    racket = collection['geometry']['racket']['body_id']
    mass = arrays[keys['.body_mass']][:, ball]
    inertia = arrays[keys['.body_inertia']][:, ball]
    inverse = arrays[keys['.body_invweight0']][:, [ball, racket], 0].sum(axis=-1)
    impratio = arrays[keys['.opt.impratio']]
    for name, value in (('mass', mass), ('inertia', inertia), ('inverse mass', inverse), ('impratio', impratio)):
        if not np.isfinite(value).all() or not np.allclose(value, value.flat[0], rtol=1e-6, atol=1e-12):
            raise ValueError(f'This fixed-context prior cannot silently average nonuniform {name}')
    pairs = collection['geometry']['pairs']
    if pairs['dim'] != [3] or pairs['margin'] != [0.] or pairs['gap'] != [0.]:
        raise ValueError('This prior only supports the actual zero-margin condim3 pair')
    if pairs['solreffriction'] != [[0., 0.]]:
        raise ValueError('Separate tangential softness requires an explicit implementation')
    contract = SoftContactContract(float(mass[0]), float(inertia[0, 0]), .001,
        tuple(pairs['solref'][0]), tuple(pairs['solimp'][0]), tuple(pairs['friction'][0][:2]),
        float(inverse[0]), float(impratio[0]),
        not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE))
    return contract, dict(context_sha256=collection['context']['sha256'],
        context_manifest_sha256=context_ref['sha256'], compiled_model_sha256=file_hash(model_path),
        effective_mass_approximation='ball_only_diagonal', racket_acceleration='zero_in_prior',
        force_approximation='normal_row_then_anisotropic_friction_cap',
        physical_parameter_fitting=False, teacher_modified=False)


def softness(contract: SoftContactContract, distance_m):
    """Native positive-solref coefficients with finite derivatives at saturation."""
    timeconst, damping_ratio = contract.solref
    if contract.reference_safe:
        timeconst = max(timeconst, 2*contract.dt_s)
    dmin, dmax, width, midpoint, power = contract.solimp
    dmin, dmax = np.clip(dmin, mujoco.mjMINIMP, mujoco.mjMAXIMP), np.clip(dmax, mujoco.mjMINIMP, mujoco.mjMAXIMP)
    midpoint = float(np.clip(midpoint, mujoco.mjMINIMP, mujoco.mjMAXIMP))
    width, power = max(width, mujoco.mjMINVAL), max(power, 1.)
    k = 1/(dmax*dmax*timeconst*timeconst*damping_ratio*damping_ratio)
    b = 2/(dmax*timeconst)
    # Clamp before fractional powers: the saturated native value is identical,
    # while unselected negative bases must not poison autodiff with NaNs.
    x = jnp.clip(jnp.abs(distance_m)/width, 0., 1.)
    low = x**power/midpoint**(power-1)
    high = 1-(1-x)**power/(1-midpoint)**(power-1)
    impedance = jnp.clip(dmin+jnp.where(x < midpoint, low, high)*(dmax-dmin), dmin, dmax)
    return k, b, impedance


class SoftContactPrediction(NamedTuple):
    delta_velocity_world_m_s: Any
    delta_omega_world_rad_s: Any
    force_world_n: Any
    force_contact_n: Any
    impedance: Any


def soft_contact_increment(contract: SoftContactContract, geometry: dict, *,
        ball_position_m, ball_velocity_m_s, ball_omega_world_rad_s,
        geom_position_m, racket_linear_velocity_m_s, racket_angular_velocity_rad_s,
        gravity_m_s2=(0., 0., -9.81)) -> SoftContactPrediction:
    """Predict one contact increment; known gravity is integrated by the caller."""
    normal, point = geometry['contact_normal_world'], geometry['contact_position_world_m']
    frame = jax.vmap(make_frame)(normal.reshape((-1, 3))).reshape(normal.shape[:-1]+(3, 3))
    lever = point-ball_position_m
    relative = (ball_velocity_m_s+jnp.cross(ball_omega_world_rad_s, lever)
                -racket_linear_velocity_m_s-jnp.cross(racket_angular_velocity_rad_s, point-geom_position_m))
    velocity = jnp.einsum('...ij,...j->...i', frame, relative)
    gravity = jnp.einsum('...ij,j->...i', frame, jnp.asarray(gravity_m_s2, jnp.float32))
    distance = geometry['sphere_gap_m']
    k, b, impedance = softness(contract, distance)
    aref = (-b*velocity).at[..., 0].add(-k*impedance*distance)
    rotation_lever = jnp.cross(lever[..., None, :], frame)
    inverse_effective_mass = 1/contract.mass_kg+jnp.sum(rotation_lever**2, axis=-1)/contract.inertia_kg_m2
    mu = jnp.asarray(contract.friction, jnp.float32)
    ratios = jnp.stack((jnp.asarray(1.), jnp.asarray(1/contract.impratio),
                        mu[0]**2/(contract.impratio*mu[1]**2)))
    regularizer = contract.reference_inverse_mass_kg_inv*(1-impedance)/impedance
    regularizer = jnp.maximum(regularizer[..., None]*ratios, mujoco.mjMINVAL)
    desired = (aref-gravity)/(inverse_effective_mass+regularizer)
    normal_force = jnp.maximum(desired[..., 0], 0.)
    tangent = desired[..., 1:]
    friction_norm = jnp.sqrt(jnp.maximum(jnp.sum((tangent/mu)**2, axis=-1), 1e-24))
    tangent = tangent*jnp.minimum(1., normal_force/friction_norm)[..., None]
    force = jnp.concatenate((normal_force[..., None], tangent), axis=-1)
    force = jnp.where((distance < 0.)[..., None], force, 0.)
    world_force = jnp.einsum('...ji,...j->...i', frame, force)
    return SoftContactPrediction(world_force*(contract.dt_s/contract.mass_kg),
        jnp.cross(lever, world_force)*(contract.dt_s/contract.inertia_kg_m2), world_force, force, impedance)
