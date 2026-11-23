"""Authenticate existing full-chain execution and39-D contact checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import jax

from .artifacts import read_array_archive
from .audit_support_replay import load_support_chain
from .contact_geometry import surface_from_compiled_model
from .contracts import config_hash, file_hash
from .neural import init_adam, init_mlp
from .position_contact import position_scale_from_statistics
from .robot_support_model import JOINT_NAMES, support_scale_from_statistics
from .soft_contact import SoftContactContract
from .state_adapter import restore_numeric_state
from .model_numerics import precision_scope


def _authenticate_sources(metadata: dict, names: tuple[str, ...]) -> None:
    for name in names:
        if metadata['source_sha256'][name] != file_hash(Path(__file__).parent/name):
            raise ValueError(f'Physical inference source changed: {name}')


def validate_contact_conditioning(schema: str, conditioning: str) -> None:
    """Separate observed-robot and coupled-fit provenance without new inputs."""
    allowed = {
        'wmj_contact_component_checkpoint_v1': 'recorded_native_pose_twist_and_joint_positions',
        'wmj_contact_coupled_checkpoint_v1': 'own_predicted_full_chain_pose_twist_and_joint_positions',
    }
    if schema not in allowed or conditioning != allowed[schema]:
        raise ValueError('Unregistered contact checkpoint/conditioning contract')


def load_physics_components(cfg: dict, reader, *, robot_checkpoint: Path,
                            contact_checkpoint: Path) -> tuple[dict, dict, dict]:
    """Load matching data/physics identities and complete numerical model state.

    Optimizer trees are authenticated while loading, even though evaluation
    only uses parameters and frozen normalization. Returned components remain
    experimental; loading is not an accuracy or deployment qualification.
    """
    identity = reader.dataset.identity
    if (identity['control_sha256'] != config_hash(cfg['control'])
            or identity['contact_sha256'] != config_hash(cfg['physics'])):
        raise ValueError('Physics data must preserve the selected control/contact contract')
    robot, contact = (json.loads(path.read_text())
                      for path in (robot_checkpoint, contact_checkpoint))
    for metadata in (robot, contact):
        if metadata['dataset_sha256'] != reader.identity or metadata['physical_contract'] != identity:
            raise ValueError('Physics components must share the coherent replay and physical contract')
    if (robot['schema'] != 'wmj_support_robot_checkpoint_v1'
            or robot['structure']['widths'] != [47, 128, 128, 26]
            or robot['structure']['joint_names'] != list(JOINT_NAMES)
            or robot['model_signature'] != config_hash(dict(dataset=reader.identity, structure=robot['structure']))):
        raise ValueError('Expected the approved full13-joint execution checkpoint')
    s = contact['structure']
    precision_scope(s.get('composed_matmul_precision'))
    if contact['fit_recipe'].get('matmul_precision') != s.get('composed_matmul_precision'):
        raise ValueError('Checkpoint inference precision disagrees with its fitting recipe')
    if s.get('composed_matmul_precision') is not None:
        _authenticate_sources(contact, ('model_numerics.py', 'world_physics.py', 'physics_fitting.py'))
    validate_contact_conditioning(contact['schema'], s['conditioning'])
    if (s['widths'] != [39, 256, 256, 128, 8]
            or s['position_conditioning']['input_dq'] or s['position_conditioning']['input_ddq']
            or contact['model_signature'] != config_hash(dict(structure=s, dataset=reader.identity,
                physics=cfg['physics'], geometry=reader.report['geometry']))):
        raise ValueError('Expected the approved39-D contact checkpoint and normalization')
    if (contact['schema'] == 'wmj_contact_coupled_checkpoint_v1'
            and contact['frozen_robot_reference']['sha256'] != file_hash(robot_checkpoint)):
        raise ValueError('Coupled contact requires its registered frozen execution proxy')
    _authenticate_sources(robot, ('robot_model.py', 'robot_support_model.py', 'neural.py',
                                  'kinematics.py', 'support_losses.py'))
    _authenticate_sources(contact, ('contact_model.py', 'contact_dynamics.py', 'contact_surface.py',
        'contact_geometry.py', 'position_contact.py', 'hybrid_contact.py', 'soft_contact.py', 'neural.py'))
    model, chain = load_support_chain(cfg, reader.report['geometry'])
    surface = surface_from_compiled_model(model, reader.report['geometry']['racket']['geom_id'], normal_axis=0)
    if s['collision_geometry'] != surface.identity:
        raise ValueError('Contact geometry and merged mesh identity changed')
    stats = robot['normalization']
    scale = support_scale_from_statistics(stats['inputs'], stats['residuals'])
    rp = init_mlp(jax.random.PRNGKey(0), robot['structure']['widths'])
    restored_robot = restore_numeric_state(
        dict(parameters=rp, optimizer=init_adam(rp), scale=scale, rng=jax.random.PRNGKey(0)),
        read_array_archive(robot_checkpoint.parent/robot['archive_file'], robot['archive']),
        robot['state'], context_sha256=robot['model_signature'])
    cp = init_mlp(jax.random.PRNGKey(0), s['widths'])
    restored_contact = restore_numeric_state(dict(parameters=cp, optimizer=init_adam(cp)),
        read_array_archive(contact_checkpoint.parent/contact['archive_file'], contact['archive']),
        contact['state'], context_sha256=contact['model_signature'])
    response = s.get('response')
    if response is not None and response['kind'] != 'soft_residual':
        raise ValueError('Expected a learned residual or pure neural contact response')
    options = dict(chain=chain, surface=surface, robot_scale=restored_robot['scale'],
        position_scale=position_scale_from_statistics(s['position_conditioning']['statistics']),
        ball_radius_m=float(reader.report['geometry']['ball']['radius_m']),
        contact_contract=SoftContactContract(**response['contract']) if response else None,
        matmul_precision=s.get('composed_matmul_precision'))
    references = {key:dict(path=str(path.absolute()), sha256=file_hash(path),
        model_signature=metadata['model_signature'], optimizer_updates=metadata['optimizer_updates'])
        for key,path,metadata in (('robot',robot_checkpoint,robot),('contact',contact_checkpoint,contact))}
    return dict(robot=restored_robot['parameters'], contact=restored_contact['parameters']), options, references
