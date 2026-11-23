"""Offline oracle: how much prior error is due to ignored racket acceleration?

Recorded dq_after is a future label, used ONLY for this diagnosis. These
results cannot qualify a learned world or enter an imagined policy rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import write_json_unique
from .audit_support_replay import load_support_chain
from .contact_geometry import surface_from_compiled_model, surface_features
from .contracts import file_hash, load_config, repository_root
from .fit_robot import summarize_errors
from .kinematics import forward_kinematics
from .native_contact_data import prepare_native_contact_batch, read_native_contact_trial
from .resources import StorageBudget
from .soft_contact import contract_from_collection, soft_contact_increment, softness
from .support_data import SupportReplayReader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if jax.default_backend() != 'cpu':
        raise ValueError('This training-data oracle is a CPU-only diagnostic')
    cfg, root = load_config(), repository_root()
    r = cfg['resources']
    budget = StorageBudget(root/r['output_root'], r['max_new_bytes'], r['min_free_bytes'])
    reader = SupportReplayReader(args.replay, budget, max_bytes=0)
    collection = json.loads((Path(reader.report['source_collection'])/'result.json').read_text())
    prior, context = contract_from_collection(cfg, collection)
    model, chain = load_support_chain(cfg, reader.report['geometry'])
    surface = surface_from_compiled_model(model, reader.report['geometry']['racket']['geom_id'], normal_axis=0)
    radius = float(reader.report['geometry']['ball']['radius_m'])

    @jax.jit
    def predict(raw):
        batch = prepare_native_contact_batch(None, raw)
        state, motion = batch['before'], batch['racket']
        geometry = surface_features(surface, state.position_m, motion['geom_position_m'], motion['geom_rotation_world'], radius)
        fk = jax.vmap(lambda q: forward_kinematics(chain, q))(raw['q_before'])
        qacc_oracle = (raw['dq_after']-raw['dq_before'])/.001
        geom_acc = jnp.einsum('nij,nj->ni', fk.geom_jacobian, qacc_oracle)
        angular_acc = jnp.einsum('nij,nj->ni', fk.angular_jacobian, qacc_oracle)
        point_acc = geom_acc+jnp.cross(angular_acc, geometry['contact_position_world_m']-motion['geom_position_m'])
        physical = dict(ball_position_m=state.position_m, ball_velocity_m_s=state.velocity_m_s,
            ball_omega_world_rad_s=state.omega_world_rad_s,
            **{name: value for name, value in motion.items() if name != 'geom_rotation_world'})
        base = soft_contact_increment(prior, geometry, **physical)
        # In the relative acceleration equation only, a_free = g - J_racket*qacc.
        # The ball's actual gravity remains unchanged in the external integration.
        corrected = jax.vmap(lambda geom, values, accel: soft_contact_increment(prior, geom,
            **values, gravity_m_s2=jnp.array([0., 0., -9.81])-accel))(geometry, physical, point_acc)
        truth = batch['after'].velocity_m_s-state.velocity_m_s-jnp.array([0., 0., -.00981])
        normal = geometry['contact_normal_world']
        relative = (state.velocity_m_s+jnp.cross(state.omega_world_rad_s,
                    geometry['contact_position_world_m']-state.position_m)
                    -motion['racket_linear_velocity_m_s']-jnp.cross(motion['racket_angular_velocity_rad_s'],
                    geometry['contact_position_world_m']-motion['geom_position_m']))
        normal_velocity = jnp.sum(relative*normal, axis=-1)
        k, b, impedance = softness(prior, geometry['sphere_gap_m'])
        normal_force = raw['contact_normal_force_n']
        measured_normal_increment = jnp.sum(truth*normal, axis=-1)
        normal_acceleration = jnp.sum(point_acc*normal, axis=-1)
        normal_regularizer = prior.reference_inverse_mass_kg_inv*(1-impedance)/impedance
        normal_aref = -b*normal_velocity-k*impedance*geometry['sphere_gap_m']
        # Diagnostic equality only: sliding contacts can couple normal and
        # tangential forces, so this residual need not vanish in the cone zone.
        force_balance = (normal_force*(1/prior.mass_kg+normal_regularizer)
                         -(normal_aref+9.81*normal[..., 2]+normal_acceleration))
        return dict(base_error_m_s=jnp.linalg.norm(base.delta_velocity_world_m_s-truth, axis=-1),
            acceleration_oracle_error_m_s=jnp.linalg.norm(corrected.delta_velocity_world_m_s-truth, axis=-1),
            base_normal_error_m_s=jnp.sum((base.delta_velocity_world_m_s-truth)*normal, axis=-1),
            oracle_normal_error_m_s=jnp.sum((corrected.delta_velocity_world_m_s-truth)*normal, axis=-1),
            racket_point_acceleration_m_s2=jnp.linalg.norm(point_acc, axis=-1),
            racket_normal_acceleration_m_s2=jnp.sum(point_acc*normal, axis=-1),
            gap_m=geometry['sphere_gap_m'], normal_velocity_m_s=normal_velocity,
            impedance=impedance, teacher_normal_force_n=normal_force,
            prior_normal_force_n=base.force_contact_n[..., 0],
            force_integration_error_m_s=measured_normal_increment-normal_force*.001/prior.mass_kg,
            normal_force_balance_m_s2=force_balance,
            tangential_speed_m_s=jnp.linalg.norm(relative-normal_velocity[..., None]*normal, axis=-1),
            near_rim=geometry['edge_signed_distance_m'] > -.01)

    rng = np.random.default_rng(2026110703)
    accumulated, rows = {}, []
    names = list(reader.records)
    for i in range(0, len(names), 2):
        budget.check()
        data, record = read_native_contact_trial(reader, names[i], names[i+1])
        forces = [reader.read(name, names={'substep__contact_normal_force_n'})[
            'substep__contact_normal_force_n'] for name in names[i:i+2]]
        data['contact_normal_force_n'] = np.concatenate([
            force.reshape((-1,)+force.shape[2:]) for force in forces])
        train = np.array([group['split'] == 'train' for group in record['groups']])[None]
        times, lanes = np.nonzero(train & data['ball_model_valid'] & data['contact_overlap'])
        count = min(512, len(times))
        if not count:
            continue
        selected = rng.choice(len(times), size=count, replace=False)
        selected = np.pad(selected, (0, 512-count), mode='edge')
        raw = {name: jnp.asarray(value[times[selected], lanes[selected]]) for name, value in data.items()}
        values = {key: np.asarray(value)[:count] for key, value in jax.device_get(predict(raw)).items()}
        for key, value in values.items():
            accumulated.setdefault(key, []).append(value)
        rows.append(dict(trial_pair=i//2, count=count,
            base_error_m_s=summarize_errors([values['base_error_m_s']]),
            acceleration_oracle_error_m_s=summarize_errors([values['acceleration_oracle_error_m_s']])))
        print(f'[AUDIT] trial={i//2+1}/{len(names)//2}', flush=True)
        del data
    values = {name: np.concatenate(parts) for name, parts in accumulated.items()}
    groups = dict(all=np.ones(len(values['near_rim']), bool), near_rim=values['near_rim'], interior=~values['near_rim'])
    worst = np.argsort(values['base_error_m_s'])[-20:][::-1]
    report = dict(schema='wmj_soft_acceleration_oracle_v1', training_version='WMJ-V1',
        source_sha256=file_hash(Path(__file__)), context=context, replay_sha256=reader.identity,
        scope='recorded_future_robot_acceleration_oracle_not_model_or_imagination_qualification',
        teacher_modified=False, final_test_used=False, validation_used=False,
        sample='training_only_up_to512_distinct_contact_substeps_per_trial',
        by_group={group: {name: summarize_errors([np.abs(value[mask])]) for name, value in values.items() if name != 'near_rim'}
                  for group, mask in groups.items()}, trials=rows,
        worst_samples=[{name: value[index].item() for name, value in values.items()} for index in worst],
        residual_normal_acceleration_correlation=float(np.corrcoef(values['base_normal_error_m_s'],
            values['racket_normal_acceleration_m_s2'])[0, 1]))
    digest = write_json_unique(args.output, report, budget)
    print(json.dumps(dict(path=str(args.output.absolute()), sha256=digest, overall=report['by_group']['all'])))


if __name__ == '__main__':
    main()
