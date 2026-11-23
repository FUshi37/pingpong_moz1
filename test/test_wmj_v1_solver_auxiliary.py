"""Independent acceleration labels and prediction cannot leak solver inputs."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'pingpong_controller/tools/rl_sim'))
from world_model.neural import init_mlp
from world_model.robot_support_model import support_features
from world_model.solver_auxiliary import (
    execution_features, normal_projection_coefficients, normal_velocity_error,
    predict, scale_from_statistics, supervised_loss,
)
from world_model.teacher_trace import point_twist


def statistics(size):
    return dict(source='train', count=10, mean=[0.]*size, variance=[1.]*size)


def test_input_is_existing_execution_state_and_ignores_future_labels():
    q = jnp.arange(26, dtype=jnp.float32).reshape(2, 13)*.01
    dq = q*.2
    commands = dict(q=q[:, :7]+.1, dq=q[:, :7]+.2, ddq=q[:, :7]+.3)
    trace = dict(q_before=q[:, :7], support_q_before=q[:, 7:],
        dq_before=dq[:, :7], support_dq_before=dq[:, 7:],
        **{f'delayed_{k}_command': v for k, v in commands.items()})
    expected = support_features(q, dq, commands)
    np.testing.assert_array_equal(execution_features(trace), expected)
    trace.update(q_after=q*99, dq_after=dq*99, solver_qacc=q*100,
                 ball_qpos_after=q*50, contact_normal_force_n=q)
    np.testing.assert_array_equal(execution_features(trace), expected)


def test_six_terms_project_to_contact_jacobian_including_slide_and_hinge():
    origin = jnp.array([[.4, -.2, .7]])
    contact = jnp.array([[.43, -.18, .69]])
    axes = jnp.array([[[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]]])
    anchors = jnp.array([[[0., 0., 0.], [.1, -.2, .3], [.2, .1, .6]]])
    acceleration = jnp.array([[2., 30., -12.]])
    hinge = jnp.array([False, True, True])
    linear, angular = point_twist(origin, axes, anchors, acceleration, hinge)
    direct, _ = point_twist(contact, axes, anchors, acceleration, hinge)
    normal = jnp.array([[.2, .3, .9]])
    coefficients = normal_projection_coefficients(normal, contact-origin)
    terms = jnp.concatenate((linear, angular), axis=-1)
    np.testing.assert_allclose(jnp.sum(terms*coefficients, axis=-1),
                               -jnp.sum(direct*normal, axis=-1), atol=1e-6)


def test_normalization_rejects_validation_and_future_input_width():
    scale = scale_from_statistics(statistics(47), statistics(6))
    with pytest.raises(ValueError):
        scale_from_statistics({**statistics(47), 'source': 'validation'}, statistics(6))
    with pytest.raises(ValueError):
        scale_from_statistics(statistics(47), {**statistics(6), 'variance': [-1.]*6})
    params = init_mlp(jax.random.PRNGKey(4), (47, 128, 128, 6), zero_output=True)
    with pytest.raises(ValueError):
        predict(params, scale, jnp.zeros((2, 48)))
    features = jnp.ones((4, 47))
    loss, gradient = jax.value_and_grad(supervised_loss)(params, scale, features, jnp.ones((4, 6)))
    assert float(loss) == 1.
    assert np.linalg.norm(gradient[-1]['b']) > 0.


def test_scalar_equation_oracle_is_only_an_offline_diagnostic():
    rows = dict(normal_coefficients=np.array([[0., 0., -1., 0., 0., 0.]]),
        aref=np.array([100.]), gravity=np.array([-9.81]),
        regularizer=np.array([1000.]), force_n=np.array([(109.81+20.)/1250.]))
    oracle = np.array([[0., 0., 20., 0., 0., 0.]])
    np.testing.assert_allclose(normal_velocity_error(oracle, rows), 0., atol=1e-10)
    assert abs(normal_velocity_error(np.zeros_like(oracle), rows)[0]) > .001


def test_native_static_contact_addresses_remain_jittable():
    from types import SimpleNamespace
    from world_model.collect_solver_auxiliary import solver_labels
    env = SimpleNamespace(ball_geom_id=1, racket_contact_geom_id=0, ball_vadr=3,
        _wmj_support_trace=dict(ancestor_ids=np.arange(3), ancestor_dof=np.arange(3),
                                hinge_mask=jnp.zeros(3, bool)))
    contact = SimpleNamespace(geom=jnp.array([[[0, 1]]]), dist=jnp.array([[-.001]]),
        efc_address=np.array([0], np.int32), pos=jnp.zeros((1, 1, 3)))

    def observed(acceleration):
        current = SimpleNamespace(contact=contact, geom_xpos=jnp.zeros((1, 1, 3)),
            xaxis=jnp.eye(3)[None], xanchor=jnp.zeros((1, 3, 3)), qacc=acceleration,
            efc_J=jnp.array([[[0., 0., -1., 0., 0., 1.]]]),
            efc_aref=jnp.array([[100.]]), efc_D=jnp.array([[.001]]),
            efc_force=jnp.array([[.1]]))
        return solver_labels(env, current)

    result = jax.jit(observed)(jnp.array([[1., 2., 3., 0., 0., 4.]]))
    np.testing.assert_allclose(result['target'], [[1., 2., 3., 0., 0., 0.]])
    np.testing.assert_allclose(result['closure_error'], 0., atol=1e-7)


def test_independent_fit_serializes_complete_state_and_never_loads_test_rows(tmp_path):
    import json
    from types import SimpleNamespace
    from world_model.artifacts import write_array_archive_unique, read_array_archive
    from world_model.contracts import config_hash, load_config
    from world_model.fit_solver_auxiliary import fit, load_labels
    from world_model.neural import init_adam
    from world_model.resources import StorageBudget
    from world_model.state_adapter import restore_numeric_state
    cfg = load_config()
    labels, output = tmp_path/'labels', tmp_path/'fit'
    labels.mkdir()
    output.mkdir()
    budget = StorageBudget(tmp_path, 100*2**20, 0)
    guard = SimpleNamespace(budget=budget, check=lambda: None)
    rng = np.random.default_rng(13)
    features = rng.normal(size=(128, 47)).astype(np.float32)
    targets = features[:, :6]*10.
    coefficients = np.zeros((128, 6), np.float32)
    coefficients[:, 2] = -1.
    rows = dict(features=features, target=targets, normal_coefficients=coefficients,
        robot_normal_term=-targets[:, 2], closure_error=np.zeros(128, np.float32),
        aref=np.full(128, 100., np.float32),
        gravity=np.full(128, -9.81, np.float32), regularizer=np.full(128, 1000., np.float32),
        force_n=(109.81+targets[:, 2])/1250., overlap=np.arange(128) % 2 == 0,
        split=np.repeat([0, 1], 64).astype(np.int8), region_id=np.zeros(128, np.int32),
        family_id=np.repeat(np.arange(8), 16).astype(np.int32))
    archive = write_array_archive_unique(labels/'data.npz', rows, budget)
    groups = [dict(root_seed_family=f'root{i}', split='train' if i < 4 else 'validation') for i in range(8)]
    manifest = dict(status='LABELS_COMPLETE_INDEPENDENT_DIAGNOSTIC_ONLY',
        physics_sha256=config_hash(cfg['physics']), control_sha256=config_hash(cfg['control']),
        native_row_closure_m_s2={'maximum': 0.}, source_dataset_sha256='synthetic_unit_test',
        label_definition='synthetic pipeline fixture', records=[dict(archive_file='data.npz', archive=archive, groups=groups)])
    (labels/'result.json').write_text(json.dumps(manifest))
    report = dict(source_sha256={'fixture': 'synthetic'})
    fit(cfg, SimpleNamespace(labels=labels, output_dir=output, seed=3, updates=2), guard, report)
    metadata = json.loads((output/'auxiliary_u000002.json').read_text())
    scale = scale_from_statistics(**metadata['normalization'])
    params = init_mlp(jax.random.PRNGKey(3), (47, 128, 128, 6), zero_output=True)
    restored = restore_numeric_state(dict(parameters=params, optimizer=init_adam(params), scale=scale),
        read_array_archive(output/metadata['archive_file'], metadata['archive']), metadata['state'],
        context_sha256=metadata['model_signature'])
    assert int(restored['optimizer'].step) == 2
    assert report['actor_updates'] == report['critic_updates'] == 0
    assert report['integration_in_world'] is False
    assert report['split_counts'] == {'train': 64, 'validation': 64}
    predictions = read_array_archive(output/'validation_predictions.npz', report['prediction_archive'])
    np.testing.assert_allclose(predict(restored['parameters'], restored['scale'], features[64:]),
                               predictions['prediction'], atol=1e-5)
    manifest['records'][0]['groups'].append(dict(root_seed_family='root0', split='validation'))
    (labels/'result.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Family leakage'):
        load_labels(labels, cfg, guard)
