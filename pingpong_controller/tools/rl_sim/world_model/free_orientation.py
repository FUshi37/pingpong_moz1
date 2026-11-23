"""Private native-only ablation of racket roll/pitch constraints, default off."""

import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import FunctionType, MethodType, SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from cartesian_pose_action_projection_jax import project_cartesian_pose_qvel_jax

from . import evaluate_frozen
from .contracts import authenticate, file_hash, repository_root
from .imagined_env import physical_bounds
from .native_entry_collision import observed_jit, write_compact_json
from .native_stack import build_native_environment
from .paired_attribution import ball_other_contacts
from .state_adapter import numeric_tree_digest


def position_only_projection(**kwargs):
    """Keep V324's 24-row EXEC schedule; four angular rows become finite no-ops."""
    # The enclosing native make_collect(mode='project') supplies its original
    # EXEC unroll transformation, just as it does for the bounded baseline.
    return project_cartesian_pose_qvel_jax(**dict(kwargs,
        relative_rpy_rad=jnp.zeros_like(kwargs['relative_rpy_rad']),
        relative_rpy_jacobian=jnp.zeros_like(kwargs['relative_rpy_jacobian']),
        roll_pitch_limit_rad=jnp.zeros_like(kwargs['roll_pitch_limit_rad'])))


def remove_reference_attitude_backstops(source: str) -> str:
    """Remove two nonlinear angle tests without touching position or joint tests."""
    replacements = dict(current_orientation_violation='current_position_violation',
                        candidate_orientation_violation='candidate_position_violation')
    counts = {name: 0 for name in replacements}

    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name in replacements:
                    counts[name] += 1
                    node.value = ast.parse('jnp.zeros_like(' + replacements[name] + ')', mode='eval').body
            return self.generic_visit(node)

    tree = Rewrite().visit(ast.parse(textwrap.dedent(source)))
    if counts != {name: 1 for name in replacements}:
        raise ValueError('Expected exactly two native reference-attitude backstops')
    return ast.unparse(ast.fix_missing_locations(tree))


def install_position_only_reference(environment, stack) -> dict:
    """Bind one instance method, preserving shared files, class and RMP wrappers."""
    original = environment._project_cartesian_pose_reference.__func__
    if file_hash(Path(inspect.getsourcefile(original))) != stack.spec.native_env_sha256:
        raise ValueError('Reference projector is not the authenticated V324 source')
    source = remove_reference_attitude_backstops(inspect.getsource(original))
    namespace = dict(original.__globals__)
    namespace.update(project_cartesian_pose_qvel_jax=position_only_projection,
        # No Euler-rate solve is needed when angular halfspaces are absent.
        relative_xyz_euler_rate_jacobian_jax=lambda angles, reference, angular: jnp.zeros_like(angular))
    exec(compile(source, '<WMJ-V1-free-racket-attitude>', 'exec'), namespace)
    environment._project_cartesian_pose_reference = MethodType(namespace[original.__name__], environment)
    return dict(original_source_sha256=stack.spec.native_env_sha256,
        transformed_method_sha256=hashlib.sha256(source.encode()).hexdigest(),
        orientation_linear_rows='four zero rows; native 24-row schedule unchanged',
        nonlinear_orientation_checks=False, position_checks=True, joint_checks=True,
        finite_large_angle_substitute=False, euler_rate_solve=False,
        native_angle_metrics='retained as legacy diagnostics only; not WMJ termination')


def free_physical_bounds(physics, pose, control, episode):
    """Retain actual XYZ/joint bounds and finite-pose admission, omit angle bound."""
    _, tests = physical_bounds(physics, pose, control, episode)
    tests = dict(tests, orientation=jnp.all(jnp.isfinite(pose), axis=-1))
    return jnp.all(jnp.stack(list(tests.values())), axis=0), tests


def phase_plan(cfg: dict) -> dict:
    trial = cfg['orientation_limit_trial']
    plan = json.loads(authenticate(repository_root(), trial['plan']).read_text())
    if (not trial['authorized'] or not plan['authorized'] or trial['default_enabled']
            or plan['modes'] != ['bounded_2deg', 'free_orientation']
            or plan['candidate']['roll_pitch_constraint_enabled'] is not False
            or plan['candidate']['roll_pitch_limit_deg'] is not None
            or plan['total_cap_bytes'] != cfg['resources']['max_new_bytes']
            or cfg['domain']['roll_pitch_limit_deg'] != 2.):
        raise ValueError('Orientation ablation differs from the authorized scope')
    return plan


def audit(cfg, args, guard, report, *, actor, actor_reference):
    """Use the preserved full evaluation entry with explicit scoped dependencies."""
    free = args.mode == 'free_orientation'
    cases = json.loads(authenticate(repository_root(),
        cfg['frozen_evaluation_artifacts']['cases']).read_text())['stages']['S01']
    observations = {}
    initial_hashes = {}
    cold_call = True

    def build(*values, **kwargs):
        bundle = build_native_environment(*values, **kwargs)
        report['reference_projection'] = (install_position_only_reference(bundle.environment, bundle.stack)
            if free else dict(unchanged=True, roll_pitch_limit_deg=2.))
        return bundle

    def observe(function, inputs, result):
        nonlocal cold_call
        if cold_call:
            report['initial_native_sha256'] = numeric_tree_digest(inputs[0])
            report['initial_history_task_rng_sha256'] = numeric_tree_digest(inputs[1:5])
            cold_call = False
            return
        mode = 'deterministic' if float(np.asarray(inputs[-1])) == 0. else 'sampled'
        if mode not in initial_hashes:
            initial_hashes[mode] = dict(native=numeric_tree_digest(inputs[0]),
                history_task_rng=numeric_tree_digest(inputs[1:5]))
        closure = inspect.getclosurevars(function).nonlocals
        env = closure['env']; native, task = result[0], result[2]
        active = ~np.asarray(inputs[2].done)
        rotation = np.asarray(native.data.site_xmat)[:, env.racket_site_id].reshape(-1, 3, 3)
        reference = np.asarray(env.fixed_contact_racket_reference_rotation)
        relative = np.einsum('ji,njk->nik', reference, rotation)
        angles = np.stack((np.arctan2(-relative[:, 1, 2], relative[:, 2, 2]),
            np.arcsin(np.clip(relative[:, 0, 2], -1., 1.))), axis=-1)
        degrees = np.rad2deg(np.abs(angles))
        normal_tilt = np.rad2deg(np.arccos(np.clip(relative[:, 2, 2], -1., 1.)))
        elapsed_ms = np.asarray(task.elapsed_ms)
        failure_codes = np.asarray(task.failure_code)
        rows = observations.setdefault(mode, [dict(case_id=c['case_id'], active_controls=0,
            max_abs_roll_pitch_deg=[0., 0.], over_2deg_controls=0, first_over_2deg_ms=None,
            max_normal_tilt_deg=0., nonfinite_pose_controls=0, collision=None) for c in cases])
        for lane in np.flatnonzero(active):
            row = rows[lane]; row['active_controls'] += 1
            if np.isfinite(degrees[lane]).all():
                row['max_abs_roll_pitch_deg'] = np.maximum(row['max_abs_roll_pitch_deg'], degrees[lane]).tolist()
                row['max_normal_tilt_deg'] = max(row['max_normal_tilt_deg'], float(normal_tilt[lane]))
                if degrees[lane].max() > 2.:
                    row['over_2deg_controls'] += 1
                    if row['first_over_2deg_ms'] is None:
                        row['first_over_2deg_ms'] = int(elapsed_ms[lane])
            else:
                row['nonfinite_pose_controls'] += 1
            if int(failure_codes[lane]) == 2 and row['collision'] is None:
                pairs = np.asarray(native.data.contact.geom)[lane]
                distances = np.asarray(native.data.contact.dist)[lane]
                indices = ball_other_contacts(pairs, distances,
                    ball_id=env.ball_geom_id, racket_id=env.racket_contact_geom_id)
                row['collision'] = dict(elapsed_ms=int(elapsed_ms[lane]),
                    scope='contact data retained at first-failure endpoint; no retrospective replay',
                    pairs=[dict(geoms=[dict(id=int(i), geom=mujoco.mj_id2name(env.mj_model,
                        mujoco.mjtObj.mjOBJ_GEOM, int(i)), body=mujoco.mj_id2name(env.mj_model,
                        mujoco.mjtObj.mjOBJ_BODY, int(env.mj_model.geom_bodyid[i]))) for i in pairs[index]],
                        distance_m=float(distances[index])) for index in indices])

    endpoint = evaluate_frozen.endpoint_code
    if free:
        namespace = dict(endpoint.__globals__, physical_bounds=free_physical_bounds)
        endpoint = FunctionType(endpoint.__code__, namespace, endpoint.__name__,
                                endpoint.__defaults__, endpoint.__closure__)
    namespace = dict(evaluate_frozen.audit.__globals__, build_native_environment=build,
        endpoint_code=endpoint, write_json_unique=write_compact_json,
        jax=SimpleNamespace(**dict(jax.__dict__, jit=observed_jit(observe))))
    entry = FunctionType(evaluate_frozen.audit.__code__, namespace, evaluate_frozen.audit.__name__,
                         evaluate_frozen.audit.__defaults__, evaluate_frozen.audit.__closure__)
    entry.__kwdefaults__ = evaluate_frozen.audit.__kwdefaults__
    entry(cfg, args, guard, report, actor_override=actor, actor_reference=actor_reference)
    report['initial_pairing'] = initial_hashes
    report['attitude_observations'] = dict(path=str(args.output_dir/'attitude_observations.json'),
        sha256=write_compact_json(args.output_dir/'attitude_observations.json',
            dict(modes=observations, angle_units='degrees', sampling='200Hz endpoint; active intervals only'), guard.budget))
