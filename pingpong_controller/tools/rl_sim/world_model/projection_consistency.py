"""Private frozen projection probes and an opt-in quaternion FK candidate."""
import ast
import inspect

from .error_attribution import capture_source as attribution_source, instrument_with_solver
from .free_imagination import make_projector
from .control_kinematics import forward_control_kinematics


NATIVE_METRICS = {
    'fallback': 'cartesian_pose_projection_fallback_to_hold',
    'nonlinear_backstop': 'cartesian_pose_projection_nonlinear_backstop',
    'current_safe': 'cartesian_pose_projection_current_safe',
    'position_violation': 'cartesian_pose_qref_position_violation_m',
    'linear_violation': 'cartesian_pose_projection_linear_violation',
}


def capture_source(source: str) -> str:
    """Keep the original collection and expose its existing branch metrics."""
    tree = ast.parse(attribution_source(source))
    function = tree.body[0]
    collect = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == 'collect')
    result = collect.body[-1]
    if not isinstance(result, ast.Return) or ast.unparse(result.value.elts[-1]) != 'bounds':
        raise ValueError('Native collector boundary changed')
    result.value.elts[-1] = ast.parse(
        'dict(bounds, **{name: metrics[key] for name, key in NATIVE_METRICS.items()})', mode='eval').body
    final = function.body[-1].value
    final.keywords.extend([ast.keyword(arg='environment', value=ast.Name(id='env', ctx=ast.Load())),
        ast.keyword(arg='native_tail', value=ast.Name(id='native', ctx=ast.Load())),
        ast.keyword(arg='geometry', value=ast.parse("reader.report['geometry']", mode='eval').body)])
    return ast.unparse(ast.fix_missing_locations(tree))


def make_capture():
    from .audit_free_imagination import audit
    namespace = dict(audit.__globals__, instrument_environment=instrument_with_solver, NATIVE_METRICS=NATIVE_METRICS)
    exec(compile(capture_source(inspect.getsource(audit)), '<WMJ-projection-consistency-capture>', 'exec'), namespace)
    return namespace[audit.__name__]


def make_quaternion_projector(quaternion_chain):
    """Change only the private control FK; preserve halfspaces and hold rules."""
    original = make_projector()

    def project(chain, contract, measured_q, reference_q, nominal_qvel):
        if chain.joint_ids != quaternion_chain.joint_ids:
            raise ValueError('Quaternion FK joint order differs from physical chain')
        return original(quaternion_chain, contract, measured_q, reference_q,
                        nominal_qvel, kinematics=forward_control_kinematics)
    return project
