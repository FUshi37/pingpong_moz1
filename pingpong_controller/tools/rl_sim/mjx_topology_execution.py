"""Opt-in, trace-scoped MJX execution candidates with unchanged arithmetic."""

import ast
from functools import lru_cache
import hashlib
import inspect
from pathlib import Path
import textwrap
from collections.abc import Callable
from types import ModuleType
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from mjx_line_search_schedule import make_chunked_loop

SUPPORT_SHA256 = '103b6ac154ab716df46057b03800851c63971a47938a8b97923824a25b668c59'
SOLVER_SHA256 = 'ad9c611bb8fe4116b0bba10f66b3473a958fe770fa82f9080a0f66a206d299ab'
PROJECTION_SHA256 = '0f43f1f4d05a410f3fd066d7776980bb075a97f74566f00249aaadbea553929e'


@lru_cache(maxsize=16)
def _mask_from_parents(parents: tuple[int, ...]) -> np.ndarray:
    count = len(parents)
    mask = np.zeros((count, count), dtype=np.bool_)
    for i in range(count):
        j = i
        visited = set()
        while j != -1:
            if not 0 <= j < count or j in visited:
                raise ValueError('Invalid or cyclic dof_parentid topology')
            visited.add(j)
            mask[i, j] = True
            j = parents[j]
    mask.flags.writeable = False
    return mask


def topology_mask(model: Any) -> np.ndarray:
    """Materialize the same bool[nv,nv] topology without staged scatter."""
    parents = np.asarray(model.dof_parentid)
    if parents.shape != (model.nv,) or not np.issubdtype(parents.dtype, np.integer):
        raise ValueError('Expected static integer dof_parentid[nv]')
    return _mask_from_parents(tuple(int(x) for x in parents))


@lru_cache(maxsize=64)
def _certify_layout(
    addresses: tuple[int, ...], dims: tuple[int, ...], nefc: int, nefl: int,
) -> None:
    occupied = set()
    for address, dim in zip(addresses, dims):
        if dim not in (1, 3, 4, 6):
            raise ValueError('Unsupported constraint dimension')
        if address < nefl or address + dim > nefc:
            raise ValueError('Contact constraint address outside its region')
        region = set(range(address, address + dim))
        if occupied & region:
            raise ValueError('Contact constraint regions overlap')
        occupied.update(region)


def certify_contact_layout(data: Any) -> None:
    """Certify static disjoint regions before asserting unique GPU indices."""
    contact = data._impl.contact
    addresses = np.asarray(contact.efc_address)
    dims = np.asarray(contact.dim)
    if addresses.ndim != 1 or addresses.shape != dims.shape:
        raise ValueError('Static contact address/dimension layout required')
    if addresses.size and (not np.issubdtype(addresses.dtype, np.integer)
                           or not np.issubdtype(dims.dtype, np.integer)):
        raise ValueError('Static contact addresses/dimensions must be integers')
    _certify_layout(tuple(int(x) for x in addresses), tuple(int(x) for x in dims),
                    int(data._impl.nefc),
                    int(data._impl.ne + data._impl.nf + data._impl.nl))


def _make_mass(original: Callable, support: ModuleType) -> Callable:
    def make_m(model: Any, a: jax.Array, b: jax.Array,
               d: jax.Array | None = None) -> jax.Array:
        if support.is_sparse(model):
            return original(model, a, b, d)
        # Preserve native dot, diagonal addition, mask multiply and triangle
        # order exactly; only replace the topology-only boolean construction.
        qm = a @ b.T
        if d is not None:
            qm += jnp.diag(d)
        qm = qm * jnp.asarray(topology_mask(model))
        qm = qm + jnp.tril(qm, -1).T
        return qm
    return make_m


class _Rewrite(ast.NodeTransformer):
    """Change only certified scatter hints or a fixed loop's unroll factor."""

    def __init__(self, projection: bool = False) -> None:
        self.projection = projection
        self.changes = 0

    def visit_Call(self, node: ast.Call) -> ast.AST:
        self.generic_visit(node)
        if (self.projection and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'fori_loop'):
            node.keywords.append(ast.keyword(arg='unroll', value=ast.Constant(24)))
            self.changes += 1
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr in ('set', 'add')
                and isinstance(node.func.value, ast.Subscript)):
            return node
        reference = node.func.value
        if not (isinstance(reference.value, ast.Attribute)
                and reference.value.attr == 'at'):
            return node
        index = ast.unparse(reference.slice)
        if index in ('jp.array(adr_i)', 'efc_address', 'efc_adr', 'dim1'):
            node.keywords.append(ast.keyword(arg='unique_indices',value=ast.Constant(True)))
            self.changes += 1
        return node


def _clone(function: Callable, rewrite: _Rewrite, expected_changes: int,
           certify: bool = False) -> Callable:
    source, source_line = inspect.getsourcelines(function)
    tree = ast.parse(textwrap.dedent(''.join(source)))
    node = tree.body[0]
    node.decorator_list = []
    tree = rewrite.visit(tree)
    if rewrite.changes != expected_changes:
        raise RuntimeError('Native function pattern differs from the reviewed source')
    if certify:
        index = 1 if isinstance(node.body[0], ast.Expr) and isinstance(
            node.body[0].value, ast.Constant) else 0
        node.body.insert(index, ast.Expr(value=ast.Call(
            func=ast.Name(id='_certify_contact_layout',ctx=ast.Load()),
            args=[ast.Name(id='d',ctx=ast.Load())],keywords=[])))
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, source_line - 1)
    namespace = dict(function.__globals__)
    namespace.update(_certify_contact_layout=certify_contact_layout)
    exec(compile(tree, function.__code__.co_filename, 'exec'), namespace)
    return namespace[node.name]


def make_collect(collect: Callable, mode: str = 'mask') -> tuple[Callable, dict]:
    """Apply a private execution candidate only during collector tracing.

    All native functions are restored on return and on failure. The initial
    reset and evaluation keep their existing implementations.
    """
    import mujoco
    from mujoco.mjx._src import solver, support

    if mode not in ('mask', 'unique', 'project'):
        raise ValueError('Unknown explicit execution candidate')
    if jax.__version__ != '0.10.1' or mujoco.__version__ != '3.8.1':
        raise RuntimeError('Unvalidated JAX/MuJoCo versions')
    for module, expected in ((solver, SOLVER_SHA256), (support, SUPPORT_SHA256)):
        if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Unvalidated native MJX source')
    for module, name in ((support, 'make_m'), (solver, '_while_loop_scan'),
                         (solver, '_update_constraint'), (solver, '_linesearch')):
        function = getattr(module, name)
        if (getattr(function, '_pingpong_execution_version', None) is not None
                or function.__module__ != module.__name__ or function.__name__ != name
                or function.__code__.co_filename != module.__file__
                or function.__code__.co_firstlineno < 30):
            raise RuntimeError('Execution dependency already patched')
    replacements = [(support, 'make_m', _make_mass(support.make_m, support)),
                    (solver, '_while_loop_scan', make_chunked_loop(5))]
    if mode in ('unique', 'project'):
        replacements.append((solver, '_update_constraint', _clone(
            solver._update_constraint, _Rewrite(), 3, certify=True)))
        replacements.append((solver._LSPoint, 'create', classmethod(_clone(
            solver._LSPoint.create.__func__, _Rewrite(), 1, certify=True))))
    if mode == 'project':
        import cartesian_pose_action_projection_jax as projection
        if hashlib.sha256(Path(projection.__file__).read_bytes()).hexdigest() != PROJECTION_SHA256:
            raise RuntimeError('Unvalidated projection source')
        replacements.append((projection, '_project_halfspaces_dykstra_jax', _clone(
            projection._project_halfspaces_dykstra_jax,
            _Rewrite(projection=True), 1)))
    for _, _, candidate in replacements:
        function = candidate.__func__ if isinstance(candidate, classmethod) else candidate
        function._pingpong_execution_version = 'mjx_topology_execution_v2'
    originals = [(target, name, target.__dict__[name])
                 for target, name, _ in replacements]

    def scoped(*args, **kwargs):
        for target, name, original in originals:
            if target.__dict__[name] is not original:
                raise RuntimeError('Execution dependency changed outside collector scope')
        try:
            for target, name, candidate in replacements:
                setattr(target, name, candidate)
            return collect(*args, **kwargs)
        finally:
            for target, name, original in reversed(originals):
                setattr(target, name, original)

    version = ('mjx_topology_execution_v2' if mode == 'project'
               else 'mjx_topology_execution_'+mode+'_v1')
    return scoped, dict(version=version,mode=mode,
        default_enabled=False,scope='training collect tracing only',
        solver_sha256=SOLVER_SHA256,support_sha256=SUPPORT_SHA256,
        jax_version=jax.__version__,mujoco_version=mujoco.__version__,
        solver_iterations=100,line_search_iterations=50,
        projection_unroll=24 if mode == 'project' else 1,
        repeated_line_search_additions='native unchanged',
        differentiation='primal simulator execution only')
