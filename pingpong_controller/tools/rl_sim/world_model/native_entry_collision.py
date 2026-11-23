"""Host observation around the complete preserved native evaluation entry point."""

from collections import deque
import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile
from types import FunctionType, SimpleNamespace

import jax
import numpy as np
import mujoco

from . import evaluate_frozen
from .attribution_common import apex_replay_deltas
from .audit_full_prior_world import release_cpu_heap
from .contracts import authenticate, repository_root
from .paired_attribution import ball_other_contacts
from .paired_precision import load_actor
from .state_adapter import numeric_tree_digest


def write_compact_json(path, payload, budget):
    """Publish lossless compact diagnostic JSON within the existing byte cap."""
    path = Path(path).absolute()
    if not path.resolve().is_relative_to(budget.root.resolve()):
        raise ValueError('Diagnostic output must remain inside the budget root')
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    data = (json.dumps(payload, ensure_ascii=False, allow_nan=False,
                       separators=(',', ':')) + '\n').encode()
    budget.check(additional_bytes=2 * len(data) + 8192)
    descriptor, temporary = tempfile.mkstemp(prefix='.wmj-diagnostic-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        json.loads(Path(temporary).read_text())
        os.link(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return hashlib.sha256(data).hexdigest()
    finally:
        Path(temporary).unlink(missing_ok=True)


def observed_jit(observer):
    """Observe host arguments/results of policy_step without changing its JIT graph."""
    def compile_function(function, *args, **kwargs):
        compiled = jax.jit(function, *args, **kwargs)
        candidates = inspect.getclosurevars(function).nonlocals.values()
        policy = next((v for v in candidates if callable(v)
                       and getattr(v, '__name__', '') == 'policy_step'), None)
        if policy is None:
            return compiled

        def call(*values, **options):
            result = compiled(*values, **options)
            observer(policy, values, result)
            return result
        return call
    return compile_function


def audit(cfg, args, guard, report, plan):
    """Use original preparation, deterministic warm call and sampled evaluation."""
    if jax.config.x64_enabled:
        raise ValueError('Native evaluation requires original FP32 runtime')
    cases = json.loads(authenticate(repository_root(),
        cfg['frozen_evaluation_artifacts']['cases']).read_text())['stages']['S01']
    lane = next(i for i, c in enumerate(cases) if c['case_id'] == plan['collision']['case_id'])
    report.update(runs={}, case_id=plan['collision']['case_id'], original_population=128,
        entry_source='unchanged evaluate_frozen.audit code object with host-only JIT observer',
        original_entry_documents='captured in memory; compared with preserved original evaluation artifacts')
    # The preserved V324 loader permits one native stack per process.
    # Each actor therefore uses a separate worker, as in its original evaluation.
    for name in (args.entry_actor,):
        if name not in plan['collision']['actor_names']:
            raise ValueError('Unregistered original-entry actor')
        actor, _ = load_actor(cfg, plan['actors'][name])
        before = numeric_tree_digest(actor)
        captured = {}; contacts = []; tail = deque(maxlen=20); initial = []

        def observer(function, inputs, result):
            if not initial:
                initial.append(numeric_tree_digest(inputs[0]))
            if float(np.asarray(inputs[-1])) == 0.:
                return
            if bool(np.asarray(inputs[2].done)[lane]):
                return
            closure = inspect.getclosurevars(function).nonlocals
            env = closure['env']; adapter = closure['adapter']
            native, task, trace = result[0], result[2], jax.device_get(result[-1])
            q = np.asarray(native.data.qpos)
            tail.append(dict(elapsed_ms=int(task.elapsed_ms[lane]),
                ball_position_m=q[lane, env.ball_qadr:env.ball_qadr+3].tolist(),
                ball_velocity_m_s=np.asarray(native.data.qvel)[lane, env.ball_vadr:env.ball_vadr+3].tolist(),
                racket_position_m=np.asarray(native.data.site_xpos)[lane, env.racket_site_id].tolist(),
                racket_rotation=np.asarray(native.data.site_xmat)[lane, env.racket_site_id].reshape(3, 3).tolist(),
                measured_q_rad=q[lane, adapter.feedback_qadr].tolist(),
                q_ref_rad=np.asarray(native.arm_cmd_q)[lane].tolist(),
                accepted_action=np.asarray(native.prev_action)[lane].tolist()))
            if np.any(trace['failure_code'][:, lane] == 2):
                pairs = np.asarray(native.data.contact.geom)[lane]
                distances = np.asarray(native.data.contact.dist)[lane]
                active = ball_other_contacts(pairs, distances,
                    ball_id=env.ball_geom_id, racket_id=env.racket_contact_geom_id)
                for index in active:
                    geoms = [dict(id=int(i), geom=mujoco.mj_id2name(env.mj_model,
                        mujoco.mjtObj.mjOBJ_GEOM, int(i)), body=mujoco.mj_id2name(env.mj_model,
                        mujoco.mjtObj.mjOBJ_BODY, int(env.mj_model.geom_bodyid[i]))) for i in pairs[index]]
                    contacts.append(dict(case_id=cases[lane]['case_id'], elapsed_ms=int(task.elapsed_ms[lane]),
                        geoms=geoms, distance_m=float(distances[index])))
                if not active:
                    contacts.append(dict(case_id=cases[lane]['case_id'], elapsed_ms=int(task.elapsed_ms[lane]),
                        status='NO_ACTIVE_PAIR_RETAINED_AT_5MS_ENDPOINT'))

        def capture_document(path, document, budget):
            # The full native entry writes no new duplicate evaluation datasets.
            # Its in-memory documents are compared before compact evidence is saved.
            captured[path.name] = document
            return hashlib.sha256(json.dumps(document, sort_keys=True, allow_nan=False).encode()).hexdigest()

        namespace = dict(evaluate_frozen.audit.__globals__)
        namespace.update(jax=SimpleNamespace(**dict(jax.__dict__, jit=observed_jit(observer))),
                         write_json_unique=capture_document)
        entry = FunctionType(evaluate_frozen.audit.__code__, namespace,
            name=evaluate_frozen.audit.__name__, argdefs=evaluate_frozen.audit.__defaults__,
            closure=evaluate_frozen.audit.__closure__)
        entry.__kwdefaults__ = evaluate_frozen.audit.__kwdefaults__
        invocation = SimpleNamespace(output_dir=args.output_dir,
            initialization_dir=authenticate(repository_root(),
                cfg['frozen_evaluation_artifacts']['static_initialization']).parent,
            stage=1, seed=plan['collision']['seed'], smoke=False)
        native_report = {}
        entry(cfg, invocation, guard, native_report, actor_override=actor,
              actor_reference=plan['actors'][name])
        sampled = captured['sampled.json']
        old_ref = plan['collision']['references'][name.removeprefix('u30_')]
        old = json.loads(authenticate(repository_root(), old_ref).read_text())
        changed = [dict(current=a, previous=b) for a, b in zip(sampled['cases'], old['cases'])
            if any(a[k] != b[k] for k in ('case_id', 'hits', 'physical_failure', 'elapsed_s'))]
        deltas = apex_replay_deltas(sampled['events'], old['events'])
        report['runs'][name] = dict(actor_reference=plan['actors'][name], original_evaluation=old_ref,
            initial_native_sha256=initial[0], outcomes=sampled['cases'],
            replay_outcome_differences=changed, matched_apex_count=len(deltas),
            matched_apex_max_abs_delta_m=max(deltas, default=0.), collision_contacts=contacts,
            selected_case_events=[x for x in sampled['events'] if x['case_id'] == cases[lane]['case_id']],
            selected_case_last20_controls=list(tail), zero_force_pair=native_report['zero_force_pair'],
            execution=native_report['execution'], native_xml=native_report['native_xml'],
            parameters_unchanged=numeric_tree_digest(actor) == before,
            native_entry_complete=native_report['status'] == 'COMPLETE')
        if not report['runs'][name]['parameters_unchanged']:
            raise ValueError('Frozen Actor changed')
        print(f'[ENTRY-COLLISION] {name} changed={len(changed)} apex_delta={max(deltas,default=0.):.6g} contacts={contacts}', flush=True)
        del captured, entry, namespace, actor
        jax.clear_caches(); release_cpu_heap()
    report.update(status='PAIRED_ORIGINAL_ENTRY_COLLISION_ATTRIBUTION_COMPLETE', parameters_unchanged=True)
