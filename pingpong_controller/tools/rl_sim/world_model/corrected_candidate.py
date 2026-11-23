"""Frozen corrected-control captures and explicitly paired action pulses."""
from __future__ import annotations
import ast
import inspect

import numpy as np

from .projection_consistency import capture_source as projection_capture_source, NATIVE_METRICS


def capture_source(source: str) -> str:
    """Retain the real full native/reset history before the existing collection."""
    tree=ast.parse(projection_capture_source(source));function=tree.body[0]
    body=[];found=[]
    for node in function.body:
        body.append(node);text=ast.unparse(node)
        if text.startswith('native = restore_native_sources('):
            body.extend(ast.parse('native_seed = native').body);found.append('native')
        if text=='history, _ = adapter.initialize(native)':
            body.extend(ast.parse('history_seed = history').body);found.append('history')
    if found!=['native','history']:raise ValueError('Native reset capture anchors changed')
    function.body=body
    function.body[-1].value.keywords.extend(ast.keyword(arg=k,value=ast.Name(id=v,ctx=ast.Load()))
        for k,v in [('native_seed','native_seed'),('history_seed','history_seed'),('adapter','adapter'),('stack','bundle'),('native_unobserved_step','original')])
    return ast.unparse(ast.fix_missing_locations(tree))


def make_capture():
    from .audit_free_imagination import audit
    from .audit_projection_precision_integrated import install_candidate
    from .projection_precision import make_runtime
    from .error_attribution import instrument_with_solver
    namespace=dict(audit.__globals__,install_position_only_reference=install_candidate,
        make_runtime=make_runtime,instrument_environment=instrument_with_solver,NATIVE_METRICS=NATIVE_METRICS)
    exec(compile(capture_source(inspect.getsource(audit)),'<WMJ-corrected-candidate-capture>','exec'),namespace)
    return namespace[audit.__name__]


def pulse_assignments(count: int, seed: int) -> list[dict]:
    """Balance joints, sizes and10/20ms durations independently before outcomes."""
    rng=np.random.default_rng(seed)
    joints=rng.permutation(np.arange(count)%7)
    epsilon=rng.permutation(np.asarray([.005,.01,.02])[np.arange(count)%3])
    duration=rng.permutation(np.asarray([2,4])[np.arange(count)%2])
    return [dict(joint=int(j),epsilon=float(e),duration_controls=int(t))
            for j,e,t in zip(joints,epsilon,duration)]


def branch_actions(base, lanes, assignments):
    """Four groups of identical source lanes: base, plus, minus, duplicate."""
    lanes=np.asarray(lanes,int)
    if base.shape!=(128,32,7) or lanes.shape!=(8,) or len(set(lanes.tolist()))!=8:
        raise ValueError('Expected128controls,32sources and8distinct selected lanes')
    mapping=np.tile(lanes,4);actions=base[:,mapping].copy();cases=[]
    for i,lane in enumerate(lanes):
        row=assignments[lane];t,j,e=row['duration_controls'],row['joint'],row['epsilon']
        actions[:t,8+i,j]+=e;actions[:t,16+i,j]-=e
        cases.append(dict(row,lane=int(lane),pulse_ms=5*t,effective_clipped_central_pulse_norm=float(np.linalg.norm(
            np.clip(actions[:,8+i],-1,1)-np.clip(actions[:,16+i],-1,1)))))
    return mapping,actions,cases


def resolved_effects(native,world,repeat,cases):
    """Keep every case; resolve direction only above observed repeat drift."""
    result={}
    if any(len(v)!=len(cases) for v in (*native.values(),*world.values(),repeat)):
        raise ValueError('Counterfactual population mismatch')
    for name,floor in [('outgoing_velocity_m_s',.001),('analytic_apex_m',.0001),('actual_apex_m',.0001)]:
        rows=[]
        for i,case in enumerate(cases):
            all_rows=[d[k][i] for d in (native,world) for k in ('base','plus','minus')]+[repeat[i]]
            paired=all(r['release_present'] and r.get(name) is not None for r in all_rows)
            paired=paired and len({r['impact_id'] for r in all_rows})==1
            row=dict(case_index=i,paired=bool(paired),effective_pulse=case['effective_clipped_central_pulse_norm']>0.)
            if paired:
                observed=np.asarray(native['plus'][i][name])-native['minus'][i][name]
                predicted=np.asarray(world['plus'][i][name])-world['minus'][i][name]
                drift=np.asarray(repeat[i][name])-native['base'][i][name]
                magnitude=float(np.linalg.norm(observed));noise=float(np.linalg.norm(drift));dot=float(np.dot(observed,predicted))
                row.update(native_effect_norm=magnitude,repeat_difference_norm=noise,predicted_effect_norm=float(np.linalg.norm(predicted)),
                    resolved=bool(row['effective_pulse'] and magnitude>max(floor,3*noise)),direction_agrees=dot>0.,
                    direction_margin_exceeds_repeat=dot>3*noise*float(np.linalg.norm(predicted)))
            rows.append(row)
        paired=[r for r in rows if r['paired']];resolved=[r for r in paired if r['resolved']]
        result[name]=dict(cases=len(rows),paired_cases=len(paired),missing_cases=len(rows)-len(paired),
            resolved_cases=len(resolved),unresolved_cases=len(rows)-len(resolved),floor=floor,repeat_multiplier=3.,
            resolved_direction_agreement_fraction=float(np.mean([r['direction_agrees'] for r in resolved])) if resolved else None,
            resolved_direction_margin_fraction=float(np.mean([r['direction_margin_exceeds_repeat'] for r in resolved])) if resolved else None,
            rows=rows,qualification=False)
    return result
