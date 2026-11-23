"""Bounded expanded-motion sources, labels and auxiliary checkpoint lineage."""
from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
from types import MethodType

import jax
import numpy as np

from . import teacher_trace
from .artifacts import read_array_archive
from .collect_solver_auxiliary import solver_labels
from .contracts import authenticate,file_hash,repository_root
from .error_attribution import capture_source
from .neural import init_adam,init_mlp
from .solver_auxiliary import WIDTHS,execution_features,scale_from_statistics
from .state_adapter import restore_numeric_state
from .velocity_reward import clone_globals


def select_sources(groups,initial_valid,*,excluded,seed,counts=(48,16)):
    """Stratify by region before outcomes, retaining each original family split."""
    rng=np.random.default_rng(seed);selected=[];partitions={}
    for group in groups:
        name=group['root_seed_family']
        if name in partitions and partitions[name]!=group['split']:
            raise ValueError('Common family crosses partitions')
        partitions[name]=group['split']
    for split,count in zip(('train','validation'),counts):
        families={}
        for lane,g in enumerate(groups):
            if (g['split']==split and initial_valid[lane] and g['initial_geometry_admitted']
                    and g['root_seed_family'] not in excluded):
                families.setdefault(g['root_seed_family'],[]).append(dict(g,lane=lane))
        candidates=[rows[int(rng.integers(len(rows)))] for rows in families.values()]
        if len(candidates)<count:raise ValueError('Insufficient independent admitted '+split+' sources')
        buckets={}
        for row in candidates:buckets.setdefault(row['region_id'],[]).append(row)
        order=rng.permutation(sorted(buckets)).tolist()
        for rows in buckets.values():rng.shuffle(rows)
        result=[]
        while len(result)<count:
            for region in order:
                if buckets[region]:result.append(buckets[region].pop())
                if len(result)==count:break
        selected.extend(result)
    return selected


def retained_label_mask(overlap,prefix,rng,*,background_fraction=.02):
    """Keep all contact +/-25ms labels and a fixed random background sample."""
    overlap=np.asarray(overlap,bool);near=overlap.copy()
    for offset in range(1,26):
        near[offset:] |= overlap[:-offset]
        near[:-offset] |= overlap[offset:]
    return np.asarray(prefix,bool) & (near | (rng.random(overlap.shape)<background_fraction))


def excluded_family_mask(rows,family_names,excluded):
    """Exclude complete diagnosed root families, including their older samples."""
    if not set(np.unique(rows['family_id'])).issubset(family_names):
        raise ValueError('Missing row-family provenance')
    return np.array([family_names[int(i)] not in excluded for i in rows['family_id']],bool)


def instrument_full_labels(env,stack,*,support_trace=False):
    identity=teacher_trace.instrument_environment(env,stack,support_trace=support_trace)
    def recorded(substep,initial,count,sample):
        def observe(index,before,after):
            trace=sample(index,before,after)
            extra=dict(solver_labels(env,after[0]),features=execution_features(trace))
            return dict(trace,**{'aux_'+k:v for k,v in extra.items()})
        return teacher_trace.recorded_loop(substep,initial,count,observe)
    observe=clone_globals(teacher_trace._observe_native_loop,recorded_loop=recorded)
    env._step_impl=MethodType(clone_globals(env._step_impl.__func__,_wmj_recorded_loop=observe),env)
    return dict(identity,labels='native J*qacc6 and current47; diagnostic/training labels only')


def source_capture_code(source):
    """Reuse the a39 capture, replacing its train-only bank comparison explicitly."""
    tree=ast.parse(capture_source(source));body=tree.body[0].body;result=[];removed=[]
    for node in body:
        text=ast.unparse(node)
        if text.startswith(('bank, metadata =','if [metadata[','initial_course, _ =')):
            removed.append('bank');continue
        if text.startswith(('delta =','observation_deltas =','if max((v for k, v in delta.items()',
                            'initial_course =','print(\'[FREE_SETUP] common')):
            removed.append('comparison');continue
        if text.startswith('source = json.loads('):
            result.extend(ast.parse("source=json.loads(authenticate(repository_root(),plan['source_collection']).read_text());delta={};observation_deltas={}").body)
        else:result.append(node)
    if removed.count('bank')!=3 or removed.count('comparison')!=5:
        raise ValueError('Original capture initialization changed: '+str(removed))
    tree.body[0].body=result
    return ast.unparse(ast.fix_missing_locations(tree)).replace(
        '[FREE_SETUP] authenticated models and independent bank restored',
        '[EXPANDED_SETUP] authenticated frozen models restored; original admitted source states follow')


def make_source_capture(recorded_actions=None):
    from .audit_free_imagination import audit
    code=source_capture_code(inspect.getsource(audit))
    if recorded_actions is not None:
        original='action = actor_mean(actor, adapter.observation(current, paired, goals))'
        if code.count(original)!=1:raise ValueError('Native action capture changed')
        code=code.replace(original,'action = recorded_actions[current.step_count[0]]')
    scope=dict(audit.__globals__,instrument_environment=instrument_full_labels,recorded_actions=recorded_actions)
    exec(compile(code,'<WMJ-expanded-native-capture>','exec'),scope)
    return scope[audit.__name__]


def restore_candidate(cfg,reference):
    """Require the immediate u8192 parent, its exact scale and unchanged inference."""
    setting=cfg['expanded_auxiliary_refresh'];path=authenticate(repository_root(),reference)
    m=json.loads(path.read_text());parent=json.loads(authenticate(repository_root(),setting['parent_checkpoint']).read_text())
    if (m['schema']!='wmj_solver_auxiliary_checkpoint_v1' or m['width']!=list(WIDTHS)
            or m['parent_checkpoint']!=setting['parent_checkpoint'] or m['normalization']!=parent['normalization']
            or not 0<m['additional_updates']<=4096 or m['updates']!=parent['updates']+m['additional_updates']):
        raise ValueError('Expanded auxiliary candidate lineage/layout mismatch')
    for name in ('solver_auxiliary.py','neural.py'):
        if m['source_sha256'][name]!=file_hash(Path(__file__).parent/name):raise ValueError('Inference source changed')
    scale=scale_from_statistics(**m['normalization']);parameters=init_mlp(jax.random.PRNGKey(0),WIDTHS,zero_output=True)
    restored=restore_numeric_state(dict(parameters=parameters,optimizer=init_adam(parameters),scale=scale),
        read_array_archive(path.parent/m['archive_file'],m['archive']),m['state'],context_sha256=m['model_signature'])
    if int(restored['optimizer'].step)!=m['updates']:raise ValueError('Restored Adam count differs')
    return restored,m
