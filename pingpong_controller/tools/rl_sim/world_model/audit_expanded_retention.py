"""Reuse all original validation cases without duplicating per-step arrays."""
from __future__ import annotations

import ast
import inspect
from copy import deepcopy

from . import audit_auxiliary_refresh as existing
from .auxiliary_refresh import restore_auxiliary
from .expanded_auxiliary import restore_candidate
from .normal_checkpoint import load_components
from .native_entry_collision import write_compact_json


def retention_source(source):
    """Keep physical calculations and every case; omit only duplicate files."""
    tree=ast.parse(source);removed=[];appends=[]
    class Rewrite(ast.NodeTransformer):
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                name=node.targets[0].id
                if name in ('filename','archive','window','path','digest'):
                    removed.append(name);return None
            return self.generic_visit(node)
        def visit_Expr(self,node):
            if ast.unparse(node).startswith("report['windows'].append("):
                appends.append(True)
                return ast.parse("report['windows'].append(dict(source=source,trial=trial,batch=batch_number,horizon=horizon,lanes=lanes[:useful].tolist(),starts=starts[:useful].tolist(),ordinal_release_diagnostic=event_results,storage='metrics_and_events_only;no_duplicate_step_arrays'))").body[0]
            return self.generic_visit(node)
    tree=Rewrite().visit(tree)
    if sorted(removed)!=sorted(['filename','archive','window','path','digest']) or len(appends)!=1:
        raise ValueError('Original retention writer structure changed')
    return ast.unparse(ast.fix_missing_locations(tree))


def audit(cfg,args,guard,report,plan):
    def components(*a,**kw):
        parameters,options,refs,_,_=load_components(*a,**kw)
        parent,_=restore_auxiliary(cfg,plan['parent_checkpoint'])
        return parameters,options,refs,{k:parent[k] for k in ('parameters','scale')},plan['parent_checkpoint']
    def restore(config,reference):
        if reference==cfg['expanded_auxiliary_refresh']['candidate_checkpoint']:return restore_candidate(cfg,reference)
        return restore_auxiliary(config,reference)
    source=retention_source(inspect.getsource(existing.audit))
    scope=dict(existing.audit.__globals__,load_components=components,restore_auxiliary=restore)
    exec(compile(source,'<WMJ-expanded-auxiliary-retention>','exec'),scope)
    private=deepcopy(cfg)
    private['auxiliary_refresh']['candidate_checkpoint']=cfg['expanded_auxiliary_refresh']['candidate_checkpoint']
    scope['audit'](private,args,guard,report)
    # Keep every regional metric/event while bounding each atomic write reserve.
    regional=report.pop('by_region');groups={}
    for key,value in regional.items():
        groups.setdefault('/'.join(key.split('/')[:3]),{})[key]=value
    report['by_region_files']=[]
    for index,(group,values) in enumerate(sorted(groups.items())):
        path=args.output_dir/f'region_metrics_{index:02d}.json'
        sha=write_compact_json(path,values,guard.budget)
        report['by_region_files'].append(dict(group=group,path=str(path),sha256=sha))
    windows=report.pop('windows');groups={}
    for window in windows:groups.setdefault(window['batch'],[]).append(window)
    report['window_files']=[]
    for batch,values in sorted(groups.items()):
        path=args.output_dir/f'windows_b{batch:02d}.json'
        sha=write_compact_json(path,values,guard.budget)
        report['window_files'].append(dict(batch=batch,path=str(path),sha256=sha))
    report.update(status='EXPANDED_AUXILIARY_RETENTION_COMPLETE',
        retained_population='every original old/targeted validation lane;original bounded control',
        per_step_array_storage='not requested;all numeric metrics and event cases retained in result',
        actor_updates=0,critic_updates=0,world_updates=0,default_replaced=False)
