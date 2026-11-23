"""Auxiliary-only continuation with the approved 50/50 old/new source mixture."""
from __future__ import annotations

import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np

from .artifacts import read_array_archive,write_array_archive_unique
from .auxiliary_refresh import restore_auxiliary
from .contracts import authenticate,config_hash,file_hash,repository_root
from .expanded_auxiliary import excluded_family_mask
from .fit_solver_auxiliary import evaluate,family_pool,sample_pool,summarize_prediction
from .native_entry_collision import write_compact_json
from .neural import adam_update
from .solver_auxiliary import WIDTHS,supervised_loss
from .state_adapter import numeric_tree_digest,pack_numeric_state


def read_old_labels(cfg,guard,plan):
    parts={};domains=[];all_names={};excluded_counts={};partitions={}
    for domain,reference in enumerate(plan['old_labels']):
        path=authenticate(repository_root(),reference);label=json.loads(path.read_text())
        if label['status']!='LABELS_COMPLETE_INDEPENDENT_DIAGNOSTIC_ONLY':raise ValueError('Incomplete older labels')
        if (label['physics_sha256']!=config_hash(cfg['physics']) or label['control_sha256']!=config_hash(cfg['control'])):
            raise ValueError('Previous physics/control mismatch')
        excluded_counts[str(domain)]=0
        for record in label['records']:
            guard.check();groups=record['groups']
            if domain==0:
                names={record['trial']*len(groups)+i:g['root_seed_family'] for i,g in enumerate(groups)}
            else:names=dict(enumerate(sorted({g['root_seed_family'] for g in groups})))
            for g in groups:
                name=g['root_seed_family']
                if name in partitions and partitions[name]!=g['split']:raise ValueError('Older source family crosses split')
                partitions[name]=g['split'];all_names.setdefault(name,len(all_names))
            rows=read_array_archive(path.parent/record['archive_file'],record['archive'])
            keep=excluded_family_mask(rows,names,set(plan['excluded_diagnostic_families']))
            excluded_counts[str(domain)]+=int((~keep).sum())
            rows={k:v[keep] for k,v in rows.items()}
            if not np.isin(rows['split'],[0,1]).all():raise ValueError('Final test label leak')
            rows['family_id']=np.array([all_names[names[int(x)]] for x in rows['family_id']],np.int32)
            for k,v in rows.items():parts.setdefault(k,[]).append(v)
            domains.append(np.full(len(rows['split']),domain,np.int8))
    return {k:np.concatenate(v) for k,v in parts.items()},np.concatenate(domains),excluded_counts,partitions


def fit(cfg,args,guard,report,plan):
    setting=cfg['expanded_auxiliary_refresh'];path=authenticate(repository_root(),setting['collection'])
    collection=json.loads(path.read_text())
    if collection['status']!='EXPANDED_AUXILIARY_LABELS_COMPLETE' or collection['plan']!=setting['plan']:
        raise ValueError('Wrong expanded collection')
    new=read_array_archive(path.parent/'labels.npz',collection['labels'])
    old,domains,excluded,partitions=read_old_labels(cfg,guard,plan)
    for row in plan['source_rows']:
        if partitions.get(row['root_seed_family'])!=row['split']:raise ValueError('New source changed original split')
    restored,parent=restore_auxiliary(cfg,plan['parent_checkpoint'])
    parameters,optimizer,scale=(restored[k] for k in ('parameters','optimizer','scale'))
    norm_hash=numeric_tree_digest(scale);start_step=int(optimizer.step)
    if start_step!=parent['updates'] or start_step!=8192:raise ValueError('Unexpected parent Adam count')
    pools=[[family_pool(rows,(rows['split']==0)&mask) for mask in (rows['overlap'],~rows['overlap'])] for rows in (old,new)]
    validation={'old_original':(old,np.flatnonzero((old['split']==1)&(domains==0))),
        'old_targeted':(old,np.flatnonzero((old['split']==1)&(domains==1))),
        'new_free_attitude':(new,np.flatnonzero(new['split']==1))}
    if any(not len(index) for rows,index in validation.values()):raise ValueError('Missing independent validation labels')
    report.update(parent_checkpoint=plan['parent_checkpoint'],parent_updates=start_step,
        parent_parameters_sha256=numeric_tree_digest(parameters),parent_optimizer_sha256=numeric_tree_digest(optimizer),
        normalization_sha256=norm_hash,normalization=parent['normalization'],excluded_old_rows=excluded,
        recipe=plan['fit'],source_counts={name:dict(train_rows=int((rows['split']==0).sum()),
        train_families=len(np.unique(rows['family_id'][rows['split']==0]))) for name,rows in [('old',old),('new',new)]},
        parent_validation={},validation_history=[],checkpoints=[],actual_sampled_rows=dict(old=0,new=0))
    for name,(rows,index) in validation.items():report['parent_validation'][name]=evaluate(parameters,scale,rows,index)[0]
    write_compact_json(args.output_dir/'prerun.json',report,guard.budget)
    @jax.jit
    def update(parameters,optimizer,features,targets):
        loss,gradient=jax.value_and_grad(supervised_loss)(parameters,scale,features,targets)
        parameters,optimizer,diag=adam_update(parameters,gradient,optimizer,learning_rate=.0003,max_gradient_norm=1.)
        return parameters,optimizer,loss,diag['gradient_norm']
    signature=config_hash(dict(plan=setting['plan'],collection=setting['collection'],source=report['source_sha256']))
    rng=np.random.default_rng(plan['fit']['seed']);started=time.monotonic()
    for step in range(1,4097):
        if step%64==1:guard.check()
        indexes=[np.concatenate([sample_pool(rng,pool,128) for pool in pair]) for pair in pools]
        for name,rows,index in zip(('old','new'),(old,new),indexes):
            if not np.all(rows['split'][index]==0) or rows['overlap'][index].sum()!=128:
                raise ValueError('Fit source/bin quota changed')
            report['actual_sampled_rows'][name]+=len(index)
        features,target=(np.concatenate([rows[key][index] for rows,index in zip((old,new),indexes)]) for key in ('features','target'))
        parameters,optimizer,loss,norm=update(parameters,optimizer,jnp.asarray(features),jnp.asarray(target))
        if step==1 or step%256==0:
            values=[float(loss),float(norm)]
            if not np.isfinite(values).all():raise FloatingPointError('Nonfinite auxiliary update')
            print(f'[EXPANDED_FIT] additional={step}/4096 loss={values[0]:.6g} grad={values[1]:.6g}',flush=True)
        if step%1024==0:
            guard.check();metrics={};predictions={}
            for name,(rows,index) in validation.items():metrics[name],predictions[name]=evaluate(parameters,scale,rows,index)
            report['validation_history'].append(dict(additional_updates=step,metrics=metrics))
            if numeric_tree_digest(scale)!=norm_hash or int(optimizer.step)!=start_step+step:
                raise ValueError('Scale or optimizer continuation changed')
            arrays,state=pack_numeric_state(dict(parameters=parameters,optimizer=optimizer,scale=scale),context_sha256=signature)
            filename=f'auxiliary_u{start_step+step:06d}'
            archive=write_array_archive_unique(args.output_dir/(filename+'.npz'),arrays,guard.budget)
            metadata=dict(schema='wmj_solver_auxiliary_checkpoint_v1',training_version='WMJ-V1',width=list(WIDTHS),
                model_signature=signature,archive_file=filename+'.npz',archive=archive,state=state,
                parent_checkpoint=plan['parent_checkpoint'],additional_updates=step,updates=start_step+step,
                numpy_rng=rng.bit_generator.state,normalization=parent['normalization'],source_sha256=report['source_sha256'],
                plan=setting['plan'],collection=setting['collection'],recipe=plan['fit'],integration_in_world=False)
            target=args.output_dir/(filename+'.json');sha=write_compact_json(target,metadata,guard.budget)
            report['checkpoints'].append(dict(path=str(target),sha256=sha))
            print('[EXPANDED_VALID] additional='+str(step)+' scalar-v P95='+str({k:v['positive_force_scalar_velocity_error_m_s']['p95'] for k,v in metrics.items()}),flush=True)
    report['final_validation_by_region']={}
    for name,(rows,index) in validation.items():
        selected={k:v[index] for k,v in rows.items()};pred=predictions[name]
        report['final_validation_by_region'][name]={str(int(region)):summarize_prediction(pred[selected['region_id']==region],
            {k:v[selected['region_id']==region] for k,v in selected.items()}) for region in np.unique(selected['region_id'])}
    report.update(status='EXPANDED_AUXILIARY_FIT_COMPLETE_NOT_PROMOTED',additional_updates=4096,world_updates=4096,
        total_updates=int(optimizer.step),fit_wall_s=time.monotonic()-started,normalization_preserved=True,
        final_parameter_sha256=numeric_tree_digest(parameters),final_validation=metrics,candidate=report['checkpoints'][-1],
        robot_updates=0,actor_updates=0,critic_updates=0,default_replaced=False)
