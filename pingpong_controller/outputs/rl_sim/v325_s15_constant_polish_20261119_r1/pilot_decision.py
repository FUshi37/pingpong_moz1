"""Preregistered paired physical acceptance; return alone never selects a model."""
import numpy as np


def compare(candidate,reference):
    failures=[]
    for name in ('clean','wind160'):
        new,old=candidate['groups'][name],reference['groups'][name]
        for field in ('survival','mean_seconds','hit12'):
            if new[field]<old[field]-1.e-9:failures.append(name+':'+field)
        qnew,qold=new['quality'],old['quality']
        for key,tolerance in (('center_m',.0005),('apex_error_rms_m',.001)):
            if qnew.get(key,float('inf'))>qold[key]+tolerance:failures.append(name+':'+key)
        if qnew.get('in_band',0.)<qold['in_band']-.02:failures.append(name+':band')
        if new['sums'].get('done/racket_z_out_of_bounds',0.)>old['sums'].get('done/racket_z_out_of_bounds',0.):
            failures.append(name+':height_failures')
    old_worst=min(v['mean_seconds'] for v in reference['wind_sectors'].values())
    new_worst=min(v['mean_seconds'] for v in candidate['wind_sectors'].values())
    if new_worst<old_worst-1.e-9:failures.append('wind:worst_sector_lifetime')
    if 'wind_vectors_m_s' in candidate and 'wind_vectors_m_s' in reference:
        vectors=np.asarray(reference['wind_vectors_m_s'])
        for name,target in (('negative_x',[-1.6,0.,0.]),('negative_y',[0.,-1.6,0.]),('positive_x',[1.6,0.,0.])):
            mask=np.linalg.norm(vectors-np.asarray(target),axis=-1)<1.e-5
            if not np.any(mask):raise ValueError('Missing criticalwindaxis '+name)
            if np.mean(np.asarray(candidate['lifetime_s'])[mask])<np.mean(np.asarray(reference['lifetime_s'])[mask])-1.e-9:
                failures.append('wind:axis_'+name+'_lifetime')
    return dict(passed=not failures,failures=failures,
                worst_sector_lifetime_s=new_worst,reference_worst_sector_lifetime_s=old_worst)


def decide(candidate,parent,baseline):
    source=compare(candidate,parent)
    control=compare(candidate,baseline)
    wind=candidate['groups']['wind160']
    improved=any(wind[key]>parent['groups']['wind160'][key]+1.e-9
                 for key in ('survival','mean_seconds'))
    return dict(accepted=source['passed'] and control['passed'] and improved,
                versus_parent=source,versus_trained_control=control,wind_improved=improved)


def select_polish(candidate,parent,baseline):
    candidate_verdict=decide(candidate,parent,baseline)
    control_verdict=compare(baseline,parent)
    control_improved=any(baseline['groups']['wind160'][key]>parent['groups']['wind160'][key]+1.e-9
                         for key in ('survival','mean_seconds'))
    selected=('fixed_direction_speed_cohorts' if candidate_verdict['accepted'] else
              'original_sampling' if control_verdict['passed'] and control_improved else None)
    return dict(sampler=selected,candidate=candidate_verdict,
                original_sampling_finetune=control_verdict,original_sampling_wind_improved=control_improved)
