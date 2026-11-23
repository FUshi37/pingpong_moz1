"""Offline action-effect labels and paired changes; missing outcomes stay visible."""

from __future__ import annotations

import numpy as np

from .contact_geometry import surface_features
from .fit_robot import summarize_errors
from .tangent_contact import tangent_prior_increment


def model_event_trace(physical, *, surface, ball_radius_m, contact_contract):
    """Evaluate the selected physical prior's force for event diagnostics only."""
    before, after = physical['contact']['before'], physical['contact']['after']
    pose = physical['racket_before']
    geometry = surface_features(surface,before.position_m,pose['geom_position'],
                                 pose['rotation'],ball_radius_m)
    prior = tangent_prior_increment(contact_contract,geometry,physical['solver_terms'],
        ball_position_m=before.position_m,ball_velocity_m_s=before.velocity_m_s,
        ball_omega_world_rad_s=before.omega_world_rad_s,geom_position_m=pose['geom_position'],
        racket_linear_velocity_m_s=pose['velocity'],racket_angular_velocity_rad_s=pose['omega'])
    return dict(contact_overlap=physical['contact']['overlap'],
        contact_normal_force_n=prior.force_contact_n[...,0],
        ball_qpos_before=before.position_m,ball_qpos_after=after.position_m,
        ball_qvel_before=before.velocity_m_s,ball_qvel_after=after.velocity_m_s)


def summarize_outcomes(labels):
    """First controlled release and its actual apex, with explicit censoring."""
    rows=[]
    for lane in range(labels['valid'].shape[1]):
        found=np.flatnonzero(labels['controlled_release'][:,lane])
        row=dict(release_present=bool(len(found)),impact_id=None,release_time_s=None,
            outgoing_velocity_m_s=None,release_position_m=None,analytic_apex_m=None,
            actual_apex_present=False,actual_apex_m=None,actual_apex_time_s=None,
            entries=int(labels['contact_entry'][:,lane].sum()),
            releases=int(labels['contact_release'][:,lane].sum()),
            invalid_substeps=int((~labels['valid'][:,lane]).sum()))
        if len(found):
            t=int(found[0]); impact=int(labels['impact_id'][t,lane])
            position=labels['release_position_m'][t,lane]
            velocity=labels['release_velocity_m_s'][t,lane]
            flight=max(float(velocity[2]),0.)/9.81
            apex=position+velocity*flight+np.array([0.,0.,-.5*9.81*flight**2])
            row.update(impact_id=impact,release_time_s=t*.001,
                outgoing_velocity_m_s=velocity.tolist(),release_position_m=position.tolist(),analytic_apex_m=apex.tolist())
            peaks=np.flatnonzero(labels['actual_apex'][:,lane]
                & (labels['impact_id'][:,lane]==impact) & (np.arange(len(labels['valid']))>t))
            if len(peaks):
                a=int(peaks[0])
                row.update(actual_apex_present=True,actual_apex_m=labels['apex_position_m'][a,lane].tolist(),
                           actual_apex_time_s=a*.001)
        rows.append(row)
    return rows


def compare_effects(native, world):
    """Report central differences across all pairs, and separate event switches."""
    n=len(native['base'])
    if any(len(rows)!=n for rows in (*native.values(),*world.values())):
        raise ValueError('Counterfactual case counts differ')
    paired=[]; stable=[]; missing_native=missing_world=0
    for i in range(n):
        observed=[native[k][i] for k in ('base','plus','minus')]
        predicted=[world[k][i] for k in ('base','plus','minus')]
        native_present=all(x['release_present'] for x in observed)
        world_present=all(x['release_present'] for x in predicted)
        missing_native+=not native_present; missing_world+=not world_present
        same_ids=native_present and world_present and len({x['impact_id'] for x in observed+predicted})==1
        paired.append(same_ids)
        stable.append(same_ids and len({(x['entries'],x['releases']) for x in observed+predicted})==1)
    paired,stable=np.array(paired),np.array(stable)
    report=dict(cases=n,paired_release_cases=int(paired.sum()),
        same_event_count_cases=int(stable.sum()),event_switch_or_missing_cases=int((~stable).sum()),
        missing_native_release_cases=int(missing_native),missing_world_release_cases=int(missing_world))
    for name,floor in (('outgoing_velocity_m_s',.001),('analytic_apex_m',.0001),('actual_apex_m',.0001)):
        valid=paired.copy()
        if name=='actual_apex_m':
            valid &= np.array([all(d[k][i]['actual_apex_present'] for d in (native,world)
                                for k in ('plus','minus')) for i in range(n)])
        ids=np.flatnonzero(valid)
        actual=np.array([np.array(native['plus'][i][name])-native['minus'][i][name] for i in ids]).reshape(-1,3)
        predicted=np.array([np.array(world['plus'][i][name])-world['minus'][i][name] for i in ids]).reshape(-1,3)
        magnitude=np.linalg.norm(actual,axis=-1); prediction_norm=np.linalg.norm(predicted,axis=-1)
        informative=magnitude>floor
        stable_informative=stable[ids]&informative
        dot=np.sum(actual*predicted,axis=-1)
        cosine=dot/np.maximum(magnitude*prediction_norm,1e-20)
        report[name]=dict(pairs=len(ids),effect_floor=floor,informative_pairs=int(informative.sum()),
            small_effect_pairs=int((~informative).sum()),
            direction_agreement_fraction=float(np.mean(dot[informative]>0.)) if informative.any() else None,
            delta_error=summarize_errors([np.linalg.norm(predicted-actual,axis=-1)]),
            native_effect=summarize_errors([magnitude]),
            cosine=summarize_errors([cosine[informative]]),
            magnitude_ratio=summarize_errors([prediction_norm[informative]/magnitude[informative]]),
            same_event_count_informative_pairs=int(stable_informative.sum()),
            same_event_count_direction_agreement_fraction=float(np.mean(dot[stable_informative]>0.))
                if stable_informative.any() else None,
            same_event_count_cosine=summarize_errors([cosine[stable_informative]]))
    return report


def compare_with_repeat(native, world, repeat, cases):
    """Stratify effects by observed duplicate drift, without declaring readiness.

    Three times one observed repeat difference is a sensitivity diagnostic,
    not a confidence interval or a guarantee on native numerical error.
    All cases, including clipped-zero pulses and missing outcomes, stay counted.
    """
    if len(cases)!=len(repeat) or len(cases)!=len(native['base']):
        raise ValueError('Repeat and counterfactual populations differ')
    report=dict(cases=len(cases),repeat_multiplier=3.,qualification=False)
    for name,floor in (('outgoing_velocity_m_s',.001),('actual_apex_m',.0001)):
        rows=[]
        for i,case in enumerate(cases):
            participants=[native[k][i] for k in ('base','plus','minus')]
            participants += [world[k][i] for k in ('base','plus','minus')]+[repeat[i]]
            paired=all(p['release_present'] and p.get(name) is not None for p in participants)
            paired=paired and len({p['impact_id'] for p in participants})==1
            row=dict(case_index=i,paired=paired,effective_pulse=case['effective_clipped_central_pulse_norm']>0.)
            if paired:
                effect=np.asarray(native['plus'][i][name])-native['minus'][i][name]
                prediction=np.asarray(world['plus'][i][name])-world['minus'][i][name]
                drift=np.asarray(repeat[i][name])-native['base'][i][name]
                magnitude=float(np.linalg.norm(effect)); noise=float(np.linalg.norm(drift))
                dot=float(np.dot(effect,prediction))
                row.update(native_effect_norm=magnitude,repeat_difference_norm=noise,
                    exceeds_repeat=row['effective_pulse'] and magnitude>max(floor,3.*noise),
                    direction_agrees=dot>0.,
                    direction_margin_exceeds_repeat=dot>3.*noise*float(np.linalg.norm(prediction)))
            rows.append(row)
        paired=[r for r in rows if r['paired']]
        resolved=[r for r in paired if r['exceeds_repeat']]
        report[name]=dict(paired_cases=len(paired),missing_cases=len(rows)-len(paired),
            clipped_zero_cases=sum(not r['effective_pulse'] for r in rows),
            effect_exceeds_repeat_cases=len(resolved),unresolved_cases=len(rows)-len(resolved),
            resolved_direction_agreement_fraction=float(np.mean([r['direction_agrees'] for r in resolved])) if resolved else None,
            resolved_direction_margin_fraction=float(np.mean([r['direction_margin_exceeds_repeat'] for r in resolved])) if resolved else None,
            repeat_difference=summarize_errors([np.array([r['repeat_difference_norm'] for r in paired])]),
            cases=rows)
    return report
