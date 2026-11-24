"""V325/V326 admission: fresh frozen evidence replaces stale mixed history."""
from __future__ import annotations

import math
import re
from pathlib import Path

from fixed_plane_v321_long_episode import quality_passed
from fixed_plane_v325_wind import MODE_NAMES

PROTOCOL = 'wind_frozen_current_policy'


def validation_seed(base_seed: int, stage: int, update: int) -> int:
    """Pair clean/wind resets, changing the root on each validation retry."""
    return (int(base_seed) + 100000 + int(stage)*100 + int(update)) % (2**32)


def validation_eligibility(window_size: int, quality: dict,
                           stage_age_s: float, required_s: float,
                           *, quality_gate=quality_passed) -> bool:
    """Retain rolling32 and duration requirements; old endpoints are telemetry."""
    return (window_size == 32 and math.isfinite(stage_age_s)
            and stage_age_s >= required_s and quality_gate(quality))


def fresh_acceptance(stack, stage: int, evaluations: dict, policy_sha256: str) -> dict:
    """Recompute original gates for1024 fresh lanes per domain/action mode.

    The old training prerequisite required64 endpoints for each enabled wind
    mode. Retain that stronger sample floor in addition to the original frozen
    grader's32-lane floor. No threshold or physical episode is shortened.
    """
    from fixed_plane_wind_validation import grade_validation
    failures, recomputed = [], {}
    if not re.fullmatch('[a-f0-9]{64}', policy_sha256 or ''):
        failures.append('missing_current_policy_digest')
    if set(evaluations) != {'wind', 'clean'}:
        failures.append('missing_domain')
    lesson=stack.spec.course[stage]
    for domain in ('wind','clean'):
        result=evaluations.get(domain,{})
        quiet=domain=='clean'
        if (not result.get('complete') or result.get('observed_s') != lesson.seconds
                or result.get('quiet') is not quiet or not result.get('fixed_policy')
                or result.get('policy_sha256') != policy_sha256):
            failures.append(domain+'/incomplete_or_policy_mismatch')
        graded=grade_validation(stack,stage,result.get('modes',{}),quiet=quiet,
            seconds=lesson.seconds,n_envs=1024)
        recomputed[domain]=graded
        failures.extend(domain+'/'+x for x in graded['failures'])
        for mode,m in result.get('modes',{}).items():
            for index,name in enumerate(MODE_NAMES):
                required=(name=='clean') if quiet else lesson.mode_weights[index]>0.
                if required and m.get('wind',{}).get(name,{}).get('n',0)<64:
                    failures.append(f'{domain}/{mode}/{name}/fewer_than64')
    return dict(passed=not failures,failures=failures,protocol=PROTOCOL,
                policy_sha256=policy_sha256,recomputed=recomputed)


def impossible_frozen_gate(active, modes, valid_steps, elapsed_s: float,
                           lesson, origin, *, quiet: bool, conditions=None) -> list[str]:
    """Reject only when first-failure outcomes cannot reach an original floor."""
    import numpy as np
    active,modes,valid_steps=map(np.asarray,(active,modes,valid_steps))
    total_steps=round(lesson.seconds/.005)
    remaining=max(0,total_steps-round(elapsed_s/.005))
    failures=[]
    def check(name,mask,min_n,min_survival,min_length):
        n=int(mask.sum())
        if n<min_n:
            failures.append(name+'/insufficient_population')
            return
        if active[mask].mean()<min_survival:
            failures.append(name+'/survival_upper_bound')
        maximum=(valid_steps[mask].sum()+remaining*active[mask].sum())/(n*total_steps)
        if maximum<min_length:
            failures.append(name+'/length_upper_bound')
    for index,name in enumerate(MODE_NAMES):
        required=(name=='clean') if quiet else lesson.mode_weights[index]>0.
        if not required:
            continue
        mask=modes==index
        if conditions is not None:
            mask &= np.asarray(conditions)==0
        check(name,mask,64,origin.min_survival,origin.min_length_fraction)
    if conditions is not None:
        from fixed_plane_v322_condition_gates import condition_gate_settings
        for index,(name,limits) in enumerate(condition_gate_settings().items()):
            check('condition/'+name,np.asarray(conditions)==index,
                  limits['min_n'],limits['min_survival'],limits['min_length_fraction'])
    return failures


def validate_no_wind_review(path: Path, source: Path, stack) -> None:
    """Require a completed, source-bound30s CUDA no-wind preservation review."""
    import json
    from fixed_plane_v323_ball_obs_noise import digest_file
    d=json.loads(path.read_text())
    expected=(('M12',True),('M12',False),('V325_u506',False),('V327_u3386',False))
    cases=d.get('cases',[])
    if (d.get('backend')!='gpu' or not d.get('complete') or d.get('ppo_updates')!=0
            or d.get('gpu_uuid')!=stack.spec.origin.gpu_uuid
            or d.get('xml_sha256')!=digest_file(stack.xml)
            or d.get('checkpoints',{}).get('M12',{}).get('sha256')!=stack.spec.origin.parent_sha256
            or [(c.get('actor'),c.get('all_clean')) for c in cases]!=list(expected)
            or any(not c.get('complete') or c.get('observed_s')!=30 for c in cases)):
        raise ValueError('A completed native/zero-wind review is required')
    snapshot=path.parent/'source_snapshot'
    if d.get('source_sha256')!=digest_file(snapshot/'audit_fixed_plane_wind_nowind.py'):
        raise ValueError('No-wind review source hash changed')
    for name in ('fixed_plane_v325_wind.py','fixed_plane_wind_env.py'):
        if digest_file(snapshot/name)!=digest_file(source/name):
            raise ValueError('Wind kernel changed after no-wind review')
    case=cases[0]
    rows=case.get('samples',[])
    if [x.get('second') for x in rows]!=list(range(1,31)):
        raise ValueError('Incomplete30s no-wind state pairing')
    for row in rows:
        a,b=row.get('native_clean_digest',''),row.get('adapter_clean_digest','')
        if (not re.fullmatch('[a-f0-9]{64}',a) or a!=b
                or any(row.get('diff',{}).get(k,math.inf)!=0.
                       for k in ('action','done','obs','qpos','qvel','reward'))):
            raise ValueError('Native/zero-wind states or actions diverged')
    for label in ('native','wind_adapter'):
        for name in ('clean_mean','clean_sampled'):
            r=case.get(label,{}).get(name,{})
            if r.get('survival',0.)<.95 or r.get('numerical_errors',1):
                raise ValueError('M12 no-wind baseline did not retain stability')
