"""Authenticate physical-height improvement before a full-state continuation."""

import json
from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np

from assess_fixed_plane_v184 import read_blocks, combine, paired_interval, sha256
from fixed_plane_v191_gpu1 import PARENT_SHA, PROFILES


def population(directory: str | Path, *, seeds: Sequence[int]):
    """Read complete hashed stochastic CUDA populations and config identity."""
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest.get('deterministic',False):
        raise ValueError('Reward admission requires stochastic policy evidence')
    cp=Path(manifest['checkpoint'])
    if sha256(cp)!=manifest['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint hash mismatch')
    from train_juggle_mjx_curriculum import build_curriculum
    stages=build_curriculum(curriculum_profile=manifest['profile'],
        measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,
        critic_command_history_steps=12,right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    stage=next(s for s in stages if s.name==manifest['stage'])
    import hashlib
    if hashlib.sha256(repr(vars(stage.cfg)).encode()).hexdigest()!=manifest['cfg_sha256']:
        raise ValueError('Frozen physical/reward config mismatch')
    return read_blocks(directory,prefix='',checkpoint_sha=manifest['checkpoint_sha256'],seeds=seeds)


def assess_trial(candidate_dir: str | Path, control_dir: str | Path,
                 source_dir: str | Path, *,
                 seeds: Sequence[int]=(20265661,20265662)) -> dict[str,Any]:
    """Compare full physical quality against the parent and matched PPO control."""
    cm,candidate=population(candidate_dir,seeds=seeds)
    mm,matched=population(control_dir,seeds=seeds)
    pm,parent=population(source_dir,seeds=seeds)
    if (cm['profile'] not in PROFILES[1:] or mm['profile']!=PROFILES[0]
            or pm['checkpoint_sha256']!=PARENT_SHA):
        raise ValueError('Wrong candidate/control/protected parent')
    if '_s13_' not in cm['stage'] or '_s13_' not in mm['stage'] or '_s13_' not in pm['stage']:
        raise ValueError('Height repair requires complete S13 populations')
    metrics=dict(candidate=combine(candidate),matched=combine(matched),parent=combine(parent))
    c=metrics['candidate'];checks={};uncertainty={}
    for label,reference in [('parent',parent),('matched',matched)]:
        r=metrics[label]
        band_ci=paired_interval(candidate,reference,'late_in_band_rate')
        center_ci=paired_interval(candidate,reference,'mean_hit_contact_center_dist_m')
        uncertainty[label]=dict(band_delta_90pct_ci=band_ci,
            center_delta_90pct_ci_m=center_ci)
        gain=.005 if label=='parent' else .003
        checks[f'band_gain_vs_{label}']=c['late_in_band_rate']>=r['late_in_band_rate']+gain
        checks[f'band_ci_vs_{label}']=band_ci[0]>0.
        checks[f'tails_stable_vs_{label}']=all(c[k]<=r[k]+.001 for k in (
            'late_below_floor_rate','late_above_ceiling_rate'))
        checks[f'center_stable_vs_{label}']=(
            c['mean_hit_contact_center_dist_m']<=r['mean_hit_contact_center_dist_m']+.0001
            and center_ci[1]<=.0002)
        checks[f'motion_stable_vs_{label}']=all(c[k]<=1.03*r[k] for k in (
            'steady_racket_vxy_rms_m_s','steady_racket_omega_rms_rad_s'))
        checks[f'adjacent_stable_vs_{label}']=c['late_adjacent_jump_m']<=r['late_adjacent_jump_m']+.0001
        checks[f'view_stable_vs_{label}']=all(
            cp['block'][k]>=rp['block'][k]-.002
            for cp,rp in zip(candidate,reference,strict=True)
            for k in ('advance_eval/ball_view_in_bounds','advance_eval/ball_view_z_ideal'))
        checks[f'joint_safety_stable_vs_{label}']=all(
            cp['arrays'][k].mean()/1200.<=rp['arrays'][k].mean()/1200.+.0012
            for cp,rp in zip(candidate,reference,strict=True)
            for k in ('arm_qvel_limit_exceed_fraction','arm_qacc_limit_exceed_fraction'))
        checks[f'acquisition_stable_vs_{label}']=all(
            cp['block'][k]>=rp['block'][k]-.002
            for cp,rp in zip(candidate,reference,strict=True)
            for k in ('advance_eval/hit1_rate','advance_eval/hit3_rate'))
    checks['full_survival_and_physical_hits']=all(
        p['block']['advance_eval/mean_len_frac']>=1.
        and p['block']['advance_eval/episode_truncation_rate']>=.995
        and 11.<=p['block']['advance_eval/mean_hits']<=15. for p in candidate)
    checks['preserved_achieved_center_motion_limits']=(
        c['mean_hit_contact_center_dist_m']<=.0065
        and c['steady_racket_vxy_rms_m_s']<=.053
        and c['steady_racket_omega_rms_rad_s']<=.28)
    checks={k:bool(v) for k,v in checks.items()}
    return dict(accepted=all(checks.values()),profile=cm['profile'],
        candidate_directory=str(candidate_dir),control_directory=str(control_dir),
        source_directory=str(source_dir),eval_seeds=list(seeds),
        candidate_sha256=cm['checkpoint_sha256'],control_sha256=mm['checkpoint_sha256'],
        parent_sha256=PARENT_SHA,metrics=metrics,checks=checks,
        failed_checks=[k for k,v in checks.items() if not v],paired_uncertainty=uncertainty)


def strict_s9(directory: str | Path, *, profile: str, checkpoint_sha: str,
              seeds: Sequence[int]) -> dict[str,Any]:
    """Recompute physical S9 gates and retain its original full-block checks."""
    from train_juggle_mjx_curriculum import build_curriculum,rapid_steady_quality_gate
    manifest,pops=population(directory,seeds=seeds)
    if manifest['checkpoint_sha256']!=checkpoint_sha or manifest['profile']!=profile:
        raise ValueError('Wrong S9 retention policy')
    stages=build_curriculum(curriculum_profile=profile,
        measured_ball_mass_range_kg=(.0039,.0041),asymmetric_critic=True,
        critic_command_history_steps=12,right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    stage=stages[8]
    if manifest['stage']!=stage.name: raise ValueError('Wrong S9 stage')
    result=[]
    for p in pops:
        a=p['arrays'];v=combine([p]);n=a['rapid_steady_steady_events'].sum()
        v.update(hit4_apex_z_m=float(a['rapid_steady_hit4_apex_sum_z_m'].sum()/a['rapid_steady_hit4_events'].sum()),
            steady_band_rate=float(a['rapid_steady_steady_band_events'].sum()/n),
            steady_surface_z_abs_error_m=float(a['rapid_steady_steady_surface_z_abs_error_sum_m'].sum()/n),
            steady_events=float(n),late_physical_events=float(a['v80_late_event'].sum()),
            late_pair_events=float(a['v80_late_pair_event'].sum()))
        raw_pass=rapid_steady_quality_gate(stage,v)
        other=[k for k,x in p['block'].items() if k.endswith('_ok') and not bool(x)]
        if not raw_pass or other or not bool(p['block']['advance_eval/passed']):
            raise ValueError(f'Unchanged S9 failed seed{p["seed"]}: {other}; raw={raw_pass}')
        result.append(dict(seed=p['seed'],physical_metrics=v))
    return dict(directory=str(directory),manifest_sha256=sha256(Path(directory)/'manifest.json'),seeds=result)


def validate_probe_payload(payload: Mapping[str, Any], *, ppo_seed: int,
                           parent_checkpoint: str | Path) -> None:
    """Authenticate the source and full-state budget with saved Path objects."""
    source = Path(payload['args']['resume_from'])
    if (payload['args']['seed'] != ppo_seed or payload['stage_update'] != 128
            or source.resolve() != Path(parent_checkpoint).resolve()
            or sha256(source) != PARENT_SHA
            or tuple(payload[k] for k in ('obs_dim', 'critic_obs_dim', 'act_dim')) != (72, 477, 7)
            or payload.get('opt') is None
            or not {'pi', 'v'}.issubset(payload.get('params', {}))):
        raise ValueError('Wrong full-state PPO source/seed/budget')


def require_gate_only_continuation(path: str | Path, *, source_sha: str) -> dict[str, Any]:
    """Authenticate unchanged rewards and the physical hit range, without a gain claim.

    This separate path cannot promote a changed reward or a bounded probe actor.
    All non-hit graduation and retention checks remain active during training.
    """
    from dataclasses import asdict
    from fixed_plane_v191_gpu1 import PARENT_PROFILE, PARENT_STAGE, physical_hit_envelope
    from train_juggle_mjx_curriculum import build_curriculum
    e = json.loads(Path(path).read_text())
    if (source_sha != PARENT_SHA or e.get('parent_sha256') != PARENT_SHA
            or e.get('profile') != PROFILES[0]
            or e.get('kind') != 'unchanged_reward_physical_hit_range_continuation'
            or e.get('reward_optimization_succeeded') is not False):
        raise ValueError('Unchanged reward continuation requires its protected parent and explicit mode')
    kwargs = dict(measured_ball_mass_range_kg=(.0039, .0041), asymmetric_critic=True,
        critic_command_history_steps=12, right_arm_pd_profile='recovered_rmp_rmpmd_v2')
    source = build_curriculum(curriculum_profile=PARENT_PROFILE, **kwargs)
    target = build_curriculum(curriculum_profile=PROFILES[0], **kwargs)
    envelope = physical_hit_envelope(episode_s=6., third_hit_time_s=(.74, 1.24),
        late_contact_z_m=(1.07, 1.12), apex_z_m=(1.4, 1.45), gravity_m_s2=9.81)
    if e.get('mean_hits_range') != list(envelope):
        raise ValueError('Nonzero physical mean-hit envelope required')
    for index, (before, after) in enumerate(zip(source, target, strict=True)):
        if asdict(before.cfg) != asdict(after.cfg):
            raise ValueError('Gate-only path cannot change any reward or physical config')
        if index < 12:
            if asdict(before) != asdict(after):
                raise ValueError('Gate-only path cannot reinterpret earlier stages')
        else:
            left, right = asdict(before), asdict(after)
            for key in ('name', 'notes', 'target_mean_hits', 'max_recent_mean_hits'):
                left.pop(key); right.pop(key)
            if (left != right or (after.target_mean_hits, after.max_recent_mean_hits) != envelope):
                raise ValueError('All non-hit graduation and retention gates must remain exact')
    manifest, pops = population(e['source_directory'], seeds=(20265661, 20265662))
    if (manifest['checkpoint_sha256'] != PARENT_SHA or manifest['profile'] != PARENT_PROFILE
            or manifest['stage'] != PARENT_STAGE):
        raise ValueError('Wrong protected stochastic S13 population')
    if not all(p['block']['advance_eval/mean_len_frac'] >= 1.
            and envelope[0] <= p['block']['advance_eval/mean_hits'] <= envelope[1]
            for p in pops):
        raise ValueError('Source survival and physical hit range must be retained')
    cadence_file = Path(e['cadence_file'])
    if sha256(cadence_file) != e['cadence_sha256']:
        raise ValueError('Physical cadence evidence hash mismatch')
    cadence = json.loads(cadence_file.read_text())
    third_times, late_contact_z = [], []
    for seed in (20265661, 20265662):
        with np.load(Path(e['source_directory']) / f'impact_{seed}.npz') as raw:
            count, events, z_sum = (raw[k] for k in ('hit_count', 'hit_event_count', 'hit_ball_z_sum'))
            if (any(a.shape != (1200, 288) or not np.isfinite(a).all()
                    for a in (count, events, z_sum)) or not np.all(np.any(count >= 3, axis=0))):
                raise ValueError('Complete finite third-hit timing population required')
            third_times.extend(((np.argmax(count >= 3, axis=0) + 1) * .005).tolist())
            late = (count >= 4) & (events > 0)
            late_contact_z.extend((z_sum[late] / events[late]).tolist())
    if not late_contact_z:
        raise ValueError('Late physical contact population missing')
    quantiles = cadence['quantiles']
    if (not np.allclose(np.quantile(third_times, quantiles), cadence['third_hit_time_s'], atol=1e-9, rtol=0.)
            or not np.allclose(np.quantile(late_contact_z, quantiles), cadence['late_ball_center_z_m'], atol=1e-9, rtol=0.)):
        raise ValueError('Cadence summary disagrees with hashed pointwise impacts')
    if (cadence['populations'] != 576 or cadence['skipped_missing_third'] != 0
            or cadence['seeds'] != [20265661, 20265662]
            or not .74 <= cadence['third_hit_time_s'][0] <= cadence['third_hit_time_s'][-1] <= 1.24
            or not 1.07 <= cadence['late_ball_center_z_m'][0] <= cadence['late_ball_center_z_m'][-1] <= 1.12):
        raise ValueError('Physical envelope must retain the complete recorded population')
    return dict(profile=PROFILES[0], mode=e['kind'], reward_optimization_succeeded=False,
        parent_sha256=PARENT_SHA, mean_hits_range=list(envelope), rewards_unchanged=True,
        non_hit_gates_unchanged=True, source_metrics=combine(pops),
        cadence_recomputed=dict(populations=len(third_times), late_contacts=len(late_contact_z),
            third_hit_time_s=[min(third_times), max(third_times)],
            late_contact_z_m=[min(late_contact_z), max(late_contact_z)]),
        note='Preserved parent continuation only. No claim of height improvement or graduation.')


def require_admission(path: str | Path, *, profile: str, source_sha: str) -> dict[str,Any]:
    """Require two positive PPO trials and new independent retained-domain evidence."""
    if profile == PROFILES[0]:
        return require_gate_only_continuation(path, source_sha=source_sha)
    e=json.loads(Path(path).read_text())
    if source_sha!=PARENT_SHA or profile not in PROFILES[1:] or e.get('profile')!=profile:
        raise ValueError('Reward continuation requires its protected parent and profile')
    contract=Path(e['preregister_file'])
    if sha256(contract)!=e['preregister_sha256']: raise ValueError('Preregistration hash mismatch')
    c=json.loads(contract.read_text())
    if c['ppo_seeds']!=[20263632,20266012] or c['probe_updates']!=128:
        raise ValueError('Wrong independent PPO contract')
    if profile == PROFILES[6]:
        if c.get('arms', {}).get('v197') != {'fixed_plane_v191_absolute_height_multiplier': 1.25}:
            raise ValueError('Milder height repair requires its preregistered reward contract')
        diagnostic = Path(c['exploratory_diagnostic_file'])
        if sha256(diagnostic) != c['exploratory_diagnostic_sha256']:
            raise ValueError('Height-tradeoff source evidence hash mismatch')
    elif profile == PROFILES[5]:
        if c.get('arms', {}).get('v196') != {'fixed_plane_v196_late_action_variation_weight': 1.}:
            raise ValueError('Action-variation repair requires its preregistered reward contract')
        diagnostic = Path(c['exploratory_diagnostic_file'])
        if sha256(diagnostic) != c['exploratory_diagnostic_sha256']:
            raise ValueError('Action-noise diagnostic source evidence hash mismatch')
    elif profile in PROFILES[3:5]:
        arm = 'v194' if profile == PROFILES[3] else 'v195'
        expected = {'fixed_plane_v194_damping_focus_margin_weight': 24.}
        if arm == 'v195':
            expected.update(fixed_plane_v191_absolute_height_multiplier=1.5,
                fixed_plane_v195_late_vxy_excess_weight=6000.)
        if c.get('arms', {}).get(arm) != expected:
            raise ValueError('Focused repair requires its preregistered reward contract')
        diagnostic = Path(c['exploratory_diagnostic_file'])
        if sha256(diagnostic) != c['exploratory_diagnostic_sha256']:
            raise ValueError('Damping-cohort source evidence hash mismatch')
    trials=e.get('trials',[])
    if len(trials)!=2 or {t['ppo_seed'] for t in trials}!=set(c['ppo_seeds']):
        raise ValueError('Two independent passing PPO seeds required')
    audited=[]
    for trial in trials:
        result=assess_trial(trial['candidate_directory'],trial['control_directory'],
            trial['source_directory'],seeds=trial['eval_seeds'])
        if result['profile']!=profile or not result['accepted']:
            raise ValueError('Reward candidate failed preregistered no-regression screen')
        for label,directory in [('candidate',trial['candidate_directory']),('control',trial['control_directory'])]:
            m=json.loads((Path(directory)/'manifest.json').read_text())
            p=pickle.loads(Path(m['checkpoint']).read_bytes())
            validate_probe_payload(p, ppo_seed=trial['ppo_seed'],
                parent_checkpoint=c['parent']['checkpoint'])
        result['s9']=strict_s9(trial['s9_directory'],profile=profile,
            checkpoint_sha=result['candidate_sha256'],seeds=trial['eval_seeds'])
        audited.append(result)
    confirmation=e['fresh_confirmation']
    if confirmation['eval_seeds']!=c['confirmation_eval_seeds']:
        raise ValueError('Fresh confirmation population required')
    fresh=assess_trial(confirmation['candidate_directory'],confirmation['control_directory'],
        confirmation['source_directory'],seeds=confirmation['eval_seeds'])
    if not fresh['accepted'] or fresh['candidate_sha256'] not in {t['candidate_sha256'] for t in audited}:
        raise ValueError('Fresh policy confirmation failed')
    return dict(profile=profile,parent_sha256=PARENT_SHA,trials=audited,fresh_confirmation=fresh)
