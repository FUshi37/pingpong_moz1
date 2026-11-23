"""Independent native forks retain complete seeds and explicit pulse effects."""
import ast
import inspect
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))


def test_capture_retains_original_loop_and_real_reset_seed():
    from world_model.audit_free_imagination import audit
    from world_model.corrected_candidate import capture_source
    from world_model.projection_consistency import capture_source as previous
    new=ast.parse(capture_source(inspect.getsource(audit)))
    old=ast.parse(previous(inspect.getsource(audit)))
    collect=lambda t:next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name=='collect')
    assert ast.dump(collect(new))==ast.dump(collect(old))
    body=new.body[0].body
    start=next(i for i,n in enumerate(body) if ast.unparse(n).startswith('native = restore_native_sources('))
    assert ast.unparse(body[start+1])=='native_seed = native'
    assert 'history_seed=history_seed' in ast.unparse(body[-1])


def test_pulses_preserve_counterfactual_pairing_and_fixed_population():
    from world_model.corrected_candidate import pulse_assignments,branch_actions
    assignments=pulse_assignments(32,2026111044)
    assert assignments==pulse_assignments(32,2026111044)
    assert sorted({r['joint'] for r in assignments})==list(range(7))
    base=np.zeros((128,32,7),np.float32);base[:,0]=1.5
    for first in range(0,32,8):
        lanes=np.arange(first,first+8)
        mapping,actions,cases=branch_actions(base,lanes,assignments)
        np.testing.assert_array_equal(mapping,np.tile(lanes,4))
        np.testing.assert_array_equal(actions[:,:8],actions[:,24:])
        np.testing.assert_allclose(actions[:,8:16]+actions[:,16:24],2*actions[:,:8],atol=1e-7)
        np.testing.assert_array_equal(actions[4:,8:16],actions[4:,:8])
        for row in cases:
            if row['lane']==0:assert row['effective_clipped_central_pulse_norm']==0.
            else:assert row['effective_clipped_central_pulse_norm']>0.


def test_repeat_comparison_retains_unresolved_missing_and_wrong_directions():
    from world_model.corrected_candidate import resolved_effects
    def row(v):return dict(release_present=v is not None,impact_id=1 if v is not None else None,
        outgoing_velocity_m_s=None if v is None else [v,0.,0.],analytic_apex_m=None if v is None else [v,0.,0.],actual_apex_m=None)
    native=dict(base=[row(0),row(0),row(0)],plus=[row(.02),row(.001),row(None)],minus=[row(-.02),row(-.001),row(0)])
    world=dict(base=[row(0)]*3,plus=[row(-.02),row(.01),row(.01)],minus=[row(.02),row(-.01),row(-.01)])
    repeat=[row(0),row(.01),row(0)]
    result=resolved_effects(native,world,repeat,[dict(effective_clipped_central_pulse_norm=1.)]*3)
    s=result['outgoing_velocity_m_s']
    assert s['cases']==3 and s['paired_cases']==2 and s['resolved_cases']==1
    assert s['resolved_direction_agreement_fraction']==0.
    assert s['missing_cases']==1 and s['unresolved_cases']==2
