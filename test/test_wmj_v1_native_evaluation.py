"""The private native task hook must require explicit substep anchors."""

from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.native_evaluation import augment_source

SOURCE='''def _step_impl(self,state,action):
    def one_substep(index,carry):
        return carry
    (data,*metadata)=jax.lax.fori_loop(0,5,one_substep,(state,))
    metrics={}
    return state,action,action,action,metrics
'''


def test_private_native_task_hook_requires_unique_loop_and_return():
    source=augment_source(SOURCE)
    compile(source,'<private-evaluation-test>','exec')
    assert 'wmj_payload' in source and 'wmj_goals' in source
    with pytest.raises(ValueError,match='one native'):
        augment_source(SOURCE.replace('one_substep,(state,)','different_loop,(state,)'))
    with pytest.raises(ValueError,match='one native'):
        augment_source(SOURCE.replace('return state,action,action,action,metrics','return state'))
