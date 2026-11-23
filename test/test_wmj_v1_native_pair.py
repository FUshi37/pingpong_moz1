"""Native pairing uses physical scales without accepting state-clock changes."""

from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.native_pair import assess_native_pair


def fixture():
    Data=namedtuple('Data','qpos qvel')
    State=namedtuple('State','data arm_cmd_q prev_action step_count')
    state=State(Data(np.zeros((1,10)),np.zeros((1,10))),np.zeros((1,7)),np.zeros((1,7)),np.zeros(1,int))
    return state,SimpleNamespace(ball_vadr=7)


def test_integrated_submicrometer_roundoff_is_recorded_without_bitwise_gate():
    left,env=fixture()
    right=left._replace(data=left.data._replace(qpos=left.data.qpos+9e-7,
        qvel=np.array([[5.3e-4]+[0.]*9])))
    report=assess_native_pair(left,right,np.array([True]),env)
    assert report['pass_'] and not report['strict_velocity_smoke_pass']


def test_ball_velocity_or_discrete_state_changes_cannot_hide_in_arm_tolerance():
    left,env=fixture()
    qvel=left.data.qvel.copy();qvel[0,7]=5e-4
    assert not assess_native_pair(left,left._replace(data=left.data._replace(qvel=qvel)),np.array([True]),env)['pass_']
    assert not assess_native_pair(left,left._replace(step_count=np.ones(1,int)),np.array([True]),env)['pass_']


def test_nonfinite_pair_remains_json_serializable_negative_evidence():
    import json
    left,env=fixture()
    right=left._replace(data=left.data._replace(qvel=left.data.qvel+np.nan))
    report=assess_native_pair(left,right,np.array([True]),env)
    assert not report['pass_']
    json.dumps(report,allow_nan=False)
