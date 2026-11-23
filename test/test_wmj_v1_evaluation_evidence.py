"""Invalid physical labels remain explicit JSON evidence, never fake zeros."""

from pathlib import Path
import sys
import json
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'pingpong_controller/tools/rl_sim'))
from world_model.evaluate_frozen import json_numeric


def test_nonfinite_event_labels_are_explicit_nulls_in_failure_evidence():
    value=json_numeric(np.array([.1,np.nan,np.inf]))
    assert value==[.1,None,None]
    assert json_numeric(np.array(np.nan)) is None
    assert json_numeric(np.array(True)) is True
    json.dumps(value,allow_nan=False)
