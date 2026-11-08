"""G2 one-use wrapper: pure frozen admission and source/authority boundary."""
import json
from pathlib import Path
import pytest

from scripts.native_p9_gpu_randomized_once import (
    SCHEMA,PARENT_SHA,CHILD_TIMEOUT_SECONDS,MAX_PYTORCH_MIB,
    frozen_admission,guard,Blocked
)


def test_frozen_protocol_admitted_without_gpu_access():
    frozen=frozen_admission()
    assert frozen["rounds"]==24
    assert frozen["wall_speed_ratio_goal"]==.75
    assert frozen["mse_ratio_goal"]==1.01
    assert CHILD_TIMEOUT_SECONDS==150
    assert MAX_PYTORCH_MIB==256
    assert PARENT_SHA=="cda3885e15d3ecee91b6a55931849002543ffb15"


def test_fail_closed_admission_negative():
    with pytest.raises(Blocked,match="NO_PERMISSION"):
        guard(False,"NO_PERMISSION")
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/"experiments/native_p9_gpu_randomized_dev_frozen.json").read_text())
    assert p["science"]["authorized"] is False
    assert p["science"]["world_seeds"]=={}
    assert p["science"]["first_use_consumed"] is False
    assert p["design"]["no_install"] is True
    assert p["design"]["no_training"] is True
    assert p["design"]["no_pipeline_integration"] is True
