"""F7 training cannot load or score a new external source in hosted CI."""
from __future__ import annotations
import os
from pathlib import Path
import pytest
import torch
from experiments import p85mr_f7_train_fit as f7
from experiments import p85mr_f7_protocol as p7


def test_f7_fit_owner_denial_happens_before_checkpoint_or_source_load():
    with pytest.raises(PermissionError,match="F7_TRAIN_UNQUALIFIED_OWNER_OR_ATTEMPT"):
        f7._owner()
    assert f7.cfg.MODEL_SEEDS==(985201,985203,985207)
    assert f7.cfg.ARMS==p7.ARMS


def test_f7_source_finite_zero_and_immutable_canonical_negatives(tmp_path):
    with pytest.raises(ValueError,match="F7_MISSING_NONZERO_VALID_STEPS"):
        f7._zero_baseline([{"effect":torch.zeros(1,16)}])
    ok=tmp_path/"receipt.json"
    payload={"source":"fixture:only","scientific_pass":False}
    ok.write_bytes(f7._canon(payload))
    assert f7._read_immutable(ok)==payload
    ok.write_text('{"scientific_pass":false, "source":"fixture:only"}')
    with pytest.raises(ValueError,match="F7_TRAIN_NONCANONICAL_SOURCE"):
        f7._read_immutable(ok)


def test_f7_training_source_has_no_external_effect_or_source_import():
    src=Path(f7.__file__).read_text()
    for token in ("env.do(","env.reset(","import causal_gym",
                  "from causal_gym","dispatch_workflow(","requests.get(",
                  "torch.load(","subprocess.run("):
        assert token not in src
    assert f7.MAX_SECONDS==7200
    assert f7.SOURCE_ROOT_ALIAS=="f7-freshfit-1821000000-once-v1"
