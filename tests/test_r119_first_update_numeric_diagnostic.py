"""R119 negative controls: only public FIT, no original R116 scientific first use."""
import base64
import hashlib

import numpy as np
import pytest
import torch

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r119_source_custody as custody
from experiments import r119_first_update_numeric_diagnostic as r119


def test_r119_prospective_freeze_and_no_scored_original_source(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="a1529830e8e89f46762c10667abc15e79040f124"
    assert f["stage"]=="PROSPECTIVELY_FROZEN_BEFORE_DIAGNOSTIC_IMPLEMENTATION"
    assert f["lineage"]["parent_R118_SHA"]=="eca962608adfb1f409003f112bf30e2edd59e33b"
    assert tuple(f["trace"]["stages"])==custody.STAGES
    assert len(custody.STAGES)==20
    assert f["evaluation"]["scientific_pass"] is False
    assert f["inputs"]["no_original_R116_UNIT_UNSEEN"]
    assert not any(f["authority"].values())
    assert custody.allow_fit("UNIT_FIT")=="UNIT_FIT"
    for panel in ("UNIT_VALID","UNIT_UNSEEN","original_scientific","protected"):
        with pytest.raises(PermissionError,match="ONLY_PUBLIC_FIT"):
            custody.allow_fit(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r119_canonical_actual_ieee754_FP32_and_int64_stage_capture():
    a=torch.tensor([[1.,-2.],[0.,.5]],dtype=torch.float32)
    item=r119._pack_one(custody.STAGES[0],"public",a)
    assert item["dtype"]=="float32-le"
    assert item["shape"]==[2,2]
    assert item["n_elements"]==4
    assert np.array_equal(
        np.frombuffer(base64.b64decode(item["raw_le_base64"]),dtype="<f4").reshape(2,2),
        a.numpy())
    b=r119._pack_one(custody.STAGES[4],"selected",np.asarray([2,17],dtype=np.int64))
    assert b["dtype"]=="int64-le"
    assert len(item["sha256"])==64 and len(b["sha256"])==64
    assert (r119._stage(custody.STAGES[0],[("public",a)])["sha256"]
            ==r119._stage(custody.STAGES[0],[("public",a)])["sha256"])
    with pytest.raises(PermissionError):
        r119._stage("UNFROZEN_STEAL_HELDOUT",[("private",a)])
    with pytest.raises(RuntimeError,match="NONFINITE"):
        r119._pack_one(custody.STAGES[0],"bad",torch.tensor([float("nan")]))


def test_r119_native_one_original_factual_fit_optimizer_step_unhooked():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    trace=r119.trace_first_update(seed)
    assert trace["number_original_FIT_optimizer_updates"]==1
    assert trace["source_panel"]=="UNIT_FIT"
    assert trace["no_original_scored_or_protected_firstuse"] is True
    assert tuple(stage["stage"] for stage in trace["actual_first_step_trace"])==custody.STAGES
    assert all(len(x["sha256"])==64 for x in trace["actual_first_step_trace"])
    assert all(all(len(p["raw_le_base64"])>0 for p in stage["parts"])
               for stage in trace["actual_first_step_trace"])
    assert all(trace["actual_first_step_trace"][k]["total_elements"]>0
               for k in range(20))
    assert (trace["native_raw_FP32_after_one_optimizer_step"]
            ==r119.original_unhooked_first_update_parameter_hash(seed))
    assert (trace["native_raw_FP32_after_one_optimizer_step"]
            !=trace["initial_native_original_R117_FP32_sha256"])
    assert ([p["name"] for p in trace["actual_first_step_trace"][12]["parts"]]
            ==[name for name,_ in r116.new_models(seed)[r116.NATIVE].named_parameters()])


def test_r119_canonical_first_minibatch_and_native_initial_state_are_repeatable():
    torch.set_num_threads(1)
    seed=custody.SEEDS[1]
    a,ix,_=r119._make_original_public_fit_first_batch(seed)
    b,ix2,_=r119._make_original_public_fit_first_batch(seed)
    assert np.array_equal(ix,ix2)
    assert torch.equal(a[0],b[0])
    assert torch.equal(a[-1],b[-1])
    model=r116.new_models(seed)[r116.NATIVE]
    init,_=r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    fresh=r119.r118._canonical_native_check_against_initial(seed)
    assert init==r118._parameter_bytes(fresh)[0]
    assert "UNIT_UNSEEN" not in r119._make_original_public_fit_first_batch.__code__.co_consts
    with pytest.raises(PermissionError):
        r119._make_original_public_fit_first_batch(1130011)


def test_r119_trace_is_fit_only_and_never_creates_scored_source():
    assert "UNIT_UNSEEN" not in r119.trace_first_update.__code__.co_consts
    assert "UNIT_UNSEEN" not in r119.original_unhooked_first_update_parameter_hash.__code__.co_consts
    assert r119.RECEIPT.name=="r119_first_update_FIT_numeric_boundary.json"
    assert r119.environment()["num_threads"]>=1
