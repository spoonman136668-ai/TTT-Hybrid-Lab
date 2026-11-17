"""R121 only-public-FIT first-step source identity and signed-gradient controls."""
import base64
import numpy as np
import pytest
import torch

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r121_source_custody as custody
from experiments import r121_first_step_repeat_diagnostic as r121


def test_r121_frozen_authority_and_original_spent_source(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="1df5056dab851178d4ae3cd2ad168faec00d1f4d"
    assert f["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_FIT_ONLY_DIAGNOSTIC_NOT_SCIENTIFIC_FIRST_USE"
    assert f["lineage"]["parent_R120_SHA"]=="99fcefefc49d38f199b79ffbee456c084a8d1128"
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert tuple(f["source"]["two_clip_methods"])==custody.MODES
    assert tuple(f["capture"]["frozen_stages"])==custody.STAGES
    assert len(custody.STAGES)==21
    assert f["source"]["number_independent_processes_per_platform"]==2
    assert f["evaluation"]["scientific_pass"] is False
    assert not any(f["authority"].values())
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    for panel in ("UNIT_VALID","UNIT_UNSEEN","original_scientific","protected","R116_spent"):
        with pytest.raises(PermissionError,match="PUBLIC_FIT_ONLY"):
            custody.allow(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r121_canonical_capture_bytes_and_negative_controls():
    f=torch.tensor([[1.,-2.],[0.,.5]],dtype=torch.float32)
    z=r121._capture("all_raw_FP32_named_gradients","first",f)
    assert z["shape"]==[2,2] and z["dtype"]=="float32-le"
    assert np.array_equal(np.frombuffer(base64.b64decode(z["raw_le_base64"]),
                                      dtype="<f4").reshape(2,2),f.numpy())
    t=r121._stage("all_raw_FP32_named_gradients",[("first",f)])
    assert len(t["sha256"])==64
    assert t["sha256"]==r121._stage(
        "all_raw_FP32_named_gradients",[("first",f)])["sha256"]
    assert t["sha256"]!=r121._stage(
        "all_raw_FP32_named_gradients",[("changed",f)])["sha256"]
    with pytest.raises(PermissionError):
        r121._stage("UNFROZEN_original_heldout",[("test",f)])
    with pytest.raises(RuntimeError,match="NONFINITE"):
        r121._capture("all_raw_FP32_named_gradients","bad",
                      torch.tensor([float("nan")]))


def test_r121_original_public_FIT_batch_does_not_reach_UNSEEN():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    x,indices,counts=r121._original_first_batch(seed)
    y,indices2,_=r121._original_first_batch(seed)
    assert np.array_equal(indices,indices2)
    assert torch.equal(x[0],y[0])
    assert torch.equal(x[-1],y[-1])
    assert indices.shape==(16,)
    assert len(x[-1])==256
    assert counts["factual_positive_diagnostic"]>=90
    with pytest.raises(PermissionError,match="UNREGISTERED_FIT_SEED"):
        r121._original_first_batch(1150011)


def test_r121_two_actual_methods_have_identical_raw_gradients_and_no_hook_effect():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    first=r121.trace(seed,custody.MODES[0])
    second=r121.trace(seed,custody.MODES[1])
    assert first["number_AdamW_model_updates"]==1
    assert second["number_AdamW_model_updates"]==1
    assert first["negative_control_pass"] is True
    assert second["negative_control_pass"] is True
    assert first["source"]==second["source"]=="UNIT_FIT"
    assert first["no_original_spent_science"] and second["no_original_spent_science"]
    assert len(first["all_actual_frozen_stages"])==len(custody.STAGES)
    for i in range(13):
        a=first["all_actual_frozen_stages"][i]
        b=second["all_actual_frozen_stages"][i]
        assert a["stage"]==b["stage"]==custody.STAGES[i]
        assert a["sha256"]==b["sha256"],custody.STAGES[i]
    assert first["final_first_step_raw_FP32_SHA256"]==first[
        "negative_control_unhooked_SHA256"]
    assert second["final_first_step_raw_FP32_SHA256"]==second[
        "negative_control_unhooked_SHA256"]
    assert sum(x["n_elements"] for x in first[
        "all_actual_frozen_stages"][12]["parts"])==3409
    assert "UNIT_UNSEEN" not in r121.trace.__code__.co_consts


def test_r121_first_mismatch_finder_reports_full_frozen_stage():
    data={"seeds":{str(s):{m:{"all_actual_frozen_stages":[
        {"stage":name,"sha256":name} for name in custody.STAGES]}
        for m in custody.MODES} for s in custody.SEEDS}}
    assert r121._first_divergence(data,data) is None
    import copy
    altered=copy.deepcopy(data)
    altered["seeds"][str(custody.SEEDS[0])][custody.MODES[0]][
       "all_actual_frozen_stages"][12]["sha256"]="changed"
    result=r121._first_divergence(data,altered)
    assert result=={"seed":custody.SEEDS[0],"mode":custody.MODES[0],
                    "stage_index":12,"stage":"all_raw_FP32_named_gradients"}
