"""R123 FIT-only native GELU original forward/derivative source and candidate tests."""
from decimal import Decimal
import base64
import copy

import numpy as np
import pytest
import torch

from experiments import r123_source_custody as custody
from experiments import r123_gelu_primitive_diagnostic as r123


def test_r123_preimplementation_protocol_and_spent_original_science_denied(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="c615affeb16477fa62e2ca287891f87d4e289226"
    assert f["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_PUBLIC_FIT_ONLY"
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert tuple(x["name"] for x in f["candidates"])==custody.CANDIDATES
    assert f["capture"]["decimal_precision"]==70
    assert f["capture"]["decimal_erf_taylor_terms"]==96
    assert f["evaluation"]["scientific_pass"] is False
    assert not any(f["authority"].values())
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    for panel in ("UNIT_VALID","UNIT_UNSEEN","protected","original_science"):
        with pytest.raises(PermissionError,match="SPENT_SCIENCE"):
            custody.allow(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r123_decimal_series_forward_and_analytic_gradient_reference():
    a=np.asarray([-2.,-1.,-.5,0.,.5,1.,2.],dtype="<f4")
    forward,grad=r123.decimal_forward_gradient(a)
    assert forward.shape==grad.shape==a.shape
    assert forward[3]==0.0 and grad[3]==0.5
    expected,expected_grad=r123._torch_forward_grad(
        torch.from_numpy(a),"torch_original_GELU_none")
    assert r123._error(forward,expected)["max_abs"]<1e-5
    assert r123._error(grad,expected_grad)["max_abs"]<1e-5
    # Not even one arbitrary source input beyond preregistered series domain.
    with pytest.raises(RuntimeError,match="OUTSIDE_PRECOMMITTED"):
        r123.decimal_forward_gradient(np.asarray([6.0],dtype=np.float32))


def test_r123_fp32_raw_provenance_and_signed_ULP_comparison():
    x=np.asarray([-1.0,0.0,0.5],dtype="<f4")
    capture=r123._tensor_bytes("public",x)
    assert capture["shape"]==[3] and capture["n_elements"]==3
    assert len(capture["float32_le_SHA256"])==64
    assert np.array_equal(np.frombuffer(base64.b64decode(
        capture["float32_le_base64"]),dtype="<f4"),x)
    assert capture==r123._tensor_bytes("public",x)
    assert capture["float32_le_SHA256"]!=r123._tensor_bytes(
        "modified",x)["float32_le_SHA256"]
    y=x.copy()
    y[0]=np.nextafter(np.float32(-1),np.float32(-2))
    result=r123._error(y,x)
    assert result["max_ULP"]==1
    assert result["different_FP32_elements"]==1
    assert result["first_mismatch_flat_index"]==0
    with pytest.raises(RuntimeError,match="NONFINITE"):
        r123._tensor_bytes("nonfinite",np.asarray([float("nan")],dtype=np.float32))


def test_r123_torch_alternatives_return_actual_finite_gradients():
    t=torch.tensor([-1.,0.,1.],dtype=torch.float32)
    for name in custody.CANDIDATES[:3]:
        f,g=r123._torch_forward_grad(t,name)
        assert f.shape==g.shape==(3,)
        assert f.dtype==g.dtype==np.float32
        assert np.isfinite(f).all() and np.isfinite(g).all()
        assert abs(float(g[1])-.5)<1e-6
    with pytest.raises(PermissionError,match="NOT_FROZEN"):
        r123._torch_forward_grad(t,"unregistered")


def test_r123_actual_original_native_first_FIT_batch_layer_inputs_and_outputs():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    record=r123.one_seed(seed)
    assert record["seed"]==seed
    assert record["original_native_model_parameter_updates"]==0
    assert record["original_R116_spent_UNSEEN_reused"] is False
    assert set(record["candidate_GELU_primitive_panels"])==set(custody.LAYERS)
    assert sum(record["candidate_GELU_primitive_panels"][l][
        "actual_GELU_input"]["n_elements"] for l in custody.LAYERS)==2304
    for layer in custody.LAYERS:
        entry=record["candidate_GELU_primitive_panels"][layer]
        original=entry["candidates"]["torch_original_GELU_none"]
        assert entry["actual_native_original_forward"]["float32_le_base64"]==(
            original["forward"]["float32_le_base64"])
        assert original["forward_error_from_original_GELU"]["max_ULP"]==0
        assert original["gradient_error_from_original_GELU"]["max_ULP"]==0
        assert set(entry["candidates"])==set(custody.CANDIDATES)
        assert len(entry["actual_GELU_input"]["float32_le_SHA256"])==64


def test_r123_sourced_mismatch_finder_is_specific_to_seed_layer_and_candidate():
    template={"FIT_source_SHA256":{"n":"source"},
              "initial_native_FP32_sha256":"initial",
              "candidate_GELU_primitive_panels":{}}
    for layer in custody.LAYERS:
        template["candidate_GELU_primitive_panels"][layer]={
           "actual_GELU_input":{"float32_le_SHA256":"input"},
           "actual_native_original_forward":{"float32_le_SHA256":"observed"},
           "candidates":{c:{"forward":{"float32_le_SHA256":"forward"},
                            "input_gradient":{"float32_le_SHA256":"gradient"}}
                         for c in custody.CANDIDATES}}
    left={"seeds":{str(seed):copy.deepcopy(template) for seed in custody.SEEDS}}
    right=copy.deepcopy(left)
    assert r123._first_worker_difference(left,right) is None
    right["seeds"][str(custody.SEEDS[1])][
        "candidate_GELU_primitive_panels"]["GELU2"][
        "candidates"]["decimal_erf_series_96"][
        "input_gradient"]["float32_le_SHA256"]="changed"
    assert r123._first_worker_difference(left,right)=={
        "seed":custody.SEEDS[1],"layer":"GELU2",
        "candidate":"decimal_erf_series_96","stage":"input_gradient"}
