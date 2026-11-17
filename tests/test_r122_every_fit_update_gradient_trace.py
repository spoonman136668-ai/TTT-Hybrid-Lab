"""R122 exact public FIT full-update native provenance negative controls."""
import copy
import numpy as np
import pytest
import torch

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r122_source_custody as custody
from experiments import r122_every_fit_update_gradient_trace as r122


def test_r122_preimplementation_frozen_and_original_science_fence(monkeypatch):
    freeze=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="ebf335b8873cee4109b34644aa3498eeb7791557"
    assert freeze["stage"]=="PREIMPLEMENTATION_FROZEN_ENGINEERING_FIT_ONLY"
    assert freeze["lineage"]["parent_R121_source_sha"]=="7a34405e687878ac05a4489d1603beacb1b04e11"
    assert tuple(freeze["inputs"]["seed_ids"])==custody.SEEDS
    assert freeze["inputs"]["num_workers_each_OS"]==2
    assert freeze["inputs"]["model_updates_per_seed_each_worker"]==512
    assert tuple(freeze["capture"]["ordered_step_stages"])==custody.STAGES
    assert freeze["acceptance"]["scientific_pass"] is False
    assert not any(freeze["authority"].values())
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    for panel in ("UNIT_VALID","UNIT_UNSEEN","original_scientific","protected_science"):
        with pytest.raises(PermissionError,match="ONLY_ORIGINAL_PUBLIC_FIT"):
            custody.allow(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r122_actual_FP32_and_F64_digest_preserves_name_bits_shape():
    f=torch.tensor([1.,-2.,.125],dtype=torch.float32)
    x=r122._hash_actual("raw_named_FP32_gradients",[("scorer.0.weight",f)])
    assert x["stage"]=="raw_named_FP32_gradients"
    assert x["parts"][0]["shape"]==[3]
    assert x["parts"][0]["dtype"]=="float32"
    assert x==r122._hash_actual("raw_named_FP32_gradients",[("scorer.0.weight",f)])
    assert x["sha256"]!=r122._hash_actual(
        "raw_named_FP32_gradients",[("scorer.0.bias",f)])["sha256"]
    assert x["sha256"]!=r122._hash_actual(
        "clipped_named_FP32_gradients",[("scorer.0.weight",f)])["sha256"]
    n=r122._hash_actual("canonical_sqrt_norm_F64",[("x",np.asarray(1.0,dtype="<f8"))])
    assert n["parts"][0]["dtype"]=="float64"
    with pytest.raises(RuntimeError,match="NONFROZEN_CAPTURE_DTYPE"):
        r122._hash_actual("x",[("test",np.asarray([1],dtype=np.int32))])
    with pytest.raises(RuntimeError,match="NONFINITE"):
        r122._hash_actual("x",[("test",np.asarray([np.nan],dtype=np.float32))])


def test_r122_first_FIT_source_and_batch_cannot_access_original_unseen():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,batches,source,counts=r122._fit_and_batches(seed)
    fit2,batches2,source2,counts2=r122._fit_and_batches(seed)
    assert batches.shape==(512,16)
    assert np.array_equal(batches,batches2)
    assert source==source2
    assert counts==counts2
    assert len(source)==5
    assert len(fit[-1])==256
    assert sum(x.numel() for x in [fit[-1]])==512
    with pytest.raises(PermissionError,match="NOT_FROZEN_FIT_SEED"):
        r122._fit_and_batches(1150011)
    assert "UNIT_UNSEEN" not in r122.trace_native_full_fit.__code__.co_consts


def test_r122_full_512_step_FIT_trace_and_unhooked_step_negative():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    result=r122.trace_native_full_fit(seed)
    assert result["seed"]==seed
    assert result["number_updates"]==512
    assert len(result["steps"])==512
    assert result["first_step_negative_control_pass"] is True
    assert result["first_step_native_FP32_SHA256"]==result[
        "first_step_unhooked_negative_control_SHA256"]
    assert len(result["initial_native_FP32_parameter_SHA256"])==64
    assert len(result["last_step_native_FP32_SHA256"])==64
    assert result["last_step_native_FP32_SHA256"]!=result[
        "initial_native_FP32_parameter_SHA256"]
    assert result["original_spent_R116_UNIT_UNSEEN_reused"] is False
    for i,step in enumerate(result["steps"],start=1):
        assert step["update"]==i
        assert tuple(x["stage"] for x in step["stages"])==custody.STAGES
        assert all(len(x["sha256"])==64 for x in step["stages"])
    assert r122._first_difference(result,result) is None


def test_r122_first_change_reports_precise_step_stage_and_named_parameter():
    seed=custody.SEEDS[0]
    # This control does not create scientific worlds; verifies ordered
    # comparison against synthetic full length 512-stage identities.
    template=[{"stage":stage,"sha256":str(j),
               "parts":[{"name":"scorer.0.weight","sha256":"a"}]}
              for j,stage in enumerate(custody.STAGES)]
    original={"FIT_source_digest":{"n":"x"},
              "initial_native_FP32_parameter_SHA256":"init",
              "steps":[{"update":i,"stages":copy.deepcopy(template)}
                       for i in range(1,custody.UPDATES+1)]}
    altered=copy.deepcopy(original)
    altered["steps"][156]["stages"][8]["sha256"]="changed"
    altered["steps"][156]["stages"][8]["parts"][0]["sha256"]="b"
    assert r122._first_difference(original,altered)=={
        "step":157,"stage_index":8,"stage":"raw_named_FP32_gradients",
        "differing_parameter_or_tensor_names":["scorer.0.weight"]}
    assert r122._first_difference(original,original) is None
    upstream=copy.deepcopy(original)
    upstream["FIT_source_digest"]["n"]="altered"
    assert r122._first_difference(original,upstream)=={
        "stage":"public_FIT_source","step":0}
