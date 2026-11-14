"""R61 original R60 FIT-only engineering trace admission and deterministic replay."""
import pytest
import torch

from experiments import p85mr_f8r61_fp64_first_divergence as r61


def test_freeze_exact_source_and_no_original_firstuse():
    f=r61.locked()
    assert f["schema"]==r61.SCHEMA
    assert r61.git_blob(r61.FREEZE.read_bytes())==r61.FREEZE_BLOB
    assert f["stage"]=="FROZEN_BEFORE_R61_DIAGNOSTIC_CODE_OR_CROSS_PLATFORM_RUN"
    assert f["parent"]["r60_sha"]==r61.PARENT
    assert f["method"]["trace_at_checkpoints"]==list(r61.CHECKPOINTS)
    assert f["method"]["seeds"]==list(r61.SEEDS)
    assert f["method"]["updates"]==1536
    assert f["decision"]["scientific_pass"] is False
    assert not any(f["authority"].values())
    with pytest.raises(PermissionError,match="ORIGINAL_EXTERNAL_FIRSTUSE_DENIED"):
        r61.reject_original_firstuse()
    with pytest.raises(ValueError,match="UNREGISTERED_SEED"):
        r61.trace_one(99999,torch.zeros((1,10)),torch.zeros(1,dtype=torch.long))


def test_raw_tensor_and_optimizer_hashes_are_deterministic_and_precise():
    a=torch.tensor([1.,-2.,3.],dtype=torch.float64)
    b=torch.tensor([1.,-2.,3.],dtype=torch.float64)
    c=torch.tensor([1.,-2.,3.0000000000000004],dtype=torch.float64)
    assert r61.hash_named([("x",a)])==r61.hash_named([("x",b)])
    assert r61.hash_named([("x",a)])!=r61.hash_named([("x",c)])
    assert r61.hash_named([("x",a)])!=r61.hash_named([("y",a)])
    assert r61.hash_named([("x",a)])!=r61.hash_named([
        ("x",a.float())])
    assert r61.hash_named([("x",a),("y",b)])==r61.hash_named([
        ("y",b),("x",a)])


def test_original_FIT_source_and_r60_MLP_unmodified_inputs():
    r61.locked()
    x,y=r61.r60.cohort("FIT_train")
    assert x.dtype==torch.float64
    assert x.shape==(4096,10)
    assert y.shape==(4096,)
    assert r61.r60.SALT["FIT_train"]==23410000
    assert len(r61.data_digest(x,y))==64
    for seed in r61.SEEDS:
        model=r61.r60.Policy(r61.ARM,seed)
        assert sum(p.numel() for p in model.parameters())==4994
        assert all(p.dtype==torch.float64 for p in model.parameters())
        assert len(r61.weights_digest(model))==64


def test_original_R60_sensitive_MLP_seed_repeatable_within_single_host():
    r61.locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled=False
    x,y=r61.r60.cohort("FIT_train")
    a=r61.trace_one(985207,x,y)
    b=r61.trace_one(985207,x,y)
    assert a["initial_weights_sha256"]==b["initial_weights_sha256"]
    assert a["actual_FIT_updates"]==b["actual_FIT_updates"]==1536
    assert a["productive_gradient_updates"]==b["productive_gradient_updates"]==1536
    for x,y in zip(a["trace_every_step"],b["trace_every_step"]):
        assert x==y
    for step in r61.CHECKPOINTS:
        assert a["milestone_diagnostics"][str(step)]==(
            b["milestone_diagnostics"][str(step)])


def test_full_first_divergence_receipts_all_steps_and_no_science_claim():
    z=r61.run()
    assert z["schema"]==r61.SCHEMA
    assert z["freeze_blob"]==r61.FREEZE_BLOB
    assert z["parent"]==r61.PARENT
    assert z["classification"]=="ENGINEERING_ONLY_ORIGINAL_R60_FIT_SOURCE"
    assert z["source_salt"]==23410000
    assert z["no_heldout_CAL_or_OOD_access"]
    assert z["CPU_dtype"]=="float64"
    assert not z["mkldnn_enabled"]
    assert z["num_threads"]==1
    assert z["actual_original_recipe_FIT_updates"]==3072
    assert z["comparison_status"]==(
        "UNVERIFIED_UNTIL_ORIGINAL_WINDOWS_AND_UBUNTU_TRACES")
    assert not z["independent_scientific_pass"]
    assert z["original_R60_source_or_science_unchanged"]
    assert z["no_new_model_scientific_claim"]
    assert not z["CUDA_first_use"]
    assert z["deployment"]=="NONE"
    assert len(z["seeds"])==2
    assert len(z["training_source_sha256"])==64
    for seed_report in z["seeds"]:
        assert seed_report["seed"] in r61.SEEDS
        assert seed_report["arm"]==r61.ARM
        assert seed_report["actual_FIT_updates"]==1536
        assert seed_report["productive_gradient_updates"]==1536
        assert seed_report["trainable_parameters"]==4994
        assert seed_report["original_R60_FIT_selected_source_only"]
        assert len(seed_report["trace_every_step"])==1536
        assert set(map(int,seed_report["milestone_diagnostics"]))==set(r61.CHECKPOINTS)
        for step,record in enumerate(seed_report["trace_every_step"],1):
            assert record["step"]==step
            assert len(record["selected_batch_sha256"])==64
            assert len(record["raw_gradient_sha256"])==64
            assert len(record["post_step_weights_sha256"])==64
            assert record["raw_gradient_global_l2_norm"]>=0
            assert record["loss_before_update_float64"]>=0
    r61.retain_ci_receipt(z)
