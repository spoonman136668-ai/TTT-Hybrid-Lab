"""R75: mathematically equivalent fast weights, proper source and resource gates."""
import hashlib
import json
import os

import numpy as np
import pytest
import torch

from model import TTTLinearMemory
from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r75_cumulative_training as training


@pytest.fixture(autouse=True)
def single_cpu_thread():
    old=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def test_prospectively_frozen_original_maths_and_parent():
    spec=r75.frozen()
    assert spec["schema"]==r75.SCHEMA
    assert r75.git_blob(r75.FREEZE.read_bytes())==r75.FREEZE_BLOB
    assert spec["parent"]["R74_SHA"]==r75.PARENT
    assert spec["parent"]["R74_original_freeze_git_blob"]==r74.FREEZE_BLOB
    assert spec["parent"]["legacy_reference_model_git_blob"]==r74.MODEL_BLOB
    assert spec["exploration_disclosure"]["already_observed_before_freeze"] is True
    assert spec["math"]["strict_lower_triangular"]
    assert spec["qualification"]["full_CPU_speedup_target_min"]==2.0
    assert spec["training_custody"]["new_engine_identity_required"]
    assert all(v is False for v in spec["authority"].values())
    assert r75.git_blob(b"abc\r\n") == r75.git_blob(b"abc\n")
    with pytest.raises(ValueError,match="R75_UNSUPPORTED_TEXT_CR"):
        r75.git_blob(b"abc\raltered")


@pytest.mark.parametrize("width,length",((16,8),(32,32),(128,64)))
def test_cpu_forward_all_parameter_gradients_and_one_adamw_parity(width,length):
    metrics=r75.check_math_parity(width=width,length=length)
    spec=r75.frozen()["qualification"]
    assert metrics["initial_parameter_tensors_identical"]
    assert metrics["parameters"]>0
    assert metrics["output_max_absolute_error"]<=spec["forward_max_absolute_error"]
    assert metrics["gradient_max_absolute_error"]<=spec["all_parameter_gradient_max_absolute_error"]
    assert metrics["reference_and_triangular_complete_real_AdamW_step"]
    assert np.isfinite(metrics["original_loss"])
    assert np.isfinite(metrics["triangular_loss"])


def test_original_no_memory_update_and_strict_causal_prefix_negative():
    torch.manual_seed(997601)
    reference=TTTLinearMemory(32)
    triangular=r75.TriangularFastWeightMemory(32)
    triangular.load_state_dict(reference.state_dict(),strict=True)
    inp=torch.randn(2,32,32)
    with torch.no_grad():
        no_update=reference(inp,update=False)
        no_update_new=triangular(inp,update=False)
        old=reference(inp,update=True)
        new=triangular(inp,update=True)
        altered=inp.clone()
        altered[:,12:]+=13.0
        different=triangular(altered,update=True)
        shortened=triangular(inp[:,:12],update=True)
    torch.testing.assert_close(no_update,no_update_new,rtol=1e-6,atol=1e-6)
    torch.testing.assert_close(old,new,rtol=1e-5,atol=1e-6)
    assert not torch.allclose(new,no_update_new)
    torch.testing.assert_close(new[:,:12],different[:,:12],
                               rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(new[:,:12],shortened,rtol=1e-5,atol=1e-6)
    with pytest.raises(ValueError,match="R75_SHAPE_GUARD"):
        triangular(torch.randn(2,32,15))


def test_full_16_million_parameter_model_forward_backward_parity():
    result=r75.check_math_parity(width=1024,length=128)
    limits=r75.frozen()["qualification"]
    assert result["parameters"]==limits["exact_parameter_count"]==16818178
    assert result["initial_parameter_tensors_identical"]
    assert result["output_max_absolute_error"]<=limits["forward_max_absolute_error"]
    assert result["gradient_max_absolute_error"]<=limits["all_parameter_gradient_max_absolute_error"]
    assert result["reference_and_triangular_complete_real_AdamW_step"]


def test_proper_frozen_selected_fit_uses_real_observed_consequences():
    source=r74.factual_worlds("FIT",count_override=2,sequence=32)
    model=training.optimized_model("HybridPilot_FULL",64,r74.SEEDS[0])
    assert isinstance(model.ttt,r75.TriangularFastWeightMemory)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    fit_loss=training.one_fit_update(model,optimizer,source,np.array([0,1]))
    assert np.isfinite(fit_loss)
    assert fit_loss>=0
    with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
        r74.factual_worlds("R73_HOLDOUT",count_override=2,sequence=32)
    with pytest.raises(ValueError,match="R75_UNREGISTERED_ARCHITECTURE"):
        training.optimized_model("OFFICIAL_UNQUALIFIED_GPU",64,r74.SEEDS[0])


def test_exact_engine_custody_and_final_65536_gate_blocks_early(tmp_path):
    receipt=training.identity("HybridPilot_FULL",1024,128,r74.SEEDS[0])
    assert receipt["engine"]==training.ENGINE
    assert receipt["R75_original_freeze_blob"]==r75.FREEZE_BLOB
    assert receipt["R74_original_freeze_blob"]==r74.FREEZE_BLOB
    assert receipt["reference_model_git_blob"]==r74.MODEL_BLOB
    assert receipt["optimizer"]=="AdamW"
    assert receipt["no_spent_DEV_OOD_used"]
    assert len(receipt["R75_math_engine_source_sha256"])==64
    assert len(receipt["R75_training_driver_source_sha256"])==64
    with pytest.raises(PermissionError,match="R75_UNFROZEN_TARGET_UPDATE_COUNT"):
        training.train_explicit_fit_only("HybridPilot_FULL",1024,128,r74.SEEDS[0],16)
    with pytest.raises(FileNotFoundError,match="R75_INCOMPLETE_65536_TRAINING"):
        training.require_all_control_checkpoints_before_holdout(tmp_path,1024,128)
    with pytest.raises(PermissionError,match="R75_OFFICIAL_GPU_OR_EXTERNAL_FIRSTUSE_DENIED"):
        r75.deny_protected_first_use()



def test_rng_optimizer_and_model_cold_checkpoint_resume_same_next_factual_step(tmp_path):
    seed=r74.SEEDS[0]
    fit=r74.factual_worlds("FIT",count_override=6,sequence=32)
    one=training.optimized_model("HybridPilot_FULL",64,seed)
    opt=torch.optim.AdamW(one.parameters(),lr=.001,weight_decay=.0001)
    rng=np.random.default_rng(seed+24619999)
    first=rng.integers(0,len(fit["input"]),2)
    initial_loss=training.one_fit_update(one,opt,fit,first)
    assert np.isfinite(initial_loss)
    identity=training.identity("HybridPilot_FULL",64,32,seed)
    path=tmp_path/"R75-factual-resume.pt"
    torch.save({"identity":identity,"updates_completed":1,
                "model_state":one.state_dict(),
                "optimizer_state":opt.state_dict(),
                "FIT_sampling_rng":rng.bit_generator.state,
                "FIT_validation_history":[]},path)
    state=torch.load(path,map_location="cpu",weights_only=True)
    assert state["identity"]==identity
    assert state["identity"]["engine"]==training.ENGINE
    two=training.optimized_model("HybridPilot_FULL",64,seed)
    restored=torch.optim.AdamW(two.parameters(),lr=.001,weight_decay=.0001)
    two.load_state_dict(state["model_state"],strict=True)
    restored.load_state_dict(state["optimizer_state"])
    rng2=np.random.default_rng(1)
    rng2.bit_generator.state=state["FIT_sampling_rng"]
    next1=rng.integers(0,len(fit["input"]),2)
    next2=rng2.integers(0,len(fit["input"]),2)
    np.testing.assert_array_equal(next1,next2)
    loss1=training.one_fit_update(one,opt,fit,next1)
    loss2=training.one_fit_update(two,restored,fit,next2)
    assert loss1==pytest.approx(loss2,rel=0,abs=0)
    for x,y in zip(one.parameters(),two.parameters()):
        assert torch.equal(x,y)


def test_complete_scientific_scoring_fails_closed_before_source_creation(tmp_path,monkeypatch):
    original=r74.factual_worlds
    def no_protected_data(panel,*args,**kwargs):
        if panel in ("DEV","OOD"):
            raise AssertionError("R75_HOLDOUT_WAS_READ_BEFORE_FULL_MODEL_CUSTODY")
        return original(panel,*args,**kwargs)
    monkeypatch.setattr(r74,"factual_worlds",no_protected_data)
    with pytest.raises(FileNotFoundError,match="R75_INCOMPLETE_65536_TRAINING"):
        training.score_completed_heldout(1024,128,checkpoint_dir=tmp_path)


def test_source_neutral_cpu_exact_host_profile_and_retained_original_ci_receipt():
    if os.environ.get("GITHUB_ACTIONS")!="true":
        pytest.skip("Original full-width CPU benchmark written only by G4 CI")
    report=r75.profile_two_engines(per_engine_seconds=35)
    assert report["schema"]==r75.SCHEMA
    assert report["freeze_blob"]==r75.FREEZE_BLOB
    assert len(report["trials"])==2
    assert [z["engine"] for z in report["trials"]]==[
        "REFERENCE_SERIAL","TRIANGULAR"]
    assert report["full_65536_training"]=="NOT_EXECUTED"
    assert report["model_learning_science"]=="UNVERIFIED"
    assert report["GPU_first_use"] is False
    assert report["external_science_first_use"] is False
    if report["measured_speedup"] is not None:
        assert report["trials"][0]["receipt"]["parameters"]==16818178
        assert report["trials"][1]["receipt"]["parameters"]==16818178
        assert report["measured_speedup"]>0
    output=r75.ROOT/"local_runs/r75_triangular_cpu_profile.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2,allow_nan=False)
        f.write("\n")
