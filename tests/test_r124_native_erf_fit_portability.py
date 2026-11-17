"""R124 FIT/VALID-only native GELU replacement and exact source controls."""
import numpy as np
import pytest
import torch
from torch import nn

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r123_gelu_primitive_diagnostic as r123
from experiments import r124_source_custody as custody
from experiments import r124_native_erf_gelu_fit_portability as r124


def test_r124_preregistered_source_firstuse_fences(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="dbff7d2b2ccef7a0371848f97732bdf34d0921b2"
    assert f["stage"]=="PREIMPLEMENTATION_FROZEN_NATIVE_ONLY_CPU_DEVELOPMENT"
    assert f["lineage"]["parent_R123_SHA"]=="179c3c886da1701fdf0f6c34b16626c82c7245d9"
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert f["source"]["trainable_arms"]==[
        "new_native_erf","original_native_GELU_control",
        "conventional_MLP","conventional_GRU"]
    assert f["source"]["AdamW_steps_per_arm"]==512
    assert f["source"]["no_original_spent_R116_UNSEEN_reuse"] is True
    assert f["evaluation"]["scientific_pass"] is False
    assert not any(f["authority"].values())
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    assert custody.allow("UNIT_VALID")=="UNIT_VALID"
    for panel in ("UNIT_UNSEEN","protected","previous_scored","original_scientific"):
        with pytest.raises(PermissionError,match="SPENT_OR_PROTECTED"):
            custody.allow(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r124_native_only_nontrainable_gelu_replacement_preserves_original_all_weights():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    before,n=r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    params=[(name,tuple(p.shape)) for name,p in model.named_parameters()]
    r124.replace_only_native_gelu_(model)
    assert isinstance(model.scorer[1],r124.NativeExplicitErfGELU)
    assert isinstance(model.scorer[3],r124.NativeExplicitErfGELU)
    assert isinstance(model.scorer[0],nn.Linear)
    assert isinstance(model.scorer[2],nn.Linear)
    assert isinstance(model.scorer[4],nn.Linear)
    assert [(name,tuple(p.shape)) for name,p in model.named_parameters()]==params
    assert n==3409
    with pytest.raises(PermissionError,match="ARCHITECTURE_DRIFT"):
        r124.replace_only_native_gelu_(model)


def test_r124_explicit_erf_module_forward_and_backward_match_R123_original_primitive():
    torch.set_num_threads(1)
    x=torch.tensor([-.8,-.2,0.,.2,.8],dtype=torch.float32)
    for shape in ((5,),):
        t=x.reshape(shape).clone().detach().requires_grad_(True)
        y=r124.NativeExplicitErfGELU()(t)
        (grad,)=torch.autograd.grad(y.sum(),t)
        expected,reference_grad=r123._torch_forward_grad(x,"torch_explicit_erf")
        assert np.array_equal(y.detach().cpu().numpy().view("<u4"),expected.view("<u4"))
        assert np.array_equal(grad.detach().cpu().numpy().view("<u4"),
                              reference_grad.view("<u4"))
    with pytest.raises(RuntimeError,match="CPU_FP32"):
        r124.NativeExplicitErfGELU()(torch.tensor([1.],dtype=torch.float64))


def test_r124_same_public_first_FIT_source_and_R123_actual_module_equivalence():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    measured=r124.original_R123_primitive_behavior_audit(seed)
    assert set(measured["GELU"])=={"GELU1","GELU2"}
    assert all(v["exact_R123_explicit_erf_comparison_pass"]
               for v in measured["GELU"].values())
    assert len(measured["public_first_indices"])==16
    assert all(len(v["erf_output_exact_reference_SHA"])==64
               for v in measured["GELU"].values())


def test_r124_native_full_original_512_update_fit_uses_unchanged_clip():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,_=r116.make_fit(seed)
    batch=np.random.default_rng(np.random.SeedSequence(
        [27601119,seed,116])).integers(0,256,size=(512,16))
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    r124.replace_only_native_gelu_(model)
    receipt=r124._train_new_native(model,fit,batch)
    assert receipt["updates"]==512 and receipt["parameters"]==3409
    assert len(receipt["raw_FP32_sha256_each_step_0_through_512"])==513
    assert receipt["raw_FP32_sha256_each_step_0_through_512"][0]!=(
        receipt["raw_FP32_sha256_each_step_0_through_512"][-1])
    assert receipt["first_step_original_R120_clip"]["parameters_processed"]==3409
    layers,_=r116.r115.quantize_native(model)
    assert r116.r115.integer_cold_restore(layers)==0
    with pytest.raises(PermissionError,match="512_X_16"):
        r124._train_new_native(model,fit,batch[:1])


def test_r124_one_seed_new_and_old_public_valid_actual_chosen_probes_controls():
    torch.set_num_threads(1)
    s=r124.run_seed(custody.SEEDS[0])
    assert s["original_R116_scored_UNIT_UNSEEN_replayed"] is False
    assert s["scientific_pass"] is False
    assert s["new_erf_native_training"]["updates"]==512
    assert s["unchanged_original_GELU_native_control"]["updates"]==512
    assert all(v["updates"]==512
               for v in s["unchanged_conventional_MLP_GRU_training"].values())
    assert s["cold_new_native_integer_mismatch"]==0
    assert s["cold_original_native_integer_mismatch"]==0
    assert len(s["new_native_original_R116_public_VALID"][
        "all_public_fourth_action_choices"])==96
    assert set(s["new_native_original_R116_public_VALID"][
        "all_public_fourth_action_choices"])<={12,14}
    assert 0<=s["new_native_original_R116_public_VALID"]["correct"]<=96
    assert 0<=s["original_GELU_native_public_VALID"]["correct"]<=96
    assert "UNIT_UNSEEN" not in r124.run_seed.__code__.co_consts
