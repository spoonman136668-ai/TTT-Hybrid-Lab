"""R120 FIT/VALID only scientific source and deterministic clip controls."""
import math
import struct

import numpy as np
import pytest
import torch

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r120_source_custody as custody
from experiments import r120_canonical_clip_portability_diagnostic as r120


def test_r120_prospective_freeze_and_original_science_source_guard(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="af6596e9bd40d8212e186dfe57c06f41339b3024"
    assert f["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_FIT_VALID_ENGINEERING_ONLY"
    assert f["lineage"]["parent_R119_exact_SHA"]=="9d3547baf8fd0914ce33aa0f4000ff217ade84ea"
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert tuple(f["snapshot"]["post_step_indices"])==(1,32,128,256,512)
    assert f["evaluation"]["scientific_pass_from_FIT_VALID"] is False
    assert not any(f["authority"].values())
    assert custody.allow_public("UNIT_FIT")=="UNIT_FIT"
    assert custody.allow_public("UNIT_VALID")=="UNIT_VALID"
    for forbidden in ("UNIT_UNSEEN","original_scientific","protected","R116_consumed"):
        with pytest.raises(PermissionError,match="SOURCE_DENIED"):
            custody.allow_public(forbidden)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_r120_exact_fixed_order_norm_clip_on_synthetic_gradients():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    for p in model.parameters():p.grad=torch.zeros_like(p)
    p=next(model.parameters())
    p.grad.view(-1)[0]=3.
    p.grad.view(-1)[1]=4.
    result=r120.canonical_clip_grad_norm_(model)
    expected=float(np.float32(1./(5.+1e-6)))
    assert result["scaled_gradient_norm_before_clipping"]==5.
    assert result["coefficient"]==expected
    assert result["parameters_processed"]==3409
    assert result["raw_grad_norm_F64_hex"]==struct.pack("<d",5.).hex()
    assert result["clipping_multiplier_exact_FP32_hex"]==struct.pack("<f",expected).hex()
    assert p.grad.view(-1)[0].item()==pytest.approx(3.*expected,abs=1e-6)
    assert p.grad.view(-1)[1].item()==pytest.approx(4.*expected,abs=1e-6)
    with pytest.raises(RuntimeError,match="NONFINITE"):
        p.grad.view(-1)[0]=float("nan")
        r120.canonical_clip_grad_norm_(model)


def test_r120_scalar_compensation_preserves_small_norm_gradient_when_under_1():
    torch.set_num_threads(1)
    model=r116.new_models(custody.SEEDS[0])[r116.NATIVE]
    for p in model.parameters():p.grad=torch.zeros_like(p)
    p=next(model.parameters())
    p.grad.view(-1)[0]=.2
    status=r120.canonical_clip_grad_norm_(model)
    assert status["coefficient"]==1.0
    assert p.grad.view(-1)[0].item()==pytest.approx(.2)


def test_r120_native_full_512_FIT_steps_snapshots_three_scratch_controls():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,_=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=rng.integers(0,256,size=(512,16))
    models=r116.new_models(seed)
    r118.seed_shared_initial_bytes(models[r116.NATIVE],seed,r116.NATIVE)
    receipt=r120.train_original_native_with_stable_clipping(
        models[r116.NATIVE],fit,batches)
    assert receipt["updates"]==512
    assert receipt["model_trainable_parameters"]==3409
    assert set(receipt["native_raw_FP32_hash_by_update"])=={
        str(x) for x in custody.SNAPSHOTS}
    assert receipt["native_raw_FP32_hash_by_update"]["0"]!=receipt[
        "native_raw_FP32_hash_by_update"]["512"]
    assert receipt["first_clip"]["parameters_processed"]==3409
    assert receipt["first_clip"]["coefficient"]>0
    assert receipt["first_clip"]["coefficient"]<=1
    with pytest.raises(PermissionError,match="512_MINIBATCHES"):
        r120.train_original_native_with_stable_clipping(
            models[r116.NATIVE],fit,batches[:5])


def test_r120_original_public_VALID_actual_four_chosen_actions_only():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    layers,sha=r116.r115.quantize_native(model)
    result=r117.public_VALID_trial(seed,384,layers)
    assert len(sha)==64
    assert len(result["next_actions"])==96
    assert set(result["next_actions"])<={12,14}
    assert 0<=result["correct"]<=96
    assert len(result["all_chosen_actions_sha256"])==64
    assert r116.r115.integer_cold_restore(layers)==0
    assert "UNIT_UNSEEN" not in r120.run_seed.__code__.co_consts


def test_r120_original_R116_torch_clip_negative_control_is_public_fit_only():
    torch.set_num_threads(1)
    seed=custody.SEEDS[0]
    fit,_=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    ix=rng.integers(0,256,size=(512,16))[0]
    receipt=r120.first_original_torch_clip_control(seed,fit,ix)
    assert receipt["original_torch_updates"]==1
    assert len(receipt["original_torch_clip_returned_norm_FP32_hex"])==8
    assert len(receipt["original_torch_native_after_first_optimizer_FP32_sha256"])==64
