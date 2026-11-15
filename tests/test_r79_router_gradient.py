"""R79 prospective exact-hard-forward surrogate gradient mechanism tests."""
import json
import os

import numpy as np
import pytest
import torch

from experiments import r79_router_gradient as r79
from experiments import r74_capability_ceiling as r74


def test_original_prospective_freeze_and_no_source_reuse():
    f=r79.frozen()
    assert f["schema"]==r79.SCHEMA
    assert r79.git_blob(r79.FREEZE.read_bytes())==r79.FREEZE_BLOB
    assert f["parent"]["R78_exact_SHA"]=="5b03322fb3f1e5e03b02c6bbb719ec61f4a577eb"
    assert f["parent"]["source_original_optimizer_counters"]=={
        "full_model":4096,"depth_router":39}
    assert f["scope"]["arms"]==[
        "HARD_REFERENCE","INACTIVE_SOFT_SURROGATE_STRAIGHT_THROUGH"]
    assert f["scope"]["source"].startswith("Fresh independent synthetic")
    assert f["scope"]["no_protected_OOD_firstuse"]
    assert f["qualification"]["actual_learning_score"]=="NOT_TESTED"
    assert not any(f["authority"].values())


def test_hard_forward_parity_and_all_eight_off_route_grads_at_small_width():
    for seed in r79.SEEDS:
        got=r79.probe(width=32,sequence=8,seed=seed)
        assert got["mechanistic_pass"]
        assert all(got["checks"].values())
        assert got["output_absolute_difference"]<=1e-6
        assert got["hard_utilization"]==got["surrogate_utilization"]==0
        assert all(v==0 for v in got["hard_depth_optimizer_step"].values())
        assert all(v==1 for v in got["surrogate_depth_optimizer_step"].values())
        assert len(got["surrogate_depth_grad_abs"])==8
        assert all(v is not None and v>0
                   for v in got["surrogate_depth_grad_abs"].values())
    with pytest.raises(ValueError,match="R79_NONFROZEN_SEED"):
        r79.probe(32,8,12)
    with pytest.raises(ValueError,match="R79_NONFROZEN_PROBE_DIMENSIONS"):
        r79.probe(128,16,r79.SEEDS[0])


def test_exact_same_hard_inference_prefix_after_future_edit():
    torch.set_num_threads(1)
    for width in (32,128):
        hard=r79.make_model(width,r79.SEEDS[0],False)
        surrogate=r79.make_model(width,r79.SEEDS[0],True)
        first={k:v.detach().clone() for k,v in hard.named_parameters()}
        second={k:v.detach().clone() for k,v in surrogate.named_parameters()}
        assert first.keys()==second.keys()
        assert all(torch.equal(first[k],second[k]) for k in first)
        torch.manual_seed(998811)
        observed=torch.randn(2,32,13)
        changed=observed.clone()
        changed[:,16:]+=20.0
        hard.eval()
        surrogate.eval()
        with torch.no_grad():
            reference=hard(observed)
            straight=surrogate(observed)
            after=surrogate(changed)
            prefix_only=surrogate(observed[:,:16])
        torch.testing.assert_close(reference,straight,rtol=0,atol=1e-6)
        torch.testing.assert_close(straight[:,:16],after[:,:16],rtol=0,atol=1e-6)
        torch.testing.assert_close(straight[:,:16],prefix_only,rtol=0,atol=1e-6)
    with pytest.raises(ValueError,match="R79_UNFROZEN_DIAGNOSTIC_MODEL"):
        r79.make_model(400,r79.SEEDS[0],True)


def test_existing_frozen_source_is_not_scored_or_replayed():
    assert r74.frozen()["learning"]["cumulative_training_checkpoints"]==[1024,4096,16384,65536]
    for panel in ("R78_PROTECTED_HOLDOUT","R79_DEV","R73_EXPIRED"):
        with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r74.factual_worlds(panel,count_override=1,sequence=32)
    with pytest.raises(PermissionError,match="R79_OFFICIAL_GPU_OR_EXTERNAL_FIRSTUSE_DENIED"):
        r79.deny_external_first_use()


def test_actual_original_full_width_cpu_1024_gradient_gate():
    if os.getenv("GITHUB_ACTIONS")!="true":
        pytest.skip("Full width original two-seed mechanistic gate only in original G4")
    receipt=r79.original_ci_receipt()
    assert receipt["classification"]=="ENGINEERING_CAUSAL_ROUTER_GRADIENT_RECOVERY_ONLY"
    assert len(receipt["observed"])==2
    assert [v["seed"] for v in receipt["observed"]]==list(r79.SEEDS)
    for v in receipt["observed"]:
        assert v["full_model_parameters"]==16818178
        assert v["width"]==1024 and v["sequence"]==128
        assert v["mechanistic_pass"]
    assert receipt["independent_scientific_pass"] is False
    assert receipt["R78_original_weights_modified"] is False
    assert receipt["DEV_OOD_access"] is False
