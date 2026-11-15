"""R74: exact frozen source and real CPU reference HybridPilot high-first gates."""
import hashlib
import json
import subprocess

import numpy as np
import pytest
import torch

from experiments import r74_capability_ceiling as r74


def test_original_reference_and_freeze_before_any_probe():
    spec=r74.frozen()
    assert spec["schema"]==r74.SCHEMA
    assert r74.git_blob(r74.FREEZE.read_bytes())==r74.FREEZE_BLOB
    assert r74.git_blob((r74.ROOT/"model.py").read_bytes())==r74.MODEL_BLOB
    assert spec["stage"]=="PRE_FROZEN_BEFORE_SOURCE_OR_PROBES"
    assert spec["capacity"]["widths"]==[1024,768,512,384,256,128,64]
    assert spec["capacity"]["sequence_lengths"]==[128,64,32]
    assert spec["learning"]["cumulative_training_checkpoints"]==[1024,4096,16384,65536]
    assert spec["learning"]["source"]["FIT"]["count"]==512
    assert spec["parent"]["no_spent_R64_to_R73_scored_source"]
    assert all(v is False for v in spec["authority"].values())


def test_real_hybrid_reference_forward_backward_adamw_cpu():
    # Unit-only width32/sequence8 is NEVER called a capacity ceiling.
    result=r74.device_probe(width=32,length=8)
    assert result["type"]=="ACTUAL_HYBRID_REFERENCE_CPU_FORWARD_BACKWARD_UPDATE"
    assert result["device"]=="cpu"
    assert result["width"]==32 and result["sequence"]==8
    assert result["parameters"]>0
    assert result["nonzero_grad_parameters"]>0
    assert np.isfinite(result["final_loss"])
    assert result["final_loss"] >= 0
    assert result["interpretation"]=="ENGINEERING_FEASIBILITY_ONLY_NOT_LEARNED_INTELLIGENCE"
    with pytest.raises(ValueError,match="R74_UNFROZEN_WIDTH"):
        r74.device_probe(width=33,length=8)
    with pytest.raises(ValueError,match="R74_UNFROZEN_SEQUENCE"):
        r74.device_probe(width=32,length=13)


def test_high_first_backoff_does_not_fabricate_untried_tiers(monkeypatch):
    called=[]
    class Fake:
        def __init__(self,code,out,err=""):
            self.returncode=code
            self.stdout=out
            self.stderr=err
    def fake_run(argv,**kw):
        assert argv[:3]==[r74.sys.executable,"-m","experiments.r74_capability_ceiling"]
        width=int(argv[argv.index("--width")+1])
        called.append(width)
        if width==1024:
            raise subprocess.TimeoutExpired(argv,kw["timeout"])
        return Fake(0,json.dumps({"width":width,"sequence":32,
            "peak_process_RSS_MiB":200,"parameters":123456}))
    monkeypatch.setattr(r74.subprocess,"run",fake_run)
    result=r74.high_first_probe(per_seconds=5,overall_seconds=20,
                                max_rss_MiB=1000,
                                levels=(1024,768,512),lengths=(32,))
    assert called==[1024,768]
    assert [x["status"] for x in result["trials"]]==["TIMEOUT","PASS"]
    assert result["highest_feasible_measured"]["width"]==768
    assert result["scientific_learning"]=="NOT_TESTED"
    assert result["GPU_and_official_backend"]=="NOT_TESTED"
    assert result["unprobed_tiers_NOT_QUALIFIED"]
    with pytest.raises(ValueError,match="R74_NOT_FROZEN_HIGH_FIRST_ORDER"):
        r74.high_first_probe(levels=(64,512),lengths=(32,))
    with pytest.raises(PermissionError,match="R74_CPU_RESOURCE_BUDGET_ESCALATION_DENIED"):
        r74.high_first_probe(per_seconds=100)


def test_selected_numeric_factual_source_no_current_outcome_leakage():
    assert r74.factual_causality_negative("FIT",sequence=32)
    fit=r74.factual_worlds("FIT",count_override=6,sequence=32)
    val=r74.factual_worlds("FIT_VALIDATION",count_override=6,sequence=32)
    identical=r74.factual_worlds("FIT",count_override=6,sequence=32)
    assert fit["input"].shape==(6,32,13)
    assert fit["selected_reward"].shape==(6,32,1)
    assert not np.shares_memory(fit["input"],fit["selected_reward"])
    np.testing.assert_array_equal(fit["input"],identical["input"])
    np.testing.assert_array_equal(fit["selected_reward"],identical["selected_reward"])
    assert not np.array_equal(fit["input"],val["input"])
    assert fit["input"].dtype==np.float32
    assert np.all(np.isfinite(fit["selected_reward"]))
    assert np.all(fit["input"][:,:,:4].sum(axis=-1)==1)
    assert np.all(fit["input"][:,0,4:12]==0)
    np.testing.assert_array_equal(fit["input"][:,1:,8],
                                  fit["selected_reward"][:,:-1,0])
    with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
        r74.factual_worlds("R73_HOLDOUT",count_override=2,sequence=32)


def test_full_model_ablations_and_strong_conventional_control_gradients():
    torch.manual_seed(9966)
    x=torch.randn(2,8,13)
    for arm in r74.TRAIN_ARMS:
        model=r74.make_model(arm,32)
        y=model(x)
        assert y.shape==(2,8,1)
        assert bool(torch.isfinite(y).all())
        y.square().mean().backward()
        assert all(p.grad is None or bool(torch.isfinite(p.grad).all())
                   for p in model.parameters())
        assert sum(p.numel() for p in model.parameters())>0
        # Non-anticipation: changed future observations must not affect prefix.
        model.eval()
        with torch.no_grad():
            original=model(x)
            perturbed=x.clone()
            perturbed[:,4:]+=12
            next_output=model(perturbed)
            prefix_only=model(x[:,:4])
        torch.testing.assert_close(original[:,:4],next_output[:,:4],
                                   rtol=1e-4,atol=1e-4)
        torch.testing.assert_close(original[:,:4],prefix_only,
                                   rtol=1e-4,atol=1e-4)


def test_no_spent_source_or_uncapped_training_side_effects():
    with pytest.raises(PermissionError,match="R74_UNFROZEN_SEED_OR_TRAINING_BUDGET"):
        r74.train_bounded("HybridPilot_FULL",64,123,1024)
    with pytest.raises(PermissionError,match="R74_UNFROZEN_SEED_OR_TRAINING_BUDGET"):
        r74.train_bounded("HybridPilot_FULL",64,r74.SEEDS[0],32)
    with pytest.raises(ValueError,match="R74_UNREGISTERED_ARM"):
        r74.make_model("OFFICIAL_GPU_UNQUALIFIED",64)
    with pytest.raises(ValueError,match="R74_UNFROZEN_WIDTH"):
        r74.make_model("HybridPilot_FULL",1000)


def test_highest_first_1024_reference_on_actual_CI_host_with_original_receipt():
    import os
    from pathlib import Path

    if os.environ.get("GITHUB_ACTIONS")!="true":
        pytest.skip("Separate isolated CPU capability probe is qualified on CI host")
    # Actual highest-first frozen tier, NOT a performance projection. The
    # run is bounded by wall time and RSS; missing upper tier pass remains
    # INCOMPLETE rather than a false software/science rejection.
    report=r74.high_first_probe(
        per_seconds=30,overall_seconds=100,max_rss_MiB=3072,
        levels=r74.LEVELS,lengths=(128,))
    assert report["schema"]==r74.SCHEMA
    assert report["freeze_blob"]==r74.FREEZE_BLOB
    assert report["trials"]
    assert report["trials"][0]["width"]==1024
    assert report["trials"][0]["sequence"]==128
    assert report["scientific_learning"]=="NOT_TESTED"
    assert report["GPU_and_official_backend"]=="NOT_TESTED"
    if report["highest_feasible_measured"] is not None:
        assert any(z["status"]=="PASS" for z in report["trials"])
    # This original selected-host receipt is retained by existing G4
    # local_runs/*.json GitHub workflow artifact, with never-overwrite.
    output=r74.ROOT/"local_runs/r74_cpu_highfirst_profiling.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as writer:
        json.dump(report,writer,sort_keys=True,indent=2,allow_nan=False)
        writer.write("\n")


def test_final_scientific_gate_denies_missing_complete_controls_before_OOD(tmp_path):
    with pytest.raises(FileNotFoundError,match="R74_INCOMPLETE_TRAINED_MODEL"):
        r74.final_evaluate_original_checkpoints(tmp_path,width=64,sequence=32)
    with pytest.raises(ValueError,match="R74_UNFROZEN_EVALUATION_SIZE"):
        r74.final_evaluate_original_checkpoints(tmp_path,width=55,sequence=32)
    with pytest.raises(PermissionError,match="R74_UNFROZEN_CHECKPOINT_ID"):
        r74.checkpoint_basename("UNQUALIFIED_ORACLE",64,32,r74.SEEDS[0])
