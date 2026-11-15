"""R77 exact checkpoint custody and non-replay tests; no long FIT job in pytest."""
import io
import json

import numpy as np
import pytest
import torch

from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as r75
from experiments import r77_verified_resume as r77


def test_only_original_frozen_checkpoint_and_qualified_engine():
    z=r77.frozen()
    assert z["schema"]==r77.SCHEMA
    assert r77.git_blob(r77.FREEZE.read_bytes())==r77.FREEZE_BLOB
    assert z["source"]["original_parent_SHA"]==r77.ORIGINAL_SHA
    assert z["source"]["original_artifact_id"]==r77.ORIGINAL_ARTIFACT_ID
    assert z["source"]["original_G4_run_id"]==43711769367
    assert len(z["source"]["original_artifact_zip_sha256"])==64
    assert z["source"]["original_engine"]==r75.ENGINE
    assert z["execution"]["from_updates"]==1024
    assert z["execution"]["to_updates"]==2048
    assert z["execution"]["qualification_milestone_not_science_checkpoint"]
    assert z["criteria"]["independent_DEV_OOD_not_generated"]
    assert all(value is False for value in z["authority"].values())


def test_protected_runner_and_all_unknown_sources_fail_closed(monkeypatch):
    for name in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                 "RUNNER_OS","RUNNER_ENVIRONMENT","GITHUB_RUN_ATTEMPT",
                 "GITHUB_REPOSITORY","GITHUB_REF","EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(name,raising=False)
    with pytest.raises(PermissionError,match="R77_UNAUTHORIZED_FIRST_ORIGINAL_PR_HOST"):
        r77.admit_first_pr_runner()
    with pytest.raises(PermissionError,match="R77_UNKNOWN_ORIGINAL_ARM"):
        r77.original_name("OFFICIAL_GPU_FIRST_USE")
    for panel in ("R72_HOLDOUT","R73_HOLDOUT","R76_ALREADY_SPENT_DEV","UNAUTHORIZED_EXTERNAL"):
        with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r74.factual_worlds(panel,count_override=1,sequence=32)
    with pytest.raises(ValueError,match="R77_INVALID_TEXT_CARRIAGE_RETURN"):
        r77.git_blob(b"foo\rbar")
    assert r77.git_blob(b"foo\r\n")==r77.git_blob(b"foo\n")


def test_original_R75_optimizer_identity_is_unchanged_except_git_commit():
    for arm in r77.ARMS:
        original=r77.old_identity(arm)
        current=r75.identity(arm,1024,128,997601)
        assert original["checkout_commit_sha"]==r77.ORIGINAL_SHA
        assert original["engine"]==r75.ENGINE
        assert original["R75_training_driver_source_sha256"]==current["R75_training_driver_source_sha256"]
        assert original["R75_math_engine_source_sha256"]==current["R75_math_engine_source_sha256"]
        original["checkout_commit_sha"]=current["checkout_commit_sha"]
        assert original==current


def test_source_artifact_must_be_present_before_any_real_update(tmp_path,monkeypatch):
    monkeypatch.setattr(r77,"SOURCE_API",tmp_path/"api.json")
    monkeypatch.setattr(r77,"SOURCE_DIR",tmp_path/"input")
    monkeypatch.setattr(r77,"admit_first_pr_runner",lambda:"a"*40)
    with pytest.raises(FileNotFoundError,match="R77_SOURCE_ARTIFACT_API_CUSTODY_RECEIPT_MISSING"):
        r77.source_preflight()
    frozen=r77.frozen()
    report={
        "R77_candidate_SHA":"a"*40,
        "original_artifact_digest":"sha256:"+frozen["source"]["original_artifact_zip_sha256"],
        "original_artifact_id":r77.ORIGINAL_ARTIFACT_ID}
    (tmp_path/"api.json").write_text(json.dumps(report))
    with pytest.raises(FileNotFoundError,match="R77_ORIGINAL_RESULT_RECEIPT_MISSING"):
        r77.source_preflight()


def test_selected_consequence_optimizer_and_PCG64_cold_resume_exact():
    """Small reference semantics: same original next selected actions, no replay."""
    torch.set_num_threads(1)
    world=r74.factual_worlds("FIT",count_override=4,sequence=32)
    model=r75.optimized_model("GRU64",64,r74.SEEDS[0])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    random=np.random.default_rng(r74.SEEDS[0]+24619999)
    chosen=random.integers(0,4,2)
    r75.one_fit_update(model,opt,world,chosen)
    buffer=io.BytesIO()
    torch.save({"model":model.state_dict(),"optimizer":opt.state_dict(),
                "rng":random.bit_generator.state},buffer)
    buffer.seek(0)
    old=torch.load(buffer,map_location="cpu",weights_only=True)
    resumed=r75.optimized_model("GRU64",64,r74.SEEDS[0])
    resumed.load_state_dict(old["model"],strict=True)
    opt_resumed=torch.optim.AdamW(resumed.parameters(),lr=.001,weight_decay=.0001)
    opt_resumed.load_state_dict(old["optimizer"])
    restored_rng=np.random.default_rng(0)
    restored_rng.bit_generator.state=old["rng"]
    next_before=random.integers(0,4,2)
    next_after=restored_rng.integers(0,4,2)
    np.testing.assert_array_equal(next_before,next_after)
    loss1=r75.one_fit_update(model,opt,world,next_before)
    loss2=r75.one_fit_update(resumed,opt_resumed,world,next_after)
    assert loss1==pytest.approx(loss2,rel=0,abs=0)
    for before,after in zip(model.parameters(),resumed.parameters()):
        assert torch.equal(before,after)


def test_intermediate_not_frozen_4096_and_protected_science_unverified():
    protocol=r77.frozen()
    assert protocol["execution"]["checkpoint_escrow"].startswith("distinct R77-only")
    assert protocol["criteria"]["on_pass"].startswith("FIT_ONLY_2048")
    assert protocol["criteria"]["scientific_result"].startswith("UNVERIFIED")
    assert r74.TRAIN_STEPS==(1024,4096,16384,65536)
    assert 2048 not in r74.TRAIN_STEPS
