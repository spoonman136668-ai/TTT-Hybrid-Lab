"""R62 pre-frozen portable canonical initializer and R60 FIT-only replay."""
import pytest
import torch
from experiments import p85mr_f8r62_canonical_initialization as r62


def test_exact_prospective_freeze_and_external_firstuse_denied():
    f=r62.locked()
    assert f["stage"]=="FROZEN_BEFORE_R62_CODE_OR_FIRST_CANONICAL_REPLAY"
    assert f["schema"]==r62.SCHEMA
    assert r62.git_blob(r62.FREEZE.read_bytes())==r62.FREEZE_BLOB
    assert f["parent"]["r61_sha"]==r62.PARENT
    assert f["parent"]["r61_freeze_blob"]==r62.r61.FREEZE_BLOB
    assert f["replay"]["original_R60_source_hash"]==r62.SOURCE_HASH
    assert f["replay"]["steps_per_seed"]==1536
    assert f["replay"]["no_evaluator_OOD_data_optimization"]
    assert not f["qualification"]["independent_scientific_pass"]
    assert not any(f["authority"].values())
    assert len(r62.SHAPE)==6
    with pytest.raises(PermissionError,match="EXTERNAL_ORIGINAL_FIRSTUSE_DENIED"):
        r62.reject_external_original_firstuse()


def test_canonical_initialization_exact_raw_byte_hash_and_source_custody():
    r62.locked()
    x,y=r62.r60.cohort("FIT_train")
    assert x.dtype==torch.float64 and x.shape==(4096,10)
    assert y.shape==(4096,)
    assert r62.r61.data_digest(x,y)==r62.SOURCE_HASH
    for seed in r62.SEEDS:
        model=r62.r60.Policy(r62.ARM,seed)
        observed=r62.canonical_init(model,seed)
        assert observed==r62.EXPECTED[seed]
        assert sum(p.numel() for p in model.parameters())==4994
        initial=r62.r61.weights_digest(model)
        assert initial==observed
        for name,p in model.named_parameters():
            assert p.dtype==torch.float64
            assert tuple(p.shape)==r62.SHAPE[name]
        other=r62.r60.Policy(r62.ARM,seed)
        assert r62.canonical_init(other,seed)==observed
        assert all(torch.equal(a,b) for a,b in zip(
            model.parameters(),other.parameters()))
    assert r62.EXPECTED[r62.SEEDS[0]]!=r62.EXPECTED[r62.SEEDS[1]]
    assert r62.portable_value(985201,"head.0.weight",0)==(
        r62.portable_value(985201,"head.0.weight",0))
    with pytest.raises(ValueError,match="UNREGISTERED_INITIALIZATION"):
        r62.portable_value(999999,"head.0.weight",0)


def test_canonical_replay_full_original_FIT_3072_steps_and_17408_actions():
    r=r62.run()
    assert r["schema"]==r62.SCHEMA
    assert r["freeze_blob"]==r62.FREEZE_BLOB
    assert r["parent"]==r62.PARENT
    assert r["classification"]=="ENGINEERING_REPLAY_SPENT_R60_SOURCE_NOT_NEW_SCIENCE"
    assert r["source_scores_OLD_SPENT_FOR_DIAG_ONLY"]
    assert r["reference_previous_R60_science_NO_GO"]
    assert r["source_FIT_sha256"]==r62.SOURCE_HASH
    assert r["canonical_initialization_method"]==r62.SALT
    assert r["total_scored_categorical_actions"]==17408
    assert r["original_FIT_optimizer_steps"]==3072
    assert len(r["trials"])==2
    assert r["CPU_dtype"]=="float64"
    assert r["num_threads"]==1 and not r["mkldnn_enabled"]
    assert not r["independent_scientific_pass"]
    assert not r["Native_specific_gain"]
    assert not r["external_original_firstuse"]
    assert r["deployment"]=="NONE"
    for tr in r["trials"]:
        seed=tr["seed"]
        assert tr["initial_weight_sha256"]==r62.EXPECTED[seed]
        assert tr["cold_restore"]
        assert tr["original_spent_FIT_only_training"]
        assert tr["optimizer_updates"]==1536
        assert tr["productive_gradient_updates"]==1536
        assert len(tr["trace_every_step"])==1536
        assert set(map(int,tr["milestone_diagnostics"]))==set(r62.CHECKPOINTS)
        assert tr["categorical_actions_total"]==8704
        for step,entry in enumerate(tr["trace_every_step"],1):
            assert entry["step"]==step
            for key in ("sample_batch_sha256","preclip_gradient_sha256",
                        "poststep_model_weights_sha256"):
                assert len(entry[key])==64
        for panel,count in [
            ("FIT_train",4096),("FIT_heldout",768),("CAL",768),("OOD",1024)
        ]:
            assert len(tr["scores"][panel]["active"]["choices"])==count
        for name in ("erase_selected_terminal","erase_noncausal_proxy"):
            assert len(tr["scores"]["OOD"][name]["choices"])==1024
    assert r["crossplatform_initial_hashes"]==(
        "UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS")
    assert r["crossplatform_training_weights"]==(
        "UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS")
    assert r["crossplatform_actions"]==(
        "UNVERIFIED_UNTIL_BOTH_ORIGINAL_OS_ARTIFACTS")
    r62.retain_ci_receipt(r)
