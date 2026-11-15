"""R80: freeze/source/negative controls, no held-out first-use in tests."""
import numpy as np
import pytest
import torch

from experiments import r80_fresh_router_learning as r80
from experiments import r74_capability_ceiling as r74
from experiments import r79_router_gradient as r79


def test_precommitted_frozen_learnability_protocol_and_original_source():
    f=r80.frozen()
    assert r80.git_blob(r80.FREEZE.read_bytes())==r80.FREEZE_BLOB
    assert f["schema"]==r80.SCHEMA
    assert f["parent"]["R79_exact_SHA"]==r80.PARENT
    assert f["parent"]["original_R78_models_unmodified"]
    assert f["design"]["seeds"]==list(r80.SEEDS)
    assert f["design"]["arms"]==list(r80.ARMS)
    assert f["design"]["width"]==1024
    assert f["design"]["sequence"]==64
    assert f["design"]["train_updates"]==512
    assert f["design"]["source_salts"]==r80.SALT
    assert f["design"]["source_sizes"]==r80.COUNT
    assert f["design"]["holdout_fence"].startswith("Generate DEV and OOD ONLY after")
    assert all(flag is False for flag in f["authority"].values())


def test_real_factual_selected_action_consequence_source_no_leakage():
    assert r80.factual_negative()
    original=r80.fresh_worlds("FIT",count_override=4,sequence=32)
    again=r80.fresh_worlds("FIT",count_override=4,sequence=32)
    assert original["input"].dtype==np.float32
    assert original["selected_reward"].dtype==np.float32
    assert original["input"].shape==(4,32,13)
    np.testing.assert_array_equal(original["input"],again["input"])
    np.testing.assert_array_equal(original["selected_reward"],again["selected_reward"])
    assert np.all(original["input"][:,:,:4].sum(-1)==1)
    assert np.all(original["input"][:,0,4:12]==0)
    np.testing.assert_array_equal(original["input"][:,1:,8],
                                  original["selected_reward"][:,:-1,0])
    assert np.any(original["cold_A_return_mask"])
    assert np.any(original["world_switched"])
    with pytest.raises(PermissionError,match="R80_PROTECTED_OR_SPENT_SOURCE_DENIED"):
        r80.fresh_worlds("R78_TRAINED_ORIGINAL_HOLDOUT",count_override=2)
    with pytest.raises(PermissionError,match="R80_UNFROZEN_SOURCE_COUNT"):
        r80.fresh_worlds("FIT",count_override=385)
    with pytest.raises(ValueError,match="R80_UNFROZEN_SOURCE_SEQUENCE"):
        r80.fresh_worlds("FIT",count_override=2,sequence=17)


def test_full_native_max_width_exact_forward_and_causal_prefix_parity():
    torch.set_num_threads(1)
    models=r80.models_for_seed(r80.SEEDS[0])
    hard=models["HARD_NATIVE"]
    soft=models["SURROGATE_NATIVE"]
    assert hard.depth.__class__!=soft.depth.__class__
    assert isinstance(soft.depth,r79.InactiveSoftSurrogateRouter)
    assert sum(p.numel() for p in hard.parameters())==16818178
    assert sum(p.numel() for p in soft.parameters())==16818178
    hp={n:p.detach() for n,p in hard.named_parameters()}
    sp={n:p.detach() for n,p in soft.named_parameters()}
    assert hp.keys()==sp.keys()
    assert all(torch.equal(hp[n],sp[n]) for n in hp)
    sample=torch.from_numpy(r80.fresh_worlds("FIT",count_override=2,sequence=32)["input"])
    with torch.no_grad():
        a=hard(sample)
        b=soft(sample)
    assert float((a-b).abs().max())<=1e-6
    assert r80.prefix_negative(hard,sample)<=1e-5
    assert r80.prefix_negative(soft,sample)<=1e-5
    assert r80.prefix_negative(models["GRU64"],sample)<=1e-5
    assert r80.prefix_negative(models["MLP64"],sample)<=1e-5


def test_unapproved_training_host_and_candidate_source_rejected(monkeypatch):
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                "GITHUB_REPOSITORY","RUNNER_OS","RUNNER_ENVIRONMENT",
                "GITHUB_RUN_ATTEMPT","GITHUB_REF","R80_SEED",
                "EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="R80_ONLY_FIRST_EXACT_BRANCH_HOSTED_CPU_JOB"):
        r80.host_admit(r80.SEEDS[0])
    with pytest.raises(ValueError,match="R80_UNFROZEN_TRAIN_SEED"):
        r80.models_for_seed(123)


def test_original_scientific_DEV_OOD_remains_protected():
    f=r80.frozen()
    assert f["design"]["acceptance"]["no_scientific_pass"]
    assert f["design"]["acceptance"]["on_all_thresholds"].endswith("NOT_INDEPENDENT_SCIENTIFIC_PASS")
    for panel in ("R74_DEV","R74_OOD","R73_HOLDOUT","R79_SCORED"):
        with pytest.raises(PermissionError,match="R80_PROTECTED_OR_SPENT_SOURCE_DENIED"):
            r80.fresh_worlds(panel,count_override=2)
        with pytest.raises(PermissionError,match="R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED"):
            r74.factual_worlds(panel,count_override=2,sequence=32)
