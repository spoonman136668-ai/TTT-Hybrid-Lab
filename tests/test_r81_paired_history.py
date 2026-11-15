"""R81 prospective true factual intervention and observable-history negatives.

Tests never generate first-use scored DEV or OOD. The only such source
generation is inside the existing G4 bounded per-seed experiment after all
six original model arms have finished the pre-frozen 512 FIT updates.
"""
import numpy as np
import pytest
import torch

from experiments import r81_paired_history as r81


def test_immutable_prospective_freeze_and_fenced_original_science():
    f=r81.frozen()
    assert r81.r75.git_blob(r81.FREEZE.read_bytes())==r81.FREEZE_BLOB
    assert f["schema"]==r81.SCHEMA
    assert f["parent"]["R80_exact_SHA"]==r81.PARENT
    assert f["parent"]["R80_scored_salts_ineligible"]
    assert f["models"]["seeds"]==list(r81.SEEDS)
    assert f["models"]["arms"]==list(r81.ARMS)
    assert f["models"]["width"]==1024
    assert f["training"]["steps"]==512
    assert f["worlds"]["salts"]==r81.SALT
    assert f["worlds"]["pair_counts"]==r81.PAIRS
    assert f["execution"]["route"].startswith("existing original G4")
    assert not any(f["authority"].values())


def test_two_actual_physical_intervention_rollouts_no_counterfactual_features():
    d=r81.paired_worlds("FIT",count_override=4)
    e=r81.paired_worlds("FIT",count_override=4)
    assert d["pairs"]==4 and d["source_count"]==8
    assert d["input"].shape==(8,64,13)
    assert d["selected_reward"].shape==(8,64,1)
    assert d["input"].dtype==np.float32
    assert d["selected_reward"].dtype==np.float32
    np.testing.assert_array_equal(d["input"],e["input"])
    np.testing.assert_array_equal(d["selected_reward"],e["selected_reward"])
    np.testing.assert_array_equal(d["input"][:,1:,8],d["selected_reward"][:,:-1,0])
    assert np.all(d["input"][:,0,4:12]==0)
    assert np.all(d["input"][:,:,:4].sum(axis=-1)==1)
    p=r81.PIVOT["FIT"]
    assert np.all(d["input"][::2,p,0]==1)
    assert np.all(d["input"][1::2,p,1]==1)
    # Both worlds choose the exact same subsequent factual exploitation actions.
    np.testing.assert_array_equal(d["input"][::2,p+1:p+9,:4],
                                  d["input"][1::2,p+1:p+9,:4])
    assert np.all(d["pair_window"].sum(axis=1)==8)
    assert np.any(d["cold_A_return_mask"])
    # Identical background laws/noise; the actual chosen pivot action causes
    # the later factual reward difference in the paired exploited window.
    target=d["selected_reward"].reshape(4,2,64,1)
    difference=(target[:,0,p+1:p+9]-target[:,1,p+1:p+9])
    assert np.any(np.abs(difference)>0.05)


def test_observable_history_erasure_preserves_current_actions_and_targets():
    d=r81.paired_worlds("FIT",count_override=3)
    blind=r81.blinded_worlds(d)
    assert np.all(blind["input"][:,:,4:12]==0)
    np.testing.assert_array_equal(blind["input"][:,:,:4],d["input"][:,:,:4])
    np.testing.assert_array_equal(blind["input"][:,:,12:],d["input"][:,:,12:])
    np.testing.assert_array_equal(blind["selected_reward"],d["selected_reward"])
    assert np.any(d["input"][:,:,8:12]!=0)
    assert not np.shares_memory(blind["input"],d["input"])


def test_large_max_first_native_identical_start_and_causal_prefix():
    torch.set_num_threads(1)
    m=r81.models(r81.SEEDS[0])
    a=dict(m["NATIVE_FULL_HISTORY"].named_parameters())
    b=dict(m["NATIVE_NO_HISTORY"].named_parameters())
    assert a.keys()==b.keys()
    assert all(torch.equal(a[n],b[n]) for n in a)
    assert sum(p.numel() for p in m["NATIVE_FULL_HISTORY"].parameters())==16818178
    assert sum(p.numel() for p in m["NATIVE_NO_HISTORY"].parameters())==16818178
    source=r81.paired_worlds("FIT",count_override=2)
    blind=r81.blinded_worlds(source)
    for arm,model in m.items():
        err=r81.prefix_negative(model,blind if arm in r81.BLIND else source)
        assert err<=1e-5,(arm,err)
    assert len(m)==6


def test_new_source_isolation_and_unapproved_work_refused(monkeypatch):
    with pytest.raises(PermissionError,match="R81_PROTECTED_OR_SPENT_SOURCE_DENIED"):
        r81.paired_worlds("R80_SCORED_OOD",count_override=2)
    with pytest.raises(ValueError,match="R81_UNFROZEN_SOURCE_PAIR_COUNT"):
        r81.paired_worlds("FIT",count_override=193)
    with pytest.raises(ValueError,match="R81_UNFROZEN_TRAIN_SEED"):
        r81.models(123)
    for key in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                "GITHUB_REPOSITORY","RUNNER_OS","RUNNER_ENVIRONMENT",
                "GITHUB_RUN_ATTEMPT","GITHUB_REF","R81_SEED","EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(key,raising=False)
    with pytest.raises(PermissionError,match="R81_ONLY_FIRST_EXACT_BRANCH_HOSTED_CPU"):
        r81.host_admit(r81.SEEDS[0])


def test_original_paired_bootstrap_and_strict_label():
    f=r81.frozen()
    assert f["frozen_gates"]["bootstrap_lower95_native_full_vs_native_blind_DEV_positive"]
    assert f["frozen_gates"]["both_seeds_every_check"]
    assert f["frozen_gates"]["classification_if_fail"]=="DEV_HISTORY_HYPOTHESIS_NO_GO"
    assert f["frozen_gates"]["independent_scientific_pass"] is False
    a=np.full(64,2.0,dtype=float)
    b=np.full(64,1.0,dtype=float)
    assert r81.bootstrap_lower(a,b)>0
    assert r81.bootstrap_lower(b,a)<0
