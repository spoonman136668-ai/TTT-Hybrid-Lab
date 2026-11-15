"""R82 first genuine action choice: frozen fresh source/negative controls only.

All DEV/OOD worlds remain ungenerated in repository unit tests. The first-use
scoring job is branch-guarded and runs only after full Windows+Ubuntu G4 PASS.
"""
import copy

import numpy as np
import pytest
import torch

from experiments import r82_active_policy as r82


def test_original_freeze_and_nonpromotion_boundaries():
    f=r82.frozen()
    assert r82.r75.git_blob(r82.FREEZE.read_bytes())==r82.FREEZE_BLOB
    assert f["stage"]=="PROSPECTIVE_FROZEN_AFTER_R81_NO_GO_BEFORE_R82_SOURCE_AND_FIRST_USE"
    assert f["parent"]["R81_SHA"]==r82.PARENT
    assert f["parent"]["R81_outcome"]=="DEV_HISTORY_HYPOTHESIS_NO_GO_both_seeds"
    assert f["parent"]["R81_scored_salts_spent"]
    assert f["model"]["arms"]==list(r82.ARMS)
    assert f["physics"]["salts"]==r82.SALT
    assert f["physics"]["world_count"]==r82.COUNT
    assert f["training"]["updates"]==512
    assert f["evaluation"]["thresholds"]["independent_scientific_pass"] is False
    assert not any(f["authority"].values())


def test_fresh_training_source_exposes_only_one_actually_selected_return():
    actual=r82.factual_worlds("FIT",pairs_override=8)
    second=r82.factual_worlds("FIT",pairs_override=8)
    assert actual["returns_only_factual"]
    assert set(actual)=={"panel","count","input","target","pivot",
                         "logged_action","cold","returns_only_factual"}
    assert actual["input"].shape==(8,64,13)
    assert actual["target"].shape==(8,)
    assert actual["input"].dtype==np.float32
    assert set(actual["logged_action"])<=set(r82.CANDIDATES)
    np.testing.assert_array_equal(actual["input"],second["input"])
    np.testing.assert_array_equal(actual["target"],second["target"])
    assert not np.shares_memory(actual["input"],actual["target"])
    for i in range(8):
        pivot=int(actual["pivot"][i])
        choice=int(actual["logged_action"][i])
        assert actual["input"][i,pivot,choice]==1
        assert np.all(actual["input"][i,:pivot+9,:4].sum(-1)==1)
        assert np.all(actual["input"][i,0,4:12]==0)
        original=r82._rollout(r82._world_draw("FIT",i),"FIT",choice)
        assert np.isclose(original["return"],actual["target"][i],rtol=0,atol=1e-6)
        np.testing.assert_allclose(
            actual["input"][i,1:pivot+9,8],
            original["reward"][:pivot+8],rtol=0,atol=0)


def test_factual_physical_action_changes_subsequent_realized_rewards():
    differences=[]
    for i in range(8):
        w=r82._world_draw("FIT",i)
        a=r82._rollout(w,"FIT",0)
        b=r82._rollout(w,"FIT",2)
        pivot=w["pivot"]
        np.testing.assert_array_equal(a["input"][:pivot],b["input"][:pivot])
        assert a["input"][pivot,0]==1
        assert b["input"][pivot,2]==1
        np.testing.assert_array_equal(a["input"][pivot+1:pivot+9,:4],
                                      b["input"][pivot+1:pivot+9,:4])
        assert np.all(a["input"][:pivot+9,:4].sum(-1)==1)
        differences.append(a["return"]-b["return"])
    assert np.any(np.asarray(differences)>1.0)
    assert np.any(np.asarray(differences)<-1.0)


def test_same_initial_huge_native_weights_and_blind_action_prefix():
    torch.set_num_threads(1)
    m=r82.models(r82.SEEDS[0])
    assert len(m)==4
    a=dict(m["NATIVE_FULL_VALUE"].named_parameters())
    b=dict(m["NATIVE_NO_HISTORY_VALUE"].named_parameters())
    assert a.keys()==b.keys()
    assert all(torch.equal(a[n],b[n]) for n in a)
    assert sum(p.numel() for p in m["NATIVE_FULL_VALUE"].parameters())==16818178
    inp=r82.factual_worlds("FIT",pairs_override=2)["input"]
    blinded=r82.blind_input(inp)
    assert np.all(blinded[:,:,4:12]==0)
    np.testing.assert_array_equal(inp[:,:,:4],blinded[:,:,:4])
    np.testing.assert_array_equal(inp[:,:,12],blinded[:,:,12])
    for arm in ("NATIVE_FULL_VALUE","NATIVE_NO_HISTORY_VALUE"):
        model=m[arm]
        model.eval()
        with torch.no_grad():
            length=13
            input_array=blinded if arm=="NATIVE_NO_HISTORY_VALUE" else inp
            x=torch.from_numpy(input_array[:1,:length])
            full=model(x)
            later=x.clone()
            later[:,8:,:]*=-2.7
            changed=model(later)
            short=model(x[:,:8])
            torch.testing.assert_close(full[:,:8],changed[:,:8],atol=1e-6,rtol=0)
            torch.testing.assert_close(full[:,:8],short,atol=1e-6,rtol=0)


def test_scoring_is_realized_paired_return_not_value_prediction(monkeypatch):
    # This control uses only FIT-derived physical worlds, NOT first-use DEV/OOD.
    train=r82.factual_worlds("FIT",pairs_override=4)
    m=r82.models(r82.SEEDS[0])["GRU256_FULL_VALUE"]
    r82.torch.set_num_threads(1)
    assert np.isfinite(r82.pivot_score(m,train,False))
    assert r82.pivot_score(m,train,True)>=0
    assert r82.frozen()["physics"]["worlds"].startswith("Fresh independent stochastic")
    with pytest.raises(PermissionError,match="R82_PROTECTED_OR_SPENT_SOURCE_DENIED"):
        r82.factual_worlds("R81_SCORED_DEV",pairs_override=4)
    with pytest.raises(ValueError,match="R82_WORLD_COUNT_INVALID"):
        r82.factual_worlds("FIT",pairs_override=385)
    with pytest.raises(ValueError,match="R82_UNKNOWN_FROZEN_SEED"):
        r82.models(10)


def test_unapproved_host_and_uncertain_replay_denied(monkeypatch):
    for name in ("GITHUB_ACTIONS","GITHUB_EVENT_NAME","GITHUB_HEAD_REF",
                 "GITHUB_REPOSITORY","RUNNER_OS","RUNNER_ENVIRONMENT",
                 "GITHUB_RUN_ATTEMPT","GITHUB_REF","R82_SEED",
                 "EXPECTED_CANDIDATE_SHA"):
        monkeypatch.delenv(name,raising=False)
    with pytest.raises(PermissionError,match="R82_UNAPPROVED_FIRST_USE_RUNNER"):
        r82.admit(r82.SEEDS[0])
    assert r82.frozen()["execution"]["first_attempt_only"]
    assert r82.frozen()["evaluation"]["thresholds"]["on_fail"]=="DEV_ACTIVE_POLICY_NO_GO"
    assert r82.frozen()["evaluation"]["thresholds"]["on_incomplete"].startswith("INCOMPLETE_CI_OR_RESOURCE")
