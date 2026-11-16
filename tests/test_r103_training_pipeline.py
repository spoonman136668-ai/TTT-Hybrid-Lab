"""R103 original science denied; engineering-only causal pipeline contract."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r103_relational_binding_diagnostic as r103
from experiments import r103_training_pipeline as pipe


def test_frozen_pipeline_all_eleven_arms_and_no_first_use():
    f=pipe.protocol()
    assert pipe.ARMS==r103.ARMS
    assert pipe.SEEDS==r103.SEEDS
    assert pipe.STEPS==3072 and pipe.BATCH==16 and pipe.SUPPORT==16
    assert len(pipe.SCORED)==5
    assert f["training"]["training_all_11_fully_trained_required_before_DEV"]
    assert not f["authority"]["protected_scored_first_use"]
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        pipe.original_ci(seed=r103.SEEDS[0])


@pytest.mark.parametrize("panel",tuple(r103.PANEL_COUNTS))
def test_all_original_fit_valid_dev_scored_paths_denied_before_world_construction(panel):
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.episode(panel,r103.SEEDS[0],0,"a"*40,paired=(panel in pipe.SCORED))
    with pytest.raises(PermissionError,match="R103_PROTECTED_ORIGINAL_PHYSICAL_SOURCE_NOT_ADMITTED"):
        pipe.source(panel,r103.SEEDS[0],"a"*40)


def test_unregistered_source_or_unfrozen_train_counterfactual_denied():
    with pytest.raises(PermissionError,match="R103_FORBIDDEN_TRAIN_COUNTERFACTUAL_QUERY"):
        pipe.episode("FIT",r103.SEEDS[0],0,paired=True)
    with pytest.raises(PermissionError,match="R103_UNREGISTERED_PROSPECTIVE_EPISODE"):
        pipe.episode("R102_DEV_UNKNOWN",r103.SEEDS[0],0)
    with pytest.raises(PermissionError,match="R103_PROTECTED_SOURCE_SCOPE_NOT_REGISTERED"):
        pipe.source("R102_DEV_UNKNOWN",r103.SEEDS[0])


def unit_two_sample_score_set():
    # Independent fresh R103 UNIT_ONLY fixtures. Never a reserved dataset.
    fixtures=[r103.unit_fixture(9),r103.unit_fixture(10)]
    p=[x["public"] for x in fixtures]
    e=[x["evaluator_only"] for x in fixtures]
    shuffled=[pipe._observed_shuffled_outcomes(
        v["support_slots"],v["support_actions"],v["support_rewards"],
        "FIT",r103.SEEDS[0],i)[0] for i,v in enumerate(p)]
    return {
        "support_slots":np.stack([v["support_slots"] for v in p]),
        "support_rewards":np.stack([v["support_rewards"] for v in p]),
        "support_rewards_shuffled":np.stack(shuffled),
        "support_actions":np.stack([v["support_actions"] for v in p]),
        "query0":np.stack([v["query0"] for v in p]),
        "query2":np.stack([v["query2"] for v in p]),
        "reward0":np.asarray([v["return0"] for v in e],np.float32),
        "reward2":np.asarray([v["return2"] for v in e],np.float32),
        "private_law":np.asarray([v["private_law"] for v in e],np.int8),
        "cold":np.array([False,True],dtype=np.bool_),
    }


def test_prospective_stratified_shuffle_is_public_and_reproducible():
    p=r103.unit_fixture(9)["public"]
    shuffled,indices=pipe._observed_shuffled_outcomes(
        p["support_slots"],p["support_actions"],p["support_rewards"],
        "FIT",r103.SEEDS[0],0)
    again,indices2=pipe._observed_shuffled_outcomes(
        p["support_slots"],p["support_actions"],p["support_rewards"],
        "FIT",r103.SEEDS[0],0)
    np.testing.assert_array_equal(indices,indices2)
    np.testing.assert_array_equal(shuffled,again)
    assert not np.any(indices==np.arange(16))
    np.testing.assert_array_equal(p["support_actions"][indices],
                                  p["support_actions"])
    np.testing.assert_array_equal(p["support_slots"][indices,-1,12],
                                  p["support_slots"][:,-1,12])


def test_all_11_arms_produce_unit_only_evaluator_wiring_and_frozen_metrics():
    torch.set_num_threads(1)
    d=unit_two_sample_score_set()
    models=r103.make_models(r103.SEEDS[0])
    result=pipe.score(models,"DEV_UNKNOWN",d)
    assert set(result["metrics"])==set(r103.ARMS)
    assert result["physical_episodes"]==2
    assert result["raw_reward"][r103.NATIVE].shape==(2,)
    assert 0<=result["role_relabel_action_agreement"]<=1
    measures=pipe.measures(r103.SEEDS[0],
                           {panel:result for panel in pipe.SCORED})
    assert set(measures["measured"])==set(r103.frozen()["evaluation"]["checks"])
    assert measures["strongest_nonengineered"] in (
        r103.GRU,r103.ROLE_SET,r103.MLP,r103.STATIC)
    assert isinstance(measures["all_pass"],bool)
    # Synthetic UNIT_ONLY is never eligible to be reported as science.


def test_train_fit_denies_mini_fixture_and_private_counterfactual_leak():
    data=unit_two_sample_score_set()
    fit=dict(data);fit["query"]=fit.pop("query0");fit["target"]=fit.pop("reward0")
    for key in ("query2","reward2","private_law","cold"):
        fit.pop(key)
    model=r103.make_models(r103.SEEDS[0])[r103.GRU]
    ix=np.zeros((3072,16),np.int64)
    with pytest.raises(RuntimeError,match="R103_FULL_FIT_COHORT_SIZE_INVALID"):
        pipe.fit_arm(model,r103.GRU,fit,fit,ix)
    fit["private_law"]=np.ones(2,dtype=np.int8)
    with pytest.raises(PermissionError,match="R103_TRAINING_INVALID_OR_PRIVILEGED_INPUT"):
        pipe.fit_arm(model,r103.GRU,fit,fit,ix)


def test_bootstrap_2000_draws_repeatable_and_prior_seed_unadmitted():
    a=np.asarray([1.,2.,-1.,3.])
    b=np.asarray([0.,0.,0.,0.])
    assert pipe.bootstrap_lower(a,b,r103.SEEDS[0])==pipe.bootstrap_lower(
        a,b,r103.SEEDS[0])
    with pytest.raises(ValueError,match="R103_BOOTSTRAP_NONMATCHED_COHORT"):
        pipe.bootstrap_lower(a,b,1002711)


def test_unit_query_role_relabel_is_public_only():
    p=r103.unit_fixture(10)["public"]
    sample=p["support_slots"][None]
    q=p["query0"][None]
    moved,query=pipe.permute_public_roles(sample,q)
    assert moved.shape==sample.shape and query.shape==q.shape
    np.testing.assert_array_equal(query[0,-1],q[0,-1])
    np.testing.assert_array_equal(
        np.sort(moved[:,:,:4,8],axis=2),
        np.sort(sample[:,:,:4,8],axis=2))



def test_all_eleven_models_take_one_actual_UNIT_ONLY_factual_optimizer_step():
    """Engineering-only DEV smoke; NO FIT/DEV original samples or scored claim."""
    torch.set_num_threads(1)
    f=r103.unit_fixture(9)
    p=f["public"]
    x=torch.from_numpy(p["support_slots"][None])
    r=torch.from_numpy(p["support_rewards"][None])
    q=torch.from_numpy(p["query0"][None])
    # The physical CALIBRATE chosen action supplies the only target.
    target=torch.tensor([f["evaluator_only"]["return0"]],dtype=torch.float32)
    models=r103.make_models(r103.SEEDS[0])
    for arm,model in models.items():
        opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        out=model(x,r,q)
        assert out.shape==(1,) and bool(torch.isfinite(out).all()),arm
        loss=torch.nn.functional.mse_loss(out,target)
        assert bool(torch.isfinite(loss)),arm
        opt.zero_grad(set_to_none=True)
        loss.backward()
        assert any(v.grad is not None and bool(torch.isfinite(v.grad).all())
                   for v in model.parameters()),arm
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        with torch.no_grad():
            after=model(x,r,q)
            assert bool(torch.isfinite(after).all()),arm



def test_original_still_rejects_direct_science_without_manifest(monkeypatch):
    monkeypatch.delenv("EXPECTED_CANDIDATE_SHA",raising=False)
    with pytest.raises(PermissionError,
                       match="R103_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        pipe.original_ci(r103.SEEDS[0])



def test_frozen_budget_checks_unit_only_without_accessing_original_sources(monkeypatch):
    """Expired budget blocks source/training/evaluation BEFORE any physics."""
    from experiments import r103_training_pipeline as pipe
    from experiments import r103_relational_binding_diagnostic as r103
    import torch
    torch.set_num_threads(1)
    never=lambda *_a,**_kw: (_ for _ in ()).throw(
        AssertionError("R103_PROTECTED_SOURCE_MUST_NOT_OPEN"))
    monkeypatch.setattr(r103,"_world",never)
    seed=r103.SEEDS[0]
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe.source("FIT",seed,deadline=-1)
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe.episode("FIT",seed,0,deadline=-1)
    arm=r103.GRU
    from experiments import r102_contradiction_evidence_diagnostic as r102
    model=r102.ConventionalEvidenceValue("gru")
    # Training must halt BEFORE any call to the optimizer/model even if its
    # input claims a full cohort count. All examples are unit-only placeholders.
    indices=np.zeros((pipe.STEPS,pipe.BATCH),dtype=np.int64)
    fake_fit={"target":np.zeros(1024,dtype=np.float32)}
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe.fit_arm(model,arm,fake_fit,fake_fit,indices,deadline=-1)
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe.predict(model,arm,{},deadline=-1)
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe.score({name:model for name in r103.ARMS},
                   "DEV_UNKNOWN",{"reward0":np.array([1.])},deadline=-1)


def test_frozen_budget_is_monotonic_not_a_science_gate(monkeypatch):
    from experiments import r103_training_pipeline as pipe
    class FakeTimer:
        n=0
        @classmethod
        def next(cls):
            cls.n+=1
            return float(cls.n)
    monkeypatch.setattr(pipe.time,"perf_counter",FakeTimer.next)
    pipe._enforce_budget(2.5)
    pipe._enforce_budget(2.5)
    with pytest.raises(TimeoutError,match="R103_FROZEN_WHOLE_PROCESS_CPU_900_S_EXCEEDED"):
        pipe._enforce_budget(2.5)


@pytest.mark.parametrize("family",r103.UNSEEN)
@pytest.mark.parametrize("fixture_id",(9,10))
def test_role_relabel_is_an_actual_causal_physical_equivalence(family,fixture_id):
    """Metamorphic physical oracle, UNIT_ONLY. Never open FIT/DEV.

    A mere permutation of feature rows is insufficient: transform the R85
    physical factors, every pre-pivot A probe's role, and the registered
    three-factor operand assignment, then verify both chosen-action returns.
    Only after that compare the alleged role-invariance transformation.
    """
    from copy import deepcopy

    physical=r103._unit_world(fixture_id,3,family)
    transformed=deepcopy(physical)
    role_map=(2,0,3,1)  # Old physical role -> newly named physical role
    new_factors=np.empty(4,dtype=np.int64)
    for old_role,new_role in enumerate(role_map):
        new_factors[new_role]=physical["factors_a"][old_role]
    transformed["factors_a"]=new_factors
    transformed["r103_roles"]=tuple(
        role_map[i] for i in physical["r103_roles"])
    transformed["probe"]={
        tick:(phase,role_map[role] if phase=="A" else role,marker)
        for tick,(phase,role,marker) in physical["probe"].items()
    }
    transformed["a_probe_permutation"]=np.asarray(
        [role_map[int(i)] for i in physical["a_probe_permutation"]],
        dtype=np.int64)
    transformed["law_a"]=r103._operator_truth(
        new_factors,transformed["r103_roles"],family)
    assert transformed["law_a"]==physical["law_a"]

    for chosen in (r85.CALIBRATE,r85.EXPLOIT_POS):
        old,actual=r103.public_view(physical,chosen)
        expected,transformed_actual=r103.public_view(transformed,chosen)
        assert transformed_actual==actual
        support=np.repeat(old[None,None],pipe.SUPPORT,axis=1)
        out,query=pipe.permute_public_roles(support,old[None])
        np.testing.assert_array_equal(out[0,0],expected)
        np.testing.assert_array_equal(query[0],expected)
        np.testing.assert_array_equal(query[0,-1],old[-1])
        assert not np.array_equal(query[0,:4],old[:4])



def test_unit_cpu_capacity_probe_samples_only_unprotected_worlds(monkeypatch):
    """UNIT_ONLY batch has 16 factual returns; never touches R103 FIT/DEV."""
    from experiments import r103_capacity_probe as probe
    def protected_forbidden(*_a,**_kw):
        raise AssertionError("R103_PROTECTED_SCIENCE_SOURCE_WAS_OPENED")
    monkeypatch.setattr(r103,"_world",protected_forbidden)
    if (probe.ROOT/"experiments/r103_cpu_firstuse_authorization.json").is_file():
        # Correct behavior AFTER owner approval: the unit capacity probe
        # itself must refuse to run. No original FIT/DEV opened.
        with pytest.raises(
                PermissionError,match="R103_CAPACITY_PROBE_NOT_FOR_ORIGINAL_FIRSTUSE"):
            probe.prepare_unit_only()
        return
    batch=probe.prepare_unit_only()
    assert batch["x"].shape==(16,16,5,13)
    assert batch["factual"].shape==(16,16)
    assert batch["shuffled"].shape==(16,16)
    assert batch["query"].shape==(16,5,13)
    assert batch["target"].shape==(16,)
    assert batch["units_count"]==16
    assert batch["unit_physics_seconds"]>=0
    assert torch.isfinite(batch["target"]).all()
    assert not r103.frozen()["authority"]["protected_scored_first_use"]


def test_probe_never_reports_synthetic_capacity_as_scientific_pass():
    from experiments import r103_capacity_probe as probe
    assert probe.STEPS_RECORDED==6
    assert probe.WARMUP_STEPS==2
    assert probe.MAX_PROBE_SECONDS==120
    assert probe.SOURCE_UNIT_COUNT==16
    assert probe.RECEIPT.name=="r103_unit_only_capacity_probe.json"
