"""R115 FIT-only/VALID-only negative controls. NEVER generate UNIT_UNSEEN."""
import numpy as np
import pytest
import torch

from experiments import r115_source_custody as custody
from experiments import r115_structured_query_source as source
from experiments import r115_query_targeted_diagnostic as r115


def test_R115_immutable_prospective_source_and_original_firstuse_fences(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="1c67231758eeb85644f5844f0c940f412ccc3cce"
    assert f["stage"]=="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
    assert f["source"]["query_public_before_support"]
    assert f["source"]["query_true_outcome_hidden_until_all_probes"]
    assert f["source"]["tests_may_generate_only_FIT_and_VALID"]
    assert f["evaluation"]["protected_scientific_pass_from_unit_only"] is False
    assert not any(f["authority"].values())
    assert r115.frozen()["training"]["FIT_training_examples_per_seed"]==2560
    assert custody.guard_public_panel("UNIT_VALID")=="UNIT_VALID"
    for panel in ("UNIT_UNSEEN","original_protected","R113_original"):
        with pytest.raises(PermissionError,match="SCORING_SOURCE"):
            custody.guard_public_panel(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R115_structural_four_factor_compositions_are_unseen_as_functions():
    source.audit_novelty()
    assert len(source.ALL)==216 and len(source.FIT)==194
    assert len({source.mask(t) for t in source.FIT})==82
    assert len(source.RESERVED)==8
    assert not ({source.mask(t) for t in source.FIT} & source.RESERVED)
    assert [f"{source.mask(t):04x}" for t in source.UNSEEN]==[
        "1eee","0008","0990","6fff","fff7","1ee1","0666","8778"]
    assert all(any(t[i]==o for t in source.FIT)
               for i in range(3) for o in source.OPS)
    for x in range(16):
        a=(x>>3)&1;b=(x>>2)&1;c=(x>>1)&1;d=x&1
        assert source.bit(("AND","OR","XOR"),x)==((a&b)^(c|d))
    with pytest.raises(ValueError):
        source.bit(("BAD","OR","XOR"),1)
    with pytest.raises(ValueError):
        source.bit(("AND","OR","XOR"),16)


def test_R115_real_probing_target_public_but_private_effect_sealed():
    a=source.PhysicalCausalWorld("UNIT_FIT",r115.SEEDS[0],5,1)
    b=source.PhysicalCausalWorld("UNIT_FIT",r115.SEEDS[0],5,1)
    assert a.public_query==b.public_query and 0<=a.public_query<16
    with pytest.raises(PermissionError,match="TARGET_SIGN"):
        a.sealed_query()
    with pytest.raises(PermissionError,match="UNCHO"):
        a.intervene(16)
    for step in range(source.STEPS):
        offered=a.offers()
        assert len(offered)==2 and offered==b.offers()
        choice=offered[step%2]
        assert a.intervene(choice)==b.intervene(choice)
        assert len(a.history)==step+1
    assert a.sealed_query()==b.sealed_query()
    with pytest.raises(PermissionError,match="BUDGET_EXHAUSTED"):
        a.offers()
    with pytest.raises(PermissionError,match="FRESH_UNSEEN_BEFORE"):
        source.PhysicalCausalWorld("UNIT_UNSEEN",r115.SEEDS[0],0,0)
    with pytest.raises(PermissionError):
        source.PhysicalCausalWorld("PROTECTED",r115.SEEDS[0],0,0)
    with pytest.raises(PermissionError):
        source.PhysicalCausalWorld("UNIT_FIT",1159999,0,0)


def test_R115_exact_query_targeted_information_benefit_is_nonnegative_and_real():
    for q in (0,3,7,10,15):
        sign,tie=source.posterior_prediction(q,[])
        assert sign==1 and tie==1
        gain,denom=source.target_query_information_value(q,q,[],full=False)
        assert gain>0 and denom>0
        assert gain/denom==pytest.approx(.4)
        for offered in (0,4,8,11,15):
            candidate,_=source.target_query_information_value(offered,q,[],full=False)
            assert candidate<=gain
        plus,_=source.posterior_prediction(q,[(q,1)])
        minus,_=source.posterior_prediction(q,[(q,-1)])
        assert plus==1 and minus==-1
    assert source.public_query_value_for_fit(3,3,[])==pytest.approx(8.)
    with pytest.raises(ValueError):
        source.target_query_information_value(16,0,[])
    with pytest.raises(ValueError):
        source.posterior_prediction(16,[])
    with pytest.raises(ValueError):
        source.posterior_prediction(0,[(0,0)])


def test_R115_native_query_and_observed_effect_changes_public_integer_state():
    history=[(0,1),(0,-1),(7,1)]
    n=r115.native_integer_features(history,(0,9),7,1)
    qblind=r115.native_integer_features(history,(0,9),7,1,query_blind=True)
    fblind=r115.native_integer_features(history,(0,9),7,1,feedback_blind=True)
    assert n.shape==(2,45) and n.dtype==np.int64
    assert np.any(n[:,32:36]!=0) and np.all(qblind[:,32:36]==0)
    assert np.all(n[:,:32]==qblind[:,:32])
    assert np.all(fblind[:,16:32]==0)
    assert np.any(n[:,16:32]!=0)
    assert np.all(n[:,:16]==fblind[:,:16])
    assert n[0,40]==384 and n[0,41]==r115._norm(3)
    x,y,g=r115.public_inputs(history,(0,9),7,1)
    assert x.shape==(2,45) and y.shape==(90,) and g.shape==(5,18)
    assert g[:3,17].tolist()==[1,1,1]
    with pytest.raises(ValueError):
        r115.native_integer_features([(0,0)],(0,9),7,1)
    with pytest.raises(ValueError):
        r115.native_integer_features(history,(0,0),7,1)


def test_R115_three_fully_trainable_policy_shapes_and_parameter_grads():
    torch.set_num_threads(1)
    models=r115.new_models(r115.SEEDS[0])
    assert set(models)==set(r115.TRAINABLE)
    n,m,g=r115.public_inputs([(1,1)],(3,8),5,1)
    x=torch.from_numpy(n[None]);y=torch.from_numpy(m[None]);z=torch.from_numpy(g[None])
    for arm,model in models.items():
        predicted=model(x,y,z)
        assert predicted.shape==(1,2) and bool(torch.isfinite(predicted).all())
        predicted.square().mean().backward()
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()),arm
    with pytest.raises(PermissionError):
        r115.new_models(1130011)


def test_R115_integer_native_scorer_is_portable_with_fresh_process_only():
    torch.set_num_threads(1)
    models=r115.new_models(r115.SEEDS[0])
    layers,sha=r115.quantize_native(models[r115.NATIVE])
    assert len(layers)==3 and len(sha)==64
    assert [len(x["weight"][0]) for x in layers]==[45,48,24]
    scores=r115.native_scores(layers,[(0,1)],(3,9),6,0)
    assert len(scores)==2 and all(type(v) is int for v in scores)
    assert r115.integer_cold_restore(layers)==0
    assert r115.quantize_native(models[r115.NATIVE])[1]==sha
    with pytest.raises(RuntimeError,match="SHAPE"):
        r115.native_scores(layers[:2]+[{"weight":[[0]],"bias":[0]}],
                           [],(3,9),6,1)


def test_R115_public_FIT_only_training_source_fulfills_hard_budget():
    torch.set_num_threads(1)
    data=r115.fit_examples(r115.SEEDS[0])
    assert tuple(x.shape for x in data)==(
        (2560,2,45),(2560,90),(2560,5,18),(2560,2))
    assert float(data[-1].min())>=0
    assert float(data[-1].max())<=10
    assert all(torch.isfinite(z).all() for z in data)
    assert "UNIT_UNSEEN" not in r115.fit_examples.__code__.co_consts
    model=r115.new_models(r115.SEEDS[0])[r115.NATIVE]
    with pytest.raises(PermissionError,match="UNMATCHED_UPDATE"):
        r115.train(model,r115.NATIVE,data,np.zeros((1,16),int),None)


def test_R115_per_seed_frozen_science_checks_cannot_become_acceptance_from_tests():
    fake={arm:{
      "new_after_accuracy":.5,"old_before_accuracy":.5,"old_after_accuracy":.5,
      "changed_law_new_after_accuracy":.5,
      "unchanged_law_new_after_accuracy":.5,
      "actual_chosen_probe_count":1280} for arm in r115.ARMS}
    result=r115.frozen_checks(fake,0,0)
    assert set(result["checks"])==set(
        r115.frozen()["evaluation"]["precommitted_each_seed_all_required"])
    assert len(result["checks"])==16
    assert result["scientific_pass"] is False
    assert not result["all_unit_only_checks_pass"]
