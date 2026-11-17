"""R116 public-FIT-only source, actual action and signed-feedback controls.

Never build UNIT_UNSEEN worlds in tests, including positive-control tests.
"""
import numpy as np
import pytest
import torch

from experiments import r115_structured_query_source as structural
from experiments import r116_source_custody as custody
from experiments import r116_pair_source as source
from experiments import r116_feedback_probe_diagnostic as r116


def test_R116_original_future_source_is_frozen_and_spent_history_denied(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="0aeda5baff861abb8ebc10b4d19ae844216bb39d"
    assert f["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_UNIT_ONLY_DEV_NEVER_PROTECTED_SCIENCE"
    assert f["lineage"]["base_R115_exact_sha"]=="89adbdb916b9fb21ab4e383cfd2577eaf12f9d77"
    assert f["source"]["UNIT_UNSEEN_generation_only_after_all_trainable_512_updates"]
    assert not any(f["authority"].values())
    assert f["evaluation"]["protected_scientific_PASS_from_UNIT_ONLY"] is False
    assert len(f["evaluation"]["all_3_seed_checks_frozen"])==15
    for name in ("UNIT_UNSEEN","original_protected","R115_consumed"):
        with pytest.raises(PermissionError,match="FIRSTUSE_DENIED"):
            custody.permit_public_panel(name)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R116_exact_paired_compositions_have_novel_truth_tables_and_branch_balance():
    source.admit()
    assert len(source.FIT)==24 and len({structural.mask(x) for x in source.FIT})==9
    assert len(source.UNSEEN)==8
    assert len(source.RESERVED_R116)==8
    assert source.RESERVED_R116.isdisjoint(source.PREVIOUS_R115)
    assert source.RESERVED_R116.isdisjoint({structural.mask(x) for x in source.FIT})
    assert [f"{structural.mask(t):04x}" for t in source.UNSEEN]==[
        "0110","111e","e111","f111","0999","6996","9990","f999"]
    assert [structural.bit(x,source.QUERY) for x in source.UNSEEN]==[
        0,0,1,1,0,0,1,1]
    assert [structural.bit(x,7) for x in source.UNSEEN]==[0]*4+[1]*4
    assert all((structural.bit(t,8),structural.bit(t,9))==(1,0)
               for t in source.FIT+source.UNSEEN)


def test_R116_real_four_chosen_interventions_do_not_reveal_unchosen_or_query():
    for seed in source.SEEDS:
        w=source.PhysicalPairWorld("UNIT_FIT",seed,0)
        assert w.public_query==15
        with pytest.raises(PermissionError,match="QUERY_TRUTH_BEFORE"):
            w.sealed_query()
        for p in (8,9,7):
            assert w.offers()==(p,)
            with pytest.raises(PermissionError,match="UNCHO"):
                w.intervene(15)
            outcome=w.intervene(p)
            assert outcome in (-1,1)
        assert [x[0] for x in w.history]==[8,9,7]
        assert w.offers()==(12,14)
        w.intervene(14)
        assert w.sealed_query()[0]==15
        assert w.sealed_query()[1] in (-1,1)
        with pytest.raises(PermissionError,match="BUDGET_EXHAUSTED"):
            w.offers()
    with pytest.raises(PermissionError,match="SEALED_UNSEEN"):
        source.PhysicalPairWorld("UNIT_UNSEEN",source.SEEDS[0],0)
    with pytest.raises(PermissionError):
        source.PhysicalPairWorld("UNIT_FIT",1150011,0)


def test_R116_public_diagnostic_sign_alone_switches_the_optimal_next_probe():
    hneg=[(8,1),(9,-1),(7,-1)]
    hpos=[(8,1),(9,-1),(7,1)]
    gn=source.public_exact_voi_for_next(hneg)
    gp=source.public_exact_voi_for_next(hpos)
    assert gn[0][1]==gn[1][1] and gp[0][1]==gp[1][1]
    assert gn[1][0]>gn[0][0]
    assert gp[0][0]>gp[1][0]
    tn=source.public_observation_target(hneg)
    tp=source.public_observation_target(hpos)
    assert tn[1]>tn[0] and tp[0]>tp[1]
    assert r116.choose(r116.BAYES,{},[],hneg,source.SEEDS[0],0)==14
    assert r116.choose(r116.BAYES,{},[],hpos,source.SEEDS[0],0)==12
    assert r116.choose(r116.RULE,{},[],hneg,source.SEEDS[0],0)==14
    assert r116.choose(r116.RULE,{},[],hpos,source.SEEDS[0],0)==12
    with pytest.raises(PermissionError,match="CALIBRATION"):
        source.public_exact_voi_for_next([(8,-1),(9,-1),(7,1)])


def test_R116_FIT_only_public_signs_and_teacher_targets():
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(source.SEEDS[0])
    assert tuple(x.shape for x in fit)==(
        (256,2,45),(256,90),(256,5,18),(256,2))
    assert 90<=counts["factual_positive_diagnostic"]<=166
    assert 90<=counts["factual_negative_diagnostic"]<=166
    assert fit[-1].shape==(256,2)
    assert torch.isfinite(fit[-1]).all()
    assert "UNIT_UNSEEN" not in r116.make_fit.__code__.co_consts


def test_R116_all_three_models_trainable_without_oracle_truth():
    torch.set_num_threads(1)
    models=r116.new_models(source.SEEDS[0])
    assert set(models)==set(r116.TRAINABLE)
    n,m,g= r116.r115.public_inputs([(8,1),(9,-1),(7,1)],(12,14),15,0)
    x,y,z=(torch.from_numpy(v[None]) for v in (n,m,g))
    for name,model in models.items():
        output=model(x,y,z)
        assert output.shape==(1,2) and torch.isfinite(output).all()
        output.square().mean().backward()
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()),name
    with pytest.raises(PermissionError):
        r116.new_models(1150011)


def test_R116_native_integer_score_is_cold_restorable_no_test_scoring():
    torch.set_num_threads(1)
    native=r116.new_models(source.SEEDS[0])[r116.NATIVE]
    layers,sha=r116.r115.quantize_native(native)
    assert len(sha)==64 and len(layers)==3
    assert [len(x["weight"][0]) for x in layers]==[45,48,24]
    assert r116.r115.integer_cold_restore(layers)==0
    assert len(r116.r115.native_scores(layers,
               [(8,1),(9,-1),(7,-1)],(12,14),15,0))==2


def test_R116_frozen_science_criteria_all_absent_for_fake_no_effect_policy():
    data={arm:{"query_accuracy":.5,
               "same_public_calibration_actions":True,
               "actual_four_chosen_actions":512} for arm in r116.ARMS}
    trained={arm:{"updates":512} for arm in r116.TRAINABLE}
    score=r116.frozen_checks(data,0,0,0,trained)
    assert set(score["checks"])==set(
        r116.freeze()["evaluation"]["all_3_seed_checks_frozen"])
    assert len(score["checks"])==15
    assert not score["all_unit_only_checks_pass"]
    assert score["scientific_pass"] is False
