"""FIT-only and schema-only tests for R111. NEVER generate UNIT_UNSEEN here."""
import numpy as np
import pytest
import torch

from experiments import r111_source_custody as custody
from experiments import r111_active_probe_diagnostic as r111


def test_R111_frozen_novel_source_and_authority(monkeypatch):
    f=r111.frozen()
    a=custody.audit_freeze()
    assert a["seeds"]==[1011011,1011023,1011037]
    assert len(a["novel_masks"])==8
    assert f["evaluation"]["original_protected_scientific_pass_from_unit_only"] is False
    assert all(v is False for v in f["authority"].values())
    assert f["source"]["total_chosen_physical_probes_per_arm_per_world"]==12
    assert len(r111.ARMS)==8 and len(r111.TRAINABLE)==3
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="FREEZE_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R111_actual_physical_chosen_probe_only_and_sealed_queries():
    w=r111.PhysicalCausalWorld("UNIT_FIT",r111.SEEDS[0],0,0)
    with pytest.raises(PermissionError,match="QUERY_BEFORE"):
        w.sealed_query()
    seen=[]
    for step in range(r111.STEPS):
        choices=w.offers()
        assert len(choices)==2 and choices[0]<choices[1] and 0<=choices[0]<8
        if step==0:
            with pytest.raises(PermissionError,match="UNOFFERED"):
                w.intervene(8)
        action=choices[step%2]
        reward=w.intervene(action)
        assert reward in (-1,1)
        assert len(w.records)==step+1
        assert set(w.records[-1])=={"chosen","observed_sign"}
        seen.append((action,reward))
    assert w.history==seen and len(w.records)==6
    assert len(w.sealed_query())==2
    with pytest.raises(PermissionError,match="PUBLIC_PROBES_FINISHED"):
        w.intervene(0)
    with pytest.raises(PermissionError):
        r111.PhysicalCausalWorld("UNIT_ORIGINAL_PROTECTED",r111.SEEDS[0],0,0)
    with pytest.raises(PermissionError):
        r111.PhysicalCausalWorld("UNIT_FIT",1010011,0,0)


def test_R111_physical_intervention_reproducibility_and_separate_phase():
    a=r111.PhysicalCausalWorld("UNIT_FIT",r111.SEEDS[0],7,0)
    b=r111.PhysicalCausalWorld("UNIT_FIT",r111.SEEDS[0],7,0)
    for _ in range(6):
        assert a.offers()==b.offers()
        p=a.offers()[0]
        assert a.intervene(p)==b.intervene(p)
    assert a.records==b.records
    old=r111.PhysicalCausalWorld("UNIT_FIT",r111.SEEDS[0],7,0)
    new=r111.PhysicalCausalWorld("UNIT_FIT",r111.SEEDS[0],7,1)
    assert new._phase==1 and old._phase==0


def test_R111_public_posterior_information_gain_and_feedback_sensitivity():
    p=4
    g0=r111.information_gain(p,[])
    assert 0.38<g0<0.40
    same=[(p,1)]*4
    conflict=[(p,1),(p,-1),(p,1),(p,-1)]
    assert r111.information_gain(p,same)<g0/4
    assert abs(r111.information_gain(p,conflict)-g0)<1e-7
    assert r111.posterior(p,[(p,1)])>0.5
    assert r111.fixed_readout(p,[(p,-1)])==-1
    assert r111.fixed_readout(0,[])==1
    assert r111.information_gain(5,[(4,1)])==pytest.approx(g0)
    with pytest.raises(ValueError):
        r111.posterior(8,[])


def test_R111_native_eight_slot_and_conventional_histories_have_same_public_inputs():
    torch.set_num_threads(1)
    hist=[(0,1),(3,-1),(0,1)]
    n,m,g=r111.features(hist,(0,5),1)
    assert n.shape==(2,5) and m.shape==(34,) and g.shape==(6,10)
    nblind,_,gblind=r111.features(hist,(0,5),1,blind=True)
    assert nblind[0,1]==0 and float(gblind[:,8].sum())==0
    assert n[0,1]!=0
    models=r111.new_models(r111.SEEDS[0])
    for arm,model in models.items():
        model.train()
        value=model(torch.from_numpy(n[None]),
                    torch.from_numpy(m[None]),
                    torch.from_numpy(g[None]))
        assert value.shape==(1,2) and torch.isfinite(value).all(),arm
        value.square().sum().backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters()),arm
    with pytest.raises(PermissionError):
        r111.choose("PROTECTED_ORACLE",models,(1,2),hist,1,r111.SEEDS[0],0,2)


def test_R111_FIT_only_training_examples_and_exact_budget():
    torch.set_num_threads(1)
    examples=r111.fit_examples(r111.SEEDS[0],None)
    assert tuple(x.shape for x in examples)==(
        (2112,2,5),(2112,34),(2112,6,10),(2112,2))
    assert torch.isfinite(examples[-1]).all()
    assert float(examples[-1].min())>=0
    assert float(examples[-1].max())<.41
    model=r111.new_models(r111.SEEDS[0])[r111.NATIVE]
    cold=r111.independent_cold_reload(model,examples,None)
    assert cold==0


def test_R111_frozen_check_names_and_failure_does_not_promote():
    by={}
    for arm in r111.ARMS:
        by[arm]={"new_after_accuracy":.5,"old_before_accuracy":.5,
                 "old_after_accuracy":.5,
                 "changed_law_new_after_accuracy":.5,
                 "unchanged_law_new_after_accuracy":.5,
                 "physically_chosen_probe_count":128*12}
    result=r111.frozen_checks(by,0,0)
    expected=r111.frozen()["evaluation"]["frozen_checks_all_seeds"]
    assert set(expected)==set(result["checks"])
    assert len(expected)==15
    assert not result["scientific_pass"]
    assert not result["all_unit_only_checks_pass"]
