"""R113 FIT-only pre-first-use unit tests; NEVER generate scored UNIT_UNSEEN."""
import numpy as np
import pytest
import torch

from experiments import r113_source_custody as custody
from experiments import r113_structured_active_probe_diagnostic as r113


def test_R113_preimplementation_freeze_identity_source_novelty_and_authority(monkeypatch):
    f=r113.frozen()
    assert custody.FREEZE_BLOB=="53d9250db550019ebfc01e34685021927af93536"
    assert f["stage"]=="PREIMPLEMENTATION_FROZEN_UNIT_ONLY_DEV_NO_PROTECTED_FIRST_USE"
    assert f["evaluation"]["protected_scientific_pass_from_unit_only"] is False
    assert all(value is False for value in f["authority"].values())
    r113.admit_templates()
    assert len(r113.ALL)==36 and len(r113.FIT)==27 and len(r113.UNSEEN)==8
    assert len({r113.mask_of(x) for x in r113.FIT})==22
    assert len({r113.mask_of(x) for x in r113.UNSEEN})==8
    assert not {r113.mask_of(x) for x in r113.FIT}&r113.RESERVED_MASKS
    assert r113.SEEDS==(1130011,1130023,1130037)
    assert len(r113.TRAINABLE)==3 and len(r113.ARMS)==8
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="BLOB_FREEZE_DRIFT"):
        custody.audit_freeze()


def test_R113_composition_is_genuinely_two_stage_and_not_a_flat_R111_table():
    bits=[r113.truth_bit(("XOR","AND"),p) for p in range(8)]
    for p in range(8):
        x=(p>>2)&1;y=(p>>1)&1;z=p&1
        assert r113.truth_bit(("XOR","AND"),p)==((x^y)&z)
    assert r113.mask_of(("AND","XOR"))==0x6a
    assert r113.mask_of(("OR","NAND"))==0x57
    assert r113.mask_of(("NOR","NAND"))==0xfd
    assert r113.TRUTHS.shape==(36,8)
    with pytest.raises(ValueError):
        r113.truth_bit(("UNKNOWN","AND"),0)
    with pytest.raises(PermissionError):
        r113.truth_bit(("AND","OR"),8)


def test_R113_real_chosen_action_only_and_frozen_final_query():
    w=r113.PhysicalStructuredWorld("UNIT_FIT",r113.SEEDS[0],4,1)
    with pytest.raises(PermissionError,match="QUERY_BEFORE"):
        w.sealed_query()
    for step in range(6):
        options=w.offers()
        assert len(options)==2 and options[0]<options[1]
        assert all(0<=p<8 for p in options)
        if step==0:
            with pytest.raises(PermissionError,match="UNCHO"):
                w.intervene(8)
        choice=options[step%2]
        reward=w.intervene(choice)
        assert reward in (-1,1)
        assert w.records[-1]=={"chosen_do":choice,
                               "observed_signed_effect":reward}
        assert len(w.records)==step+1
    p,sign=w.sealed_query()
    assert 0<=p<8 and sign in (-1,1)
    with pytest.raises(PermissionError,match="BUDGET_EXHAUSTED"):
        w.intervene(0)
    with pytest.raises(PermissionError,match="SEALED_UNIT_UNSEEN"):
        r113.PhysicalStructuredWorld("UNIT_UNSEEN",r113.SEEDS[0],0,0)
    with pytest.raises(PermissionError):
        r113.PhysicalStructuredWorld("UNIT_FIT",1011011,0,0)
    with pytest.raises(PermissionError):
        r113.PhysicalStructuredWorld("PROTECTED_ORIGINAL",r113.SEEDS[0],0,0)


def test_R113_paired_public_worlds_are_deterministic_on_same_physical_actions():
    a=r113.PhysicalStructuredWorld("UNIT_FIT",r113.SEEDS[0],3,0)
    b=r113.PhysicalStructuredWorld("UNIT_FIT",r113.SEEDS[0],3,0)
    for _ in range(6):
        offers=a.offers()
        assert offers==b.offers()
        selected=offers[0]
        assert a.intervene(selected)==b.intervene(selected)
    assert a.history==b.history
    assert a.sealed_query()==b.sealed_query()
    assert r113.assign("UNIT_UNSEEN",0)[2] is False
    assert r113.assign("UNIT_UNSEEN",8)[2] is True
    assert sum(r113.assign("UNIT_UNSEEN",i)[2] for i in range(128))==64


def test_R113_public_information_gain_and_hypothesis_posterior_sanity():
    p=r113._posterior([])
    assert p.shape==(36,) and float(p.sum())==pytest.approx(1.)
    assert all(p>0) and float(p.max())==pytest.approx(1/36)
    ig=r113.information_gain(4,[])
    assert 0<ig<1.
    for pattern in range(8):
        post=r113._posterior([(pattern,1)]*6)
        assert np.isfinite(post).all() and np.sum(post)==pytest.approx(1.)
        assert r113.information_gain(pattern,[(pattern,1)]*6)<ig
        assert r113.fixed_readout(pattern,[(pattern,1)]*6)==1
        assert r113.fixed_readout(pattern,[(pattern,-1)]*6)==-1
    with pytest.raises(ValueError):
        r113._posterior([(8,1)])
    with pytest.raises(ValueError):
        r113._posterior([(0,0)])


def test_R113_native_integer_tokenless_features_bind_factual_consequences():
    history=[(0,1),(3,-1),(0,1)]
    native=r113.integer_native_features(history,(0,5),1)
    blind=r113.integer_native_features(history,(0,5),1,blind=True)
    assert native.shape==(2,24) and native.dtype==np.int64
    assert native[0,0]==2*64 and native[0,8]==2*64
    assert native[0,16:19].tolist()==[0,0,0]
    assert native[1,16:19].tolist()==[384,0,384]
    assert np.array_equal(native[:,:8],blind[:,:8])
    assert np.any(native[:,8:16]!=0)
    assert np.all(blind[:,8:16]==0)
    n,m,g=r113.features(history,(0,5),1)
    assert n.shape==(2,24) and m.shape==(48,) and g.shape==(6,10)
    assert n[0,0]==pytest.approx(2/6)
    assert g[:3,9].tolist()==[1,1,1]
    with pytest.raises(ValueError):
        r113.integer_native_features([(9,1)],(0,5),0)
    with pytest.raises(ValueError):
        r113.integer_native_features([(0,0)],(0,5),0)


def test_R113_scratch_native_and_strong_conventional_are_trainable():
    torch.set_num_threads(1)
    models=r113.new_models(r113.SEEDS[0])
    n,m,g=r113.features([(0,1)],(0,5),1)
    a=torch.from_numpy(n[None]);b=torch.from_numpy(m[None])
    c=torch.from_numpy(g[None])
    for name,model in models.items():
        out=model(a,b,c)
        assert out.shape==(1,2) and bool(torch.isfinite(out).all()),name
        out.square().mean().backward()
        assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
                   for p in model.parameters()),name
    with pytest.raises(PermissionError):
        r113.new_models(1011011)


def test_R113_quantized_native_integer_cold_reload_without_heldout():
    torch.set_num_threads(1)
    model=r113.new_models(r113.SEEDS[0])[r113.NATIVE]
    layers,sha=r113.quantize_native(model)
    assert len(sha)==64 and len(layers)==3
    assert [len(z["weight"][0]) for z in layers]==[24,32,16]
    assert r113.quantize_native(model)[1]==sha
    first=r113.integer_scores(layers,[(0,1)],(0,5),0)
    assert all(isinstance(x,int) for x in first)
    assert r113.isolated_integer_cold_reload(layers)==0
    with pytest.raises(PermissionError):
        r113.choose("ORIGINAL_PROTECTED",{},layers,(0,5),[],0,
                    r113.SEEDS[0],0,0)


def test_R113_FIT_only_training_data_and_zero_heldout_generation():
    torch.set_num_threads(1)
    records=r113.fit_examples(r113.SEEDS[0])
    assert tuple(x.shape for x in records)==(
        (2112,2,24),(2112,48),(2112,6,10),(2112,2))
    assert bool(torch.isfinite(records[-1]).all())
    assert float(records[-1].min())>=0.
    assert float(records[-1].max())<1.
    assert not any(x=="UNIT_UNSEEN" for x in r113.fit_examples.__code__.co_consts)


def test_R113_precommitted_science_checks_never_mean_scientific_acceptance():
    base={arm:{"new_after_accuracy":.5,"old_before_accuracy":.5,
               "old_after_accuracy":.5,"changed_law_new_after_accuracy":.5,
               "unchanged_law_new_after_accuracy":.5,
               "physically_chosen_probe_count":128*12}
          for arm in r113.ARMS}
    result=r113.frozen_checks(base,0,0)
    assert set(result["checks"])==set(
        r113.frozen()["evaluation"]["per_seed_criteria_all_three_required"])
    assert len(result["checks"])==15
    assert not result["all_unit_only_checks_pass"]
    assert result["scientific_pass"] is False
