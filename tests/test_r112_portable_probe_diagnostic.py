"""R112 FIT/VALID-only engineering tests. Never touch spent R111 UNIT_UNSEEN."""
import numpy as np
import pytest
import torch

from experiments import r111_active_probe_diagnostic as r111
from experiments import r112_portable_custody as custody
from experiments import r112_portable_probe_diagnostic as r112


def _weights(*, sign=1):
    q=custody.Q
    w1=[[0]*5 for _ in range(32)]
    w2=[[0]*32 for _ in range(16)]
    w3=[[0]*16]
    w1[0][1]=sign*q
    w2[0][0]=q
    w3[0][0]=q
    return [{"weight":w1,"bias":[0]*32},
            {"weight":w2,"bias":[0]*16},
            {"weight":w3,"bias":[0]}]


def test_R112_sealed_preimplementation_original_lineage_and_authority(monkeypatch):
    f=custody.audit_freeze()
    assert f["stage"].startswith("PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION")
    assert custody.FREEZE_BLOB=="b41eb9dd59e27cc86f1cb0c153b2da1c79431e1e"
    assert tuple(f["source"]["seeds"])==custody.SEEDS
    assert f["source"]["no_original_R111_unseen_reuse"]
    assert all(value is False for value in f["authority"].values())
    assert not f["evaluation"]["SCIENCE_PASS_from_THIS_study"]
    assert custody.fit_valid_only("UNIT_FIT")=="UNIT_FIT"
    assert custody.fit_valid_only("UNIT_VALID")=="UNIT_VALID"
    for name in ("UNIT_UNSEEN","original_scientific_source","R111_UNSEEN"):
        with pytest.raises(PermissionError,match="UNSEEN"):
            custody.fit_valid_only(name)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R112_integer_feature_matches_public_physical_count_exactly():
    q=custody.Q
    history=[(0,1),(1,-1),(0,1)]
    f=r112._integer_feature(history,(0,5),1)
    assert f.dtype==np.int64 and f.shape==(2,5)
    assert f.tolist()==[[2*q//6,2*q//6,2*q//6,q,3*q//6],
                        [0,0,0,q,3*q//6]]
    with pytest.raises(ValueError):
        r112._integer_feature([(9,1)],(0,5),0)
    with pytest.raises(ValueError):
        r112._integer_feature([(0,0)],(0,5),0)
    with pytest.raises(ValueError):
        r112._integer_feature([], (1,1),0)


def test_R112_symmetric_rounding_signed_and_hardswish_bounded():
    inp=np.asarray([-768,-577,-384,-1,0,1,384,768],np.int64)
    got=r112._hard_swish(inp)
    assert got[0]==0 and got[-1]==768
    assert got[4]==0
    assert all(isinstance(int(v),int) for v in got)
    assert r112._rdiv([-768,-385,-192,-1,0,1,192,385,768],384).tolist()==[
        -2,-1,-1,0,0,0,1,1,2]
    with pytest.raises(ValueError):
        r112._rdiv([1,2,3],0)


def test_R112_portable_integer_action_changes_with_observed_fact_not_oracle():
    layers=_weights()
    s_pos=r112.portable_scores(layers,[(0,1)],(0,5),1)
    s_neg=r112.portable_scores(layers,[(0,-1)],(0,5),1)
    assert all(isinstance(x,int) for x in s_pos+s_neg)
    assert s_pos[0]>s_pos[1]
    assert s_neg[0]<s_neg[1]
    assert r112.choose_portable(layers,(0,5),[(0,1)],1)==0
    assert r112.choose_portable(layers,(0,5),[(0,-1)],1)==5
    assert r112.choose_portable(layers,(0,5),[],1)==0
    with pytest.raises(RuntimeError,match="SHAPE_DRIFT"):
        r112.portable_scores(layers[:2]+[{"weight":[[0]],"bias":[0]}],
                            [],(0,5),0)


def test_R112_quantization_matches_model_exact_layer_shapes():
    torch.set_num_threads(1)
    native=r111.new_models(r111.SEEDS[0])[r111.NATIVE]
    layer,sha=r112.quantize_model(native)
    assert len(layer)==3 and len(sha)==64
    assert [len(x["weight"]) for x in layer]==[32,16,1]
    assert [len(x["bias"]) for x in layer]==[32,16,1]
    assert r112.quantize_model(native)[1]==sha
    assert len(r112.portable_scores(layer,[],(0,3),0))==2
    assert r112.independent_cold_integer_parity(layer)==0


def test_R112_selected_intervention_not_unchosen_and_no_new_unseen_source():
    w=r111.PhysicalCausalWorld("UNIT_VALID",r111.SEEDS[0],0,1)
    p=w.offers()
    step=r112.choose_portable(_weights(),p,[],1)
    assert step in p
    signed=w.intervene(step)
    assert signed in (-1,1) and w.records==[{"chosen":step,"observed_sign":signed}]
    with pytest.raises(PermissionError,match="QUERY_BEFORE"):
        w.sealed_query()


def test_R112_integer_validation_does_not_recreate_R111_heldout():
    f=custody.audit_freeze()
    assert f["source"]["public_panels_allowed"]==["UNIT_FIT","UNIT_VALID"]
    assert f["source"]["chosen_action_budget_per_world"]==12
    assert r112.LAYERS==(5,32,16,1)
    assert "UNIT_UNSEEN" not in r112._validate_actor.__code__.co_consts
