"""R114 public FIT-only tests; no R113 UNIT_UNSEEN generation or scoring."""
import hashlib
import pytest
import torch

from experiments import r113_structured_active_probe_diagnostic as r113
from experiments import r114_source_custody as custody
from experiments import r114_exact_readout_diagnostic as r114


def test_R114_prospective_exact_freeze_and_no_spent_or_original_access(monkeypatch):
    f=custody.audit_freeze()
    assert custody.FREEZE_BLOB=="a00e102ae6c7ae19c8f03a14ee54f7481f6e3961"
    assert f["stage"]=="PROSPECTIVE_PREIMPLEMENTATION_FIT_VALID_ONLY_NO_R113_UNSEEN_REUSE"
    assert f["lineage"]["parent_R113_exact_sha"]=="2910f2974da86bcd74c1b3411784d821b1d002f9"
    assert f["source"]["R113_no_scored_holdout_generated"] is True
    assert all(x is False for x in f["authority"].values())
    assert f["evaluation"]["original_scientific_PASS_from_fit_valid"] is False
    assert custody.allow("UNIT_FIT")=="UNIT_FIT"
    assert custody.allow("UNIT_VALID")=="UNIT_VALID"
    for panel in ("UNIT_UNSEEN","original_protected","R111_original","R113_consumed"):
        with pytest.raises(PermissionError,match="SPENT_HELDOUT"):
            custody.allow(panel)
    monkeypatch.setattr(custody,"FREEZE_BLOB","0"*40)
    with pytest.raises(RuntimeError,match="GIT_BLOB_DRIFT"):
        custody.audit_freeze()


def test_R114_exact_rational_posterior_integer_ties_deterministic():
    for q in range(8):
        weights=r114.exact_weights_from_public([])
        assert len(weights)==36 and weights==[1]*36
        answer,tie=r114.integer_prediction(q,[])
        assert answer==1 and tie==1
        assert r114.integer_prediction(q,[(q,1)])[0]==1
        assert r114.integer_prediction(q,[(q,-1)])[0]==-1
    hist=[(0,1),(0,-1),(0,1),(0,-1),(3,1),(3,-1)]
    weights=r114.exact_weights_from_public(hist)
    assert all(type(x) is int and x>0 for x in weights)
    assert len(set(weights))>=1
    assert r114.integer_prediction(5,hist)[0] in (-1,1)
    with pytest.raises(ValueError):
        r114.exact_weights_from_public([(0,0)])
    with pytest.raises(ValueError):
        r114.exact_weights_from_public([(8,1)])
    with pytest.raises(ValueError):
        r114.integer_prediction(8,[])


def test_R114_causal_integer_likelihood_does_not_access_hidden_mechanisms():
    # Hand check: one public positive sign at p=0 gives 17 to a matching
    # structural hypothesis and 3 to a contradicting hypothesis.
    weights=r114.exact_weights_from_public([(0,1)])
    for i,pair in enumerate(r113.ALL):
        assert weights[i]==(17 if r113.truth_bit(pair,0) else 3)
    plus=sum(w for i,w in enumerate(weights) if r113.truth_bit(r113.ALL[i],0))
    minus=sum(w for i,w in enumerate(weights) if not r113.truth_bit(r113.ALL[i],0))
    assert plus>minus
    assert r114.integer_prediction(0,[(0,1)])==(1,0)


def test_R114_r113_scored_original_worlds_have_no_public_access():
    with pytest.raises(PermissionError,match="SEALED_UNIT_UNSEEN"):
        r113.PhysicalStructuredWorld("UNIT_UNSEEN",r113.SEEDS[0],0,0)
    valid=r113.PhysicalStructuredWorld("UNIT_VALID",r113.SEEDS[0],0,0)
    for i in range(6):
        p=valid.offers()[i%2]
        assert valid.intervene(p) in (-1,1)
    q,truth=valid.sealed_query()
    assert r114.integer_prediction(q,valid.history)[0] in (-1,1)
    assert truth in (-1,1)


def test_R114_immutable_event_hashes_require_valid_actions():
    assert r114._hash_octets([0,1,2,7])==hashlib.sha256(bytes([0,1,2,7])).hexdigest()
    for bad in ([],[256],[-1]):
        with pytest.raises(RuntimeError):
            r114._hash_octets(bad)


def test_R114_native_quantized_FIT_only_cold_state_preserves_exact_scores():
    torch.set_num_threads(1)
    models=r113.new_models(r113.SEEDS[0])
    native=models[r113.NATIVE]
    layers,sha=r113.quantize_native(native)
    assert len(sha)==64
    assert r113.isolated_integer_cold_reload(layers)==0
    assert r113.integer_scores(layers,[(0,1)],(0,5),1)==r113.integer_scores(
        layers,[(0,1)],(0,5),1)
