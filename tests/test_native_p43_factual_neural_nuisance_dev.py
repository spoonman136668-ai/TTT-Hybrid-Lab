"""P43 no-source custody and information-parity controls (not scientific qualification)."""
import hashlib
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_p43_factual_neural_nuisance_dev import (
    authored, FactorizedHistoryMLP, ConventionalMLP, ConventionalGRU,
    wrong_credit_history, params, HISTORY, FIT_N, VALID_N, OOD_N,
    SOURCE_SEEDS, FREQUENCIES, UPDATES, BATCH_SIZE, SAMPLE_SEED,
)

SOURCE_BLOB = "c3ced61c21be36b6da411ad9a3863559e01a98b5"


def test_custody_and_no_hidden_counterfactual_training():
    source=Path(__file__).resolve().parents[1]/"native_p43_factual_neural_nuisance_dev.py"
    data=source.read_bytes()
    sha=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    assert sha==SOURCE_BLOB
    fit=authored(FIT_N,"fit")
    assert set(fit)=={"history","query_action","factual_query","scored_both","beta","source_trace_sha256","cohort"}
    assert fit["history"].shape==(FIT_N,HISTORY,3)
    assert fit["query_action"].shape==(FIT_N,1)
    assert fit["factual_query"].shape==(FIT_N,1)
    assert fit["scored_both"].shape==(FIT_N,2)
    assert fit["source_trace_sha256"]=="60a02c7c9dbd7916b85559529b45dba863103a9f4b4db18031050d3afb2ed286"


def test_source_disjoint_registered_and_out_of_range_rejected():
    valid=authored(VALID_N,"validation")
    ood=authored(OOD_N,"ood_dev")
    assert valid["source_trace_sha256"]=="3d4a5262531d9853264c6aaddfbfcc4bfa11792d80511e1e4735aaf975a1ecc9"
    assert ood["source_trace_sha256"]=="ffaad800187f3d277d45fc710ab4e37bf11bb9aa9186bb2241aa3b71009f4044"
    assert FREQUENCIES["ood_dev"][0]>FREQUENCIES["fit"][1]
    assert len(set(SOURCE_SEEDS.values()))==3
    try:
        authored(1234,"ood_dev")
    except ValueError as exc:
        assert str(exc)=="P43_NONFROZEN_SOURCE_COHORT"
    else:
        raise AssertionError("Unregistered source count accepted")


def test_matched_capacity_and_deterministic_wrong_credit_only_shifts_past_action():
    n0,n1,n2=map(params,(FactorizedHistoryMLP(),ConventionalMLP(),ConventionalGRU()))
    assert (n0,n1,n2)==(1642,1649,1747)
    assert max(n0,n1,n2)/min(n0,n1,n2)<1.2
    h=authored(VALID_N,"validation")["history"]
    corrupted=wrong_credit_history(h)
    assert torch.equal(corrupted[:,:,0],h[:,:,0])
    assert torch.equal(corrupted[:,:,2],h[:,:,2])
    assert torch.equal(corrupted[:,:,1],h[:,:,1].roll(1,dims=1))
    assert UPDATES==400 and BATCH_SIZE==128 and SAMPLE_SEED==720001


def test_each_model_read_only_two_query_actions_have_valid_shapes():
    h=authored(VALID_N,"validation")["history"][:3]
    for cls in (FactorizedHistoryMLP,ConventionalMLP,ConventionalGRU):
        model=cls().eval()
        with torch.no_grad():
            p=model(h,torch.ones((3,1)))
            m=model(h,-torch.ones((3,1)))
        assert p.shape==(3,1) and m.shape==(3,1)
        assert torch.isfinite(p).all() and torch.isfinite(m).all()
