"""P101: frozen intervention-value study, true choice-before-effect and cold retention."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p101_information_interventions_dev as p101

ROOT=Path(__file__).resolve().parents[1]


def test_p101_original_frozen_contract_and_source_hashes():
    f=p101.pinned_protocol()
    assert p101.FREEZE_SHA=="2f27ebcf70b06a27b2182b11e3473fda1b62a55d"
    assert f["schema"]=="ttt-hybrid.p101-information-directed-causal-intervention-dev-freeze.v1"
    assert f["source"]["interventions_per_task"]==8
    assert f["source"]["worlds_per_seed"]=={"FIT":24,"CAL":12}
    assert f["learner"]["prior_lambda"]==.5
    assert f["acceptance"]["scientific_pass"] is False
    assert set(p101.POLICIES)=={"max_variance","hash_random","cyclic","max_norm"}
    hashes={p101.world_hash(fam,split,i) for fam in p101.FAMILIES
            for split in p101.COUNTS for i in range(3)}
    assert len(hashes)==12


def test_p101_source_fences_and_chosen_action_only_leakage_negative():
    for args in (("isotropic","OOD",0,0),("isotropic","CAL",0,12),
                 ("unknown","FIT",0,0),("anisotropic","FIT",3,0),
                 ("anisotropic","NEGATIVE",0,0)):
        with pytest.raises(ValueError,match="P101_PROTECTED_OR_UNKNOWN_SOURCE_DENIED"):
            p101.build_world(*args)
    w=p101.build_world("anisotropic","FIT",0,0)
    assert w.pool.shape==(2,8,8,16)
    assert w.queries.shape==(2,12,16)
    assert len(w.maps)==2
    assert bool(torch.isfinite(w.pool).all())
    st=p101.initial()
    base=p101.choose("max_variance",st,w.pool[0,0],w.raw[0,0],1010211,0,0,0)
    # Counterfactual true outcomes are inaccessible to choose: its entire
    # allowed signature is unchanged when latent causal maps are replaced.
    alternate=w._replace(maps=(torch.zeros_like(w.maps[0]),
                               torch.ones_like(w.maps[1])))
    n=p101.choose("max_variance",st,alternate.pool[0,0],
                  alternate.raw[0,0],1010211,0,0,0)
    assert base==n
    with pytest.raises(ValueError,match="P101_UNQUALIFIED_CHOICE_SET"):
        p101.choose("oracle",st,w.pool[0,0],w.raw[0,0],1010211,0,0,0)


def test_p101_all_arms_equal_causal_effect_calls_and_same_learner():
    w=p101.build_world("isotropic","CAL",0,0)
    results={}
    for p in p101.POLICIES:
        results[p]=p101.run_world(w,p,1010127,0,wrong_credit=(p=="max_variance"))
        assert results[p]["n_confirmed_effects"]==16
        assert results[p]["persistent_floats"]==1024
        assert results[p]["cold_task0_bitwise"] is True
        assert results[p]["mse_all"]>=0
        assert results[p]["mse_all"]<float("inf")
    assert results["max_variance"]["wrong_credit_mse"]>=0
    assert all(len(v["selected_indices_sha256"])==64 for v in results.values())


def test_p101_factual_update_not_unconfirmed_and_queries_are_read_only():
    w=p101.build_world("isotropic","FIT",0,1)
    s=p101.initial()
    k=w.pool[0,0,0]
    v=w.maps[0]@k
    next_state=p101.confirmed_write(s,k,v)
    assert not torch.equal(s.weights,next_state.weights)
    assert not torch.equal(s.precision,next_state.precision)
    blocked=p101.confirmed_write(s,k,torch.full_like(v,float("nan")),confirmed=False)
    assert torch.equal(s.weights,blocked.weights)
    assert torch.equal(s.precision,blocked.precision)
    with pytest.raises(ValueError,match="P101_UNQUALIFIED_CONFIRMED_EFFECT"):
        p101.confirmed_write(s,k,torch.full_like(v,float("nan")))
    before=next_state.weights.clone(),next_state.precision.clone()
    _=p101.read(next_state,w.queries[0,0])
    assert torch.equal(before[0],next_state.weights)
    assert torch.equal(before[1],next_state.precision)


def test_p101_real_new_cal_falsifier_receipt_from_existing_full_ci(tmp_path):
    result=p101.run()
    assert result["status"] in (
        "DEV_FAIL_INTERVENTION_VALUE_OR_COST",
        "DEV_CAL_ONLY_PASS_OOD_STILL_SEALED",
    )
    assert result["scientific_pass"] is False
    assert result["external_simulator_used"] is False
    assert result["reserved_OOD_generated"] is False
    assert result["reserved_negative_generated"] is False
    assert result["native_hybrid_trained"] is False
    assert len(set(result["source_sha256"].values()))==12
    assert result["freeze_commit"]==p101.FREEZE_SHA
    for family in p101.FAMILIES:
        x=result["families"][family]
        assert len(x["seeds"])==3
        for row in x["seeds"]:
            assert row["fit_selection"] in p101.PASSIVES
            assert set(row["FIT_passives"])==set(p101.PASSIVES)
            assert set(row["CAL_policies"])==set(p101.POLICIES)
            assert all(z["total_confirmed_outcomes"]==12*16
                       for z in row["CAL_policies"].values())
            assert all(z["persistent_floats"]==1024
                       for z in row["CAL_policies"].values())
            assert all(z["cold_restore_bitwise"]
                       for z in row["CAL_policies"].values())
            assert all(v>0 for v in row["cpu_median_ms"].values())
    sha=os.getenv("EXPECTED_CANDIDATE_SHA")
    if sha:
        result["capture_provenance"]={
            "candidate_sha":sha,"github_run_id":os.getenv("GITHUB_RUN_ID"),
            "classification":"FRESH_AUTHOR_SYNTHETIC_DEV_NOT_SCIENTIFIC_ACCEPTANCE"
        }
        dest=ROOT/"local_runs/p101_dev_result.json"
    else:
        dest=tmp_path/"p101_dev_result.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")
