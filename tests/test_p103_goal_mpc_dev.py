"""P103 frozen goal-conditioned causal MPC with strong conventional controls."""
import json
import os
from pathlib import Path
import pytest
import torch
from experiments import p101_information_interventions_dev as p101
from experiments import p103_goal_mpc_dev as p103

ROOT=Path(__file__).resolve().parents[1]


def test_p103_precise_prospective_freeze_without_prior_CAL_reuse():
    f=p103.freeze()
    assert p103.FREEZE_COMMIT=="40a70dca2120a96404eaf10e0bc7538ec39fbf20"
    assert f["status"]=="PROSPECTIVELY_FROZEN_BEFORE_ANY_P103_SOURCE_IMPLEMENTATION_OR_DATA"
    assert f["parent"]["sha"]=="52553526ceea2ce01f26cc035bb486b351974fd8"
    assert f["source"]["seeds"]["reserved_OOD"]==[1030401,1030403]
    assert f["source"]["seeds"]["reserved_NEGATIVE"]==[1030501,1030503]
    assert f["acceptance"]["scientific_pass"] is False
    assert set(p103.SEEDS)==set(p103.FAMILIES)
    assert p103.COUNT=={"FIT":16,"CAL":12}
    hashes=[p103.fingerprint(fam,split,i) for fam in p103.FAMILIES
            for split in p103.COUNT for i in range(3)]
    assert len(hashes)==12 and len(set(hashes))==12


def test_p103_source_qualification_and_reserved_data_hard_refusal():
    for args in (("isotropic","OOD",0,0),
                 ("isotropic","NEGATIVE",0,0),
                 ("unknown","FIT",0,0),
                 ("anisotropic","CAL",0,12),
                 ("isotropic","FIT",3,0),
                 ("isotropic","FIT",0,-1)):
        with pytest.raises(ValueError,match="P103_PROTECTED_SOURCE_REFUSAL"):
            p103.build_world(*args)
    w=p103.build_world("anisotropic","FIT",0,0)
    w2=p103.build_world("anisotropic","FIT",0,0)
    assert w.calibrate.shape==(2,8,8,16)
    assert w.pool.shape==(2,8,8,16)
    assert w.goals.shape==(2,16)
    assert w.queries.shape==(2,12,16)
    assert all(torch.equal(a,b) for a,b in zip(w.maps,w2.maps))
    assert torch.equal(w.goals,w2.goals)
    assert torch.equal(w.pool,w2.pool)


def test_p103_no_oracle_goal_aware_and_future_source_blindness():
    w=p103.build_world("isotropic","FIT",0,0)
    s=p101.initial()
    physical=torch.zeros(16,dtype=torch.float64)
    pool=w.pool[0,0]
    next_pool=w.pool[0,1]
    # The action chooser's only inputs: model state, current position, public
    # goal and two permissible candidate pools. It never receives hidden M.
    decision=p103.chooser("epistemic_mpc",s,physical,w.goals[0],
                          pool,next_pool,1030111,0,0,0,.08)
    replaced=w._replace(maps=(torch.zeros_like(w.maps[0]),
                              torch.ones_like(w.maps[1])),
                        queries=torch.full_like(w.queries,float("nan")))
    same=p103.chooser("epistemic_mpc",s,physical,replaced.goals[0],
                      replaced.pool[0,0],replaced.pool[0,1],1030111,0,0,0,.08)
    assert same==decision
    assert 0<=decision<8
    wrong_goal=p103.chooser("goal_sham",s,physical,w.goals[0],
                             pool,next_pool,1030111,0,0,0,.08)
    assert 0<=wrong_goal<8
    with pytest.raises(ValueError,match="P103_CHOOSER_BOUNDARY_VIOLATION"):
        p103.chooser("oracle",s,physical,w.goals[0],
                      pool,next_pool,1030111,0,0,0,.08)
    with pytest.raises(ValueError,match="P103_CHOOSER_BOUNDARY_VIOLATION"):
        p103.chooser("epistemic_mpc",s,physical,
                      torch.full_like(w.goals[0],float("nan")),
                      pool,next_pool,1030111,0,0,0,.08)


def test_p103_equal_32_confirmed_effects_per_world_cold_task0_and_negative():
    w=p103.build_world("anisotropic","CAL",0,0)
    all_results={}
    for policy in p103.ARMS:
        z=p103.one_world(w,policy,1030227,0,.08,
                         wrong_credit=(policy=="epistemic_mpc"))
        all_results[policy]=z
        assert z["confirmed_effects"]==32
        assert z["persistent_float64_state"]==1024
        assert z["cold_bitwise"]
        assert len(z["journal_sha256"])==64
        assert 0<=z["tracking_all"]<float("inf")
    assert all_results["epistemic_mpc"]["wrong_credit_query_mse"]>=0
    model=p101.initial()
    a=w.calibrate[0,0,0]
    blocked=p101.confirmed_write(model,a,torch.full_like(a,float("nan")),False)
    assert torch.equal(blocked.weights,model.weights)
    assert torch.equal(blocked.precision,model.precision)
    with pytest.raises(ValueError,match="P101_UNQUALIFIED_CONFIRMED_EFFECT"):
        p101.confirmed_write(model,a,torch.full_like(a,float("nan")))


def test_p103_existing_full_ci_captures_frozen_DEV_not_science(tmp_path):
    report=p103.study()
    assert report["status"] in (
        "DEV_FAIL_GOAL_MPC_VS_CONVENTIONAL_CONTROLS",
        "DEV_CAL_GOAL_MPC_POSITIVE_NOT_SCIENTIFIC_PASS",
    )
    assert report["scientific_pass"] is False
    assert report["external_source_used"] is False
    assert report["OOD_generated"] is False
    assert report["NEGATIVE_reserved_generated"] is False
    assert report["native_model_trained"] is False
    assert len(report["source_sha256"])==12
    assert len(set(report["source_sha256"].values()))==12
    assert report["freeze_commit"]==p103.FREEZE_COMMIT
    for fam in p103.FAMILIES:
        rows=report["families"][fam]["seeds"]
        assert len(rows)==3
        for row in rows:
            assert row["FIT_beta"] in p103.BETAS
            assert row["FIT_strongest"] in p103.CONVENTIONAL
            assert set(row["CAL_controls"])==set(p103.CONVENTIONAL)
            assert set(row["ratios"])=={"FIT_strongest","pure_mpc"}
            assert all(z["confirmed_outcomes"]==12*32 for z in (
                row["CAL_candidate"],row["CAL_goal_sham"],
                *row["CAL_controls"].values()))
            assert all(z["cold_bitwise"] for z in (
                row["CAL_candidate"],row["CAL_goal_sham"],
                *row["CAL_controls"].values()))
            assert row["CPU_ratio"]>0
    if os.getenv("EXPECTED_CANDIDATE_SHA"):
        report["capture_provenance"]={
            "candidate_sha":os.environ["EXPECTED_CANDIDATE_SHA"],
            "github_run_id":os.getenv("GITHUB_RUN_ID"),
            "classification":"AUTHOR_SYNTHETIC_GOAL_DEV_NOT_EXTERNAL_SCIENCE",
        }
        dest=ROOT/"local_runs/p103_dev_result.json"
    else:
        dest=tmp_path/"p103_dev_result.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n")
