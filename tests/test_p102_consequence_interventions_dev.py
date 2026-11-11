"""P102 frozen three-regime consequence-aware intervention acquisition DEV qualification."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p101_information_interventions_dev as p101
from experiments import p102_consequence_interventions_dev as p102


def test_p102_freeze_prospective_custody_and_all_source_fences():
    doc=p102.frozen()
    assert p102.FREEZE=="16dab90c352af558ae1db152eaf292065652745d"
    assert doc["identity"]["parent_sha"]=="fcf260b966e8e6dd12a2dc1594103f5a5cc44725"
    assert doc["source"]["seeds"]["OOD_RESERVED"]==[1020401,1020403]
    assert doc["source"]["seeds"]["NEGATIVE_RESERVED"]==[1020501,1020503]
    assert doc["parameter_selection"]["beta_grid"]==[.25,1.,4.]
    assert doc["acceptance"]["scientific_pass"] is False
    for path,blob in p102.PARENTS.items():
        assert p86.blob(Path(__file__).resolve().parents[1]/path)==blob
    for bad in (("unknown","FIT",0,0),("isotropic","OOD",0,0),
                ("isotropic","CAL",0,12),("isotropic","CAL",4,0),
                ("anisotropic","NEGATIVE",0,0)):
        with pytest.raises(ValueError,match="P102_PROTECTED_OR_UNKNOWN_SOURCE_DENIED"):
            p102.build_world(*bad)


def test_p102_independent_action_cohorts_deterministic_and_sealed_probes():
    values=[]
    for family in p102.FAMILIES:
        for split in p102.COUNT:
            for seed_index in range(3):
                values.append(p102.source_sha(family,split,seed_index))
    assert len(values)==18 and len(set(values))==18
    a=p102.build_world("mildly_anisotropic","FIT",0,0)
    b=p102.build_world("mildly_anisotropic","FIT",0,0)
    assert a.pool.shape==(2,8,8,16)
    assert a.queries.shape==(2,12,16)
    assert all(torch.equal(x,y) for x,y in zip(a.maps,b.maps))
    assert torch.equal(a.pool,b.pool)
    assert torch.equal(a.queries,b.queries)
    assert bool((a.pool.norm(dim=-1)>0).all())
    # Query-specific effects are not part of the chooser's arguments.
    st=p101.initial()
    choice1=p102.choice("impact",st,a.pool[0,0],a.raw[0,0],
                        p102.SEEDS["mildly_anisotropic"]["FIT"][0],0,0,0)
    changed=a._replace(maps=(torch.zeros_like(a.maps[0]),
                             torch.ones_like(a.maps[1])))
    choice2=p102.choice("impact",st,changed.pool[0,0],changed.raw[0,0],
                        p102.SEEDS["mildly_anisotropic"]["FIT"][0],0,0,0)
    assert choice1==choice2
    with pytest.raises(ValueError,match="P102_INVALID_CHOICE"):
        p102.choice("oracle",st,a.pool[0,0],a.raw[0,0],1020311,0,0,0)


def test_p102_confirmed_outcome_model_conditioning_is_distinct_from_geometry():
    st=p101.initial()
    W=st.weights.clone()
    W[0,1]=20.
    state=p101.RLS(W,st.precision)
    pool=torch.eye(16,dtype=torch.float64)[:8].clone()
    geom=p102.choice("geometry",state,pool,pool,1020111,0,0,1)
    action=p102.choice("impact",state,pool,pool,1020111,0,0,1,beta=1.)
    assert geom==0
    assert action==1
    empty=p101.initial()
    assert p102.choice("impact",empty,pool,pool,1020111,0,0,0,beta=1.)==geom
    bad=torch.full_like(pool,float("nan"))
    with pytest.raises(ValueError,match="P102_INVALID_CHOICE"):
        p102.choice("impact",state,bad,pool,1020111,0,0,0)


def test_p102_equal_confirmed_consequences_cold_and_negative_credit():
    world=p102.build_world("anisotropic","CAL",0,0)
    for arm in ("impact","geometry","hash_random","cyclic",
                "max_norm","impact_misaligned"):
        scored=p102.run_world(world,arm,1020227,0,beta=1.,
                              wrong_credit=(arm=="impact"))
        assert scored["confirmed_effects"]==16
        assert scored["persistent_float64_state"]==1024
        assert scored["cold_bitwise"]
        assert len(scored["action_journal_sha256"])==64
        assert 0 <= scored["mse_all"] < float("inf")
        if arm=="impact":
            assert 0 <= scored["wrong_credit_mse"] < float("inf")
    state=p101.initial()
    k=world.pool[0,0,0]
    consequence=world.maps[0]@k
    saved=p101.confirmed_write(state,k,torch.full_like(consequence,float("nan")),
                              confirmed=False)
    assert torch.equal(saved.weights,state.weights)
    assert torch.equal(saved.precision,state.precision)


def test_p102_full_frozen_multiseed_study_captures_original_ci_receipt(tmp_path):
    receipt=p102.study()
    assert receipt["status"] in (
        "DEV_FAIL_CONSEQUENCE_CONDITIONED_ACQUISITION_OR_COST",
        "DEV_CAL_ONLY_POSITIVE_NONPROMOTING_INDEPENDENT_FIRSTUSE_REQUIRED",
    )
    assert receipt["freeze_sha"]==p102.FREEZE
    assert receipt["freeze_blob"]==p102.FREEZE_BLOB
    assert receipt["scientific_pass"] is False
    assert receipt["native_hybrid_trained"] is False
    assert receipt["protected_OOD_generated"] is False
    assert receipt["protected_NEGATIVE_generated"] is False
    assert receipt["external_source_used"] is False
    assert receipt["original_model_edited"] is False
    assert len(receipt["source_sha256"])==18
    assert len(set(receipt["source_sha256"].values()))==18
    assert set(receipt["families"])==set(p102.FAMILIES)
    for family,group in receipt["families"].items():
        assert len(group["seeds"])==3
        for row in group["seeds"]:
            assert row["FIT_selected_beta"] in p102.BETAS
            assert row["FIT_selected_passive"] in p102.PASSIVE
            assert len(row["FIT_impact_grid"])==3
            assert len(row["FIT_passives"])==3
            assert set(row["ratios_impact_vs"])=={"geometry","FIT_selected_passive"}
            assert row["CAL_impact"]["total_confirmed_outcomes"]==192
            assert row["CAL_geometric"]["total_confirmed_outcomes"]==192
            assert row["CAL_selected_passive"]["total_confirmed_outcomes"]==192
            assert row["CAL_misaligned_ablation"]["total_confirmed_outcomes"]==192
            assert all(x["cold_bitwise"] for x in (
                row["CAL_impact"],row["CAL_geometric"],
                row["CAL_selected_passive"],row["CAL_misaligned_ablation"]))
            assert all(z["persistent_float64_state"]==1024 for z in (
                row["CAL_impact"],row["CAL_geometric"],
                row["CAL_selected_passive"],row["CAL_misaligned_ablation"]))
            assert row["cpu_ratio_vs_geometry"]>0
            assert row["cpu_ratio_vs_passive"]>0
    candidate=os.environ.get("EXPECTED_CANDIDATE_SHA")
    if candidate:
        receipt["capture_provenance"]={
            "candidate_sha":candidate,
            "github_run_id":os.environ.get("GITHUB_RUN_ID"),
            "classification":"FROZEN_AUTHOR_SYNTHETIC_DEV_NO_SCIENCE_ACCEPTANCE",
        }
        output=Path(__file__).resolve().parents[1]/"local_runs"/"p102_dev_result.json"
    else:
        output=tmp_path/"p102_dev_result.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True,
                                  allow_nan=False)+"\n",encoding="utf-8")
