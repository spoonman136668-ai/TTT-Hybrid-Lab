"""P104 independent-source replay, frozen source fences and original-chooser qualification."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p101_information_interventions_dev as p101
from experiments import p104_independent_active_design_dev as p104


def test_p104_prospective_freeze_and_original_model_custody():
    spec=p104.frozen()
    root=Path(__file__).resolve().parents[1]
    assert p104.FREEZE=="4b16585d437bc3903bdb28c4188e065679681a38"
    assert spec["identity"]["parent_pr"]==228
    assert spec["identity"]["parent_sha"]=="fcf260b966e8e6dd12a2dc1594103f5a5cc44725"
    assert spec["acceptance"]["scientific_pass"] is False
    assert p86.blob(root/"experiments/p104_independent_active_design_freeze.json")==p104.FREEZE_BLOB
    assert set(p104.FAMILIES)=={"isotropic","anisotropic","correlated"}
    for rel,blob in p104.PINNED.items():
        assert p86.blob(root/rel)==blob


def test_p104_new_independent_cohorts_and_sealed_sources():
    fingerprints=[]
    for fam in p104.FAMILIES:
        for split in p104.COUNTS:
            for index in range(3):
                fingerprints.append(p104.source_hash(fam,split,index))
    assert len(fingerprints)==18
    assert len(set(fingerprints))==18
    assert set(v for fam in p104.FAMILIES
               for split in p104.COUNTS for v in p104.SEEDS[fam][split]).isdisjoint(
               set(v for fam in p101.FAMILIES
                   for split in p101.COUNTS for v in p101.SEEDS[fam][split]))
    for arguments in (("isotropic","OOD",0,0),
                      ("anisotropic","NEGATIVE",0,0),
                      ("correlated","CAL",0,12),
                      ("isotropic","FIT",3,0),
                      ("unknown","FIT",0,0),
                      ("correlated","FIT",0,-1)):
        with pytest.raises(ValueError,match="P104_PROTECTED_OR_UNKNOWN_SOURCE_DENIED"):
            p104.build_world(*arguments)
    w=p104.build_world("correlated","CAL",0,0)
    assert w.raw.shape==(2,8,8,16)
    assert w.pool.shape==(2,8,8,16)
    assert w.queries.shape==(2,12,16)
    assert bool(torch.isfinite(w.pool).all())
    assert bool(torch.isfinite(w.queries).all())
    assert torch.equal(w.pool,p104.build_world("correlated","CAL",0,0).pool)


def test_p104_candidate_is_exact_unmodified_P101_action_chooser_and_RLS():
    world=p104.build_world("anisotropic","CAL",0,0)
    seed=p104.SEEDS["anisotropic"]["CAL"][0]
    original=p101.run_world(world,"max_variance",seed,0,wrong_credit=True)
    fresh=p104.one_world(world,"max_variance",seed,0,"anisotropic",True)
    assert original==fresh
    assert fresh["n_confirmed_effects"]==16
    assert fresh["persistent_floats"]==1024
    assert fresh["cold_task0_bitwise"] is True
    assert fresh["wrong_credit_mse"]>fresh["mse_all"]
    for policy in p104.POLICIES:
        z=p104.one_world(world,policy,seed,0,"anisotropic")
        assert z["n_confirmed_effects"]==16
        assert z["persistent_floats"]==1024
        assert z["cold_task0_bitwise"]
    model=p101.initial()
    first=world.pool[0,0]
    decision=p104.aoptimal_choice(model,first,"anisotropic")
    changed=world._replace(maps=(torch.ones_like(world.maps[0]),
                                 torch.zeros_like(world.maps[1])),
                           queries=torch.full_like(world.queries,float("nan")))
    assert decision==p104.aoptimal_choice(model,changed.pool[0,0],"anisotropic")
    with pytest.raises(ValueError,match="P104_AOPT_BAD_POOL"):
        p104.aoptimal_choice(model,torch.full_like(first,float("nan")),"anisotropic")


def test_p104_no_unconfirmed_or_query_target_learning():
    world=p104.build_world("isotropic","FIT",0,0)
    st=p101.initial()
    act=world.pool[0,0,0]
    effect=world.maps[0]@act
    next_state=p101.confirmed_write(st,act,effect)
    assert not torch.equal(next_state.weights,st.weights)
    bad=p101.confirmed_write(st,act,torch.full_like(effect,float("nan")),False)
    assert torch.equal(st.weights,bad.weights)
    assert torch.equal(st.precision,bad.precision)
    original=next_state.weights.clone(),next_state.precision.clone()
    _=p101.read(next_state,world.queries[0,0])
    assert torch.equal(next_state.weights,original[0])
    assert torch.equal(next_state.precision,original[1])
    with pytest.raises(ValueError,match="P101_UNQUALIFIED_CONFIRMED_EFFECT"):
        p101.confirmed_write(st,act,torch.full_like(effect,float("nan")))


def test_p104_complete_new_dev_replication_in_existing_G4_receipt(tmp_path):
    report=p104.study()
    assert report["status"] in (
        "DEV_FAIL_ANISOTROPIC_P101_REPLICATION_OR_COST",
        "DEV_ANISOTROPIC_PASSIVE_REPLICATION_ONLY_NOT_CLASSICAL",
        "DEV_ANISOTROPIC_REPLICATION_PLUS_CLASSICAL_PASS_NOT_SCIENTIFIC",
    )
    assert report["scientific_pass"] is False
    assert report["author_synthetic_only"] is True
    assert report["external_source_used"] is False
    assert report["reserved_OOD_generated"] is False
    assert report["reserved_negative_generated"] is False
    assert report["native_hybrid_trained"] is False
    assert report["original_model_modified"] is False
    assert report["freeze_commit"]==p104.FREEZE
    assert len(set(report["source_sha256"].values()))==18
    assert set(report["families"])==set(p104.FAMILIES)
    for fam,group in report["families"].items():
        assert len(group["seeds"])==3
        for row in group["seeds"]:
            assert row["FIT_selected_passive"] in p104.PASSIVES
            assert row["FIT_selected_classical"] in p104.CONVENTIONAL
            assert set(row["CAL_all_policies"])==set(p104.POLICIES)
            assert set(row["FIT_all_policies"])==set(p104.POLICIES)
            assert row["wrong_credit_ratio"]>=1.05
            assert all(z["total_confirmed_outcomes"]==12*16
                       and z["persistent_floats"]==1024
                       and z["cold_bitwise"]
                       for z in row["CAL_all_policies"].values())
            assert row["cpu_ratio_vs_a_opt"]>0
            assert row["cpu_ratio_vs_passive"]>0
    candidate=os.getenv("EXPECTED_CANDIDATE_SHA")
    if candidate:
        report["capture_provenance"]={
            "candidate_sha":candidate,
            "github_run_id":os.getenv("GITHUB_RUN_ID"),
            "classification":"FRESH_INDEPENDENT_AUTHOR_SYNTHETIC_P101_DEV_VALIDATION",
        }
        dest=Path(__file__).resolve().parents[1]/"local_runs"/"p104_dev_result.json"
    else:
        dest=tmp_path/"p104_dev_result.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf-8")
