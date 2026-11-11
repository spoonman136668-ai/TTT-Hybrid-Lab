"""P99 freeze, confirmation, source custody, cold-resume and full DEV CAL falsifier."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97
from experiments import p99_event_checkpoint_dev as p99


def test_p99_freeze_before_source_and_original_blobs_unchanged():
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "experiments/p99_event_checkpoint_frozen.json")
                        .read_text(encoding="utf-8"))
    assert freeze["schema"] == "ttt-hybrid.p99-event-driven-checkpoint-dev-freeze.v1"
    assert freeze["identity"]["parent_sha"] == "d42629b3176feb3bed0e9f166cc53707f2a69237"
    assert freeze["new_source"]["seeds"]["OOD_RESERVED"] == 990117
    assert freeze["new_source"]["seeds"]["NEGATIVE_RESERVED"] == 990119
    assert freeze["frozen_acceptance"]["CPU_full_episode_ratio_to_fit_W256_max"] == 3.
    assert p99.FREEZE == "910ffd2d0fe68ec1f52d49b4e266e31db2d1736a"
    for path, git_blob in p99.BLOBS.items():
        assert p86.blob(root / path) == git_blob


def test_p99_fresh_data_is_distinct_and_fenced():
    hashes=[]
    for key in p99.SEEDS:
        w=p99.source(key)
        assert p97.audit(w)
        assert int(w["changed_world"].sum())==p99.N[key.split("_")[1]]//2
        assert w["history"].shape[-2:]==(72,32)
        assert w["probes"].shape[-2:]==(3,16)
        assert bool(torch.isfinite(w["targets"]).all())
        hashes.append(p97.fingerprint(w))
    assert len(set(hashes)) == 4
    assert len(set(p99.SEEDS.values()) & set(p97.SEEDS.values())) == 0
    for forbidden in ("OOD", "NEGATIVE", "A_OOD", "B_NEGATIVE", "P98_CAL", ""):
        with pytest.raises(ValueError, match="P99_PROTECTED_SOURCE_DENIED"):
            p99.source(forbidden)


def test_p99_lazy_checkpoint_confirmation_and_bitwise_cold_retention():
    w=p99.source("B_CAL")
    tiny={key:value[:8].clone() for key,value in w.items()}
    m=p99.EventCheckpointMemory(4.,2)
    outcome,state,route=m.episode(tiny["history"],tiny["probes"],trace=True)
    assert outcome.shape==(8,3,16)
    assert route.shape==(8,72)
    assert state.active.shape==(8,16,16)
    assert state.checkpoint.shape==(8,16,16)
    assert state.energy.shape==(8,)
    assert state.active.numel()+state.checkpoint.numel()+state.energy.numel()==8*513
    assert bool((state.step==72).all())
    assert bool(torch.isfinite(outcome).all())
    assert set(route.unique().tolist()) <= {0,1,2}
    negatives=p99.negative(m,w)
    assert negatives=={
        "cold_exact":True,"masked_nan_inert":True,"query_cannot_write":True,
    }


def test_p99_no_oracle_label_or_query_effect_credit_and_denied_observation():
    w=p99.source("A_CAL")
    tiny={key:value[:8].clone() for key,value in w.items()}
    m=p99.EventCheckpointMemory(6.,3)
    base,state=m.episode(tiny["history"],tiny["probes"])
    changed={key:value.clone() for key,value in tiny.items()}
    changed["changed_world"]=~tiny["changed_world"]
    changed["targets"][:]=float("nan")
    replay,state2=m.episode(changed["history"],changed["probes"])
    assert torch.equal(base,replay)
    assert torch.equal(state.active,state2.active)
    assert torch.equal(state.checkpoint,state2.checkpoint)
    zero=torch.zeros((8,72),dtype=torch.bool)
    poisoned=tiny["history"][:,:,16:].clone()
    poisoned[:]=float("nan")
    start=m.initial(tiny["probes"][:,0])
    rejected=m.ingest(tiny["history"][:,:,:16],poisoned,zero,start)
    assert all(torch.equal(getattr(rejected,k),getattr(start,k))
               for k in p99.CheckpointState.__dataclass_fields__)
    with pytest.raises(ValueError,match="P99_NONFINITE_CONFIRMED_EFFECT"):
        m.ingest(tiny["history"][:,:,:16],poisoned,~zero,start)


def test_p99_frozen_dev_result_preserved_to_existing_ci_artifact(tmp_path):
    report=p99.run()
    assert report["freeze_commit"] == p99.FREEZE
    assert report["scientific_pass"] is False
    assert report["external_source_used"] is False
    assert report["ood_generated"] is False
    assert report["negative_reserved_generated"] is False
    assert report["original_model_edited"] is False
    assert report["trained_neural_weights"] is False
    assert report["predictive_float_state"] == 513
    assert report["status"] in (
        "DEV_FAIL_QUALITY_OR_COST_OR_ROUTING",
        "DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED",
    )
    assert set(report["families"])=={"A","B"}
    assert len(set(report["source_sha256"].values()))==4
    for family,row in report["families"].items():
        assert len(row["FIT_grid"]) == 9
        assert len(row["FIT_candidate_grid"]) == 4
        assert len(row["FIT_prior_P97_grid"]) == 4
        assert "prior_P97_two_context" in row["CAL"]
        assert row["changed_world_count"] == 32
        assert row["matched_inert_513_baseline_parity"] is True
        assert all(row["negative_controls"].values())
        assert row["cpu"]["ratio"] > 0
        assert row["CAL"]["wrong_credit"]["all"] >= 0
        assert 0 <= row["routing"]["false_stable_switch_rate"] <= 1
    candidate = os.environ.get("EXPECTED_CANDIDATE_SHA")
    if candidate:
        report["capture_provenance"] = {
            "candidate_sha":candidate,
            "github_run_id":os.environ.get("GITHUB_RUN_ID"),
            "classification":"AUTHOR_SYNTHETIC_FROZEN_DEV_ONLY",
        }
        dest = Path(__file__).resolve().parents[1]/"local_runs"/"p99_dev_result.json"
    else:
        dest = tmp_path/"p99_dev_result.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
