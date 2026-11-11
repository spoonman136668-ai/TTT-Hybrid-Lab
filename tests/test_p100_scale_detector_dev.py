"""P100: frozen sources, detector timing, parity, negative controls and DEV result."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97
from experiments import p100_scale_detector_dev as p100


def test_p100_prospectively_frozen_contract_and_unchanged_originals():
    root = Path(__file__).resolve().parents[1]
    f = json.loads((root/"experiments/p100_scale_detector_freeze.json").read_text())
    assert f["schema"] == "ttt-hybrid.p100-scale-normalized-causal-change-detector-freeze.v1"
    assert f["origin"]["parent_sha"] == "5f932c24f37c396e7d1d0f89f40321303a14be24"
    assert f["source"]["seeds"]["OOD_RESERVED"] == 1000117
    assert f["source"]["seeds"]["NEGATIVE_RESERVED"] == 1000119
    assert f["scope"].startswith("Detector-only")
    assert p100.FREEZE == "0d85bf542c425c02e40a1035123a94f4247a527c"
    assert p86.blob(root/"experiments/p100_scale_detector_freeze.json") == p100.FREEZE_BLOB
    for path, blob in p100.BLOBS.items():
        assert p86.blob(root/path) == blob


def test_p100_new_sources_unique_audited_and_protected():
    hashes = []
    for key in p100.SEEDS:
        w = p100.source(key)
        assert p97.audit(w)
        assert w["history"].shape == (p100.N[key.split("_")[1]],72,32)
        assert int(w["changed_world"].sum()) == len(w["history"])//2
        hashes.append(p97.fingerprint(w))
    assert len(set(hashes)) == 4
    for forbidden in ("OOD","NEGATIVE","A_OOD","B_NEGATIVE","P99_CAL",""):
        with pytest.raises(ValueError, match="P100_PROTECTED_SOURCE_DENIED"):
            p100.source(forbidden)


def test_p100_causal_memory_exact_parity_and_negative_controls():
    w = p100.source("A_CAL")
    h = w["history"][:8]
    state, alarms, pred = p100.stream(h,2.,2)
    assert state.W.shape == (8,16,16)
    assert state.energy.shape == (8,)
    assert state.W.numel()+state.energy.numel() == 8*257
    assert bool((state.count==72).all())
    assert torch.equal(state.W,p100.fixed_w256(h))
    assert alarms.shape == (8,72) and pred.shape == (8,72,16)
    assert not bool(alarms[:,:16].any())
    assert p100.negative((2.,2),w) == {
        "bitwise_cold_resume":True, "unconfirmed_nan_inert":True,
    }
    _, static, _ = p100.stream(h,2.,2,adaptive=False)
    assert static.shape == alarms.shape
    # Oracle source labels and withheld probes have no influence on detector.
    modified = {k:v.clone() for k,v in w.items()}
    modified["changed_world"] = ~modified["changed_world"]
    modified["targets"][:] = float("nan")
    x, a, y = p100.stream(modified["history"][:8],2.,2)
    assert torch.equal(x.W,state.W) and torch.equal(a,alarms)
    assert torch.equal(y,pred)


def test_p100_detector_alarm_events_are_not_prior_occupancy():
    w = {"changed_world":torch.tensor([True,False],dtype=torch.bool)}
    alarm = torch.zeros((2,72),dtype=torch.bool)
    alarm[0,20] = True
    alarm[0,45] = True
    early = p100.detection(alarm,w)
    assert early["middle_entry_count"] == 0
    assert early["return_entry_count"] == 0
    assert early["stable_false_alarm_count"] == 0
    alarm[0,20] = False
    alarm[0,45] = False
    alarm[0,27] = True
    alarm[0,50] = True
    alarm[1,33] = True
    correct = p100.detection(alarm,w)
    assert correct["middle_entry_count"] == 1
    assert correct["return_entry_count"] == 1
    assert correct["stable_false_alarm_count"] == 1


def test_p100_invalid_and_unconfirmed_cannot_write_or_alarm():
    w = p100.source("B_CAL")
    h = w["history"][:8]
    previous = p100.initial(h)
    mask = torch.zeros((8,72),dtype=torch.bool)
    bad = h.clone()
    bad[:,:,16:] = float("nan")
    state,alarm,_ = p100.stream(bad,1.5,2,confirmed=mask)
    assert all(torch.equal(getattr(state,k),getattr(previous,k))
               for k in p100.DetectorState.__dataclass_fields__)
    assert not bool(alarm.any())
    with pytest.raises(ValueError,match="P100_NONFINITE_CONFIRMED_EFFECT"):
        p100.stream(bad,1.5,2,confirmed=~mask)


def test_p100_full_prospective_dev_result_and_existing_ci_receipt(tmp_path):
    result = p100.run()
    assert result["freeze_commit"] == p100.FREEZE
    assert result["science_pass"] is False
    assert result["detector_only"] is True
    assert result["causal_recovery_proven"] is False
    assert result["ood_generated"] is False
    assert result["negative_generated"] is False
    assert result["external_real_source_used"] is False
    assert result["original_models_modified"] is False
    assert result["persistent_float_state"] == 257
    assert result["status"] in (
        "DEV_FAIL_CHANGE_DETECTION_OR_COST",
        "DEV_DETECTOR_ONLY_PASS_NEEDS_RECOVERY_AND_EXTERNAL_FIRST_USE",
    )
    assert set(result["families"]) == {"A","B"}
    assert len(set(result["source_sha256"].values())) == 4
    for fam, row in result["families"].items():
        assert len(row["FIT_grid_normalized"]) == 6
        assert len(row["FIT_grid_static"]) == 6
        assert len(row["FIT_grid_p99"]) == 4
        assert row["stable_world_count"] == 32
        assert row["changed_world_count"] == 32
        assert all(row["negative_controls"].values())
        assert row["wrong_credit_ratio"] >= 1.05
        assert row["CPU"]["ratio"] > 0
        assert set(row["performance_gates"]) == {
            "stable_false","B_entry","A_return",
            "better_than_static","whole_episode_cost",
        }
    sha = os.getenv("EXPECTED_CANDIDATE_SHA")
    if sha:
        result["capture_provenance"] = {
            "candidate_sha":sha,"github_run_id":os.getenv("GITHUB_RUN_ID"),
            "classification":"AUTHOR_SYNTHETIC_DETECTOR_ONLY_DEV_NONPROMOTING",
        }
        dest = Path(__file__).resolve().parents[1]/"local_runs"/"p100_dev_result.json"
    else:
        dest = tmp_path/"p100_dev_result.json"
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
