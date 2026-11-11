"""P98 prospective blind routing diagnostic; author-synthetic DEV only."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p97_recurrent_context_dev as p97
from experiments import p98_blind_switch_diagnostics as p98


def test_p98_freeze_source_isolation_and_unmodified_P97():
    f = p98.frozen_protocol()
    assert f["schema"] == "ttt-hybrid.p98-blinded-context-switch-diagnostics-dev-freeze.v1"
    assert f["source"]["worlds"] == {"FIT":128,"CAL":64}
    assert f["source"]["seeds"]["OOD_RESERVED"] == 980117
    assert f["source"]["seeds"]["NEGATIVE_RESERVED"] == 980119
    assert f["candidate"]["no_change_to_existing_P97"] is True
    assert p98.SEEDS == {"A_FIT":980111,"A_CAL":980113,"B_FIT":980211,"B_CAL":980213}
    assert len({*p98.SEEDS.values(), *p97.SEEDS.values()}) == 8


def test_p98_new_source_is_factual_without_protected_birth():
    source_hashes=[]
    for key in p98.SEEDS:
        c=p98.source(key)
        assert p97.audit(c)
        assert int(c["changed_world"].sum()) == len(c["history"])//2
        assert c["history"].shape[-2:] == (72,32)
        assert c["probes"].shape[-2:] == (3,16)
        source_hashes.append(p97.fingerprint(c))
    assert len(set(source_hashes))==4
    for invalid in ("OOD","NEGATIVE","A_OOD","B_NEGATIVE","P97_CAL",""):
        with pytest.raises(ValueError, match="P98_PROTECTED_OR_UNKNOWN_SOURCE_DENIED"):
            p98.source(invalid)


def test_p98_step_telemetry_is_bitwise_equal_to_original_P97():
    c=p98.source("A_CAL")
    tiny={k:v[:4] for k,v in c.items()}
    model=p97.RecurrentContextMemory(2.,.7)
    pred,route,state=p98.route_trace(model,tiny)
    assert pred.shape == (4,3,16)
    assert route.shape==(4,72)
    assert set(route.unique().tolist()) <= {0,1}
    assert bool((state.confirmed_count==72).all())
    diag=p98.route_diagnostics(pred,route,tiny)
    assert diag["stable_worlds"]+diag["changed_worlds"]==4
    assert diag["per_event_vs_continuous_parity"] is True
    assert 0 <= diag["stable_false_switch_rate"] <= 1
    assert 0 <= diag["changed_detect_within_12_rate"] <= 1
    assert 0 <= diag["return_detect_within_12_rate"] <= 1


def test_p98_stable_false_switch_audit_is_posthoc_only():
    c=p98.source("B_CAL")
    tiny={k:v[:4] for k,v in c.items()}
    m=p97.RecurrentContextMemory(3.,.5)
    y,route,state=p98.route_trace(m,tiny)
    altered={k:v.clone() for k,v in tiny.items()}
    altered["changed_world"][:] = ~altered["changed_world"]
    altered["targets"][:] = float("nan")
    y2,route2,state2=p98.route_trace(m,altered)
    assert torch.equal(y,y2)
    assert torch.equal(route,route2)
    assert torch.equal(state.weights,state2.weights)
    # Oracle labels are consumed by score/audit only, never by causal write.
    diag=p98.route_diagnostics(y,route,tiny)
    assert diag["stable_worlds"]+diag["changed_worlds"]==4


def test_p98_frozen_dev_diagnostic_with_exact_ci_artifact(tmp_path):
    report=p98.run()
    assert report["frozen_before_source_commit"]==p98.FREEZE_COMMIT
    assert report["science_pass"] is False
    assert report["real_external_source_used"] is False
    assert report["ood_generated"] is False
    assert report["negative_reserved_generated"] is False
    assert report["original_model_edited"] is False
    assert report["new_model_trained"] is False
    assert report["candidate_predictive_state_floats"]==514
    assert report["status"] in (
        "DEV_DIAGNOSTIC_HYPOTHESIS_FAIL",
        "DEV_DIAGNOSTIC_PASS_STILL_NOT_SCIENCE",
    )
    assert len(set(report["source_sha256"].values()))==4
    for key in ("A","B"):
        row=report["families"][key]
        assert len(row["FIT_candidate_grid"])==4
        assert len(row["FIT_control_grid"])==9
        assert row["blinded_route_diagnostics"]["stable_worlds"]==32
        assert row["blinded_route_diagnostics"]["changed_worlds"]==32
        assert all(row["negative_controls"].values())
        assert row["cpu"]["P97_W256_ratio"]>0
    sha=os.getenv("EXPECTED_CANDIDATE_SHA")
    if sha:
        report["capture_provenance"]={
            "candidate_sha":sha,
            "github_run_id":os.getenv("GITHUB_RUN_ID"),
            "classification":"AUTHOR_SYNTHETIC_DEV_DIAGNOSTIC_NOT_SCIENTIFIC_PASS",
        }
        path=Path(__file__).resolve().parents[1]/"local_runs"/"p98_dev_result.json"
    else:
        path=tmp_path/"p98_dev_result.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
