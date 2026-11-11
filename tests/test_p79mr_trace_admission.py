"""P79-MR toy-fixture engineering tests: no external source ever scored."""
import copy
import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from p79mr_trace_admission import (
    SCHEMA, canonical_sha256, toy_manifest, validate_trace,
)
from p79mr_toy_transfer_replay import pinned_policy, toy_replay


def unit(axis, sign=1):
    x = [0.0] * 16
    x[axis] = float(sign)
    return x


def fixture():
    events = [
        {"event_id": i+1, "action": unit(i % 2),
         "effect": unit((i+1) % 2), "receipt_ref": f"fixture:event-{i+1}"}
        for i in range(6)
    ]
    probes = [
        {"probe_id": "early", "after_events": 2, "action": unit(0),
         "realized_effect": unit(1), "group": "stable",
         "receipt_ref": "fixture:early"},
        {"probe_id": "late", "after_events": 6, "action": unit(0),
         "realized_effect": unit(1, -1), "group": "changed",
         "receipt_ref": "fixture:late"},
    ]
    worlds = [{"world_id": "toy-world-a", "events": events, "probes": probes}]
    return toy_manifest(worlds), worlds


def test_manifest_schema_and_digest_do_not_claim_independent_science():
    manifest, worlds = fixture()
    receipt = validate_trace(manifest, worlds)
    assert manifest["schema"] == SCHEMA
    assert receipt["probes"] == 2 and receipt["worlds"] == 1
    assert receipt["worlds_sha256"] == canonical_sha256(worlds)
    assert receipt["admission"] == "SCHEMA_PASS_SCIENCE_UNVERIFIED"
    assert receipt["scientific_pass"] is False


def test_trace_digest_mismatch_rejects_mutation():
    manifest, worlds = fixture()
    worlds = copy.deepcopy(worlds)
    worlds[0]["events"][0]["effect"][0] += 0.1
    with pytest.raises(ValueError, match="P79MR_DIGEST_MISMATCH"):
        validate_trace(manifest, worlds)


@pytest.mark.parametrize("change,code", [
    ("extra_world_field", "P79MR_BAD_WORLD"),
    ("wrong_event_index", "P79MR_EVENT_ORDER"),
    ("duplicate_probe", "P79MR_DUPLICATE_PROBE"),
    ("bad_probe_cut", "P79MR_BAD_PROBE_CUT"),
    ("extra_hidden_role", "P79MR_BAD_EVENT"),
    ("missing_event_receipt", "P79MR_MISSING_EVENT_CUSTODY"),
    ("nonfinite_effect", "P79MR_BAD_EFFECT"),
    ("nonunit_action", "P79MR_ACTION_NOT_UNIT"),
])
def test_malformed_event_and_probe_fail_closed(change, code):
    _, worlds = fixture()
    worlds = copy.deepcopy(worlds)
    w = worlds[0]
    if change == "extra_world_field":
        w["hidden_phase"] = 7
    elif change == "wrong_event_index":
        w["events"][1]["event_id"] = 99
    elif change == "duplicate_probe":
        w["probes"][1]["probe_id"] = "early"
    elif change == "bad_probe_cut":
        w["probes"][1]["after_events"] = len(w["events"]) + 1
    elif change == "extra_hidden_role":
        w["events"][1]["oracle_role"] = 1
    elif change == "missing_event_receipt":
        w["events"][1]["receipt_ref"] = ""
    elif change == "nonfinite_effect":
        w["events"][2]["effect"][0] = float("nan")
        # Digest cannot include NaN; failure happens earlier during validation.
    elif change == "nonunit_action":
        w["events"][0]["action"][0] = 2.0
    manifest = toy_manifest(fixture()[1])
    with pytest.raises(ValueError, match=code):
        validate_trace(manifest, worlds)


def test_external_manifest_can_be_checked_but_never_scored():
    manifest, worlds = fixture()
    manifest["source_kind"] = "external_simulator"
    assert validate_trace(manifest, worlds)["scientific_pass"] is False
    with pytest.raises(ValueError, match="P79MR_OWNER_FIRST_USE_REQUIRED"):
        toy_replay(manifest, worlds)
    manifest["source_kind"] = "physical"
    with pytest.raises(ValueError, match="P79MR_OWNER_FIRST_USE_REQUIRED"):
        toy_replay(manifest, worlds)


def test_toy_replay_both_frozen_policies_without_training():
    manifest, worlds = fixture()
    for kind in ("fixed", "learned3"):
        out = toy_replay(manifest, worlds, kind=kind)
        assert out["classification"] == "TOY_ENGINEERING_ONLY"
        assert out["science_pass"] is False
        assert out["external_source_evaluated"] is False
        assert len(out["worlds"][0]["probes"]) == 2
        assert all(math.isfinite(p["mse_toy_only"])
                   for p in out["worlds"][0]["probes"])


@pytest.mark.parametrize("seed", (978201, 978203, 978207))
def test_three_pinned_learned_policies_are_finite_and_frozen(seed):
    model = pinned_policy("learned3", seed)
    assert model.theta.numel() == 3 and not model.theta.requires_grad
    assert all(math.isfinite(float(x)) for x in model.theta)
    with pytest.raises(ValueError, match="P79MR_BAD_POLICY"):
        pinned_policy("learned3", 978999)


def test_future_probe_targets_labels_and_receipts_never_change_predictions():
    manifest, worlds = fixture()
    original = toy_replay(manifest, worlds)
    copyworlds = copy.deepcopy(worlds)
    for probe in copyworlds[0]["probes"]:
        probe["realized_effect"] = [3.0] * 16
        probe["group"] = "unknown"
        probe["receipt_ref"] = "fixture:replaced-receipt"
    changed = toy_replay(toy_manifest(copyworlds), copyworlds)
    for x, y in zip(original["worlds"][0]["probes"],
                    changed["worlds"][0]["probes"]):
        assert x["estimate"] == y["estimate"]
    assert original["worlds"][0]["probes"][0]["mse_toy_only"] != (
        changed["worlds"][0]["probes"][0]["mse_toy_only"])


def test_future_observed_events_cannot_change_early_predictions():
    manifest, worlds = fixture()
    original = toy_replay(manifest, worlds)
    modified = copy.deepcopy(worlds)
    modified[0]["events"][4]["effect"] = [2.0] * 16
    out = toy_replay(toy_manifest(modified), modified)
    assert original["worlds"][0]["probes"][0]["estimate"] == (
        out["worlds"][0]["probes"][0]["estimate"])
    assert original["worlds"][0]["probes"][1]["estimate"] != (
        out["worlds"][0]["probes"][1]["estimate"])


@pytest.mark.parametrize("kind,seed", [("fixed", None), ("learned3", 978201)])
def test_tensor_only_midpoint_restore_bitwise(kind, seed):
    model = pinned_policy(kind, seed)
    _, worlds = fixture()
    events = worlds[0]["events"]
    facts = lambda es: {
        "action": torch.tensor([[x["action"] for x in es]]),
        "effect": torch.tensor([[x["effect"] for x in es]]),
    }
    with torch.no_grad():
        full = model.encode(facts(events))
        half = model.encode(facts(events[:3]))
        clone = half.clone()
        restored = model.encode(facts(events[3:]), carry=half)
    assert torch.equal(full, restored)
    assert torch.equal(clone, half)


def test_policy_custody_and_replay_reject_unregistered_parameters():
    with pytest.raises(ValueError, match="P79MR_BAD_POLICY"):
        pinned_policy("fixed", 978201)
    with pytest.raises(ValueError, match="P79MR_BAD_POLICY"):
        pinned_policy("oracle", 978201)


def test_external_source_has_no_science_execution_entrypoint():
    root = Path(__file__).resolve().parents[1] / "experiments"
    source = (root / "p79mr_toy_transfer_replay.py").read_text()
    assert "torch.save" not in source
    assert "if __name__" not in source
    assert "subprocess" not in source
    assert "requests." not in source
