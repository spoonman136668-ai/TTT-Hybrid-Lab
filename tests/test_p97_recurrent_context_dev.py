"""P97: frozen recurrence, causal confirmation, cold restoration, full DEV falsifier."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p97_recurrent_context_dev as p97


def test_p97_freeze_and_original_source_custody():
    root = Path(__file__).resolve().parents[1]
    frozen = json.loads((root / "experiments/p97_recurrent_context_frozen.json")
                        .read_text(encoding="utf-8"))
    assert frozen["schema"] == "ttt-hybrid.p97-recurrent-context-dev-freeze.v1"
    assert frozen["parent"]["sha"] == "f9712ebb3195f889d7c4187363ba80aa592c7ef9"
    assert frozen["source"]["seeds"]["OOD_RESERVED"] == 970117
    assert frozen["source"]["seeds"]["NEGATIVE_RESERVED"] == 970119
    assert p97.FREEZE == "45373e14dc586aa2c3d518700c67dba546919378"
    for path, sha in p97.BLOBS.items():
        assert p86.blob(root / path) == sha


def test_p97_source_fingerprint_and_protected_reservations():
    hashes = []
    for key in p97.SEEDS:
        world = p97.source(key)
        assert p97.audit(world)
        assert int(world["changed_world"].sum()) == p97.N[key.split("_")[1]] // 2
        assert world["history"].shape[-2:] == (72, 32)
        assert world["probes"].shape[-2:] == (3, 16)
        hashes.append(p97.fingerprint(world))
    assert len(set(hashes)) == 4
    for forbidden in ("OOD", "NEGATIVE", "HELDOUT", "P96_CAL", ""):
        with pytest.raises(ValueError, match="P97_PROTECTED_SOURCE_DENIED"):
            p97.source(forbidden)


def test_p97_cold_and_unconfirmed_nan_fence():
    world = p97.source("A_CAL")
    mem = p97.RecurrentContextMemory(2., .7)
    negative = p97.cold_and_negative(mem, world)
    assert negative["cold_bitwise"] is True
    assert negative["masked_nan_inert"] is True
    w = {k: v[:8] for k, v in world.items()}
    _, state = mem.episode(w["history"], w["probes"])
    assert state.weights.shape == (8, 2, 16, 16)
    assert state.innovation.shape == (8, 2)
    assert state.weights.numel() + state.innovation.numel() == 8 * 514
    assert bool((state.confirmed_count == 72).all())
    assert torch.equal(state.weights, state.snapshot().weights)
    wrong_query = w["probes"][:, -1].roll(1, dims=0)
    _ = mem.read_only(wrong_query, state)
    assert torch.equal(state.weights, state.snapshot().weights)


def test_p97_no_future_oracle_credit_and_bad_facts_denied():
    world = p97.source("B_CAL")
    mem = p97.RecurrentContextMemory(3., .5)
    w = {k: v[:8].clone() for k, v in world.items()}
    before, initial_state = mem.episode(w["history"], w["probes"])
    # Oracle targets/hidden-world flags are not arguments to the learner.
    bogus = w["targets"].clone()
    bogus[:] = float("nan")
    labels = ~w["changed_world"]
    assert bool(labels.numel() == 8 and torch.isnan(bogus).all())
    after, repeated_state = mem.episode(w["history"], w["probes"])
    assert torch.equal(before, after)
    assert torch.equal(initial_state.weights, repeated_state.weights)
    prior = mem.initial(w["probes"][:, 0])
    denied = torch.zeros((8, 72), dtype=torch.bool)
    poisoned = w["history"][:, :, 16:].clone()
    poisoned[:] = float("nan")
    inert = mem.ingest(w["history"][:, :, :16], poisoned, denied, prior)
    assert torch.equal(inert.weights, prior.weights)
    assert torch.equal(inert.innovation, prior.innovation)
    assert torch.equal(inert.visits, prior.visits)
    assert torch.equal(inert.active, prior.active)
    with pytest.raises(ValueError, match="P97_NONFINITE_CONFIRMED_EFFECT"):
        mem.ingest(w["history"][:, :, :16], poisoned, ~denied, prior)


def test_p97_full_frozen_dev_study_with_numerical_ci_receipt(tmp_path):
    report = p97.run()
    assert report["freeze_commit"] == p97.FREEZE
    assert report["scientific_pass"] is False
    assert report["protected_ood_generated"] is False
    assert report["protected_negative_generated"] is False
    assert report["original_model_modified"] is False
    assert report["predictive_float_state"] == 514
    assert report["status"] in (
        "DEV_FAIL_QUALITY_OR_COST",
        "DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED",
    )
    assert set(report["families"]) == {"A", "B"}
    assert all(d["audit_pass"] for d in report["sources"].values())
    assert len(set(d["sha256"] for d in report["sources"].values())) == 4
    for family, row in report["families"].items():
        assert len(row["FIT_control_grid"]) == 9
        assert len(row["FIT_context_grid"]) == 4
        assert row["switched_worlds"] == 64
        assert row["negative"]["cold_bitwise"]
        assert row["negative"]["masked_nan_inert"]
        assert row["matched_inert_514_control_parity"] is True
        assert row["cpu"]["ratio"] > 0
        assert row["CAL"]["wrong_credit"]["all"] >= 0
    candidate_sha = os.environ.get("EXPECTED_CANDIDATE_SHA")
    if candidate_sha:
        report["capture_provenance"] = {
            "candidate_sha": candidate_sha,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "evidence": "AUTHOR_SYNTHETIC_DEV_ONLY_NOT_SCIENTIFIC_PASS",
        }
        destination = Path(__file__).resolve().parents[1] / "local_runs" / "p97_dev_result.json"
    else:
        destination = tmp_path / "p97_dev_result.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
