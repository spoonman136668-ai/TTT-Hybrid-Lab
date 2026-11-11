"""P96 frozen source, new mechanism, negative controls, cold resume and DEV-only trial."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p96_diagonal_precision_dev as p96


def test_p96_protocol_is_frozen_before_new_source_and_originals_unchanged():
    root = Path(__file__).resolve().parents[1]
    doc = json.loads((root / "experiments/p96_diagonal_precision_frozen.json")
                     .read_text(encoding="utf-8"))
    assert doc["schema"] == "ttt-hybrid.p96-diagonal-precision-dev-freeze.v1"
    assert doc["parent"]["sha"] == "f9712ebb3195f889d7c4187363ba80aa592c7ef9"
    assert doc["seeds"]["OOD_RESERVED"] == 960117
    assert doc["seeds"]["NEGATIVE_RESERVED"] == 960119
    assert doc["prospective_dev_acceptance"]["all_two_families_required"] is True
    assert p96.FREEZE == "3750d547626f048711e24dd6cdadb2d101d75283"
    for path, sha in p96.ORIGINAL_BLOBS.items():
        assert p86.blob(root / path) == sha


def test_p96_new_sources_are_independent_and_reservation_is_fail_closed():
    hashes = []
    for name in ("A_FIT", "A_CAL", "B_FIT", "B_CAL"):
        w = p96.source(name)
        assert w["history"].shape == (
            p96.NUM_WORLDS[name.split("_")[1]], 48, 32)
        assert w["query"].shape[-1] == 32
        assert torch.count_nonzero(w["query"][:, 16:]) == 0
        assert p96.source_receipt(name, w)["audit"]["pass"]
        hashes.append(p86.world_sha(w))
    assert len(set(hashes)) == 4
    for name in ("OOD", "NEGATIVE", "VAL", "P95_CAL", ""):
        with pytest.raises(ValueError, match="P96_PROTECTED_OR_UNKNOWN_SOURCE"):
            p96.source(name)


def test_p96_precision_ingest_is_masked_and_cold_restorable():
    m = p96.DiagonalPrecisionCausalMemory()
    w = {key: value[:8].clone() for key, value in p96.source("B_CAL").items()}
    before = m.initial(w["query"][:, :16])
    result, state = m.episode(w)
    assert result.shape == (8, 16)
    assert state.weights.shape == (8, 16, 16)
    assert state.precision.shape == (8, 16)
    assert state.weights.numel() + state.precision.numel() == 8 * 272
    assert bool((state.precision >= 0).all())
    assert bool((state.confirmed_count == 48).all())
    assert torch.equal(before.precision, torch.zeros_like(before.precision))
    assert p96.cold_restore(m, w)
    assert p96.confirmation_negative(m, w)
    q = w["query"][:, :16]
    assert torch.equal(m.read_only(q, state), result)
    before_again = state.snapshot()
    shifted_q = q.roll(1, dims=0)
    _ = m.read_only(shifted_q, before_again)
    assert torch.equal(state.weights, before_again.weights)
    assert torch.equal(state.precision, before_again.precision)


def test_p96_unconfirmed_nan_cannot_learn_and_bad_confirmed_values_rejected():
    m = p96.DiagonalPrecisionCausalMemory()
    w = {key: value[:8].clone() for key, value in p96.source("A_CAL").items()}
    prior = m.initial(w["query"][:, :16])
    denied = torch.zeros((8, 48), dtype=torch.bool)
    poisoned = w["history"][:, :, 16:].clone()
    poisoned[:] = float("nan")
    empty = m.ingest(w["history"][:, :, :16], poisoned, denied, prior)
    assert torch.equal(empty.weights, prior.weights)
    assert torch.equal(empty.precision, prior.precision)
    assert torch.equal(empty.confirmed_count, prior.confirmed_count)
    with pytest.raises(ValueError, match="P96_NONFINITE_CONFIRMED_EFFECT"):
        m.ingest(w["history"][:, :, :16], poisoned, ~denied, prior)


def test_p96_full_study_keeps_dev_failure_nonpromoting_and_receipts_in_repo(tmp_path):
    report = p96.run()
    assert report["freeze_commit"] == p96.FREEZE
    assert report["scientific_pass"] is False
    assert report["protected_ood_generated"] is False
    assert report["protected_negative_generated"] is False
    assert report["original_model_modified"] is False
    assert report["learned_memory_floats"] == 272
    assert report["status"] in (
        "DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED",
        "DEV_FAIL_QUALITY_OR_COST_OR_CONTROLS",
    )
    assert set(report["families"]) == {"A", "B"}
    for label, value in report["families"].items():
        assert len(value["FIT_grid"]) == 9
        assert value["matched_memory_floats"] == 272
        assert value["hidden_switch_world_count"] == 64
        assert value["controls"]["bitwise_cold_restore"] is True
        assert value["controls"]["unconfirmed_nan_mask"] is True
        assert value["cost"]["CPU_ratio"] > 0
        assert value["CAL"]["fit_selected_W256"] == value["CAL"]["matched_272_inert_W256"]
    # Do not rerun this trial to collect metrics: keep results from same pytest
    # invocation using the repository's existing G4 local_runs/*.json artifact.
    candidate = os.environ.get("EXPECTED_CANDIDATE_SHA")
    if candidate:
        report["capture_provenance"] = {
            "candidate_sha": candidate,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "evidence": "AUTHOR_SYNTHETIC_DEV_ONLY_NOT_SCIENTIFIC_PASS",
        }
        dest = Path(__file__).resolve().parents[1] / "local_runs" / "p96_dev_result.json"
    else:
        dest = tmp_path / "p96_dev_result.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
