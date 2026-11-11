"""P95: frozen source/custody, confirmed-credit, cold and DEV-only negative controls."""
import json
import os
from pathlib import Path

import pytest
import torch

from experiments import p86_portable_causal_preflight as p86
from experiments import p95_native_residual_credit_dev as p95
from native_p94_confirmed_causal_memory import AnchoredHybridPilot, ConfirmedCausalMemory


def test_p95_source_and_freeze_custody():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    freeze = json.loads((root / "experiments/p95_native_residual_credit_frozen.json").read_text())
    assert freeze["schema"] == "ttt-hybrid.p95-native-residual-credit-falsifier-freeze.v1"
    assert freeze["parent"]["sha"] == "f9712ebb3195f889d7c4187363ba80aa592c7ef9"
    assert freeze["data"]["seeds"]["OOD_RESERVED"] == 950117
    assert freeze["data"]["seeds"]["NEGATIVE_RESERVED"] == 950119
    for path, sha in p95.ORIGINAL_BLOBS.items():
        assert p86.blob(root / path) == sha


def test_p95_fresh_sources_have_exact_platform_identical_sha():
    assert p95.check_cohort("FIT", p95.source("FIT"))["sha256"] == p95.EXPECTED_SHA["FIT"]
    assert p95.check_cohort("CAL", p95.source("CAL"))["sha256"] == p95.EXPECTED_SHA["CAL"]
    for forbidden in ("OOD", "NEGATIVE", "HELDOUT", "VAL", ""):
        with pytest.raises(ValueError, match="P95_RESERVED_SOURCE_DENIED"):
            p95.source(forbidden)


def test_p95_parent_W256_full_state_parity_and_cold_retention():
    world = {k: v[:8] for k, v in p95.source("CAL").items()}
    got, state = ConfirmedCausalMemory().forward_episode(
        world["history"], world["query"][:, :16])
    expected, w = p86.online(world, 1., .99)
    assert torch.equal(got, expected)
    assert torch.equal(state.fast_weights, w)
    assert state.fast_weights.numel() == 8 * 256
    assert p95.cold_restore(world)


def test_p95_query_outcome_never_written_and_unconfirmed_effect_masked():
    world = {k: v[:8] for k, v in p95.source("CAL").items()}
    mem = ConfirmedCausalMemory()
    history, q = world["history"], world["query"][:, :16]
    mask = torch.ones((8, 48), dtype=torch.bool)
    mask[:, 1::3] = False
    tampered = history.clone()
    tampered[:, 1::3, 16:] = float("nan")
    a, state_a = mem.forward_episode(history, q, mask)
    b, state_b = mem.forward_episode(tampered, q, mask)
    assert torch.equal(a, b)
    assert torch.equal(state_a.fast_weights, state_b.fast_weights)
    new_query = q.roll(1, dims=0)
    _, state_c = mem.forward_episode(history, new_query, mask)
    assert torch.equal(state_a.fast_weights, state_c.fast_weights)


def test_p95_initial_native_residual_agrees_with_strong_W256_and_receives_gradient():
    w = {k: v[:4] for k, v in p95.source("CAL").items()}
    torch.manual_seed(950201)
    m = AnchoredHybridPilot()
    expected, _ = p86.online(w, 1., .99)
    with torch.no_grad():
        assert torch.equal(m(w["history"], w["query"][:, :16]), expected)
        assert torch.equal(p95.model_prediction(m, w), expected)
    loss = (p95.model_prediction(m, w) - w["effect"]).square().mean()
    loss.backward()
    assert m.residual_scale.grad is not None
    assert bool(torch.isfinite(m.residual_scale.grad))


def test_p95_full_frozen_dev_study_nonpromoting(tmp_path):
    # This is a prospective development falsifier; a genuine negative is PASS
    # for engineering. Never assert that the scientific hypothesis succeeded.
    report = p95.run()
    (tmp_path / "p95_dev_result.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert report["freeze_commit"] == p95.FREEZE
    assert report["scientific_pass"] is False
    assert report["protected_OOD_generated"] is False
    assert report["protected_NEGATIVE_generated"] is False
    assert report["original_model_modified"] is False
    assert report["status"] in (
        "DEV_CAL_ONLY_PASS_PROTECTED_FIRST_USE_REQUIRED",
        "DEV_FAIL_QUALITY_OR_COST",
    )
    assert len(report["models"]) == 3
    assert all(row["updates"] == 300 for row in report["models"])
    assert report["cohorts"]["FIT"]["sha256"] == p95.EXPECTED_SHA["FIT"]
    assert report["cohorts"]["CAL"]["sha256"] == p95.EXPECTED_SHA["CAL"]
    assert report["cpu_ratio"] > 0
    # Preserve the numerical result from this exact test invocation in the
    # existing G4 artifact; do not train a second time or alter the protocol.
    candidate = os.environ.get("EXPECTED_CANDIDATE_SHA")
    if candidate:
        report["capture_provenance"] = {
            "candidate_sha": candidate,
            "github_run_id": os.environ.get("GITHUB_RUN_ID"),
            "purpose": "DEV_ONLY_NUMERICAL_EVIDENCE_NONPROMOTING",
        }
        output = Path(__file__).resolve().parents[1] / "local_runs" / "p95_dev_result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
