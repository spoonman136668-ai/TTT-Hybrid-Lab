"""P88 bounded confirmed-only native TTT parity and causal-source guards.

Engineering interface tests only. The offline 15-arm negative is not rerun.
"""
import json
from pathlib import Path
import pytest
import torch
from model import HybridPilot, Configuration
from experiments import p86_portable_causal_preflight as p86
from native_p88_prequential_confirmation import confirmed_mask, native_ttt_confirmed, paired_stream

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)


def _fixture():
    w = p86.source("CAL", 8)  # reused P86 calibration only for interface testing
    return w, paired_stream(w["history"], w["query"])


def _model():
    torch.manual_seed(880201)
    return HybridPilot(32, 16, Configuration(width=16, heads=2, ssm=True, attention=True, ttt=True, depth=True)).eval()


def test_original_source_git_blob_custody():
    assert p86.blob(ROOT / "model.py") == "5d9ea57b827aa1796548447868fa5ee14c251b49"
    assert p86.blob(ROOT / "native_p78_bounded_raw_ttt.py") == "db948c53bb1a28af5fe0e6a62ef71463db6d95ba"


def test_action_query_precedes_confirmed_effect_no_leak():
    w, seq = _fixture()
    assert seq.shape == (8, 97, 32)
    assert torch.count_nonzero(seq[:, ::2, 16:]) == 0
    assert torch.equal(seq[:, 1::2], w["history"])
    assert torch.equal(seq[:, -1, :], w["query"])
    bad = w["query"].clone()
    bad[:, 16:] = 1
    with pytest.raises(ValueError, match="P88_FUTURE_EFFECT_NOT_HIDDEN"):
        paired_stream(w["history"], bad)


def test_unknown_or_malformed_confirmation_refused():
    with pytest.raises(ValueError, match="P88_EXPECTED_97_INTERLEAVED_TOKENS"):
        confirmed_mask(96)
    m = _model()
    _, seq = _fixture()
    with pytest.raises(ValueError, match="P88_INVALID_FACT_CONFIRMATION_MASK"):
        native_ttt_confirmed(m.ttt, m.stem(seq), torch.ones(97))


def test_adapter_exact_original_when_all_writes_enabled():
    _, seq = _fixture()
    m = _model()
    with torch.no_grad():
        z = m.attn(m.ssm(m.stem(seq)))
        original = m.ttt(z)
        from_adapter, _ = native_ttt_confirmed(m.ttt, z, torch.ones(97, dtype=torch.bool))
    torch.testing.assert_close(original, from_adapter, rtol=0, atol=0)


def test_only_48_confirmed_events_affect_fast_weight_state():
    _, seq = _fixture()
    m = _model()
    with torch.no_grad():
        z = m.attn(m.ssm(m.stem(seq)))
        _, guarded = native_ttt_confirmed(m.ttt, z, confirmed_mask(97))
        _, only_facts = native_ttt_confirmed(m.ttt, z[:, 1::2], torch.ones(48, dtype=torch.bool))
        _, unguarded = native_ttt_confirmed(m.ttt, z, torch.ones(97, dtype=torch.bool))
    assert guarded.shape == (8, 16, 16)
    torch.testing.assert_close(guarded, only_facts, rtol=0, atol=0)
    assert not torch.allclose(guarded, unguarded, rtol=1e-5, atol=1e-6)


def test_future_confirmed_effect_does_not_leak_to_current_query():
    _, seq = _fixture()
    changed = seq.clone()
    changed[:, 2*20+1, 16:] *= -1  # fact after query 20
    m = _model()
    def preview(x):
        z = m.attn(m.ssm(m.stem(x)))
        z, _ = native_ttt_confirmed(m.ttt, z, confirmed_mask(97))
        return m.head(m.depth(z))[:, ::2]
    with torch.no_grad():
        original, altered = preview(seq), preview(changed)
    torch.testing.assert_close(original[:, :21], altered[:, :21], rtol=1e-6, atol=1e-6)


def test_negative_scientific_receipt_and_unqualified_monolith_preserved():
    r = json.loads((ROOT / "experiments/p88_prequential_ttt_local_negative_receipt.json").read_text())
    assert r["decision"]["status"] == "DEV_FAIL_FROZEN_PREQUENTIAL_OR_STRONG_CONTROL_AND_UNQUALIFIED_MONOLITH"
    assert r["execution"]["individual_arm_shards"] == 15
    assert r["execution"]["monolithic_attempts_timed_out"] == 2
    assert not r["execution"]["monolithic_CPU_wall_qualified"]
    assert not r["decision"]["scientific_pass"] and not r["decision"]["OOD_generated"]
    assert len(r["rows"]) == 15
    assert all(len(row["validation_final_mse_steps_20_40_60_80_100_120"]) == 6 for row in r["rows"])
    assert all(row["best_final_mse"] > r["controls"]["P86_CAL_locked_W256_eta1_decay099"]["final"] for row in r["rows"])
