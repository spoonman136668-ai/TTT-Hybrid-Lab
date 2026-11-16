"""R90 source custody and full-regression diagnostic on R89 TEST-only worlds."""
import json
import math

import pytest

from experiments import r90_native_credit_path_diagnostic as r90


def test_r90_source_freeze_predates_implementation_and_denies_first_use():
    contract = r90.frozen()
    assert r90.r85.git_blob(r90.FREEZE.read_bytes()) == r90.FREEZE_BLOB
    assert contract["parent"]["R89_candidate_sha"] == r90.PARENT
    assert contract["parent"]["R89_original_result"] == "R89_NATIVE_ROLE_ACCESS_NO_GO_BOTH_SEEDS"
    assert contract["source"]["only"] == "R89_EXISTING_TEST_WORLD_GENERATOR_25319991"
    assert contract["source"]["indexes"] == [0, 1]
    assert contract["design"]["training_steps_width1024"] == 0
    assert contract["future_science"]["new_independent_source_required"]
    assert contract["future_science"]["freeze_before_first_use"]
    assert not any(value for key, value in contract["authority"].items()
                   if key != "dev_cpu_test_only")
    assert contract["authority"]["dev_cpu_test_only"]
    with pytest.raises(PermissionError, match="R89_ORIGINAL_FIRST_USE_SOURCE_NOT_ADMITTED"):
        r90.r89._world("FIT", r90.r89.SEEDS[0], 0)


def test_r90_tampered_freeze_fails_closed(tmp_path, monkeypatch):
    data = json.loads(r90.FREEZE.read_text(encoding="utf-8"))
    data["authority"]["external_protected_science"] = True
    fake = tmp_path / "r90_modified.json"
    fake.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(r90, "FREEZE", fake)
    with pytest.raises(RuntimeError, match="R90_PROSPECTIVE_DIAGNOSTIC_FREEZE_DRIFT"):
        r90.frozen()


def test_r90_original_native_gradients_and_64_step_tiny_fit(monkeypatch):
    # The diagnostic must not accidentally invoke original protected R89
    # FIT/VALID/DEV/OOD even when the local environment is misconfigured.
    def forbidden(*_args, **_kwargs):
        raise AssertionError("R90_ATTEMPTED_PROTECTED_R89_SOURCE")
    monkeypatch.setattr(r90.r89, "_world", forbidden)
    result = r90.diagnostic()
    assert result["status"] == "R90_ENGINEERING_DIAGNOSTIC_OBSERVED_NOT_SCIENCE"
    assert result["only_existing_R89_TEST_world_indexes"] == [0, 1]
    assert result["freeze_blob"] == r90.FREEZE_BLOB
    assert result["parent_R89_candidate"] == r90.PARENT
    assert set(result["diagnostics"]) == {"32", "1024"}
    for width in ("32", "1024"):
        arms = result["diagnostics"][width]
        assert set(arms) == set(r90.ARMS)
        for arm, receipt in arms.items():
            assert receipt["updates"] == (64 if width == "32" else 0)
            assert math.isfinite(receipt["initial"]["loss"])
            assert math.isfinite(receipt["final_mse"])
            assert math.isfinite(receipt["final_over_initial"])
            assert receipt["initial"]["parameters"] > 0
            assert "stem" in receipt["initial"]["gradients"]
            assert "ssm" in receipt["initial"]["gradients"]
            assert "attn" in receipt["initial"]["gradients"]
            assert "ttt" in receipt["initial"]["gradients"]
            assert "head" in receipt["initial"]["gradients"]
            for entry in receipt["initial"]["gradients"].values():
                assert math.isfinite(entry["grad_l2"])
                assert entry["tensors"] >= entry["missing"] + entry["zero"]
        assert (arms["HARD_ROUTER"]["initial"]["parameters"]
                == arms["SOFT_ROUTER"]["initial"]["parameters"])
        assert (arms["HARD_ROUTER"]["initial"]["parameters"]
                > arms["NO_DEPTH"]["initial"]["parameters"])
        if width == "1024":
            assert arms["HARD_ROUTER"]["initial"]["parameters"] == 16818178
    reader = result["public_reader_width32"]
    assert math.isfinite(reader["loss"])
    assert set(reader["gradient_norms"]) == {
        "query", "role_strength_raw", "key.0.weight", "key.2.weight",
        "native.stem.weight", "native.head.weight"}
    assert all(v is None or math.isfinite(v)
               for v in reader["gradient_norms"].values())
    receipt = r90.ROOT / "local_runs/r90_native_credit_path_diagnostic.json"
    assert json.loads(receipt.read_text(encoding="utf-8"))["schema"] == r90.SCHEMA
