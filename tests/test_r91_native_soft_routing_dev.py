"""R91 full-regression TEST-only native multiworld routing comparison."""
import json
import math

import numpy as np
import pytest

from experiments import r91_native_soft_routing_dev as r91


def test_r91_prospective_freeze_and_strict_source_fences():
    f=r91.frozen()
    assert r91.r85.git_blob(r91.FREEZE.read_bytes())==r91.FREEZE_BLOB
    assert f["parent"]["R90_candidate_SHA"]==r91.PARENT
    assert f["parent"]["R89_original_science"]=="R89_NATIVE_ROLE_ACCESS_NO_GO_BOTH_SEEDS"
    assert len(r91.FIT)==128 and len(r91.DEV_TEST)==64
    assert set(r91.FIT).isdisjoint(set(r91.DEV_TEST)|{0,1})
    assert f["sources"]["protected_science_sources_used"] is False
    assert f["sources"]["hidden_law_or_oracle_training_labels"] is False
    assert f["models"]["arms"]==list(r91.ARMS)
    assert not f["evaluation"]["independent_scientific_pass"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="existing_TEST_only_dev_training")
    assert f["authority"]["existing_TEST_only_dev_training"]
    with pytest.raises(PermissionError,match="R91_UNREGISTERED_TEST_INDEX"):
        r91._chosen(0)
    with pytest.raises(PermissionError,match="R89_ORIGINAL_FIRST_USE_SOURCE_NOT_ADMITTED"):
        r91.r89._world("FIT",r91.r89.SEEDS[0],0)


def test_r91_modified_protocol_rejected_before_any_source(monkeypatch,tmp_path):
    changed=json.loads(r91.FREEZE.read_text(encoding="utf-8"))
    changed["sources"]["protected_science_sources_used"]=True
    path=tmp_path/"altered.json"
    path.write_text(json.dumps(changed),encoding="utf-8")
    monkeypatch.setattr(r91,"FREEZE",path)
    with pytest.raises(RuntimeError,match="R91_FROZEN_DEVELOPMENT_SOURCE_DRIFT"):
        r91.frozen()


def test_r91_original_width_and_complete_test_only_development(monkeypatch):
    def forbidden(*_args,**_kwargs):
        raise AssertionError("R91_ACCESSED_PROTECTED_R89_SOURCE")
    monkeypatch.setattr(r91.r89,"_world",forbidden)
    result=r91.development_ci()
    assert result["status"] in ("R91_SOFT_ROUTING_DEV_SIGNAL_ONLY",
                                "R91_SOFT_ROUTING_DEV_NO_GO")
    assert result["scientific_pass"] is False
    assert result["real_PROTECTED_scientific_first_use"] is False
    assert result["original_R89_spent_or_sealed_source_used"] is False
    assert result["all_weights_frozen_before_DEV_TEST"]
    assert set(result["completed_arms"])==set(r91.ARMS)
    for arm,info in result["completed_arms"].items():
        assert info["steps"] == r91.UPDATES[arm]
        assert info["factual_selected_exposures"]==r91.UPDATES[arm]*r91.BATCH
        assert math.isfinite(info["final_train_MSE"])
        assert math.isfinite(info["initial_train_MSE"])
        assert info["wall_seconds"]>=0
        if arm in r91.NATIVE:
            assert info["parameter_count"]>=16818178
    assert (result["completed_arms"]["NATIVE_HARD_SLOTS5"]["parameter_count"]
            ==result["completed_arms"]["NATIVE_SOFT_SLOTS5"]["parameter_count"]
            ==16818178)
    assert result["completed_arms"]["NATIVE_SOFT_ROLE_READER48"]["parameter_count"]>16818178
    assert set(result["DEV_TEST"])==set(r91.ARMS)|{
        "ALWAYS_CALIBRATE","ALWAYS_EXPLOIT_POS2",
        "PUBLIC_A_RULE_EVALUATOR_ONLY","PHYSICAL_ORACLE_EVALUATOR_ONLY"}
    assert result["comparisons"]["checks"].keys()=={
        "soft_vs_hard_reward_at_least_0p5",
        "soft_vs_shuffled_reward_at_least_0p5",
        "soft_within_0p5_of_fully_trained_GRU64",
        "soft_lawful_agreement_at_least_65pct",
        "soft_oracle_regret_at_most_1p5"}.keys()
    assert all(math.isfinite(info["DEV_TEST_mean_actual_reward"])
               for info in result["DEV_TEST"].values())
    assert all(isinstance(v,bool) for v in result["comparisons"]["checks"].values())
    receipt=r91.ROOT/"local_runs/r91_native_soft_routing_dev.json"
    persisted=json.loads(receipt.read_text(encoding="utf-8"))
    assert persisted["schema"]==r91.SCHEMA
    assert persisted["freeze_blob"]==r91.FREEZE_BLOB
    assert persisted["status"]==result["status"]
