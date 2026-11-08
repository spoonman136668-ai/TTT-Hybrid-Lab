"""Synthetic engineering controls only: no S26/S27/native science first-use."""
import json
from pathlib import Path

import pytest
import torch

from scripts.native_p1_conventional_controls import (
    BaselineAdmissionError, ConventionalAnalogGRU, ConventionalAnalogSSM,
    qualify_conventional_record)
from scripts.native_p1_seed_admission import (
    ROLES, inspect_candidate_seed_plan)


def inputs():
    hist = torch.zeros(2, 3, 4, dtype=torch.float32)
    hist[:, :, -1] = .2
    sensor = torch.tensor([[.1, -.2], [.5, .3]], dtype=torch.float32)
    action = torch.tensor([[.8], [-.8]], dtype=torch.float32)
    dt = torch.tensor([[.2], [.6]], dtype=torch.float32)
    return hist, sensor, action, dt


@pytest.mark.parametrize("model", (ConventionalAnalogGRU, ConventionalAnalogSSM))
def test_continuous_predictor_and_trainable_gradients(model):
    torch.manual_seed(21)
    net = model(2, 1, width=8)
    h,s,a,dt = inputs()
    y = net(h,s,a,dt)
    assert y.shape == (2, 2)
    assert torch.isfinite(y).all()
    y.square().mean().backward()
    assert any(p.grad is not None and bool(torch.isfinite(p.grad).all())
               for p in net.parameters())
    with torch.no_grad():
        assert not torch.allclose(net(h,s,a,dt), net(h,s,-a,dt))


@pytest.mark.parametrize("model", (ConventionalAnalogGRU, ConventionalAnalogSSM))
def test_noncontinuous_and_future_time_rejected(model):
    net = model(2, 1, width=8, max_history=3)
    h,s,a,dt=inputs()
    with pytest.raises(BaselineAdmissionError, match="NONCONTINUOUS"):
        net(h.long(),s,a,dt)
    with pytest.raises(BaselineAdmissionError, match="NONPOSITIVE"):
        net(h,s,a,dt * 0)
    with pytest.raises(BaselineAdmissionError, match="UNBOUNDED"):
        net(torch.cat((h,h[:, :1]),dim=1),s,a,dt)
    future=h.clone();future[0,0,0]=float('nan')
    with pytest.raises(BaselineAdmissionError, match="NONCONTINUOUS_OR_NONFINITE"):
        net(future,s,a,dt)


def valid_receipt():
    return {
        "update_count": 960,
        "validation_mse_last_five": [.002008,.002004,.002003,.002002,.002001],
        "static_validation_mse": .01,
        "effect_sign_accuracy": .98,
        "effect_eligible_coverage": .97,
        "all_controls_same_information": True,
        "checkpoint_cold_restore": True,
        "source_and_seed_custody_verified": True,
        "training_completed": True,
        "compute_budget_verified": True,
    }


def test_numeric_receipt_gate_is_not_training_or_provenance():
    r=qualify_conventional_record(valid_receipt(),arm="conventional_analog_gru")
    assert r["status"]=="CONTROL_RECEIPT_NUMERICALLY_ADMISSIBLE"
    assert r["qualified"] is True


@pytest.mark.parametrize("field,value,code", (
    ("validation_mse_last_five",[.01,.02,.01,.02,.01],"NO_PLATEAU"),
    ("static_validation_mse",.001,"NOT_COMPETENT"),
    ("effect_sign_accuracy",.94,"NOT_COMPETENT"),
    ("effect_eligible_coverage",.92,"NOT_COMPETENT"),
    ("source_and_seed_custody_verified",False,"CUSTODY_OR_TRAINING"),
    ("training_completed",False,"CUSTODY_OR_TRAINING"),
    ("update_count",12,"INVALID_BUDGET"),
    ("validation_mse_last_five",[.002,float('nan'),.002,.002,.002],"INVALID_WINDOW"),
))
def test_receipts_fail_closed(field,value,code):
    r=valid_receipt();r[field]=value
    assert not qualify_conventional_record(r,arm="conventional_analog_gru")["qualified"]
    assert code in qualify_conventional_record(r,arm="conventional_analog_gru")["status"]


def test_missing_receipt_and_unregistered_arm():
    assert "MISSING_EVIDENCE" in qualify_conventional_record({},arm="conventional_analog_ssm")["status"]
    with pytest.raises(BaselineAdmissionError,match="UNREGISTERED"):
        qualify_conventional_record(valid_receipt(),arm="token_lm")


def candidate():
    return {role:[105001+11*i] for i,role in enumerate(ROLES)}


def test_future_science_roles_start_unregistered():
    root=Path(__file__).resolve().parents[1]
    manifest=json.loads((root/"experiments/native_p1_control_admission_preflight.json").read_text())
    assert manifest["schema"]=="ttt-hybrid.native-p1.control-admission-preflight.v1"
    assert manifest["science"]["authorized"] is False
    assert manifest["science"]["seeds"] == {}
    assert manifest["science"]["data_consumed"] is False
    assert manifest["source"]["parent_p0_sha"]=="35501945e619298ce5a093f0b23f20a95f55c380"
    assert not inspect_candidate_seed_plan({})["admitted"]


def test_protected_predecessor_and_s27_collisions():
    for blocked,reason in ((7207,"S26_CONSUMED"),(7403,"S26_SEALED"),(92017,"S27_RESERVED")):
        roles=candidate();roles["fit"]=[blocked]
        x=inspect_candidate_seed_plan(roles)
        assert not x["admitted"] and reason in x["reason"]


def test_overlap_invalid_and_unassigned_fails_closed():
    x=candidate();x["validation"]=x["fit"][:]
    assert "OVERLAPPING" in inspect_candidate_seed_plan(x)["reason"]
    x=candidate();x["fit"]=[]
    assert "UNASSIGNED" in inspect_candidate_seed_plan(x)["reason"]
    x=candidate();x["fit"]=[True]
    assert "INVALID_SEED" in inspect_candidate_seed_plan(x)["reason"]
    x=candidate();x.pop("bootstrap")
    assert "MISSING_OR_EXTRA" in inspect_candidate_seed_plan(x)["reason"]


def test_owner_source_lock_and_scoped_first_use_are_independent():
    x=candidate()
    assert "OWNERSHIP_UNVERIFIED" in inspect_candidate_seed_plan(x)["reason"]
    assert "SOURCE_LOCK_ABSENT" in inspect_candidate_seed_plan(
        x,exclusive_owner_verified=True)["reason"]
    assert "NOT_AUTHORIZED" in inspect_candidate_seed_plan(
        x,exclusive_owner_verified=True,exact_science_source_locked=True)["reason"]
    result=inspect_candidate_seed_plan(x,exclusive_owner_verified=True,
                                      exact_science_source_locked=True,
                                      first_use_authorized=True)
    assert result["reason"]=="SOURCE_ADMISSION_INPUTS_ASSERTED_ONLY"
    assert "separately" in result["warning"]
