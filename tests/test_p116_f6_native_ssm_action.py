"""P116 native F4 SSM action readout: original parity and causality negatives."""
from __future__ import annotations

import copy
import math
import os

import pytest
import torch

from experiments import p114_f6_native_optimizer_smoke as p114
from experiments import p116_f6_native_ssm_action as p116
from experiments import p113_f6_normalized_ssm_preflight as p113
from p75_prequential_baselines import before_write_predictions

torch.set_num_threads(1)


def _case():
    events=p114.fictional_events(0,1)
    return events[:2],events[2]["state_before"]


def test_p116_original_native_readout_matches_p75_before_write_at_next_step():
    prefix,now=_case()
    model=p114.new_baseline("ssm256",p114.SEEDS[0]).eval()
    got=p116.infer_two_actions(model,now,prefix)
    assert got["policy"]=="ssm256_model_based"
    assert got["action"] in (0,1)
    assert got["before_write_factual_prefix_steps"]==2
    assert got["unselected_source_effects_observed"]==0
    assert not got["F6_first_use_authorized"] and not got["scientific_pass"]
    assert len(got["predicted_normalized_effects_fixture"])==2
    for action in (0,1):
        # The evaluator fills a deliberately fictional current effect; the
        # model must read only the prefix when forecasting this effect.
        item=p113.fixture_event(3,action,now,[x+.001 for x in now],
                                1.0,terminated=True)
        full=p114._facts([prefix+[item]])
        native,_=before_write_predictions(model,full)
        expected=native[0,2].detach().tolist()
        assert got["predicted_normalized_effects_fixture"][action]==pytest.approx(
            expected,abs=1e-6,rel=1e-6)
        changed=copy.deepcopy(full)
        changed["effect"][0,2,0]+=0.7
        changed["effect"][0,2,4]-=0.4
        altered,_=before_write_predictions(model,changed)
        assert torch.equal(native[:,:3],altered[:,:3])
    assert p116.infer_two_actions(model,now,prefix)==got


def test_p116_two_action_decision_uses_native_readout_not_fixed_heuristic():
    model=p114.new_baseline("ssm256",p114.SEEDS[0]).eval()
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
        model.head.weight[2,256+5]=10.0
    a=p116.infer_two_actions(model,[0.,0.,0.,0.])
    assert a["action"]==1
    assert a["estimated_physical_costs_fixture"][0]>a["estimated_physical_costs_fixture"][1]
    with torch.no_grad():
        model.head.weight[2,256+5]=0.
        model.head.weight[2,256+6]=10.0
    b=p116.infer_two_actions(model,[0.,0.,0.,0.])
    assert b["action"]==0
    assert b["estimated_physical_costs_fixture"][1]>b["estimated_physical_costs_fixture"][0]
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.zero_()
    tie=p116.infer_two_actions(model,[0.,0.,0.,0.])
    assert tie["action"]==0
    # Prediction-padding channels do not contribute to the physical objective.
    with torch.no_grad():
        model.head.bias[7:]=1000.
    padded=p116.infer_two_actions(model,[0.,0.,0.,0.])
    assert padded["action"]==tie["action"]
    assert padded["estimated_physical_costs_fixture"]==tie["estimated_physical_costs_fixture"]


def test_p116_toy_trained_native_checkpoint_can_be_read_without_oracle():
    train,_=p114.fictional_corpus()
    facts=p114._facts([events for _,_,events in train])
    model=p114.new_baseline("ssm256",p114.SEEDS[0]).cpu().eval()
    prefix,now=_case()
    before=p116.infer_two_actions(model,now,prefix)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    for _ in range(8):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=p114.factual_objective(model,facts)["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    model.eval()
    after=p116.infer_two_actions(model,now,prefix)
    assert after["model_checkpoint_sha256_fixture_only"]!=before["model_checkpoint_sha256_fixture_only"]
    assert any(abs(x-y)>1e-7 for pa,pb in zip(
        before["predicted_normalized_effects_fixture"],
        after["predicted_normalized_effects_fixture"]) for x,y in zip(pa,pb))
    assert after["original_source_effects_executed"]==0
    assert after["F6_real_trained_model_qualified"] is False


def test_p116_strict_factual_history_shape_custody_and_termination():
    prefix,now=_case()
    model=p114.new_baseline("ssm256",p114.SEEDS[0]).eval()
    cases=[
        ("unknown_effect", "P116_UNCHOSEN_EFFECT_OR_UNKNOWN_STEP_FIELD"),
        ("wrong_receipt", "P116_UNSEALED_FIXTURE_OUTCOME"),
        ("not_do", "P116_NOT_A_CONFIRMED_BINARY_ACTION"),
        ("wrong_action", "P116_NOT_A_CONFIRMED_BINARY_ACTION"),
        ("wrong_id", "P116_NOT_A_CONFIRMED_BINARY_ACTION"),
        ("broken_chain", "P116_BROKEN_FACTUAL_PREFIX"),
        ("future_terminated", "P116_AFTER_TERMINAL_NO_NEXT_ACTION"),
        ("bad_reward", "P113_BAD_REALIZED_REWARD"),
    ]
    for change,why in cases:
        events=copy.deepcopy(prefix)
        if change=="unknown_effect":events[0]["unchosen_effect"]=1
        if change=="wrong_receipt":events[0]["receipt_ref"]="fixture:bad"
        if change=="not_do":events[0]["mode"]="observe"
        if change=="wrong_action":events[0]["executed_action"]=2
        if change=="wrong_id":events[0]["event_id"]=7
        if change=="broken_chain":events[1]["state_before"][0]+=1.
        if change=="future_terminated":events[1]["terminated"]=True
        if change=="bad_reward":events[0]["reward"]=float("nan")
        with pytest.raises((ValueError,PermissionError),match=why):
            p116.infer_two_actions(model,now,events)
    with pytest.raises(ValueError,match="P116_NEXT_STATE_DOES_NOT_MATCH_FACTUAL_LAST"):
        p116.infer_two_actions(model,[0.,0.,0.,0.],prefix)
    with pytest.raises(ValueError,match="P113_BAD_PHYSICAL_STATE"):
        p116.infer_two_actions(model,[0.,0.,float("nan"),0.])
    with pytest.raises(ValueError,match="P116_INVALID_FACTUAL_HISTORY_LENGTH"):
        p116.infer_two_actions(model,now,prefix*48)


def test_p116_native_model_state_validation_and_firstuse_denial():
    model=p114.new_baseline("ssm256",p114.SEEDS[0])
    with pytest.raises(ValueError,match="P116_ORIGINAL_F4_SSM_EVAL_REQUIRED"):
        p116.infer_two_actions(model,[0.,0.,0.,0.])
    model.eval()
    with torch.no_grad():
        model.head.bias[0]=float("nan")
    with pytest.raises(ValueError,match="P116_CPU_FLOAT32_FINITE_WEIGHTS_REQUIRED"):
        p116.infer_two_actions(model,[0.,0.,0.,0.])
    with pytest.raises(PermissionError,match="P116_REAL_F6_FIRSTUSE_NOT_AUTHORIZED"):
        p116.refuse_external_source(seed=1820000000)
    from pathlib import Path
    body=Path(p116.__file__).read_text("utf-8")
    for forbidden in ("import causal_gym","from causal_gym","env.do(",
                      "env.reset(","dispatch_workflow(","torch.load(",
                      "requests.get(","subprocess.run(","owner_execute("):
        assert forbidden not in body
    f=p116.locked()
    assert f["classification"]["source_firstuse_authorized"] is False
    assert f["classification"]["protocol_fork_selected"] is False
