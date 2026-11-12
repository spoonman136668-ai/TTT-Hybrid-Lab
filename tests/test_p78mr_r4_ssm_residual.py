"""P78-R4: fictional in-memory causal sequences only; no F3/F4 artifact access."""
import copy
from pathlib import Path

import pytest
import torch

from experiments.p69_trainable_competitors import DiagonalContinuousSSM
from experiments.p75_prequential_baselines import before_write_predictions
from experiments.p77_write_policy import WritePolicy
from experiments.p78mr_r4_ssm_residual import (
    SSM_STATE, FAST_STATE, lock_protocol, start_states, step, score_world,
    frozen_ssm, retrospect,
)


def toy(batch=2, events=6):
    g = torch.Generator().manual_seed(984555)
    a = torch.nn.functional.normalize(
        torch.randn(batch, events, 16, generator=g), dim=-1)
    y = torch.randn(batch, events, 16, generator=g) / 10.0
    return {"action": a, "effect": y}


def test_exact_original_ssm_prequential_without_memory_is_unchanged():
    torch.manual_seed(22)
    base = DiagonalContinuousSSM().eval()
    facts = toy()
    with torch.no_grad():
        original, tail = before_write_predictions(base, facts)
        ssm, fast = start_states(facts["action"][:, 0])
        pred = []
        for t in range(facts["action"].shape[1]):
            p, ssm, fast = step(base, None, ssm, fast,
                                facts["action"][:, t], facts["effect"][:, t])
            pred.append(p)
        new = torch.stack(pred, dim=1)
    assert torch.equal(original, new)
    assert torch.equal(tail, ssm)
    assert torch.count_nonzero(fast) == 0
    assert ssm.shape == (2, 256)
    assert fast.shape == (2, 16, 16)


def test_no_current_effect_in_prediction_and_cold_restore():
    torch.manual_seed(23)
    base = DiagonalContinuousSSM().eval()
    policy = WritePolicy("fixed_eta", eta=0.4).eval()
    facts = toy(batch=1)
    with torch.no_grad():
        state, memory = start_states(facts["action"][:, 0])
        first, state, memory = step(
            base, policy, state, memory, facts["action"][:, 0],
            facts["effect"][:, 0])
        checkpoint = state.clone(), memory.clone()
        second, s1, m1 = step(
            base, policy, state, memory,
            facts["action"][:, 1], facts["effect"][:, 1])
        future = facts["effect"][:, 1].clone() + 15.0
        same_pred, s2, m2 = step(
            base, policy, *checkpoint, facts["action"][:, 1], future)
        assert torch.equal(second, same_pred)
        assert not torch.equal(s1, s2)
        assert not torch.equal(m1, m2)
        assert torch.equal(checkpoint[0], state)
        assert torch.equal(checkpoint[1], memory)
        assert first.shape == second.shape == (1, 16)


def test_full_six_step_cold_restore_exact_256_plus_256_state():
    torch.manual_seed(24)
    base = DiagonalContinuousSSM().eval()
    policy = WritePolicy("fixed_eta", eta=0.4).eval()
    f = toy(batch=1)
    with torch.no_grad():
        s, w = start_states(f["action"][:, 0])
        for i in range(6):
            _, s, w = step(base, policy, s, w,
                           f["action"][:, i], f["effect"][:, i])
        full_s, full_w = s, w
        s, w = start_states(f["action"][:, 0])
        for i in range(3):
            _, s, w = step(base, policy, s, w,
                           f["action"][:, i], f["effect"][:, i])
        s, w = s.clone(), w.clone()
        for i in range(3, 6):
            _, s, w = step(base, policy, s, w,
                           f["action"][:, i], f["effect"][:, i])
    assert torch.equal(s, full_s)
    assert torch.equal(w, full_w)
    assert s.numel() == 256 and w.numel() == 256


def test_wrong_action_credit_changes_both_updates():
    torch.manual_seed(25)
    base = DiagonalContinuousSSM().eval()
    policy = WritePolicy("fixed_eta", eta=0.4).eval()
    a = torch.zeros(1,16)
    a[0,0] = 1.0
    wrong = torch.zeros(1,16)
    wrong[0,1] = 1.0
    observed = torch.randn(1,16)
    s,w = start_states(a)
    with torch.no_grad():
        _,s1,w1 = step(base,policy,s,w,a,observed)
        _,s2,w2 = step(base,policy,s,w,wrong,observed)
    assert not torch.equal(s1,s2) and not torch.equal(w1,w2)


@pytest.mark.parametrize("bad_action,bad_effect", [
    ("nan",None),("zero",None),("scale",None),(None,"nan"),
])
def test_bad_factual_inputs_fail_closed(bad_action,bad_effect):
    base=DiagonalContinuousSSM().eval()
    a=torch.zeros((1,16)); a[0,0]=1.0
    y=torch.zeros_like(a)
    if bad_action=="nan": a[0,0]=float("nan")
    if bad_action=="zero": a[0,0]=0.0
    if bad_action=="scale": a[0,0]=2.0
    if bad_effect=="nan": y[0,0]=float("nan")
    s,w=start_states(torch.eye(16,dtype=torch.float32)[:1])
    with pytest.raises(ValueError,match="P78MR_R4_INVALID_FACT_OR_STATE"):
        step(base,None,s,w,a,y)


def test_frozen_protocol_source_and_owner_denial():
    f=lock_protocol()
    assert f["classification"].startswith("PRE-SCORE-FROZEN")
    assert f["trained_dynamics"]["all_weights_frozen"] is True
    assert f["authority"]["new_external_do"] is False
    assert f["authority"]["model_train"] is False
    assert f["authority"]["OOD"] is False


def test_invalid_or_missing_actual_owner_archives_cannot_be_scored(tmp_path):
    bogus=tmp_path/"untrusted.zip"
    bogus.write_bytes(b"not source")
    with pytest.raises(ValueError):
        frozen_ssm(bogus)
    with pytest.raises(ValueError):
        retrospect(bogus,bogus)


def test_toy_world_scoring_cannot_be_confused_with_external_study():
    torch.manual_seed(21)
    base=DiagonalContinuousSSM().eval()
    raw = []
    before=[0.0,0.0,0.0,0.0]
    for i in range(3):
        after=[v+0.01 for v in before]
        raw.append({"state_before":before,"state_after":after,
                    "executed_action":i%2,"reward":1.0,
                    "terminated":False,"truncated":False})
        before=after
    with torch.no_grad():
        score=score_world(base,None,{"transitions":raw})
    assert score>=0 and score<float("inf")


def test_no_simulator_training_or_runner_api():
    source=(Path(__file__).resolve().parents[1]/
            "experiments"/"p78mr_r4_ssm_residual.py").read_text()
    for forbidden in ("make_world(", "make_batch(", "optimizer.step(",
                      "torch.save(", "env.do(", "env.reset(", "requests.get(",
                      "urlopen(", "subprocess.run("):
        assert forbidden not in source
