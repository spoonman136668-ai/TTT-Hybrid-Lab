"""P78-MR R3 source-free regression: fabricated traces only; no F3 access."""
import hashlib
import json
from pathlib import Path

import pytest
import torch

from experiments.p78mr_r3_f3_transfer import (
    FREEZE, FREEZE_BLOB, locked_protocol, factual_event, score_world,
    read_prior_cal, read_f4_controls,
)
from experiments.p78mr_gain_posthoc_audit import read_locked
from experiments.p78mr_adaptive_controls import new_learned
from experiments.p77_write_policy import WritePolicy


def fake_step(i, action=1, reward=1., effect_offset=0.):
    a=[0.1,0.2,0.03,0.04]
    return dict(event_id=i,executed_action=action,mode="do",
                receipt_ref=f"fixture:{i}",reward=reward,
                state_before=a,state_after=[x+effect_offset for x in a],
                terminated=False,truncated=False)


def test_freeze_is_original_immutable_pre_source_and_no_science():
    d=locked_protocol()
    assert read_locked(FREEZE,FREEZE_BLOB)
    assert d["source"]["reuse_only"]
    assert not d["authority"]["new_effects"]
    assert not d["authority"]["training"]
    assert not d["authority"]["OOD"]
    assert not d["report"]["scientific"].startswith("SCIENTIFIC_PASS")


def test_p80_physical_encoding_factual_key_and_zero_padding():
    a,y=factual_event(fake_step(1,effect_offset=0.07))
    assert a.shape == y.shape == (16,)
    assert a.norm().item() == pytest.approx(1.,abs=1e-6)
    assert a[0]>0 and a[5]==0 and a[6]>0
    assert (a[7:]==0).all()
    assert (y[7:]==0).all()
    assert y[4]==pytest.approx(0.761594176,abs=1e-6)


def test_read_before_write_denies_current_observation_leakage():
    step=fake_step(1,effect_offset=0.07)
    future=fake_step(2,effect_offset=90.)
    old=dict(event_id=2,executed_action=1,mode="do",
             receipt_ref="fixture:2",reward=1.,
             state_before=step["state_after"],
             state_after=[v+0.07 for v in step["state_after"]],
             terminated=False,truncated=False)
    w={"transitions":[step,old]}
    changed={"transitions":[step,dict(old,state_after=[v+90 for v in old["state_before"]])]}
    # The current effect changes MSE but not previous memory or its query.
    model=WritePolicy("fixed_eta",eta=.4)
    t=torch.tensor([factual_event(step)[0].tolist()])
    state=model.encode({"action":factual_event(step)[0].reshape(1,1,16),
                        "effect":factual_event(step)[1].reshape(1,1,16)})
    pred=model.read(state,t)
    assert torch.equal(pred,model.read(state,t))
    assert score_world(model,w) != score_world(model,changed)


def test_unqualified_archives_fail_closed(tmp_path):
    fake=tmp_path/"spoof.zip"
    fake.write_bytes(b"wrong source")
    with pytest.raises(ValueError,match="PRIOR_ARCHIVE_DRIFT"):
        read_prior_cal(fake)
    with pytest.raises(ValueError,match="PRIOR_ARCHIVE_DRIFT"):
        read_f4_controls(fake)


def test_evaluator_statically_has_no_model_fit_or_simulator_actions():
    code=(Path(__file__).resolve().parents[1]/"experiments"/
          "p78mr_r3_f3_transfer.py").read_text()
    for forbidden in ("make_world(", "make_batch(", "optimizer.step(",
                      "torch.save(", "env.do(", "env.reset(", "requests.get(",
                      "urlopen(", "subprocess.run("):
        assert forbidden not in code


def test_original_learned_model_is_not_modified_by_scoring():
    model=new_learned("non_directional3",978201)
    before=model.theta.detach().clone()
    world={"transitions":[fake_step(1,effect_offset=.02),
                          fake_step(2,effect_offset=.03)]}
    score_world(model,world)
    assert torch.equal(before,model.theta.detach())
