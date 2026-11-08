"""P16: champion-protected bounded learned action substitution factual controls."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch

from scripts.native_p16_champion_guard_dev import (
    LearnedGuard,batch_champion_two_step,train_gate,evaluate,
    misalign_rewards,guard_dev_only,GATE_HORIZON,GATE_WORLDS,
    MAX_EVAL_CHALLENGER_DEVIATIONS,P14_SEEDS,P11_SEEDS,
    GATE_UPDATES,ARMS)
from scripts.native_p11_temporal_credit_dev import (
    ConventionalCausalGRU,plan)
from scripts.native_p14_reward_trained_intervention_dev import (
    CausalFeaturePolicy)


def fixtures():
    torch.manual_seed(67)
    return {"champion":ConventionalCausalGRU().eval(),
            "challenger":CausalFeaturePolicy().eval(),
            "learned_guard":LearnedGuard().eval(),
            "miscredit_guard":LearnedGuard().eval()}


def test_guard_only_completed_prior_factual_history_and_logits():
    torch.manual_seed(353)
    g=LearnedGuard()
    h=torch.randn(3,8,2)
    ca=torch.tensor([1,-1,1])
    actor=torch.randn(3)
    assert g(h,ca,actor).shape==(3,)
    with pytest.raises(ValueError,match="P16_GUARD_OBSERVATION_SHAPE"):
        g(h[:,:7,:],ca,actor)


def test_batch_champion_matches_original_P11_2step_planner():
    torch.manual_seed(353)
    model=ConventionalCausalGRU().eval()
    x=torch.randn(4,8,2)
    x[:,:,0]=torch.tensor([[-1.,1.,-1.,1.,-1.,1.,-1.,1.]]).repeat(4,1)
    b=batch_champion_two_step(model,x)
    for i in range(4):
        original,_=plan(model,[(int(a),float(y))
                               for a,y in x[i].tolist()],lookahead=True)
        assert int(b[i])==original


def test_wrong_reward_credit_is_misaligned_not_harmless_column_permutation():
    a=torch.arange(GATE_HORIZON*GATE_WORLDS,dtype=torch.float32).reshape(
        GATE_HORIZON,GATE_WORLDS)
    wrong=misalign_rewards(a)
    assert not torch.equal(a,wrong)
    assert torch.equal(a.flatten().sort().values,
                       wrong.flatten().sort().values)


def test_genuine_policy_gradient_training_weight_updates_same_worlds():
    torch.set_num_threads(1)
    f=fixtures()
    aligned,record=train_gate(f["champion"],f["challenger"],
                               P14_SEEDS[0],updates=2)
    broken,neg=train_gate(f["champion"],f["challenger"],P14_SEEDS[0],
                          wrong_credit=True,updates=2)
    assert record["factual_gradient_changed_weights"]
    assert neg["factual_gradient_changed_weights"]
    assert record["same_world_cohort_sha256"]==neg["same_world_cohort_sha256"]
    assert record["actual_gradient_updates"]==neg["actual_gradient_updates"]==2
    assert record["actual_chosen_training_actions"]==2*GATE_WORLDS*GATE_HORIZON


def test_factual_action_cap_and_sham_same_observed_result():
    torch.set_num_threads(1)
    f=fixtures()
    for fam,seed in (("instant",88031),("delay2",88421),
                     ("changing_delay",88421),("sham_no_action",88421)):
        e=evaluate(fam,seed,0,f)
        assert e["integrity"]=="PASS",e["checks"]
        assert all(e["checks"].values())
        assert set(e["arms"])==set(ARMS)
        for arm in ("learned_guard","miscredit_guard","random_guard"):
            assert e["arms"][arm]["actual_nonprobe_challenger_deviations"]<=MAX_EVAL_CHALLENGER_DEVIATIONS
        if fam=="sham_no_action":
            assert e["checks"]["sham_no_action_identical_all_factual_returns"]


def test_cli_frozen_protocol_denies_original_science(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    m=json.loads((root/"experiments/native_p16_champion_guard_frozen.json").read_text())
    assert m["schema"]=="ttt-hybrid.native-p16.champion-guard-frozen.v1"
    assert m["source"]["P15_parent_sha"]=="205291c7e1fcd7a3df0343ba7dbcd17a2fa225e0"
    assert m["science"]["authorized"] is False
    assert m["science"]["world_seeds"]=={}
    assert m["science"]["first_use_consumed"] is False
    assert m["design"]["fixed_forced_probes"]==16
    assert m["design"]["max_evaluation_challenger_deviations"]==12
    assert m["design"]["gate_gradient_updates"]==GATE_UPDATES==32
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p16_champion_guard_dev.py"),
                      "--help"],cwd=root,text=True,capture_output=True,timeout=20)
    assert p.returncode==0,p.stderr[-900:]
    monkeypatch.setenv("NATIVE_P16_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P16_SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        guard_dev_only()
