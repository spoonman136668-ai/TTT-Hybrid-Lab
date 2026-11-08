"""P20 PPO real environment factual learning and sealed scientific source controls."""
import json
from pathlib import Path
import subprocess
import sys
import torch
import pytest
from scripts.native_p20_reward_only_ppo_dev import (
    model_for_arm,actor_tensor,corrupt_action_credit,
    set_public_environment_dynamics,require_development,SEEDS,
    EVAL_SEEDS,EVAL_CONDITIONS,TRAIN_ARMS,EVAL_ARMS,
    NUM_ENVS,ROLLOUTS,STEPS_PER_ROLLOUT,PPO_EPOCHS,MINIBATCH
)
from scripts.native_p18_real_cartpole_factual_dev import (
    HistoryGRU,FullHistoryMLP,factual_row)


def test_original_two_P18_tokenfree_competing_architectures_unchanged():
    m=model_for_arm("reward_gru",SEEDS[0])
    f=model_for_arm("reward_mlp",SEEDS[0])
    assert type(m) is HistoryGRU and type(f) is FullHistoryMLP
    assert sum(p.numel() for p in m.parameters())==489
    assert sum(p.numel() for p in f.parameters())==471
    x=actor_tensor([factual_row([0.,.04,.03,.07],0,0.)])
    assert x.shape==(1,8,6)
    assert m(x)[0].shape==(1,2)
    assert f(x)[0].shape==(1,2)
    assert torch.isfinite(x).all()


def test_real_bad_credit_negative_is_nonidentity_with_same_histogram():
    reward=torch.arange(STEPS_PER_ROLLOUT*NUM_ENVS,dtype=torch.float32).reshape(
        STEPS_PER_ROLLOUT,NUM_ENVS)
    bad=corrupt_action_credit(reward)
    assert not torch.equal(bad,reward)
    assert torch.equal(bad.flatten().sort().values,
                       reward.flatten().sort().values)
    with pytest.raises(ValueError,match="P20_CREDIT_CONTROL_SHAPE"):
        corrupt_action_credit(reward[:-1])


def test_source_pole_change_does_not_change_frozen_source_or_actor_inputs():
    class ActualPinnedUpstreamObject:
        length=.5
        masspole=.1
        polemass_length=.05
    class Container:
        unwrapped=ActualPinnedUpstreamObject()
    e=Container()
    assert set_public_environment_dynamics(e,"short_pole")["model_observes_ratio"] is False
    assert e.unwrapped.length==pytest.approx(.35)
    assert e.unwrapped.polemass_length==pytest.approx(.035)
    e.unwrapped.length=.5
    assert set_public_environment_dynamics(e,"long_pole")["pole_length_ratio"]==1.3
    with pytest.raises(ValueError,match="P20_UNREGISTERED_TRANSFER_STRUCTURE"):
        set_public_environment_dynamics(e,"future_science")


def test_frozen_protocol_auth_and_strict_no_science(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    p=json.loads((root/"experiments/native_p20_reward_only_ppo_frozen.json").read_text())
    assert p["schema"]=="ttt-hybrid.native-p20.ppo-reward-only-frozen.v1"
    assert p["source"]["P19_parent_sha"]=="631647d8d03d150a6c537be6b8b0e3361c27e5a0"
    assert p["source"]["S29_owner_sha"]=="30c27ca4865e16b7574ee2b562879108db2b3796"
    assert p["science"]["authorized"] is False
    assert p["science"]["first_use_consumed"] is False
    assert p["science"]["protected_science_seeds"]=={}
    assert p["design"]["ppo_rollouts"]==ROLLOUTS
    assert p["design"]["parallel_environments"]==NUM_ENVS
    assert p["design"]["train_arm_types"]==list(TRAIN_ARMS)
    assert p["design"]["dev_eval_conditions"]==list(EVAL_CONDITIONS)
    assert p["design"]["dev_eval_seeds"]==list(EVAL_SEEDS)
    assert p["design"]["eval_arms"]==list(EVAL_ARMS)
    assert p["design"]["ppo_epochs"]==PPO_EPOCHS
    assert p["design"]["ppo_minibatch"]==MINIBATCH
    monkeypatch.setenv("NATIVE_P20_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P20_OFFICIAL_SCIENCE_FIRST_USE_DENIED"):
        require_development()


def test_one_use_source_script_cli_does_not_install_packages():
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
        str(root/"scripts/native_p20_reward_only_ppo_dev.py"),"--help"],
        cwd=root,text=True,capture_output=True,timeout=20)
    assert p.returncode==0,p.stderr[-2000:]
    s=(root/"scripts/native_p20_reward_only_ppo_dev.py").read_text()
    assert "pip install" not in s
    assert "admit_gymnasium" in s
    assert "no_expert_action_labels_used" in s
    assert "original_scientific_first_use" in s
