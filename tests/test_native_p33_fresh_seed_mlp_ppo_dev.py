"""P33 frozen original-source reward-only replication and cold-reload negatives."""
import hashlib
import json
import numpy as np
import pytest
import torch
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p30_reward_motion_ppo_dev as p30
from scripts import native_p32_component_damage_dev as p32
from scripts import native_p33_fresh_seed_mlp_ppo_dev as p33

class FakeCartPole:
    def __init__(self):
        self.unwrapped=self
        self.gravity=9.8
        self.length=.5
        self.masspole=.1
        self.polemass_length=.05
        self.n=0
    def reset(self,seed=None):
        self.n=0
        self.x=float(seed%19)*.0001
        self.vel=.003
        self.theta=.009
        self.omega=.003
        return self.obs(),{}
    def obs(self):
        return np.array((self.x,self.vel,self.theta,self.omega),
                        dtype=np.float32)
    def step(self,action):
        assert action in (0,1)
        impulse=2*action-1
        self.vel+=.002*impulse
        self.omega+=.007*impulse/self.length
        self.x+=.02*self.vel
        self.theta+=.02*self.omega
        self.n+=1
        return self.obs(),1.,self.n>=18,False,{}
    def close(self): pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="CartPole-v1" and render_mode is None
        return FakeCartPole()

def factual_history():
    return [p20.factual_row((.1+.02*i,999+i,.01+.001*i*i,
                             -888-i),0 if i==0 else 1,0 if i==0 else 1.)
            for i in range(8)]

def test_preimplementation_frozen_seeds_thresholds_and_science_denied(monkeypatch):
    p=p33.frozen()
    assert p["parent"]["sha"]=="7bc49166b83788efc7bfb9da82d39fc420630c46"
    assert p["phase"]=="PREIMPLEMENTATION_FROZEN_ORIGINAL_PUBLIC_SOURCE_FIRST_USE_UNCONSUMED"
    assert p["source"]["no_protected_science_first_use"]
    assert p["training"]["new_model_seeds"]==list(p33.SEEDS)
    assert set(p33.SEEDS).isdisjoint(p32.SEEDS)
    assert p["training"]["new_public_train_source_reset_base"]==164000
    assert p["training"]["MLP_total_parameters"]==591
    assert p["training"]["GRU_total_parameters"]==597
    assert p["frozen_thresholds"]["three_fresh_seeds_must_pass_all"]
    assert p["frozen_thresholds"]["no_posthoc_threshold_or_seed_changes"]
    monkeypatch.setenv("NATIVE_P33_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P33_PROTECTED_FIRST_USE_DENIED"):
        p33.require_dev()

def test_mlp_is_591_parameter_nonrecurrent_full_eight_history_neural_baseline():
    torch.set_num_threads(1)
    m=p33.FullyTrainedHistoryMLP()
    assert sum(x.numel() for x in m.parameters())==591
    assert sum(x.numel() for x in p30.RewardGRU10().parameters())==597
    assert not any(isinstance(x,torch.nn.GRU) for x in m.modules())
    x=torch.zeros(3,8,10)
    logits,value=m(x)
    assert logits.shape==(3,2)
    assert value.shape==(3,)
    x[0,0,0]=1.
    x[1,7,0]=1.
    with torch.no_grad():
        before,_=m(torch.zeros_like(x))
        after,_=m(x)
    assert not torch.equal(before[0],after[0])
    assert not torch.equal(before[1],after[1])
    with pytest.raises(ValueError,match="FULL_SOURCE_SHAPE"):
        m(torch.zeros(1,8,6))

def test_frozen_velocity_mask_original_motion_view_and_source_scopes():
    torch.set_num_threads(1)
    old=(p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,
         p32.PROBE_SEEDS,p30.SEEDS,p30.RewardGRU10)
    with p33.scoped_original_p32():
        assert p32.SEEDS==p33.SEEDS
        assert p32.TRAIN_BASE==p33.TRAIN_BASE
        assert p32.EVAL_SEEDS==p33.EVAL_SEEDS
        tensor=p32.motion(factual_history(),p33.SEEDS[0])
        assert tensor.shape==(1,8,10)
        assert torch.count_nonzero(tensor[:,:,[1,3]])==0
        assert torch.count_nonzero(tensor[:,:,6:])>0
        with p33.scoped_conventional_mlp():
            assert p30.RewardGRU10 is p33.FullyTrainedHistoryMLP
            assert isinstance(p32.make_initial(p33.SEEDS[0]),
                              p33.FullyTrainedHistoryMLP)
        assert p30.RewardGRU10 is old[5]
    assert (p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,
            p32.PROBE_SEEDS,p30.SEEDS,p30.RewardGRU10)==old
    with pytest.raises(RuntimeError,match="TEST_ROLLBACK"):
        with p33.scoped_original_p32():
            with p33.scoped_conventional_mlp():
                raise RuntimeError("TEST_ROLLBACK")
    assert (p32.SEEDS,p32.TRAIN_BASE,p32.EVAL_SEEDS,
            p32.PROBE_SEEDS,p30.SEEDS,p30.RewardGRU10)==old

def test_original_factual_mlp_PPO_and_GRU_actor_freeze_same_budget(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p20,"ROLLOUTS",1)
    monkeypatch.setattr(p20,"NUM_ENVS",2)
    monkeypatch.setattr(p20,"STEPS_PER_ROLLOUT",8)
    monkeypatch.setattr(p20,"PPO_EPOCHS",2)
    monkeypatch.setattr(p20,"MINIBATCH",8)
    gym=FakeGym()
    receipts={}
    with p33.scoped_original_p32():
        for arm in ("full_actor_recurrent_ppo","actor_head_only_ppo",
                    "conventional_mlp_full_ppo","critic_only_frozen_policy"):
            if arm=="conventional_mlp_full_ppo":
                with p33.scoped_conventional_mlp():
                    model,initial,rec=p32.train_arm(
                        gym,p33.SEEDS[0],"full_actor_recurrent_ppo")
            else:
                model,initial,rec=p32.train_arm(gym,p33.SEEDS[0],arm)
            assert rec["real_original_train_source_actions"]==16
            assert rec["original_ppo_minibatch_updates"]==4
            assert rec["cold_exact_logits_and_values"]
            assert rec["learning_executed_real_actions_without_expert"]
            assert len(rec["original_actual_action_consequence_sha256"])==64
            receipts[arm]=rec
        sha={rec["first_512_real_source_actions_trace_sha256"]
             for arm,rec in receipts.items()
             if arm!="conventional_mlp_full_ppo"}
        assert len(sha)==1
        assert receipts["conventional_mlp_full_ppo"]["all_model_parameters"]==591
        assert receipts["full_actor_recurrent_ppo"]["all_model_parameters"]==597
        assert receipts["actor_head_only_ppo"][
            "components_bit_exact_unchanged_from_original"]["recurrent"]
        assert receipts["critic_only_frozen_policy"][
            "components_bit_exact_unchanged_from_original"]["actor"]
        assert not receipts["conventional_mlp_full_ppo"][
            "components_bit_exact_unchanged_from_original"]["recurrent"]
        assert not receipts["conventional_mlp_full_ppo"][
            "components_bit_exact_unchanged_from_original"]["actor"]

def test_original_world_source_real_reward_means_and_frozen_comparison():
    vals={
      condition:{
        "actor_head_only_ppo":180.,
        "full_actor_recurrent_ppo":130.,
        "conventional_mlp_full_ppo":190.,
        "critic_only_frozen_policy":20.,
        "finite_difference_public_two_frame":200.,
        "initial_no_update":20.,
        "seeded_random":22.}
      for condition in p33.CONDITIONS}
    gates=p33.frozen_arm_gates(vals)
    assert all(all(v.values()) for v in gates.values())
    vals["original"]["actor_head_only_ppo"]=130.
    assert not p33.frozen_arm_gates(vals)["original"]["reward_min"]
    vals["original"]["actor_head_only_ppo"]=170.
    vals["original"]["conventional_mlp_full_ppo"]=200.
    assert not p33.frozen_arm_gates(vals)["original"][
        "actor_within_15_of_fully_trained_MLP"]
    with p33.scoped_original_p32():
        result=p33.original_eval(FakeGym(),None,
             "finite_difference_public_two_frame",p33.EVAL_SEEDS[0],
             "original",p33.SEEDS[0])
    assert result["actual_source_reward_return"]>0
    assert not result["true_velocity_seen_by_model"]
    assert len(result["source_real_action_consequence_sha256"])==64

def test_cold_replay_one_frozen_real_source_world_and_corrupt_checkpoint(
        tmp_path,monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(p33,"SEEDS",(4903,))
    monkeypatch.setattr(p33,"TRAIN_ARMS",("conventional_mlp_full_ppo",))
    monkeypatch.setattr(p33,"EVAL_SEEDS",(165011,))
    monkeypatch.setattr(p33,"CONDITIONS",("original",))
    directory=tmp_path/"models"
    directory.mkdir()
    with p33.scoped_original_p32():
        model=p33.model_for_eval(4903,"conventional_mlp_full_ppo")
        world=p33.original_eval(FakeGym(),model,
                  "conventional_mlp_full_ppo",165011,"original",4903)
        state=directory/"seed_4903_conventional_mlp_full_ppo.pt"
        torch.save(model.state_dict(),state)
        weights=p20.model_sha(model)
    record={
      "schema":p33.SCHEMA,
      "engineering":"PASS_ORIGINAL_PUBLIC_DEV_ONLY",
      "original_factual_train_actions":p33.EXPECTED_TRAIN_ACTIONS,
      "original_PPO_optimizer_minibatches":p33.EXPECTED_UPDATES,
      "records":[{
        "new_model_seed":4903,
        "checkpoint_manifest":{"conventional_mlp_full_ppo":{
          "relative_name":state.name,
          "checkpoint_file_sha256":hashlib.sha256(state.read_bytes()).hexdigest(),
          "state_weight_sha256":weights}},
        "actual_source_heldout_episodes":[{
          "source_reset_seed":165011,
          "source_condition":"original",
          "factual_original_source_episodes":{
              "conventional_mlp_full_ppo":world}}]
      }]}
    receipt=tmp_path/"receipt.json"
    receipt.write_text(json.dumps(record),encoding="utf-8")
    okay=p33.cold_verify(FakeGym(),receipt,directory)
    assert okay["status"]=="PASS_BIT_EXACT_ORIGINAL_SOURCE_TRACE_AND_REWARD_PARITY"
    assert okay["independent_fresh_process_source_episodes"]==1
    state.write_bytes(b"tampered checkpoint")
    with pytest.raises(RuntimeError,match="P33_COLD_CHECKPOINT_FILE_SHA_DRIFT"):
        p33.cold_verify(FakeGym(),receipt,directory)
