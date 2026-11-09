"""P36: action attribution, true source interventions and never-retrain guards."""
import math,types
import numpy as np
import pytest
import torch
from scripts import native_p35_mountaincar_factual_aux_ppo_dev as p35
from scripts import native_p36_cold_action_intervention_dev as p36

class FakeMountainCar:
    def __init__(self):
        self.unwrapped=self
        self.action_space=types.SimpleNamespace(n=3)
        self.force=.001;self.gravity=.0025;self.goal_position=.5
        self.position=-.5;self.velocity=0.;self.n=0
    def reset(self,seed=None):
        self.n=0;self.position=-.5+int(seed%31)*.0003;self.velocity=0.
        return np.asarray((self.position,self.velocity),dtype=np.float32),{}
    def step(self,a):
        assert a in (0,1,2)
        self.velocity+=self.force*(a-1)-self.gravity*math.cos(3*self.position)
        self.velocity=min(.07,max(-.07,self.velocity))
        self.position=min(.6,max(-1.2,self.position+self.velocity))
        if self.position<=-1.2 and self.velocity<0:self.velocity=0.
        self.n+=1;done=self.position>=.5
        return np.asarray((self.position,self.velocity),dtype=np.float32),-1.,done,self.n>=200,{}
    def close(self):pass

class FakeGym:
    def make(self,name,render_mode=None):
        assert name=="MountainCar-v0" and render_mode is None
        return FakeMountainCar()

def test_original_P35_checkpoints_only_frozen_science_and_no_new_model_train(monkeypatch):
    q=p36.frozen()
    assert q["parent"]["sha"]=="6cbd0a9b390e793df3f24a09a9ccc66b5011ca5d"
    assert q["phase"]=="FROZEN_BEFORE_NEW_SOURCE_FIRST_USE"
    assert q["parent"]["archive_sha256"]==p36.ARCHIVE_SHA
    assert q["source"]["trained_model_seeds"]==list(p36.SEEDS)
    assert set(q["probe"]["seeds"]).isdisjoint(q["heldout"]["seeds"])
    assert q["intervention"]["arms"]==list(p36.ARMS)
    assert q["thresholds"]["no_postresult_tuning"]
    assert q["safety"]["model_parameter_updates"]==0
    assert q["safety"]["protected_science_first_use"] is False
    monkeypatch.setenv("NATIVE_P36_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="P36_PROTECTED_SCIENCE_FIRST_USE_NOT_AUTHORIZED"):
        p36.guard()
    monkeypatch.setenv("NATIVE_P36_DEV_ONLY","1")
    assert p36.guard()["heldout"]["episodes"]==432

def test_actual_action_label_mapping_changes_decision_with_identical_model_weights():
    torch.set_num_threads(1)
    m=p35.make_model(p36.SEEDS[0],"factual_action_aux_ppo")
    with torch.no_grad():
        m.aux_predictor.weight.zero_()
        m.aux_predictor.bias.zero_()
        m.aux_predictor.weight[0,7]=-1.
        m.aux_predictor.weight[0,8]=0.
        m.aux_predictor.weight[0,9]=1.
    hist=[p35.row(-.5),p35.row(-.48,2,-1,.02)]
    pred=p36.effects(m,hist)
    perm=p36.effects(m,hist,flipped=True)
    assert pred==pytest.approx((-1,0,1))
    assert perm==pytest.approx((1,0,-1))
    before={name:weight.detach().clone() for name,weight in m.named_parameters()}
    assert p36.action_select(m,hist)==2
    assert p36.action_select(m,hist,flipped=True)==0
    hist=[p35.row(-.5),p35.row(-.52,0,-1,-.02)]
    assert p36.action_select(m,hist)==0
    assert p36.action_select(m,hist,flipped=True)==2
    assert all(torch.equal(v,dict(m.named_parameters())[k]) for k,v in before.items())

def test_three_actual_public_interventions_replay_original_source_prefix_only():
    torch.set_num_threads(1)
    e=FakeGym()
    first=p36.paired_probe(e,"original",p36.PROBE_SEEDS[0],0)
    assert first["real_probe_actions"]==3
    assert len(first["real_physical_acceleration"])==3
    assert first["real_physical_acceleration"][0]<first["real_physical_acceleration"][1]
    assert first["real_physical_acceleration"][1]<first["real_physical_acceleration"][2]
    assert len(first["original_source_trace_sha256"])==64
    x=p36.paired_probe(e,"low_force_x0.75",p36.PROBE_SEEDS[1],3)
    assert x["real_probe_actions"]==3*(3+1)
    assert len(x["public_history"])==4
    assert p36.paired_probe(e,"low_force_x0.75",p36.PROBE_SEEDS[1],3)==x
    with pytest.raises(ValueError,match="P36_CAUSAL_PROBE_NOT_PRECOMMITTED"):
        p36.paired_probe(e,"original",p36.PROBE_SEEDS[0],9)

def test_original_source_action_choice_true_reward_cold_determinism_and_no_hidden_velocity():
    torch.set_num_threads(1)
    seed=p36.SEEDS[0]
    models={a:p35.make_model(seed,a) for a in p36.MODEL_ARMS}
    gym=FakeGym()
    for world in p36.WORLDS:
        a=p36.episode(gym,p36.EVAL_SEEDS[0],seed,world,
                      "factual_aux_energy_choice",models)
        b=p36.episode(gym,p36.EVAL_SEEDS[0],seed,world,
                      "factual_aux_energy_choice",models)
        assert a==b
        assert a["true_real_source_reward"]==-a["real_source_actions"]
        assert 0<a["real_source_actions"]<=200
        assert a["source_training_updates"]==0
        assert a["source_privileged_velocity_seen_by_policy"] is False
        c=p36.episode(gym,p36.EVAL_SEEDS[0],seed,world,"two_frame_public_momentum_rule",models)
        assert a["initial_public_source_observation"]==c["initial_public_source_observation"]
    with pytest.raises(ValueError,match="P36_EVAL_SOURCE_NOT_PRECOMMITTED"):
        p36.episode(gym,1,seed,"original","seeded_random",models)

def test_frozen_success_requires_actual_source_goals_not_equal_failures():
    score={seed:{world:{arm:{"true_source_reward_mean":-200.,
       "physical_goal_successes_of_8":0} for arm in p36.ARMS}
       for world in p36.WORLDS} for seed in p36.SEEDS}
    ranks={seed:{arm:{"exact_three_action_order_fraction":0.0}
             for arm in p36.ARMS[:3]} for seed in p36.SEEDS}
    assert not any(all(g.values()) for g in p36.precommitted_gates(score,ranks).values())
    for seed in p36.SEEDS:
        for world in p36.WORLDS:
            score[seed][world][p36.ARMS[0]]={
                "true_source_reward_mean":-110.,"physical_goal_successes_of_8":8}
            score[seed][world][p36.ARMS[1]]["true_source_reward_mean"]=-200.
            score[seed][world][p36.ARMS[2]]["true_source_reward_mean"]=-200.
            score[seed][world]["two_frame_public_momentum_rule"]["true_source_reward_mean"]=-105.
        ranks[seed][p36.ARMS[0]]["exact_three_action_order_fraction"]=1.
    assert all(all(g.values()) for g in p36.precommitted_gates(score,ranks).values())

def test_missing_prior_owner_archive_fails_closed_without_retraining(tmp_path):
    with pytest.raises(RuntimeError,match="P36_P35_ORIGINAL_OWNER_ARCHIVE_UNAVAILABLE_OR_SHA_CHANGED"):
        p36.load_previous_owner(tmp_path)
