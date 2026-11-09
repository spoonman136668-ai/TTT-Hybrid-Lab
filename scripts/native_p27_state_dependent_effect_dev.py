"""P27: factual physical state-conditioned action effects, source-pinned public DEV.

Original S29 and MIT Gymnasium only. Two actual, identical-prefix engine
instances execute opposite factual intervention actions, followed by seven
identical real actions. Models cannot inspect hidden physical parameters,
future observations, expert labels, or protected science worlds.
Prospectively frozen metrics and seeds precede this implementation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p25_paired_action_consequence_dev as p25

SCHEMA="ttt-hybrid.native-p27.state-dependent-effect-dev.v1"
PROTOCOL=ROOT/"experiments/native_p27_state_dependent_effect_frozen.json"
MODEL_SEEDS=(3851,3863,3877)
TRAIN_SEEDS=range(120000,120128)
VAL_SEEDS=range(120500,120532)
EVAL_SEEDS=range(121000,121096)
TRAIN_LENGTHS=(0.35,0.50,0.65)
HELDOUT_LENGTHS=(0.42,0.58)
PREFIXES=(4,8,12)
AFTER_ACTIONS=(0,1,1,0,0,1,1)
EPOCHS=96
BATCH=256
LR=0.005
ARMS=("contextual_factual_GRU","additive_constant_effect_GRU",
      "shuffled_history_context_GRU","trained_contextual_MLP")
HISTORY=p20.HISTORY
RANDOM_SEED_SALT=31079

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p.get("schema")!="ttt-hybrid.native-p27.state-dependent-physical-effect-frozen.v1"
        or p["state"]!="FROZEN_BEFORE_IMPLEMENTATION_OR_EXTERNAL_SOURCE_USE"
        or p["parent"]["sha"]!="c6e26fd43c783403f6037cc1e960b80f66320c42"
        or p["source"]["scientific_first_use"] is not False
        or p["models"]["seeds"]!=list(MODEL_SEEDS)
        or p["models"]["train_epochs"]!=EPOCHS
        or p["models"]["batch_size"]!=BATCH
        or p["physical_protocol"]["train_lengths"]!=list(TRAIN_LENGTHS)
        or p["physical_protocol"]["heldout_lengths"]!=list(HELDOUT_LENGTHS)
        or p["physical_protocol"]["prefix_lengths"]!=list(PREFIXES)
        or p["physical_protocol"]["independent_public_seeds"]["training"]!=
           [TRAIN_SEEDS.start,TRAIN_SEEDS.stop-1]
        or p["physical_protocol"]["independent_public_seeds"]["evaluation"]!=
           [EVAL_SEEDS.start,EVAL_SEEDS.stop-1]):
        raise RuntimeError("P27_FROZEN_PROTOCOL_OR_SOURCE_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P27_DEV_ONLY","0")!="1":
        raise RuntimeError("P27_PROTECTED_SCIENCE_FIRST_USE_DENIED")
    return frozen()

def configure_physics(env,length):
    if not (length in TRAIN_LENGTHS or length in HELDOUT_LENGTHS):
        raise RuntimeError("P27_UNREGISTERED_OR_SEALED_PHYSICS")
    base=env.unwrapped
    if (not math.isclose(float(base.length),.5,abs_tol=1e-8)
        or not math.isclose(float(base.masspole),.1,abs_tol=1e-8)
        or not math.isclose(float(base.polemass_length),.05,abs_tol=1e-8)
        or not math.isclose(float(base.gravity),9.8,abs_tol=1e-7)):
        raise RuntimeError("P27_ORIGINAL_UPSTREAM_PHYSICS_DRIFT")
    base.length=float(length)
    base.polemass_length=float(base.masspole)*float(length)
    return env

def prefix_actions(seed,prefix_length):
    if prefix_length not in PREFIXES:
        raise RuntimeError("P27_UNREGISTERED_PREFIX_LENGTH")
    rng=random.Random(seed*577+prefix_length*RANDOM_SEED_SALT)
    return tuple(action for _ in range(prefix_length//2)
                 for action in ((0,1) if rng.randrange(2)==0 else (1,0)))

def source_pairs(gym,seeds,lengths,split):
    legal={"train":(TRAIN_SEEDS,TRAIN_LENGTHS),
           "validation":(VAL_SEEDS,TRAIN_LENGTHS),
           "evaluation":(EVAL_SEEDS,HELDOUT_LENGTHS)}
    if split not in legal or list(seeds)!=list(legal[split][0]) or tuple(lengths)!=legal[split][1]:
        raise RuntimeError("P27_UNREGISTERED_OR_PROTECTED_SOURCE_DATA")
    samples=[]
    records=[]
    counts={}
    events_sha=hashlib.sha256()
    real_actions=0
    begin=time.perf_counter()
    for length in lengths:
        accepted=0
        for seed in seeds:
            for prefix in PREFIXES:
                e0=configure_physics(gym.make("CartPole-v1",render_mode=None),length)
                e1=configure_physics(gym.make("CartPole-v1",render_mode=None),length)
                event={"length":length,"seed":seed,"prefix":prefix,
                       "executed_prefix_actions":list(prefix_actions(seed,prefix)),
                       "first_interventions":[0,1],
                       "subsequent_identical_actions":list(AFTER_ACTIONS),
                       "valid":False}
                factual=[]
                try:
                    initial0,_=e0.reset(seed=seed)
                    initial1,_=e1.reset(seed=seed)
                    if p25.real_floats(initial0)!=p25.real_floats(initial1):
                        raise RuntimeError("P27_SISTER_WORLD_INITIAL_STATE_MISMATCH")
                    factual.append(p20.factual_row(initial0,0,0.))
                    prev=p25.real_floats(initial0)
                    failed=False
                    for action in event["executed_prefix_actions"]:
                        a0,r0,t0,z0,_=e0.step(action)
                        a1,r1,t1,z1,_=e1.step(action)
                        real_actions+=2
                        if (p25.real_floats(a0)!=p25.real_floats(a1)
                            or float(r0)!=float(r1) or t0!=t1 or z0!=z1):
                            raise RuntimeError("P27_PRE_INTERVENTION_WORLD_REPLAY_DIFFERED")
                        row=p20.factual_row(a0,2*action-1,float(r0))
                        factual.append(row)
                        prev=p25.real_floats(a0)
                        if t0 or z0:
                            failed=True
                            event["exclusion"]="prefix_real_termination"
                            break
                    if not failed:
                        # No privileged world length/true velocity reaches models.
                        history=p25.visible_history(factual)
                        before=prev
                        ends=[]
                        branch_hashes=[]
                        for env,intervention in ((e0,0),(e1,1)):
                            actions=(intervention,)+AFTER_ACTIONS
                            h=hashlib.sha256()
                            outcome=None
                            for action in actions:
                                obs,reward,terminated,truncated,_=env.step(action)
                                real_actions+=1
                                obsf=p25.real_floats(obs)
                                event_row=(action,obsf,float(reward),
                                           bool(terminated),bool(truncated))
                                h.update(json.dumps(event_row,separators=(",",":"),
                                                    allow_nan=False).encode())
                                outcome=obsf
                                if terminated or truncated:
                                    failed=True
                                    event["exclusion"]="post_intervention_real_termination"
                                    break
                            ends.append(outcome)
                            branch_hashes.append(h.hexdigest())
                        event["factual_branch_sha256"]=branch_hashes
                        if not failed:
                            y0=torch.tensor([ends[0][0]-before[0],
                                             10*(ends[0][2]-before[2])],
                                            dtype=torch.float32)
                            y1=torch.tensor([ends[1][0]-before[0],
                                             10*(ends[1][2]-before[2])],
                                            dtype=torch.float32)
                            if not (bool(torch.isfinite(y0).all()) and
                                    bool(torch.isfinite(y1).all())):
                                raise RuntimeError("P27_NONFINITE_FACTUAL_SOURCE_OBSERVATION")
                            event.update({"valid":True,
                                          "original_factual_preintervention_history":
                                              [list(r) for r in factual[-HISTORY:]],
                                          "real_branch_end_observation_0":ends[0],
                                          "real_branch_end_observation_1":ends[1],
                                          "target_public_position_delta0":y0.tolist(),
                                          "target_public_position_delta1":y1.tolist()})
                            samples.append((history,y0,y1,length,seed,prefix))
                            accepted+=1
                finally:
                    e0.close()
                    e1.close()
                records.append(event)
                events_sha.update(json.dumps(event,sort_keys=True,separators=(",",":"),
                                             allow_nan=False).encode())
        counts[str(length)]=accepted
        if accepted < math.ceil(len(seeds)*len(PREFIXES)*.6):
            raise RuntimeError("P27_INSUFFICIENT_SURVIVING_REAL_PAIRED_WORLDS")
    return samples,{
        "split":split,
        "physical_lengths":list(lengths),
        "registered_seed_range":[seeds.start,seeds.stop-1],
        "valid_paired_counts_by_physical_length":counts,
        "valid_pairs_total":len(samples),
        "original_real_environment_actions":real_actions,
        "all_real_sister_world_events_sha256":events_sha.hexdigest(),
        "raw_factual_real_sister_world_events":records,
        "collection_wall_seconds":time.perf_counter()-begin,
        "unexecuted_counterfactual_targets":0,
        "true_velocities_given_to_models":False,
        "source_world_length_given_to_models":False,
    }

def arrays(samples):
    h=[];act=[];targets=[]
    for history,y0,y1,*_ in samples:
        h.extend((history,history))
        act.extend((-1.,1.))
        targets.extend((y0,y1))
    return (torch.stack(h),
            torch.tensor(act,dtype=torch.float32).reshape(-1,1),
            torch.stack(targets))

class ContextGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(6,9,batch_first=True)
        self.mean=nn.Linear(9,2)
        self.effect=nn.Linear(9,2)
    def forward(self,h,a,mode="contextual"):
        if h.ndim!=3 or h.shape[1:]!=(HISTORY,6) or a.shape!=(h.shape[0],1):
            raise ValueError("P27_GRU_HISTORY_AND_ACTION_SHAPE")
        z=self.recurrent(h)[0][:,-1,:]
        if mode=="additive":
            e=self.effect(torch.zeros_like(z))
        elif mode=="contextual":
            e=self.effect(z)
        else:
            raise ValueError("P27_INVALID_INTERACTION_ABLATION")
        return self.mean(z)+a*e

class ContextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Linear(HISTORY*6,9),nn.Tanh())
        self.mean=nn.Linear(9,2)
        self.effect=nn.Linear(9,2)
    def forward(self,h,a):
        if h.ndim!=3 or h.shape[1:]!=(HISTORY,6) or a.shape!=(h.shape[0],1):
            raise ValueError("P27_MLP_HISTORY_AND_ACTION_SHAPE")
        z=self.project(h.reshape(h.shape[0],-1))
        return self.mean(z)+a*self.effect(z)

def model_forward(model,h,a,arm):
    if arm=="additive_constant_effect_GRU":
        return model(h,a,mode="additive")
    return model(h,a)

def train_arm(data,val,model_seed,arm):
    if arm not in ARMS or model_seed not in MODEL_SEEDS:
        raise RuntimeError("P27_UNREGISTERED_MODEL_OR_SEED")
    torch.manual_seed(model_seed)
    model=ContextMLP() if arm=="trained_contextual_MLP" else ContextGRU()
    initial=p20.model_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    x,a,y=arrays(data)
    vx,va,vy=arrays(val)
    perm_gen=torch.Generator().manual_seed(model_seed+73001)
    data_gen=torch.Generator().manual_seed(model_seed+83511)
    updates=0
    curve=[]
    begin=time.perf_counter()
    for epoch in range(EPOCHS):
        # A context-shuffled negative destroys historical pair association but
        # retains actual action/target and marginal history distribution.
        contextual=x
        if arm=="shuffled_history_context_GRU":
            contextual=x[torch.randperm(len(x),generator=perm_gen)]
        order=torch.randperm(len(a),generator=data_gen)
        model.train()
        for start in range(0,len(a),BATCH):
            idx=order[start:start+BATCH]
            pred=model_forward(model,contextual[idx],a[idx],arm)
            loss=F.mse_loss(pred,y[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P27_NONFINITE_SOURCE_TRAINING_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise RuntimeError("P27_NONFINITE_TRAINING_GRADIENT")
            opt.step()
            updates+=1
        if (epoch+1)%24==0:
            model.eval()
            with torch.no_grad():
                vloss=float(F.mse_loss(model_forward(model,vx,va,arm),vy))
            curve.append({"epoch":epoch+1,"heldout_training_length_validation_mse":vloss})
    model.eval()
    return model,{
        "arm":arm,"seed":model_seed,
        "training_factual_branch_examples":len(a),
        "trained_epochs":EPOCHS,"optimizer_minibatches":updates,
        "initial_model_sha256":initial,
        "trained_model_sha256":p20.model_sha(model),
        "trained_parameters":sum(z.numel() for z in model.parameters()),
        "learning_changed_weights":p20.model_sha(model)!=initial,
        "factual_action_and_consequence_only":True,
        "same_full_data_and_update_budget_as_all_arms":True,
        "sham_history_context_per_epoch_permuted":
            arm=="shuffled_history_context_GRU",
        "constant_effect_head_cannot_depend_on_history":
            arm=="additive_constant_effect_GRU",
        "validation_not_selected_or_used_to_tune_weights":True,
        "validation_curves":curve,
        "fit_seconds":time.perf_counter()-begin,
    }

def ridge_features(h,a):
    v=h.double().reshape(h.shape[0],-1)
    dpos=h[:,-1,0].double()-h[:,-2,0].double()
    dtheta=h[:,-1,2].double()-h[:,-2,2].double()
    velocity=torch.stack((dpos,dtheta),dim=1)
    stat=torch.cat((torch.ones((len(h),1),dtype=torch.float64),v,velocity),dim=1)
    return torch.cat((stat,a.double()*stat),dim=1)

def ridge_fit(data):
    h,a,y=arrays(data)
    phi=ridge_features(h,a)
    eye=torch.eye(phi.shape[1],dtype=torch.float64)
    return torch.linalg.solve(phi.T@phi+1e-5*eye,phi.T@y.double())

@torch.no_grad()
def effect_metrics(model,data,arm,constant_effect):
    if not data:
        raise RuntimeError("P27_EMPTY_REAL_PHYSICAL_EVALUATION")
    h,a,y=arrays(data)
    hs=h[::2]
    aa0=torch.full((len(hs),1),-1.,dtype=torch.float32)
    aa1=-aa0
    truth=y[1::2]-y[::2]
    if arm=="trained_interaction_ridge":
        pred0=ridge_features(hs,aa0)@model
        pred1=ridge_features(hs,aa1)@model
        effect=(pred1-pred0).float()
    elif arm=="train_fitted_constant_effect":
        effect=constant_effect.reshape(1,2).expand_as(truth)
    elif arm=="zero_action_effect":
        effect=torch.zeros_like(truth)
    else:
        effect=(model_forward(model,hs,aa1,arm)-
                model_forward(model,hs,aa0,arm))
    if not bool(torch.isfinite(effect).all()):
        raise RuntimeError("P27_NONFINITE_FACTUAL_EFFECT_PREDICTION")
    mse=float(((effect-truth)**2).mean())
    absolute_var=float(((truth-constant_effect.reshape(1,2))**2).mean())
    if absolute_var<=1e-12:
        raise RuntimeError("P27_UNINFORMATIVE_NO_CAUSAL_EFFECT_VARIANCE")
    centered_truth=float(((truth-truth.mean(dim=0,keepdim=True))**2).mean())
    centered_prediction=float(((effect-effect.mean(dim=0,keepdim=True))**2).mean())
    return {
        "factual_sister_pairs":len(data),
        "two_step_source_only_no_oracle":True,
        "action_effect_MSE":mse,
        "factual_heterogeneity_MSE_vs_train_constant":absolute_var,
        "effect_skill_vs_train_constant":1-mse/absolute_var,
        "centered_true_action_effect_variance":centered_truth,
        "centered_predicted_action_effect_variance":centered_prediction,
        "predicted_context_effect_variance_fraction":
            centered_prediction/max(centered_truth,1e-12),
        "no_eval_optimizer_steps":True,
    }

def group_by_length(samples,length):
    return [entry for entry in samples if entry[3]==length]

def run(s29_root,source_root,cloudpickle_root):
    p=require_dev()
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,owner=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    train,train_custody=source_pairs(gym,TRAIN_SEEDS,TRAIN_LENGTHS,"train")
    valid,val_custody=source_pairs(gym,VAL_SEEDS,TRAIN_LENGTHS,"validation")
    evaluation,eval_custody=source_pairs(gym,EVAL_SEEDS,HELDOUT_LENGTHS,"evaluation")
    train_effects=torch.stack([row[2]-row[1] for row in train])
    constant=train_effects.mean(dim=0).detach().clone()
    ridge=ridge_fit(train)
    ridge_sha=hashlib.sha256(ridge.numpy().tobytes()).hexdigest()
    records=[]
    for seed in MODEL_SEEDS:
        trained={}
        receipts={}
        for arm in ARMS:
            trained[arm],receipts[arm]=train_arm(train,valid,seed,arm)
        before={arm:p20.model_sha(v) for arm,v in trained.items()}
        by_length={}
        for length in HELDOUT_LENGTHS:
            rows=group_by_length(evaluation,length)
            by_length[str(length)]={
                arm:effect_metrics(trained.get(arm,ridge),rows,arm,constant)
                for arm in ARMS+("trained_interaction_ridge",)
            }
            by_length[str(length)]["train_fitted_constant_effect"] = effect_metrics(
                None,rows,"train_fitted_constant_effect",constant)
            by_length[str(length)]["zero_action_effect"] = effect_metrics(
                None,rows,"zero_action_effect",constant)
        after={arm:p20.model_sha(v) for arm,v in trained.items()}
        if before!=after:
            raise RuntimeError("P27_FACTUAL_EVALUATION_CHANGED_TRAINED_WEIGHTS")
        rules={}
        variance=[]
        predicted=[]
        strongest=[]
        for length in HELDOUT_LENGTHS:
            scores=by_length[str(length)]
            m=scores["contextual_factual_GRU"]
            constant_mse=scores["train_fitted_constant_effect"]["action_effect_MSE"]
            sham_mse=scores["shuffled_history_context_GRU"]["action_effect_MSE"]
            additive_mse=scores["additive_constant_effect_GRU"]["action_effect_MSE"]
            mlp_mse=scores["trained_contextual_MLP"]["action_effect_MSE"]
            ridge_mse=scores["trained_interaction_ridge"]["action_effect_MSE"]
            condition=str(length)
            rules["context_variance_qualified_"+condition]=constant_mse>=0.0001
            rules["context_skill_over_constant_"+condition]=m["action_effect_MSE"]<=0.8*constant_mse
            rules["context_over_sham_"+condition]=m["action_effect_MSE"]<=0.8*sham_mse
            rules["context_over_additive_"+condition]=m["action_effect_MSE"]<=0.8*additive_mse
            variance.append(m["centered_true_action_effect_variance"])
            predicted.append(m["centered_predicted_action_effect_variance"])
            strongest.append(m["action_effect_MSE"]<=.9*min(mlp_mse,ridge_mse))
        pred_fraction=sum(predicted)/max(sum(variance),1e-12)
        rules["predicted_heterogeneity_variance_at_least_15pct"]=pred_fraction>=.15
        rules["predicted_heterogeneity_variance_at_most_4x"]=pred_fraction<=4.
        records.append({
            "model_seed":seed,"training":receipts,
            "real_source_effects_by_unseen_length":by_length,
            "frozen_primary_rules":rules,
            "context_variance_fraction_pooled":pred_fraction,
            "GRU_strongest_conventional_superiority_by_length":strongest,
            "original_source_model_weights_not_modified_during_eval":True,
        })
    wall=time.perf_counter()-started
    expected=EPOCHS*math.ceil((2*len(train))/BATCH)
    engineering=(wall<=p["qualification"]["bounded_dev_wall_seconds"] and
       len(records)==3 and all(
           v["optimizer_minibatches"]==expected and v["learning_changed_weights"]
           for rec in records for v in rec["training"].values()))
    return {
        "schema":SCHEMA,
        "engineering":"PASS_PUBLIC_SOURCE_DEV_ONLY" if engineering else "FAIL_P27_ENGINEERING",
        "public_dev_frozen_state_dependent_mechanism":
            all(all(r["frozen_primary_rules"].values()) for r in records),
        "novel_hybrid_superiority_over_fully_trained_conventional":
            all(all(r["GRU_strongest_conventional_superiority_by_length"]) for r in records),
        "scientific_first_use":"NOT_RUN",
        "sealed_independent_worlds_consumed":[],
        "original_owner_source":owner,
        "preregistered_parent_sha":p["parent"]["sha"],
        "train_factual_constant_effect":constant.tolist(),
        "source_custody":{"train":train_custody,"validation":val_custody,
                          "unseen_length_evaluation":eval_custody,
                          "trained_ridge_sha256":ridge_sha},
        "experiment_wall_seconds":wall,
        "original_P25R2_result_unchanged":True,
        "independent_P26_branch_untouched":True,
        "FORGE_enrollment":False,
        "model_promotion":False,
        "records":records,
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--s29-root",type=Path,required=True)
    ap.add_argument("--source-root",type=Path,required=True)
    ap.add_argument("--cloudpickle-root",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    result=run(a.s29_root,a.source_root,a.cloudpickle_root)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    summary=[{
      "seed":r["model_seed"],
      "unseen_length":{
        l:{arm:{
            "MSE":score["action_effect_MSE"],
            "context_variance":score["factual_heterogeneity_MSE_vs_train_constant"],
            "effect_skill_vs_train_constant":score["effect_skill_vs_train_constant"],
        } for arm,score in data.items()}
        for l,data in r["real_source_effects_by_unseen_length"].items()},
      "rules":r["frozen_primary_rules"],
      "strongest_conventional":r["GRU_strongest_conventional_superiority_by_length"]
    } for r in result["records"]]
    print("P27_ORIGINAL_FACTUAL_SOURCE_FINAL="+json.dumps({
        "engineering":result["engineering"],
        "frozen_development_support":
            result["public_dev_frozen_state_dependent_mechanism"],
        "architectural_superiority":
            result["novel_hybrid_superiority_over_fully_trained_conventional"],
        "science":result["scientific_first_use"],
        "training_real_paired_worlds":result["source_custody"]["train"]["valid_pairs_total"],
        "heldout_real_paired_worlds":result["source_custody"]["unseen_length_evaluation"]["valid_pairs_total"],
        "elapsed":result["experiment_wall_seconds"],
        "seeds":summary
    },sort_keys=True))
    return 0 if result["engineering"]=="PASS_PUBLIC_SOURCE_DEV_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
