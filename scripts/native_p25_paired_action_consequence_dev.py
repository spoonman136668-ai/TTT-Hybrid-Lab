"""P25 public DEV: independent real sister-world action interventions.

No expert decisions, hidden source physics or official scientific first-use.
Two interventions are independently ACTUALLY EXECUTED after an identical
source-observed prefix; model predicts their two-step positional consequences.
A priori protocol and metric contract are frozen in earlier Git commits.
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

SCHEMA="ttt-hybrid.native-p25.paired-action-consequence-dev.v1"
PROTOCOL=ROOT/"experiments/native_p25_paired_action_consequence_frozen.json"
MODEL_SEEDS=(3671,3691,3701)
TRAIN_SEEDS=range(104000,104256)
VAL_SEEDS=range(105000,105064)
EVAL_SEEDS=range(106000,106096)
PREFIX_LENGTHS=(0,3,6,9)
EVAL_CONDITIONS=("original","gravity_x0.75","gravity_x1.25")
MODEL_ARMS=("action_conditioned_factual_GRU","sham_independent_action_GRU",
            "action_blind_same_capacity_GRU","factual_action_MLP")
MAX_WALL=600.0
EPOCHS=64
BATCH=256
LR=.008

def frozen():
    j=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (j.get("schema")!="ttt-hybrid.native-p25.paired-action-consequence-frozen.v1"
        or j["state"]!="PROSPECTIVE_FROZEN_BEFORE_IMPLEMENTATION"
        or j["parent"]["exact_sha"]!="45b7f925c28c3c58b54a7c5b61f11398474a388c"
        or j["source"]["protected_science_first_use"] is not False
        or j["models"]["model_initializations"]!=list(MODEL_SEEDS)
        or j["models"]["training_epochs"]!=EPOCHS
        or j["models"]["batch_size"]!=BATCH
        or j["paired_intervention"]["prefix_lengths"]!=list(PREFIX_LENGTHS)
        or j["paired_intervention"]["train_seed_range"]!=[101000,101255]
        or j["paired_intervention"]["validation_seed_range"]!=[102000,102063]
        or j["paired_intervention"]["eval_seed_range"]!=[103000,103095]
        or "metric_contract" not in j
        or "zero_effect_MSE" not in j["metric_contract"]):
        raise RuntimeError("P25_FROZEN_PROTOCOL_DRIFT")
    repair=json.loads((ROOT/"experiments/native_p25r1_json_custody_repair_frozen.json").read_text(encoding="utf-8"))
    if (repair.get("schema")!="ttt-hybrid.native-p25r1.json-factual-custody-repair-frozen.v1"
        or repair["state"]!="FROZEN_BEFORE_REPAIR_SOURCE_AND_FRESH_RETRY"
        or repair["origin"]["original_P25_head_sha"]!="74e1fd846a00dce4b2e0dc48f182c91cc29cbf03"
        or repair["fresh_public_dev"]["train"]!=[104000,104255]
        or repair["fresh_public_dev"]["validation"]!=[105000,105063]
        or repair["fresh_public_dev"]["evaluation"]!=[106000,106095]
        or repair["qualification"]["science_first_use_authorized"] is not False):
        raise RuntimeError("P25R1_FRESH_SOURCE_RECOVERY_NOT_FROZEN")
    return j

def require_dev():
    if os.environ.get("NATIVE_P25_DEV_ONLY","0")!="1":
        raise RuntimeError("P25_PROTECTED_SCIENCE_FIRST_USE_DENIED")
    return frozen()

def visible_history(rows):
    if not rows or any(len(r)!=6 for r in rows[-p20.HISTORY:]):
        raise ValueError("P25_INVALID_FACTUAL_INPUT")
    mask=[list(r) for r in rows[-p20.HISTORY:]]
    for r in mask:
        r[1]=0.0
        r[3]=0.0
    return p20.scaled_history(mask).reshape(p20.HISTORY,6).detach().clone()

def real_floats(observation):
    """Canonical JSON of observed upstream NumPy float32/64, not hidden state."""
    return [float(v) for v in observation]

def condition_env(env,condition):
    if condition not in EVAL_CONDITIONS:
        raise ValueError("P25_UNREGISTERED_SOURCE_CONDITION")
    expected=9.8
    if not math.isclose(float(env.unwrapped.gravity),expected,abs_tol=1e-7):
        raise RuntimeError("P25_SOURCE_GRAVITY_DRIFT")
    ratio={"original":1.0,"gravity_x0.75":.75,"gravity_x1.25":1.25}[condition]
    env.unwrapped.gravity=expected*ratio
    return ratio

def source_pairs(gym,seeds,condition,split):
    """Two physical source engines, same seed/prefix, then both intervention actions."""
    if condition not in EVAL_CONDITIONS or split not in ("train","validation","evaluation"):
        raise ValueError("P25_UNREGISTERED_SOURCE_SPLIT")
    legal={"train":TRAIN_SEEDS,"validation":VAL_SEEDS,
           "evaluation":EVAL_SEEDS}[split]
    if (list(seeds)!=list(legal) or
        (split!="evaluation" and condition!="original")):
        raise RuntimeError("P25_SEALED_OR_UNREGISTERED_WORLD_SEEDS")
    samples=[];records=[]
    digest=hashlib.sha256()
    real_steps=0
    invalid=0
    for seed in seeds:
        for prefix_len in PREFIX_LENGTHS:
            e0=gym.make("CartPole-v1",render_mode=None)
            e1=gym.make("CartPole-v1",render_mode=None)
            try:
                condition_env(e0,condition)
                condition_env(e1,condition)
                obs0,_=e0.reset(seed=seed)
                obs1,_=e1.reset(seed=seed)
                if list(obs0)!=list(obs1):
                    raise RuntimeError("P25_SISTER_RESET_NOT_EXACT")
                rows=[p20.factual_row(obs0,0,0.)]
                prefix_rng=random.Random(seed*997+prefix_len*31+71)
                prefix_actions=[prefix_rng.randrange(2) for _ in range(prefix_len)]
                bad=False
                event={"seed":seed,"prefix":prefix_len,
                       "condition":condition,"prefix_actions":prefix_actions}
                for a in prefix_actions:
                    nxt0,r0,t0,z0,_=e0.step(a)
                    nxt1,r1,t1,z1,_=e1.step(a)
                    real_steps+=2
                    if (list(nxt0)!=list(nxt1) or r0!=r1 or
                        t0!=t1 or z0!=z1):
                        raise RuntimeError("P25_SISTER_PREFIX_MISMATCH")
                    digest.update(json.dumps([seed,prefix_len,a,real_floats(nxt0),float(r0),
                                               bool(t0),bool(z0)],
                                              allow_nan=False,separators=(",",":")).encode())
                    rows.append(p20.factual_row(nxt0,2*a-1,float(r0)))
                    if t0 or z0:
                        bad=True
                        break
                if bad:
                    invalid+=1
                    records.append({**event,"valid":False,"reason":"PREFIX_TERMINATED"})
                    continue
                before=list(nxt0) if prefix_len else list(obs0)
                x=visible_history(rows)
                # Both choices really execute, in separate causal sister worlds.
                next0,r0,t0,z0,_=e0.step(0)
                next1,r1,t1,z1,_=e1.step(1)
                real_steps+=2
                if t0 or z0 or t1 or z1:
                    invalid+=1
                    records.append({**event,"valid":False,
                                    "reason":"FIRST_INTERVENTION_TERMINATED"})
                    digest.update(json.dumps([seed,prefix_len,
                                              real_floats(next0),real_floats(next1),True],
                                             allow_nan=False,separators=(",",":")).encode())
                    continue
                last0,reward0,term0,trunc0,_=e0.step(0)
                last1,reward1,term1,trunc1,_=e1.step(0)
                real_steps+=2
                y0=torch.tensor([float(last0[0])-float(before[0]),
                                 10*(float(last0[2])-float(before[2]))],
                                dtype=torch.float32)
                y1=torch.tensor([float(last1[0])-float(before[0]),
                                 10*(float(last1[2])-float(before[2]))],
                                dtype=torch.float32)
                if not (bool(torch.isfinite(y0).all()) and
                        bool(torch.isfinite(y1).all())):
                    raise RuntimeError("P25_NONFINITE_FACTUAL_CONSEQUENCE")
                samples.append((x,y0,y1,seed,prefix_len))
                event.update({
                    "valid":True,
                    "original_masked_observation":rows[-1][0:4],
                    "executed_intervention_actions":[0,1],
                    "identical_followup_actions":[0,0],
                    "actual_2step_0":real_floats(last0),
                    "actual_2step_1":real_floats(last1),
                    "factual_2step_position_targets_0":y0.tolist(),
                    "factual_2step_position_targets_1":y1.tolist(),
                    "terminated_second_action":[bool(term0 or trunc0),bool(term1 or trunc1)],
                })
                digest.update(json.dumps(event,sort_keys=True,
                                         separators=(",",":"),allow_nan=False).encode())
                records.append(event)
            finally:
                e0.close()
                e1.close()
    if len(samples)<len(legal)*len(PREFIX_LENGTHS)//2:
        raise RuntimeError("P25_INADEQUATE_VALID_PAIRED_INTERVENTIONS")
    return samples,{"split":split,"condition":condition,
                    "registered_worlds":len(legal)*len(PREFIX_LENGTHS),
                    "valid_sister_pairs":len(samples),"invalid_pairs":invalid,
                    "real_original_engine_steps":real_steps,
                    "factual_action_consequence_sha256":digest.hexdigest(),
                    "original_seed_range":[legal.start,legal.stop-1],
                    "all_sister_action_consequence_events":records}

def arrays(samples):
    xs=[];aa=[];yy=[]
    for x,y0,y1,seed,prefix in samples:
        xs.extend((x,x))
        aa.extend((-1.,1.))
        yy.extend((y0,y1))
    return (torch.stack(xs),torch.tensor(aa,dtype=torch.float32).reshape(-1,1),
            torch.stack(yy))

class ActionGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.recurrent=nn.GRU(6,9,batch_first=True)
        self.head=nn.Linear(10,2)
    def forward(self,x,a):
        if x.ndim!=3 or x.shape[1:]!=(p20.HISTORY,6) or a.shape!=(x.shape[0],1):
            raise ValueError("P25_GRU_FACTUAL_SHAPE")
        z=self.recurrent(x)[0][:,-1,:]
        return self.head(torch.cat((z,a),dim=-1))

class ActionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(p20.HISTORY*6+1,16),
                                nn.Tanh(),nn.Linear(16,2))
    def forward(self,x,a):
        if x.ndim!=3 or x.shape[1:]!=(p20.HISTORY,6) or a.shape!=(x.shape[0],1):
            raise ValueError("P25_MLP_FACTUAL_SHAPE")
        return self.head(torch.cat((x.flatten(1),a),dim=-1))

def action_input(a,arm,pair_flip=None):
    if arm=="action_blind_same_capacity_GRU":
        return torch.zeros_like(a)
    if arm=="sham_independent_action_GRU" and pair_flip is not None:
        return a*pair_flip
    return a

def train_model(samples,validation,seed,arm):
    if seed not in MODEL_SEEDS or arm not in MODEL_ARMS:
        raise ValueError("P25_UNREGISTERED_MODEL")
    # Same seed means identical initial weights and optimizer data order
    # across all three GRU variants.
    torch.manual_seed(seed)
    model=ActionMLP() if arm=="factual_action_MLP" else ActionGRU()
    initial=p20.model_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    x,a,y=arrays(samples)
    vx,va,vy=arrays(validation)
    rng=torch.Generator().manual_seed(seed+900001)
    flips=torch.where(torch.rand(len(samples),generator=torch.Generator().manual_seed(
            seed+77001))>=.5,1.,-1.).repeat_interleave(2).reshape(-1,1)
    curve=[];updates=0
    begin=time.perf_counter()
    for epoch in range(EPOCHS):
        order=torch.randperm(len(a),generator=rng)
        model.train()
        for offset in range(0,len(a),BATCH):
            ii=order[offset:offset+BATCH]
            pred=model(x[ii],action_input(a[ii],arm,flips[ii]))
            loss=F.mse_loss(pred,y[ii])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P25_NONFINITE_ACTUAL_SOURCE_TRAIN_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for param in model.parameters():
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise RuntimeError("P25_NONFINITE_FACTUAL_GRADIENT")
            opt.step()
            updates+=1
        if (epoch+1)%16==0:
            model.eval()
            with torch.no_grad():
                pred=model(vx,action_input(va,arm))
                curve.append({"epoch":epoch+1,"validation_mean_squared_observation_delta":
                              float(F.mse_loss(pred,vy))})
    model.eval()
    return model,{"arm":arm,"seed":seed,
                  "initial_weight_sha256":initial,
                  "trained_weights_sha256":p20.model_sha(model),
                  "weights_updated":p20.model_sha(model)!=initial,
                  "model_parameters":sum(p.numel() for p in model.parameters()),
                  "optimizer_minibatches":updates,
                  "training_factual_real_source_examples":len(a),
                  "validation_curves_monitor_only":curve,
                  "train_wall_seconds":time.perf_counter()-begin,
                  "no_teacher_action_labels":True,
                  "used_same_factual_pairs_for_all_arms":True,
                  "sham_random_action_pair_sign":arm=="sham_independent_action_GRU",
                  "no_protected_science_first_use":True}

def ridge_features(x,a):
    # Only current/past public position, angle and finite differences.
    cart=x[:,-1,0].double()/0.5
    theta=x[:,-1,2].double()/10.0
    prev_cart=x[:,-2,0].double()/0.5
    prev_theta=x[:,-2,2].double()/10.0
    dcart=(cart-prev_cart)/0.02
    dtheta=(theta-prev_theta)/0.02
    action=a[:,0].double()
    return torch.stack((torch.ones_like(cart),cart,theta,dcart,dtheta,
        action,torch.sin(theta),torch.cos(theta),action*cart,
        action*theta,action*dcart,action*dtheta,theta*dcart,
        theta*dtheta),dim=1)

def ridge_fit(samples):
    x,a,y=arrays(samples)
    features=ridge_features(x,a)
    # Fixed small Tikhonov, fully trained on identical factual source pairs.
    gram=features.T @ features
    eye=torch.eye(gram.shape[0],dtype=torch.float64)
    weights=torch.linalg.solve(gram+1e-5*eye,features.T @ y.double())
    return weights

@torch.no_grad()
def effect_metrics(model,samples,arm,device=None):
    x,a,y=arrays(samples)
    x_even=x[::2]
    y_effect=y[1::2]-y[::2]
    act0=torch.full((len(samples),1),-1.,dtype=torch.float32)
    act1=-act0
    if arm=="factual_finite_difference_ridge":
        pred0=ridge_features(x_even,act0) @ model
        pred1=ridge_features(x_even,act1) @ model
        effect=(pred1-pred0).float()
    else:
        pred0=model(x_even,action_input(act0,arm))
        pred1=model(x_even,action_input(act1,arm))
        effect=pred1-pred0
    if not bool(torch.isfinite(effect).all()):
        raise RuntimeError("P25_NONFINITE_ACTUAL_SOURCE_PREDICTION")
    mse=float(((effect-y_effect)**2).mean())
    zero=float((y_effect**2).mean())
    if zero<=1e-12:
        raise RuntimeError("P25_NO_SOURCE_ACTION_EFFECT")
    # Prespecified: action-effect skill against zero difference, not centered R2.
    return {"real_paired_worlds":len(samples),
            "action_effect_mse":mse,
            "zero_action_effect_reference_mse":zero,
            "effect_R2_against_zero":1.-mse/zero,
            "model_action_consequence_pair_MSE":mse,
            "actual_action_effect_mean":y_effect.mean(dim=0).tolist(),
            "real_intervention_sign_accuracy_theta":
                float((torch.sign(effect[:,1])==torch.sign(y_effect[:,1]))
                      .float().mean()),
            "actual_action_consequences_not_oracle":True}

def run(s29_root,source_root,cloudpickle_root):
    protocol=require_dev()
    started=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,source=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    train,train_receipt=source_pairs(gym,TRAIN_SEEDS,"original","train")
    val,val_receipt=source_pairs(gym,VAL_SEEDS,"original","validation")
    evaluation={}
    for condition in EVAL_CONDITIONS:
        evaluation[condition]=source_pairs(gym,EVAL_SEEDS,condition,"evaluation")
    ridge=ridge_fit(train)
    ridge_sha=hashlib.sha256(ridge.numpy().tobytes()).hexdigest()
    records=[]
    for seed in MODEL_SEEDS:
        models={};fit={}
        for arm in MODEL_ARMS:
            m,receipt=train_model(train,val,seed,arm)
            models[arm]=m
            fit[arm]=receipt
        models["factual_finite_difference_ridge"]=ridge
        before={arm:p20.model_sha(m) for arm,m in models.items()
                if arm in MODEL_ARMS}
        scores={}
        for condition,(samples,receipt) in evaluation.items():
            scores[condition]={arm:effect_metrics(model,samples,arm)
                               for arm,model in models.items()}
        after={arm:p20.model_sha(m) for arm,m in models.items()
               if arm in MODEL_ARMS}
        if before!=after:
            raise RuntimeError("P25_ONLINE_EVAL_CHANGED_TRAINED_WEIGHTS")
        s=scores["original"]
        main=s["action_conditioned_factual_GRU"]["action_effect_mse"]
        sham=s["sham_independent_action_GRU"]["action_effect_mse"]
        blind=s["action_blind_same_capacity_GRU"]["action_effect_mse"]
        ridge_mse=s["factual_finite_difference_ridge"]["action_effect_mse"]
        mlp_mse=s["factual_action_MLP"]["action_effect_mse"]
        gates={
          "original_vs_sham_at_least_2x_mse_gain":main<=.5*sham,
          "original_vs_action_blind_at_least_2x_mse_gain":main<=.5*blind,
          "original_effect_skill_min_0_8":
              s["action_conditioned_factual_GRU"]["effect_R2_against_zero"]>=.8,
          "gravity_x0_75_effect_skill_min_0_7":
              scores["gravity_x0.75"]["action_conditioned_factual_GRU"]["effect_R2_against_zero"]>=.7,
          "gravity_x1_25_effect_skill_min_0_7":
              scores["gravity_x1.25"]["action_conditioned_factual_GRU"]["effect_R2_against_zero"]>=.7,
        }
        records.append({
            "seed":seed,"factual_training":fit,"evaluation_scores":scores,
            "frozen_mechanism_rules":gates,
            "trained_conventional_ridge_and_MLP_compared":True,
            "strongest_conventional_mse":min(ridge_mse,mlp_mse),
            "GRU_better_than_strongest_control_by_10pct":
                main<=.9*min(ridge_mse,mlp_mse),
            "weights_not_updated_in_evaluation":True,
        })
    wall=time.perf_counter()-started
    expected_train_updates=EPOCHS*math.ceil(2*len(train)/BATCH)
    engineering=(wall<=MAX_WALL and len(records)==3 and
        all(x["optimizer_minibatches"]==expected_train_updates
            and x["weights_updated"] for rec in records
            for x in rec["factual_training"].values()))
    return {
        "schema":SCHEMA,
        "status":"PASS_PUBLIC_DEV_ENGINEERING_ONLY" if engineering else "FAIL_P25_ENGINEERING",
        "frozen_mechanism_support_all_seeds":
            all(all(rec["frozen_mechanism_rules"].values()) for rec in records),
        "strongest_conventional_architecture_superiority_proven":False,
        "source":source,
        "frozen_source_sha":protocol["parent"]["exact_sha"],
        "source_custody":{
            "train":train_receipt,"validation":val_receipt,
            "eval_conditions":{c:receipt for c,(_,receipt) in evaluation.items()},
            "ridge_trained_weights_sha256":ridge_sha,
            "fresh_train_pairs":len(train),"fresh_validation_pairs":len(val),
            "future_eval_paired_worlds":{c:len(v[0]) for c,v in evaluation.items()}
        },
        "science_first_use":"NOT_RUN",
        "official_sealed_world_data_consumed":[],
        "original_P24_failed_reward_only_result_preserved":True,
        "independent_P23_source_modified":False,
        "FORGE_activation":False,
        "project_model_promotion":False,
        "trained_action_choice_policy":False,
        "wall_seconds":wall,
        "records":records,
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    output=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(output,sort_keys=True,indent=2)+"\n",
                            encoding="utf-8")
    print(json.dumps({
        "status":output["status"],"mechanism":output["frozen_mechanism_support_all_seeds"],
        "wall":output["wall_seconds"],"science":output["science_first_use"],
        "evaluated_pairs":output["source_custody"]["future_eval_paired_worlds"],
        "mean_original_per_seed":[{
            "seed":r["seed"],"original":{
               a:{k:v[k] for k in ("action_effect_mse","effect_R2_against_zero")}
               for a,v in r["evaluation_scores"]["original"].items()},
            "frozen":r["frozen_mechanism_rules"]} for r in output["records"]]
    },sort_keys=True))
    return 0 if output["status"]=="PASS_PUBLIC_DEV_ENGINEERING_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
