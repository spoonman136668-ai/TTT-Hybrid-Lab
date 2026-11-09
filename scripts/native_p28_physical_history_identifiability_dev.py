"""P28: source-locked factual-history identifiability under real physics changes.

The past-only observation cannot contain the hidden authored pole length
directly. Only the separate privileged diagnostic oracle may inspect length.
All action-effect labels come from TWO actually executed physical branches
from the unmodified original upstream source; never an unexecuted oracle.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from torch.nn import functional as F
from scripts import native_p20_reward_only_ppo_dev as p20
from scripts import native_p27_state_dependent_effect_dev as p27

SCHEMA="ttt-hybrid.native-p28.physical-history-identifiability-dev.v1"
PROTOCOL=ROOT/"experiments/native_p28_physical_history_identifiability_frozen.json"
SEEDS=(4013,4027,4049)
TRAIN_SEEDS=range(122000,122128)
VAL_SEEDS=range(122500,122532)
TEST_SEEDS=range(123000,123096)
TRAIN_LENGTHS=(.30,.42,.55,.70)
HELDOUT_LENGTHS=(.36,.62)
EPOCHS=64
BATCH=256
LR=.003
ARMS=("history_effect_MLP","shuffled_history_effect_MLP","history_length_MLP")

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p.get("schema")!="ttt-hybrid.native-p28.physical-history-identifiability-frozen.v1"
        or p["state"]!="FROZEN_BEFORE_NEW_SOURCE_USE_OR_IMPLEMENTATION"
        or p["parent"]["sha"]!="5a1eb9569aa7e44278a57d196f6c2a8c2c6f7c0d"
        or p["source"]["no_protected_scientific_first_use"] is not True
        or p["training"]["initialization_seeds"]!=list(SEEDS)
        or p["training"]["epochs"]!=EPOCHS
        or p["training"]["batch_size"]!=BATCH
        or p["public_source"]["fresh_training_seed_range"]!=[TRAIN_SEEDS.start,TRAIN_SEEDS.stop-1]
        or p["public_source"]["fresh_validation_seed_range"]!=[VAL_SEEDS.start,VAL_SEEDS.stop-1]
        or p["public_source"]["fresh_heldout_seed_range"]!=[TEST_SEEDS.start,TEST_SEEDS.stop-1]
        or p["public_source"]["train_lengths"]!=list(TRAIN_LENGTHS)
        or p["public_source"]["heldout_lengths"]!=list(HELDOUT_LENGTHS)):
        raise RuntimeError("P28_PROSPECTIVE_PROTOCOL_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P28_DEV_ONLY","0")!="1":
        raise RuntimeError("P28_PROTECTED_FIRST_USE_DENIED")
    return frozen()

@contextmanager
def scoped_fresh_source():
    """Uses original P27 source interpreter, never changes its committed source."""
    keys=("TRAIN_SEEDS","VAL_SEEDS","EVAL_SEEDS","TRAIN_LENGTHS","HELDOUT_LENGTHS")
    values=(TRAIN_SEEDS,VAL_SEEDS,TEST_SEEDS,TRAIN_LENGTHS,HELDOUT_LENGTHS)
    saved=[getattr(p27,k) for k in keys]
    try:
        for k,v in zip(keys,values):
            setattr(p27,k,v)
        yield
    finally:
        for k,v in zip(keys,saved):
            setattr(p27,k,v)

def true_arrays(samples):
    if not samples:
        raise RuntimeError("P28_NO_SOURCE_PAIRS")
    xs=torch.stack([s[0] for s in samples])
    effects=torch.stack([s[2]-s[1] for s in samples])
    length=torch.tensor([float(s[3]) for s in samples],dtype=torch.float32).reshape(-1,1)
    if not (xs.shape[1:]==(p20.HISTORY,6)
            and bool(torch.isfinite(xs).all())
            and bool(torch.isfinite(effects).all())):
        raise RuntimeError("P28_INVALID_HISTORICAL_SOURCE_LABEL")
    if torch.count_nonzero(xs[:,:,1]) or torch.count_nonzero(xs[:,:,3]):
        raise RuntimeError("P28_HIDDEN_VELOCITY_IN_MODEL_INPUT")
    return xs,effects,length

def visible_features(hist):
    if hist.ndim!=3 or hist.shape[1:]!=(p20.HISTORY,6):
        raise ValueError("P28_VISIBLE_HISTORY_SHAPE")
    flat=hist.double().flatten(1)
    # No knowledge of the authored length, source physics or future observations.
    cart=hist[:,:,0].double()
    theta=hist[:,:,2].double()
    actions=hist[:,:,4].double()
    derived=torch.stack((
        cart[:,-1]-cart[:,-2],
        theta[:,-1]-theta[:,-2],
        theta[:,-1]-2*theta[:,-2]+theta[:,-3],
        cart[:,-1]-2*cart[:,-2]+cart[:,-3],
        actions[:,-1], actions[:,-2],
        (theta[:,-1]-theta[:,-2])*actions[:,-1],
        (theta[:,-1]-2*theta[:,-2]+theta[:,-3])*actions[:,-2],
    ),dim=1)
    return torch.cat((torch.ones((len(hist),1),dtype=torch.float64),flat,derived),dim=1)

def oracle_features(length):
    # A separately labeled privileged source-parameter diagnostic, NOT a learner input.
    l=length.double().reshape(-1)
    return torch.stack((torch.ones_like(l),l,l*l,1/l,1/(l*l)),dim=1)

def ridge_fit(phi,targets,regularizer=1e-5):
    if phi.ndim!=2 or targets.ndim!=2 or len(phi)!=len(targets):
        raise RuntimeError("P28_MALFORMED_FIT_DATA")
    p=phi.double()
    y=targets.double()
    gram=p.T@p
    eye=torch.eye(gram.shape[0],dtype=torch.float64)
    return torch.linalg.solve(gram+regularizer*eye,p.T@y)

def predict_ridge(phi,coef):
    out=(phi.double()@coef).float()
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("P28_NONFINITE_ORACLE_OR_RIDGE")
    return out

class HistoryMLP(nn.Module):
    def __init__(self,output_dim):
        super().__init__()
        self.layers=nn.Sequential(nn.Linear(p20.HISTORY*6,32),
                                  nn.Tanh(),nn.Linear(32,16),
                                  nn.Tanh(),nn.Linear(16,output_dim))
    def forward(self,h):
        if h.ndim!=3 or h.shape[1:]!=(p20.HISTORY,6):
            raise ValueError("P28_MLP_PAST_OBSERVATION_SHAPE")
        return self.layers(h.flatten(1))

def train_model(train,val,seed,arm):
    if seed not in SEEDS or arm not in ARMS:
        raise RuntimeError("P28_UNREGISTERED_TRAINING_ARM")
    x,e,l=true_arrays(train)
    vx,ve,vl=true_arrays(val)
    output_dim=1 if arm=="history_length_MLP" else 2
    target=l if output_dim==1 else e
    vtarget=vl if output_dim==1 else ve
    torch.manual_seed(seed)
    model=HistoryMLP(output_dim)
    init=p20.model_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    data_generator=torch.Generator().manual_seed(seed+85003)
    shuffle_generator=torch.Generator().manual_seed(seed+77003)
    updates=0
    curve=[]
    start=time.perf_counter()
    for epoch in range(EPOCHS):
        features=x
        if arm=="shuffled_history_effect_MLP":
            features=x[torch.randperm(len(x),generator=shuffle_generator)]
        order=torch.randperm(len(x),generator=data_generator)
        model.train()
        for offset in range(0,len(x),BATCH):
            idx=order[offset:offset+BATCH]
            prediction=model(features[idx])
            loss=F.mse_loss(prediction,target[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P28_NONFINITE_REAL_SOURCE_TRAIN_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                    raise RuntimeError("P28_NONFINITE_FACTUAL_GRADIENT")
            opt.step()
            updates+=1
        if (epoch+1)%16==0:
            model.eval()
            with torch.no_grad():
                value=float(F.mse_loss(model(vx),vtarget))
            curve.append({"epoch":epoch+1,"validation_mse_not_used_for_selection":value})
    model.eval()
    digest=p20.model_sha(model)
    return model,{
        "arm":arm,"seed":seed,
        "original_init_model_sha256":init,
        "final_trained_model_sha256":digest,
        "learned_weights_changed":digest!=init,
        "train_source_pairs":len(train),
        "optimizer_updates":updates,
        "factual_epoch_budget":EPOCHS,
        "model_parameters":sum(p.numel() for p in model.parameters()),
        "source_length_supervision_diagnostic_only":output_dim==1,
        "sham_history_randomly_misaligned":
            arm=="shuffled_history_effect_MLP",
        "fit_wall_seconds":time.perf_counter()-start,
        "validation_curves":curve,
        "model_received_source_length_as_INPUT":False,
        "no_expert_action_labels":True,
    }

@torch.no_grad()
def effect_eval(pred,truth,train_constant):
    if not (pred.shape==truth.shape and truth.shape[1]==2):
        raise RuntimeError("P28_PAIRED_EFFECT_EVAL_SHAPE")
    mse=float(((pred-truth)**2).mean())
    reference=float(((truth-train_constant)**2).mean())
    if reference<=1e-12:
        raise RuntimeError("P28_NO_FACTUAL_PHYSICS_EFFECT_VARIANCE")
    return {
        "mse":mse,
        "train_constant_effect_mse":reference,
        "relative_skill_vs_train_constant":1-mse/reference,
        "source_pair_count":len(truth),
        "no_eval_training_or_fit":True
    }

def run(s29_root,source_root,cloudpickle_root):
    protocol=require_dev()
    start=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,source=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    with scoped_fresh_source():
        training,tc=p27.source_pairs(gym,TRAIN_SEEDS,TRAIN_LENGTHS,"train")
        validation,vc=p27.source_pairs(gym,VAL_SEEDS,TRAIN_LENGTHS,"validation")
        evaluation,ec=p27.source_pairs(gym,TEST_SEEDS,HELDOUT_LENGTHS,"evaluation")
    tx,te,tl=true_arrays(training)
    constant=te.mean(dim=0,keepdim=True)
    length_mean=float(tl.mean())
    visible_coef=ridge_fit(visible_features(tx),te)
    oracle_coef=ridge_fit(oracle_features(tl),te)
    length_coef=ridge_fit(visible_features(tx),tl)
    ridge_hashes={
      "visible_effect_sha256":hashlib.sha256(visible_coef.numpy().tobytes()).hexdigest(),
      "source_oracle_effect_sha256":hashlib.sha256(oracle_coef.numpy().tobytes()).hexdigest(),
      "visible_length_sha256":hashlib.sha256(length_coef.numpy().tobytes()).hexdigest()
    }
    data={}
    for physical_length in HELDOUT_LENGTHS:
        rows=p27.group_by_length(evaluation,physical_length)
        vx,truth,vl=true_arrays(rows)
        constant_prediction=constant.expand_as(truth)
        visible_prediction=predict_ridge(visible_features(vx),visible_coef)
        oracle_prediction=predict_ridge(oracle_features(vl),oracle_coef)
        pred_length=predict_ridge(visible_features(vx),length_coef)
        data[str(physical_length)]={
          "rows":rows,"history_x":vx,"truth":truth,"true_source_length":vl,
          "train_fitted_constant_effect":effect_eval(constant_prediction,truth,constant),
          "trained_visible_finite_difference_ridge":effect_eval(visible_prediction,truth,constant),
          "privileged_source_length_oracle_diagnostic":effect_eval(oracle_prediction,truth,constant),
          "trained_visible_ridge_length_MAE":float(torch.abs(pred_length-vl).mean()),
          "train_constant_length_MAE":float(torch.abs(length_mean-vl).mean()),
          "actual_true_effect_mean":truth.mean(dim=0).tolist()
        }
    means=[torch.tensor(data[str(l)]["actual_true_effect_mean"]) for l in HELDOUT_LENGTHS]
    physical_shift=float(((means[1]-means[0])**2).mean())
    source_oracle_pass=all(
       data[str(l)]["privileged_source_length_oracle_diagnostic"]["mse"] <=
       .75*data[str(l)]["train_fitted_constant_effect"]["mse"]
       for l in HELDOUT_LENGTHS)
    records=[]
    for seed in SEEDS:
        trained={}
        receipts={}
        for arm in ARMS:
            trained[arm],receipts[arm]=train_model(training,validation,seed,arm)
        before={arm:p20.model_sha(model) for arm,model in trained.items()}
        scores={}
        rules={}
        for physical_length in HELDOUT_LENGTHS:
            key=str(physical_length)
            d=data[key]
            x=d["history_x"]
            truth=d["truth"]
            length=d["true_source_length"]
            model_scores={}
            for arm in ARMS:
                predictions=trained[arm](x)
                if arm=="history_length_MLP":
                    model_scores[arm]={
                       "source_length_MAE":float(torch.abs(predictions-length).mean()),
                       "train_constant_length_MAE":d["train_constant_length_MAE"],
                       "source_length_provided_as_input":False}
                else:
                    model_scores[arm]=effect_eval(predictions,truth,constant)
            for field in (
                "train_fitted_constant_effect","trained_visible_finite_difference_ridge",
                "privileged_source_length_oracle_diagnostic","trained_visible_ridge_length_MAE",
                "train_constant_length_MAE","actual_true_effect_mean"):
                model_scores[field]=d[field]
            scores[key]=model_scores
            mse=model_scores["history_effect_MLP"]["mse"]
            cmse=model_scores["train_fitted_constant_effect"]["mse"]
            sham=model_scores["shuffled_history_effect_MLP"]["mse"]
            probe=model_scores["history_length_MLP"]["source_length_MAE"]
            const_mae=model_scores["train_constant_length_MAE"]
            rules["physical_variation_in_real_source"] = physical_shift>=1e-4
            rules["source_length_oracle_effect_gain_"+key]=(
                model_scores["privileged_source_length_oracle_diagnostic"]["mse"]<=.75*cmse)
            rules["visible_history_length_probe_gain_"+key]=probe<=.75*const_mae
            rules["visible_history_effect_vs_constant_"+key]=mse<=.75*cmse
            rules["visible_history_effect_vs_shuffled_"+key]=mse<=.8*sham
        after={arm:p20.model_sha(model) for arm,model in trained.items()}
        if before!=after:
            raise RuntimeError("P28_EVALUATION_MUTATED_TRAINED_MODEL")
        records.append({
          "seed":seed,"training":receipts,
          "heldout_physics_source_scores":scores,
          "frozen_identifiability_rules":rules,
          "real_effect_heterogeneity_between_source_conditions":physical_shift,
          "original_models_not_modified_during_test":True,
          "original_source_length_hidden_from_eligible_models":True,
          "trained_conventional_eligible_vs_MLP_by_length":{
            key:(scores[key]["history_effect_MLP"]["mse"] <=
                 .9*scores[key]["trained_visible_finite_difference_ridge"]["mse"])
            for key in scores
          }
        })
    elapsed=time.perf_counter()-start
    updates=EPOCHS*math.ceil(len(training)/BATCH)
    engineering=(elapsed<=protocol["qualification"]["max_public_experiment_wall_seconds"]
        and len(records)==3 and all(
          x["optimizer_updates"]==updates and x["learned_weights_changed"]
          for rec in records for x in rec["training"].values()))
    return {
      "schema":SCHEMA,
      "engineering":"PASS_SOURCE_DEV_ONLY" if engineering else "FAIL_P28_ENGINEERING",
      "frozen_public_identifiability_mechanism_all_seeds":
          all(all(r["frozen_identifiability_rules"].values()) for r in records),
      "physical_heterogeneity_squared_mean_shift":physical_shift,
      "source_oracle_upper_diagnostic_support":source_oracle_pass,
      "eligible_architecture_superiority_not_claimed":True,
      "scientific_first_use":"NOT_RUN",
      "protected_source_worlds_consumed":[],
      "original_source":source,
      "original_parent_sha":protocol["parent"]["sha"],
      "source_custody":{"training":tc,"validation":vc,"evaluation":ec,
                        "strong_conventional_fit_weight_sha256":ridge_hashes},
      "training_mean_source_length_diagnostic_only":length_mean,
      "training_mean_factual_action_effect":constant.flatten().tolist(),
      "whole_factual_process_wall_seconds":elapsed,
      "original_P27_negative_unchanged":True,
      "independent_P26_unchanged":True,
      "FORGE_activation":False,
      "model_promotion":False,
      "records":records
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
    print("P28_PUBLIC_SOURCE_IDENTIFIABILITY_RESULTS="+json.dumps({
       "engineering":result["engineering"],
       "development_supported":result["frozen_public_identifiability_mechanism_all_seeds"],
       "source_oracle_supported":result["source_oracle_upper_diagnostic_support"],
       "physical_context_variance":result["physical_heterogeneity_squared_mean_shift"],
       "train_factual_sister_pairs":result["source_custody"]["training"]["valid_pairs_total"],
       "test_factual_sister_pairs":result["source_custody"]["evaluation"]["valid_pairs_total"],
       "real_train_actions":result["source_custody"]["training"]["original_real_environment_actions"],
       "wall_seconds":result["whole_factual_process_wall_seconds"],
       "sources":"original MIT public DEV only",
       "seeds":[{
         "seed":rec["seed"],
         "rules":rec["frozen_identifiability_rules"],
         "heldout":{
           length:{
             k:({"MSE":v["mse"],"length_MAE":v.get("source_length_MAE")} if isinstance(v,dict) and "mse" in v else v)
             for k,v in scores.items() if k in (
               "history_effect_MLP","shuffled_history_effect_MLP",
               "trained_visible_finite_difference_ridge","train_fitted_constant_effect",
               "privileged_source_length_oracle_diagnostic","history_length_MLP",
               "trained_visible_ridge_length_MAE","train_constant_length_MAE")
           } for length,scores in rec["heldout_physics_source_scores"].items()}
       } for rec in result["records"]]
    },sort_keys=True))
    return 0 if result["engineering"]=="PASS_SOURCE_DEV_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
