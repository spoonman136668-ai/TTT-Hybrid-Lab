"""P29: identical-width raw versus past-only derivative neural source learning.

Only original S29-admitted MIT CartPole source and ACTUALLY executed sister-world
action differences. Freeze experiment thresholds/seeds before implementation.
No policy/expert actions, source parameters or future information given to models.
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
from scripts import native_p28_physical_history_identifiability_dev as p28

SCHEMA="ttt-hybrid.native-p29.past-only-feature-access-dev.v1"
PROTOCOL=ROOT/"experiments/native_p29_feature_access_frozen.json"
SEEDS=(4201,4217,4229)
TRAIN_SEEDS=range(124000,124128)
VAL_SEEDS=range(124500,124532)
TEST_SEEDS=range(125000,125096)
TRAIN_LENGTHS=(.28,.40,.56,.72)
HELDOUT_LENGTHS=(.33,.63)
EPOCHS=64
BATCH=256
LR=.003
HISTORY=p20.HISTORY
DELTA_T=.02
ARMS=("raw_MLP","derived_MLP","shuffled_derived_MLP","raw_GRU",
      "derived_GRU","shuffled_derived_GRU","raw_length_MLP","derived_length_MLP")

def frozen():
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    if (p.get("schema")!="ttt-hybrid.native-p29.past-only-feature-access-frozen.v1"
        or p["status"]!="FROZEN_BEFORE_IMPLEMENTATION_AND_SOURCE_USE"
        or p["parent"]["exact_sha"]!="15ebf21b5e1799428635a0ea7182ccec604084dd"
        or p["source"]["protected_science_first_use_not_authorized"] is not True
        or p["training"]["model_seeds"]!=list(SEEDS)
        or p["training"]["epochs"]!=EPOCHS
        or p["training"]["batch_size"]!=BATCH
        or p["data"]["training_source_reset_seeds"]!=[TRAIN_SEEDS.start,TRAIN_SEEDS.stop-1]
        or p["data"]["validation_reset_seeds"]!=[VAL_SEEDS.start,VAL_SEEDS.stop-1]
        or p["data"]["heldout_reset_seeds"]!=[TEST_SEEDS.start,TEST_SEEDS.stop-1]
        or p["data"]["train_engine_lengths"]!=list(TRAIN_LENGTHS)
        or p["data"]["heldout_engine_lengths"]!=list(HELDOUT_LENGTHS)
        or p["data"]["actual_source_prefix_lengths"]!=list(p27.PREFIXES)):
        raise RuntimeError("P29_FROZEN_SOURCE_HYPOTHESIS_DRIFT")
    return p

def require_dev():
    if os.environ.get("NATIVE_P29_DEV_ONLY")!="1":
        raise RuntimeError("P29_SEALED_SCIENCE_FIRST_USE_DENIED")
    return frozen()

@contextmanager
def scoped_source():
    """Use unchanged original physical engine with fresh P29 world ranges."""
    fields=("TRAIN_SEEDS","VAL_SEEDS","EVAL_SEEDS","TRAIN_LENGTHS","HELDOUT_LENGTHS")
    v=(TRAIN_SEEDS,VAL_SEEDS,TEST_SEEDS,TRAIN_LENGTHS,HELDOUT_LENGTHS)
    old=[getattr(p27,k) for k in fields]
    try:
        for k,value in zip(fields,v):
            setattr(p27,k,value)
        yield
    finally:
        for k,value in zip(fields,old):
            setattr(p27,k,value)

def raw_and_targets(pairs):
    # The source-locked P28 implementation refuses true source velocity columns.
    return p28.true_arrays(pairs)

def two_views(raw):
    """Six original channels + four deterministic previous-only motion features."""
    if (raw.ndim!=3 or raw.shape[1:]!=(HISTORY,6)
        or bool(torch.count_nonzero(raw[:,:,1]))
        or bool(torch.count_nonzero(raw[:,:,3]))):
        raise ValueError("P29_VISIBLE_PAST_ONLY_SOURCE_SHAPE_OR_VELOCITY")
    v=torch.zeros((len(raw),HISTORY,10),dtype=torch.float32)
    v[:,:,:6]=raw.clone()
    delta=raw[:,1:,:]-raw[:,:-1,:]
    v[:,1:,6]=delta[:,:,0]/DELTA_T
    v[:,1:,7]=delta[:,:,2]/DELTA_T
    second=raw[:,2:,:]-2*raw[:,1:-1,:]+raw[:,:-2,:]
    v[:,2:,8]=second[:,:,0]/(DELTA_T*DELTA_T)
    v[:,2:,9]=second[:,:,2]/(DELTA_T*DELTA_T)
    if not bool(torch.isfinite(v).all()):
        raise RuntimeError("P29_NONFINITE_DETERMINISTIC_PAST_DIFFERENCES")
    raw_padded=v.clone()
    raw_padded[:,:,6:]=0.
    return raw_padded,v

class TrainOnlyNormalizer:
    """Training history statistics. Both arms have equal raw-channel scaling."""
    def __init__(self,training_hist):
        raw,derived=two_views(training_hist)
        self.raw_mean=raw[:,:,:6].mean(dim=(0,1))
        self.raw_std=raw[:,:,:6].std(dim=(0,1),unbiased=False).clamp_min(1e-4)
        self.derived_mean=derived[:,:,6:].mean(dim=(0,1))
        self.derived_std=derived[:,:,6:].std(dim=(0,1),unbiased=False).clamp_min(1e-4)
        self.training_hist_sha256=hashlib.sha256(
            training_hist.contiguous().numpy().tobytes()).hexdigest()
        self.stats_sha256=hashlib.sha256(torch.cat((
            self.raw_mean,self.raw_std,self.derived_mean,self.derived_std
        )).numpy().tobytes()).hexdigest()
    def views(self,history):
        r,d=two_views(history)
        r[:,:,:6]=(r[:,:,:6]-self.raw_mean)/self.raw_std
        d[:,:,:6]=(d[:,:,:6]-self.raw_mean)/self.raw_std
        d[:,:,6:]=(d[:,:,6:]-self.derived_mean)/self.derived_std
        if not torch.allclose(r[:,:,:6],d[:,:,:6],atol=0.,rtol=0.):
            raise RuntimeError("P29_RAW_SCALING_NOT_IDENTICAL_ACROSS_ARMS")
        return r,d

class EffectMLP(nn.Module):
    def __init__(self,out_channels):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(HISTORY*10,32),
                                 nn.Tanh(),nn.Linear(32,16),
                                 nn.Tanh(),nn.Linear(16,out_channels))
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,10):
            raise ValueError("P29_MLP_SOURCE_HISTORY_SHAPE")
        return self.model(x.flatten(1))

class EffectGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(10,9,batch_first=True)
        self.head=nn.Linear(9,2)
    def forward(self,x):
        if x.ndim!=3 or x.shape[1:]!=(HISTORY,10):
            raise ValueError("P29_GRU_SOURCE_HISTORY_SHAPE")
        return self.head(self.gru(x)[0][:,-1,:])

def is_derived(arm):
    return arm.startswith("derived_") or arm.startswith("shuffled_derived_")

def is_sham(arm):
    return arm.startswith("shuffled_derived_")

def make_model(arm):
    if arm not in ARMS:
        raise RuntimeError("P29_UNREGISTERED_MODEL")
    if arm.endswith("_GRU"):
        return EffectGRU()
    return EffectMLP(1 if arm.endswith("_length_MLP") else 2)

def train_arm(train,val,normalizer,arm,seed):
    if seed not in SEEDS or arm not in ARMS:
        raise RuntimeError("P29_UNREGISTERED_REAL_SOURCE_TRAINING")
    x,y,l=raw_and_targets(train)
    vx,vy,vl=raw_and_targets(val)
    train_raw,train_derived=normalizer.views(x)
    valid_raw,valid_derived=normalizer.views(vx)
    target=l if arm.endswith("_length_MLP") else y
    val_target=vl if arm.endswith("_length_MLP") else vy
    torch.manual_seed(seed)
    model=make_model(arm)
    before=p20.model_sha(model)
    opt=torch.optim.AdamW(model.parameters(),lr=LR)
    train_gen=torch.Generator().manual_seed(seed+71003)
    sham_gen=torch.Generator().manual_seed(seed+81007)
    updates=0
    trace=[]
    begin=time.perf_counter()
    for epoch in range(EPOCHS):
        if is_derived(arm):
            feat=train_derived
            if is_sham(arm):
                shuffle=torch.randperm(len(x),generator=sham_gen)
                feat=train_derived.clone()
                feat[:,:,6:]=train_derived[shuffle,:,6:]
        else:
            feat=train_raw
        order=torch.randperm(len(x),generator=train_gen)
        model.train()
        for offset in range(0,len(x),BATCH):
            idx=order[offset:offset+BATCH]
            predicted=model(feat[idx])
            loss=F.mse_loss(predicted,target[idx])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("P29_REAL_FACTUAL_TRAINING_LOSS_NONFINITE")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            for par in model.parameters():
                if par.grad is not None and not bool(torch.isfinite(par.grad).all()):
                    raise RuntimeError("P29_NONFINITE_REAL_SOURCE_GRADIENT")
            opt.step()
            updates+=1
        if (epoch+1)%16==0:
            model.eval()
            with torch.no_grad():
                val_input=valid_derived if is_derived(arm) else valid_raw
                val_loss=float(F.mse_loss(model(val_input),val_target))
            trace.append({"epoch":epoch+1,
                          "validation_no_best_epoch_selection_mse":val_loss})
    model.eval()
    final=p20.model_sha(model)
    return model,{
        "model_arm":arm,
        "initialization_seed":seed,
        "input_channels":10,
        "raw_visible_channels_preserved":6,
        "additional_features_only_own_past":4 if is_derived(arm) else 0,
        "equal_model_parameters":sum(p.numel() for p in model.parameters()),
        "fixed_factual_training_epochs":EPOCHS,
        "trained_on_factual_public_pairs":len(x),
        "optimizer_minibatches":updates,
        "initial_weight_sha256":before,
        "final_weight_sha256":final,
        "weights_changed":before!=final,
        "normalizer_training_source_sha256":normalizer.training_hist_sha256,
        "normalizer_train_only_stats_sha256":normalizer.stats_sha256,
        "sham_derivatives_misaligned_each_epoch":is_sham(arm),
        "source_length_is_never_an_input":True,
        "no_expert_policy_actions":True,
        "development_validation_curve":trace,
        "training_wall_seconds":time.perf_counter()-begin,
    }

@torch.no_grad()
def evaluate_effect(model,source_pairs,normalizer,arm,constant):
    raw,y,l=raw_and_targets(source_pairs)
    r,d=normalizer.views(raw)
    input_tensor=d if is_derived(arm) else r
    pred=model(input_tensor)
    if pred.shape!=y.shape or not bool(torch.isfinite(pred).all()):
        raise RuntimeError("P29_FAILED_REAL_SOURCE_EVAL_SHAPE_OR_FINITE")
    mse=float(((pred-y)**2).mean())
    reference=float(((y-constant)**2).mean())
    if reference<=1e-12:
        raise RuntimeError("P29_NO_ACTUAL_SOURCE_ACTION_VARIANCE")
    return {
        "real_source_sister_pairs":len(source_pairs),
        "real_action_effect_mse":mse,
        "constant_effect_mse":reference,
        "effect_skill_vs_train_constant":1-mse/reference,
        "no_eval_training":True
    }

@torch.no_grad()
def evaluate_length(model,source_pairs,normalizer,arm,constant_length):
    raw,y,l=raw_and_targets(source_pairs)
    r,d=normalizer.views(raw)
    pred=model(d if is_derived(arm) else r)
    if pred.shape!=l.shape or not bool(torch.isfinite(pred).all()):
        raise RuntimeError("P29_NONFINITE_PUBLIC_LENGTH_PROBE")
    return {
        "real_source_sister_pairs":len(source_pairs),
        "source_length_MAE":float(torch.abs(pred-l).mean()),
        "train_mean_length_MAE":float(torch.abs(constant_length-l).mean()),
        "source_length_used_only_as_supervision_target":True,
        "no_eval_training":True
    }

def run(s29_root,source_root,cloudpickle_root):
    contract=require_dev()
    begin=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    gym,admission=p20.admit_gymnasium(s29_root,source_root,cloudpickle_root)
    with scoped_source():
        train,train_c=p27.source_pairs(gym,TRAIN_SEEDS,TRAIN_LENGTHS,"train")
        val,val_c=p27.source_pairs(gym,VAL_SEEDS,TRAIN_LENGTHS,"validation")
        heldout,eval_c=p27.source_pairs(gym,TEST_SEEDS,HELDOUT_LENGTHS,"evaluation")
    raw,y,l=raw_and_targets(train)
    normalizer=TrainOnlyNormalizer(raw)
    constant=y.mean(dim=0,keepdim=True)
    constant_length=l.mean().reshape(1,1)
    ridge=p28.ridge_fit(p28.visible_features(raw),y)
    strong_ridge_sha256=hashlib.sha256(ridge.numpy().tobytes()).hexdigest()
    length_ridge=p28.ridge_fit(p28.visible_features(raw),l)
    diagnostic_oracle=p28.ridge_fit(p28.oracle_features(l),y)
    separate={}
    for length in HELDOUT_LENGTHS:
        rows=p27.group_by_length(heldout,length)
        hx,ey,hl=raw_and_targets(rows)
        vr=p28.predict_ridge(p28.visible_features(hx),ridge)
        diagnostic=p28.predict_ridge(p28.oracle_features(hl),diagnostic_oracle)
        length_ridge_prediction=p28.predict_ridge(p28.visible_features(hx),length_ridge)
        separate[str(length)]={
            "rows":rows,
            "original_fully_trained_visible_ridge_effect_MSE":float(((vr-ey)**2).mean()),
            "separate_privileged_length_oracle_MSE":float(((diagnostic-ey)**2).mean()),
            "eligible_visible_ridge_length_MAE":float(torch.abs(length_ridge_prediction-hl).mean()),
            "train_fitted_constant_effect_MSE":float(((ey-constant)**2).mean()),
            "train_fitted_constant_length_MAE":float(torch.abs(hl-constant_length).mean()),
            "real_factual_effect_mean":ey.mean(dim=0).tolist()
        }
    effect_shift=torch.stack([
        torch.tensor(separate[str(v)]["real_factual_effect_mean"])
        for v in HELDOUT_LENGTHS
    ])
    physical_shift=float(((effect_shift[1]-effect_shift[0])**2).mean())
    receipts=[]
    for seed in SEEDS:
        trained={}
        training={}
        for arm in ARMS:
            trained[arm],training[arm]=train_arm(train,val,normalizer,arm,seed)
        before={arm:p20.model_sha(m) for arm,m in trained.items()}
        evaluations={}
        primary={}
        eligible_superiority=[]
        for length in HELDOUT_LENGTHS:
            key=str(length)
            data=separate[key]
            sample=data["rows"]
            scores={
                arm:(evaluate_length(trained[arm],sample,normalizer,arm,constant_length)
                     if arm.endswith("_length_MLP")
                     else evaluate_effect(trained[arm],sample,normalizer,arm,constant))
                for arm in ARMS
            }
            scores["fully_fitted_past_visible_ridge"]={
                "real_action_effect_mse":data["original_fully_trained_visible_ridge_effect_MSE"],
                "source_length_MAE":data["eligible_visible_ridge_length_MAE"],
                "strong_conventional_source_length_input":False}
            scores["separate_privileged_length_oracle_diagnostic"]={
                "real_action_effect_mse":data["separate_privileged_length_oracle_MSE"],
                "source_length_used_as_privileged_input":True}
            scores["train_fitted_constant_effect"]={
                "real_action_effect_mse":data["train_fitted_constant_effect_MSE"]}
            scores["train_fitted_constant_length"]={
                "source_length_MAE":data["train_fitted_constant_length_MAE"]}
            evaluations[key]=scores
            rawmlp=scores["raw_MLP"]["real_action_effect_mse"]
            made=scores["derived_MLP"]["real_action_effect_mse"]
            sham=scores["shuffled_derived_MLP"]["real_action_effect_mse"]
            rawgru=scores["raw_GRU"]["real_action_effect_mse"]
            mgru=scores["derived_GRU"]["real_action_effect_mse"]
            shamgru=scores["shuffled_derived_GRU"]["real_action_effect_mse"]
            longraw=scores["raw_length_MLP"]["source_length_MAE"]
            longderived=scores["derived_length_MLP"]["source_length_MAE"]
            constlen=scores["train_fitted_constant_length"]["source_length_MAE"]
            primary.update({
                "real_physical_effect_variation_"+key:physical_shift>=1e-4,
                "derived_MLP_at_least_25pct_better_than_raw_"+key:made<=.75*rawmlp,
                "derived_MLP_at_least_20pct_better_than_sham_"+key:made<=.80*sham,
                "derived_GRU_at_least_25pct_better_than_raw_"+key:mgru<=.75*rawgru,
                "derived_GRU_at_least_20pct_better_than_sham_"+key:mgru<=.80*shamgru,
                "derived_length_MLP_at_least_20pct_better_than_raw_"+key:
                    longderived<=.80*longraw,
                "derived_length_MLP_at_least_25pct_better_than_train_mean_"+key:
                    longderived<=.75*constlen
            })
            best_ridge=data["original_fully_trained_visible_ridge_effect_MSE"]
            eligible_superiority.append({
                "heldout_length":length,
                "derived_MLP_better_than_ridge_10pct":made<=.90*best_ridge,
                "derived_GRU_better_than_ridge_10pct":mgru<=.90*best_ridge,
            })
        after={arm:p20.model_sha(m) for arm,m in trained.items()}
        if before!=after:
            raise RuntimeError("P29_SOURCE_EVAL_MODIFIED_TRAINED_WEIGHTS")
        receipts.append({
            "seed":seed,
            "trained_arms":training,
            "unseen_physics_effects":evaluations,
            "frozen_rules":primary,
            "strong_conventional_comparisons":eligible_superiority,
            "no_source_oracle_labels_in_neural_inputs":True,
            "all_models_unchanged_during_source_eval":True
        })
    wall=time.perf_counter()-begin
    expected_updates=EPOCHS*math.ceil(len(train)/BATCH)
    engineering=(wall<=contract["qualification"]["experiment_wall_limit_seconds"]
       and len(receipts)==3 and all(
         arm["optimizer_minibatches"]==expected_updates and arm["weights_changed"]
         for rec in receipts for arm in rec["trained_arms"].values()))
    architecture_superiority=all(all(
        x["derived_MLP_better_than_ridge_10pct"]
        and x["derived_GRU_better_than_ridge_10pct"]
        for x in rec["strong_conventional_comparisons"]) for rec in receipts)
    return {
        "schema":SCHEMA,
        "engineering":"PASS_ORIGINAL_SOURCE_DEV_ONLY" if engineering else "FAIL_P29_ENGINEERING",
        "frozen_feature_access_mechanism_all_three_seeds":
            all(all(rec["frozen_rules"].values()) for rec in receipts),
        "architecture_superiority_over_strong_original_ridge":architecture_superiority,
        "real_source_physical_effect_between_lengths_variance":physical_shift,
        "original_source_admission":admission,
        "science_first_use":"NOT_RUN",
        "independent_P26_not_modified":True,
        "P28_original_negative_unchanged":True,
        "FORGE_activation":False,
        "accepted_model_promotion":False,
        "fixed_train_only_normalizer_sha256":normalizer.stats_sha256,
        "original_strong_ridge_weights_sha256":strong_ridge_sha256,
        "source_custody":{"training":train_c,"validation":val_c,"evaluation":eval_c},
        "training_examples":len(train),
        "heldout_examples":len(heldout),
        "whole_physical_process_wall_seconds":wall,
        "records":receipts
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--s29-root",type=Path,required=True)
    parser.add_argument("--source-root",type=Path,required=True)
    parser.add_argument("--cloudpickle-root",type=Path,required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    result=run(args.s29_root,args.source_root,args.cloudpickle_root)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                            encoding="utf-8")
    summary=[{
        "seed":r["seed"],"frozen_rules":r["frozen_rules"],
        "heldout":{
            length:{
                arm:(result["real_action_effect_mse"]
                     if "real_action_effect_mse" in result
                     else result.get("source_length_MAE"))
                for arm,result in controls.items()}
            for length,controls in r["unseen_physics_effects"].items()}
    } for r in result["records"]]
    print("P29_FROZEN_REAL_SOURCE_FEATURE_ACCESS_SUMMARY="+json.dumps({
        "engineering":result["engineering"],
        "feature_access_mechanism":result["frozen_feature_access_mechanism_all_three_seeds"],
        "strong_ridge_superiority":result["architecture_superiority_over_strong_original_ridge"],
        "science_first_use":result["science_first_use"],
        "physical_effect_shift":result["real_source_physical_effect_between_lengths_variance"],
        "training_real_pairs":result["training_examples"],
        "heldout_real_pairs":result["heldout_examples"],
        "real_train_source_actions":result["source_custody"]["training"]["original_real_environment_actions"],
        "real_heldout_source_actions":result["source_custody"]["evaluation"]["original_real_environment_actions"],
        "wall_seconds":result["whole_physical_process_wall_seconds"],
        "seeds":summary
    },sort_keys=True))
    return 0 if result["engineering"]=="PASS_ORIGINAL_SOURCE_DEV_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
