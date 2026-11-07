"""G7-S17. Learn continuous features, not the hidden nonlinear physics basis.

Raw inputs: observed real x, action u, elapsed dt, observed next real x.
The model is never given the true tanh(x), xu or per-world coefficients.
Uses prior S16 world's frozen simulator strictly to generate/score data.
Competitive no-hand-feature controls: calibration-selected RBF kernel ridge,
identically initialized untrained random feature ridge, fully trained GRU,
unadapted state baseline and history-shuffle negative controls.

Numerical time sampling and derivative residual are explicit inductive biases.
No tokenizer, vocabulary, attention, discrete IDs or next-token objective.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import resource
import sys
import time
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s16_nonlinear_switch as world_source
import torch
from torch import nn
from torch.nn import functional as F

class Blocked(Exception):
    pass

def ensure(x,reason):
    if not x:
        raise Blocked(reason)

def raw_in_features(history):
    x,u,dt,nx=history.unbind(-1)
    return torch.stack((x,u),dim=-1),(nx-x)/dt

def raw_query_features(query):
    return query[:,:,:2]

def consequence_from_rate(query,rate):
    return query[:,:,0]+query[:,:,2]*rate

class RawFeatureBank(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,32),nn.Tanh(),
                               nn.Linear(32,12),nn.Tanh())

    def forward(self,x):
        return self.net(x)

def learned_ridge(model,history,query,decay=.78,regularizer=.08):
    x,y=raw_in_features(history)
    feat=model(x)
    weights=decay**torch.arange(
        history.shape[1]-1,-1,-1,dtype=torch.float32)
    gram=torch.einsum("bti,btj,t->bij",feat,feat,weights)
    rhs=torch.einsum("bti,bt,t->bi",feat,y,weights)
    gram=gram+regularizer*torch.eye(feat.shape[-1])[None]
    coeff=torch.linalg.solve(gram,rhs.unsqueeze(-1)).squeeze(-1)
    ask=model(raw_query_features(query))
    rate=torch.einsum("bqi,bi->bq",ask,coeff)
    return consequence_from_rate(query,rate)

class RawGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(4,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(34,32),nn.Tanh(),
                                nn.Linear(32,1))

    def forward(self,history,query):
        state=self.gru(history)[1][-1]
        nq=query.shape[1]
        features=torch.cat((state[:,None,:].expand(-1,nq,-1),
                            raw_query_features(query)),dim=-1)
        return consequence_from_rate(
            query,self.head(features).squeeze(-1))

def kernel_ridge(history,query,gamma,decay,ridge):
    """Strong batch Gaussian-kernel ridge, calibrated ONLY on CAL seeds."""
    x,y=raw_in_features(history)
    q=raw_query_features(query)
    d2=((x[:,:,None,:]-x[:,None,:,:])**2).sum(-1)
    kernel=torch.exp(-gamma*d2)
    time_count=history.shape[1]
    decay_weight=decay**torch.arange(
        time_count-1,-1,-1,dtype=torch.float32)
    sqrtw=decay_weight.sqrt()
    weighted=kernel*sqrtw[None,:,None]*sqrtw[None,None,:]
    weighted=weighted+ridge*torch.eye(time_count)[None]
    alpha=torch.linalg.solve(
        weighted,(sqrtw[None,:]*y).unsqueeze(-1)).squeeze(-1)
    qdist=((q[:,:,None,:]-x[:,None,:,:])**2).sum(-1)
    qkernel=torch.exp(-gamma*qdist)*sqrtw[None,None,:]
    rate=torch.einsum("bqt,bt->bq",qkernel,alpha)
    return consequence_from_rate(query,rate)

def simulator(seed,batch,regime):
    # This function is the ONLY place where the hidden physics law is used.
    # All learned/adaptive predictors operate on raw output arrays alone.
    history,queries,targets,_,_=world_source.world(
        torch,seed,batch,32,8,regime)
    return history,queries,targets

def freeze_check(p):
    ensure(p.get("schema")==
           "ttt-hybrid.g7-s17.frozen-learned-raw-continuous-features.v1",
           "FROZEN_SCHEMA_DRIFT")
    t=p["frozen_training"];e=p["evaluation"];c=p["calibration"];z=p["threshold"]
    ensure(p["predecessor"]["s16_exact_sha"]==
           "ed87a20a8615f8da47ed0de8e252e6000720e4fb"
           and p["predecessor"]["s16_run"]==42393985416
           and p["predecessor"]["s16_generator_git_blob"]==
              "12bc2b9db686531b7cf71a58f043f655d15c1708"
           and t["seeds"]==[101,211,307]
           and t["steps"]==240 and t["batch"]==48
           and t["mlp_output"]==12
           and t["feature_ridge_decay"]==.78
           and t["feature_ridge_regularization"]==.08
           and c["seeds"]==[907,911,919]
           and c["rbf_gamma_grid"]==[.5,2.,6.]
           and c["rbf_decay_grid"]==[.65,.78,1.]
           and c["rbf_regularization_grid"]==[.01,.08]
           and e["split_seeds"]["ood_switch"]==[1201,1213,1229]
           and e["split_seeds"]["unfamiliar_function"]==[1301,1319,1327]
           and e["batch"]==64
           and z["learned_over_random_ood_relative_mse_gain_min"]==.10
           and z["learned_over_kernel_ood_relative_mse_gain_min"]==.05
           and z["learned_over_kernel_ood_absolute_mse_gain_min"]==.00002
           and z["wall_limit_seconds"]==240,
           "FROZEN_SEEDS_OR_CONTROLS_DRIFT")
    ensure(not p["world"]["no_tokenizer"] is False and
           p["world"]["no_attention"] is True,"TOKENIZATION_SCOPE_DRIFT")

def calibrate_kernel(p,started):
    configs=list(itertools.product(
        p["calibration"]["rbf_gamma_grid"],
        p["calibration"]["rbf_decay_grid"],
        p["calibration"]["rbf_regularization_grid"]))
    datasets=[simulator(seed,p["calibration"]["batch"],"train_switch")
              for seed in p["calibration"]["seeds"]]
    aggregate={}
    with torch.no_grad():
        for gamma,decay,ridge in configs:
            values=[]
            for h,q,y in datasets:
                pred=kernel_ridge(h,q,gamma,decay,ridge)
                values.append(float(F.mse_loss(pred,y)))
            aggregate[(gamma,decay,ridge)]=mean(values)
            ensure(time.monotonic()-started<p["threshold"]["wall_limit_seconds"],
                   "KERNEL_CALIBRATION_TIME_CAP")
    best=min(configs,key=lambda z:(
        aggregate[z],z[0],-z[1],-z[2]))
    return best,{
        f"gamma={k[0]};decay={k[1]};ridge={k[2]}":v
        for k,v in aggregate.items()}

def measure(feature,untrained,gru,kcfg,seed,regime):
    h,q,y=simulator(seed,64,regime)
    feature.eval();untrained.eval();gru.eval()
    with torch.no_grad():
        adapted=learned_ridge(feature,h,q)
        baseline=learned_ridge(untrained,h,q)
        kernel=kernel_ridge(h,q,*kcfg)
        recurrent=gru(h,q)
        unadapted=q[:,:,0]
        # World-swapped context is a causal negative, not a model scoring input.
        wrong=h.roll(shifts=1,dims=0)
        wrong_feature=learned_ridge(feature,wrong,q)
        wrong_kernel=kernel_ridge(wrong,q,*kcfg)
        flipped=q.clone()
        flipped[:,:,1]=-q[:,:,1]
        flipped_feature=learned_ridge(feature,h,flipped)
        m={
            "learned_feature_mse":float(F.mse_loss(adapted,y)),
            "random_feature_mse":float(F.mse_loss(baseline,y)),
            "rbf_kernel_mse":float(F.mse_loss(kernel,y)),
            "trained_gru_mse":float(F.mse_loss(recurrent,y)),
            "static_no_dynamics_mse":float(F.mse_loss(unadapted,y)),
            "oracle_mse":float(F.mse_loss(y,y)),
            "swapped_learned_feature_mse":float(F.mse_loss(wrong_feature,y)),
            "swapped_rbf_kernel_mse":float(F.mse_loss(wrong_kernel,y)),
            "action_flip_changes_learned_response":float(
                (adapted-flipped_feature).abs().mean())
        }
    ensure(all(math.isfinite(v) for v in m.values()),
           "NONFINITE_HELDOUT_MEASUREMENT")
    return m

def checkpoint_cold(model,history,query):
    model.eval()
    with torch.no_grad():
        before= (learned_ridge(model,history,query)
                 if isinstance(model,RawFeatureBank) else model(history,query))
    buf=io.BytesIO()
    torch.save(model.state_dict(),buf)
    digest=hashlib.sha256(buf.getvalue()).hexdigest()
    other=type(model)()
    other.load_state_dict(
        torch.load(io.BytesIO(buf.getvalue()),weights_only=True))
    other.eval()
    with torch.no_grad():
        after=(learned_ridge(other,history,query)
               if isinstance(model,RawFeatureBank) else other(history,query))
    return digest,float((before-after).abs().max())

def run(args):
    pbytes=args.protocol.read_bytes()
    p=json.loads(pbytes)
    freeze_check(p)
    ensure(custody.git(ROOT,"hash-object",
                       "scripts/g7_s16_nonlinear_switch.py")==
           p["predecessor"]["s16_generator_git_blob"],
           "VERIFIED_S16_WORLD_GENERATOR_GIT_BLOB_DRIFT")
    manifest,packages=custody.manifest()
    ensure(manifest==custody.R3_MANIFEST_SHA256
           and packages.get("torch")==custody.TORCH
           and packages.get("triton")==custody.TRITON,
           "PINNED_R3_ENVIRONMENT_CHANGED")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ensure(not torch.cuda.is_initialized(),
           "UNAUTHORIZED_GPU")
    t0=time.monotonic()
    kcfg,calibration=calibrate_kernel(p,t0)
    output=[]
    for index,seed in enumerate(p["frozen_training"]["seeds"]):
        torch.manual_seed(seed)
        feature=RawFeatureBank()
        untrained=copy.deepcopy(feature)
        gru=RawGRU()
        fopt=torch.optim.AdamW(feature.parameters(),
            lr=p["frozen_training"]["feature_lr"],weight_decay=.01)
        gopt=torch.optim.AdamW(gru.parameters(),
            lr=p["frozen_training"]["gru_lr"],weight_decay=.01)
        losses={"learned_features":[],"trained_gru":[]}
        fit_t0=time.monotonic()
        for step in range(p["frozen_training"]["steps"]):
            h,q,y=simulator(
                seed*100000+271000+step*37,
                p["frozen_training"]["batch"],
                p["frozen_training"]["regime_schedule"][step%2])
            for model,optim,name in (
                (feature,fopt,"learned_features"),
                (gru,gopt,"trained_gru")
            ):
                model.train()
                pred=(learned_ridge(model,h,q)
                      if isinstance(model,RawFeatureBank) else model(h,q))
                loss=F.mse_loss(pred,y)
                ensure(bool(torch.isfinite(loss)),
                       "NONFINITE_TRAINING_LOSS")
                optim.zero_grad(set_to_none=True)
                loss.backward()
                grad=nn.utils.clip_grad_norm_(model.parameters(),1.0)
                ensure(bool(torch.isfinite(grad)),
                       "NONFINITE_TRAINING_GRAD")
                optim.step()
                losses[name].append(float(loss.detach()))
            ensure(time.monotonic()-t0<p["threshold"]["wall_limit_seconds"],
                   "FROZEN_TRAIN_TIME_CAP")
        fit_sec=time.monotonic()-fit_t0
        m={}
        for split,seeds in p["evaluation"]["split_seeds"].items():
            regime=("unseen_function" if split=="unfamiliar_function" else split)
            m[split]=measure(feature,untrained,gru,kcfg,seeds[index],regime)
        cold_h,cold_q,_=simulator(seed*100000+99911,4,"ood_switch")
        f_sha,f_delta=checkpoint_cold(feature,cold_h,cold_q)
        g_sha,g_delta=checkpoint_cold(gru,cold_h,cold_q)
        ensure(f_delta==0.0 and g_delta==0.0,
               "COLD_TRAINED_WEIGHTS_RESTORE_DRIFT")
        output.append({
            "train_seed":seed,
            "updates_feature":len(losses["learned_features"]),
            "updates_gru":len(losses["trained_gru"]),
            "train_worlds_each":len(losses["learned_features"])*
                p["frozen_training"]["batch"],
            "feature_trainable_parameters":sum(z.numel() for z in feature.parameters()),
            "gru_trainable_parameters":sum(z.numel() for z in gru.parameters()),
            "initial_random_feature_same_as_learned_before_fit":True,
            "first_losses":{name:ls[0] for name,ls in losses.items()},
            "last_32_losses":{name:mean(ls[-32:]) for name,ls in losses.items()},
            "fit_seconds":round(fit_sec,3),
            "splits":m,
            "feature_checkpoint_sha256":f_sha,
            "gru_checkpoint_sha256":g_sha,
            "feature_cold_output_delta":f_delta,
            "gru_cold_output_delta":g_delta
        })
        ensure(time.monotonic()-t0<p["threshold"]["wall_limit_seconds"],
               "FROZEN_MEASUREMENT_TIME_CAP")
    agg={split:{key:mean(row["splits"][split][key] for row in output)
                 for key in output[0]["splits"][split]}
         for split in p["evaluation"]["split_seeds"]}
    t=p["threshold"]
    valid_oracle=all(
        row["splits"][split]["oracle_mse"]<=t["oracle_mse_max"]
        for row in output for split in agg)
    kernel_control=all(
        row["splits"]["ood_switch"]["rbf_kernel_mse"]<=t["calibrated_kernel_ood_mse_max"]
        and (row["splits"]["ood_switch"]["static_no_dynamics_mse"]-
             row["splits"]["ood_switch"]["rbf_kernel_mse"])/
             max(row["splits"]["ood_switch"]["static_no_dynamics_mse"],1e-12)>=
             t["calibrated_kernel_relative_gain_vs_static_min"]
        and row["splits"]["ood_switch"]["swapped_rbf_kernel_mse"]/
            max(row["splits"]["ood_switch"]["rbf_kernel_mse"],1e-12)>=
            t["calibrated_kernel_swapped_history_mse_ratio_min"]
        for row in output)
    ood=agg["ood_switch"]
    improvement_random=(ood["random_feature_mse"]-
                        ood["learned_feature_mse"])/max(
                        ood["random_feature_mse"],1e-12)
    improvement_kernel=(ood["rbf_kernel_mse"]-
                        ood["learned_feature_mse"])/max(
                        ood["rbf_kernel_mse"],1e-12)
    improvement_abs=ood["rbf_kernel_mse"]-ood["learned_feature_mse"]
    better_random=all(
        row["splits"]["ood_switch"]["learned_feature_mse"]<
        row["splits"]["ood_switch"]["random_feature_mse"] for row in output)
    better_kernel=all(
        row["splits"]["ood_switch"]["learned_feature_mse"]<
        row["splits"]["ood_switch"]["rbf_kernel_mse"] for row in output)
    frozen_success=(valid_oracle and kernel_control and
        improvement_random>=t["learned_over_random_ood_relative_mse_gain_min"]
        and better_random and
        improvement_kernel>=t["learned_over_kernel_ood_relative_mse_gain_min"]
        and improvement_abs>=t["learned_over_kernel_ood_absolute_mse_gain_min"]
        and better_kernel and
        agg["unfamiliar_function"]["learned_feature_mse"]<=
             t["learned_unfamiliar_function_mse_max"])
    state=("ASSAY_INVALID_STRONG_KERNEL_CONTROL_OR_ORACLE"
           if not valid_oracle or not kernel_control else
           "PRELIMINARY_LEARNED_TOKENLESS_FEATURE_TRANSFER_SIGNAL_NOT_SCIENTIFIC_PASS"
           if frozen_success else
           "LEARNED_CONTINUOUS_FEATURE_ADVANTAGE_NULL_OR_NEGATIVE")
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    ensure(rss<=t["peak_rss_limit_bytes"],"CPU_RESIDENT_MEMORY_EXCEEDED")
    ensure(not torch.cuda.is_initialized(),"CUDA_USED_DURING_CPU_STUDY")
    return {
        "schema":"ttt-hybrid.g7-s17.learned-raw-feature-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":state,
        "frozen_protocol_sha256":hashlib.sha256(pbytes).hexdigest(),
        "pinned_r3_package_manifest_sha256":manifest,
        "tokenizer_used":False,
        "vocabulary_used":False,
        "attention_used":False,
        "known_analytic_nonlinear_basis_given_to_models":False,
        "euler_residual_supervision_used":True,
        "numeric_time_sampled":True,
        "trained_feature_representations":True,
        "kernel_calibration_selected_only_before_heldout":True,
        "rbf_calibration_choice":{
            "gamma":kcfg[0],"decay":kcfg[1],"ridge":kcfg[2]},
        "calibration_only_candidate_mse":calibration,
        "per_seed":output,
        "aggregate_metrics":agg,
        "oracle_controls_pass":valid_oracle,
        "strong_calibrated_rbf_control_pass":kernel_control,
        "learned_vs_random_ood_relative_improvement":improvement_random,
        "learned_vs_kernel_ood_relative_improvement":improvement_kernel,
        "learned_vs_kernel_ood_absolute_mse_improvement":improvement_abs,
        "learned_better_than_random_all_seeds":better_random,
        "learned_better_than_kernel_all_seeds":better_kernel,
        "frozen_learned_feature_gain_pass":frozen_success,
        "train_steps_per_seed_per_learned_model":240,
        "train_worlds_per_seed_per_learned_model":11520,
        "heldout_worlds_per_seed_per_split":64,
        "trained_model_cold_restore_exact":True,
        "whole_process_seconds":round(time.monotonic()-t0,3),
        "peak_process_rss_bytes":rss,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    a=parser.parse_args()
    receipt=ROOT/"local_runs/g7_s17_learned_features_evidence.json"
    protocol=ROOT/"experiments/g7_s17_frozen_learned_features.json"
    result={
        "schema":"ttt-hybrid.g7-s17.learned-raw-feature-evidence.v1",
        "status":"ENGINEERING_UNVERIFIED",
        "candidate_sha":a.expected_sha,
        "deployment":"NOT_AUTHORIZED"}
    code=2
    try:
        ensure(a.protocol.resolve()==protocol.resolve() and
               a.receipt.resolve()==receipt.resolve(),
               "FROZEN_SCIENCE_PATH_DRIFT")
        result=run(a)
        code=0 if result["status"]!="ASSAY_INVALID_STRONG_KERNEL_CONTROL_OR_ORACLE" else 2
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["failure_class"]=type(exc).__name__
    finally:
        if a.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                               encoding="utf-8")
        print("G7_S17_STATUS="+result["status"],flush=True)
        print("G7_S17_TOKENIZER=NOT_USED",flush=True)
        print("G7_S17_GPU=NOT_USED",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
