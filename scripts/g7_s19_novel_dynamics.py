"""G7-S19: frozen two-new-family token-free transfer and sequential interference.

All predictive models see ONLY numerically sampled continuous (x,action,dt,
measured_next_x) and intervention queries. The true dynamics formulas below
are used solely inside the synthetic evaluator to generate and score data;
they are never passed to the learned MLP, 32-unit GRU, or RBF-KRR control.

Training phase A uses the ORIGINAL S16 world and the byte-identical S18
model/evaluation implementations. Phase B trains 96 additional steps on a
novel sine/action-tanh world then checks old-world retention and new-world
acquisition, with completely untouched frozen heldout observations.

No vocabulary, token embeddings, attention, or next-token loss. Discrete
sensor sampling/Euler-difference supervision remain explicit inductive biases.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import resource
from statistics import mean
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s18_causal_feature_replication as s18
import torch
from torch import nn
from torch.nn import functional as F

class Blocked(Exception):
    pass

def ensure(ok,message):
    if not ok:
        raise Blocked(message)

ORIGINAL_SIMULATOR=s18.simulator

def novel_law(q,hidden,family):
    """World-generator truth only. Never called by a predictive model."""
    x,u,dt=q.unbind(-1)
    a=hidden[:,0:1]
    b=hidden[:,1:2]
    c=hidden[:,2:3]
    d=hidden[:,3:4]
    if family=="novel_sine":
        derivative=-a*x+b*u+c*torch.sin(4*x)+d*torch.tanh(3*u)
    elif family=="novel_rational":
        derivative=(-a*torch.tanh(2*x)+b*u+
                    c*x.pow(3)/(1+x.square())+d*torch.sin(3*u))
    else:
        raise Blocked("UNREGISTERED_NOVEL_PHYSICS_FAMILY")
    return x+dt*derivative

def sample_hidden(generator,batch,ood):
    r=lambda:torch.rand(batch,1,generator=generator)
    if ood:
        coeffs=(1.55+.70*r(),1.55+.55*r(),
                -.8+1.6*r(),-.65+1.3*r())
    else:
        coeffs=(.35+1.05*r(),.35+1.1*r(),
                -.45+.9*r(),-.35+.7*r())
    return torch.cat(coeffs,dim=-1)

def novel_simulator(seed,batch,regime):
    if regime not in ("novel_sine_train_switch",
                      "novel_sine_ood_switch",
                      "novel_rational_ood_switch"):
        return ORIGINAL_SIMULATOR(seed,batch,regime)
    family="novel_sine" if regime.startswith("novel_sine") else "novel_rational"
    ood=regime!="novel_sine_train_switch"
    generator=torch.Generator(device="cpu").manual_seed(int(seed))
    old=sample_hidden(generator,batch,ood)
    changed=sample_hidden(generator,batch,ood)
    state=-1+2*torch.rand(batch,1,generator=generator)
    observations=[]
    for step in range(32):
        u=-1+2*torch.rand(batch,1,generator=generator)
        dt=.05+.11*torch.rand(batch,1,generator=generator)
        query=torch.stack(
            (state.squeeze(-1),u.squeeze(-1),dt.squeeze(-1)),dim=-1
        )[:,None,:]
        coef=changed if step>=20 else old
        next_state=novel_law(query,coef,family)
        observed_next=next_state+.002*torch.randn(
            batch,1,generator=generator)
        observations.append(torch.cat((state,u,dt,observed_next),dim=-1))
        state=next_state
    context=torch.stack(observations,dim=1)
    query_x=state.expand(-1,8)
    query_u=-1+2*torch.rand(batch,8,generator=generator)
    query_dt=.05+.11*torch.rand(batch,8,generator=generator)
    queries=torch.stack((query_x,query_u,query_dt),dim=-1)
    target=novel_law(queries,changed,family)
    ensure(tuple(context.shape)==(batch,32,4)
           and tuple(queries.shape)==(batch,8,3)
           and tuple(target.shape)==(batch,8)
           and bool((queries[:,:,2]>=.05).all())
           and bool((queries[:,:,2]<=.16).all())
           and bool(torch.isfinite(context).all())
           and bool(torch.isfinite(target).all()),
           "NOVEL_WORLD_SHAPE_OR_PHYSICAL_FINITE_INVARIANT")
    return context,queries,target

# The S18 source generator and numerical model functions are reused without
# byte edits. Only its evaluation data source is injected to handle novel laws.
s18.simulator=novel_simulator

def check_frozen(p):
    t=p["frozen_training"]
    e=p["evaluation"]
    c=p["calibration"]
    q=p["threshold"]
    ensure(
        p.get("schema")==
            "ttt-hybrid.g7-s19.novel-dynamics-sequential-interference.v1"
        and p["predecessor"]["s18_exact_sha"]==
            "affe574f00d69abb8d816589ed12d81e716fc458"
        and p["predecessor"]["s18_science_run"]==42396385331
        and p["predecessor"]["s18_original_source_git_blob"]==
            "cfc1959b8812e9abad581018462302bc89a31d59"
        and t["seeds"]==[701,809,907]
        and t["steps"]==240 and t["batch"]==48
        and t["mlp_output"]==12
        and t["feature_ridge_decay"]==.78
        and t["feature_ridge_regularization"]==.08
        and c["seeds"]==[1901,1913,1931]
        and c["rbf_gamma_grid"]==[.5,2.,6.]
        and c["rbf_decay_grid"]==[.65,.78,1.]
        and c["rbf_regularization_grid"]==[.01,.08]
        and e["split_seeds"]=={
            "source_ood_switch":[2027,2039,2053],
            "novel_sine_ood_switch":[2101,2113,2129],
            "novel_rational_ood_switch":[2203,2213,2221]}
        and e["sequential_interference"]["phase_B_updates"]==96
        and e["sequential_interference"]["phase_B_seed_formula"]==
            "train_seed*100000+711000+step*37"
        and q["phase_B_source_retention_MSE_ratio_max"]==1.5
        and q["phase_B_sine_improvement_relative_min"]==.10
        and q["novel_sine_zero_shot_vs_kernel_gain_relative_min"]==.05
        and q["novel_rational_zero_shot_vs_kernel_gain_relative_min"]==.05
        and q["wall_limit_seconds"]==240
        and p["world"]["no_tokenizer"] is True
        and p["world"]["no_attention"] is True
        and p["resource_authority"]["cpu_only"] is True,
        "FROZEN_S19_SOURCE_PROTOCOL_OR_THRESHOLDS_DRIFT")

def evaluate_seed(feature,random_features,gru,kcfg,p,index):
    measured={}
    for name,seed_list in p["evaluation"]["split_seeds"].items():
        seed=seed_list[index]
        regime="ood_switch" if name=="source_ood_switch" else name
        measured[name]=s18.measure(
            feature,random_features,gru,kcfg,seed,regime)
    return measured

def train_phase(feature,gru,opt_feature,opt_gru,p,seed,stage,start):
    t=p["frozen_training"]
    steps=t["steps"] if stage=="A" else p["evaluation"][
        "sequential_interference"]["phase_B_updates"]
    losses={"learned_features":[],"trained_gru":[]}
    t0=time.monotonic()
    for step in range(steps):
        if stage=="A":
            episode_seed=seed*100000+513000+step*37
            regime=t["regime_schedule"][step%2]
        else:
            episode_seed=seed*100000+711000+step*37
            regime="novel_sine_train_switch"
        h,q,y=novel_simulator(episode_seed,t["batch"],regime)
        for model,opt,name in (
            (feature,opt_feature,"learned_features"),
            (gru,opt_gru,"trained_gru")
        ):
            model.train()
            pred=(s18.learned_ridge(model,h,q)
                  if isinstance(model,s18.RawFeatureBank) else model(h,q))
            loss=F.mse_loss(pred,y)
            ensure(bool(torch.isfinite(loss)),
                   "NONFINITE_PHASE_"+stage+"_LOSS")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
            ensure(bool(torch.isfinite(norm)),
                   "NONFINITE_PHASE_"+stage+"_GRADIENT")
            opt.step()
            losses[name].append(float(loss.detach()))
        ensure(time.monotonic()-start<p["threshold"]["wall_limit_seconds"],
               "WHOLE_PROCESS_FROZEN_TIME_LIMIT")
    return {"steps":steps,
            "train_worlds":steps*t["batch"],
            "last_32_loss":{k:mean(v[-32:]) for k,v in losses.items()},
            "elapsed_seconds":round(time.monotonic()-t0,3)}

def run(a):
    pbytes=a.protocol.read_bytes()
    p=json.loads(pbytes)
    check_frozen(p)
    ensure(custody.git(ROOT,"hash-object",
                       "scripts/g7_s18_causal_feature_replication.py")==
           p["predecessor"]["s18_original_source_git_blob"],
           "ORIGINAL_SOURCE_IMPLEMENTATION_DRIFT")
    manifest,packages=custody.manifest()
    ensure(manifest==custody.R3_MANIFEST_SHA256 and
           packages.get("torch")==custody.TORCH and
           packages.get("triton")==custody.TRITON,
           "R3_PINNED_ENVIRONMENT_DRIFT")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    ensure(not torch.cuda.is_initialized(),"UNAUTHORIZED_GPU")
    started=time.monotonic()
    kcfg,calib=s18.calibrate_kernel(p,started)
    per_seed=[]
    for index,seed in enumerate(p["frozen_training"]["seeds"]):
        torch.manual_seed(seed)
        feature=s18.RawFeatureBank()
        random_features=copy.deepcopy(feature)
        gru=s18.RawGRU()
        fopt=torch.optim.AdamW(feature.parameters(),
           lr=p["frozen_training"]["feature_lr"],weight_decay=.01)
        gopt=torch.optim.AdamW(gru.parameters(),
           lr=p["frozen_training"]["gru_lr"],weight_decay=.01)
        phase_a=train_phase(
            feature,gru,fopt,gopt,p,seed,"A",started)
        first=evaluate_seed(feature,random_features,gru,kcfg,p,index)
        chk_h,chk_q,_=novel_simulator(
            seed*100000+99123,4,"novel_sine_ood_switch")
        before_fsha,before_fdelta=s18.checkpoint_cold(
            feature,chk_h,chk_q)
        before_gsha,before_gdelta=s18.checkpoint_cold(
            gru,chk_h,chk_q)
        ensure(before_fdelta==0.0 and before_gdelta==0.0,
               "COLD_CHECKPOINT_PHASE_A_DRIFT")
        phase_b=train_phase(
            feature,gru,fopt,gopt,p,seed,"B",started)
        # Same immutable heldout seeds as phase A, and no optimizer
        # uses any of these heldout examples.
        after={}
        for name in ("source_ood_switch","novel_sine_ood_switch"):
            seen=p["evaluation"]["split_seeds"][name][index]
            regime="ood_switch" if name=="source_ood_switch" else name
            after[name]=s18.measure(
                feature,random_features,gru,kcfg,seen,regime)
        after_fsha,after_fdelta=s18.checkpoint_cold(
            feature,chk_h,chk_q)
        after_gsha,after_gdelta=s18.checkpoint_cold(
            gru,chk_h,chk_q)
        ensure(after_fdelta==0.0 and after_gdelta==0.0,
               "COLD_CHECKPOINT_PHASE_B_DRIFT")
        per_seed.append({
            "seed":seed,
            "phase_A":phase_a,
            "phase_B":phase_b,
            "phase_A_model_cold_shas":{
                "feature":before_fsha,"gru":before_gsha},
            "phase_B_model_cold_shas":{
                "feature":after_fsha,"gru":after_gsha},
            "cold_max_abs_deltas":[before_fdelta,before_gdelta,
                                   after_fdelta,after_gdelta],
            "learned_feature_params":sum(
                x.numel() for x in feature.parameters()),
            "trained_gru_params":sum(x.numel() for x in gru.parameters()),
            "phase_A_heldout":first,
            "phase_B_heldout":after,
            "phase_B_original_mse_retention_ratio":(
                after["source_ood_switch"]["learned_feature_mse"]/
                max(first["source_ood_switch"]["learned_feature_mse"],1e-12)),
            "phase_B_sine_mse_improvement_fraction":(
                first["novel_sine_ood_switch"]["learned_feature_mse"]-
                after["novel_sine_ood_switch"]["learned_feature_mse"])/
                max(first["novel_sine_ood_switch"]["learned_feature_mse"],1e-12)
        })
        ensure(time.monotonic()-started<p["threshold"]["wall_limit_seconds"],
               "TOTAL_STUDY_FROZEN_TIME_LIMIT")
    splits=list(p["evaluation"]["split_seeds"])
    agg={
        split:{metric:mean(row["phase_A_heldout"][split][metric]
                          for row in per_seed)
               for metric in per_seed[0]["phase_A_heldout"][split]}
        for split in splits}
    after_agg={
        split:{metric:mean(row["phase_B_heldout"][split][metric]
                          for row in per_seed)
               for metric in per_seed[0]["phase_B_heldout"][split]}
        for split in ("source_ood_switch","novel_sine_ood_switch")}
    t=p["threshold"]
    oracle_valid=all(
        row["phase_A_heldout"][name]["oracle_mse"]<=t["oracle_mse_max"]
        for row in per_seed for name in splits)
    source_controls=all(
        row["phase_A_heldout"]["source_ood_switch"]["rbf_kernel_mse"]<=
            t["calibrated_kernel_source_ood_mse_max"]
        and (row["phase_A_heldout"]["source_ood_switch"][
                 "static_no_dynamics_mse"]-
             row["phase_A_heldout"]["source_ood_switch"][
                 "rbf_kernel_mse"])/
            max(row["phase_A_heldout"]["source_ood_switch"][
                 "static_no_dynamics_mse"],1e-12)>=
            t["calibrated_kernel_source_gain_over_static_min"]
        and row["phase_A_heldout"]["source_ood_switch"][
             "post_switch_ablated_kernel_mse"]/
            max(row["phase_A_heldout"]["source_ood_switch"][
                 "rbf_kernel_mse"],1e-12)>=
            t["calibrated_kernel_source_post_switch_ablation_ratio_min"]
        and row["phase_A_heldout"]["source_ood_switch"][
             "post_switch_ablated_learned_mse"]/
            max(row["phase_A_heldout"]["source_ood_switch"][
                 "learned_feature_mse"],1e-12)>=
            t["learned_source_post_switch_ablation_ratio_min"]
        for row in per_seed)
    novel_checks={}
    for name in ("novel_sine_ood_switch","novel_rational_ood_switch"):
        m=agg[name]
        rel_kernel=(m["rbf_kernel_mse"]-
                    m["learned_feature_mse"])/max(
                    m["rbf_kernel_mse"],1e-12)
        rel_random=(m["random_feature_mse"]-
                    m["learned_feature_mse"])/max(
                    m["random_feature_mse"],1e-12)
        absolute=m["rbf_kernel_mse"]-m["learned_feature_mse"]
        each_seed_better=all(
            row["phase_A_heldout"][name]["learned_feature_mse"]<
            row["phase_A_heldout"][name]["rbf_kernel_mse"]
            and row["phase_A_heldout"][name]["learned_feature_mse"]<
            row["phase_A_heldout"][name]["random_feature_mse"]
            for row in per_seed)
        novel_checks[name]={
            "learned_over_kernel_relative_mse_reduction":rel_kernel,
            "learned_over_random_relative_mse_reduction":rel_random,
            "learned_over_kernel_absolute_mse_reduction":absolute,
            "all_three_seeds_better_than_both_controls":each_seed_better,
            "frozen_thresholds_met":(
                rel_kernel>=t[
                    "novel_sine_zero_shot_vs_kernel_gain_relative_min"
                    if name.startswith("novel_sine") else
                    "novel_rational_zero_shot_vs_kernel_gain_relative_min"]
                and absolute>=t[
                    "novel_sine_zero_shot_vs_kernel_abs_MSE_gain_min"
                    if name.startswith("novel_sine") else
                    "novel_rational_zero_shot_vs_kernel_abs_MSE_gain_min"]
                and rel_random>=t[
                    "novel_sine_vs_random_gain_min"
                    if name.startswith("novel_sine") else
                    "novel_rational_vs_random_gain_min"]
                and each_seed_better)
        }
    zero_shot_positive=(oracle_valid and source_controls
                        and all(x["frozen_thresholds_met"]
                                for x in novel_checks.values()))
    retention_each=[
        row["phase_B_original_mse_retention_ratio"]<=
            t["phase_B_source_retention_MSE_ratio_max"] and
        row["phase_B_sine_mse_improvement_fraction"]>=
            t["phase_B_sine_improvement_relative_min"]
        for row in per_seed]
    sequential_positive=all(retention_each)
    if not oracle_valid or not source_controls:
        status="ASSAY_INVALID_SOURCE_CONTROL_OR_ORACLE"
    elif zero_shot_positive:
        status="PRELIMINARY_CROSS_FAMILY_TOKENLESS_TRANSFER_REQUIRES_INDEPENDENT_QUALIFICATION"
    else:
        status="CROSS_FAMILY_RAW_FEATURE_TRANSFER_NULL_OR_NEGATIVE"
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
    ensure(rss<=t["peak_rss_limit_bytes"],"CPU_MEMORY_LIMIT_EXCEEDED")
    ensure(not torch.cuda.is_initialized(),"GPU_USED_IN_CPU_ONLY_RUN")
    return {
        "schema":"ttt-hybrid.g7-s19.novel-dynamics-evidence.v1",
        "candidate_sha":a.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(pbytes).hexdigest(),
        "r3_pinned_package_manifest_sha256":manifest,
        "tokenizer_used":False,
        "vocabulary_used":False,
        "attention_used":False,
        "true_nonlinear_feature_formula_exposed_to_models":False,
        "numeric_time_is_sampled":True,
        "euler_transition_residual_prior_used":True,
        "original_s18_model_implementations_unchanged":True,
        "original_s18_source_blob":
            p["predecessor"]["s18_original_source_git_blob"],
        "calibrated_RBF_selected_only_on_source_train_worlds":True,
        "rbf_calibration_choice":{"gamma":kcfg[0],"decay":kcfg[1],
                                  "ridge":kcfg[2]},
        "rbf_calibration_receipts":calib,
        "per_seed":per_seed,
        "aggregate_phase_A":agg,
        "aggregate_phase_B":after_agg,
        "original_source_oracle_controls_pass":oracle_valid,
        "original_source_conventional_controls_pass":source_controls,
        "novel_zero_shot_frozen_checks":novel_checks,
        "novel_zero_shot_frozen_thresholds_met":zero_shot_positive,
        "sequential_retention_each_seed":retention_each,
        "sequential_retention_and_new_learning_frozen_thresholds_met":
            sequential_positive,
        "trained_model_steps_phase_A_per_seed":240,
        "trained_model_steps_phase_B_per_seed":96,
        "training_worlds_per_seed_per_model":(240+96)*48,
        "heldout_worlds_per_split_per_seed":64,
        "all_trained_cold_restores_exact":True,
        "peak_cpu_rss_bytes":rss,
        "whole_experiment_wall_seconds":round(time.monotonic()-started,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    a=parser.parse_args()
    destination=ROOT/"local_runs/g7_s19_novel_dynamics_evidence.json"
    protocol=ROOT/"experiments/g7_s19_frozen_novel_dynamics.json"
    result={
        "schema":"ttt-hybrid.g7-s19.novel-dynamics-evidence.v1",
        "candidate_sha":a.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    exit_status=2
    try:
        ensure(a.receipt.resolve()==destination.resolve()
               and a.protocol.resolve()==protocol.resolve(),
               "SOURCE_CUSTODY_PROTOCOL_OR_OUTPUT_PATH")
        result=run(a)
        exit_status=0 if result["status"]!="ASSAY_INVALID_SOURCE_CONTROL_OR_ORACLE" else 2
    except Exception as e:
        result["status"]="ENGINEERING_BLOCKED_"+type(e).__name__
        result["failure_class"]=type(e).__name__
    finally:
        if a.receipt.resolve()==destination.resolve():
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_text(
                json.dumps(result,sort_keys=True,indent=2)+"\n",
                encoding="utf-8")
        print("G7_S19_STATUS="+result["status"],flush=True)
        print("G7_S19_TOKENIZER=NOT_USED",flush=True)
        print("G7_S19_GPU=NOT_USED",flush=True)
    return exit_status

if __name__=="__main__":
    raise SystemExit(main())
