"""G7-S20 frozen partial-observation and four-phase token-free learning.

A (source 240) -> B (sine 96) -> C (rational 96) -> D (source 96).
Observed physical sensor is ONLY y = x + 0.25 z, with a two-dimensional
latent (x,z) and noisy sensory history. Models never receive x,z, hidden
coefficients, a law selector or hand-supplied nonlinear features.

Reuses exact G7-S18 learned-feature + 32-unit GRU and RBF control source,
verified by git blob. Strong kernel control calibrated only on source
training-range worlds. All heldout worlds are frozen and reused across phases.
Retaining the source after phase C is not confused with recovery after D.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from statistics import mean

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s18_causal_feature_replication as s18
import torch
from torch import nn
from torch.nn import functional as F
from g7_process_peak_memory import read_peak_memory

class Blocked(Exception):
    pass

def require(ok,why):
    if not ok:
        raise Blocked(why)

def coefficients(gen,b,ood):
    rand=lambda:torch.rand((b,1),generator=gen)
    if ood:
        return torch.cat((
            1.55+.70*rand(),1.55+.55*rand(),
            -.80+1.60*rand(),-.65+1.30*rand()),dim=1)
    return torch.cat((
        .35+1.05*rand(),.35+1.10*rand(),
        -.45+.90*rand(),-.35+.70*rand()),dim=1)

def rate(x,u,c,family):
    a,b,g,d=c.unbind(-1)
    a=a[:,None];b=b[:,None];g=g[:,None];d=d[:,None]
    if family=="source":
        return -a*x+b*u+g*torch.tanh(2*x)+d*x*u
    if family=="sine":
        return -a*x+b*u+g*torch.sin(4*x)+d*torch.tanh(3*u)
    if family=="rational":
        return -a*torch.tanh(2*x)+b*u+g*x.pow(3)/(1+x.square())+d*torch.sin(3*u)
    raise Blocked("UNKNOWN_PHYSICS_FAMILY")

def partial_world(seed,batch,regime):
    """Privileged evaluator; returned arrays contain observed y, never x/z."""
    if regime=="train_switch":
        family,ood="source",False
    elif regime=="source_ood":
        family,ood="source",True
    elif regime=="sine_ood":
        family,ood="sine",True
    elif regime=="rational_ood":
        family,ood="rational",True
    elif regime=="sine_train":
        family,ood="sine",False
    elif regime=="rational_train":
        family,ood="rational",False
    elif regime=="source_train":
        family,ood="source",False
    else:
        raise Blocked("UNFROZEN_PARTIAL_OBSERVATION_REGIME")
    gen=torch.Generator(device="cpu").manual_seed(int(seed))
    old=coefficients(gen,batch,ood)
    new=coefficients(gen,batch,ood)
    x=-1+2*torch.rand((batch,1),generator=gen)
    z=-.8+1.6*torch.rand((batch,1),generator=gen)
    observation=x+.25*z+.002*torch.randn((batch,1),generator=gen)
    frames=[]
    for step in range(32):
        u=-1+2*torch.rand((batch,1),generator=gen)
        dt=.05+.11*torch.rand((batch,1),generator=gen)
        hidden=new if step>=20 else old
        dx=rate(x,u,hidden,family)+.35*z
        dz=-.85*z+.25*u+.15*x
        xn=x+dt*dx
        zn=z+dt*dz
        observed_next=xn+.25*zn+.002*torch.randn(
            (batch,1),generator=gen)
        frames.append(torch.cat(
            (observation,u,dt,observed_next),dim=1))
        x=xn
        z=zn
        observation=observed_next
    history=torch.stack(frames,dim=1)
    query_y=observation.expand(-1,8)
    query_u=-1+2*torch.rand((batch,8),generator=gen)
    query_dt=.05+.11*torch.rand((batch,8),generator=gen)
    q=torch.stack((query_y,query_u,query_dt),dim=-1)
    xq=x.expand(-1,8);zq=z.expand(-1,8)
    future_x=xq+query_dt*(rate(xq,query_u,new,family)+.35*zq)
    future_z=zq+query_dt*(-.85*zq+.25*query_u+.15*xq)
    future_sensor=future_x+.25*future_z
    require(history.shape==(batch,32,4)
            and q.shape==(batch,8,3)
            and future_sensor.shape==(batch,8)
            and bool(torch.isfinite(history).all())
            and bool(torch.isfinite(future_sensor).all()),
            "PARTIAL_WORLD_SENSOR_SHAPE_OR_FINITE")
    return history,q,future_sensor

# Only the data-generation interface is substituted. All predictive model
# and calibration implementations are the EXACT G7-S18 source blob.
s18.simulator=partial_world

def frozen(p):
    tr=p["training"];ev=p["evaluation"];cal=p["calibration"];t=p["frozen_thresholds"]
    require(
        p.get("schema")=="ttt-hybrid.g7-s20.hidden-state-multicycle.v1"
        and p["predecessor"]["s19_sha"]==
            "65308a66956b2bca70c48600fd5a879b5b22d388"
        and p["predecessor"]["s19_run"]==42398803072
        and p["predecessor"]["s18_unchanged_model_blob"]==
            "cfc1959b8812e9abad581018462302bc89a31d59"
        and tr["seeds"]==[1091,1217,1301]
        and tr["batch"]==48
        and [tr[k]["steps"] for k in
             ("phase_A","phase_B","phase_C","phase_D")]==[240,96,96,96]
        and tr["total_updates_per_trained_arm_per_seed"]==528
        and tr["total_worlds_per_trained_arm_per_seed"]==25344
        and cal["seeds"]==[2401,2411,2423]
        and cal["kernel_gamma"]==[.5,2.,6.]
        and cal["kernel_decay"]==[.65,.78,1.]
        and cal["kernel_ridge"]==[.01,.08]
        and ev["heldout_seeds"]=={
            "source":[2503,2521,2531],
            "sine":[2609,2617,2621],
            "rational":[2707,2713,2729]}
        and ev["evaluated_after_phases"]==["A","B","C","D"]
        and t["zero_shot_sine_kernel_relative_mse_gain_min"]==.05
        and t["zero_shot_rational_kernel_relative_mse_gain_min"]==.05
        and t["source_retention_after_C_over_after_A_ratio_max"]==1.5
        and t["source_recovery_after_D_over_after_A_ratio_max"]==1.25
        and t["wall_seconds_max"]==240
        and p["environment"]["no_tokenizer"] is True
        and p["environment"]["no_attention"] is True
        and p["authority"]["cpu_only"] is True,
        "FROZEN_S20_PROTOCOL_SOURCE_OR_LIMITS_DRIFT")

def calibration(p,start):
    pc=p["calibration"]
    q={
        "calibration":{
            "rbf_gamma_grid":pc["kernel_gamma"],
            "rbf_decay_grid":pc["kernel_decay"],
            "rbf_regularization_grid":pc["kernel_ridge"],
            "seeds":pc["seeds"],"batch":pc["batch"]},
        "threshold":{"wall_limit_seconds":p["frozen_thresholds"]["wall_seconds_max"]}
    }
    return s18.calibrate_kernel(q,start)

def fit(feature,gru,optim_f,optim_g,seed,p,phase,start):
    spec=p["training"]["phase_"+phase]
    count=spec["steps"]
    label=spec["family"]
    train_seconds={"feature":0.0,"gru":0.0}
    losses={"feature":[],"gru":[]}
    t0=time.monotonic()
    for step in range(count):
        sample_seed=seed*100000+{
            "A":701000,"B":801000,"C":901000,"D":1001000}[phase]+step*37
        h,q,y=partial_world(
            sample_seed,p["training"]["batch"],
            label+"_train" if phase!="A" else "train_switch")
        for model,optim,name in (
            (feature,optim_f,"feature"),
            (gru,optim_g,"gru")
        ):
            tick=time.monotonic()
            model.train()
            output=(s18.learned_ridge(model,h,q)
                    if isinstance(model,s18.RawFeatureBank) else model(h,q))
            loss=F.mse_loss(output,y)
            require(bool(torch.isfinite(loss)),
                    "NONFINITE_"+phase+"_TRAIN_LOSS")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            norm=nn.utils.clip_grad_norm_(model.parameters(),1.0)
            require(bool(torch.isfinite(norm)),
                    "NONFINITE_"+phase+"_TRAIN_GRAD")
            optim.step()
            train_seconds[name]+=time.monotonic()-tick
            losses[name].append(float(loss.detach()))
        require(time.monotonic()-start<
                p["frozen_thresholds"]["wall_seconds_max"],
                "WHOLE_S20_WALL_CAP")
    return {
        "phase":phase,"family":label,"steps":count,
        "worlds_per_trained_arm":count*p["training"]["batch"],
        "per_model_training_seconds":{
            name:round(v,4) for name,v in train_seconds.items()},
        "phase_total_seconds":round(time.monotonic()-t0,3),
        "first_loss":{name:vals[0] for name,vals in losses.items()},
        "last32_loss":{name:mean(vals[-32:]) for name,vals in losses.items()}
    }

def observe(feature,random,gru,kcfg,p,index):
    results={}
    for family,seed_list in p["evaluation"]["heldout_seeds"].items():
        seed=seed_list[index]
        results[family]=s18.measure(
            feature,random,gru,kcfg,seed,family+"_ood")
    return results

def cold(feature,gru,seed):
    h,q,_=partial_world(seed*100000+119911,4,"source_ood")
    a,da=s18.checkpoint_cold(feature,h,q)
    b,db=s18.checkpoint_cold(gru,h,q)
    require(da==0.0 and db==0.0,"COLD_WEIGHT_RESTORE_DRIFT")
    return {"learned_features_sha256":a,"gru_sha256":b,
            "feature_max_abs_delta":da,"gru_max_abs_delta":db}

def verify_stable_controls(phases):
    """No baseline parameters or calibration can change across phases."""
    for family in ("source","sine","rational"):
        reference=phases["A"]["heldout"][family]
        for stage in ("B","C","D"):
            other=phases[stage]["heldout"][family]
            for key in ("random_feature_mse","rbf_kernel_mse",
                        "static_no_dynamics_mse","oracle_mse",
                        "post_switch_ablated_kernel_mse"):
                require(reference[key]==other[key],
                        "UNTRAINED_OR_CALIBRATED_CONTROL_DRIFT_"+key)

def run(args):
    # The frozen S20 assay has a Linux ru_maxrss/2 GiB criterion. Windows
    # working-set bytes cannot be substituted or claimed equivalent.
    # S25 may still import the original physics functions on Windows.
    require(sys.platform == "linux", "S20_FROZEN_RSS_REQUIRES_LINUX")
    pbytes=args.protocol.read_bytes()
    p=json.loads(pbytes)
    frozen(p)
    require(custody.git(ROOT,"hash-object",
                        "scripts/g7_s18_causal_feature_replication.py")==
            p["predecessor"]["s18_unchanged_model_blob"],
            "S18_ORIGINAL_MODEL_SOURCE_DRIFT")
    manifest,packages=custody.manifest()
    require(manifest==custody.R3_MANIFEST_SHA256
            and packages.get("torch")==custody.TORCH
            and packages.get("triton")==custody.TRITON,
            "QUALIFIED_PINNED_VENV_MANIFEST_CHANGED")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    require(not torch.cuda.is_initialized(),"CPU_ONLY_CUDA_FORBIDDEN")
    started=time.monotonic()
    kcfg,calibration_receipts=calibration(p,started)
    all_seeds=[]
    for index,seed in enumerate(p["training"]["seeds"]):
        torch.manual_seed(seed)
        feature=s18.RawFeatureBank()
        random=copy.deepcopy(feature)
        gru=s18.RawGRU()
        optf=torch.optim.AdamW(feature.parameters(),
            lr=p["training"]["feature_lr"],
            weight_decay=p["training"]["weight_decay"])
        optg=torch.optim.AdamW(gru.parameters(),
            lr=p["training"]["gru_lr"],
            weight_decay=p["training"]["weight_decay"])
        phases={}
        for stage in p["evaluation"]["evaluated_after_phases"]:
            train=fit(feature,gru,optf,optg,seed,p,stage,started)
            data=observe(feature,random,gru,kcfg,p,index)
            restored=cold(feature,gru,seed)
            phases[stage]={"training":train,
                           "heldout":data,
                           "cold_restore":restored}
        verify_stable_controls(phases)
        a=phases["A"]["heldout"]
        b=phases["B"]["heldout"]
        c=phases["C"]["heldout"]
        d=phases["D"]["heldout"]
        def ratio(numerator,denominator):
            return numerator/max(denominator,1e-12)
        all_seeds.append({
            "seed":seed,
            "trainable_feature_parameters":sum(x.numel()
                                              for x in feature.parameters()),
            "trainable_gru_parameters":sum(x.numel()
                                          for x in gru.parameters()),
            "shared_initial_random_features":True,
            "phases":phases,
            "source_after_C_over_after_A_mse":
                ratio(c["source"]["learned_feature_mse"],
                      a["source"]["learned_feature_mse"]),
            "source_after_D_over_after_A_mse":
                ratio(d["source"]["learned_feature_mse"],
                      a["source"]["learned_feature_mse"]),
            "sine_B_relative_mse_gain":
                1-ratio(b["sine"]["learned_feature_mse"],
                        a["sine"]["learned_feature_mse"]),
            "rational_C_relative_mse_gain":
                1-ratio(c["rational"]["learned_feature_mse"],
                        b["rational"]["learned_feature_mse"])
        })
        require(time.monotonic()-started<
                p["frozen_thresholds"]["wall_seconds_max"],
                "FROZEN_TOTAL_S20_TIME_CAP")
    families=("source","sine","rational")
    stages=("A","B","C","D")
    agg={stage:{
        family:{key:mean(row["phases"][stage]["heldout"][family][key]
                         for row in all_seeds)
                for key in all_seeds[0]["phases"][stage]["heldout"][family]}
        for family in families} for stage in stages}
    t=p["frozen_thresholds"]
    oracle_ok=all(
        row["phases"][stage]["heldout"][family]["oracle_mse"]<=
            t["exact_oracle_mse_max"]
        for row in all_seeds for stage in stages for family in families)
    source_positive=all(
        row["phases"]["A"]["heldout"]["source"]["rbf_kernel_mse"]<=
            t["calibrated_kernel_source_ood_mse_max"]
        and (row["phases"]["A"]["heldout"]["source"][
                 "static_no_dynamics_mse"]-
             row["phases"]["A"]["heldout"]["source"]["rbf_kernel_mse"])/
            max(row["phases"]["A"]["heldout"]["source"][
                "static_no_dynamics_mse"],1e-12)>=
            t["calibrated_kernel_source_gain_vs_static_min"]
        and row["phases"]["A"]["heldout"]["source"][
                "post_switch_ablated_kernel_mse"]/
            max(row["phases"]["A"]["heldout"]["source"][
                "rbf_kernel_mse"],1e-12)>=
            t["calibrated_kernel_source_post_switch_ablation_ratio_min"]
        for row in all_seeds)
    source_learned_causal=all(
        row["phases"]["A"]["heldout"]["source"][
            "post_switch_ablated_learned_mse"]/
        max(row["phases"]["A"]["heldout"]["source"][
            "learned_feature_mse"],1e-12)>=
        t["source_learned_post_switch_ablation_ratio_min"]
        for row in all_seeds)
    zero_shot={}
    for family in ("sine","rational"):
        m=agg["A"][family]
        kern=m["rbf_kernel_mse"]
        learned=m["learned_feature_mse"]
        random=m["random_feature_mse"]
        relative_kernel=(kern-learned)/max(kern,1e-12)
        relative_random=(random-learned)/max(random,1e-12)
        per_seed_better=[
            row["phases"]["A"]["heldout"][family]["learned_feature_mse"]<
            row["phases"]["A"]["heldout"][family]["rbf_kernel_mse"]
            and row["phases"]["A"]["heldout"][family]["learned_feature_mse"]<
            row["phases"]["A"]["heldout"][family]["random_feature_mse"]
            for row in all_seeds]
        ok=(relative_kernel>=t[
                 "zero_shot_"+family+"_kernel_relative_mse_gain_min"]
            and relative_random>=t[
                 "zero_shot_"+family+"_random_relative_mse_gain_min"]
            and all(per_seed_better))
        zero_shot[family]={
            "learned_over_kernel_relative_gain":relative_kernel,
            "learned_over_random_relative_gain":relative_random,
            "all_seeds_better_than_both_controls":per_seed_better,
            "frozen_thresholds_met":ok}
    zero_shot_pass=(oracle_ok and source_positive
                    and source_learned_causal
                    and all(m["frozen_thresholds_met"]
                            for m in zero_shot.values()))
    memory_each=[
        row["source_after_C_over_after_A_mse"]<=
            t["source_retention_after_C_over_after_A_ratio_max"]
        and row["source_after_D_over_after_A_mse"]<=
            t["source_recovery_after_D_over_after_A_ratio_max"]
        and row["sine_B_relative_mse_gain"]>=
            t["sine_learning_after_B_relative_error_reduction_min"]
        and row["rational_C_relative_mse_gain"]>=
            t["rational_learning_after_C_relative_error_reduction_min"]
        for row in all_seeds]
    memory_pass=all(memory_each)
    status=("ASSAY_INVALID_PARTIAL_OBSERVABILITY_OR_CONVENTIONAL_CONTROL"
            if not oracle_ok or not source_positive else
            "PRELIMINARY_PARTIAL_OBSERVATION_MULTICYCLE_TRANSFER_NOT_SCIENTIFIC_PASS"
            if zero_shot_pass and memory_pass else
            "PARTIAL_OBSERVATION_OR_MULTICYCLE_LEARNING_FROZEN_NULL")
    peak=read_peak_memory()
    require(peak.metric == "linux_ru_maxrss_bytes"
            and peak.comparable_to_linux_ru_maxrss,
            "S20_FROZEN_RSS_METRIC_NOT_EQUIVALENT")
    rss=peak.peak_bytes
    require(rss<=t["max_cpu_rss_bytes"],"FROZEN_RSS_CAP_EXCEEDED")
    require(not torch.cuda.is_initialized(),
            "UNAUTHORIZED_CUDA_USED")
    return {
        "schema":"ttt-hybrid.g7-s20.hidden-state-multicycle-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(pbytes).hexdigest(),
        "r3_manifest_sha256":manifest,
        "observed_sensor_channels":1,
        "hidden_physical_dimensions":2,
        "hidden_state_or_latent_coefficients_supplied_to_predictor":False,
        "known_nonlinear_analytic_basis_supplied_to_predictor":False,
        "tokenizer_used":False,"attention_used":False,"vocabulary_used":False,
        "numeric_time_sampled":True,
        "per_seed":all_seeds,
        "aggregate_heldout_by_stage":agg,
        "calibration_only_krr_hyperparameters":{
            "gamma":kcfg[0],"decay":kcfg[1],"ridge":kcfg[2]},
        "calibration_receipts":calibration_receipts,
        "oracle_controls_pass":oracle_ok,
        "source_conventional_controls_pass":source_positive,
        "source_learned_causal_ablation_pass":source_learned_causal,
        "novel_family_zero_shot_controls":zero_shot,
        "zero_shot_frozen_thresholds_met":zero_shot_pass,
        "multicycle_each_seed":memory_each,
        "multicycle_frozen_thresholds_met":memory_pass,
        "same_heldout_worlds_every_phase":True,
        "untrained_kernel_and_random_controls_unchanged":True,
        "all_model_cold_restores_exact":True,
        "steps_per_model_per_seed":528,
        "worlds_per_model_per_seed":25344,
        "peak_cpu_rss_bytes":rss,
        "whole_process_wall_seconds":round(time.monotonic()-started,3),
        "training_compute_matched_to_krr":False,
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--protocol",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    a=parser.parse_args()
    expected_protocol=ROOT/"experiments/g7_s20_frozen_hidden_multicycle.json"
    receipt=ROOT/"local_runs/g7_s20_hidden_multicycle_evidence.json"
    result={"schema":"ttt-hybrid.g7-s20.hidden-state-multicycle-evidence.v1",
            "candidate_sha":a.expected_sha,
            "status":"ENGINEERING_UNVERIFIED",
            "independent_scientific_qualification":"NOT_RUN",
            "deployment":"NOT_AUTHORIZED"}
    rc=2
    try:
        require(a.protocol.resolve()==expected_protocol.resolve() and
                a.receipt.resolve()==receipt.resolve(),
                "FROZEN_SOURCE_PATH_DRIFT")
        result=run(a)
        rc=(2 if result["status"]==
           "ASSAY_INVALID_PARTIAL_OBSERVABILITY_OR_CONVENTIONAL_CONTROL"
           else 0)
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["blocked_type"]=type(exc).__name__
    finally:
        if a.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",
                               encoding="utf-8")
        print("G7_S20_STATUS="+result["status"],flush=True)
        print("G7_S20_TOKENIZER=NOT_USED",flush=True)
        print("G7_S20_CUDA=NOT_USED",flush=True)
    return rc

if __name__=="__main__":
    raise SystemExit(main())
