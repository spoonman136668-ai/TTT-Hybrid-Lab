"""G7-S8: three real trained arms; same training episodes and fresh evaluation.

Reuses the *exact* frozen G7-S6 conventional generator/model source (same
Git blob) and the S7 native official Mamba2+TTT source-custody loader. This
is equal-data/equal-steps, not equal-parameter or equal-FLOP science.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import statistics
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/"scripts")]
import g7_r5_oneshot as custody
import g7_s6_trained_conventional_baseline as conventional_source
import g7_s7_native_extended_binding as native_source

class Blocked(Exception):
    pass

PHASE="NOT_STARTED"

def require(ok,label):
    if not ok:
        raise Blocked(label)

def phase(value):
    global PHASE
    PHASE=value

def native_tokens(torch,k,v,q):
    n=k.shape[0]
    x=torch.zeros(n,16,64,dtype=torch.float32)
    for idx in range(4):
        x[:,2*idx,:16]=k[:,idx]
        x[:,2*idx+1,16:32]=v[:,idx]
        x[:,2*idx,32]=1
        x[:,2*idx+1,33]=1
    x[:,8:,:16]=q
    x[:,8:,34]=1
    require(bool((x[:,8:,16:32]==0).all()) and
            bool((x[:,8:,32:34]==0).all()),
            "QUERY_VALUE_LEAKAGE")
    return x

def thresholds_valid(spec):
    require(spec.get("schema")=="ttt-hybrid.g7-s8.same-episodes-three-arm.v1",
            "FROZEN_PROTOCOL_SCHEMA_DRIFT")
    data=spec["data"]
    b=spec["budget"]
    require(data["seeds"]==[101,211,307] and
            data["fit_steps"]==512 and
            data["batch_size"]==64 and
            data["heldout_batches_per_seed"]==8 and
            data["heldout_batch_size"]==128 and
            data["ordered_query_keys"]==[0,1,2,3,3,2,1,0] and
            data["fit_episodes_per_arm_per_seed"]==32768 and
            b["peak_torch_allocated_bytes_max"]==1073741824 and
            b["wall_seconds_max"]==900 and
            spec["predecessor"]["s6_generator_model_blob"]==
            "72ba66eebcebd3856d5251ccc765ea52ddf84500" and
            [a["id"] for a in spec["arms"]]==
            ["conventional","online_ttt","static_matched"],
            "PRE_REGISTERED_TRAINING_OR_ARMS_DRIFT")
    t=spec["thresholds"]
    require(t["online_heldout_mse_max"]==0.10 and
            t["conventional_heldout_mse_max"]==0.1 and
            t["online_vs_static_relative_reduction_min"]==0.05,
            "PRE_REGISTERED_THRESHOLDS_DRIFT")

def baseline_cold(torch,model,seed,order):
    phase("CONVENTIONAL_COLD_RESTORE")
    with tempfile.TemporaryDirectory(prefix="ttt-g7-s8-baseline-cold-") as d:
        checkpoint=Path(d)/"model-state.pt"
        torch.save(model.state_dict(),checkpoint)
        digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        restored=conventional_source.new_model(torch)
        restored.load_state_dict(torch.load(
            checkpoint,map_location="cpu",weights_only=True))
        k,v,q,_=conventional_source.get_batch(
            torch,seed*10000000+9600000,16,order)
        model.eval(); restored.eval()
        with torch.no_grad():
            delta=float((model(k,v,q)-restored(k,v,q)).abs().max().item())
        return digest,delta

def run(args):
    phase("FROZEN_PROTOCOL")
    raw=args.protocol.read_bytes()
    spec=json.loads(raw)
    thresholds_valid(spec)
    import torch
    from torch.nn import functional as F
    order=spec["data"]["ordered_query_keys"]
    require(os.geteuid()==1000,"NORMAL_UID_REQUIRED")
    phase("PIN_SOURCE_AND_DRIVER")
    official,manifest=native_source.check_environment(
        torch,{"budget":{"required_free_cuda_bytes_min":
                             spec["budget"]["min_cuda_free_bytes"]}},
        args)
    reference_blob=custody.git(
        ROOT,"hash-object","scripts/g7_s6_trained_conventional_baseline.py")
    require(reference_blob==
            spec["predecessor"]["s6_generator_model_blob"],
            "QUALIFIED_S6_GENERATOR_BLOB_DRIFT")
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    device=torch.device("cuda:0")
    t0=time.monotonic()
    arm_results=[]
    max_vram=0
    for seed in spec["data"]["seeds"]:
        models={}
        fingerprints={}
        arm_reports={}
        for arm in spec["arms"]:
            phase("INITIALIZE_"+arm["id"])
            is_native=arm["id"]!="conventional"
            if is_native:
                model,fingerprint=native_source.new_hybrid(
                    torch,official,arm["ttt_inner_update"],seed)
                fingerprints[arm["id"]]=fingerprint
                model=model.to(device=device,dtype=torch.float32)
            else:
                torch.manual_seed(seed)
                model=conventional_source.new_model(torch)
            parameter_count=sum(p.numel() for p in model.parameters())
            require(parameter_count==arm["params"],
                    "PARAMETER_COUNT_DRIFT_"+arm["id"])
            trainable=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(
                trainable,lr=arm["train_lr"],weight_decay=0.0)
            model.train()
            train_start=time.monotonic()
            losses=[]
            if is_native:
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(spec["data"]["fit_steps"]):
                phase("TRAIN_FORWARD_"+arm["id"])
                k,v,q,y=conventional_source.get_batch(
                    torch,seed*10000000+step*17,
                    spec["data"]["batch_size"],order)
                if is_native:
                    x=native_tokens(torch,k,v,q).to(device)
                    y=y.to(device)
                    prediction=model(x)[:,8:,:16]
                else:
                    prediction=model(k,v,q)
                require(tuple(prediction.shape)==tuple(y.shape),
                        "TRAIN_PREDICTION_SHAPE")
                loss=F.mse_loss(prediction,y)
                require(bool(torch.isfinite(loss)),"NONFINITE_LOSS")
                optimizer.zero_grad(set_to_none=True)
                phase("TRAIN_BACKWARD_"+arm["id"])
                loss.backward()
                gn=torch.nn.utils.clip_grad_norm_(trainable,1.0)
                require(bool(torch.isfinite(gn)),"NONFINITE_GRADIENT")
                phase("OPTIMIZER_STEP_"+arm["id"])
                optimizer.step()
                losses.append(float(loss.detach().item()))
                if is_native:
                    torch.cuda.synchronize(device)
                    peak=torch.cuda.max_memory_allocated(device)
                    require(peak<=
                            spec["budget"]["peak_torch_allocated_bytes_max"],
                            "NATIVE_VRAM_BUDGET")
                require(time.monotonic()-t0<
                        spec["budget"]["wall_seconds_max"],
                        "FROZEN_STUDY_WALLTIME_BUDGET")
            train_seconds=time.monotonic()-train_start
            phase("FRESH_HELDOUT_"+arm["id"])
            model.eval()
            measures={name:[] for name in (
                "correct_mse","shuffled_support_mse","erased_support_mse",
                "zero_predictor_mse","oracle_correct_mse",
                "oracle_shuffled_mse","prediction_movement_mse"
            )}
            eval_start=time.monotonic()
            with torch.no_grad():
                for batch_idx in range(spec["data"]["heldout_batches_per_seed"]):
                    k,v,q,y=conventional_source.get_batch(
                        torch,seed*10000000+9800000+batch_idx*17,
                        spec["data"]["heldout_batch_size"],order)
                    matches=(q[:,:,None,:]==k[:,None,:,:]).all(dim=-1)
                    require(bool((matches.sum(dim=-1)==1).all()),
                            "HELDOUT_UNIQUE_KEY_ORACLE_FAIL")
                    oracle=torch.einsum("bqi,biv->bqv",matches.float(),v)
                    altered=torch.roll(v,1,dims=1)
                    oracle_bad=torch.einsum(
                        "bqi,biv->bqv",matches.float(),altered)
                    require(bool(torch.equal(oracle,y)),
                            "HELDOUT_ORACLE_TARGET_IDENTITY")
                    if is_native:
                        predictions=[]
                        for support in (v,altered,torch.zeros_like(v)):
                            chunks=[]
                            for a in range(0,k.shape[0],16):
                                x=native_tokens(
                                    torch,k[a:a+16],
                                    support[a:a+16],q[a:a+16]).to(device)
                                parts=model(x)[:,8:,:16]
                                chunks.append(parts.cpu())
                            predictions.append(torch.cat(chunks,dim=0))
                        pred,bad,erased=predictions
                    else:
                        pred=model(k,v,q)
                        bad=model(k,altered,q)
                        erased=model(k,torch.zeros_like(v),q)
                    require(all(bool(torch.isfinite(out).all())
                                for out in (pred,bad,erased)),
                            "NONFINITE_HELDOUT")
                    vals={
                        "correct_mse":F.mse_loss(pred,y),
                        "shuffled_support_mse":F.mse_loss(bad,y),
                        "erased_support_mse":F.mse_loss(erased,y),
                        "zero_predictor_mse":y.square().mean(),
                        "oracle_correct_mse":F.mse_loss(oracle,y),
                        "oracle_shuffled_mse":F.mse_loss(oracle_bad,y),
                        "prediction_movement_mse":F.mse_loss(pred,bad),
                    }
                    for k2,value in vals.items():
                        measures[k2].append(float(value.item()))
            if is_native:
                torch.cuda.synchronize(device)
                peak=int(torch.cuda.max_memory_allocated(device))
                max_vram=max(max_vram,peak)
                require(peak<=spec["budget"]["peak_torch_allocated_bytes_max"],
                        "HELDOUT_VRAM_BUDGET")
                phase("COLD_RESTORE_"+arm["id"])
                ck_sha,cold_delta=native_source.cold_restore(
                    torch,model,official,arm["ttt_inner_update"],seed,device)
            else:
                ck_sha,cold_delta=baseline_cold(torch,model,seed,order)
                peak=0
            eval_seconds=time.monotonic()-eval_start
            require(cold_delta==0.0,"MODEL_COLD_RESTORE_FAILED")
            arm_reports[arm["id"]]={
                "parameters":parameter_count,
                "trainable_parameters":sum(p.numel() for p in trainable),
                "optimizer_steps":len(losses),
                "training_episodes":len(losses)*spec["data"]["batch_size"],
                "first_train_mse":losses[0],
                "last32_train_mse":statistics.mean(losses[-32:]),
                "heldout_batch_metrics":measures,
                "heldout_means":{key:statistics.mean(v)
                                 for key,v in measures.items()},
                "training_elapsed_seconds":round(train_seconds,3),
                "evaluation_elapsed_seconds":round(eval_seconds,3),
                "torch_peak_allocated_bytes":peak,
                "cold_checkpoint_sha256":ck_sha,
                "cold_restored_output_max_abs_delta":cold_delta,
            }
            del optimizer,model
            if is_native:
                torch.cuda.empty_cache()
            require(time.monotonic()-t0<
                    spec["budget"]["wall_seconds_max"],
                    "FINAL_STUDY_WALLTIME_EXCEEDED")
        require(fingerprints["online_ttt"]==
                fingerprints["static_matched"],
                "NATIVE_ARM_INITIAL_WEIGHT_HASH_DRIFT")
        require(all(arm_reports[key]["optimizer_steps"]==512 and
                    arm_reports[key]["training_episodes"]==32768
                    for key in arm_reports),"ARM_EXPOSURE_MISMATCH")
        arm_results.append({
            "seed":seed,
            "native_matching_initial_weight_sha256":
                fingerprints["online_ttt"],
            "arms":arm_reports,
        })
    phase("AGGREGATE_ALL_FROZEN_ARMS")
    metrics={}
    for arm in ("conventional","online_ttt","static_matched"):
        metrics[arm]={name:statistics.mean(
            row["arms"][arm]["heldout_means"][name]
            for row in arm_results)
            for name in arm_results[0]["arms"][arm]["heldout_means"]}
    thresholds=spec["thresholds"]
    valid=all(
        m["oracle_correct_mse"]<=thresholds["oracle_exact_mse_max"]
        and m["oracle_shuffled_mse"]>=
            thresholds["corrupted_oracle_mse_min"]
        and abs(m["zero_predictor_mse"]-1.0)<1e-12
        for m in metrics.values()
    )
    c,o,s=(metrics[n] for n in
            ("conventional","online_ttt","static_matched"))
    conventional_pass=(
        c["correct_mse"]<=thresholds["conventional_heldout_mse_max"]
        and c["shuffled_support_mse"]>=
            thresholds["conventional_corrupted_mse_min"]
        and c["erased_support_mse"]>=
            thresholds["conventional_erased_mse_min"]
        and all(
            row["arms"]["conventional"]["heldout_means"]["correct_mse"]<=
            thresholds["conventional_heldout_mse_max"]
            for row in arm_results)
    )
    require(s["correct_mse"]>0 and o["correct_mse"]>0,
            "INVALID_MODEL_ERROR_DENOMINATOR")
    online_vs_static=(s["correct_mse"]-
                      o["correct_mse"])/s["correct_mse"]
    online_shuffle=(o["shuffled_support_mse"]-
                    o["correct_mse"])/o["correct_mse"]
    online_erased=(o["erased_support_mse"]-
                   o["correct_mse"])/o["correct_mse"]
    every_seed_online_better_static=all(
        row["arms"]["online_ttt"]["heldout_means"]["correct_mse"] <
        row["arms"]["static_matched"]["heldout_means"]["correct_mse"]
        for row in arm_results)
    every_seed_online_better_zero=all(
        row["arms"]["online_ttt"]["heldout_means"]["correct_mse"] <
        thresholds["zero_mse_exact"]
        for row in arm_results)
    native_thresholds_met=(
        valid and conventional_pass and
        o["correct_mse"]<=thresholds["online_heldout_mse_max"]
        and every_seed_online_better_static and
        every_seed_online_better_zero and
        online_vs_static>=
            thresholds["online_vs_static_relative_reduction_min"]
        and online_shuffle>=
            thresholds["online_shuffled_relative_harm_min"]
        and online_erased>=
            thresholds["online_erased_relative_harm_min"]
    )
    status=(
        "ASSAY_INVALID_CONVENTIONAL_CONTROL" if not valid or
        not conventional_pass else
        "PRELIMINARY_ASSOCIATIVE_LEARNING_SIGNAL_REQUIRES_INDEPENDENT_REPLICATION"
        if native_thresholds_met else
        "ONLINE_TTT_INCREMENTAL_ADVANTAGE_NOT_DEMONSTRATED"
    )
    return {
        "schema":"ttt-hybrid.g7-s8.equal-episodes-three-arm-evidence.v1",
        "candidate_sha":args.expected_sha,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "status":status,
        "s6_exact_generator_blob":reference_blob,
        "source_and_runtime_manifest_sha256":manifest,
        "fit_steps_per_seed_all_arms":512,
        "training_episodes_per_seed_all_arms":32768,
        "fresh_eval_episodes_per_seed_all_arms":1024,
        "arms_by_seed":arm_results,
        "aggregate_heldout_metrics":metrics,
        "conventional_positive_control_pass":conventional_pass,
        "oracle_controls_pass":valid,
        "online_vs_static_relative_mse_reduction":online_vs_static,
        "online_shuffle_relative_harm":online_shuffle,
        "online_erased_relative_harm":online_erased,
        "online_better_than_static_all_seeds":
            every_seed_online_better_static,
        "online_better_than_zero_all_seeds":every_seed_online_better_zero,
        "native_frozen_thresholds_met":native_thresholds_met,
        "parameters_matched_between_hybrid_arms":True,
        "parameters_matched_to_conventional":False,
        "compute_cost_matched_to_conventional":False,
        "actual_gpu_peak_torch_allocated_bytes":max_vram,
        "actual_python_process_wall_seconds":round(time.monotonic()-t0,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    receipt=ROOT/"local_runs/g7_s8_equal_episodes_evidence.json"
    report={
        "schema":"ttt-hybrid.g7-s8.equal-episodes-three-arm-evidence.v1",
        "status":"ENGINEERING_UNVERIFIED",
        "candidate_sha":args.expected_sha,
        "phase":"NOT_STARTED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }
    code=2
    try:
        require(args.receipt.resolve()==receipt.resolve() and
                args.protocol.resolve()==
                (ROOT/"experiments/g7_s8_frozen_episode_matched.json").resolve(),
                "FROZEN_PROTOCOL_OR_RECEIPT_DESTINATION_DENIED")
        report=run(args)
        code=0 if report["status"]!="ASSAY_INVALID_CONVENTIONAL_CONTROL" else 2
    except (Blocked,custody.Blocked,native_source.Blocked) as exc:
        report["status"]="ENGINEERING_BLOCKED_"+str(exc)
        report["phase"]=PHASE
    except Exception as exc:
        report["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        report["phase"]=PHASE
        report["error_class"]=type(exc).__name__
    finally:
        if args.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                               encoding="utf-8")
        print("G7_S8_STATUS="+report["status"],flush=True)
        print("G7_S8_PHASE="+report.get("phase","COMPLETE"),flush=True)
        print("G7_S8_INDEPENDENT=NOT_RUN",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
