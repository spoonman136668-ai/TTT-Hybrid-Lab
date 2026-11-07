"""G7-S9: OOD random query/support permutations after fixed-layout training.

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


def permuted_support_queries(torch,k,v,q,y,episode_seed):
    """Independent deterministic nonidentity permutations for every episode."""
    count=k.shape[0]
    qperms=[]
    sperms=[]
    for index in range(count):
        rng=torch.Generator(device="cpu").manual_seed(
            int(episode_seed + 100003 + index*73))
        qidx=torch.randperm(8,generator=rng)
        sidx=torch.randperm(4,generator=rng)
        if torch.equal(qidx,torch.arange(8)):
            qidx=qidx.roll(1)
        if torch.equal(sidx,torch.arange(4)):
            sidx=sidx.roll(1)
        qperms.append(qidx)
        sperms.append(sidx)
    qp=torch.stack(qperms)
    sp=torch.stack(sperms)
    qswap=q.gather(1,qp.unsqueeze(-1).expand(-1,-1,16))
    yswap=y.gather(1,qp.unsqueeze(-1).expand(-1,-1,16))
    kswap=k.gather(1,sp.unsqueeze(-1).expand(-1,-1,16))
    vswap=v.gather(1,sp.unsqueeze(-1).expand(-1,-1,16))
    # No per-example target changes from support *pair* permutations.
    return qswap,yswap,kswap,vswap

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
    transfer_bytes=args.transfer_protocol.read_bytes()
    transfer=json.loads(transfer_bytes)
    require(transfer.get("schema")=="ttt-hybrid.g7-s9.permutation-transfer.v1"
            and transfer["predecessor"]["s8_run"]==42370667214
            and transfer["predecessor"]["s8_generator_blob"]==
            "72ba66eebcebd3856d5251ccc765ea52ddf84500"
            and transfer["training"]["optimizer_steps_per_arm"]==512
            and transfer["heldout"]["heldout_batches_per_seed"]==8
            and transfer["heldout"]["examples_per_batch"]==128
            and transfer["heldout"]["novel_split_after_s8"] is True
            and transfer["controls"]["native_joint_over_fixed_mse_ratio_max"]==2.0,
            "FROZEN_S9_TRANSFER_PROTOCOL_DRIFT")
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
            phase("FRESH_S9_PERMUTATION_HELDOUT_"+arm["id"])
            model.eval()
            variants=transfer["heldout"]["variants"]
            measures={name:[] for name in variants}
            measures["oracle_fixed_mse"]=[]
            measures["oracle_joint_mse"]=[]
            measures["oracle_corrupted_mse"]=[]
            measures["zero_predictor_mse"]=[]
            measures["joint_prediction_corruption_mse"]=[]
            eval_start=time.monotonic()
            with torch.no_grad():
                for batch_idx in range(transfer["heldout"]["heldout_batches_per_seed"]):
                    frozen_seed=seed*10000000+9900000+batch_idx*17
                    k,v,q,y=conventional_source.get_batch(
                        torch,frozen_seed,
                        transfer["heldout"]["examples_per_batch"],order)
                    qp,yp,kp,vp=permuted_support_queries(
                        torch,k,v,q,y,frozen_seed)
                    dataset={
                        "fixed_layout":(k,v,q,y),
                        "query_order_permuted":(k,v,qp,yp),
                        "support_pair_positions_permuted":(kp,vp,q,y),
                        "joint_support_and_query_permuted":(kp,vp,qp,yp),
                        "joint_permuted_with_corrupted_support_values":
                            (kp,torch.roll(vp,1,dims=1),qp,yp),
                    }
                    pred_cache={}
                    for tag,(keys,values,queries,labels) in dataset.items():
                        matches=(queries[:,:,None,:]==keys[:,None,:,:]).all(dim=-1)
                        require(bool((matches.sum(dim=-1)==1).all()),
                                "AMBIGUOUS_S9_CONTENT_LOOKUP")
                        oracle=torch.einsum(
                            "bqi,biv->bqv",matches.float(),values)
                        if tag!="joint_permuted_with_corrupted_support_values":
                            require(bool(torch.equal(oracle,labels)),
                                    "S9_POSITIVE_ORACLE_INCORRECT")
                        if is_native:
                            pieces=[]
                            for index in range(0,keys.shape[0],16):
                                x=native_tokens(
                                    torch,keys[index:index+16],
                                    values[index:index+16],
                                    queries[index:index+16]).to(device)
                                pieces.append(model(x)[:,8:,:16].cpu())
                            prediction=torch.cat(pieces,dim=0)
                        else:
                            prediction=model(keys,values,queries)
                        require(bool(torch.isfinite(prediction).all()),
                                "NONFINITE_S9_TRANSFER_OUTPUT")
                        pred_cache[tag]=prediction
                        measures[tag].append(float(
                            F.mse_loss(prediction,labels).item()))
                        if tag=="fixed_layout":
                            measures["oracle_fixed_mse"].append(
                                float(F.mse_loss(oracle,labels).item()))
                            measures["zero_predictor_mse"].append(
                                float(labels.square().mean().item()))
                        elif tag=="joint_support_and_query_permuted":
                            measures["oracle_joint_mse"].append(
                                float(F.mse_loss(oracle,labels).item()))
                        elif tag=="joint_permuted_with_corrupted_support_values":
                            measures["oracle_corrupted_mse"].append(
                                float(F.mse_loss(oracle,labels).item()))
                    measures["joint_prediction_corruption_mse"].append(
                        float(F.mse_loss(
                            pred_cache["joint_support_and_query_permuted"],
                            pred_cache["joint_permuted_with_corrupted_support_values"]).item()))
            # New independent heldout is untouched by training.
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
    phase("AGGREGATE_PRE_REGISTERED_S9_TRANSFER")
    metrics={}
    for arm in ("conventional","online_ttt","static_matched"):
        metrics[arm]={name:statistics.mean(
            item["arms"][arm]["heldout_means"][name]
            for item in arm_results)
            for name in arm_results[0]["arms"][arm]["heldout_means"]}
    thresholds=transfer["controls"]
    assays_valid=all(
        m["oracle_fixed_mse"]<=thresholds["exact_oracle_mse_max"] and
        m["oracle_joint_mse"]<=thresholds["exact_oracle_mse_max"] and
        m["oracle_corrupted_mse"]>=
            thresholds["corrupted_support_oracle_mse_min"] and
        abs(m["zero_predictor_mse"]-1.0)<1e-12
        for m in metrics.values()
    )
    keys=(
        "fixed_layout","query_order_permuted",
        "support_pair_positions_permuted",
        "joint_support_and_query_permuted"
    )
    conventional=metrics["conventional"]
    conventional_valid=(
        assays_valid and
        all(conventional[k]<=
            thresholds["conventional_all_4_permutations_mse_max"]
            for k in keys) and
        all(all(item["arms"]["conventional"]["heldout_means"][k] <=
                thresholds["conventional_all_4_permutations_mse_max"]
                for k in keys)
            for item in arm_results)
    )
    results={}
    for arm in ("online_ttt","static_matched"):
        m=metrics[arm]
        fixed=m["fixed_layout"]
        joint=m["joint_support_and_query_permuted"]
        corrupt=m["joint_permuted_with_corrupted_support_values"]
        require(fixed>0 and joint>0,"INVALID_TRANSFER_DENOMINATOR")
        ratio=joint/fixed
        harm=(corrupt-joint)/joint
        meets=(
            m["fixed_layout"]<=thresholds["native_fixed_layout_mse_max"] and
            m["query_order_permuted"]<=thresholds["native_query_permuted_mse_max"] and
            m["support_pair_positions_permuted"]<=
                thresholds["native_support_permuted_mse_max"] and
            joint<=thresholds["native_joint_permuted_mse_max"] and
            ratio<=thresholds["native_joint_over_fixed_mse_ratio_max"] and
            harm>=thresholds["native_joint_corruption_relative_harm_min"] and
            all(
                item["arms"][arm]["heldout_means"]["joint_support_and_query_permuted"]<=
                    thresholds["native_joint_permuted_mse_max"] and
                item["arms"][arm]["heldout_means"]["joint_support_and_query_permuted"] <=
                    thresholds["native_joint_over_fixed_mse_ratio_max"] *
                    item["arms"][arm]["heldout_means"]["fixed_layout"]
                for item in arm_results)
        )
        results[arm]={
            "joint_vs_fixed_mse_ratio":ratio,
            "joint_corrupted_support_relative_harm":harm,
            "all_frozen_flexible_binding_controls_met":meets,
        }
    status=(
        "ASSAY_INVALID_CONVENTIONAL_PERMUTATION_CONTROL" if not
        conventional_valid else
        "PRELIMINARY_FLEXIBLE_BINDING_SIGNAL_NOT_SCIENTIFIC_PASS" if
        all(results[a]["all_frozen_flexible_binding_controls_met"]
            for a in results) else
        "FIXED_LAYOUT_LEARNING_DOES_NOT_TRANSFER_UNDER_PERMUTATION"
    )
    return {
        "schema":"ttt-hybrid.g7-s9.permutation-transfer-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":status,
        "original_s8_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "new_s9_frozen_protocol_sha256":hashlib.sha256(transfer_bytes).hexdigest(),
        "qualified_s6_source_blob":reference_blob,
        "runtime_package_manifest_sha256":manifest,
        "fit_steps_per_seed_all_arms":512,
        "fit_episodes_per_seed_all_arms":32768,
        "fresh_heldout_episodes_per_seed_all_arms":1024,
        "arms_by_seed":arm_results,
        "aggregate_heldout_metrics":metrics,
        "oracle_controls_pass":assays_valid,
        "conventional_permutation_control_pass":conventional_valid,
        "native_transfer_results":results,
        "model_parameters_equal_within_native_pair":True,
        "model_parameters_or_compute_matched_to_conventional":False,
        "torch_gpu_peak_allocated_bytes":max_vram,
        "elapsed_seconds":round(time.monotonic()-t0,3),
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--transfer-protocol",required=True,type=Path)
    p.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    receipt=ROOT/"local_runs/g7_s9_permutation_transfer_evidence.json"
    report={
        "schema":"ttt-hybrid.g7-s9.permutation-transfer-evidence.v1",
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
                (ROOT/"experiments/g7_s8_frozen_episode_matched.json").resolve() and
                args.transfer_protocol.resolve()==
                (ROOT/"experiments/g7_s9_frozen_permutation_transfer.json").resolve(),
                "FROZEN_PROTOCOL_OR_RECEIPT_DESTINATION_DENIED")
        report=run(args)
        code=0 if report["status"]!="ASSAY_INVALID_CONVENTIONAL_PERMUTATION_CONTROL" else 2
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
        print("G7_S9_STATUS="+report["status"],flush=True)
        print("G7_S9_PHASE="+report.get("phase","COMPLETE"),flush=True)
        print("G7_S9_INDEPENDENT=NOT_RUN",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
