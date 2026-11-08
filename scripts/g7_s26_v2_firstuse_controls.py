"""S26-V2 separate, operator-authorized conventional baseline fit/validation first use.

Science FIT + VALIDATION seeds ONLY; all heldout, transfer, negative and bootstrap
remain unopened. Uses original S25 GRU fitter and KRR calibrator unmodified.
All model targets are produced by the same-state paired S20 simulator after
constructing prior-only history. This is NOT real-world causal identification,
not S26/TTT adaptation acceptance, and not original S25/S26 scientific first use.

Run only on pinned source/registered role-scoped project-owned CPU runner.
All three fit/validation seeds and both conventional arms are attempted once;
null/nonconvergence is preserved, not tuned away or silently skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import dataclass

sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import Tensor
from torch.nn import functional as F
import g7_s25_numeric_ttt as numeric
from g7_s25_paired_interventions import PairedS20World,MAX_STEPS,PAIR_ACTIONS
from g7_s26_v2_baseline_observations import fingerprint_step
from g7_process_peak_memory import read_peak_memory

ROOT=Path(__file__).resolve().parents[1]
PROTOCOL=ROOT/"experiments/g7_s26_v2_fitval_firstuse_frozen.json"
SCHEMA="ttt-hybrid.g7-s26-v2.fit-validation-firstuse-frozen.v1"
PROTOCOL_BLOB="d72b9bb41e5e7e5dd9229c7bc5053f5a42f9750d"
SCIENCE_WORLD_BLOB="ec5442ecacb44f594b142b8666ca64fa4abeeab3"
NUMERIC_BLOB="7608b55f8abe5238ea37f16a2039e99c20fbb38f"
PARENT_SHA="255aa654bd3dfc6d5b36f137555e4d33dcf95cb1"
FIT=(7207,7219,7229)
VALIDATION=(7307,7319,7327)
NO_SEEDS=(7403,7411,7429,7507,7517,7529,7603,7613,7627,7807)
DEV=(3101,3109,3119)
TRAIN_SET_ROWS=128
BATCH=32
STEPS=32
STRIDE=8
MAX_ARM_SECONDS=900
MAX_RSS=2147483648
REGIME="source_stable"
ACTION_SIGN=0.012


class FirstUseBlocked(RuntimeError):
    pass


def require(ok,why):
    if not ok:
        raise FirstUseBlocked(why)


def git(*args):
    p=subprocess.run(["git","-C",str(ROOT),*args],text=True,
                     capture_output=True,check=False,timeout=12)
    require(p.returncode==0,"FIRSTUSE_GIT_CUSTODY_UNAVAILABLE")
    return p.stdout.strip()


def authorize_exact_source(sha):
    require(type(sha) is str and len(sha)==40 and
            all(x in "0123456789abcdef" for x in sha) and
            git("rev-parse","HEAD")==sha and
            git("status","--porcelain","--untracked-files=all")=="",
            "FIRSTUSE_CHECKOUT_NOT_CLEAN_EXACT_SHA")
    git("merge-base","--is-ancestor",PARENT_SHA,"HEAD")
    for path,blob in (
        ("experiments/g7_s26_v2_fitval_firstuse_frozen.json",PROTOCOL_BLOB),
        ("scripts/g7_s25_paired_interventions.py",SCIENCE_WORLD_BLOB),
        ("scripts/g7_s25_numeric_ttt.py",NUMERIC_BLOB),
        ("experiments/g7_s26_frozen_quarter_reserve.json",
         "34251b2cde6373ed484dbc21e4617c9165047e93"),
        ("experiments/g7_s25_frozen_contradiction_ttt.json",
         "31633b341403729b7ffbb7d73f3a04098ad44380"),
        ("scripts/g7_s20_hidden_multicycle.py",
         "bc3e9f255a4d1b967252ada8782ac9f8c000f7b7"),
        ("source_pins.json","1438b51f8548db6bfe7e04805ba8c156e4b5b630"),
    ):
        require(git("hash-object","--",path)==blob,
                "FIRSTUSE_SOURCE_OR_FROZEN_PROTOCOL_DRIFT_"+path)
    p=json.loads(PROTOCOL.read_text(encoding="utf-8"))
    a=p["authority"]
    require(p["schema"]==SCHEMA and
            p["parent_qualified_engineering_sha"]==PARENT_SHA and
            a["first_use_authorized_for_this_new_protocol"] is True and
            a["heldout_transfer_negative_bootstrap_forbidden"] is True and
            a["GPU_training_forbidden"] is True and
            p["role_seeds"]["fit"]==list(FIT) and
            p["role_seeds"]["validation"]==list(VALIDATION) and
            p["controls"]["GRU_fully_trained"]["max_updates"]==1600 and
            p["world"]["selected_samples_per_role_seed"]==TRAIN_SET_ROWS and
            p["world"]["dataset_selection"]==
            "exact first sample index 0 then every 8th prequential row from 1024, yields 128 world-step examples in natural chronological order",
            "FIRSTUSE_REGISTERED_ROLE_OR_DESIGN_DRIFT")
    groups=list(p["role_seeds"].values())
    require(set(FIT).isdisjoint(DEV+VALIDATION+NO_SEEDS) and
            set(VALIDATION).isdisjoint(DEV+NO_SEEDS) and
            not any(k in ("heldout","transfer","negative","bootstrap")
                    for k in ("fit","validation")),
            "FIRSTUSE_RESEARCH_SEED_COLLISION")
    return p


@dataclass(frozen=True)
class SourceBatch:
    h:Tensor
    q:Tensor
    y:Tensor
    selected:Tensor
    source_sha256:str
    selected_sha256:str
    seed:int
    role:str
    raw_exposures:int
    selected_exposures:int


@torch.no_grad()
def paired_source(seed:int,role:str)->SourceBatch:
    require(type(seed) is int and
            ((role=="fit" and seed in FIT) or
             (role=="validation" and seed in VALIDATION)),
            "FIRSTUSE_UNREGISTERED_FIT_VALIDATION_SEED")
    w=PairedS20World(seed,BATCH,REGIME,registered_role=role)
    hlist=[];qlist=[];ylist=[];alist=[]
    whole=hashlib.sha256()
    last=None
    for step in range(STEPS):
        row=w.emit()
        require(row.history.shape==(BATCH,1 if step==0 else step,4) and
                row.query_pair.shape==(BATCH,2,3) and
                row.observed_pair.shape==(BATCH,2) and
                row.realized_action_index.shape==(BATCH,),
                "FIRSTUSE_WORLD_SHAPE_OR_STEP_DRIFT")
        hist=numeric.causal_history32(row.history)
        require(bool(torch.isfinite(hist).all() and
                     torch.isfinite(row.query_pair).all() and
                     torch.isfinite(row.observed_pair).all() and
                     torch.equal(row.query_pair[:,0,0],row.query_pair[:,1,0]) and
                     torch.equal(row.query_pair[:,0,2],row.query_pair[:,1,2])),
                "FIRSTUSE_NONFINITE_OR_NONCAUSAL_PAIR")
        if last is not None:
            require(bool(torch.equal(row.history[:,-1,3],last)),
                    "FIRSTUSE_REALIZED_HISTORY_MISMATCH")
        last=row.observed_pair.gather(
            1,row.realized_action_index[:,None]).squeeze(1).clone()
        fingerprint_step(whole,step,row,hist)
        hlist.append(hist)
        qlist.append(row.query_pair.detach().clone())
        ylist.append(row.observed_pair.detach().clone())
        alist.append(row.realized_action_index.detach().clone())
    closed=w.close()
    require(closed["worlds"]==BATCH and closed["steps"]==STEPS and
            closed["paired_observations"]==BATCH*STEPS*2 and
            closed["realized_transitions"]==BATCH*STEPS,
            "FIRSTUSE_INCOMPLETE_FROZEN_WORLD")
    a=torch.cat(hlist);b=torch.cat(qlist);y=torch.cat(ylist);idx=torch.cat(alist)
    require(a.shape==(1024,32,4) and b.shape==(1024,2,3) and
            y.shape==(1024,2),"FIRSTUSE_BAD_PUBLIC_ARRAY_SHAPES")
    chosen=torch.arange(0,1024,STRIDE,dtype=torch.long)
    require(chosen.numel()==TRAIN_SET_ROWS,
            "FIRSTUSE_PRECOMMITTED_ROW_SAMPLE_DRIFT")
    ah=a[chosen];bq=b[chosen];ty=y[chosen];ix=idx[chosen]
    # The 128-sample content digest is independent from the full 1024
    # stream provenance, role, seed, tensor dtype and shape are included.
    fingerprint=hashlib.sha256()
    for name,t in (("h",ah),("q",bq),("y",ty),("index",ix)):
        fingerprint.update(name.encode())
        fingerprint.update(str(tuple(t.shape)).encode())
        fingerprint.update(str(t.dtype).encode())
        fingerprint.update(t.contiguous().numpy().tobytes())
    require(bool(((ix==0)|(ix==1)).all() and
                 (bq[:,0,1]==PAIR_ACTIONS[0]).all() and
                 (bq[:,1,1]==PAIR_ACTIONS[1]).all()),
            "FIRSTUSE_PAIR_SELECTOR_OR_ACTION_DRIFT")
    return SourceBatch(ah,bq,ty,ix,whole.hexdigest(),
                       fingerprint.hexdigest(),seed,role,1024,128)


def validation_readout(predicted:Tensor,data:SourceBatch):
    require(predicted.shape==data.y.shape and
            bool(torch.isfinite(predicted).all()),
            "FIRSTUSE_NONFINITE_VALIDATION_PREDICTIONS")
    d=predicted.detach().float()
    truth=data.y
    observed_effect=truth[:,1]-truth[:,0]
    predicted_effect=d[:,1]-d[:,0]
    observed_significant=observed_effect.abs()>=ACTION_SIGN
    predicted_significant=predicted_effect.abs()>=ACTION_SIGN
    eligible=observed_significant&predicted_significant
    correct=int(((predicted_effect*observed_effect>0)&eligible).sum())
    wrong=int(((predicted_effect*observed_effect<0)&eligible).sum())
    total=int(observed_significant.sum())
    coverage=(correct+wrong)/total if total else 0.0
    static=data.q[:,:,0]
    naive=float(F.mse_loss(static,truth).item())
    mse=float(F.mse_loss(d,truth).item())
    effect_mae=float((predicted_effect-observed_effect).abs().mean().item())
    require(all(math.isfinite(x) for x in (naive,mse,effect_mae,coverage)),
            "FIRSTUSE_NONFINITE_METRICS")
    return {
        "validation_paired_mse":mse,
        "static_naive_paired_mse":naive,
        "action_effect_significant_actual":total,
        "action_effect_eligible_predicted":int(eligible.sum()),
        "action_effect_correct_sign":correct,
        "action_effect_wrong_sign":wrong,
        "action_effect_coverage":coverage,
        "action_effect_sign_accuracy_when_evaluable":(
            correct/(correct+wrong) if correct+wrong else None),
        "action_effect_mean_absolute_error":effect_mae,
        "validation_pair_examples":int(truth.shape[0]),
        "validation_action_probes":int(truth.numel()),
        "predict_before_target_scoring":True,
        "source_is_simulator_paired_clones_not_real_world":True,
        "precommitted_admission_criteria_met":(
            mse<naive and correct>wrong and coverage>0.5),
    }


def serialize_checkpoint(obj:dict,where:Path)->str:
    # No code loading and no untrusted pickle classes. Only torch tensors,
    # integers, floats and strings. Persist a proof artifact for cold read.
    where.parent.mkdir(parents=True,exist_ok=True)
    torch.save(obj,where)
    return hashlib.sha256(where.read_bytes()).hexdigest()


def source_resource_check(start:float):
    peak=read_peak_memory()
    require(peak.metric=="linux_ru_maxrss_bytes" and
            peak.comparable_to_linux_ru_maxrss and
            0<peak.peak_bytes<=MAX_RSS,
            "FIRSTUSE_LINUX_RSS_CUSTODY_OR_BUDGET_FAIL")
    elapsed=time.monotonic()-start
    require(elapsed<MAX_ARM_SECONDS,"FIRSTUSE_WALL_BUDGET_EXCEEDED")
    return {"peak_rss_bytes":peak.peak_bytes,
            "peak_rss_metric":peak.metric,
            "monotonic_elapsed_seconds":elapsed}


def run_gru(fit:SourceBatch,valid:SourceBatch,path:Path):
    start=time.monotonic();cpu0=time.process_time()
    torch.manual_seed(fit.seed)
    m=numeric.ConventionalGRU().cpu()
    val_h=valid.h.clone();val_q=valid.q.clone();val_y=valid.y.clone()
    def training_batch(step):
        require(type(step) is int and 1<=step<=1600,
                "FIRSTUSE_GRU_TRAIN_STEP_OUT_OF_REGISTERED_RANGE")
        k=((step-1)%4)*BATCH
        return fit.h[k:k+BATCH],fit.q[k:k+BATCH],fit.y[k:k+BATCH]
    def validation_batch():
        return val_h,val_q,val_y
    # Invoke the pre-existing fully trained control without custom tuning.
    training=numeric.fit_conventional_gru(
        m,training_batch,validation_batch,
        learning_rate=.001,max_updates=1600)
    m.eval()
    with torch.no_grad():
        pred=m.predict(val_h,val_q).detach().clone()
    result=validation_readout(pred,valid)
    bytes_sha=serialize_checkpoint(
        {"arm":"GRU_fully_trained","fit_seed":fit.seed,
         "model_state":m.state_dict(),"optimizer_updates":training.updates},
        path)
    cold=torch.load(path,map_location="cpu",weights_only=True)
    restored=numeric.ConventionalGRU().cpu()
    restored.load_state_dict(cold["model_state"],strict=True)
    restored.eval()
    with torch.no_grad():
        new=restored.predict(val_h,val_q)
    require(torch.equal(pred,new),"FIRSTUSE_GRU_COLD_RESTORE_DIVERGENCE")
    elapsed=time.monotonic()-start
    cost=source_resource_check(start)
    cpu=time.process_time()-cpu0
    require(cpu<=MAX_ARM_SECONDS,"FIRSTUSE_GRU_CPU_SECONDS_BUDGET_EXCEEDED")
    return {
        "arm":"GRU_fully_trained","fit_seed":fit.seed,"validation_seed":valid.seed,
        "checkpoint_sha256":bytes_sha,"checkpoint_bytes":path.stat().st_size,
        "optimizer_updates":training.updates,
        "validation_history":[{"step":640+64*j,"paired_mse":float(v)}
                              for j,v in enumerate(training.validation_mse)],
        "trainer_status":training.status,
        "fully_converged":bool(training.converged),
        "training_observation_samples":fit.selected_exposures,
        "training_observation_pairs":fit.selected_exposures*2,
        "training_gradient_pair_exposures":training.updates*BATCH*2,
        "same_unique_training_examples_as_KRR":True,
        "validation_stream_sha256":valid.selected_sha256,
        "training_stream_sha256":fit.selected_sha256,
        "validation_readout":result,
        "checkpoint_cold_restore":"PASS_BIT_IDENTICAL",
        "cpu_seconds":cpu,"wall_seconds":elapsed,**cost,
        "scientific_baseline_admission":(
            "PASS_TRAINED_CONTROL_ONLY" if training.converged and
            result["precommitted_admission_criteria_met"] else
            "FAIL_NONCONVERGED_OR_NOT_COMPETENT"),
    }


def run_krr(fit:SourceBatch,valid:SourceBatch,path:Path):
    start=time.monotonic();cpu0=time.process_time()
    m=numeric.CalibratedRBFKernelRidge()
    calibration=m.fit_calibrated((fit.h,fit.q,fit.y),
                                 (valid.h,valid.q,valid.y))
    with torch.no_grad():
        pred=m.predict(valid.h,valid.q).detach().clone()
    result=validation_readout(pred,valid)
    sha=serialize_checkpoint({
        "arm":"RBF_KRR","fit_seed":fit.seed,
        "mean":m.mean,"scale":m.scale,
        "train_vectors":m.train_vectors,"weights":m.weights,
        "gamma":m.gamma,"ridge":m.ridge
    },path)
    cold=torch.load(path,map_location="cpu",weights_only=True)
    restored=numeric.CalibratedRBFKernelRidge()
    for field in ("mean","scale","train_vectors","weights","gamma","ridge"):
        setattr(restored,field,cold[field])
    with torch.no_grad():
        new=restored.predict(valid.h,valid.q)
    require(torch.equal(pred,new),"FIRSTUSE_KRR_COLD_RESTORE_DIVERGENCE")
    elapsed=time.monotonic()-start
    cost=source_resource_check(start)
    cpu=time.process_time()-cpu0
    require(cpu<=MAX_ARM_SECONDS,"FIRSTUSE_KRR_CPU_SECONDS_BUDGET_EXCEEDED")
    return {
        "arm":"RBF_KRR","fit_seed":fit.seed,"validation_seed":valid.seed,
        "checkpoint_sha256":sha,"checkpoint_bytes":path.stat().st_size,
        "gamma":calibration["gamma"],"ridge":calibration["ridge"],
        "selected_validation_mse":calibration["validation_mse"],
        "train_query_samples":calibration["train_query_samples"],
        "train_only_normalization":True,
        "validation_only_grid_selection":True,
        "frozen_six_hyperparameter_grid":True,
        "training_observation_samples":fit.selected_exposures,
        "training_observation_pairs":fit.selected_exposures*2,
        "training_gradient_pair_exposures":0,
        "same_unique_training_examples_as_GRU":True,
        "validation_stream_sha256":valid.selected_sha256,
        "training_stream_sha256":fit.selected_sha256,
        "validation_readout":result,
        "checkpoint_cold_restore":"PASS_BIT_IDENTICAL",
        "cpu_seconds":cpu,"wall_seconds":elapsed,**cost,
        "scientific_baseline_admission":(
            "PASS_TRAINED_CONTROL_ONLY" if result[
                "precommitted_admission_criteria_met"] else
            "FAIL_NOT_COMPETENT"),
    }


def experiment(expected_sha:str,output:Path)->dict:
    p=authorize_exact_source(expected_sha)
    require(sys.platform=="linux" and
            os.environ.get("CUDA_VISIBLE_DEVICES")=="-1" and
            os.environ.get("S26_V2_PROJECT_FIRST_USE")=="FIT_VALIDATION_ONLY" and
            os.environ.get("GITHUB_REPOSITORY")=="spoonman136668-ai/TTT-Hybrid-Lab" and
            os.environ.get("GITHUB_REF")==
                "refs/heads/research/g7-s26-v2-fitval-firstuse-20261107" and
            os.environ.get("GITHUB_SHA")==expected_sha and
            os.environ.get("RUNNER_NAME")=="TTT-HYBRID-GPU-WIN" and
            os.environ.get("GITHUB_EVENT_NAME")=="push" and
            os.environ.get("GITHUB_RUN_ATTEMPT")=="1",
            "FIRSTUSE_RUNNER_OWNER_OR_SCOPE_UNAUTHORIZED")
    require(not torch.cuda.is_initialized() and
            not torch.backends.cuda.is_built() or
            os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "FIRSTUSE_CUDA_INADMISSIBLE")
    torch.set_num_threads(1)
    whole_start=time.monotonic()
    records=[]
    firstuse={
        "schema":"ttt-hybrid.g7-s26-v2.authentic-fit-validation-conventional-controls.v1",
        "status":"FIRSTUSE_EXECUTION_IN_PROGRESS_NOT_SCIENCE_PASS",
        "expected_sha":expected_sha,
        "preregistration_blob":PROTOCOL_BLOB,
        "actual_science_seed_roles":{"fit":list(FIT),"validation":list(VALIDATION)},
        "excluded_heldout_negative_transfer_bootstrap":"NOT_ACCESSED",
        "trained_controls":records,
        "cross_platform_full_regression_required_before_firstuse":True,
        "original_s25_s26_scientific_firstuse":"STILL_BLOCKED",
        "gpu":"NOT_USED","merges":"NOT_PERFORMED",
        "deployment":"NOT_PERFORMED",
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=output.parent/"g7_s26_v2_scientific_checkpoints"
    try:
        for fit_seed,val_seed in zip(FIT,VALIDATION):
            # Same two actual deterministic S20 data streams for BOTH arms.
            source=paired_source(fit_seed,"fit")
            validation=paired_source(val_seed,"validation")
            require(source.selected_sha256!=validation.selected_sha256 and
                    source.source_sha256!=validation.source_sha256 and
                    source.selected_exposures==validation.selected_exposures==128,
                    "FIRSTUSE_FIT_VALIDATION_DATA_COLLISION")
            for arm in ("GRU_fully_trained","RBF_KRR"):
                checkpoint=checkpoint_dir/f"{arm}_{fit_seed}.pt"
                if arm=="GRU_fully_trained":
                    result=run_gru(source,validation,checkpoint)
                else:
                    result=run_krr(source,validation,checkpoint)
                result["original_seed_role_and_data_stream"]={
                    "fit_seed":fit_seed,"validation_seed":val_seed,
                    "full_fit_stream_sha256":source.source_sha256,
                    "full_validation_stream_sha256":validation.source_sha256,
                    "selected_fit_sha256":source.selected_sha256,
                    "selected_validation_sha256":validation.selected_sha256,
                }
                records.append(result)
                firstuse["records_completed"]=len(records)
                # Immutable files are retained even if a later arm is invalid.
                output.write_text(json.dumps(firstuse,indent=2,sort_keys=True)+"\n",
                                  encoding="utf-8")
        require(len(records)==6 and
                len({x["checkpoint_sha256"] for x in records})==6 and
                all(records[i]["training_stream_sha256"]==
                    records[i+1]["training_stream_sha256"] and
                    records[i]["validation_stream_sha256"]==
                    records[i+1]["validation_stream_sha256"]
                    for i in (0,2,4)),
                "FIRSTUSE_MISSING_ARM_OR_MISMATCHED_INFORMATION")
        success=all(r["scientific_baseline_admission"]==
                    "PASS_TRAINED_CONTROL_ONLY" for r in records)
        firstuse["status"]=(
            "PASS_CONVENTIONAL_FIT_VALIDATION_BASELINES_ONLY"
            if success else "FAIL_CONVENTIONAL_BASELINE_COMPETENCE_OR_PLATEAU")
        firstuse["all_strong_conventional_controls_competent"]=success
        firstuse["scientific_TTT_efficacy"]="NOT_EVALUATED"
        firstuse["independent_reproduction"]="NOT_PERFORMED"
        firstuse["crossplatform_science_reproduction"]="NOT_PERFORMED"
        firstuse["scientific_firstuse_scope"]="FIT_VALIDATION_ONLY"
        firstuse["whole_experiment_wall_seconds"]=time.monotonic()-whole_start
        firstuse["whole_experiment_cpu_seconds"]=sum(
            x["cpu_seconds"] for x in records)
        firstuse["whole_experiment_peak_rss_bytes"]=read_peak_memory().peak_bytes
        firstuse["different_training_compute_but_matched_unique_information"]=True
        require(firstuse["whole_experiment_peak_rss_bytes"]<=MAX_RSS,
                "FIRSTUSE_WHOLE_PROCESS_RSS_CAP_EXCEEDED")
        return firstuse
    except Exception as exc:
        firstuse["status"]="BLOCKED_SCIENTIFIC_FIRSTUSE_EXECUTION"
        firstuse["error_code"]=str(exc)[:350]
        firstuse["records_completed"]=len(records)
        raise
    finally:
        output.write_text(json.dumps(firstuse,indent=2,sort_keys=True)+"\n",
                          encoding="utf-8")


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--expected-sha",required=True)
    parser.add_argument("--receipt",type=Path,required=True)
    a=parser.parse_args()
    try:
        r=experiment(a.expected_sha,a.receipt)
    except Exception as exc:
        print("S26_V2_SCIENCE_FIRSTUSE=BLOCKED_"+str(exc)[:150],flush=True)
        return 1
    print("S26_V2_CONVENTIONAL_BASELINES="+r["status"],flush=True)
    print("S26_V2_GRU_KRR_RECORDS="+str(r["records_completed"]),flush=True)
    print("S26_V2_SCIENTIFIC_TTT_EFFICACY=NOT_EVALUATED",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
