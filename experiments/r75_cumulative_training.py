"""R75 explicit, source-neutral, FIT-only sustained training on fast equivalent engine.

Original R74 freeze, official torch/source, selected-action numerical input,
reference HybridPilot, all ablations and conventional controls remain untouched.
No implicit launch, no GPU, no DEV/OOD until separate original full-training
qualification. Checkpoints are engine-specific and cannot silently reuse R74.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.r75-accelerated-cumulative-fit.v1"
ENGINE="R75_STRICT_LOWER_TRIANGULAR_FASTWEIGHT_V1"
SOURCE="R74_FROZEN_AUTHOR_SYNTHETIC_FIT_ONLY"
FROZEN_CHECKPOINTS=(1024,4096,16384,65536)
CHECKPOINT_ROOT=ROOT/"local_runs"/"r75_cumulative_fit"


def optimized_model(arm,width,seed):
    r75.frozen()
    if arm not in r74.TRAIN_ARMS:
        raise ValueError("R75_UNREGISTERED_ARCHITECTURE")
    if width not in r74.LEVELS or seed not in r74.SEEDS:
        raise ValueError("R75_UNFROZEN_CAPACITY_OR_SEED")
    torch.manual_seed(seed)
    model=r74.make_model(arm,width)
    if hasattr(model,"ttt") and model.ttt is not None:
        old=model.ttt
        replacement=r75.TriangularFastWeightMemory(
            width,learning_rate=old.eta)
        replacement.load_state_dict(old.state_dict(),strict=True)
        model.ttt=replacement
    return model.cpu()


def canonical_sha(path):
    raw=Path(path).read_bytes().replace(b"\r\n",b"\n")
    if b"\r" in raw:
        raise PermissionError("R75_CHECKPOINT_UNSUPPORTED_TEXT_ENCODING")
    return hashlib.sha256(raw).hexdigest()


def identity(arm,width,sequence,seed):
    frozen=r75.frozen()
    if sequence not in r74.LENGTHS:
        raise ValueError("R75_UNFROZEN_SEQUENCE")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
                       capture_output=True,text=True,check=False)
    sha=git.stdout.strip() if git.returncode==0 else "NO_GIT_CHECKOUT"
    return {
       "schema":SCHEMA,
       "engine":ENGINE,
       "R75_original_freeze_blob":r75.FREEZE_BLOB,
       "R74_original_freeze_blob":r74.FREEZE_BLOB,
       "reference_model_git_blob":r74.MODEL_BLOB,
       "R75_math_engine_source_sha256":canonical_sha(r75.__file__),
       "R75_training_driver_source_sha256":canonical_sha(__file__),
       "checkout_commit_sha":sha,
       "torch":str(torch.__version__),
       "numpy":str(np.__version__),
       "cpu_device":"cpu",
       "seed":seed,"arm":arm,
       "width":width,"length":sequence,
       "FIT_seed":frozen["parent"]["R74_original_freeze_git_blob"] and
                  r74.frozen()["learning"]["source"]["FIT"]["seed"],
       "optimizer":"AdamW","optimizer_lr":.001,"optimizer_weight_decay":.0001,
       "batch":2,"clip":1.0,
       "no_spent_DEV_OOD_used":True
    }


def checkpoint_path(arm,width,sequence,seed):
    if arm not in r74.TRAIN_ARMS or seed not in r74.SEEDS:
        raise PermissionError("R75_UNKNOWN_TRAINED_MODEL")
    if width not in r74.LEVELS or sequence not in r74.LENGTHS:
        raise PermissionError("R75_UNKNOWN_FROZEN_MODEL_SIZE")
    return CHECKPOINT_ROOT/(
       f"R75-{arm}-width{width}-seq{sequence}-seed{seed}.pt")


def one_fit_update(model,optimizer,source,chosen):
    """Actual selected FIT reward prediction; no future/hidden outcome feature."""
    x=torch.from_numpy(source["input"][chosen])
    y=torch.from_numpy(source["selected_reward"][chosen])
    pred=model(x)
    if pred.shape!=y.shape:
        raise RuntimeError("R75_FIT_PREDICTION_SHAPE_DRIFT")
    mse=(pred-y).square().mean()
    if not bool(torch.isfinite(mse)):
        raise RuntimeError("R75_NONFINITE_ACTUAL_SELECTED_FIT_LOSS")
    optimizer.zero_grad(set_to_none=True)
    mse.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1.0)
    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all())
           for p in model.parameters()):
        raise RuntimeError("R75_NONFINITE_GRADIENT")
    optimizer.step()
    return float(mse.detach())


def score_fit_validation(model,source):
    x=torch.from_numpy(source["input"])
    y=torch.from_numpy(source["selected_reward"])
    model.eval()
    sums=[]
    with torch.no_grad():
        for i in range(0,len(x),2):
            output=model(x[i:i+2])
            if not bool(torch.isfinite(output).all()):
                raise RuntimeError("R75_NONFINITE_FIT_VALIDATION")
            sums.append(float((output-y[i:i+2]).square().mean()))
    model.train()
    return float(np.mean(sums))


def train_explicit_fit_only(arm,width,sequence,seed,updates):
    """A requested real cumulative FIT run, never an automatic background task.

    Exact-source checkpoint uses one canonical engine namespace and preserves
    original torch optimizer and NumPy selection RNG. No original R74
    checkpoints are touched; no DEV/OOD source ever instantiated.
    """
    spec=r75.frozen()
    if updates not in FROZEN_CHECKPOINTS:
        raise PermissionError("R75_UNFROZEN_TARGET_UPDATE_COUNT")
    if width not in r74.LEVELS or sequence not in r74.LENGTHS:
        raise PermissionError("R75_UNFROZEN_CAPACITY_OR_SEQUENCE")
    if seed not in r74.SEEDS or arm not in r74.TRAIN_ARMS:
        raise PermissionError("R75_UNAUTHORIZED_TRAINING_ARM_OR_SEED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    model=optimized_model(arm,width,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    random=np.random.default_rng(seed+24619999)
    proof=identity(arm,width,sequence,seed)
    save=checkpoint_path(arm,width,sequence,seed)
    prior=0
    history=[]
    if save.exists():
        old=torch.load(save,map_location="cpu",weights_only=True)
        if old.get("identity")!=proof:
            raise PermissionError("R75_OLD_OR_OTHER_ENGINE_CHECKPOINT_DENIED")
        prior=int(old.get("updates_completed",-1))
        if prior<0 or prior>=updates or prior not in FROZEN_CHECKPOINTS:
            raise PermissionError("R75_DUPLICATE_OR_INEXACT_REPLAY_DENIED")
        model.load_state_dict(old["model_state"],strict=True)
        opt.load_state_dict(old["optimizer_state"])
        random.bit_generator.state=old["FIT_sampling_rng"]
        history=list(old["FIT_validation_history"])
    fit=r74.factual_worlds("FIT",sequence=sequence)
    validate=r74.factual_worlds("FIT_VALIDATION",sequence=sequence)
    loss=0.0
    for step in range(prior+1,updates+1):
        chosen=random.integers(0,len(fit["input"]),2)
        loss=one_fit_update(model,opt,fit,chosen)
        if step in FROZEN_CHECKPOINTS:
            validation=score_fit_validation(model,validate)
            record={"updates":step,"FIT_last_loss":loss,
                    "FIT_validation_MSE":validation,
                    "elapsed_since_launch_seconds":time.perf_counter()-start}
            history.append(record)
            checkpoint={
              "identity":proof,
              "updates_completed":step,
              "model_state":model.state_dict(),
              "optimizer_state":opt.state_dict(),
              "FIT_sampling_rng":random.bit_generator.state,
              "FIT_validation_history":history
            }
            save.parent.mkdir(parents=True,exist_ok=True)
            tmp=save.with_suffix(save.suffix+".tmp")
            if tmp.exists():
                raise FileExistsError("R75_UNCERTAIN_TEMPORARY_CHECKPOINT_EXISTS")
            torch.save(checkpoint,tmp)
            os.replace(tmp,save)
    return {
      "schema":SCHEMA,"source":SOURCE,"identity":proof,
      "updates_completed":updates,
      "training_exposures":updates*2*sequence,
      "trainable_parameters":sum(p.numel() for p in model.parameters()),
      "FIT_validation_history":history,
      "checkpoint_engine_specific":save.name,
      "time_this_invocation_seconds":time.perf_counter()-start,
      "DEV_OOD_access":"NOT_PERFORMED",
      "scientific_pass":"UNVERIFIED",
      "CPU_only":True
    }


def require_all_control_checkpoints_before_holdout(checkpoint_dir,width,sequence=128):
    """Read-only preflight. Only complete exact-engine models can be scored.

    This deliberate guard is separate from the original R74 evaluation and
    never rebrands a short training milestone as a completed ceiling.
    """
    r75.frozen()
    if width not in r74.LEVELS or sequence not in r74.LENGTHS:
        raise ValueError("R75_UNFROZEN_FINAL_SIZE")
    files=[]
    directory=Path(checkpoint_dir)
    for seed in r74.SEEDS:
        for arm in r74.TRAIN_ARMS:
            path=directory/checkpoint_path(arm,width,sequence,seed).name
            if not path.is_file():
                raise FileNotFoundError(f"R75_INCOMPLETE_65536_TRAINING:{path.name}")
            rec=torch.load(path,map_location="cpu",weights_only=True)
            if (rec.get("identity")!=identity(arm,width,sequence,seed)
                or rec.get("updates_completed")!=FROZEN_CHECKPOINTS[-1]):
                raise PermissionError("R75_INCOMPLETE_OR_DRIFTED_CHECKPOINT_CUSTODY")
            files.append(path.name)
    return {"source":"R75_ORIGINAL_EXACT_ENGINE_FULL_TRAINING_PRECHECK",
            "engine":ENGINE,"full_controls":files,
            "source_DEV_OOD_generated":False,
            "scientific_qualification":"NOT_YET_EVALUATED"}



def score_completed_heldout(width,sequence=128,checkpoint_dir=CHECKPOINT_ROOT):
    """Only after ALL 14 exactly custodied 65,536-step checkpoints exist.

    Max-first training and best conventional selection use frozen R74 source,
    objective and scientific thresholds. No failed or incomplete model is
    silently left out. DEV and OOD are created only after preflight succeeds.
    This is local DEV adjudication, NOT independent science/GPU promotion.
    """
    import gc

    preflight=require_all_control_checkpoints_before_holdout(
        checkpoint_dir,width,sequence)
    spec=r74.frozen()["learning"]
    thresholds=r74.frozen()["learning"]["metrics"]
    # This is the FIRST generation/access to pre-frozen scored data here.
    dev=r74.factual_worlds("DEV",sequence=sequence)
    ood=r74.factual_worlds("OOD",sequence=sequence)
    all_scores={}
    decisions=[]
    for seed in r74.SEEDS:
        candidate_source={}
        for arm in r74.TRAIN_ARMS:
            ckpt=Path(checkpoint_dir)/checkpoint_path(
                arm,width,sequence,seed).name
            payload=torch.load(ckpt,map_location="cpu",weights_only=True)
            model=optimized_model(arm,width,seed)
            model.load_state_dict(payload["model_state"],strict=True)
            model.eval()
            candidate_source[arm]={
                "DEV":r74._score_model(model,dev),
                "OOD":r74._score_model(model,ood),
                "parameters":sum(p.numel() for p in model.parameters()),
                "FIT_65536_checkpoint":ckpt.name
            }
            del payload,model
            gc.collect()
        comparator=min(("GRU64","MLP64"),
            key=lambda arm:candidate_source[arm]["DEV"]["mean_selected_reward_MSE"])
        base=candidate_source[comparator]
        candidate=candidate_source["HybridPilot_FULL"]
        def relative_gain(panel,key):
            x=base[panel][key]
            y=candidate[panel][key]
            if not x>0 or not np.isfinite(x) or not np.isfinite(y):
                raise ValueError("R75_INVALID_HELDOUT_MSE_SCORE")
            return (x-y)/x
        d=relative_gain("DEV","mean_selected_reward_MSE")
        o=relative_gain("OOD","mean_selected_reward_MSE")
        c=relative_gain("DEV","cold_A_return_MSE")
        lower=r74._bootstrap_lower_gain(
            base["DEV"]["per_world_MSE"],
            candidate["DEV"]["per_world_MSE"],seed)
        checks={
            "DEV_relative_gain_min_5pct":d>=.05,
            "OOD_relative_gain_min_5pct":o>=.05,
            "cold_return_relative_gain_min_5pct":c>=.05,
            "paired_lower_bootstrap_positive":lower>0,
            "all_14_full_65536_updates_exact_custody":True,
            "reference_CPU_not_official_GPU":True
        }
        decisions.append({"seed":seed,
           "strongest_fully_trained_conventional_control":comparator,
           "DEV_relative_gain":d,"OOD_relative_gain":o,
           "cold_A_return_relative_gain":c,
           "paired_bootstrap_lower95_baseline_minus_hybrid_MSE":lower,
           "checks":checks,"all_frozen_checks":all(checks.values())})
        all_scores[str(seed)]={
           arm:{
               "params":data["parameters"],
               "checkpoint":data["FIT_65536_checkpoint"],
               "DEV":{k:v for k,v in data["DEV"].items()
                      if k!="per_world_MSE"},
               "OOD":{k:v for k,v in data["OOD"].items()
                      if k!="per_world_MSE"}
           } for arm,data in candidate_source.items()}
    result=("DEV_REFERENCE_HYBRID_MECHANISM_SIGNAL_ONLY"
            if all(d["all_frozen_checks"] for d in decisions)
            else "DEV_SCIENTIFIC_NO_GO")
    return {"schema":SCHEMA,"engine":ENGINE,
        "source":"ORIGINAL_FROZEN_R74_NEW_DEV_OOD",
        "checkpoint_preflight":preflight,
        "source_counts":{"DEV":len(dev["input"]),"OOD":len(ood["input"])},
        "scores":all_scores,"decisions":decisions,
        "development_result_only":result,
        "independent_scientific_pass":False,
        "full_official_Mamba2_TTT_GPU_qualification":False,
        "action_selection_learning_proven":False,
        "deployment":"NONE"}



def main():
    p=argparse.ArgumentParser()
    p.add_argument("--train",action="store_true")
    p.add_argument("--preflight",action="store_true")
    p.add_argument("--score-complete",action="store_true")
    p.add_argument("--arm",choices=r74.TRAIN_ARMS,default="HybridPilot_FULL")
    p.add_argument("--width",type=int,default=1024)
    p.add_argument("--sequence",type=int,default=128)
    p.add_argument("--seed",type=int,default=r74.SEEDS[0])
    p.add_argument("--updates",type=int,default=1024)
    a=p.parse_args()
    if a.train:
        result=train_explicit_fit_only(a.arm,a.width,a.sequence,a.seed,a.updates)
    elif a.preflight:
        result=require_all_control_checkpoints_before_holdout(
            CHECKPOINT_ROOT,a.width,a.sequence)
    elif a.score_complete:
        result=score_completed_heldout(a.width,a.sequence)
    else:
        p.error("use --train, --preflight or --score-complete")
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
