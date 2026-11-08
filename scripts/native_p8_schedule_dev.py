"""P8: fixed vs predeclared cosine LR, two factual-history neural models.

Engineering-only: no hidden context labels in training, same exact P7 synthetic
source/splits and registered initializations. No scientific first-use, original
seed consumption, model promotion or FORGE runtime integration. A negative
outcome or nonplateau is recorded, not converted into a passing science claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
from torch.nn import functional as F

from scripts.native_p7_learned_router_dev import (
    LearnedRouter, ConventionalMLP, synthetic_dev, eval_model,
    handcoded_prior_pair, HISTORY, INIT_SEEDS, TRAIN_DEV_SEED,
    VALIDATION_DEV_SEED, TRAIN_EPISODES, VALID_EPISODES,
    SCRAMBLE_DEV_SEED, BATCH,
)

ROOT=Path(__file__).resolve().parents[1]
SCHEMA="ttt-hybrid.native-p8.learning-rate-stability-dev.v1"
ARMS=("LearnedRouter","ConventionalMLP")
SCHEDULES=("fixed","cosine_decay")
UPDATES=1200
CHECKPOINT_STEPS=tuple(range(100,UPDATES+1,100))
INITIAL_LR=.006
MIN_LR=.0006
CLIP_NORM=5.
STRICT_RELATIVE_PLATEAU_SPREAD=.01
MAX_WALL_SECONDS=100
SAMPLE_SEED_OFFSET=9999


def source_sha() -> str:
    result=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],
                          check=False,capture_output=True,text=True,timeout=10)
    if result.returncode!=0 or len(result.stdout.strip())!=40:
        raise RuntimeError("P8_EXACT_SOURCE_MISSING")
    sha=result.stdout.strip()
    expected=os.environ.get("EXPECTED_CANDIDATE_SHA")
    if expected and sha!=expected:
        raise RuntimeError("P8_CANDIDATE_SOURCE_DRIFT")
    return sha


def _tensor_state_sha(model:torch.nn.Module)->str:
    digest=hashlib.sha256()
    for name,weight in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8")+b"\0")
        digest.update(weight.detach().contiguous().cpu().numpy().tobytes())
    return digest.hexdigest()


def schedule_lr(kind:str,update:int)->float:
    """Frozen exact schedule; every arm trains for the SAME 1200 updates."""
    if kind not in SCHEDULES or type(update) is not int or not 0<=update<=UPDATES:
        raise ValueError("P8_UNREGISTERED_SCHEDULE")
    if kind=="fixed":
        return INITIAL_LR
    return MIN_LR+(.5*(INITIAL_LR-MIN_LR)*
                   (1+math.cos(math.pi*update/UPDATES)))


def _validate_train_inputs(architecture:str,schedule:str,seed:int)->None:
    if architecture not in ARMS or schedule not in SCHEDULES or (
       type(seed) is not int or seed not in INIT_SEEDS):
        raise ValueError("P8_UNREGISTERED_DEV_ARM_OR_INIT")


def train_pair_arm(architecture:str,schedule:str,seed:int,
                   training:tuple,validation:tuple,shuffled:torch.Tensor)->dict:
    _validate_train_inputs(architecture,schedule,seed)
    torch.manual_seed(seed)
    model=(LearnedRouter() if architecture=="LearnedRouter" else ConventionalMLP())
    start_weights=_tensor_state_sha(model)
    start_val,start_context_acc=eval_model(model,validation)
    optimizer=torch.optim.AdamW(model.parameters(),lr=INITIAL_LR)
    sampler=torch.Generator().manual_seed(seed+SAMPLE_SEED_OFFSET)
    sampler_hash=hashlib.sha256()
    records=[]
    start=time.perf_counter()
    for step in range(1,UPDATES+1):
        ix=torch.randint(len(training[0]),(BATCH,),generator=sampler)
        sampler_hash.update(ix.numpy().tobytes())
        prediction,_=model(training[0][ix],training[1][ix])
        loss=F.mse_loss(prediction,training[2][ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),CLIP_NORM)
        optimizer.step()
        # Explicit fixed prospective schedule, no heldout-driven scheduler.
        optimizer.param_groups[0]["lr"]=schedule_lr(schedule,step)
        if step in CHECKPOINT_STEPS:
            val,context_accuracy=eval_model(model,validation)
            records.append({
                "step":step,
                "validation_mse":val,
                "diagnostic_context_sign_accuracy":context_accuracy,
                "lr_after_step":optimizer.param_groups[0]["lr"],
            })
    train_wall=time.perf_counter()-start
    final_val,final_context_acc=eval_model(model,validation)
    with torch.no_grad():
        scrambled_pred,_=model(validation[0][shuffled],validation[1])
        missing_pred,_=model(torch.zeros_like(validation[0]),validation[1])
    scrambled=float(F.mse_loss(scrambled_pred,validation[2]))
    no_history=float(F.mse_loss(missing_pred,validation[2]))
    trailing=[x["validation_mse"] for x in records[-5:]]
    maximum=max(trailing)
    minimum=min(trailing)
    relative_spread=(maximum-minimum)/max(minimum,1e-12)
    count=sum(p.numel() for p in model.parameters())
    final_weights=_tensor_state_sha(model)
    return {
        "architecture":architecture,
        "schedule":schedule,
        "init":seed,
        "initial_state_sha256":start_weights,
        "final_state_sha256":final_weights,
        "sample_indices_sha256":sampler_hash.hexdigest(),
        "parameters":count,
        "optimizer_steps":UPDATES,
        "example_draws":UPDATES*BATCH,
        "unique_dev_fit_examples":len(training[0]),
        "batch_size":BATCH,
        "initial_validation_mse":start_val,
        "initial_diagnostic_context_accuracy":start_context_acc,
        "final_validation_mse":final_val,
        "final_diagnostic_context_accuracy":final_context_acc,
        "shuffled_history_final_mse":scrambled,
        "zero_history_final_mse":no_history,
        "initial_to_final_mse_improved":final_val<start_val,
        "model_weights_changed":start_weights!=final_weights,
        "last_five_validation_mse_spread_ratio":relative_spread,
        "strict_one_percent_plateau":relative_spread<=STRICT_RELATIVE_PLATEAU_SPREAD,
        "validation_curve":records,
        "training_wall_seconds":train_wall,
        "final_learning_rate":schedule_lr(schedule,UPDATES)
    }


def run()->dict:
    if os.environ.get("NATIVE_P8_DEV_ONLY","1")!="1":
        raise RuntimeError("P8_SCIENTIFIC_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start=time.perf_counter()
    training=synthetic_dev(TRAIN_DEV_SEED,TRAIN_EPISODES)
    validation=synthetic_dev(VALIDATION_DEV_SEED,VALID_EPISODES)
    shuffle=torch.randperm(len(validation[0]),
                          generator=torch.Generator().manual_seed(SCRAMBLE_DEV_SEED))
    prior_pred,prior_choice=handcoded_prior_pair(validation[0],validation[1])
    prior_mse=float(F.mse_loss(prior_pred,validation[2]))
    arms=[]
    for seed in INIT_SEEDS:
        for architecture in ARMS:
            for schedule in SCHEDULES:
                arms.append(train_pair_arm(architecture,schedule,seed,
                                           training,validation,shuffle))
    checks={}
    paired={}
    for seed in INIT_SEEDS:
        four={(r["architecture"],r["schedule"]):r
              for r in arms if r["init"]==seed}
        params=[r["parameters"] for r in four.values()]
        hashes={r["sample_indices_sha256"] for r in four.values()}
        checks[str(seed)]={
            "all_four_arms_complete":len(four)==4,
            "same_sample_indices_across_models_and_schedules":len(hashes)==1,
            "identical_same_architecture_initial_weights":all(
                four[(a,"fixed")]["initial_state_sha256"]==
                four[(a,"cosine_decay")]["initial_state_sha256"] for a in ARMS),
            "matched_optimizer_steps_and_draws":all(
                r["optimizer_steps"]==UPDATES and r["example_draws"]==UPDATES*BATCH
                and r["batch_size"]==BATCH for r in four.values()),
            "same_fit_validation_source":all(
                r["unique_dev_fit_examples"]==len(training[0]) for r in four.values()),
            "all_models_have_real_weight_updates":all(
                r["model_weights_changed"] for r in four.values()),
            "near_equal_parameter_budget":max(params)-min(params)<.05*max(params),
            "fixed_lr_unchanged":all(
                r["validation_curve"][i]["lr_after_step"]==INITIAL_LR
                for (a,s),r in four.items() if s=="fixed" for i in range(12)),
            "cosine_lr_reaches_fixed_floor":all(
                math.isclose(r["final_learning_rate"],MIN_LR,abs_tol=1e-9)
                for (a,s),r in four.items() if s=="cosine_decay"),
            "every_record_has_12_validation_checkpoints":all(
                len(r["validation_curve"])==12 for r in four.values()),
            "validation_losses_finite":all(
                math.isfinite(r["final_validation_mse"]) and
                math.isfinite(r["last_five_validation_mse_spread_ratio"])
                for r in four.values()),
        }
        paired[str(seed)]={
            "router_fixed_vs_cosine":{
                "fixed_mse":four[("LearnedRouter","fixed")]["final_validation_mse"],
                "cosine_mse":four[("LearnedRouter","cosine_decay")]["final_validation_mse"],
                "cosine_lower_mse":four[("LearnedRouter","cosine_decay")]["final_validation_mse"]<
                    four[("LearnedRouter","fixed")]["final_validation_mse"],
                "cosine_plateau":four[("LearnedRouter","cosine_decay")]["strict_one_percent_plateau"],
                "fixed_plateau":four[("LearnedRouter","fixed")]["strict_one_percent_plateau"]
            },
            "mlp_fixed_vs_cosine":{
                "fixed_mse":four[("ConventionalMLP","fixed")]["final_validation_mse"],
                "cosine_mse":four[("ConventionalMLP","cosine_decay")]["final_validation_mse"],
                "cosine_lower_mse":four[("ConventionalMLP","cosine_decay")]["final_validation_mse"]<
                    four[("ConventionalMLP","fixed")]["final_validation_mse"],
                "cosine_plateau":four[("ConventionalMLP","cosine_decay")]["strict_one_percent_plateau"],
                "fixed_plateau":four[("ConventionalMLP","fixed")]["strict_one_percent_plateau"],
            },
            "between_architectures":{
                s:{
                    "router_mse":four[("LearnedRouter",s)]["final_validation_mse"],
                    "mlp_mse":four[("ConventionalMLP",s)]["final_validation_mse"],
                    "router_better_than_mlp":four[("LearnedRouter",s)]["final_validation_mse"]<
                        four[("ConventionalMLP",s)]["final_validation_mse"]
                } for s in SCHEDULES
            },
        }
    elapsed=time.perf_counter()-start
    integrity=(len(arms)==12 and all(all(v.values()) for v in checks.values())
               and elapsed<MAX_WALL_SECONDS)
    support={
        "cosine_beats_fixed_router_count":sum(
            paired[str(s)]["router_fixed_vs_cosine"]["cosine_lower_mse"] for s in INIT_SEEDS),
        "cosine_beats_fixed_mlp_count":sum(
            paired[str(s)]["mlp_fixed_vs_cosine"]["cosine_lower_mse"] for s in INIT_SEEDS),
        "router_beats_mlp_fixed_count":sum(
            paired[str(s)]["between_architectures"]["fixed"]["router_better_than_mlp"]
            for s in INIT_SEEDS),
        "router_beats_mlp_cosine_count":sum(
            paired[str(s)]["between_architectures"]["cosine_decay"]["router_better_than_mlp"]
            for s in INIT_SEEDS),
        "router_cosine_plateau_count":sum(
            paired[str(s)]["router_fixed_vs_cosine"]["cosine_plateau"] for s in INIT_SEEDS),
        "mlp_cosine_plateau_count":sum(
            paired[str(s)]["mlp_fixed_vs_cosine"]["cosine_plateau"] for s in INIT_SEEDS),
        "router_fixed_plateau_count":sum(
            paired[str(s)]["router_fixed_vs_cosine"]["fixed_plateau"] for s in INIT_SEEDS),
        "mlp_fixed_plateau_count":sum(
            paired[str(s)]["mlp_fixed_vs_cosine"]["fixed_plateau"] for s in INIT_SEEDS),
    }
    return {
        "schema":SCHEMA,
        "status":"PASS_FIXED_SCHEDULE_COMPARISON_ENGINEERING_ONLY" if integrity else
                 "FAIL_FIXED_SCHEDULE_COMPARISON_ENGINEERING_ONLY",
        "exact_source_sha":source_sha(),
        "scientific_first_use":"NOT_RUN",
        "scientific_world_seeds_consumed":[],
        "native_substrate_changed":False,
        "worlds_constructed":0,
        "model_count":len(arms),
        "fit_examples":len(training[0]),
        "dev_validation_examples":len(validation[0]),
        "per_model_optimizer_updates":UPDATES,
        "per_model_example_draws":UPDATES*BATCH,
        "all_model_optimizer_steps":UPDATES*len(arms),
        "all_model_examples_drawn":UPDATES*BATCH*len(arms),
        "source_p7_handcoded_mse":prior_mse,
        "source_p7_handcoded_accuracy":float(
            (torch.where(prior_choice,1,-1)==validation[3]).float().mean()),
        "engineering_integrity_checks":checks,
        "paired_diagnostics":paired,
        "hypothesis_support_of_three":support,
        "records":arms,
        "whole_process_wall_seconds":elapsed,
        "limitations":"Known P7 public synthetic family only; unrelated science worlds never constructed. A cosine schedule is a DEVELOPMENT intervention, not a prospective scientific confirmatory comparison. Last-five validation checkpoint plateau is a bounded development heuristic, not proof of fit convergence. Train and validation use distinct dev RNG but same generator; time and parameter matching are not independent FLOP parity. Never promote based on this alone."
    }


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    result=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":result["status"],
        "head":result["exact_source_sha"],
        "scientific_first_use":result["scientific_first_use"],
        "model_count":result["model_count"],
        "hypothesis_support_of_three":result["hypothesis_support_of_three"],
        "whole_process_wall_seconds":result["whole_process_wall_seconds"],
        "final_mse":[{"architecture":r["architecture"],"schedule":r["schedule"],
                      "seed":r["init"],"mse":r["final_validation_mse"],
                      "plateau":r["strict_one_percent_plateau"],
                      "spread":r["last_five_validation_mse_spread_ratio"],
                      "seconds":r["training_wall_seconds"]}
                     for r in result["records"]],
    },sort_keys=True))
    return 0 if result["status"]=="PASS_FIXED_SCHEDULE_COMPARISON_ENGINEERING_ONLY" else 1


if __name__=="__main__":
    raise SystemExit(main())
