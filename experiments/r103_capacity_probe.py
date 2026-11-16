"""R103 UNIT_ONLY cost rehearsal; never generates scored FIT/DEV worlds.

Separate from original science. Same 16-support input shape, optimizer,
16-sample batch and all 11 arms; measures short CPU optimizer traces only.
Projections are *diagnostic estimates*, not a qualified first-use completion
claim, because data generation, full validation and scoring costs differ.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r103_relational_binding_diagnostic as r103

ROOT=Path(__file__).resolve().parents[1]
STEPS_RECORDED=6
WARMUP_STEPS=2
MAX_PROBE_SECONDS=120
SOURCE_UNIT_COUNT=16
RECEIPT=ROOT/"local_runs/r103_unit_only_capacity_probe.json"


def prepare_unit_only():
    """Build exactly 16 synthetic UNIT-only episodes; no original panel APIs."""
    r103.frozen()
    if (ROOT/"experiments/r103_cpu_firstuse_authorization.json").exists():
        raise PermissionError("R103_CAPACITY_PROBE_NOT_FOR_ORIGINAL_FIRSTUSE")
    examples=[]
    started=time.perf_counter()
    for i in range(SOURCE_UNIT_COUNT):
        family_i=i%len(r103.KNOWN+r103.UNSEEN)
        episode=r103.unit_fixture(family_i)
        public=episode["public"]
        evaluator=episode["evaluator_only"]
        action=r85.CALIBRATE if i%2==0 else r85.EXPLOIT_POS
        q=public["query0" if action==r85.CALIBRATE else "query2"]
        actual=evaluator["return0" if action==r85.CALIBRATE else "return2"]
        shuffled_order=r103.public_shuffled_order(
            public["support_actions"],public["support_slots"],family_i)
        examples.append((public["support_slots"],public["support_rewards"],
                         public["support_rewards"][shuffled_order],
                         q,actual,family_i))
    seconds=time.perf_counter()-started
    x=torch.from_numpy(np.stack([v[0] for v in examples]).astype(np.float32))
    factual=torch.from_numpy(np.stack([v[1] for v in examples]).astype(np.float32))
    shuffled=torch.from_numpy(np.stack([v[2] for v in examples]).astype(np.float32))
    query=torch.from_numpy(np.stack([v[3] for v in examples]).astype(np.float32))
    target=torch.tensor([v[4] for v in examples],dtype=torch.float32)
    if (x.shape!=(16,16,5,13) or factual.shape!=(16,16)
        or shuffled.shape!=(16,16) or query.shape!=(16,5,13)
        or target.shape!=(16,) or not bool(torch.isfinite(target).all())):
        raise RuntimeError("R103_UNIT_CAPACITY_PUBLIC_FACTUAL_SHAPE_DRIFT")
    return {"x":x,"factual":factual,"shuffled":shuffled,"query":query,
            "target":target,"unit_physics_seconds":seconds,
            "units_count":SOURCE_UNIT_COUNT}


def run_unit_only():
    """No original SHA/seed admission; no trained checkpoints saved."""
    f=r103.frozen()
    if (ROOT/"experiments/r103_cpu_firstuse_authorization.json").exists():
        return {"status":"R103_CAPACITY_PROBE_SKIPPED_AUTHORIZED_SOURCE",
                "original_scored_source_opened":False}
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,
                       text=True,timeout=10,check=True).stdout.strip()
    requested=os.getenv("EXPECTED_CANDIDATE_SHA","")
    if requested and requested!=sha:
        raise PermissionError("R103_UNIT_PROBE_EXACT_SHA_DRIFT")
    if os.getenv("GITHUB_ACTIONS")=="true" and os.getenv("RUNNER_OS") not in ("Linux","Windows"):
        raise PermissionError("R103_UNIT_PROBE_NOT_HOSTED_CPU")
    torch.set_num_threads(1)
    started=time.perf_counter()
    b=prepare_unit_only()
    t=f["training"]
    models=r103.make_models(r103.SEEDS[0])
    measured={}
    for arm,model in models.items():
        elapsed=time.perf_counter()-started
        if elapsed>MAX_PROBE_SECONDS:
            break
        rewards=b["shuffled"] if arm==r103.SHUFFLED else b["factual"]
        optimizer=torch.optim.AdamW(
            model.parameters(),lr=t["learning_rate"],
            weight_decay=t["weight_decay"])
        times=[]
        for k in range(WARMUP_STEPS+STEPS_RECORDED):
            if time.perf_counter()-started>MAX_PROBE_SECONDS:
                break
            t0=time.perf_counter()
            prediction=model(b["x"],rewards,b["query"])
            if prediction.shape!=(len(b["target"]),):
                raise RuntimeError("R103_UNIT_PROBE_TRAINED_VALUE_SHAPE_DRIFT")
            loss=F.mse_loss(prediction,b["target"])
            if not bool(torch.isfinite(loss)):
                raise RuntimeError("R103_UNIT_PROBE_NONFINITE_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),t["gradient_clip"])
            optimizer.step()
            if k>=WARMUP_STEPS:
                times.append(time.perf_counter()-t0)
        if len(times)<STEPS_RECORDED:
            break
        measured[arm]={
            "steps_after_warmup":STEPS_RECORDED,
            "warmup_steps":WARMUP_STEPS,
            "timed_step_seconds":times,
            "median_step_seconds":float(np.median(times)),
            "mean_step_seconds":float(np.mean(times)),
            "projected_3072_step_seconds_excludes_validation":
                float(np.median(times)*t["steps_per_arm"]),
            "trainable_parameters":sum(
                p.numel() for p in model.parameters() if p.requires_grad)}
        del optimizer
    total_probe_seconds=time.perf_counter()-started
    completed=set(measured)==set(r103.ARMS)
    sum_optimizer=float(sum(
        v["projected_3072_step_seconds_excludes_validation"]
        for v in measured.values()))
    # 1856 frozen episodes, each with 16 factual returns + separate query.
    episode_count=sum(f["source"]["counts"].values())
    sampled_source_seconds=b["unit_physics_seconds"]
    projected_source=sampled_source_seconds*episode_count/SOURCE_UNIT_COUNT
    projected_lower_bound=sum_optimizer+projected_source if completed else None
    if projected_lower_bound is not None and not math.isfinite(projected_lower_bound):
        raise RuntimeError("R103_UNIT_PROBE_NONFINITE_COST_PROJECTION")
    return {
        "schema":"ttt-hybrid.r103.unit-only-cost-readiness.v1",
        "status":("R103_UNIT_ONLY_ALL_11_ARMS_TIMED" if completed
                  else "R103_UNIT_ONLY_120S_PROBE_LIMIT_PARTIAL"),
        "exact_source_sha":sha,"freeze_blob":r103.FREEZE_BLOB,
        "original_scored_source_opened":False,
        "original_scored_seed_consumed":False,
        "source_seed":None,
        "unit_only_salt":r103.UNIT_SALT,
        "original_pipeline_training_executed":False,
        "training_updates_per_arm_original":t["steps_per_arm"],
        "training_batch_size":t["batch_size"],
        "sampled_optimizer_steps_after_warmup":STEPS_RECORDED,
        "CPU_threads":torch.get_num_threads(),
        "torch_version":str(torch.__version__),
        "platform":platform.platform(),
        "unit_physical_episodes":SOURCE_UNIT_COUNT,
        "original_physical_episode_count_if_authorized":episode_count,
        "unit_physics_seconds":sampled_source_seconds,
        "projected_total_source_seconds_at_unit_rate":projected_source,
        "measured_arms":measured,
        "all_11_arms_measured":completed,
        "projected_total_optimizer_seconds_excludes_validation":
            sum_optimizer if completed else None,
        "rough_total_source_plus_optimizer_seconds":
            projected_lower_bound,
        "frozen_original_per_seed_total_CPU_wall_budget_seconds":
            t["per_seed_total_CPU_wall_budget_seconds"],
        "probe_elapsed_seconds":total_probe_seconds,
        "FLOPs_measured":False,
        "science_validation":"NOT_RUN_UNVERIFIED",
        "disclaimer":"Short UNIT_ONLY throughput extrapolation, not an original-run benchmark or science result. Excludes actual FIT/VALID passes, full scoring, source custody and residual CI overhead. Neither a pass guarantee nor a formal lower bound."
    }


def main():
    result=run_unit_only()
    RECEIPT.parent.mkdir(exist_ok=True,parents=True)
    if RECEIPT.exists():
        raise FileExistsError("R103_UNIT_PROBE_RECEIPT_ALREADY_WRITTEN")
    RECEIPT.write_text(
        json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("R103_UNIT_ONLY_CAPACITY_PROBE "+
          json.dumps({k:result.get(k) for k in
                     ("exact_source_sha","status","all_11_arms_measured",
                      "rough_total_source_plus_optimizer_seconds",
                      "frozen_original_per_seed_total_CPU_wall_budget_seconds",
                      "probe_elapsed_seconds","science_validation")},
                     sort_keys=True))
    return result


if __name__=="__main__":
    main()
