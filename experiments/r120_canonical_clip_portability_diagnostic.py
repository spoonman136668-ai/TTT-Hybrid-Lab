"""R120 native one-operation intervention: fixed-order global gradient norm.

Original R116 native tokenless policy, original FIT teacher, scratch AdamW
512 updates and R118 canonical initial FP32 state retained exactly. The only
algorithmic change is the global L2 norm used to clip native gradients.
Nothing from spent UNIT_UNSEEN original research is reopened.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import r115_query_targeted_diagnostic as r115
from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r120_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r120_canonical_gradient_clip_full_fit_valid.json"
SEEDS=custody.SEEDS
SNAPSHOTS=custody.SNAPSHOTS


def canonical_clip_grad_norm_(model):
    """Single changed primitive: source-ordered FP32 gradients, F64 fsum.

    No torch.linalg.norm, reduction kernel, foreach, FMA or changed AdamW.
    Fixed gradient conversion, exact fsum order and once-rounded FP32 clip
    coefficient are required by the preimplementation R120 source freeze.
    """
    custody.audit_freeze()
    model_params=list(model.named_parameters())
    if len(model_params)!=6:
        raise RuntimeError("R120_ORIGINAL_NATIVE_PARAMETER_LAYOUT_DRIFT")
    raw=[]
    for name,p in model_params:
        if p.grad is None or p.grad.dtype!=torch.float32:
            raise RuntimeError("R120_ABSENT_OR_NON_FP32_RAW_GRADIENT")
        a=p.grad.detach().cpu().numpy()
        if not np.isfinite(a).all():
            raise RuntimeError("R120_NONFINITE_ACTUAL_RAW_GRADIENT")
        raw.extend(float(x) for x in a.ravel(order="C"))
    if len(raw)!=3409:
        raise RuntimeError("R120_NATIVE_PARAMETER_COUNT_DRIFT")
    sq=math.fsum(g*g for g in raw)
    norm=math.sqrt(sq)
    coefficient=float(np.float32(min(1.,1.0/(norm+1e-6))))
    if not math.isfinite(norm) or not math.isfinite(coefficient):
        raise RuntimeError("R120_NONFINITE_FLOAT64_NORM_OR_FP32_COEFFICIENT")
    with torch.no_grad():
        for _,p in model_params:
            p.grad.mul_(coefficient)
    return {
      "raw_grad_sum_squares_F64_hex":struct.pack("<d",sq).hex(),
      "raw_grad_norm_F64_hex":struct.pack("<d",norm).hex(),
      "clipping_multiplier_exact_FP32_hex":struct.pack("<f",coefficient).hex(),
      "scaled_gradient_norm_before_clipping":norm,
      "coefficient":coefficient,
      "parameters_processed":len(raw),
      "actual_gradient_update_applied":True
    }


def first_original_torch_clip_control(seed,fit,indices):
    model=r116.new_models(seed)[r116.NATIVE]
    r118.seed_shared_initial_bytes(model,seed,r116.NATIVE)
    n,m,g,target=fit
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train()
    v=model(n[indices],m[indices],g[indices])
    loss=F.mse_loss(v,target[indices])
    optim.zero_grad(set_to_none=True)
    loss.backward()
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
    optim.step()
    return {
      "original_torch_clip_returned_norm_FP32_hex":
         struct.pack("<f",float(norm)).hex(),
      "original_torch_native_after_first_optimizer_FP32_sha256":
         r117.raw_fp32_sha256(model),
      "original_torch_updates":1
    }


def train_original_native_with_stable_clipping(model,fit,draws):
    if draws.shape!=(512,16):
        raise PermissionError("R120_FROZEN_512_MINIBATCHES_REQUIRED")
    n,m,g,target=fit
    t0=time.perf_counter()
    optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    model.train()
    raw_snap={"0":r117.raw_fp32_sha256(model)}
    first_clip=None
    for update,indices in enumerate(draws,start=1):
        actual=model(n[indices],m[indices],g[indices])
        if actual.shape!=(16,2):
            raise RuntimeError("R120_R116_NATIVE_PUBLIC_INTERFACE_DRIFT")
        loss=F.mse_loss(actual,target[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R120_NONFINITE_PUBLIC_FIT_LOSS")
        optim.zero_grad(set_to_none=True)
        loss.backward()
        clip=canonical_clip_grad_norm_(model)
        if update==1:
            first_clip=clip
        optim.step()
        if update in SNAPSHOTS:
            raw_snap[str(update)]=r117.raw_fp32_sha256(model)
    if set(raw_snap)!={str(i) for i in SNAPSHOTS}:
        raise RuntimeError("R120_FIRST_AND_FULL_FIT_CHECKPOINTS_MISSING")
    return {
      "model_trainable_parameters":sum(p.numel() for p in model.parameters()),
      "updates":512,"first_clip":first_clip,
      "native_raw_FP32_hash_by_update":raw_snap,
      "train_seconds":time.perf_counter()-t0
    }


def run_seed(seed):
    custody.allow_public("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R120_FIT_SEED_NOT_REGISTERED")
    t0=time.perf_counter()
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(seed)
    gen=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=gen.integers(0,len(fit[-1]),size=(512,16))
    models=r116.new_models(seed)
    initial={}
    for arm in r116.TRAINABLE:
        sha,n=r118.seed_shared_initial_bytes(models[arm],seed,arm)
        initial[arm]={"sha256":sha,"parameters":n}
    first_original=first_original_torch_clip_control(seed,fit,batches[0])
    native=train_original_native_with_stable_clipping(models[r116.NATIVE],fit,batches)
    conventional={arm:r116.train_arm(models[arm],arm,fit,batches)
                  for arm in (r116.MLP,r116.GRU)}
    if (set(conventional)!={r116.MLP,r116.GRU} or
        any(r["updates"]!=512 for r in conventional.values()) or
        native["model_trainable_parameters"]!=3409 or
        native["updates"]!=512):
        raise PermissionError("R120_STRONG_CONTROL_UPDATE_PARITY_FAIL")
    layers,quantized_sha=r115.quantize_native(models[r116.NATIVE])
    native_cold=r115.integer_cold_restore(layers)
    custody.allow_public("UNIT_VALID")
    valid=r117.public_VALID_trial(seed,384,layers)
    if native_cold!=0:
        raise RuntimeError("R120_COLD_NATIVE_INTEGER_SCORE_DRIFT")
    return {
      "seed":seed,
      "canonical_initial_FP32_by_arm":initial,
      "native_original_torch_clip_negative_control":first_original,
      "new_native_training":native,
      "conventional_unchanged_training":conventional,
      "conventional_after_512_raw_FP32":{
        arm:r118._parameter_bytes(models[arm])[0]
        for arm in (r116.MLP,r116.GRU)},
      "native_Q384_trained_integer_weight_sha256":quantized_sha,
      "native_post512_raw_FP32_sha256":
         native["native_raw_FP32_hash_by_update"]["512"],
      "native_independent_integer_cold_mismatches":native_cold,
      "actual_public_VALID":{
        "correct":valid["correct"],
        "accuracy":valid["accuracy"],
        "action_history_sha256":valid["all_chosen_actions_sha256"],
        "observed_signed_effect_SHA256":valid["all_physical_outcome_signs_sha256"],
        "correctness_SHA256":valid["all_correctness_sha256"],
        "chosen_do14_count":valid["next_actions_do14_count"],
        "actual_offered_next_action_decisions":valid["next_actions"]},
      "public_FIT_diagnostic_signed_counts":counts,
      "scored_R116_original_UNIT_UNSEEN_used":False,
      "scientific_pass":False,
      "whole_process_wall_seconds":time.perf_counter()-t0
    }


def run_all():
    custody.audit_freeze()
    if (ROOT/"experiments/r120_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R120_PROTECTED_ORIGINAL_SCIENTIFIC_FIRSTUSE_DENIED")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
         capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=sha:
        raise PermissionError("R120_EXACT_SOURCE_SHA_DRIFT")
    seeds={str(seed):run_seed(seed) for seed in SEEDS}
    return {
       "schema":"ttt-hybrid.r120.exact-ordered-clip-512step-original-native-FIT-VALID.v1",
       "exact_source_SHA":sha,
       "preimplementation_freeze_blob":custody.FREEZE_BLOB,
       "status":"R120_LOCAL_FIT_VALID_SCORED_AWAIT_DUAL_OS_ORIGINAL_ARTIFACT_PARITY",
       "by_seed":seeds,
       "crossOS_native_training_parity_NOT_ASSUMED":True,
       "protected_original_scientific_firstuse":False,
       "spent_R116_UNSEEN_replayed":False,
       "independent_scientific_PASS":False,
       "model_promotion":False,
       "deployment":"HOLD"
    }


def main():
    report=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R120_FIT_VALID_ORIGINAL_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",
                       encoding="utf8")
    print("R120_FIT_VALID "+json.dumps({
       "source":report["exact_source_SHA"],
       "seeds":{k:{
           "FP32_update1":v["new_native_training"]["native_raw_FP32_hash_by_update"]["1"],
           "FP32_update512":v["native_post512_raw_FP32_sha256"],
           "Q384":v["native_Q384_trained_integer_weight_sha256"],
           "VALID_correct":v["actual_public_VALID"]["correct"],
           "clipping_coefficient_fp32":v["new_native_training"]["first_clip"]["clipping_multiplier_exact_FP32_hex"],
           "cold":v["native_independent_integer_cold_mismatches"]
       } for k,v in report["by_seed"].items()},
       "scientific_PASS":False,"heldout_replay":False
    },sort_keys=True))


if __name__=="__main__":
    main()
