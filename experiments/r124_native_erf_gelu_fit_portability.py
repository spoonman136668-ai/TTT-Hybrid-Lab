"""R124 native-only explicit-erf GELU replacement: full original public FIT/VALID.

The same original 3409-parameter R116 NativePolicy, R118 canonical initial
weights, original R116 public data/minibatches/AdamW, and original R120
source-ordered clipping. Replace only the native GELU calculation with the
high-fidelity same-input R123 explicit torch.erf forward and autograd. This
is engineering; no original heldout/protected science or GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r115_query_targeted_diagnostic as r115
from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_shared_init_optimizer_diagnostic as r118
from experiments import r120_canonical_clip_portability_diagnostic as r120
from experiments import r122_every_fit_update_gradient_trace as r122
from experiments import r123_gelu_primitive_diagnostic as r123
from experiments import r124_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r124_native_erf_gelu_full_fit_valid_portability.json"
SEEDS=custody.SEEDS
STEPS=custody.UPDATES


class NativeExplicitErfGELU(nn.Module):
    """Exactly R123's prospectively measured efficient torch-erf expression."""
    def forward(self,x):
        if x.dtype!=torch.float32 or x.device.type!="cpu":
            raise RuntimeError("R124_NATIVE_EXPLICIT_ERF_REQUIRES_CPU_FP32")
        return (0.5*x)*(1.0+torch.erf(x/math.sqrt(2.0)))


def replace_only_native_gelu_(model):
    custody.audit_freeze()
    layers=model.scorer
    if (len(layers)!=5 or
        not isinstance(layers[0],nn.Linear) or
        not isinstance(layers[1],nn.GELU) or
        not isinstance(layers[2],nn.Linear) or
        not isinstance(layers[3],nn.GELU) or
        not isinstance(layers[4],nn.Linear)):
        raise PermissionError("R124_ORIGINAL_R116_NATIVE_ARCHITECTURE_DRIFT")
    original_params=[(name,tuple(p.shape)) for name,p in model.named_parameters()]
    if sum(p.numel() for p in model.parameters())!=3409:
        raise PermissionError("R124_ORIGINAL_R116_PARAMETER_COUNT_DRIFT")
    layers[1]=NativeExplicitErfGELU()
    layers[3]=NativeExplicitErfGELU()
    if ([(name,tuple(p.shape)) for name,p in model.named_parameters()]!=original_params or
        sum(p.numel() for p in model.parameters())!=3409):
        raise RuntimeError("R124_ERF_CHANGED_NATIVE_TRAINABLE_PARAMETERS")
    return model


def original_R123_primitive_behavior_audit(seed):
    """Only PUBLIC FIT actual first GELU inputs; original is never retrained."""
    observed,source,canonical_sha,indices=r123._original_public_native_inputs(seed)
    results={}
    for name,measurement in observed.items():
        x=measurement["input"].detach().to(torch.float32)
        t=x.clone().requires_grad_(True)
        actual=NativeExplicitErfGELU()(t)
        (grad,)=torch.autograd.grad(actual.sum(),t)
        ref_output,ref_grad=r123._torch_forward_grad(x,"torch_explicit_erf")
        if (not np.array_equal(actual.detach().cpu().numpy().view("<u4"),
                               ref_output.view("<u4")) or
            not np.array_equal(grad.detach().cpu().numpy().view("<u4"),
                               ref_grad.view("<u4"))):
            raise RuntimeError("R124_CUSTOM_NATIVE_ERF_DOES_NOT_MATCH_R123_PRIMITIVE")
        results[name]={
          "input_fp32_SHA":r123._tensor_bytes(name+"|input",x)["float32_le_SHA256"],
          "erf_output_exact_reference_SHA":r123._tensor_bytes(
              name+"|native_erf",actual)["float32_le_SHA256"],
          "erf_gradient_exact_reference_SHA":r123._tensor_bytes(
              name+"|native_erf_gradient",grad)["float32_le_SHA256"],
          "exact_R123_explicit_erf_comparison_pass":True
        }
    if set(results)!={"GELU1","GELU2"}:
        raise RuntimeError("R124_ORIGINAL_R123_SOURCE_LAYER_DRIFT")
    return {"original_FIT_source_sha":source,
            "original_canonical_native_initial_sha":canonical_sha,
            "public_first_indices":indices.tolist(),
            "GELU":results}


def _train_new_native(model,fit,batches):
    """Original FIT teacher, loss, optimizer and R120 canonical clip."""
    custody.allow("UNIT_FIT")
    if batches.shape!=(STEPS,16):
        raise PermissionError("R124_ORIGINAL_R116_512_X_16_FIT_BATCH_REQUIRED")
    n,m,g,target=fit
    started=time.perf_counter()
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    hashes=[r117.raw_fp32_sha256(model)]
    first_clip=None
    for step,ix in enumerate(batches,start=1):
        prediction=model(n[ix],m[ix],g[ix])
        if prediction.shape!=(16,2):
            raise RuntimeError("R124_R116_SAME_NATIVE_OUTPUT_SHAPE_DRIFT")
        loss=F.mse_loss(prediction,target[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R124_NEW_NATIVE_NONFINITE_FIT_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        clip=r120.canonical_clip_grad_norm_(model)
        if step==1:
            first_clip=clip
        opt.step()
        hashes.append(r117.raw_fp32_sha256(model))
    if (len(hashes)!=STEPS+1 or
        sum(p.numel() for p in model.parameters())!=3409):
        raise RuntimeError("R124_ORIGINAL_NATIVE_TRAINING_BUDGET_OR_ARCHITECTURE_DRIFT")
    return {
      "updates":STEPS,"parameters":3409,
      "raw_FP32_sha256_each_step_0_through_512":hashes,
      "first_step_original_R120_clip":first_clip,
      "train_seconds":time.perf_counter()-started
    }


def run_seed(seed):
    custody.allow("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R124_UNREGISTERED_FIT_SEED")
    t0=time.perf_counter()
    torch.set_num_threads(1)
    fit,batches,source,counts=r122._fit_and_batches(seed)
    models=r116.new_models(seed)
    canonical={}
    for arm in r116.TRAINABLE:
        initial_sha,n=r118.seed_shared_initial_bytes(models[arm],seed,arm)
        canonical[arm]={"initial_SHA":initial_sha,"n_parameters":n}
    comparison_native=r116.new_models(seed)[r116.NATIVE]
    sha_old,n_old=r118.seed_shared_initial_bytes(
        comparison_native,seed,r116.NATIVE)
    if sha_old!=canonical[r116.NATIVE]["initial_SHA"] or n_old!=3409:
        raise RuntimeError("R124_NEW_AND_OLD_NATIVE_NOT_IDENTICALLY_INITIALIZED")
    new_native=replace_only_native_gelu_(models[r116.NATIVE])
    if (r117.raw_fp32_sha256(new_native)!=
        r117.raw_fp32_sha256(comparison_native)):
        raise RuntimeError("R124_NATIVE_GELU_CHANGE_ALTERED_INITIAL_WEIGHT_BITS")
    primitive=original_R123_primitive_behavior_audit(seed)
    if primitive["original_FIT_source_sha"]!=source or (
        primitive["original_canonical_native_initial_sha"]!=
        canonical[r116.NATIVE]["initial_SHA"]):
        raise RuntimeError("R124_PRIMITIVE_ORIGINAL_SOURCE_CHANGED")
    native_train=_train_new_native(new_native,fit,batches)
    original_control=r120.train_original_native_with_stable_clipping(
        comparison_native,fit,batches)
    conventional={arm:r116.train_arm(models[arm],arm,fit,batches)
                  for arm in (r116.MLP,r116.GRU)}
    if (original_control["updates"]!=STEPS or
        any(z["updates"]!=STEPS for z in conventional.values()) or
        native_train["updates"]!=STEPS):
        raise PermissionError("R124_ALL_FOUR_SCRATCH_ARMS_MUST_TRAIN_512")
    original_weights,original_quant_sha=r115.quantize_native(comparison_native)
    new_weights,new_quant_sha=r115.quantize_native(new_native)
    cold_new=r115.integer_cold_restore(new_weights)
    cold_original=r115.integer_cold_restore(original_weights)
    if cold_new!=0 or cold_original!=0:
        raise RuntimeError("R124_INDEPENDENT_NATIVE_INTEGER_COLD_RESTORE_DRIFT")
    custody.allow("UNIT_VALID")
    new_val=r117.public_VALID_trial(seed,384,new_weights)
    old_val=r117.public_VALID_trial(seed,384,original_weights)
    f=custody.audit_freeze()["evaluation"]
    local_fidelity=(new_val["correct"]>=f["public_VALID_new_native_correct_min_per_seed"] and
                    new_val["correct"]>=old_val["correct"])
    return {
      "seed":seed,
      "canonical_initial_named_FP32":canonical,
      "public_FIT_source_and_batch_hash":source,
      "native_replaced_primitive_exact_R123_audit":primitive,
      "new_erf_native_training":native_train,
      "unchanged_original_GELU_native_control":original_control,
      "unchanged_original_GELU_final_raw_FP32_SHA":
         r117.raw_fp32_sha256(comparison_native),
      "unchanged_conventional_MLP_GRU_training":conventional,
      "new_native_final_Q384_weight_SHA256":new_quant_sha,
      "original_native_final_Q384_weight_SHA256":original_quant_sha,
      "new_native_original_R116_public_VALID":{
          "correct":new_val["correct"],
          "accuracy":new_val["accuracy"],
          "all_chosen_physical_actions_SHA256":new_val["all_chosen_actions_sha256"],
          "all_chosen_physical_outcome_signs_SHA256":
             new_val["all_physical_outcome_signs_sha256"],
          "public_query_correctness_SHA256":new_val["all_correctness_sha256"],
          "next_action_do14_count":new_val["next_actions_do14_count"],
          "all_public_fourth_action_choices":new_val["next_actions"]},
      "original_GELU_native_public_VALID":{
          "correct":old_val["correct"],
          "accuracy":old_val["accuracy"],
          "all_chosen_physical_actions_SHA256":old_val["all_chosen_actions_sha256"],
          "all_chosen_physical_outcome_signs_SHA256":old_val["all_physical_outcome_signs_sha256"],
          "public_query_correctness_SHA256":old_val["all_correctness_sha256"]},
      "local_frozen_accuracy_criterion_met":local_fidelity,
      "cold_new_native_integer_mismatch":cold_new,
      "cold_original_native_integer_mismatch":cold_original,
      "public_FIT_calibration":counts,
      "original_R116_scored_UNIT_UNSEEN_replayed":False,
      "scientific_pass":False,
      "whole_process_wall_seconds":time.perf_counter()-t0
    }


def run_all():
    custody.audit_freeze()
    exact_sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    required=os.getenv("EXPECTED_CANDIDATE_SHA")
    if required and required!=exact_sha:
        raise PermissionError("R124_EXACT_SOURCE_CANDIDATE_SHA_CHANGED")
    if (ROOT/"experiments/r124_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R124_PROTECTED_FIRSTUSE_NOT_AUTHORIZED")
    seeds={str(seed):run_seed(seed) for seed in SEEDS}
    all_local=all(x["local_frozen_accuracy_criterion_met"] for x in seeds.values())
    return {
      "schema":"ttt-hybrid.r124.native-explicit-erf-GELU-full-FIT-VALID-portability.v1",
      "exact_scored_engineering_SHA":exact_sha,
      "prospective_source_freeze_blob":custody.FREEZE_BLOB,
      "status":"R124_LOCAL_FIT_VALID_COMPLETE_CROSS_OS_ORIGINAL_ARTIFACT_COMPARISON_REQUIRED",
      "local_new_native_VALID_accuracy_all_seeds_pass":all_local,
      "per_seed":seeds,
      "cross_OS_native_training_portability_NOT_ASSUMED":True,
      "R116_original_spent_UNSEEN_replayed":False,
      "scientific_PASS":False,
      "GPU_firstuse":False,"merge":False,"model_promotion":False,"deployment":"HOLD"
    }


def main():
    receipt=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R124_ORIGINAL_FIT_VALID_EVIDENCE_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",
                       encoding="utf8")
    print("R124_NATIVE_ERF_FULL_FIT "+json.dumps({
        "sha":receipt["exact_scored_engineering_SHA"],
        "local_accuracy_pass":receipt["local_new_native_VALID_accuracy_all_seeds_pass"],
        "seeds":{seed:{
            "erf_initial":x["new_erf_native_training"]
                          ["raw_FP32_sha256_each_step_0_through_512"][0],
            "erf_first":x["new_erf_native_training"]
                          ["raw_FP32_sha256_each_step_0_through_512"][1],
            "erf_final":x["new_erf_native_training"]
                          ["raw_FP32_sha256_each_step_0_through_512"][-1],
            "erf_Q384":x["new_native_final_Q384_weight_SHA256"],
            "VALID_erf":x["new_native_original_R116_public_VALID"]["correct"],
            "VALID_original":x["original_GELU_native_public_VALID"]["correct"],
            "cold":x["cold_new_native_integer_mismatch"]}
          for seed,x in receipt["per_seed"].items()},
        "original_spent_science_replayed":False,
        "science_PASS":False
    },sort_keys=True))


if __name__=="__main__":
    main()
