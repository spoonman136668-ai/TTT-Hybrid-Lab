"""R117: FIT/VALID-only native optimizer and quantization reproducibility.

Diagnostic, not a rerun of R116 science. Trains R116's unchanged FIT
architectures from scratch, snapshots raw float weights during optimizer
updates, compares preregistered fixed-point scales and scores only public
VALID worlds. No UNIT_UNSEEN source is generated or accessed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r112_portable_probe_diagnostic as r112
from experiments import r115_query_targeted_diagnostic as r115
from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r116_pair_source as source
from experiments import r117_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r117_fit_valid_native_weight_portability.json"
SEEDS=custody.SEEDS
SCALES=custody.SCALES
CHECKPOINTS=custody.CHECKPOINTS
LAYERS=(45,48,24,1)


def _native_layers(model):
    a=(model.scorer[0],model.scorer[2],model.scorer[4])
    if len(a)!=3 or not all(isinstance(z,nn.Linear) for z in a):
        raise RuntimeError("R117_ORIGINAL_NATIVE_ARCHITECTURE_DRIFT")
    return a


def raw_fp32_sha256(model):
    h=hashlib.sha256(b"R117_RAW_FP32_LE_NATIVE_45_48_24_1_V1")
    for layer in _native_layers(model):
        for name in ("weight","bias"):
            param=getattr(layer,name).detach().cpu().numpy()
            if not np.isfinite(param).all():
                raise RuntimeError("R117_NONFINITE_NATIVE_RAW_MODEL")
            if param.dtype!=np.float32:
                raise RuntimeError("R117_RAW_FLOAT32_DTYPE_DRIFT")
            packed=param.astype("<f4",copy=False)
            h.update(name.encode());h.update(struct.pack("<I",packed.ndim))
            for d in packed.shape:h.update(struct.pack("<I",d))
            h.update(packed.tobytes(order="C"))
    return h.hexdigest()


def original_native_fit_train(model,fit,draws):
    """Exact original 512-update optimizer semantics; record snapshots."""
    if draws.shape!=(512,16):
        raise PermissionError("R117_FROZEN_ORIGINAL_MINIBATCH_BUDGET")
    t=custody.audit_freeze()["source"]
    if t["trainer_identical_to_R116"] is not True:
        raise PermissionError("R117_ORIGINAL_TRAINING_METHOD_CHANGED")
    n,m,g,target=fit
    t0=time.perf_counter()
    model.train()
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    hashes={"0":raw_fp32_sha256(model)}
    for update,indices in enumerate(draws,start=1):
        estimate=model(n[indices],m[indices],g[indices])
        if estimate.shape!=(16,2):
            raise RuntimeError("R117_R116_TRAINING_OUTPUT_DRIFT")
        loss=F.mse_loss(estimate,target[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R117_NONFINITE_TRAINING_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1)
        opt.step()
        if update in CHECKPOINTS:
            hashes[str(update)]=raw_fp32_sha256(model)
    if set(hashes)!={str(x) for x in CHECKPOINTS}:
        raise RuntimeError("R117_TRAINING_SNAPSHOT_BUDGET_DRIFT")
    return {"updates":512,"parameters":sum(p.numel() for p in model.parameters()),
            "FIT_only_training_seconds":time.perf_counter()-t0},hashes


def quantized_layers(model,scale):
    if scale not in SCALES:
        raise PermissionError("R117_UNFROZEN_INTEGER_SCALE")
    h=hashlib.sha256(f"R117_NATIVE_45_48_24_1_Q{scale}_V1".encode())
    layers=[];flattened=[];margins=[]
    for layer in _native_layers(model):
        parts=[]
        for name in ("weight","bias"):
            p=getattr(layer,name).detach().cpu().numpy().astype(np.float64)
            if not np.isfinite(p).all():
                raise RuntimeError("R117_NONFINITE_QUANTIZATION_SOURCE")
            values=np.rint(p*scale).astype("<i8")
            if np.any(np.abs(values)>2**20):
                raise OverflowError("R117_TRAINED_INTEGER_PARAMETER_RANGE")
            h.update(name.encode());h.update(struct.pack("<I",values.ndim))
            for dim in values.shape:h.update(struct.pack("<I",dim))
            h.update(values.tobytes(order="C"))
            parts.append(values.tolist())
            flattened.extend(int(x) for x in values.flat)
            delta=np.abs(p*scale-np.floor(p*scale)-.5)
            margins.extend(float(x) for x in delta.flat)
        layers.append({"weight":parts[0],"bias":parts[1]})
    return layers,h.hexdigest(),flattened,{
        "nearest_half_integer_margin":min(margins),
        "within_1e_minus4_of_half_integer":sum(float(v)<1e-4 for v in margins),
        "within_1e_minus3_of_half_integer":sum(float(v)<1e-3 for v in margins)}


def scaled_native_scores(layers,scale,history,*,offers=(12,14),query=15,phase=0):
    """All posttraining inference int64, including feature requantization."""
    if (scale not in SCALES or len(layers)!=3 or tuple(offers)!=(12,14)
        or type(query) is not int or not 0<=query<16
        or phase!=0):
        raise PermissionError("R117_UNKNOWN_PUBLIC_INTEGER_INTERFACE")
    canonical=r115.native_integer_features(history,offers,query,phase)
    z=r112._rdiv(canonical.astype(np.int64)*scale,384)
    for i,layer in enumerate(layers):
        w=np.asarray(layer["weight"],dtype=np.int64)
        b=np.asarray(layer["bias"],dtype=np.int64)
        if w.shape!=(LAYERS[i+1],LAYERS[i]) or b.shape!=(LAYERS[i+1],):
            raise RuntimeError("R117_UNQUALIFIED_INTEGER_LAYER_SHAPE")
        z=r112._rdiv(z@w.T,scale)+b[None,:]
        if bool(np.any(np.abs(z)>2**25)):
            raise OverflowError("R117_TRAINED_INTEGER_INFERENCE_RANGE")
        if i<2:
            z=r112._rdiv(z*np.clip(z+2*scale,0,4*scale),4*scale)
    if z.shape!=(2,1):
        raise RuntimeError("R117_NATIVE_INTEGER_SCORE_OUTPUT_SHAPE")
    return int(z[0,0]),int(z[1,0])


def cold_original_integer_score_check(all_layers):
    """One new process checks all prenominated scales against same public facts."""
    samples=[
      ([(8,1),(9,-1),(7,-1)],15),
      ([(8,1),(9,-1),(7,1)],15),
      ([(8,1),(9,-1),(7,-1),(12,1)],15),
      ([(8,1),(9,-1),(7,1),(14,-1)],15)]
    before={str(q):[scaled_native_scores(layers,q,h,query=query)
                   for h,query in samples]
            for q,layers in all_layers.items()}
    payload={"weights":{str(k):v for k,v in all_layers.items()},"samples":samples}
    code=(
       "import json,sys\n"
       "from experiments.r117_native_weight_portability_diagnostic import scaled_native_scores\n"
       "d=json.load(sys.stdin)\n"
       "scores={k:[scaled_native_scores(v,int(k),[tuple(x) for x in hist],query=q)"
       " for hist,q in d['samples']] for k,v in d['weights'].items()}\n"
       "print(json.dumps(scores,sort_keys=True))\n")
    run=subprocess.run([sys.executable,"-c",code],
         input=json.dumps(payload).encode(),cwd=ROOT,capture_output=True,check=True,timeout=30)
    after=json.loads(run.stdout)
    return sum(tuple(x)!=tuple(y)
               for key in before
               for x,y in zip(before[key],after[key]))


def public_VALID_trial(seed,scale,layers):
    custody.permit_public_panel("UNIT_VALID")
    actions=[];effects=[];queries=[];correctness=[];positive=0
    for idx in range(96):
        w=source.PhysicalPairWorld("UNIT_VALID",seed,idx)
        for action in (8,9,7):
            if w.offers()!=(action,):
                raise RuntimeError("R117_ORIGINAL_CALIBRATION_OFFER_DRIFT")
            signed=w.intervene(action)
            actions.append(action);effects.append(int(signed>0))
        q=w.public_query
        if q!=15 or w.offers()!=(12,14):
            raise RuntimeError("R117_PUBLIC_VALID_QUERY_DRIFT")
        s0,s1=scaled_native_scores(layers,scale,w.history,query=q)
        next_action=14 if s1>s0 else 12
        signed=w.intervene(next_action)
        actions.append(next_action);effects.append(int(signed>0))
        query,truth=w.sealed_query()
        predicted,_=source.public_exact_prediction(w.history)
        positive+=int(predicted==1)
        queries.append(next_action)
        correctness.append(int(predicted==truth))
    if len(actions)!=96*4 or len(effects)!=96*4 or len(queries)!=96:
        raise RuntimeError("R117_PHYSICAL_ACTION_BUDGET_CHANGED")
    return {"correct":sum(correctness),"accuracy":sum(correctness)/96,
            "positive_predictions":positive,
            "next_actions_do14_count":sum(x==14 for x in queries),
            "next_actions":queries,
            "all_chosen_actions_sha256":hashlib.sha256(bytes(actions)).hexdigest(),
            "all_physical_outcome_signs_sha256":hashlib.sha256(bytes(effects)).hexdigest(),
            "all_correctness_sha256":hashlib.sha256(bytes(correctness)).hexdigest()}


def run_seed(seed):
    custody.audit_freeze()
    if seed not in SEEDS:
        raise PermissionError("R117_UNREGISTERED_FIT_SOURCE_SEED")
    torch.set_num_threads(1)
    started=time.perf_counter()
    # This path only calls R116's FIT physical simulator; no UNIT_UNSEEN.
    fit,calibration=r116.make_fit(seed)
    gen=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    batches=gen.integers(0,len(fit[-1]),size=(512,16))
    models=r116.new_models(seed)
    native_training,snapshots=original_native_fit_train(models[r116.NATIVE],fit,batches)
    conventional={arm:r116.train_arm(models[arm],arm,fit,batches)
                  for arm in (r116.MLP,r116.GRU)}
    training={r116.NATIVE:native_training,**conventional}
    if (any(v["updates"]!=512 for v in training.values()) or
        native_training["parameters"]!=3409):
        raise PermissionError("R117_FROZEN_TRAINABLE_BUDGET_MISMATCH")
    original_layers,original_Q384_sha=r115.quantize_native(models[r116.NATIVE])
    # Diagnose possible observation effects without touching scored original:
    # create a separate same-seed native model and invoke the exact original
    # R116 trainer, preserving its original public FIT inputs/minibatches.
    original_reference_models=r116.new_models(seed)
    original_reference_native=original_reference_models[r116.NATIVE]
    initial_original_reference_raw=raw_fp32_sha256(original_reference_native)
    original_reference_train=r116.train_arm(
        original_reference_native,r116.NATIVE,fit,batches)
    reference_R116_SHA=r115.quantize_native(original_reference_native)[1]
    osname=platform.system().lower()
    if osname not in ("linux","windows"):
        raise PermissionError("R117_UNSUPPORTED_CROSS_PLATFORM_OWNER")
    prior=custody.audit_freeze()["source"]["provenance_R116_original_quantized_sha256"]
    expected=prior[str(seed)][osname]
    original_source_reproduced=original_Q384_sha==expected
    # Fail closed *before* generating permitted VALID when original source
    # optimizer/backend is inconsistent with the qualified historical source.
    if not original_source_reproduced:
        raise RuntimeError(
            "R117_R116_ORIGINAL_FIT_ONLY_QUANTIZED_HASH_NOT_REPRODUCED"
            +f" seed={seed} expected={expected}"
            +f" observed_trainer={original_Q384_sha}"
            +f" exact_original_trainer={reference_R116_SHA}"
            +f" original_reference_initial_raw={initial_original_reference_raw}"
            +f" instrumented_initial_raw={snapshots['0']}"
            +f" instrumented_final_raw={snapshots['512']}"
            +f" original_updates={original_reference_train['updates']}")
    baseline=public_VALID_trial(seed,384,original_layers)
    scales={};layers_for_restore={}
    tolerance=custody.audit_freeze()["diagnostics"]
    for scale in SCALES:
        layers,sha,flattened,margin=quantized_layers(models[r116.NATIVE],scale)
        candidate=public_VALID_trial(seed,scale,layers)
        divergence=sum(x!=y for x,y in zip(
            baseline["next_actions"],candidate["next_actions"]))
        local_action_fidelity=1.-divergence/96
        accuracy_drop=baseline["accuracy"]-candidate["accuracy"]
        fit_local=(local_action_fidelity>=tolerance["fidelity_min_per_seed"]
                   and accuracy_drop<=
                   tolerance["VALID_correct_drop_from_original_R116_Q384_max_per_seed"])
        scales[str(scale)]={
            "scale":scale,"integer_weight_sha256":sha,
            "Q384_all_integer_parameters_for_exact_OS_coordinate_diff":
                flattened if scale==384 else None,
            "numeric_rounding_margin":margin,
            "public_VALID":candidate,
            "action_fidelity_to_unchanged_R116_Q384":local_action_fidelity,
            "count_actions_different_from_R116_Q384":divergence,
            "accuracy_drop_from_unchanged_R116_Q384":accuracy_drop,
            "local_fidelity_pass":fit_local}
        layers_for_restore[scale]=layers
    cold=cold_original_integer_score_check(layers_for_restore)
    if cold!=0:
        raise RuntimeError("R117_PORTABLE_INTEGER_COLD_SCORE_DRIFT")
    # Keep public validation available only in sealed CI receipt. Unselected
    # scale means no portability success until independent dual-OS comparison.
    return {"seed":seed,"original_R116_Q384_SHA256":original_Q384_sha,
            "R116_original_FIT_training_reproduced_on_this_OS":original_source_reproduced,
            "native_raw_FP32_sha256_by_update":snapshots,
            "R116_same_three_scratch_FIT_training":training,
            "FIT_diagnostic_signed_counts":calibration,
            "baseline_original_Q384_public_VALID":baseline,
            "candidates":scales,"cold_independent_integer_score_mismatches":cold,
            "whole_process_wall_seconds":time.perf_counter()-started,
            "original_spent_R116_UNIT_UNSEEN_replayed":False,
            "scientific_pass":False}


def run_all():
    custody.audit_freeze()
    if (ROOT/"experiments/r117_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R117_ORIGINAL_PROTECTED_FIRST_USE_DENIED")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,check=True,text=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and sha!=expected:
        raise PermissionError("R117_EXACT_CANDIDATE_SHA_DRIFT")
    result={str(seed):run_seed(seed) for seed in SEEDS}
    return {"schema":"ttt-hybrid.r117.FIT_VALID_optimizer_quantization_diagnostic.v1",
            "exact_scored_engineering_source_sha":sha,
            "preimplementation_freeze_blob":custody.FREEZE_BLOB,
            "status":"R117_FIT_VALID_LOCAL_DIAGNOSTIC_COMPLETE_NEEDS_DUAL_OS_EXACT_REVIEW",
            "cross_OS_candidate_qualification_not_assumed":True,
            "by_seed":result,"original_R116_spent_heldout_reused":False,
            "protected_science_firstuse":False,
            "scientific_pass":False,"promotion":False,"deployment":"HOLD"}


def main():
    report=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R117_IMMUTABLE_ORIGINAL_FIT_VALID_RECEIPT_OVERWRITE_DENIED")
    RECEIPT.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R117_FIT_VALID "+json.dumps({
        "exact_sha":report["exact_scored_engineering_source_sha"],
        "status":report["status"],
        "seeds":{k:{
            "R116_native_reproduced":v["R116_original_FIT_training_reproduced_on_this_OS"],
            "original_Q384_sha":v["original_R116_Q384_SHA256"],
            "local_eligible_scales":[int(s) for s,c in v["candidates"].items()
                                       if c["local_fidelity_pass"]],
            "q384_rounding_margin":v["candidates"]["384"]["numeric_rounding_margin"],
            "raw_initial_SHA":v["native_raw_FP32_sha256_by_update"]["0"],
            "raw_final_SHA":v["native_raw_FP32_sha256_by_update"]["512"],
            "cold":v["cold_independent_integer_score_mismatches"]}
              for k,v in report["by_seed"].items()},
        "heldout_reuse":False,"scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
