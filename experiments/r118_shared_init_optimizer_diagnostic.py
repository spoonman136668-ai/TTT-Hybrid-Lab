"""R118 canonical shared model initial state, then original R116 FIT optimizer.

Development-only probe of initial-state and post-initialization native
cross-platform numerical reproducibility. Only public FIT and VALID sources.
No prior UNIT_UNSEEN world, scientific first-use, GPU or promotion.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import torch

from experiments import r116_feedback_probe_diagnostic as r116
from experiments import r117_native_weight_portability_diagnostic as r117
from experiments import r118_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r118_shared_canonical_init_optimizer_fit_valid.json"
SEEDS=custody.SEEDS
SCALES=custody.SCALES
ARMS=r116.TRAINABLE


def _parameter_bytes(model):
    """Canonical dtype, name, dims and FP32 LE payload, independent of OS."""
    h=hashlib.sha256(b"R118_ALL_PARAM_FP32_CANONICAL_V1")
    count=0
    for name,param in model.named_parameters():
        a=param.detach().cpu().numpy()
        if a.dtype!=np.float32 or not np.isfinite(a).all():
            raise RuntimeError("R118_NONFINITE_OR_NONFLOAT32_WEIGHT")
        little=a.astype("<f4",copy=False)
        h.update(name.encode("utf8"));h.update(b"\0")
        h.update(struct.pack("<I",little.ndim))
        for dim in little.shape:h.update(struct.pack("<I",dim))
        h.update(little.tobytes(order="C"))
        count+=little.size
    return h.hexdigest(),count


def _pure_canonical(seed,arm,name,i):
    key=f"R118|V1|{seed}|{arm}|{name}|{i}".encode("utf8")
    d=hashlib.sha256(key).digest()
    return (int.from_bytes(d[:2],"little")-32768)/262144


def seed_shared_initial_bytes(model,seed,arm):
    """Every FP32 number is exact signed int / 2**18; no torch RNG accepted."""
    custody.audit_freeze()
    if seed not in SEEDS or arm not in ARMS:
        raise PermissionError("R118_INVALID_CANONICAL_INITIAL_MODEL")
    with torch.no_grad():
        for name,p in model.named_parameters():
            if p.dtype!=torch.float32 or p.device.type!="cpu":
                raise RuntimeError("R118_NO_CUDA_OR_DIFFERENT_INITIAL_DTYPE")
            values=[_pure_canonical(seed,arm,name,i) for i in range(p.numel())]
            x=torch.tensor(values,dtype=torch.float32).reshape(p.shape)
            p.copy_(x)
            # Independent byte check verifies the original Python arithmetic
            # is exactly FP32-representable and was not altered by a backend.
            out=p.detach().cpu().numpy().ravel()
            if any(float(out[i])!=values[i] for i in range(len(values))):
                raise RuntimeError("R118_INITIAL_FLOAT32_GRID_INEXACT")
    return _parameter_bytes(model)


def run_seed(seed):
    custody.allow("UNIT_FIT")
    if seed not in SEEDS:
        raise PermissionError("R118_SEED_NOT_FROZEN")
    t0=time.perf_counter()
    torch.set_num_threads(1)
    fit,counts=r116.make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([27601119,seed,116]))
    draws=rng.integers(0,256,size=(512,16))
    models=r116.new_models(seed)
    original_pre_replace={a:_parameter_bytes(models[a])[0] for a in ARMS}
    initial={}
    for arm in ARMS:
        sha,count=seed_shared_initial_bytes(models[arm],seed,arm)
        initial[arm]={"raw_FP32_sha256":sha,"parameter_count":count}
    if set(initial)!=set(ARMS) or any(x["parameter_count"]<100
                                    for x in initial.values()):
        raise RuntimeError("R118_CANONICAL_INITIAL_MODEL_INCOMPLETE")
    # Native path exactly preserves R116 optimizer, clipping, FIT target,
    # batch ordering and 512 updates. Snapshots are read-only.
    trained_native,snaps=r117.original_native_fit_train(models[r116.NATIVE],fit,draws)
    other={a:r116.train_arm(models[a],a,fit,draws)
           for a in (r116.MLP,r116.GRU)}
    trained={r116.NATIVE:trained_native,**other}
    if (set(trained)!=set(ARMS)
        or any(x["updates"]!=512 for x in trained.values())
        or set(snaps)!={str(i) for i in custody.CHECKPOINTS}
        or trained_native["parameters"]!=3409):
        raise RuntimeError("R118_R116_FIT_OPTIMIZER_OR_STEP_BUDGET_DRIFT")
    after={arm:_parameter_bytes(models[arm])[0] for arm in ARMS}
    scales={};layers_for_cold={}
    for q in SCALES:
        layers,weight_hash,ints,margins=r117.quantized_layers(
            models[r116.NATIVE],q)
        # This invokes only the public VALID simulator, no UNIT_UNSEEN.
        valid=r117.public_VALID_trial(seed,q,layers)
        scales[str(q)]={
            "integer_model_SHA256":weight_hash,
            "rounding_margin":margins,
            "full_public_VALID_actions_sha256":valid["all_chosen_actions_sha256"],
            "full_public_VALID_outcome_sha256":valid["all_physical_outcome_signs_sha256"],
            "full_public_VALID_correctness_sha256":valid["all_correctness_sha256"],
            "public_VALID_correct":valid["correct"],
            "public_VALID_correct_accuracy":valid["accuracy"],
            "public_VALID_do14_choice_count":valid["next_actions_do14_count"],
            "all_PUBLIC_VALID_fourth_actions":valid["next_actions"]
        }
        layers_for_cold[q]=layers
    cold=r117.cold_original_integer_score_check(layers_for_cold)
    if cold!=0:
        raise RuntimeError("R118_FROZEN_INTEGER_COLD_RELOAD_DRIFT")
    if snaps["0"]!=r117.raw_fp32_sha256(
            _canonical_native_check_against_initial(seed)):
        raise RuntimeError("R118_INIT_SNAPSHOT_NOT_CANONICAL")
    return {
        "seed":seed,
        "original_torch_init_FP32_digest":original_pre_replace,
        "canonical_initial_by_arm":initial,
        "native_raw_FP32_snapshots":snaps,
        "post512_raw_FP32_by_arm":after,
        "training":trained,
        "public_FIT_calibration_counts":counts,
        "fixed_precision_PUBLIC_VALID":scales,
        "cold_integer_independent_subprocess_score_mismatches":cold,
        "original_R116_scored_UNIT_UNSEEN_generated":False,
        "protected_science_firstuse":False,
        "process_wall_seconds":time.perf_counter()-t0
    }


def _canonical_native_check_against_initial(seed):
    # Regenerate a model with a DIFFERENT underlying torch RNG, then
    # independently apply the canonical recipe to prove randomness-independent
    # initial source custody (no original heldout / training performed).
    torch.manual_seed(seed+181181)
    model=r116.r115.NativePolicy()
    seed_shared_initial_bytes(model,seed,r116.NATIVE)
    return model


def run_all():
    custody.audit_freeze()
    if (ROOT/"experiments/r118_protected_scientific_firstuse_authorization.json").exists():
        raise PermissionError("R118_ORIGINAL_SCIENTIFIC_FIRSTUSE_NOT_ALLOWED")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=sha:
        raise PermissionError("R118_EXACT_SOURCE_SHA_DRIFT")
    results={str(seed):run_seed(seed) for seed in SEEDS}
    return {
        "schema":"ttt-hybrid.r118.same-FP32-initial-state-original-AdamW-FIT_VALID.v1",
        "source_exact_SHA":sha,
        "prospective_freeze_git_blob":custody.FREEZE_BLOB,
        "local_status":"R118_ALL_FIT_VALID_DIAGNOSTICS_COMPLETE_NEEDS_ORIGINAL_CROSS_OS_RECEIPTS",
        "cross_OS_model_portability_PASSED":False,
        "by_seed":results,
        "original_R116_scored_UNSEEN_replayed":False,
        "independent_scientific_PASS":False,
        "GPU_firstuse":False,
        "deployment":"HOLD",
        "merge_or_promotion":False
    }


def main():
    receipt=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R118_ORIGINAL_FIT_VALID_RECEIPT_REPLAY_DENIED")
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",
                       encoding="utf8")
    print("R118_FIT_VALID "+json.dumps({
        "sha":receipt["source_exact_SHA"],
        "status":receipt["local_status"],
        "seeds":{seed:{
            "init_native":result["canonical_initial_by_arm"][r116.NATIVE]["raw_FP32_sha256"],
            "initial_native_snapshot":result["native_raw_FP32_snapshots"]["0"],
            "post_native_FP32":result["post512_raw_FP32_by_arm"][r116.NATIVE],
            "Q384_integer_sha":result["fixed_precision_PUBLIC_VALID"]["384"]["integer_model_SHA256"],
            "Q384_PUBLIC_VALID_correct":result["fixed_precision_PUBLIC_VALID"]["384"]["public_VALID_correct"],
            "cold_mismatches":result["cold_integer_independent_subprocess_score_mismatches"]
        }for seed,result in receipt["by_seed"].items()},
        "original_R116_UNSEEN_replayed":False,
        "scientific_PASS":False
    },sort_keys=True))


if __name__=="__main__":
    main()
