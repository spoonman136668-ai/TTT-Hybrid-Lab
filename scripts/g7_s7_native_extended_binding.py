"""G7-S7: longer native Mamba2+TTT test against parameter-matched static hybrid.

S6 conventional retrieval remains a separate positive control, not a matched
FLOPs/model-size comparison. All hypotheses, thresholds and resource limits
are frozen in experiments/g7_s7_frozen_extended_native.json.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody
import g7_s1_online_learning as prior

class Blocked(Exception):
    pass

PHASE = "NOT_STARTED"

def check(test, why):
    if not test:
        raise Blocked(why)

def set_phase(value):
    global PHASE
    PHASE = value

def state_hash(model):
    return prior.snapshot_digest(model)

def new_hybrid(torch, official, arm, seed):
    from authentic_hybrid import AuthenticHybrid
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = AuthenticHybrid(
        official_classes=official,
        width=64, heads=4, mini_batch=8,
        mamba_mem_eff_path=False, use_ttt=True, use_depth=True,
    )
    with torch.no_grad():
        model.depth.router.weight.zero_()
        model.depth.router.bias.fill_(1.0)
    for p in model.depth.router.parameters():
        p.requires_grad_(False)
    fingerprint = state_hash(model)
    model.ttt.config.ttt_base_lr = arm
    return model, fingerprint

def frozen_validate(spec):
    check(spec.get("schema") == "ttt-hybrid.g7-s7.extended-native-binding.v1",
          "FROZEN_SCHEMA_DRIFT")
    t, v, limit = spec["train"], spec["heldout"], spec["budget"]
    check(t["seeds"] == [101, 211, 307] and
          t["steps_per_arm_per_seed"] == 128 and
          t["batch_size"] == 8 and
          t["learning_rate"] == 0.002 and
          t["train_episodes_per_seed_per_arm"] == 1024 and
          v["episodes_per_seed"] == 24 and
          v["examples_per_episode"] == 2 and
          limit["torch_peak_allocated_bytes_max"] == 1073741824 and
          limit["wall_seconds_max"] == 720 and
          [a["ttt_base_lr"] for a in spec["arms"]] == [1,0] and
          spec["thresholds"]["online_correct_mse_max"] == 0.95 and
          spec["thresholds"]["relative_online_vs_static_reduction_min"] == 0.05,
          "FROZEN_PARAMETERS_DRIFT")
    check(spec["lineage"]["s6_positive_baseline_run"] == 42368703057
          and spec["lineage"]["s5_native_diagnostic_run"] == 42367320764,
          "FROZEN_LINEAGE_DRIFT")

def check_environment(torch, spec, args):
    set_phase("SOURCE_AND_RUNTIME_CUSTODY")
    check(os.geteuid() == 1000 and "microsoft" in
          Path("/proc/sys/kernel/osrelease").read_text().lower(),
          "NORMAL_WSL_UID_REQUIRED")
    check(custody.git(ROOT, "rev-parse", "HEAD") == args.expected_sha
          and not custody.git(ROOT, "status", "--porcelain",
                              "--untracked-files=all"),
          "SOURCE_DIRTY_OR_SHA_DRIFT")
    check(custody.sha256(ROOT/"source_pins.json") ==
          custody.R3_PINS_SHA256, "PINNED_SOURCE_SPEC_DRIFT")
    root = (Path.home().resolve()/".local/share/ttt-hybrid"/
            ("g7-r3-"+custody.R3_SOURCE_SHA[:12]))
    manifest_hash, packages = custody.manifest()
    check(manifest_hash == custody.R3_MANIFEST_SHA256 and
          packages.get("torch") == custody.TORCH and
          packages.get("triton") == custody.TRITON,
          "R3_PACKAGE_MANIFEST_DRIFT")
    pins = json.loads((ROOT/"source_pins.json").read_text())
    for label in ("mamba2","ttt_linear"):
        custody.source_custody(root/"sources"/label,pins["sources"][label])
    sys.path[:0] = [str(root/"sources"/"mamba2"),
                    str(root/"sources"/"ttt_linear")]
    check(sys.version_info[:2] == (3,12) and
          torch.__version__ == custody.TORCH and
          torch.version.cuda == "12.8" and
          torch.cuda.is_available() and
          torch.cuda.device_count() == 1 and
          tuple(torch.cuda.get_device_capability(0)) == (12,0),
          "GPU_CUDA_TORCH_IDENTITY_DRIFT")
    free,_ = torch.cuda.mem_get_info()
    check(free >= spec["budget"]["required_free_cuda_bytes_min"],
          "INSUFFICIENT_FREE_GPU_MEMORY")
    check(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8" and
          os.environ.get("LIBRARY_PATH") == "/usr/lib/wsl/lib" and
          "python3.12" in os.environ.get("CPATH",""),
          "PINNED_PROCESS_LOCAL_CUDA_ENVIRONMENT_DRIFT")
    from authentic_hybrid import load_official_classes
    return load_official_classes(root/"sources"/"mamba2",
                                 root/"sources"/"ttt_linear"), manifest_hash

def cold_restore(torch, model, official, arm, seed, device):
    # Exact state round trip in a disposable, non-persistent directory.
    set_phase("COLD_RESTORE")
    with tempfile.TemporaryDirectory(prefix="ttt-g7-s7-cold-") as d:
        save=Path(d)/"trained-weights.pt"
        torch.save(model.state_dict(),save)
        checkpoint_sha=hashlib.sha256(save.read_bytes()).hexdigest()
        restored,_=new_hybrid(torch,official,arm,seed)
        restored.load_state_dict(torch.load(
            save,map_location="cpu",weights_only=True))
        restored=restored.to(device=device,dtype=torch.float32)
        restored.eval()
        model.eval()
        with torch.no_grad():
            x,_,_=prior.episode(seed*100000+970000,batch=2)
            x=x.to(device)
            a=model(x)[:,8:,:16]
            b=restored(x)[:,8:,:16]
            delta=float((a-b).abs().max().item())
        del restored
    return checkpoint_sha,delta

def run(args):
    set_phase("READ_FROZEN_PROTOCOL")
    raw = args.protocol.read_bytes()
    spec = json.loads(raw)
    frozen_validate(spec)
    import torch
    set_phase("PIN_OFFICIAL_UPSTREAM")
    official, manifest_hash = check_environment(torch,spec,args)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)
    device=torch.device("cuda:0")
    t, v, budget=spec["train"],spec["heldout"],spec["budget"]
    started=time.monotonic()
    report=[]
    max_vram=0
    for seed in t["seeds"]:
        arms={}
        hashes=[]
        for arm in spec["arms"]:
            set_phase("INITIALIZE_MATCHED_"+arm["id"])
            model,initial_sha=new_hybrid(
                torch,official,arm["ttt_base_lr"],seed)
            hashes.append(initial_sha)
            model=model.to(device=device,dtype=torch.float32)
            count_all=sum(p.numel() for p in model.parameters())
            count_fit=sum(p.numel() for p in model.parameters()
                          if p.requires_grad)
            check(count_all == t["full_native_parameters_expected"] and
                  count_fit == t["trainable_parameters_expected"],
                  "NATIVE_MODEL_PARAMETER_COUNT_DRIFT")
            trainable=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(
                trainable,lr=t["learning_rate"],
                weight_decay=t["weight_decay"])
            torch.cuda.reset_peak_memory_stats(device)
            first=None
            recent=[]
            arm_start=time.monotonic()
            model.train()
            for step in range(t["steps_per_arm_per_seed"]):
                set_phase("TRAIN_FORWARD_"+arm["id"])
                x,y,_=prior.episode(seed*100000+step*100,
                                    batch=t["batch_size"])
                x,y=x.to(device),y.to(device)
                optimizer.zero_grad(set_to_none=True)
                out=model(x)[:,8:,:16]
                check(tuple(out.shape)==tuple(y.shape),"TRAIN_TENSOR_SHAPE")
                loss=(out-y).square().mean()
                check(bool(torch.isfinite(loss)),"NONFINITE_TRAIN_LOSS")
                set_phase("TRAIN_BACKWARD_"+arm["id"])
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(
                    trainable,t["gradient_clip"])
                check(bool(torch.isfinite(norm)),"NONFINITE_TRAIN_GRADIENT")
                set_phase("TRAIN_OPTIMIZER_"+arm["id"])
                optimizer.step()
                value=float(loss.detach().item())
                if first is None: first=value
                recent.append(value)
                if len(recent)>32: recent.pop(0)
                # Measure resource envelope on every update, not a prediction.
                torch.cuda.synchronize(device)
                peak=int(torch.cuda.max_memory_allocated(device))
                check(peak<=budget["torch_peak_allocated_bytes_max"],
                      "GPU_ALLOCATED_VRAM_BUDGET_EXCEEDED")
                check(time.monotonic()-started<budget["wall_seconds_max"],
                      "BOUND_WALLTIME_EXCEEDED")
            set_phase("S7_NEW_HELDOUT_CAUSAL_"+arm["id"])
            measures=prior.evaluate_support_mechanism(
                model,seed,v["episodes_per_seed"],device=device)
            set_phase("COLD_RESTORE_"+arm["id"])
            ck_sha,cold_delta=cold_restore(
                torch,model,official,arm["ttt_base_lr"],seed,device)
            torch.cuda.synchronize(device)
            peak=int(torch.cuda.max_memory_allocated(device))
            max_vram=max(max_vram,peak)
            arms[arm["id"]]={
                "parameter_count":count_all,
                "trainable_parameter_count":count_fit,
                "fit_steps":t["steps_per_arm_per_seed"],
                "fit_examples":t["steps_per_arm_per_seed"]*t["batch_size"],
                "fit_first_mse":first,
                "fit_last32_mse":statistics.mean(recent),
                "heldout":measures,
                "cold_checkpoint_sha256":ck_sha,
                "cold_output_max_abs_delta":cold_delta,
                "elapsed_seconds":round(time.monotonic()-arm_start,3),
                "gpu_peak_allocated_bytes":peak,
            }
            del model,optimizer
            torch.cuda.empty_cache()
        check(hashes[0]==hashes[1],"ARM_INITIAL_WEIGHTS_MISMATCH")
        check(arms["online_ttt"]["parameter_count"] ==
              arms["static_matched"]["parameter_count"] and
              arms["online_ttt"]["fit_steps"] ==
              arms["static_matched"]["fit_steps"],"ARM_UPDATE_COUNT_DRIFT")
        report.append({"seed":seed,"initial_weight_sha256":hashes[0],
                       "arms":arms})
    set_phase("AGGREGATE_NEW_FROZEN_HELDOUT")
    metric=["model_correct_mse","model_shuffled_mse","model_erased_mse",
            "oracle_correct_mse","oracle_shuffled_mse","zero_baseline_mse",
            "shuffled_prediction_movement_mse","erased_prediction_movement_mse"]
    means={}
    for name in ("online_ttt","static_matched"):
        means[name]={key:statistics.mean(
            item["arms"][name]["heldout"][key] for item in report)
            for key in metric}
    online,static=means["online_ttt"],means["static_matched"]
    check(online["model_correct_mse"]>0 and
          static["model_correct_mse"]>0,"INVALID_ZERO_ERROR_RATIO")
    thresholds=spec["thresholds"]
    oracle_valid=all(
        abs(means[arm]["zero_baseline_mse"]-1.0)<1e-12 and
        means[arm]["oracle_correct_mse"]<=
        thresholds["oracle_correct_mse_max"] and
        means[arm]["oracle_shuffled_mse"]>=
        thresholds["oracle_corrupted_mse_min"]
        for arm in means
    )
    relative=(static["model_correct_mse"]-
              online["model_correct_mse"])/static["model_correct_mse"]
    shuffle=(online["model_shuffled_mse"]-
             online["model_correct_mse"])/online["model_correct_mse"]
    erased=(online["model_erased_mse"]-
            online["model_correct_mse"])/online["model_correct_mse"]
    seed_zero=[item["arms"]["online_ttt"]["heldout"]["model_correct_mse"]
               < thresholds["zero_predictor_mse"]
               for item in report]
    seed_static=[item["arms"]["online_ttt"]["heldout"]["model_correct_mse"]
                 < item["arms"]["static_matched"]["heldout"]["model_correct_mse"]
                 for item in report]
    restored=all(
        item["arms"][name]["cold_output_max_abs_delta"]<=
        thresholds["max_abs_cold_restore_output_delta"]
        for item in report for name in ("online_ttt","static_matched"))
    meet=(oracle_valid and restored and
          online["model_correct_mse"]<=
          thresholds["online_correct_mse_max"] and
          all(seed_zero) and all(seed_static) and
          relative>=thresholds["relative_online_vs_static_reduction_min"]
          and shuffle>=thresholds["online_shuffled_relative_harm_min"]
          and erased>=thresholds["online_erased_relative_harm_min"])
    status=("ASSAY_INVALID_OR_ENGINEERING_BLOCKED" if not oracle_valid
            or not restored else
            "DEVELOPMENT_SIGNAL_REQUIRES_REPLICATION_AND_MATCHED_CONTROL"
            if meet else "NULL_OR_NEGATIVE")
    return {
        "schema":"ttt-hybrid.g7-s7.native-extended-binding-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":status,
        "frozen_protocol_sha256":hashlib.sha256(raw).hexdigest(),
        "pinned_package_manifest_sha256":manifest_hash,
        "source_pins_sha256":custody.R3_PINS_SHA256,
        "seeds":report,
        "heldout_means":means,
        "oracle_controls_pass":oracle_valid,
        "cold_restore_controls_pass":restored,
        "online_better_than_zero_each_seed":seed_zero,
        "online_better_than_static_each_seed":seed_static,
        "relative_online_vs_static_mse_reduction":relative,
        "relative_online_shuffle_harm":shuffle,
        "relative_online_erased_harm":erased,
        "frozen_thresholds_met":meet,
        "s6_positive_control_reference_run":42368703057,
        "equal_training_examples_to_s6":False,
        "equal_parameters_or_compute_to_s6":False,
        "strongest_conventional_baseline_retrained_in_this_run":False,
        "gpu_peak_torch_allocated_bytes":max_vram,
        "elapsed_seconds":round(time.monotonic()-started,3),
        "whole_process_cost":"GPU, wall and model allocation measured; energy not instrumented",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--expected-sha",required=True)
    ap.add_argument("--protocol",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    receipt=ROOT/"local_runs/g7_s7_native_extended.json"
    result={
        "schema":"ttt-hybrid.g7-s7.native-extended-binding-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "phase":"NOT_STARTED",
        "independent_scientific_qualification":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED",
    }
    code=2
    try:
        check(args.receipt.resolve()==receipt.resolve() and
              args.protocol.resolve()==
              (ROOT/"experiments/g7_s7_frozen_extended_native.json").resolve(),
              "FROZEN_PATH_OR_OUTPUT_DENIED")
        result=run(args)
        code=0 if result["status"]!="ASSAY_INVALID_OR_ENGINEERING_BLOCKED" else 2
    except (Blocked,custody.Blocked,prior.Blocked) as exc:
        result["status"]="ENGINEERING_BLOCKED_"+str(exc)
        result["phase"]=PHASE
    except Exception as exc:
        result["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        result["error_class"]=type(exc).__name__
        result["phase"]=PHASE
        # Never export raw tracebacks, paths, secrets or subprocess stderr.
    finally:
        if args.receipt.resolve()==receipt.resolve():
            receipt.parent.mkdir(parents=True,exist_ok=True)
            receipt.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",
                               encoding="utf-8")
        print("G7_S7_STATUS="+result["status"],flush=True)
        print("G7_S7_PHASE="+result.get("phase","COMPLETED"),flush=True)
        print("G7_S7_SCIENCE=NO_INDEPENDENT_QUALIFICATION",flush=True)
    return code

if __name__=="__main__":
    raise SystemExit(main())
