"""G7-S6: frozen, fully trained conventional associative retrieval control.

This intentionally strong, structurally informed *positive control* qualifies
task learnability only. It does NOT establish native hybrid superiority. Runs
CPU only on the existing pinned WSL/R3 environment; no downloads or installs.
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
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import g7_r5_oneshot as custody

class Blocked(Exception):
    pass


def require(test, reason):
    if not test:
        raise Blocked(reason)


def checked(argv, timeout=20):
    p = subprocess.run(argv, check=False, text=True, capture_output=True,
                       timeout=timeout, cwd=ROOT)
    require(p.returncode == 0, "READ_ONLY_COMMAND_FAILED")
    return p.stdout


def package_state_sha256():
    fmt = ("-f=" + chr(36) + "{binary:Package}\t" +
           chr(36) + "{Status}\t" + chr(36) + "{Version}\n")
    s = checked(["/usr/bin/dpkg-query", "-W", fmt], timeout=30)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_batch(torch, seed, batch_size, order):
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    k = torch.randint(0, 2, (batch_size, 4, 16),
                      generator=g, dtype=torch.int64).float()*2-1
    v = torch.randint(0, 2, (batch_size, 4, 16),
                      generator=g, dtype=torch.int64).float()*2-1
    # Rejection sampling makes content-address lookup unique for every query.
    for attempt in range(8):
        collision = (k[:, :, None, :] == k[:, None, :, :]).all(dim=-1)
        mask = torch.triu(collision, diagonal=1).any(dim=(1,2))
        if not bool(mask.any()):
            break
        k[mask] = (torch.randint(0, 2, (int(mask.sum()), 4, 16),
                                 generator=g, dtype=torch.int64).float()*2-1)
    else:
        raise Blocked("DUPLICATE_KEYS_AFTER_EIGHT_RESAMPLES")
    idx = torch.tensor(order, dtype=torch.int64)
    q = k.index_select(1, idx)
    target = v.index_select(1, idx)
    return k, v, q, target


def new_model(torch):
    import torch.nn as nn
    class ConventionalRetrieval(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Linear(16,16,bias=False)
            self.k = nn.Linear(16,16,bias=False)
            self.v = nn.Linear(16,16,bias=False)
            self.o = nn.Linear(16,16,bias=False)
            self.log_temp = nn.Parameter(torch.tensor(0.,dtype=torch.float32))

        def forward(self, keys, support_values, query_keys):
            logits = (self.q(query_keys) @
                      self.k(keys).transpose(1,2)) / math.sqrt(16.0)
            attention = torch.softmax(logits*self.log_temp.exp(),dim=-1)
            return self.o(attention @ self.v(support_values))
    return ConventionalRetrieval()


def estimate(model, untrained, torch, seed, spec, started):
    from torch.nn import functional as F
    order = spec["task"]["query_order"]
    evalspec = spec["evaluation"]
    results = {key: [] for key in (
        "correct_mse", "corrupted_mse", "erased_mse",
        "oracle_correct_mse", "oracle_corrupted_mse",
        "zero_mse", "untrained_mse", "support_prediction_movement_mse"
    )}
    model.eval()
    untrained.eval()
    with torch.no_grad():
        for batch_idx in range(evalspec["heldout_batches_per_seed"]):
            s = seed*10000000 + 9000000 + batch_idx*17
            k,v,q,y = get_batch(
                torch,s,evalspec["heldout_batch_size"],order)
            # Positive oracle plus strict no-target-leak/uniqueness controls.
            matches=(q[:,:,None,:]==k[:,None,:,:]).all(dim=-1)
            require(bool((matches.sum(dim=-1)==1).all()),
                    "AMBIGUOUS_HELDOUT_LOOKUP")
            lookup=matches.float()
            oracle=torch.einsum("bqi,biv->bqv",lookup,v)
            require(bool(torch.equal(oracle,y)),
                    "POSITIVE_ORACLE_BROKEN")
            altered=torch.roll(v,shifts=1,dims=1)
            oracle_bad=torch.einsum("bqi,biv->bqv",lookup,altered)
            correct=model(k,v,q)
            corrupted=model(k,altered,q)
            erased=model(k,torch.zeros_like(v),q)
            initial=untrained(k,v,q)
            require(bool(torch.isfinite(correct).all()) and
                    bool(torch.isfinite(corrupted).all()) and
                    bool(torch.isfinite(erased).all()) and
                    bool(torch.isfinite(initial).all()),
                    "NONFINITE_HELDOUT_PREDICTIONS")
            values={
                "correct_mse": F.mse_loss(correct,y),
                "corrupted_mse": F.mse_loss(corrupted,y),
                "erased_mse": F.mse_loss(erased,y),
                "oracle_correct_mse": F.mse_loss(oracle,y),
                "oracle_corrupted_mse": F.mse_loss(oracle_bad,y),
                "zero_mse": y.square().mean(),
                "untrained_mse": F.mse_loss(initial,y),
                "support_prediction_movement_mse": F.mse_loss(correct,corrupted),
            }
            for key,value in values.items():
                results[key].append(float(value.item()))
            require(time.monotonic()-started <
                    spec["budget"]["wall_seconds_max"],"WALL_BUDGET")
    return {key:statistics.mean(val) for key,val in results.items()}


def run(args):
    import torch
    from torch.nn import functional as F
    from torch import optim

    raw=args.protocol.read_bytes()
    spec=json.loads(raw)
    require(spec["schema"]==
            "ttt-hybrid.g7-s6.conventional-associative-baseline.v1",
            "FROZEN_PROTOCOL_SCHEMA")
    b=spec["baseline"]
    ev=spec["evaluation"]
    require(b["train_seeds"]==[101,211,307]
            and b["steps"]==512
            and b["batch_size"]==64
            and b["model_parameters_expected"]==1025
            and ev["heldout_batches_per_seed"]==8
            and ev["heldout_batch_size"]==128,
            "FROZEN_PARAMETER_DRIFT")
    require(spec["task"]["query_order"]==[0,1,2,3,3,2,1,0]
            and spec["task"]["support_pair_count"]==4,
            "TASK_DRIFT")
    require(os.geteuid()==1000 and "microsoft" in
            Path("/proc/sys/kernel/osrelease").read_text().lower(),
            "NORMAL_WSL_USER_REQUIRED")
    require(custody.git(ROOT,"rev-parse","HEAD")==args.expected_sha
            and not custody.git(ROOT,"status","--porcelain","--untracked-files=all"),
            "SOURCE_CUSTODY_DRIFT")
    require(custody.sha256(ROOT/"source_pins.json")==
            custody.R3_PINS_SHA256,"SOURCE_PIN_DRIFT")
    manifest,pkgs=custody.manifest()
    require(manifest==custody.R3_MANIFEST_SHA256
            and pkgs.get("torch")==custody.TORCH
            and pkgs.get("triton")==custody.TRITON
            and torch.__version__==custody.TORCH
            and sys.version_info[:2]==(3,12),
            "PINNED_RUNTIME_DRIFT")
    require(torch.cuda.is_initialized() is False, "CUDA_ALREADY_INITIALIZED")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    spec_sha=hashlib.sha256(raw).hexdigest()
    results=[]
    for seed in b["train_seeds"]:
        torch.manual_seed(seed)
        model=new_model(torch)
        initial=new_model(torch)
        initial.load_state_dict(model.state_dict())
        param_count=sum(x.numel() for x in model.parameters())
        require(param_count==b["model_parameters_expected"],
                "BASELINE_PARAM_COUNT_DRIFT")
        initial_hash=hashlib.sha256(
            b"".join(t.detach().numpy().tobytes() for t in
                     model.state_dict().values())).hexdigest()
        optimizer=optim.AdamW(model.parameters(),lr=b["lr"],
                               weight_decay=b["weight_decay"])
        losses=[]
        model.train()
        for step in range(b["steps"]):
            k,v,q,y=get_batch(torch,seed*10000000+step*17,
                              b["batch_size"],spec["task"]["query_order"])
            pred=model(k,v,q)
            loss=F.mse_loss(pred,y)
            require(bool(torch.isfinite(loss)), "NONFINITE_FIT_LOSS")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                b["gradient_clip"])
            require(bool(torch.isfinite(gn)), "NONFINITE_GRAD")
            optimizer.step()
            losses.append(float(loss.detach()))
            require(time.monotonic()-started <
                    spec["budget"]["wall_seconds_max"],"WALL_BUDGET")
            maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
            require(maxrss<=spec["budget"]["peak_cpu_rss_bytes_max"],
                    "CPU_RSS_BUDGET")
        metrics=estimate(model,initial,torch,seed,spec,started)
        # Save and *cold load* only in temporary storage, then re-check outputs.
        with tempfile.TemporaryDirectory(prefix="g7-s6-cold-") as t:
            checkpoint=Path(t)/"baseline-state.pt"
            torch.save(model.state_dict(),checkpoint)
            checkpoint_sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            restored=new_model(torch)
            restored.load_state_dict(torch.load(
                checkpoint,map_location="cpu",weights_only=True))
            restored.eval()
            model.eval()
            with torch.no_grad():
                k,v,q,_=get_batch(torch,seed*10000000+9500000,
                                   32,spec["task"]["query_order"])
                before=model(k,v,q)
                after=restored(k,v,q)
                delta=float((before-after).abs().max().item())
        thresholds=spec["thresholds"]
        valid=(abs(metrics["zero_mse"]-
                   thresholds["zero_predictor_mse_exact"])<=1e-12
               and metrics["oracle_correct_mse"]<=
                   thresholds["oracle_correct_mse_max"]
               and metrics["oracle_corrupted_mse"]>=
                   thresholds["oracle_corrupted_mse_min"])
        passed=(valid
                and metrics["correct_mse"]<=
                    thresholds["trained_correct_mse_max"]
                and metrics["corrupted_mse"]>=
                    thresholds["corrupted_support_mse_min"]
                and metrics["erased_mse"]>=
                    thresholds["erased_support_mse_min"]
                and statistics.mean(losses[-32:])<=
                    thresholds["train_final_32_step_loss_max"]
                and delta<=
                    thresholds["cold_restore_max_abs_output_delta"])
        results.append({
            "seed":seed, "parameters":param_count,
            "steps":len(losses),"train_first_loss":losses[0],
            "train_last_32_loss_mean":statistics.mean(losses[-32:]),
            "initial_weights_sha256":initial_hash,
            "cold_checkpoint_sha256":checkpoint_sha,
            "cold_restore_max_abs_delta":delta,
            "heldout":metrics,
            "oracle_controls_pass":valid,
            "frozen_thresholds_pass":passed
        })
    require(not torch.cuda.is_initialized(), "CUDA_USED_IN_CPU_ONLY_STUDY")
    valid_all=all(r["oracle_controls_pass"] for r in results)
    passed_all=all(r["frozen_thresholds_pass"] for r in results)
    status=("ASSAY_INVALID_OR_ENGINEERING_BLOCKED" if not valid_all else
            "TASK_LEARNABILITY_POSITIVE_CONTROL_PASS" if passed_all else
            "TASK_LEARNABILITY_POSITIVE_CONTROL_FAIL")
    return {
        "schema":"ttt-hybrid.g7-s6.trained-conventional-evidence.v1",
        "timestamp_utc":datetime.now(timezone.utc).isoformat(),
        "status":status,"candidate_sha":args.expected_sha,
        "frozen_protocol_sha256":spec_sha,
        "science_scope":"TASK_LEARNABILITY_ONLY",
        "hybrid_superiority":"NOT_TESTED",
        "parameter_compute_matched_hybrid":"NO",
        "pinned_torch":custody.TORCH,
        "pinned_package_manifest_sha256":manifest,
        "seeds":results,
        "all_oracle_controls_pass":valid_all,
        "all_frozen_baseline_thresholds_pass":passed_all,
        "heldout_total_episodes":len(results)*
            ev["heldout_batches_per_seed"]*ev["heldout_batch_size"],
        "cpu_peak_rss_bytes":
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        "elapsed_seconds":round(time.monotonic()-started,3),
        "gpu_execution":"NOT_RUN",
        "independent_replication":"NOT_RUN",
        "deployment":"NOT_AUTHORIZED"
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--protocol",required=True,type=Path)
    p.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    report={
        "schema":"ttt-hybrid.g7-s6.trained-conventional-evidence.v1",
        "candidate_sha":args.expected_sha,
        "status":"ENGINEERING_UNVERIFIED",
        "science_scope":"TASK_LEARNABILITY_ONLY",
        "deployment":"NOT_AUTHORIZED"
    }
    code=2
    before=None
    try:
        require(args.protocol.resolve()==
                (ROOT/"experiments/g7_s6_frozen_conventional_baseline.json").resolve()
                and args.receipt.resolve()==
                (ROOT/"local_runs/g7_s6_trained_baseline.json").resolve(),
                "PROTOCOL_OR_RECEIPT_PATH_DENIED")
        before=package_state_sha256()
        report=run(args)
        code=0 if report["status"]!="ASSAY_INVALID_OR_ENGINEERING_BLOCKED" else 2
    except (Blocked,custody.Blocked) as exc:
        report["status"]="ENGINEERING_BLOCKED_"+str(exc)
    except (OSError,subprocess.TimeoutExpired) as exc:
        report["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
    except Exception as exc:
        report["status"]="ENGINEERING_BLOCKED_"+type(exc).__name__
        report["exception_class"]=type(exc).__name__
    finally:
        if before is not None:
            try:
                after=package_state_sha256()
                report["package_inventory_before_sha256"]=before
                report["package_inventory_after_sha256"]=after
                report["package_changes"]="NONE" if after==before else "DRIFT"
                if after!=before:
                    report["status"]="ENGINEERING_BLOCKED_PACKAGES_CHANGED"
                    code=2
            except Exception:
                report["status"]="ENGINEERING_BLOCKED_POSTRUN_INVENTORY"
                code=2
        if (args.receipt.resolve()==
                (ROOT/"local_runs/g7_s6_trained_baseline.json").resolve()):
            args.receipt.parent.mkdir(parents=True,exist_ok=True)
            args.receipt.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",
                                    encoding="utf-8")
        print("G7_S6_STATUS="+report["status"],flush=True)
        print("G7_S6_PACKAGES="+str(report.get("package_changes","UNVERIFIED")),
              flush=True)
        print("G7_S6_GPU=NOT_RUN",flush=True)
    return code


if __name__=="__main__":
    raise SystemExit(main())
