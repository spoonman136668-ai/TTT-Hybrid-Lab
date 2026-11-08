"""G7-S26 observed action-effect competence diagnostic, development seeds ONLY.

Diagnostic only, does not train, fit, tune, update model fast-state, use any
scientific seeds, call heldout evaluation, change frozen thresholds, or claim
generalization. Evaluation targets are used AFTER read-only slow predictions,
and never enter predictor inputs or hidden simulator coefficients.

Purpose: distinguish the magnitude/direction of untrained slow forecast
from a useful persistent sign contradiction. MSE alone is NOT evidence of
signed intervention competence.
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

# -I execution: only trust the exact checked-out scripts directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from torch import Tensor
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as physics
import g7_s25_contradiction_gate as gate

SEEDS=(3101,3109,3119)
REGIMES=("source_stable","source_signflip","sine_signflip","rational_signflip")
BATCH=4
STEPS=32
MAX_DEV_SECONDS=70
BLOB_S25="31633b341403729b7ffbb7d73f3a04098ad44380"
BLOB_S26="34251b2cde6373ed484dbc21e4617c9165047e93"


def check(ok: bool, reason: str) -> None:
    if not ok:
        raise ValueError(reason)


def contrast_score(predicted: Tensor, observed: Tensor) -> dict:
    """An unbiased, postprediction-only paired action-effect measurement.

    Counts positive sign on significant actual contrasts; tiny predicted
    contrasts are *uninformative*, not magically correct or wrong.
    Explicitly distinguishes threshold-eligible false contradictions.
    This never gates or adjusts a trained model.
    """
    check(isinstance(predicted,Tensor) and isinstance(observed,Tensor) and
          predicted.ndim==2 and predicted.shape==observed.shape and
          predicted.shape[0]>0 and predicted.shape[1]==2,
          "S26_CONTRAST_PAIRED_SAMPLE_SHAPE")
    check(predicted.device.type=="cpu" and observed.device.type=="cpu" and
          torch.isfinite(predicted).all() and torch.isfinite(observed).all(),
          "S26_CONTRAST_NONFINITE_OR_DEVICE")
    p=(predicted[:,1]-predicted[:,0]).detach()
    y=(observed[:,1]-observed[:,0]).detach()
    observed_big=y.abs()>=gate.MIN_SIGNED_EFFECT
    predicted_big=p.abs()>=gate.MIN_SIGNED_EFFECT
    eligible=observed_big & predicted_big
    opposite=(p*y<0) & eligible & (p-y).abs()>=gate.MIN_CONTRAST_DISAGREEMENT
    correct=(p*y>0) & eligible
    # Differences in pair MSE are observable but cannot substitute for sign.
    mae=(p-y).abs()
    return {
        "pairs":int(p.numel()),
        "observed_significant":int(observed_big.sum().item()),
        "predicted_significant":int(predicted_big.sum().item()),
        "both_significant":int(eligible.sum().item()),
        "sign_correct":int(correct.sum().item()),
        "sign_opposite_eligible":int(opposite.sum().item()),
        "unidentifiable_small_prediction":int((observed_big & ~predicted_big).sum().item()),
        "mean_squared_pair_loss_sum":float((predicted-observed).square().sum().item()),
        "effect_absolute_error_sum":float(mae.sum().item()),
    }


def sum_counts(rows: list[dict]) -> dict:
    keys=("pairs","observed_significant","predicted_significant",
          "both_significant","sign_correct","sign_opposite_eligible",
          "unidentifiable_small_prediction")
    counts={k:sum(r[k] for r in rows) for k in keys}
    counts["mean_squared_pair_mse"]=round(sum(
        r["mean_squared_pair_loss_sum"] for r in rows)/
        (2*counts["pairs"]),9)
    counts["mean_absolute_action_effect_error"]=round(sum(
        r["effect_absolute_error_sum"] for r in rows)/counts["pairs"],9)
    counts["effect_sign_accuracy_when_evaluable"]=(
        round(counts["sign_correct"]/counts["both_significant"],6)
        if counts["both_significant"] else None
    )
    counts["effect_sign_coverage_of_real_large_effects"]=(
        round(counts["both_significant"]/counts["observed_significant"],6)
        if counts["observed_significant"] else None
    )
    counts["readout_measures_only_no_competence_pass"]=True
    return counts


def checkpoint_sha256(module: torch.nn.Module) -> str:
    sha=hashlib.sha256()
    for key,value in module.state_dict().items():
        check(value.device.type=="cpu" and torch.isfinite(value).all(),
              "S26_NONFINITE_OR_NONCPU_UNTRAINED_CHECKPOINT")
        array=value.detach().contiguous()
        sha.update(key.encode())
        sha.update(str(tuple(array.shape)).encode())
        sha.update(array.numpy().tobytes())
    return sha.hexdigest()


@torch.no_grad()
def run_untrained_development_diagnostic(
    *, seeds=SEEDS, regimes=REGIMES, batch=BATCH,
    max_seconds=MAX_DEV_SECONDS
) -> dict:
    check(tuple(seeds)==SEEDS and tuple(regimes)==REGIMES and batch==BATCH,
          "S26_DEV_WORLD_ONLY_FIXED_MATRIX")
    check(type(max_seconds) is int and 1<=max_seconds<=MAX_DEV_SECONDS,
          "S26_DEV_BOUNDED_RUNTIME")
    torch.set_num_threads(1)
    torch.manual_seed(3101)
    slow=numeric.ConventionalGRU().eval()
    check(all(not p.is_cuda for p in slow.parameters()),
          "S26_DEV_NO_CUDA")
    checkpoint=checkpoint_sha256(slow)
    start=time.monotonic()
    rows=[]
    by_epoch={"pre20":[],"post20":[]}
    for seed in seeds:
        for regime in regimes:
            world=physics.PairedS20World(seed,batch,regime)
            pre=[];post=[]
            last_observed=None
            raw_hash=hashlib.sha256()
            for step in range(STEPS):
                check(time.monotonic()-start < max_seconds,
                      "S26_DEV_DIAGNOSTIC_WALL_TIMEOUT")
                item=world.emit()
                h=numeric.causal_history32(item.history)
                prediction=slow.predict(h,item.query_pair)
                check(prediction.shape==item.observed_pair.shape and
                      torch.isfinite(prediction).all(),
                      "S26_DEV_SLOW_FORECAST_NOT_FINITE")
                # Both raw observations are accounted for AFTER prediction.
                # The signed-effect evaluator never reads private physics x/z.
                row=contrast_score(prediction,item.observed_pair)
                (pre if step<20 else post).append(row)
                raw_hash.update(item.query_pair.contiguous().numpy().tobytes())
                raw_hash.update(item.observed_pair.contiguous().numpy().tobytes())
                raw_hash.update(item.realized_action_index.contiguous().numpy().tobytes())
            source=world.close()
            check(source["paired_observations"]==STEPS*batch*2,
                  "S26_DEV_PAIRED_WORLD_COMPLETENESS")
            pre_metric=sum_counts(pre)
            post_metric=sum_counts(post)
            by_epoch["pre20"].extend(pre)
            by_epoch["post20"].extend(post)
            rows.append({
                "seed":seed,"regime":regime,"untrained_slow_sha256":checkpoint,
                "observed_world_stream_sha256":raw_hash.hexdigest(),
                "pre20":pre_metric,"post20":post_metric,
                "untrained_slow_model":True,
                "performed_optimizer_or_fast_state_updates":False,
            })
    pre=sum_counts(by_epoch["pre20"])
    post=sum_counts(by_epoch["post20"])
    return {
        "schema":"ttt-hybrid.g7-s26.action-effect-competence-development-readout.v1",
        "status":"PASS_DEVELOPMENT_UNTRAINED_READOUT_ONLY_NOT_SCIENCE",
        "untrained_slow_checkpoint_sha256":checkpoint,
        "rows":rows,"source_seeds":list(seeds),
        "development_worlds_per_seed_regime":batch,
        "pre20":pre,"post20":post,
        "scientific_first_use":"NOT_AUTHORIZED_NOT_RUN",
        "fit_validation_heldout":"NOT_RUN",
        "model_training":"NOT_RUN",
        "fast_weight_update":"NOT_RUN",
        "scientific_baseline_competence":"UNVERIFIED",
        "scientific_contrast_gain":"NOT_MEASURED",
        "frozen_s25_s26_thresholds":"UNCHANGED",
        "gpu":"NOT_USED",
        "elapsed_seconds":round(time.monotonic()-start,3),
    }


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--sha",required=True)
    p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    check(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
          "S26_DEV_CUDA_ISOLATION_REQUIRED")
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],
                          text=True,capture_output=True,timeout=12,check=False)
    check(result.returncode==0 and args.sha==result.stdout.strip() and
          len(args.sha)==40 and
          all(c in "0123456789abcdef" for c in args.sha),
          "S26_DEV_EXACT_SHA_DRIFT")
    out=run_untrained_development_diagnostic()
    out["source_sha"]=args.sha
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print("S26_CONTRAST_COMPETENCE="+out["status"])
    print("S26_DEV_PRE20_EVALUABLE="+str(out["pre20"]["both_significant"]))
    print("S26_DEV_POST20_EVALUABLE="+str(out["post20"]["both_significant"]))
    print("S26_SCIENCE=NOT_RUN")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
