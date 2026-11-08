"""G7-S26 pre-science DEVELOPMENT-only comparison against unchanged S25.

This tool replays *untrained* S25 and S26 policies on identical paired S20
engineering worlds. It quantifies update timing and trigger/fallback but cannot
establish scientific learning or benchmark superiority. It does not train,
calibrate, touch S25 scientific/heldout seeds, update source, or install tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from g7_process_peak_memory import read_peak_memory
from pathlib import Path
import subprocess
import sys
import time

# -I disables conventional script-local import handling; only insert the
# exact checkout's scripts directory for this bounded source-only harness.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch

import g7_s25_gate_diagnostic as original
import g7_s25_numeric_ttt as numeric
import g7_s25_paired_interventions as physics
import g7_s25_cost_ledger as ledger
import g7_s26_reserve_gate as proposed

S25_HEAD="108808f7b685107a76fae1ce5605b7ff9e2bac27"
DEV_SEEDS=(3101,3109,3119)
REGIMES=("source_stable","source_signflip","sine_signflip","rational_signflip")
MODES=("periodic","random","surprise","contradiction")
BATCH=4


def require(condition,reason:str) -> None:
    if not condition:
        raise ValueError(reason)


def _shadow(model,seed:int,mode:str,batch:int):
    return [proposed.QuarterReserveGate(
        mode,engineering_seed=seed+101*j) for j in range(batch)]


@torch.no_grad()
def run_development_comparison(
    *, seeds=DEV_SEEDS,regimes=REGIMES,modes=MODES,batch=BATCH,
    max_seconds=300,
) -> dict:
    require(seeds == DEV_SEEDS and regimes == REGIMES and
            modes == MODES and batch == BATCH,
            "S26_FULL_DEV_MATRIX_ONLY_NO_SCIENCE_SEED_OVERRIDE")
    require(type(max_seconds) is int and 1 <= max_seconds <= 300,
            "S26_BOUNDED_DIAGNOSTIC_WALL_LIMIT")
    start=time.monotonic()
    torch.set_num_threads(1)
    # Run the unchanged S25 diagnostic, which itself uses ONLY dev seeds.
    frozen=original.run_diagnostic(
        seeds=seeds,regimes=regimes,modes=modes,batch=batch,
        max_seconds=min(170,max_seconds))
    require(frozen["status"]=="PASS_DEVELOPMENT_MECHANISM_DIAGNOSTIC_EXECUTED" and
            frozen["frozen_scientific_seeds_used"] is False and
            len(frozen["rows"])==len(seeds)*len(regimes)*len(modes),
            "S26_S25_REFERENCE_REPLAY_UNQUALIFIED")
    s25={(r["seed"],r["regime"],r["mode"]):r for r in frozen["rows"]}
    torch.manual_seed(3101)
    net=numeric.NumericFastWeightTTT(numeric.ConventionalGRU()).eval()
    rows=[]
    for seed in seeds:
        for regime in regimes:
            for mode in modes:
                require(time.monotonic()-start < max_seconds,
                        "S26_DIAGNOSTIC_TIME_CAP")
                world=physics.PairedS20World(seed,batch,regime)
                ep=numeric.PrequentialTTTEpisode(
                    net,mode=mode,seed=seed,batch=batch,
                    reference=torch.zeros(batch,2))
                proposed.inject_quarter_reserve(ep,seed=seed)
                record=ledger.EpisodeWorkLedger(ep,source_seed=seed)
                shadows=_shadow(net,seed,mode,batch)
                used_quarters=[0,0,0,0]
                selected_quarters=[0,0,0,0]
                forced_quarters=[0,0,0,0]
                true_conflict_quarters=[0,0,0,0]
                mse_sum=0.0
                time_counts=[]
                for t in range(32):
                    require(time.monotonic()-start < max_seconds,
                            "S26_DIAGNOSTIC_TIME_CAP")
                    sample=world.emit()
                    predicted=record.step(sample)
                    mse_sum+=float((predicted-sample.observed_pair).square().sum())
                    actual=ep.update_masks[-1]
                    times=int(actual.sum())
                    time_counts.append(times)
                    used_quarters[t//8]+=times
                    for j in range(batch):
                        primary=int(sample.realized_action_index[j].item())
                        error=float((sample.observed_pair[j,primary]-
                                     predicted[j,primary]).item())
                        d=shadows[j].observe(
                            t,
                            predicted_minus=float(predicted[j,0].item()),
                            predicted_plus=float(predicted[j,1].item()),
                            observed_minus=float(sample.observed_pair[j,0].item()),
                            observed_plus=float(sample.observed_pair[j,1].item()),
                            primary_residual=error)
                        require(d.update == bool(actual[j]) and
                                d.real_updates_used==ep.gates[j].updates_used,
                                "S26_SHADOW_GATE_DIVERGENCE")
                        selected_quarters[t//8]+=int(d.reason=="trigger")
                        forced_quarters[t//8]+=int(d.reason=="forced_budget")
                        true_conflict_quarters[t//8]+=int(
                            d.contradiction_evidence_count>=2)
                expected_updates=32
                counts=record.close()
                world_receipt=world.close()
                for shadow in shadows:
                    require(shadow.close()==8,"S26_SHADOW_BUDGET_MISMATCH")
                require(counts["real_fast_weight_update_worlds"]==expected_updates
                        and world_receipt["paired_observations"]==256
                        and used_quarters==[8,8,8,8]
                        and all(a==b+c for a,b,c in zip(
                            used_quarters,selected_quarters,forced_quarters)),
                        "S26_QUARTER_RESOURCE_OR_REASON_MISMATCH")
                previous=s25[(seed,regime,mode)]
                require(previous["stream_sha256"]==
                        counts["input_stream_sha256"] and
                        previous["slow_checkpoint_sha256"]==
                        counts["model_checkpoint_sha256"] and
                        previous["real_fast_weight_update_worlds"]==
                        counts["real_fast_weight_update_worlds"] and
                        previous["observed_action_pairs"]==
                        counts["paired_intervention_observations"],
                        "S26_S25_MATCHED_INFORMATION_OR_CHECKPOINT_FAILED")
                require(sum(time_counts)==sum(previous["updates_by_time"])==
                        expected_updates,"S26_S25_RESOURCE_PARITY_BROKEN")
                # Report metrics only. No selection, threshold tuning,
                # statistical acceptance, or scientific first use.
                rows.append({
                    "seed":seed,"regime":regime,"mode":mode,
                    "source_stream_sha256":counts["input_stream_sha256"],
                    "slow_checkpoint_sha256":
                        counts["model_checkpoint_sha256"],
                    "paired_probe_exposure":counts["paired_intervention_observations"],
                    "fast_weight_updates_each":counts["real_fast_weight_update_worlds"],
                    "s25_pre20_updates":sum(previous["updates_by_time"][:20]),
                    "s25_post20_updates":sum(previous["updates_by_time"][20:]),
                    "s26_pre20_updates":sum(time_counts[:20]),
                    "s26_post20_updates":sum(time_counts[20:]),
                    "s25_trigger_selected":previous["selected_trigger_steps"],
                    "s25_forced_budget":previous["forced_budget_updates"],
                    "s26_quarter_updates":used_quarters,
                    "s26_quarter_trigger_selected":selected_quarters,
                    "s26_quarter_forced_budget":forced_quarters,
                    "s26_quarter_persistent_contradiction":
                        true_conflict_quarters,
                    "s25_update_mask_by_step":previous["updates_by_time"],
                    "s26_update_mask_by_step":time_counts,
                    "s26_untrained_pair_prediction_mse":
                        round(mse_sum/(batch*32*2),9),
                    "s26_realized_primary_sourced_from_observable_selector":True,
                    "science":"NOT_RUN",
                })
    summary={}
    for mode in modes:
        arm=[r for r in rows if r["mode"]==mode]
        summary[mode]={
            "s25_updates_pre20":sum(r["s25_pre20_updates"] for r in arm),
            "s25_updates_post20":sum(r["s25_post20_updates"] for r in arm),
            "s26_updates_pre20":sum(r["s26_pre20_updates"] for r in arm),
            "s26_updates_post20":sum(r["s26_post20_updates"] for r in arm),
            "s26_selected_trigger":sum(sum(r["s26_quarter_trigger_selected"])
                                       for r in arm),
            "s26_forced_budget":sum(sum(r["s26_quarter_forced_budget"])
                                    for r in arm),
            "s25_selected_trigger":sum(r["s25_trigger_selected"] for r in arm),
            "s25_forced_budget":sum(r["s25_forced_budget"] for r in arm),
            "s26_last_quarter_updates":sum(r["s26_quarter_updates"][3] for r in arm),
        }
    return {
        "schema":"ttt-hybrid.g7-s26.development-only-quarter-reserve-comparison.v1",
        "status":"PASS_DEVELOPMENT_ONLY_COMPARISON_NOT_SCIENCE",
        "s25_source_head":S25_HEAD,
        "frozen_s25_untouched":True,
        "s25_dev_replay_status":frozen["summary"]["status"],
        "s26_untrained_dev_quarter_reserve_summary":summary,
        "row_count":len(rows),
        "rows":rows,
        "all_observations_and_slow_checkpoints_match":True,
        "all_k8_actual_update_counts_match":True,
        "untrained_slow_model":True,
        "scientific_fit_validation_heldout_seed_consumption":"NONE",
        "fully_trained_gru_or_kernel":"NOT_RUN",
        "scientific_first_use":"NOT_AUTHORIZED_NOT_RUN",
        "full_windows_linux_regression":"BLOCKED_PREVIOUSLY_QUALIFIED_ENVIRONMENT",
        "scientific_claim":"NONE",
        "promotion":"NOT_AUTHORIZED",
        "elapsed_wall_seconds":round(time.monotonic()-start,3),
    }


def main() -> int:
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--receipt",required=True,type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise SystemExit("S26_DIAGNOSTIC_CUDA_NOT_ISOLATED")
    result=subprocess.run(
        ["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,
        text=True,timeout=10,check=False)
    if (len(args.expected_sha)!=40 or result.returncode!=0 or
            result.stdout.strip()!=args.expected_sha):
        raise SystemExit("S26_DIAGNOSTIC_WRONG_SHA")
    result=subprocess.run(
        ["git","-C",str(root),"merge-base","--is-ancestor",S25_HEAD,"HEAD"],
        capture_output=True,text=True,timeout=10,check=False)
    if result.returncode!=0:
        raise SystemExit("S26_DIAGNOSTIC_S25_BASE_NOT_ANCESTOR")
    value=run_development_comparison()
    # Engineering only: OS-native memory peaks are NOT interchangeable.
    # The original frozen S26 science source and Linux RSS cap stay untouched.
    memory=read_peak_memory()
    require(memory.peak_bytes <= 2147483648,
            "S26_DEVELOPMENT_2GIB_OS_NATIVE_PEAK_CAP_EXCEEDED")
    value["dev_peak_rss_bytes"]=(
        memory.peak_bytes if memory.comparable_to_linux_ru_maxrss else None)
    value["dev_peak_working_set_bytes"]=(
        memory.peak_bytes
        if memory.metric=="windows_psapi_peak_working_set_bytes" else None)
    value["dev_peak_memory_metric"]=memory.metric
    value["dev_peak_memory_comparable_to_linux_ru_maxrss"]=(
        memory.comparable_to_linux_ru_maxrss)
    value["max_allowed_rss_bytes"]=2147483648
    value["source_sha"]=args.expected_sha
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(value,sort_keys=True,indent=2)+"\n")
    print("S26_DEV_COMPARISON="+value["status"])
    print("S26_SCIENTIFIC_FIRST_USE=NOT_RUN")
    for mode in MODES:
        x=value["s26_untrained_dev_quarter_reserve_summary"][mode]
        print(f"S26_{mode.upper()}_POST20={x['s26_updates_post20']};S25={x['s25_updates_post20']}")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
