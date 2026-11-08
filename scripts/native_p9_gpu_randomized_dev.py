"""P9 G2: randomized, paired CUDA-event + host-wall measurement, DEV ONLY.

The inherited P9 model and gate are NOT modified. All four arms share the same
source weights/observations/complete recurrent history. Neither optimizer
training nor scientific first-use occurs. We retain ALL samples and failures.
CUDA events measure elapsed time on the submitted stream including gaps
between Python-dispatched kernels; they are NOT pure hardware kernel time.
"""
from __future__ import annotations
import math
import random
import statistics
import time

import torch
from torch.nn import functional as F

from scripts.native_p9_parallel_refinement_dev import (
    ContinuousDualStream, SEEDS, WIDTH, WINDOW, trace,
)

ARMS=("parallel_dense","parallel_gate","parallel_random","sequential_dense")
REPETITIONS=24
WARMUPS=3
STEPS=64
BATCH=4
ORDER_RANDOM_SEED_OFFSET=310000
BOOTSTRAP_RANDOM_SEED_OFFSET=410000
BOOTSTRAP_RESAMPLES=1000
SPEED_RATIO_LIMIT=.75
QUALITY_RATIO_LIMIT=1.01
MAX_BENCHMARK_SECONDS=115


def round_schedule(seed:int,repetitions:int=REPETITIONS)->list[tuple[str,...]]:
    """Predeclared deterministic random permutations, one of each arm/round."""
    if seed not in SEEDS or type(repetitions) is not int or not 1<=repetitions<=REPETITIONS:
        raise ValueError("P9_G2_UNREGISTERED_ROUND")
    rng=random.Random(seed+ORDER_RANDOM_SEED_OFFSET)
    rounds=[]
    for _ in range(repetitions):
        arm_order=list(ARMS)
        rng.shuffle(arm_order)
        rounds.append(tuple(arm_order))
    return rounds


def paired_bootstrap(median_ratios:list[float],seed:int)->dict:
    """Paired per-round ratios; deterministic bootstrap and no outlier pruning."""
    if seed not in SEEDS or not median_ratios or any(
        not math.isfinite(v) or v<=0 for v in median_ratios
    ):
        raise ValueError("P9_G2_INVALID_PAIRED_MEASUREMENTS")
    rng=random.Random(seed+BOOTSTRAP_RANDOM_SEED_OFFSET)
    n=len(median_ratios)
    draws=[]
    for _ in range(BOOTSTRAP_RESAMPLES):
        draws.append(statistics.median(
            median_ratios[rng.randrange(n)] for _ in range(n)))
    draws.sort()
    return {"median_ratio":statistics.median(median_ratios),
            "bootstrap_95pct_median_CI_lower":draws[24],
            "bootstrap_95pct_median_CI_upper":draws[974],
            "bootstrap_resamples":BOOTSTRAP_RESAMPLES,
            "all_paired_ratios":median_ratios}


def rollout(model,obs,mode:str,gates:list[bool]):
    """No GPU-to-host target conversion in the timed forward."""
    sensors,actions,dts,*_=obs
    state=model.initial(sensors[0])
    outputs=[]
    if len(gates)!=len(sensors):
        raise ValueError("P9_G2_GATE_LENGTH_DRIFT")
    for index,active in enumerate(gates):
        output,state=model.step(sensors[index],actions[index],dts[index],
                                state,mode=mode,active=active)
        outputs.append(output)
        if state.events!=index+1 or state.cache.shape[1]>WINDOW:
            raise RuntimeError("P9_G2_TEMPORAL_STATE_INTERRUPTED")
    return torch.stack(outputs),state


def _specs(obs):
    n=len(obs[0])
    return {"parallel_dense":("parallel",[True]*n),
            "parallel_gate":("parallel",obs[4]),
            "parallel_random":("parallel",obs[5]),
            "sequential_dense":("sequential",[True]*n)}


def _all_state_continuous(outputs:dict,steps:int)->dict:
    ref=outputs["parallel_dense"][1]
    return {
        "all_recurrent_states_bit_identical":all(torch.equal(st.h,ref.h)
            for _,st in outputs.values()),
        "all_bounded_attention_caches_bit_identical":all(torch.equal(st.cache,ref.cache)
            for _,st in outputs.values()),
        "all_events_completed":all(st.events==steps for _,st in outputs.values()),
        "cache_window_eight":all(st.cache.shape[1]==WINDOW
            for _,st in outputs.values()),
    }


def _measure_cuda(model,obs,mode,gates,device):
    torch.cuda.synchronize(device)
    begin=torch.cuda.Event(enable_timing=True)
    end=torch.cuda.Event(enable_timing=True)
    wall_start=time.perf_counter()
    begin.record()
    rollout(model,obs,mode,gates)
    end.record()
    end.synchronize()
    wall_ms=(time.perf_counter()-wall_start)*1000
    event_ms=begin.elapsed_time(end)
    if not 0<wall_ms<10000 or not 0<event_ms<10000:
        raise RuntimeError("P9_G2_NONFINITE_OR_INVALID_TIMING")
    return {"wall_ms":wall_ms,"cuda_event_ms":event_ms}


def compare_one(seed:int,device:torch.device)->dict:
    if seed not in SEEDS or device.type!="cuda" or not torch.cuda.is_available():
        raise ValueError("P9_G2_GPU_AND_FROZEN_SOURCE_REQUIRED")
    torch.manual_seed(seed)
    model=ContinuousDualStream().to(device).eval()
    t=trace(seed,STEPS,BATCH)
    obs=tuple(v.to(device) if isinstance(v,torch.Tensor) else v for v in t)
    specs=_specs(obs)
    if specs["parallel_gate"][1].count(True)!=STEPS//2 or (
        specs["parallel_random"][1].count(True)!=STEPS//2):
        raise RuntimeError("P9_G2_EQUAL_SKIP_BUDGET_DRIFT")
    schedule=round_schedule(seed)
    all_records={key:[] for key in ARMS}
    with torch.inference_mode():
        reference={name:rollout(model,obs,*specs[name]) for name in ARMS}
        checks=_all_state_continuous(reference,STEPS)
        checks["exact_equal_budget_random_negative"]=(
            sum(specs["parallel_gate"][1])==sum(specs["parallel_random"][1])
            ==STEPS//2)
        checks["all_four_frozen_arms_present"]=set(specs)==set(ARMS)
        errors={}
        for name,(pred,_) in reference.items():
            errors[name]=float(F.mse_loss(pred,obs[3]))
        checks["all_finite_factual_DEV_errors"]=all(
            math.isfinite(x) and x>0 for x in errors.values())
        for _ in range(WARMUPS):
            for name in ARMS:
                rollout(model,obs,*specs[name])
        torch.cuda.synchronize(device)
        for round_index,order in enumerate(schedule):
            for name in order:
                sample=_measure_cuda(model,obs,*specs[name],device)
                all_records[name].append({"round":round_index,**sample})
    checks["every_arm_exactly_24_measurements"]=all(
        len(x)==REPETITIONS for x in all_records.values())
    checks["all_arm_round_indices_complete"]=all(
        [s["round"] for s in x]==list(range(REPETITIONS))
        for x in all_records.values())
    if not all(checks.values()):
        raise RuntimeError("P9_G2_ENGINEERING_INTEGRITY_FAIL")
    def stats(key:str):
        fields={}
        for name in ARMS:
            values=[row[key] for row in all_records[name]]
            fields[name]={
                "median_ms":statistics.median(values),
                "min_ms":min(values),"max_ms":max(values),
                "quartiles_ms":[statistics.quantiles(values,n=4)[0],
                                statistics.quantiles(values,n=4)[2]],
                "all_ms":values
            }
        return fields
    wall=stats("wall_ms")
    event=stats("cuda_event_ms")
    paired_wall=[all_records["parallel_gate"][i]["wall_ms"]/
                 all_records["parallel_dense"][i]["wall_ms"]
                 for i in range(REPETITIONS)]
    paired_event=[all_records["parallel_gate"][i]["cuda_event_ms"]/
                  all_records["parallel_dense"][i]["cuda_event_ms"]
                  for i in range(REPETITIONS)]
    wall_CI=paired_bootstrap(paired_wall,seed)
    event_CI=paired_bootstrap(paired_event,seed)
    ratios={
        "wall_ratio_of_independent_medians":(
            wall["parallel_gate"]["median_ms"]/wall["parallel_dense"]["median_ms"]),
        "cuda_event_ratio_of_independent_medians":(
            event["parallel_gate"]["median_ms"]/event["parallel_dense"]["median_ms"]),
        "paired_wall":wall_CI,"paired_cuda_event":event_CI
    }
    gains={
        # Original P9 objective: >=25% measured wall-time improvement.
        "at_least_25pct_wall_speedup":ratios["wall_ratio_of_independent_medians"]<=SPEED_RATIO_LIMIT,
        "at_least_25pct_cuda_event_speedup":ratios["cuda_event_ratio_of_independent_medians"]<=SPEED_RATIO_LIMIT,
        "paired_wall_95pct_bootstrap_upper_at_most_0_75":
            wall_CI["bootstrap_95pct_median_CI_upper"]<=SPEED_RATIO_LIMIT,
        "paired_cuda_event_95pct_bootstrap_upper_at_most_0_75":
            event_CI["bootstrap_95pct_median_CI_upper"]<=SPEED_RATIO_LIMIT,
        "no_more_than_1pct_untrained_DEV_MSE_degradation":
            errors["parallel_gate"]<=QUALITY_RATIO_LIMIT*errors["parallel_dense"],
        "variance_gate_beats_equal_budget_random_DEV_MSE":
            errors["parallel_gate"]<errors["parallel_random"],
        "parallel_dense_faster_than_sequential_dense_wall":
            wall["parallel_dense"]["median_ms"]<wall["sequential_dense"]["median_ms"],
    }
    return {
        "initialization_seed":seed,
        "source_model_changed":False,
        "device":"cuda",
        "measured_rounds":REPETITIONS,
        "warmups_per_arm":WARMUPS,
        "order_random_seed":seed+ORDER_RANDOM_SEED_OFFSET,
        "all_randomized_arm_orders":schedule,
        "all_checks":checks,
        "integrity":"PASS",
        "factual_MSE_untrained":errors,
        "wall_ms":wall,
        "cuda_event_elapsed_ms":event,
        "ratios":ratios,
        "hypotheses":gains,
        "refinement_calls_each":{
            name:sum(specs[name][1]) for name in ARMS},
        "cuda_event_caveat":"Events include idle gaps while host dispatches Python steps; not pure CUDA kernel execution time."
    }


def run()->dict:
    import os
    if os.environ.get("NATIVE_P9_DEV_ONLY","1")!="1":
        raise RuntimeError("P9_G2_SCIENTIFIC_FIRST_USE_FORBIDDEN")
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("P9_G2_CUDA_ADMISSION_REQUIRED_NO_CPU_FALLBACK")
    start=time.perf_counter()
    torch.set_num_threads(1)
    device=torch.device("cuda:0")
    records=[compare_one(seed,device) for seed in SEEDS]
    elapsed=time.perf_counter()-start
    counts={key:sum(int(x["hypotheses"][key]) for x in records)
            for key in records[0]["hypotheses"]}
    reproducible_speed_and_quality=all(
        r["hypotheses"]["at_least_25pct_wall_speedup"] and
        r["hypotheses"]["at_least_25pct_cuda_event_speedup"] and
        r["hypotheses"]["paired_wall_95pct_bootstrap_upper_at_most_0_75"] and
        r["hypotheses"]["paired_cuda_event_95pct_bootstrap_upper_at_most_0_75"] and
        r["hypotheses"]["no_more_than_1pct_untrained_DEV_MSE_degradation"]
        for r in records)
    valid=(len(records)==len(SEEDS)
           and all(r["integrity"]=="PASS" for r in records)
           and elapsed<MAX_BENCHMARK_SECONDS)
    return {
        "schema":"ttt-hybrid.native-p9.g2.randomized-cuda-dev.v1",
        "status":"PASS_GPU_DEV_INTEGRITY_NOT_SCIENCE" if valid
                 else "FAIL_GPU_DEV_INTEGRITY_NOT_SCIENCE",
        "gpu_execution_verified":True,
        "actual_cuda_event_measurements":True,
        "actual_host_wall_measurements":True,
        "trained_model":False,
        "native_substrate_unchanged":True,
        "real_fused_Mamba2":False,
        "production_MoD":False,
        "scientific_first_use":"NOT_RUN",
        "scientific_world_seeds_consumed":[],
        "seeds":list(SEEDS),"total_measurements":len(records)*len(ARMS)*REPETITIONS,
        "bootstrap_resamples_per_seed":BOOTSTRAP_RESAMPLES,
        "hypothesis_support_of_three":counts,
        "reproducible_25pct_speed_and_1pct_quality_all_three":
            reproducible_speed_and_quality,
        "records":records,
        "actual_mechanism_wall_seconds":elapsed,
        "limitations":"Same familiar public synthetic untrained P9 model. CUDA events measure intervals INCLUDING host launch gaps in eager Python, not pure kernel-time or matched FLOPs. Bootstrap across rounds within each fixed model init, not independent world replicates. 1% quality is diagnostic only. No learning, scientific first-use, unfamiliar transfer, checkpoint or FORGE integration."
    }
