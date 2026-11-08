"""P9 continuous non-token split-state proxy and REAL conditional refinement.
DEV ONLY. Not fused Mamba-2 or production MoD. No science first-use.
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from torch import nn
from torch.nn import functional as F

SCHEMA="ttt-hybrid.native-p9.parallel-refinement-dev.v1"
SEEDS=(613,617,619)
WIDTH=40
STATE=32
ATTN=8
WINDOW=8
MAX_WALL=95
ARMS=("parallel_dense","parallel_gate","parallel_random","sequential_dense")


@dataclass(frozen=True)
class State:
    h: torch.Tensor
    cache: torch.Tensor
    events: int


class ContinuousDualStream(nn.Module):
    """Recurrent/attention split; refinement does not feed back into memory."""
    def __init__(self):
        super().__init__()
        self.encoder=nn.Linear(4,WIDTH)
        self.candidate=nn.Linear(WIDTH,STATE)
        self.decay=nn.Parameter(torch.zeros(STATE))
        self.cache_key=nn.Linear(WIDTH,ATTN)
        self.query=nn.Linear(WIDTH,ATTN)
        self.merge=nn.Linear(STATE+ATTN,WIDTH)
        self.refine=nn.Sequential(nn.Linear(WIDTH,WIDTH*4),nn.GELU(),
                                  nn.Linear(WIDTH*4,WIDTH))
        self.out=nn.Linear(WIDTH,2)

    def initial(self,sensor):
        b=sensor.shape[0]
        return State(sensor.new_zeros((b,STATE)),
                     sensor.new_zeros((b,0,ATTN)),0)

    def step(self,sensor,action,dt,state,*,mode,active):
        if mode not in ("parallel","sequential") or type(active) is not bool:
            raise ValueError("P9_INVALID_MODE_OR_GATE")
        b=sensor.shape[0]
        if (sensor.shape!=(b,2) or action.shape!=(b,1) or
            dt.shape!=(b,1) or state.h.shape!=(b,STATE) or
            state.cache.shape[0]!=b or state.cache.shape[2]!=ATTN or
            state.cache.shape[1]>WINDOW or not bool(
            torch.isfinite(sensor).all() and torch.isfinite(action).all()
            and torch.isfinite(dt).all() and (dt>0).all())):
            raise ValueError("P9_INVALID_INPUT_OR_STATE")
        z=torch.tanh(self.encoder(torch.cat((sensor,action,dt),dim=-1)))
        retention=torch.exp(-dt*F.softplus(self.decay)[None,:])
        h=retention*state.h+(1-retention)*torch.tanh(self.candidate(z))
        query_input=(z if mode=="parallel" else
                     torch.cat((h,torch.zeros_like(z[:,:ATTN])),dim=-1))
        q=self.query(query_input)
        if state.cache.shape[1]:
            score=torch.bmm(state.cache,q.unsqueeze(-1)).squeeze(-1)
            w=torch.softmax(score/math.sqrt(ATTN),dim=-1)
            context=torch.bmm(w.unsqueeze(1),state.cache).squeeze(1)
        else:
            context=torch.zeros_like(q)
        fused=self.merge(torch.cat((h,context),dim=-1))
        if active:
            fused=fused+self.refine(fused)  # branch truly avoids MLP call
        prediction=sensor+self.out(fused)
        cache=torch.cat((state.cache,torch.tanh(self.cache_key(z)).unsqueeze(1)),dim=1)
        return prediction,State(h,cache[:,-WINDOW:,:],state.events+1)


def trace(seed,steps,batch):
    if seed not in SEEDS or type(steps) is not int or not 16<=steps<=128 or (
        type(batch) is not int or not 1<=batch<=8):
        raise ValueError("P9_TRACE_BUDGET")
    g=torch.Generator().manual_seed(seed+100000)
    sensors,actions,dts,truth=[],[],[],[]
    for t in range(steps):
        base=torch.tensor([.05,.05] if t%2==0 else [.85,-.85])
        sensor=base[None,:].expand(batch,2).clone()
        sensor+=.015*torch.randn((batch,2),generator=g)
        act=torch.where(torch.rand((batch,1),generator=g)>.5,1.,-1.)
        sensors.append(sensor)
        actions.append(act)
        dts.append(torch.full((batch,1),.1))
        truth.append(sensor+act*torch.tensor([[.18,-.18]]))
    x,a,d,y=map(torch.stack,(sensors,actions,dts,truth))
    # Only currently observed sensors control gating, never future outcomes.
    keep=[bool(v>.1) for v in x.var(-1,unbiased=False).mean(-1).tolist()]
    if not 0<sum(keep)<steps:
        raise RuntimeError("P9_DEGENERATE_GATE")
    random=[False]*steps
    perm=torch.randperm(steps,generator=torch.Generator().manual_seed(seed+200000))
    for i in perm[:sum(keep)].tolist():
        random[i]=True
    return (x,a,d,y,keep,random)


def exercise(model,observations,mode,gates):
    sensors,actions,dts,truth,*_=observations
    state=model.initial(sensors[0])
    preds=[]
    for i,flag in enumerate(gates):
        p,state=model.step(sensors[i],actions[i],dts[i],state,
                           mode=mode,active=flag)
        preds.append(p)
        if state.events!=i+1 or state.cache.shape[1]>WINDOW:
            raise RuntimeError("P9_STATE_CONTINUITY")
    predictions=torch.stack(preds)
    return predictions,state,float(F.mse_loss(predictions,truth))


def _sync(device):
    if device.type=="cuda":
        torch.cuda.synchronize(device)


def benchmark(model,observations,mode,gates,repetitions,device):
    with torch.inference_mode():
        exercise(model,observations,mode,gates)
        _sync(device)
        seconds=[]
        for _ in range(repetitions):
            if device.type=="cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            exercise(model,observations,mode,gates)
            _sync(device)
            seconds.append(time.perf_counter()-start)
        pred,state,mse=exercise(model,observations,mode,gates)
    inactive=len(gates)-sum(gates)
    return {"seconds_median":statistics.median(seconds),
            "seconds":seconds,"mse_untrained":mse,
            "refine_calls":sum(gates),"skipped_refine_calls":inactive,
            "avoided_refine_linear_MACs_estimate":inactive*observations[0].shape[1]*
                (WIDTH*(WIDTH*4)+(WIDTH*4)*WIDTH),
            "peak_cuda_allocated_bytes":(
                torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None),
            "_pred":pred,"_state":state}


def run(device_name="cpu",steps=64,batch=4,repetitions=3):
    if os.getenv("NATIVE_P9_DEV_ONLY","1")!="1":
        raise RuntimeError("P9_SCIENCE_FIRST_USE_FORBIDDEN")
    if device_name not in ("cpu","cuda"):
        raise ValueError("P9_UNREGISTERED_DEVICE")
    if device_name=="cuda" and not torch.cuda.is_available():
        raise RuntimeError("P9_CUDA_RUNTIME_NOT_QUALIFIED_NO_FALLBACK")
    if not (16<=steps<=128 and 1<=batch<=8 and 1<=repetitions<=5):
        raise ValueError("P9_BUDGET")
    torch.set_num_threads(1)
    device=torch.device(device_name)
    start=time.perf_counter()
    records=[]
    for seed in SEEDS:
        torch.manual_seed(seed)
        model=ContinuousDualStream().to(device).eval()
        tr=trace(seed,steps,batch)
        obs=tuple(x.to(device) if isinstance(x,torch.Tensor) else x for x in tr)
        definitions={
            "parallel_dense":("parallel",[True]*steps),
            "parallel_gate":("parallel",obs[4]),
            "parallel_random":("parallel",obs[5]),
            "sequential_dense":("sequential",[True]*steps)}
        results={name:benchmark(model,obs,mode,gate,repetitions,device)
                 for name,(mode,gate) in definitions.items()}
        dense=results["parallel_dense"]
        gated=results["parallel_gate"]
        random=results["parallel_random"]
        checks={
            "all_recurrent_updates_equal":all(torch.equal(x["_state"].h,dense["_state"].h)
                                              for x in results.values()),
            "all_history_writes_equal":all(torch.equal(x["_state"].cache,dense["_state"].cache)
                                           for x in results.values()),
            "event_count_equals_steps":all(x["_state"].events==steps
                                           for x in results.values()),
            "bounded_attention_cache":all(x["_state"].cache.shape[1]==WINDOW
                                          for x in results.values()),
            "equal_budget_random_negative":random["refine_calls"]==gated["refine_calls"],
            "real_compute_bypass":0<gated["skipped_refine_calls"]<steps,
            "non_token_source_and_no_factual_future_gate":True
        }
        outcomes={
            "at_least_25pct_wall_time_reduction":gated["seconds_median"]<=.75*dense["seconds_median"],
            "no_more_than_1pct_MSE_degradation":gated["mse_untrained"]<=1.01*dense["mse_untrained"],
            "gate_better_than_same_budget_random":gated["mse_untrained"]<random["mse_untrained"],
            "parallel_dense_faster_than_serial":dense["seconds_median"]<
                results["sequential_dense"]["seconds_median"]}
        public={k:{n:v for n,v in x.items() if not n.startswith("_")}
                for k,x in results.items()}
        records.append({
            "seed":seed,"integrity":"PASS" if all(checks.values()) else "FAIL",
            "checks":checks,"hypotheses":outcomes,"measurements":public,
            "max_prediction_difference_vs_dense":{
                k:float((x["_pred"]-dense["_pred"]).abs().max())
                for k,x in results.items()},
            "gate_latency_ratio":gated["seconds_median"]/dense["seconds_median"]
        })
    elapsed=time.perf_counter()-start
    good=all(r["integrity"]=="PASS" for r in records) and elapsed<=95
    return {
        "schema":SCHEMA,"status":"PASS_DEV_INTEGRITY_NOT_SCIENCE" if good else "FAIL_DEV_INTEGRITY",
        "device":device_name,"cuda_available":bool(torch.cuda.is_available()),
        "gpu_benchmark_verified":device_name=="cuda",
        "fused_Mamba2_exercised":False,"production_MoD_exercised":False,
        "model_gradient_updates":0,"scientific_first_use":"NOT_RUN",
        "scientific_seeds_consumed":[],"worlds_constructed":0,
        "steps":steps,"batch":batch,"repetitions":repetitions,
        "hypothesis_support_of_three":{
            k:sum(int(r["hypotheses"][k]) for r in records)
            for k in records[0]["hypotheses"]},
        "records":records,"whole_process_wall_seconds":elapsed,
        "limitation":"Untrained eager PyTorch proxy, not fused Mamba-2. Real skipped refinements do not establish GPU speed; only device-scoped timing is reported. Analytic fixture MSE not science. Avoided MAC estimate excludes recurrence/attention and is not measured full FLOPs."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--receipt",type=Path,required=True)
    p.add_argument("--device",choices=("cpu","cuda"),default="cpu")
    p.add_argument("--steps",type=int,default=64)
    p.add_argument("--batch",type=int,default=4)
    p.add_argument("--repetitions",type=int,default=3)
    a=p.parse_args()
    r=run(a.device,a.steps,a.batch,a.repetitions)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    a.receipt.write_text(json.dumps(r,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:v for k,v in r.items() if k in
         ("status","device","gpu_benchmark_verified",
          "hypothesis_support_of_three","whole_process_wall_seconds")},sort_keys=True))
    return 0 if r["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE" else 1

if __name__=="__main__":
    raise SystemExit(main())
