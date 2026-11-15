"""R79 exact-forward native depth-router gradient recovery (engineering only).

Original R78 4096 checkpoint is immutable; neither this probe nor its models
inherit any learned weights. We directly compare original hard-gated routing
against an additive backward-only surrogate on INACTIVE tokens, with
identical current native full-width model/parameters and fresh random inputs.
The surrogate makes no numerical change to the actual hard forward output,
but adds approximate gradients when hard routing is inactive. This is a
development mechanism probe, NOT learner improvement or scientific PASS.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import time

import torch

from model import CausalDepthRouter
from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r78_4096_continuation as r78

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r79_router_gradient_freeze.json"
FREEZE_BLOB="68f7b6272c0854f26720d527df3b10dce187ae12"
SCHEMA="ttt-hybrid.r79-rare-depth-router-causal-gradient-diagnostic.v1"
SEEDS=(997601,997607)
LARGE_WIDTH=1024
LARGE_LENGTH=128
BATCH=2
DEPTH_PARAMS=(
    "depth.norm.weight","depth.norm.bias",
    "depth.router.weight","depth.router.bias",
    "depth.ffn.0.weight","depth.ffn.0.bias",
    "depth.ffn.2.weight","depth.ffn.2.bias",
)


def git_blob(content):
    return r75.git_blob(content)


def frozen():
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R79_ORIGINAL_FROZEN_GIT_BLOB_CHANGED")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    s=f["scope"]
    if (f["schema"]!=SCHEMA
        or f["parent"]["R78_exact_SHA"]!="5b03322fb3f1e5e03b02c6bbb719ec61f4a577eb"
        or f["parent"]["R78_frozen_blob"]!=r78.FREEZE_BLOB
        or f["parent"]["R74_experiment_frozen_blob"]!=r74.FREEZE_BLOB
        or s["width"]!=LARGE_WIDTH or s["sequence"]!=LARGE_LENGTH
        or s["seed"]!=list(SEEDS) or s["batch"]!=BATCH
        or s["arms"]!=["HARD_REFERENCE","INACTIVE_SOFT_SURROGATE_STRAIGHT_THROUGH"]
        or s["forward_absolute_max_diff"]!=1e-6
        or f["stage"]!="FROZEN_AFTER_R78_HOLDOUT_FREE_FIT_RECEIPT_BEFORE_R79_ENGINEERING_IMPLEMENTATION"
        or any(f["authority"].values())):
        raise PermissionError("R79_PROSPECTIVE_SEMANTIC_FREEZE_CHANGED")
    # The exact prior model source + strict-causal native TTT algebra.
    r75.frozen()
    return f


class InactiveSoftSurrogateRouter(CausalDepthRouter):
    """Exact hard forward; additional soft backward ONLY where hard gate is off.

    The parent sparsely computes selected FFN and its selected-router STE.
    The expression [inactive * p * FFN(z)] - stop_grad(same expression)
    is exactly zero in forward and supplies gradients to norm/router/FFN
    on unselected tokens in backward. This is deliberately expensive
    dense FFN computation, not a proven efficient MoD implementation.
    """
    def forward(self,x):
        hard_result=super().forward(x)
        z=self.norm(x)
        p=torch.sigmoid(self.router(z))
        inactive=(p<0.5).detach()
        soft_off=inactive.to(dtype=x.dtype)*p*self.ffn(z)
        return hard_result + (soft_off-soft_off.detach())


def make_model(width,seed,soft):
    frozen()
    if seed not in SEEDS or width not in (32,128,LARGE_WIDTH):
        raise ValueError("R79_UNFROZEN_DIAGNOSTIC_MODEL")
    model=r75.make_original_or_optimized(width,True,seed)
    if soft:
        original=model.depth
        replacement=InactiveSoftSurrogateRouter(width)
        replacement.load_state_dict(original.state_dict(),strict=True)
        model.depth=replacement
    with torch.no_grad():
        model.depth.router.weight.zero_()
        model.depth.router.bias.fill_(-3.0)
    return model


def probe(width,sequence,seed):
    frozen()
    if width not in (32,128,LARGE_WIDTH) or sequence not in (8,32,LARGE_LENGTH):
        raise ValueError("R79_NONFROZEN_PROBE_DIMENSIONS")
    if seed not in SEEDS:
        raise ValueError("R79_NONFROZEN_SEED")
    torch.set_num_threads(1)
    torch.manual_seed(seed+909)
    inp=torch.randn(BATCH,sequence,13)
    target=torch.randn(BATCH,sequence,1)
    summary={}
    for soft in (False,True):
        model=make_model(width,seed,soft)
        tic=time.perf_counter()
        original_params={k:v.detach().clone() for k,v in model.named_parameters()}
        pred=model(inp)
        util=model.depth.last_utilization
        loss=(pred-target).square().mean()
        assert bool(torch.isfinite(loss))
        opt=torch.optim.AdamW(model.parameters(),lr=.001)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad={}
        for name in DEPTH_PARAMS:
            p=dict(model.named_parameters())[name]
            grad[name]=None if p.grad is None else float(p.grad.detach().abs().max())
        num_grads=sum(p.grad is not None for p in model.parameters())
        opt.step()
        optimizer_counters={}
        for name in DEPTH_PARAMS:
            param=dict(model.named_parameters())[name]
            state=opt.state.get(param,{})
            optimizer_counters[name]=int(state["step"]) if "step" in state else 0
        label="SURROGATE" if soft else "HARD"
        summary[label]={
            "before":original_params,
            "prediction":pred.detach().clone(),
            "MSE":float(loss.detach()),
            "elapsed_forward_backward_AdamW_seconds":time.perf_counter()-tic,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "depth_utilization":float(util),
            "depth_grad_max_abs":grad,
            "depth_optimizer_step":optimizer_counters,
            "other_parameter_grad_tensors":num_grads
        }
    old=summary["HARD"]
    new=summary["SURROGATE"]
    initial_equal=(old["before"].keys()==new["before"].keys()
        and all(torch.equal(old["before"][n],new["before"][n]) for n in old["before"]))
    output_err=float((old["prediction"]-new["prediction"]).abs().max())
    zero_original=all(v is None or v==0 for v in old["depth_grad_max_abs"].values())
    full_recovery=all(v is not None and v>0 and v<float("inf")
                      for v in new["depth_grad_max_abs"].values())
    baseline_updates=all(v==0 for v in old["depth_optimizer_step"].values())
    recovered_updates=all(v==1 for v in new["depth_optimizer_step"].values())
    maxerr=frozen()["scope"]["forward_absolute_max_diff"]
    checks={
        "identical_initial_parameters":initial_equal,
        "hard_route_always_inactive":old["depth_utilization"]==0
                                   and new["depth_utilization"]==0,
        "hard_reference_router_all_gradients_zero":zero_original,
        "hard_reference_router_optimizer_never_updates":baseline_updates,
        "soft_surrogate_all_eight_gradient_recovered":full_recovery,
        "soft_surrogate_all_eight_optimizer_updated":recovered_updates,
        "native_model_parameter_count_identical":old["parameter_count"]==new["parameter_count"]
            and (width!=LARGE_WIDTH or old["parameter_count"]==16818178),
        "forward_numeric_parity":output_err<=maxerr,
        "original_hard_inference_is_preserved":output_err<=maxerr,
    }
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"width":width,
        "sequence":sequence,"seed":seed,
        "output_absolute_difference":output_err,
        "hard_utilization":old["depth_utilization"],
        "surrogate_utilization":new["depth_utilization"],
        "hard_depth_grad_abs":old["depth_grad_max_abs"],
        "surrogate_depth_grad_abs":new["depth_grad_max_abs"],
        "hard_depth_optimizer_step":old["depth_optimizer_step"],
        "surrogate_depth_optimizer_step":new["depth_optimizer_step"],
        "full_model_parameters":old["parameter_count"],
        "hard_wall_seconds":old["elapsed_forward_backward_AdamW_seconds"],
        "surrogate_wall_seconds":new["elapsed_forward_backward_AdamW_seconds"],
        "checks":checks,"mechanistic_pass":all(checks.values())}


def original_ci_receipt():
    f=frozen()
    if os.getenv("GITHUB_ACTIONS")!="true":
        raise PermissionError("R79_ORIGINAL_CI_ONLY")
    results=[probe(LARGE_WIDTH,LARGE_LENGTH,s) for s in SEEDS]
    passed=all(x["mechanistic_pass"] for x in results)
    receipt={
      "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
      "parent_R78_SHA":f["parent"]["R78_exact_SHA"],
      "reference_model_blob":f["parent"]["R78_original_reference_model_blob"],
      "platform":platform.platform(),"torch":str(torch.__version__),
      "engine":"R75_STRICT_LOWER_TRIANGULAR_FASTWEIGHT_V1",
      "observed":results,
      "classification":("ENGINEERING_CAUSAL_ROUTER_GRADIENT_RECOVERY_ONLY"
                         if passed else "ENGINEERING_NO_GO"),
      "independent_scientific_pass":False,
      "R78_original_weights_modified":False,
      "R78_model_or_training_modified":False,
      "DEV_OOD_access":False,"GPU_firstuse":False,
      "deployment":"NONE"}
    output=ROOT/"local_runs/r79_original_router_gradient_diagnostic.json"
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("x",encoding="utf-8") as writer:
        json.dump(receipt,writer,indent=2,sort_keys=True,allow_nan=False)
        writer.write("\n")
    if not passed:
        raise AssertionError("R79_PREDECLARED_GRADIENT_MECHANISM_NO_GO")
    return receipt


def deny_external_first_use():
    raise PermissionError("R79_OFFICIAL_GPU_OR_EXTERNAL_FIRSTUSE_DENIED")
