"""R74 max-first native numeric HybridPilot capability and backward scaling.

Two separate axes:
(1) HIGHEST-FIRST CPU process-isolated resource *engineering* feasibility;
(2) sustained selected-consequence training with frozen FIT/VALID source and
    completely withheld DEV/OOD until all final candidate/control weights
    are frozen. Neither implies official Mamba2/TTT CUDA qualification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

from model import Configuration, HybridPilot

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r74_capability_ceiling_freeze.json"
FREEZE_BLOB="7deb2cd19bcd06ef0800e79a1588698c74ae350d"
MODEL_BLOB="5d9ea57b827aa1796548447868fa5ee14c251b49"
SCHEMA="ttt-hybrid.r74-capability-ceiling.v1"
LEVELS=(1024,768,512,384,256,128,64)
LENGTHS=(128,64,32)
INPUT_DIM=13
OUTPUT_DIM=1
ACTIONS=4
TRAIN_STEPS=(1024,4096,16384,65536)
SEEDS=(997601,997607)
TRAIN_ARMS=("HybridPilot_FULL","HybridPilot_NO_TTT","HybridPilot_NO_ATTENTION",
            "HybridPilot_NO_SSM","HybridPilot_NO_DEPTH","GRU64","MLP64")


def git_blob(raw):
    """Verify original canonical Git text blob on Windows and Unix.

    Git's Windows checkout can materialize CRLF while its committed object is
    LF. Normalize ONLY that representation difference, never changes to
    source content, constants, or protocol. Reject other stray CR bytes.
    """
    canonical=raw.replace(b"\r\n",b"\n")
    if b"\r" in canonical:
        raise RuntimeError("R74_UNSUPPORTED_TEXT_CARRIAGE_RETURN")
    return hashlib.sha1(b"blob "+str(len(canonical)).encode()+b"\0"+canonical).hexdigest()


def frozen():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB or git_blob((ROOT/"model.py").read_bytes())!=MODEL_BLOB:
        raise RuntimeError("R74_ORIGINAL_FREEZE_OR_HYBRID_SOURCE_DRIFT")
    spec=json.loads(raw)
    cap=spec["capacity"]
    learn=spec["learning"]
    if (spec["schema"]!=SCHEMA
        or spec["stage"]!="PRE_FROZEN_BEFORE_SOURCE_OR_PROBES"):
        raise RuntimeError("R74_SCHEMA_DRIFT")
    if (cap["widths"]!=list(LEVELS)
        or cap["sequence_lengths"]!=list(LENGTHS)
        or cap["batch"]!=2
        or cap["budget"]["workers"]!=1
        or learn["train_arms"]!=list(TRAIN_ARMS)
        or learn["seeds"]!=list(SEEDS)
        or learn["cumulative_training_checkpoints"]!=list(TRAIN_STEPS)
        or any(spec["authority"].values())):
        raise RuntimeError("R74_FROZEN_CONTRACT_DRIFT")
    return spec


def device_probe(width,length,seed=997601):
    """Actual original reference model forward/backward/optimizer, not a mock."""
    frozen()
    if width not in LEVELS and width!=32:
        raise ValueError("R74_UNFROZEN_WIDTH")
    if length not in LENGTHS and length!=8:
        raise ValueError("R74_UNFROZEN_SEQUENCE")
    torch.set_num_threads(1)
    started=time.perf_counter()
    torch.manual_seed(seed)
    model=HybridPilot(INPUT_DIM,OUTPUT_DIM,Configuration(width=width,heads=4))
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    data=torch.randn(2,length,INPUT_DIM)
    target=torch.randn(2,length,OUTPUT_DIM)
    y=model(data)
    assert y.shape==target.shape and bool(torch.isfinite(y).all())
    loss=(y-target).square().mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grads=[p.grad for p in model.parameters()]
    if not grads or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads):
        raise ValueError("R74_NONFINITE_GRADIENTS")
    opt.step()
    rss=None
    try:
        import resource
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform=="darwin":
            rss/=1024*1024
        else:
            rss/=1024
    except ImportError:
        # Native Windows GetProcessMemoryInfo; no optional paid package.
        if os.name=="nt":
            import ctypes
            from ctypes import wintypes
            class MemoryCounters(ctypes.Structure):
                _fields_=[("cb",wintypes.DWORD),
                          ("PageFaultCount",wintypes.DWORD),
                          ("PeakWorkingSetSize",ctypes.c_size_t),
                          ("WorkingSetSize",ctypes.c_size_t),
                          ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),
                          ("QuotaPagedPoolUsage",ctypes.c_size_t),
                          ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),
                          ("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                          ("PagefileUsage",ctypes.c_size_t),
                          ("PeakPagefileUsage",ctypes.c_size_t)]
            counter=MemoryCounters()
            counter.cb=ctypes.sizeof(counter)
            # ctypes defaults to 32-bit c_int; this is INVALID for a
            # 64-bit HANDLE (including pseudo-handle -1) on Windows x64.
            # Explicit WinAPI signatures prevent silent 32-bit truncation.
            kernel=ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype=wintypes.HANDLE
            psapi=ctypes.windll.psapi
            psapi.GetProcessMemoryInfo.argtypes=(
                wintypes.HANDLE,ctypes.POINTER(MemoryCounters),wintypes.DWORD)
            psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
            process=kernel.GetCurrentProcess()
            ok=psapi.GetProcessMemoryInfo(
                process,ctypes.byref(counter),counter.cb)
            if ok:
                rss=counter.PeakWorkingSetSize/(1024*1024)
        # Unknown other platform => UNVERIFIED_RSS, never pass falsely.
    return {"type":"ACTUAL_HYBRID_REFERENCE_CPU_FORWARD_BACKWARD_UPDATE",
            "width":width,"sequence":length,"batch":2,
            "parameters":sum(p.numel() for p in model.parameters()),
            "nonzero_grad_parameters":sum(p.grad is not None and bool((p.grad!=0).any()) for p in model.parameters()),
            "final_loss":float(loss.detach()),"device":"cpu","torch":torch.__version__,
            "peak_process_RSS_MiB":rss,
            "total_seconds":time.perf_counter()-started,
            "interpretation":"ENGINEERING_FEASIBILITY_ONLY_NOT_LEARNED_INTELLIGENCE"}


def high_first_probe(per_seconds=30,overall_seconds=240,max_rss_MiB=3072,
                     levels=LEVELS,lengths=LENGTHS):
    """Isolated one-worker high-first descending trial, never fake a timeout as success."""
    spec=frozen()
    upper=spec["capacity"]["budget"]
    if (per_seconds<=0 or per_seconds>upper["per_tier_seconds"]
        or overall_seconds<=0 or overall_seconds>upper["total_seconds"]
        or max_rss_MiB<=0 or max_rss_MiB>upper["max_peak_RSS_MiB"]):
        raise PermissionError("R74_CPU_RESOURCE_BUDGET_ESCALATION_DENIED")
    if (not levels or not lengths
        or any(w not in LEVELS for w in levels)
        or any(t not in LENGTHS for t in lengths)
        or list(levels)!=sorted(levels,reverse=True)
        or list(lengths)!=sorted(lengths,reverse=True)):
        raise ValueError("R74_NOT_FROZEN_HIGH_FIRST_ORDER")
    started=time.perf_counter()
    trials=[]
    winner=None
    budget_exhausted=False
    for length in lengths:
        for width in levels:
            remaining=overall_seconds-(time.perf_counter()-started)
            if remaining<=0:
                budget_exhausted=True
                break
            limit=min(remaining,per_seconds)
            argv=[sys.executable,"-m","experiments.r74_capability_ceiling","--worker",
                  "--width",str(width),"--sequence",str(length)]
            attempt=time.perf_counter()
            try:
                result=subprocess.run(argv,capture_output=True,text=True,
                         timeout=limit,check=False,cwd=str(ROOT))
                if result.returncode:
                    state="PROCESS_FAILED"
                    data={"exit_code":result.returncode,
                          "stderr_tail":result.stderr[-1500:]}
                else:
                    data=json.loads(result.stdout.strip().splitlines()[-1])
                    rss=data["peak_process_RSS_MiB"]
                    state=("PASS" if rss is not None and rss<=max_rss_MiB
                           else "UNVERIFIED_RSS" if rss is None
                           else "RSS_LIMIT")
            except subprocess.TimeoutExpired:
                state="TIMEOUT"
                data={"cap_seconds":round(limit,3)}
            except (ValueError,KeyError,json.JSONDecodeError) as e:
                state="BAD_RECEIPT"
                data={"error":str(e)}
            trials.append({"width":width,"sequence":length,"status":state,
                           "elapsed_seconds":time.perf_counter()-attempt,
                           "proof":data})
            if state=="PASS":
                winner={"width":width,"sequence":length,
                        "actual_parameters":data["parameters"]}
                break
        if winner or budget_exhausted:
            break
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "tested_cpu":platform.platform(),"python":platform.python_version(),
            "torch":torch.__version__,
            "highest_feasible_measured":winner,
            "resource_status":("MEASURED_ENGINEERING_FEASIBILITY_ONLY" if winner
                               else "INCOMPLETE_RESOURCE_OR_MEASUREMENT_LIMIT"),
            "unprobed_tiers_NOT_QUALIFIED":True,
            "GPU_and_official_backend":"NOT_TESTED",
            "scientific_learning":"NOT_TESTED",
            "budget_exhausted":budget_exhausted,
            "wall_seconds":time.perf_counter()-started,
            "trials":trials}


class ConventionalGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(INPUT_DIM,64,batch_first=True)
        self.head=nn.Linear(64,1)

    def forward(self,x):
        return self.head(self.gru(x)[0])


class ConventionalMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(INPUT_DIM,64),nn.Tanh(),
                               nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))

    def forward(self,x):
        return self.net(x)


def make_model(arm,width):
    if arm not in TRAIN_ARMS:
        raise ValueError("R74_UNREGISTERED_ARM")
    if arm=="GRU64":
        return ConventionalGRU()
    if arm=="MLP64":
        return ConventionalMLP()
    if width not in LEVELS and width!=32:
        raise ValueError("R74_UNFROZEN_WIDTH")
    cfg=Configuration(width=width,heads=4,
          ssm=arm!="HybridPilot_NO_SSM",
          attention=arm!="HybridPilot_NO_ATTENTION",
          ttt=arm!="HybridPilot_NO_TTT",
          depth=arm!="HybridPilot_NO_DEPTH")
    return HybridPilot(INPUT_DIM,OUTPUT_DIM,cfg)


def factual_worlds(panel,count_override=None,sequence=128):
    """Generate actual physical selected actions, reward and causal training input.

    Input[t] uses current exogenous chosen action and ONLY previous selected
    outcome. Hidden law/actuator state used *only* inside the environment.
    """
    spec=frozen()
    src=spec["learning"]["source"]
    if panel not in src:
        raise PermissionError("R74_PREVIOUS_OR_PROTECTED_SCIENCE_SOURCE_DENIED")
    if sequence not in LENGTHS and sequence!=8:
        raise ValueError("R74_UNFROZEN_SEQUENCE")
    n=src[panel]["count"] if count_override is None else count_override
    if n<1 or n>src[panel]["count"]:
        raise PermissionError("R74_NOT_PREDECLARED_SOURCE_VOLUME")
    rng=np.random.default_rng(src[panel]["seed"])
    input_arr=np.zeros((n,sequence,INPUT_DIM),np.float32)
    target=np.zeros((n,sequence,1),np.float32)
    cold=np.zeros((n,sequence),bool)
    switched=np.zeros(n,bool)
    for world in range(n):
        initial=-1 if world%2==0 else 1
        change=world%4>=2
        switched[world]=change
        if panel=="OOD":
            b=int(rng.integers(max(1,sequence//5),max(2,sequence//2)))
            a=int(rng.integers(max(3,sequence*2//3),sequence))
            gain_bad=.25
            reward_noise=.9
        else:
            b=int(rng.integers(max(1,sequence//3),max(2,sequence//2)))
            a=int(rng.integers(max(3,sequence*3//4),sequence))
            gain_bad=.35
            reward_noise=.7
        state=0
        previous_action=None
        previous_obs=np.zeros(4)
        for t in range(sequence):
            law=-initial if change and b<=t<a else initial
            action=int(rng.choice(4,p=[.13,.07,.4,.4]))
            x=input_arr[world,t]
            x[action]=1
            if previous_action is not None:
                x[4+previous_action]=1
            x[8:12]=previous_obs
            x[12]=t/40
            innovation=rng.standard_normal(4)
            obs=np.empty(4)
            if action==0:
                obs[:]=[-.35+.02*innovation[0],
                        law+.1*innovation[1],
                        innovation[2],innovation[3]]
                state=law
            elif action==1:
                obs[:]=[-.35+.02*innovation[0],innovation[1],
                        (law if panel in ("FIT","FIT_VALIDATION") else -law)+.1*innovation[2],
                        innovation[3]]
            else:
                orient=1 if action==2 else -1
                if state==law:
                    amplitude,penalty=1.,0.
                elif state==0:
                    amplitude,penalty=gain_bad,-.15
                else:
                    amplitude,penalty=gain_bad,-.65
                obs[:]=[amplitude*orient*law+penalty+
                        reward_noise*innovation[0],
                        innovation[1],innovation[2],innovation[3]]
            target[world,t,0]=obs[0]
            previous_action=action
            previous_obs=obs.copy()
            cold[world,t]=bool(change and t>=a)
    return {"input":input_arr,"selected_reward":target,
            "cold_A_return_mask":cold,"world_switched":switched,
            "source_panel":panel,"source_count":n}


def factual_causality_negative(panel="FIT",sequence=32):
    data=factual_worlds(panel,count_override=2,sequence=sequence)
    x=data["input"]
    y=data["selected_reward"]
    # A future target value cannot be a feature for its current decision.
    assert not np.shares_memory(x,y)
    assert np.all(x[:,0,8:12]==0)
    assert np.all(x[:,1,8]==y[:,0,0])
    assert x.shape==(2,sequence,INPUT_DIM)
    assert y.shape==(2,sequence,1)
    return True


def checkpoint_identity(arm,width,sequence,seed):
    f=frozen()
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
                       capture_output=True,text=True,check=False)
    commit=git.stdout.strip() if git.returncode==0 else "NO_GIT_CHECKOUT"
    return {"R74_freeze_blob":FREEZE_BLOB,"reference_model_blob":MODEL_BLOB,
            "R74_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "candidate_source_git_SHA":commit,
            "torch_version":str(torch.__version__),
            "arm":arm,"width":width,"sequence":sequence,"seed":seed,
            "optimizer":"AdamW","batch":2,"source_seed":f["learning"]["source"]["FIT"]["seed"]}


def train_bounded(arm,width,seed,steps,sequence=128,save_path=None):
    """CPU-only, FIT-only, cumulative training, optionally exact-custody checkpoints.

    Dev/OOD data are NEVER generated here. steps cap is frozen; checkpoint
    stores RNG + optimizer state for true incremental continuation, not replay.
    """
    spec=frozen()
    if seed not in SEEDS or steps not in TRAIN_STEPS:
        raise PermissionError("R74_UNFROZEN_SEED_OR_TRAINING_BUDGET")
    if width not in LEVELS:
        raise ValueError("R74_UNFROZEN_TRAINING_WIDTH")
    if sequence not in LENGTHS:
        raise ValueError("R74_UNREGISTERED_LENGTH")
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    model=make_model(arm,width).cpu()
    opt=torch.optim.AdamW(model.parameters(),lr=spec["learning"]["lr"],
                           weight_decay=spec["learning"]["weight_decay"])
    rng=np.random.default_rng(seed+24619999)
    identity=checkpoint_identity(arm,width,sequence,seed)
    began=time.perf_counter()
    seen=0
    history=[]
    if save_path and Path(save_path).exists():
        loaded=torch.load(Path(save_path),map_location="cpu",weights_only=True)
        if loaded["identity"]!=identity:
            raise PermissionError("R74_EXACT_CUSTODY_CHECKPOINT_MISMATCH")
        if loaded["updates"]>=steps:
            raise ValueError("R74_REFUSE_UNCERTAIN_REPLAY")
        model.load_state_dict(loaded["model_state"])
        opt.load_state_dict(loaded["optimizer_state"])
        rng.bit_generator.state=loaded["selection_rng"]
        seen=int(loaded["updates"])
        history=list(loaded["training_history"])
    fit=factual_worlds("FIT",sequence=sequence)
    val=factual_worlds("FIT_VALIDATION",sequence=sequence)
    x=torch.from_numpy(fit["input"])
    y=torch.from_numpy(fit["selected_reward"])
    vx=torch.from_numpy(val["input"])
    vy=torch.from_numpy(val["selected_reward"])
    for update in range(seen+1,steps+1):
        ids=rng.integers(0,len(x),2)
        out=model(x[ids])
        loss=(out-y[ids]).square().mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R74_NONFINITE_CAUSAL_SELECTED_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if update in TRAIN_STEPS:
            model.eval()
            with torch.no_grad():
                # Stream to avoid host memory pressure, no held-out data.
                pieces=[]
                for j in range(0,len(vx),2):
                    pieces.append(float((model(vx[j:j+2])-vy[j:j+2]).square().mean()))
                validation=float(np.mean(pieces))
            model.train()
            history.append({"updates":update,
                            "FIT_last_selected_MSE":float(loss.detach()),
                            "FIT_validation_MSE":validation,
                            "elapsed_total_seconds_this_invocation":time.perf_counter()-began})
            if save_path:
                # Preserve exact RNG, optimizer and weights before any DEV/OOD.
                payload={"identity":identity,"updates":update,
                         "model_state":model.state_dict(),
                         "optimizer_state":opt.state_dict(),
                         "selection_rng":rng.bit_generator.state,
                         "training_history":history}
                dest=Path(save_path)
                dest.parent.mkdir(exist_ok=True,parents=True)
                temp=dest.with_suffix(dest.suffix+".tmp")
                torch.save(payload,temp)
                os.replace(temp,dest)
    return {"identity":identity,
            "updates_completed":steps,"params":sum(p.numel() for p in model.parameters()),
            "seen_selected_episodes":steps*2,
            "FIT_validation_only":True,"NO_DEV_OR_OOD_ACCESSED":True,
            "CPU_only":True,"history":history,
            "total_wall_seconds":time.perf_counter()-began,
            "qualifier":"DEVELOPMENT_NOT_SCIENTIFIC_CEILING"}


def checkpoint_basename(arm,width,sequence,seed):
    if arm not in TRAIN_ARMS or seed not in SEEDS or width not in LEVELS:
        raise PermissionError("R74_UNFROZEN_CHECKPOINT_ID")
    if sequence not in LENGTHS:
        raise ValueError("R74_UNFROZEN_CHECKPOINT_LENGTH")
    return f"r74-{arm}-w{width}-t{sequence}-seed{seed}.pt"


def _score_model(model,source):
    x=torch.from_numpy(source["input"])
    y=torch.from_numpy(source["selected_reward"])
    model.eval()
    squared=[]
    with torch.no_grad():
        for i in range(0,len(x),2):
            prediction=model(x[i:i+2])
            if prediction.shape!=y[i:i+2].shape:
                raise ValueError("R74_SCORE_SHAPE_CHANGED")
            if not bool(torch.isfinite(prediction).all()):
                raise RuntimeError("R74_NONFINITE_SELECTED_REWARD_PREDICTION")
            squared.append((prediction-y[i:i+2]).square().cpu().numpy())
    error=np.concatenate(squared,axis=0)[:,:,0]
    switched=source["world_switched"]
    cold=source["cold_A_return_mask"]
    late=error[cold]
    if not np.any(switched) or not len(late):
        raise RuntimeError("R74_MISSING_CAUSAL_COLD_RETURN")
    return {"mean_selected_reward_MSE":float(np.mean(error)),
            "cold_A_return_MSE":float(np.mean(late)),
            "per_world_MSE":error.mean(axis=1).tolist(),
            "selected_events":int(error.size)}


def _bootstrap_lower_gain(baseline,hybrid,seed):
    a=np.asarray(baseline,dtype=np.float64)
    b=np.asarray(hybrid,dtype=np.float64)
    if a.shape!=b.shape or a.ndim!=1 or len(a)<64:
        raise ValueError("R74_BOOTSTRAP_SOURCE_MISMATCH")
    rng=np.random.default_rng(24649999+seed)
    ids=rng.integers(0,len(a),(2000,len(a)))
    return float(np.quantile((a-b)[ids].mean(axis=1),.025))


def final_evaluate_original_checkpoints(checkpoint_directory,width,sequence=128):
    """Never construct protected DEV/OOD until ALL 14 final model states verified.

    This executable scientific path intentionally requires 65,536 completed
    updates per arm+seed. A checkpoint that only profiled or trained 1,024
    steps is not a scientific candidate and cannot enter scored holdout.
    """
    frozen()
    if width not in LEVELS or sequence not in LENGTHS:
        raise ValueError("R74_UNFROZEN_EVALUATION_SIZE")
    directory=Path(checkpoint_directory)
    torch.set_num_threads(1)
    models={}
    custody={}
    for seed in SEEDS:
        for arm in TRAIN_ARMS:
            file=directory/checkpoint_basename(arm,width,sequence,seed)
            if not file.is_file():
                raise FileNotFoundError(f"R74_INCOMPLETE_TRAINED_MODEL:{file.name}")
            ckpt=torch.load(file,map_location="cpu",weights_only=True)
            expected=checkpoint_identity(arm,width,sequence,seed)
            if ckpt.get("identity")!=expected or ckpt.get("updates")!=TRAIN_STEPS[-1]:
                raise PermissionError("R74_ORIGINAL_TRAINING_CUSTODY_INCOMPLETE")
            model=make_model(arm,width).cpu()
            model.load_state_dict(ckpt["model_state"],strict=True)
            models[(seed,arm)]=model
            custody[f"{seed}:{arm}"]={
                "checkpoint":file.name,
                "updates":ckpt["updates"],
                "exact_identity":expected,
                "trained_parameters":sum(p.numel() for p in model.parameters())}
    # No trained weights change after this line. Only now construct the
    # precommitted held-out DEV and OOD world outcomes.
    dev=factual_worlds("DEV",sequence=sequence)
    ood=factual_worlds("OOD",sequence=sequence)
    out={}
    results=[]
    for seed in SEEDS:
        score={arm:{"DEV":_score_model(models[(seed,arm)],dev),
                    "OOD":_score_model(models[(seed,arm)],ood)}
               for arm in TRAIN_ARMS}
        out[str(seed)]={a:{panel:{k:v for k,v in d[panel].items()
                                if k!="per_world_MSE"}
                            for panel in ("DEV","OOD")}
                        for a,d in score.items()}
        baseline=min(("GRU64","MLP64"),
            key=lambda arm:score[arm]["DEV"]["mean_selected_reward_MSE"])
        candidate=score["HybridPilot_FULL"]
        base=score[baseline]
        rel_dev=(base["DEV"]["mean_selected_reward_MSE"]-candidate["DEV"]["mean_selected_reward_MSE"])/base["DEV"]["mean_selected_reward_MSE"]
        rel_ood=(base["OOD"]["mean_selected_reward_MSE"]-candidate["OOD"]["mean_selected_reward_MSE"])/base["OOD"]["mean_selected_reward_MSE"]
        rel_cold=(base["DEV"]["cold_A_return_MSE"]-candidate["DEV"]["cold_A_return_MSE"])/base["DEV"]["cold_A_return_MSE"]
        lower=_bootstrap_lower_gain(base["DEV"]["per_world_MSE"],
                                    candidate["DEV"]["per_world_MSE"],seed)
        checks={"DEV_relative_gain_min_5pct":rel_dev>=.05,
                "OOD_relative_gain_min_5pct":rel_ood>=.05,
                "cold_A_return_relative_gain_min_5pct":rel_cold>=.05,
                "paired_lower_bootstrap_positive":lower>0,
                "original_65536_training_updates_all_arms":True,
                "CPU_reference_only_no_official_GPU":True}
        results.append({"seed":seed,"strongest_conventional_control":baseline,
                        "DEV_relative_gain":rel_dev,"OOD_relative_gain":rel_ood,
                        "cold_return_relative_gain":rel_cold,
                        "paired_lower95_baseline_minus_hybrid_MSE":lower,
                        "checks":checks,"passes_frozen_checks":all(checks.values())})
    science=all(v["passes_frozen_checks"] for v in results)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "source":{"DEV_worlds":len(dev["input"]),
                      "OOD_worlds":len(ood["input"]),
                      "sequence":sequence},
            "model_provenance":custody,"scores":out,
            "decisions":results,
            "result":("DEV_REFERENCE_HYBRID_MECHANISM_SIGNAL_ONLY" if science
                      else "DEV_SCIENTIFIC_NO_GO"),
            "full_Mamba2_TTT_CUDA_qualification":False,
            "action_policy_learning_proven":False,
            "independent_scientific_pass":False,
            "deployment":"NONE"}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",action="store_true")
    p.add_argument("--profile",action="store_true")
    p.add_argument("--train",action="store_true")
    p.add_argument("--evaluate",action="store_true")
    p.add_argument("--checkpoint-directory")
    p.add_argument("--width",type=int,default=64)
    p.add_argument("--sequence",type=int,default=128)
    p.add_argument("--per-seconds",type=int,default=30)
    p.add_argument("--overall-seconds",type=int,default=240)
    p.add_argument("--max-rss",type=int,default=3072)
    p.add_argument("--arm",default="HybridPilot_FULL")
    p.add_argument("--seed",type=int,default=SEEDS[0])
    p.add_argument("--steps",type=int,default=1024)
    p.add_argument("--save-path")
    args=p.parse_args()
    if args.worker:
        ans=device_probe(args.width,args.sequence)
    elif args.profile:
        ans=high_first_probe(args.per_seconds,args.overall_seconds,args.max_rss)
    elif args.train:
        ans=train_bounded(args.arm,args.width,args.seed,args.steps,
                          args.sequence,args.save_path)
    elif args.evaluate:
        if not args.checkpoint_directory:
            p.error("--evaluate requires --checkpoint-directory")
        ans=final_evaluate_original_checkpoints(args.checkpoint_directory,
                                               args.width,args.sequence)
    else:
        p.error("choose --worker, --profile, or --train")
    print(json.dumps(ans,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
