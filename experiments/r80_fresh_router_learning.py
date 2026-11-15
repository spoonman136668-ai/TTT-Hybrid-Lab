"""R80 fresh factual router learning diagnostic. No R74/R78 cohorts or weights.

Prospective original freeze is independent of R64-R79 scored source. Tokenless
numeric A-B-A hidden causal law with actual selected (never counterfactual)
observations; world splits are generated only by distinct frozen fresh salts.
No causal policy optimizer; predicted target is factual selected reward.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch

from model import Configuration, HybridPilot
from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as trainer
from experiments import r75_triangular_fastweight as engine
from experiments import r79_router_gradient as r79

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r80_router_learning_freeze.json"
FREEZE_BLOB="57edf593e0b68dfd2c84199cb9c762244b7d7cc3"
SCHEMA="ttt-hybrid.r80-fresh-hard-versus-surrogate-learning-dev.v1"
PARENT="9bbbe2f40c3c8016af7a4eb671ba27d57bbbfb9e"
BRANCH="research/r80-fresh-matched-router-learning-20261115"
SEEDS=(998211,998223)
ARMS=("HARD_NATIVE","SURROGATE_NATIVE","GRU64","MLP64")
WIDTH=1024
LENGTH=64
STEPS=512
BATCH=2
SALT={"FIT":25180111,"FIT_VALIDATION":25180211,
      "DEV":25180311,"OOD":25180411}
COUNT={"FIT":384,"FIT_VALIDATION":64,"DEV":96,"OOD":96}
REPORT_DIR=ROOT/"local_runs"


def git_blob(raw):
    return r79.git_blob(raw)


def frozen():
    if git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R80_ORIGINAL_PROSPECTIVE_FREEZE_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    d=f["design"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PRE_FROZEN_BEFORE_R80_FRESH_SOURCE_IMPLEMENTATION_AND_FIRST_USE"
        or f["parent"]["R79_exact_SHA"]!=PARENT
        or f["parent"]["R79_freeze_blob"]!=r79.FREEZE_BLOB
        or d["seeds"]!=list(SEEDS)
        or d["width"]!=WIDTH or d["sequence"]!=LENGTH
        or d["batch"]!=BATCH or d["train_updates"]!=STEPS
        or d["arms"]!=list(ARMS)
        or d["source_salts"]!=SALT
        or d["source_sizes"]!=COUNT
        or any(f["authority"].values())):
        raise PermissionError("R80_ORIGINAL_FROZEN_SEMANTIC_DRIFT")
    r79.frozen()
    return f


def fresh_worlds(panel,count_override=None,sequence=LENGTH):
    """Fresh real selected action->outcome only; never unchosen outcomes.

    Current x includes chosen action, *previous* selected public observation,
    previous chosen action and current public time. Current hidden law and
    observed reward cannot enter its features at the current timestep.
    """
    frozen()
    if panel not in SALT:
        raise PermissionError("R80_PROTECTED_OR_SPENT_SOURCE_DENIED")
    if sequence not in (8,32,LENGTH):
        raise ValueError("R80_UNFROZEN_SOURCE_SEQUENCE")
    count=COUNT[panel] if count_override is None else count_override
    if count<1 or count>COUNT[panel]:
        raise PermissionError("R80_UNFROZEN_SOURCE_COUNT")
    rng=np.random.default_rng(SALT[panel])
    inp=np.zeros((count,sequence,13),dtype=np.float32)
    outcome=np.zeros((count,sequence,1),dtype=np.float32)
    cold=np.zeros((count,sequence),bool)
    switches=np.zeros(count,bool)
    for w in range(count):
        initial=1 if w%2 else -1
        switched=w%4>=2
        switches[w]=switched
        b=int(rng.integers(max(1,int(.25*sequence)),
                           max(2,int(.45*sequence))))
        a=int(rng.integers(max(b+1,int(.70*sequence)),
                           max(b+2,int(.85*sequence))))
        if panel=="OOD":
            # Prespecified unfamiliar transition cadence, cue inversion,
            # exploitation noise/cost. Held-out after model training only.
            b=max(1,int(.18*sequence))
            a=max(b+1,int(.64*sequence))
        state=0
        previous_action=None
        previous_obs=np.zeros(4,dtype=np.float32)
        for t in range(sequence):
            law=(-initial if switched and b<=t<a else initial)
            action=int(rng.choice(4,p=[.12,.10,.39,.39]))
            row=inp[w,t]
            row[action]=1.0
            if previous_action is not None:
                row[4+previous_action]=1.0
            row[8:12]=previous_obs
            row[12]=t/max(1,sequence-1)
            eps=rng.standard_normal(4)
            obs=np.zeros(4,dtype=np.float32)
            if action==0:  # factual state-changing calibration, costed
                obs[:]=[-.26+.03*eps[0],.15*eps[1],eps[2],eps[3]]
                state=law
            elif action==1:  # factual cue, hidden law never exposed directly
                cue=(law if panel!="OOD" else -law)
                obs[:]=[-.18+.03*eps[0],eps[1],cue+.20*eps[2],eps[3]]
            else:
                direction=1 if action==2 else -1
                if state==law:
                    amp,penalty=1.0,0.0
                elif state==0:
                    amp,penalty=.42,-.20
                else:
                    amp,penalty=.42,-.72
                noise=(.62 if panel!="OOD" else .87)*eps[0]
                obs[:]=[direction*law*amp+penalty+noise,
                        eps[1],eps[2],eps[3]]
            outcome[w,t,0]=obs[0]
            previous_obs=obs
            previous_action=action
            cold[w,t]=bool(switched and t>=a)
    return {"input":inp,"selected_reward":outcome,
            "cold_A_return_mask":cold,"world_switched":switches,
            "panel":panel,"count":count}


def factual_negative():
    d=fresh_worlds("FIT",count_override=4,sequence=32)
    x,y=d["input"],d["selected_reward"]
    if (x.shape!=(4,32,13) or y.shape!=(4,32,1)
        or np.shares_memory(x,y)
        or not bool(np.all(x[:,0,8:12]==0))
        or not bool(np.all(x[:,1:,8]==y[:,:-1,0]))):
        raise AssertionError("R80_CURRENT_SELECTED_OUTCOME_LEAKAGE")
    # A world with swapped future targets must not change x prefix.
    alternative=y.copy()
    alternative[:,16:]*=-1
    assert np.array_equal(x[:,:16],d["input"][:,:16])
    assert not np.array_equal(y,alternative)
    return True


def models_for_seed(seed):
    frozen()
    if seed not in SEEDS:
        raise ValueError("R80_UNFROZEN_TRAIN_SEED")
    torch.manual_seed(seed)
    native=HybridPilot(13,1,Configuration(width=WIDTH,heads=4))
    old=native.ttt
    upgraded=engine.TriangularFastWeightMemory(WIDTH,learning_rate=old.eta)
    upgraded.load_state_dict(old.state_dict(),strict=True)
    native.ttt=upgraded
    surrogate=copy.deepcopy(native)
    original=surrogate.depth
    replacement=r79.InactiveSoftSurrogateRouter(WIDTH)
    replacement.load_state_dict(original.state_dict(),strict=True)
    surrogate.depth=replacement
    hard={n:p.detach() for n,p in native.named_parameters()}
    soft={n:p.detach() for n,p in surrogate.named_parameters()}
    if (hard.keys()!=soft.keys()
        or not all(torch.equal(hard[n],soft[n]) for n in hard)):
        raise RuntimeError("R80_INITIAL_NATIVE_TENSORS_NOT_IDENTICAL")
    torch.manual_seed(seed+111)
    gru=r74.ConventionalGRU()
    torch.manual_seed(seed+222)
    mlp=r74.ConventionalMLP()
    return dict(zip(ARMS,(native,surrogate,gru,mlp)))


def prefix_negative(model,input_tensor):
    model.eval()
    with torch.no_grad():
        full=model(input_tensor)
        half=input_tensor.shape[1]//2
        revised=input_tensor.clone()
        revised[:,half:]+=17
        changed=model(revised)
        prefix=model(input_tensor[:,:half])
        maxerr=max(float((full[:,:half]-changed[:,:half]).abs().max()),
                   float((full[:,:half]-prefix).abs().max()))
    model.train()
    return maxerr


def score(model,data):
    model.eval()
    x=torch.from_numpy(data["input"])
    y=torch.from_numpy(data["selected_reward"])
    per=[]
    cold_num=cold_den=0.0
    with torch.no_grad():
        for start in range(0,len(x),2):
            error=(model(x[start:start+2])-y[start:start+2]).square()
            if not bool(torch.isfinite(error).all()):
                raise RuntimeError("R80_NONFINITE_HELDOUT_PREDICTION")
            val=error.detach().cpu().numpy()[:,:,0]
            per.extend(val.mean(axis=1).astype(float).tolist())
            cold=data["cold_A_return_mask"][start:start+2]
            cold_num+=float(val[cold].sum())
            cold_den+=int(cold.sum())
    model.train()
    if cold_den==0:
        raise RuntimeError("R80_COLD_RETURN_MISSING")
    return {"MSE":float(np.mean(per)),
            "cold_A_return_MSE":cold_num/cold_den,
            "per_world_MSE":per,"worlds":len(per)}


def bootstrap_lower(baseline,treated):
    a=np.asarray(baseline,dtype=np.float64)
    b=np.asarray(treated,dtype=np.float64)
    if a.shape!=b.shape or len(a)!=COUNT["DEV"]:
        raise ValueError("R80_BOOTSTRAP_SOURCE_MISMATCH")
    rng=np.random.default_rng(25180555)
    ids=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[ids].mean(axis=1),.025))


def host_admit(seed):
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")
        or os.getenv("R80_SEED")!=str(seed)):
        raise PermissionError("R80_ONLY_FIRST_EXACT_BRANCH_HOSTED_CPU_JOB")
    expected=os.getenv("EXPECTED_CANDIDATE_SHA","")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
        capture_output=True,text=True,timeout=10,check=False)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=str(ROOT),
        capture_output=True,text=True,timeout=10,check=False)
    if (len(expected)!=40 or git.returncode!=0
        or expected!=git.stdout.strip() or dirty.returncode!=0):
        raise PermissionError("R80_EXACT_SOURCE_CLEAN_HEAD_REQUIRED")
    return expected


def run(seed):
    """Actual fresh FIT train, then and only then first DEV/OOD generation."""
    candidate=host_admit(seed)
    torch.set_num_threads(1)
    output=REPORT_DIR/f"r80_fresh_router_learning_seed{seed}.json"
    if output.exists():
        raise FileExistsError("R80_DUPLICATE_SCIENCE_RECEIPT_DENIED")
    start=time.perf_counter()
    spec=frozen()
    report={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "candidate_SHA":candidate,"seed":seed,
        "arms":list(ARMS),"train_updates_target":STEPS,
        "completed_training_arms":{},"status":"INCOMPLETE",
        "DEV_OOD_firstuse":False,"scientific_pass":False,
        "device":"cpu","torch":str(torch.__version__),
        "platform":platform.platform()}
    try:
        fit=fresh_worlds("FIT")
        valid=fresh_worlds("FIT_VALIDATION")
        models=models_for_seed(seed)
        sample=torch.from_numpy(fit["input"][:2])
        before=models["HARD_NATIVE"](sample).detach()
        after=models["SURROGATE_NATIVE"](sample).detach()
        parity=float((before-after).abs().max())
        if parity>spec["design"]["acceptance"]["hard_forward_matches_surrogate_before_training_abs_error_le"]:
            raise RuntimeError("R80_INITIAL_FORWARD_PARITY_NOT_QUALIFIED")
        prefix={arm:prefix_negative(model,sample) for arm,model in models.items()}
        if any(value>1e-5 for value in prefix.values()):
            raise RuntimeError("R80_CAUSAL_PREFIX_NEGATIVE_FAIL")
        report["initial_forward_abs_diff"]=parity
        report["initial_prefix_max_abs"]=prefix
        stream=np.random.default_rng(seed+25180611)
        preselected=stream.integers(0,fit["count"],size=(STEPS,BATCH))
        # Same factual samples for ALL four arms, deterministic before training.
        for arm in ARMS:
            model=models[arm]
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            init=trainer.score_fit_validation(model,valid)
            tic=time.perf_counter()
            for step in range(STEPS):
                trainer.one_fit_update(model,opt,fit,preselected[step])
            secs=time.perf_counter()-tic
            fini=trainer.score_fit_validation(model,valid)
            if (not math.isfinite(fini)
                or any(not bool(torch.isfinite(p).all()) for p in model.parameters())):
                raise RuntimeError("R80_INCOMPLETE_OR_NONFINITE_TRAINING")
            grad_steps={}
            if arm in ("HARD_NATIVE","SURROGATE_NATIVE"):
                for name in r79.DEPTH_PARAMS:
                    param=dict(model.named_parameters())[name]
                    state=opt.state.get(param,{})
                    grad_steps[name]=(int(state["step"]) if "step" in state else 0)
            report["completed_training_arms"][arm]={
                "updates":STEPS,"selected_factual_events":STEPS*BATCH*LENGTH,
                "params":sum(p.numel() for p in model.parameters()),
                "FIT_VALIDATION_initial":init,"FIT_VALIDATION_final":fini,
                "train_seconds":secs,
                "depth_optimizer_steps":grad_steps}
        # All four weights are now frozen. Do not retrain or tune from scoring.
        report["all_weights_frozen_before_scoring"]=True
        report["DEV_OOD_firstuse"]=True
        dev=fresh_worlds("DEV")
        ood=fresh_worlds("OOD")
        reports={}
        for arm in ARMS:
            reports[arm]={"DEV":score(models[arm],dev),
                          "OOD":score(models[arm],ood)}
        report["scores"]={
            arm:{split:{k:v for k,v in metric.items() if k!="per_world_MSE"}
                 for split,metric in values.items()}
            for arm,values in reports.items()}
        best=min(("GRU64","MLP64"),key=lambda a:reports[a]["DEV"]["MSE"])
        hard,soft=reports["HARD_NATIVE"],reports["SURROGATE_NATIVE"]
        control=reports[best]
        def gain(p,metric,baseline):
            x=baseline[p][metric]
            return (x-soft[p][metric])/x if x>0 else float("-inf")
        lower=bootstrap_lower(control["DEV"]["per_world_MSE"],soft["DEV"]["per_world_MSE"])
        rules=spec["design"]["acceptance"]
        checks={
            "surrogate_vs_hard_DEV":gain("DEV","MSE",hard)>=rules["surrogate_vs_hard_DEV_relative_MSE_reduction_at_least"],
            "surrogate_vs_hard_OOD":gain("OOD","MSE",hard)>=rules["surrogate_vs_hard_OOD_relative_MSE_reduction_at_least"],
            "surrogate_vs_hard_cold":gain("DEV","cold_A_return_MSE",hard)>=rules["surrogate_vs_hard_cold_A_return_relative_MSE_reduction_at_least"],
            "surrogate_vs_strongest_conventional_DEV":gain("DEV","MSE",control)>=rules["surrogate_vs_strongest_trained_conventional_DEV_relative_MSE_reduction_at_least"],
            "bootstrap_lower_positive":lower>0}
        report["comparison"]={
            "strongest_matched_512_step_conventional_control":best,
            "surrogate_vs_hard_DEV_relative_gain":gain("DEV","MSE",hard),
            "surrogate_vs_hard_OOD_relative_gain":gain("OOD","MSE",hard),
            "surrogate_vs_hard_cold_relative_gain":gain("DEV","cold_A_return_MSE",hard),
            "surrogate_vs_conventional_DEV_relative_gain":gain("DEV","MSE",control),
            "paired_DEV_bootstrap_lower95":lower,
            "frozen_checks":checks}
        report["status"]="DEV_ROUTER_PREDICTION_SIGNAL_ONLY" if all(checks.values()) else "DEV_ROUTER_NO_GAIN_OR_NO_GO"
        report["scientific_pass"]=False
    except BaseException as exc:
        report["status"]="INCOMPLETE_EXECUTION"
        report["error_type"]=type(exc).__name__
        raise
    finally:
        report["wall_seconds"]=time.perf_counter()-start
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open("x",encoding="utf-8") as h:
            json.dump(report,h,sort_keys=True,indent=2,allow_nan=False)
            h.write("\n")
    return report


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    args=p.parse_args()
    result=run(args.seed)
    print(json.dumps({"seed":args.seed,"status":result["status"],
        "updates":result["train_updates_target"],
        "completed_arms":len(result["completed_training_arms"]),
        "DEV_OOD_firstuse":result["DEV_OOD_firstuse"],
        "scientific_pass":False},sort_keys=True))
