"""R82 fresh active factual CALIBRATE versus EXPLOIT intervention value.

Independent new synthetic physical worlds, two genuinely realized paired
rollouts only in heldout scorer. FIT/validation train on ONE actually chosen
branch per world, never an unchosen target. Models learn factual causal
return-to-go from a chosen action's observable prefix. During evaluation
the agent chooses a real intervention via two forward Q estimates without
access to other-branch results; evaluator uses both physical rollouts only
after choice to calculate realized reward/regret. This is NOT independent
scientific qualification, official fused GPU or general intelligence.
"""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import torch
from torch.nn import functional as F

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r81_paired_history as r81

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r82_active_value_freeze.json"
FREEZE_BLOB="4554c4ff084b80c5b43bae6b19a884f71a2bdf28"
SCHEMA="ttt-hybrid.r82-first-active-physical-intervention-value-policy-dev.v1"
PARENT="335de19a6cc74ee7f454a6839e24044ec972c363"
BRANCH="research/r82-active-calibration-value-20261115"
SEEDS=(998621,998633)
ARMS=("NATIVE_FULL_VALUE","NATIVE_NO_HISTORY_VALUE",
      "GRU256_FULL_VALUE","MLP256_FULL_VALUE")
SALT={"FIT":25230111,"FIT_VALIDATION":25230211,
      "DEV":25230311,"OOD":25230411}
COUNT={"FIT":384,"FIT_VALIDATION":64,"DEV":128,"OOD":128}
LENGTH=64
WIDTH=1024
STEPS=512
BATCH=2
CANDIDATES=(0,2)
BLIND="NATIVE_NO_HISTORY_VALUE"


def frozen():
    if r75.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R82_FROZEN_GIT_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    m=f["model"]; w=f["physics"]; train=f["training"]; eva=f["evaluation"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_AFTER_R81_NO_GO_BEFORE_R82_SOURCE_AND_FIRST_USE"
        or f["parent"]["R81_SHA"]!=PARENT
        or f["parent"]["R81_freeze_blob"]!=r81.FREEZE_BLOB
        or f["parent"]["R81_outcome"]!="DEV_HISTORY_HYPOTHESIS_NO_GO_both_seeds"
        or m["width"]!=WIDTH or m["sequence"]!=LENGTH
        or m["batch"]!=BATCH or m["seeds"]!=list(SEEDS)
        or m["arms"]!=list(ARMS)
        or w["salts"]!=SALT or w["world_count"]!=COUNT
        or w["return_horizon"]!=9
        or w["candidate_actions"]!=["CALIBRATE_0","EXPLOIT_POS_2"]
        or train["updates"]!=STEPS
        or eva["bootstrap"]["resamples"]!=2000
        or not f["execution"]["first_attempt_only"]
        or any(f["authority"].values())):
        raise PermissionError("R82_FROZEN_SEMANTIC_SCOPE_DRIFT")
    r75.frozen()
    return f


def _world_draw(panel,index):
    if panel not in SALT:
        raise PermissionError("R82_PROTECTED_OR_SPENT_SOURCE_DENIED")
    if not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise ValueError("R82_WORLD_INDEX_OUTSIDE_FROZEN_PANEL")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],index]))
    initial=1 if index%4 in (1,3) else -1
    switched=index%4>=2
    pivot=(16 if panel=="OOD" else 12) if index%2==0 else 48
    begin=int(rng.integers(24,32)) if panel=="OOD" else int(rng.integers(20,28))
    restore=int(rng.integers(40,48)) if panel=="OOD" else int(rng.integers(36,44))
    actions=rng.choice(4,size=LENGTH,p=[.12,.18,.35,.35]).astype(np.int64)
    # Freeze paired post-decision exogenous interventions as EXACT same
    # eight factual EXPLOIT_POS actions in both realized branches.
    actions[pivot+1:pivot+9]=2
    innovations=rng.standard_normal((LENGTH,4))
    logged=int(rng.choice(CANDIDATES))
    return {"initial":initial,"switched":switched,"pivot":pivot,
            "begin":begin,"restore":restore,"actions":actions,
            "innovations":innovations,"logged_action":logged,
            "cold":bool(index%2==1 and switched)}


def _law(w,t):
    return -w["initial"] if (w["switched"] and w["begin"]<=t<w["restore"]) else w["initial"]


def _physical_step(w,t,action,state,epsilon,panel):
    law=_law(w,t)
    if action==0:
        return np.array([-.5,.05*epsilon[1],epsilon[2],epsilon[3]],np.float32),law
    if action==1:
        cue=law if panel!="OOD" else -law
        return np.array([-.18+.03*epsilon[0],epsilon[1],cue+.12*epsilon[2],epsilon[3]],np.float32),state
    orient=1 if action==2 else -1
    if state==law:
        factor,penalty=1.0,0
    elif state==0:
        factor,penalty=.35,-.15
    else:
        factor,penalty=.35,-.25
    noise=(.12 if panel!="OOD" else .20)*epsilon[0]
    return np.array([orient*law*factor+penalty+noise,
                     epsilon[1],epsilon[2],epsilon[3]],np.float32),state


def _rollout(w,panel,choice):
    pivot=w["pivot"]
    x=np.zeros((LENGTH,13),np.float32)
    reward=np.zeros(LENGTH,np.float32)
    last=np.zeros(4,np.float32)
    previous=None
    state=0
    for t in range(pivot+9):
        action=int(w["actions"][t]) if t!=pivot else choice
        row=x[t]
        row[action]=1
        if previous is not None:
            row[4+previous]=1
        row[8:12]=last
        row[12]=t/(LENGTH-1)
        observed,state=_physical_step(w,t,action,state,w["innovations"][t],panel)
        reward[t]=observed[0]
        last=observed
        previous=action
    return {"input":x,"reward":reward,
            "return":float(reward[pivot:pivot+9].sum()),
            "chosen_action":choice,"pivot":pivot,"cold":w["cold"]}


def factual_worlds(panel,pairs_override=None):
    """FIT/validation expose ONE realized factual action per world.

    DEV/OOD expose paired realized outcomes only to local evaluator after
    all training is frozen. The model never sees either paired reward while
    selecting an action on a new world.
    """
    frozen()
    if panel not in SALT:
        raise PermissionError("R82_PROTECTED_OR_SPENT_SOURCE_DENIED")
    count=COUNT[panel] if pairs_override is None else pairs_override
    if not isinstance(count,int) or not 1<=count<=COUNT[panel]:
        raise ValueError("R82_WORLD_COUNT_INVALID")
    if panel in ("FIT","FIT_VALIDATION"):
        data=[]
        for i in range(count):
            w=_world_draw(panel,i)
            data.append(_rollout(w,panel,w["logged_action"]))
        return {"panel":panel,"count":count,
                "input":np.stack([d["input"] for d in data]),
                "target":np.asarray([d["return"] for d in data],np.float32),
                "pivot":np.asarray([d["pivot"] for d in data],np.int64),
                "logged_action":np.asarray([d["chosen_action"] for d in data],np.int64),
                "cold":np.asarray([d["cold"] for d in data],bool),
                "returns_only_factual":True}
    result=[]
    for i in range(count):
        w=_world_draw(panel,i)
        a=_rollout(w,panel,0)
        b=_rollout(w,panel,2)
        # The two actual futures share the previous factual observation
        # prefix and exogenous future noise but ACTUALLY CHOOSE different
        # physical pivot actions. This pair is for evaluation only.
        result.append({"x0":a["input"],"x2":b["input"],
                       "r0":a["return"],"r2":b["return"],
                       "pivot":w["pivot"],"cold":w["cold"]})
    return {"panel":panel,"worlds":result,"count":count,
            "evaluation_only":True}


def blind_input(x):
    result=x.copy()
    result[...,4:12]=0
    return result


def models(seed):
    frozen()
    if seed not in SEEDS:
        raise ValueError("R82_UNKNOWN_FROZEN_SEED")
    torch.manual_seed(seed)
    native=r74.make_model("HybridPilot_FULL",WIDTH)
    old=native.ttt
    optimized=r75.TriangularFastWeightMemory(WIDTH,learning_rate=old.eta)
    optimized.load_state_dict(old.state_dict(),strict=True)
    native.ttt=optimized
    blind=copy.deepcopy(native)
    if any(not torch.equal(a,b) for a,b in zip(native.parameters(),blind.parameters())):
        raise RuntimeError("R82_ORIGINAL_NATIVE_WEIGHTS_NOT_IDENTICAL")
    torch.manual_seed(seed+101)
    gru=r81.GRU(256)
    torch.manual_seed(seed+102)
    mlp=r81.MLP256()
    return dict(zip(ARMS,(native,blind,gru,mlp)))


def pivot_score(model,data,blinded):
    x=blind_input(data["input"]) if blinded else data["input"]
    target=data["target"]
    pivot=data["pivot"]
    model.eval()
    losses=[]
    with torch.no_grad():
        for first in range(0,len(x),BATCH):
            n=min(BATCH,len(x)-first)
            t=torch.from_numpy(x[first:first+n])
            vals=model(t)[torch.arange(n),torch.from_numpy(pivot[first:first+n]),0]
            truth=torch.from_numpy(target[first:first+n])
            losses.extend((vals-truth).square().cpu().numpy().tolist())
    model.train()
    return float(np.mean(losses))


def train_one(model,fit,valid,indices,blinded):
    x=torch.from_numpy(blind_input(fit["input"]) if blinded else fit["input"])
    yy=torch.from_numpy(fit["target"])
    pos=torch.from_numpy(fit["pivot"])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before=pivot_score(model,valid,blinded)
    t0=time.perf_counter()
    last_loss=None
    for row in indices:
        pred=model(x[row])
        predq=pred[torch.arange(BATCH),pos[row],0]
        loss=F.mse_loss(predq,yy[row])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        last_loss=float(loss.detach())
    elapsed=time.perf_counter()-t0
    after=pivot_score(model,valid,blinded)
    if (not math.isfinite(after) or not math.isfinite(last_loss)
        or not all(bool(torch.isfinite(p).all()) for p in model.parameters())):
        raise RuntimeError("R82_NONFINITE_TRAINED_VALUE")
    return {"updates":STEPS,"logged_factual_worlds_per_update":BATCH,
            "parameters":sum(p.numel() for p in model.parameters()),
            "train_seconds":elapsed,"FIT_VAL_Q_MSE_before":before,
            "FIT_VAL_Q_MSE_after":after,"last_FIT_loss":last_loss,
            "uses_history":not blinded}


def choose(model,w,blinded):
    pivot=w["pivot"]
    x=np.stack([w["x0"][:pivot+1],w["x2"][:pivot+1]])
    if blinded:
        x=blind_input(x)
    model.eval()
    with torch.no_grad():
        predicted=model(torch.from_numpy(x))[:,-1,0].cpu().numpy()
    model.train()
    if not np.isfinite(predicted).all():
        raise RuntimeError("R82_NONFINITE_ACTION_VALUE_AT_CHOICE")
    selected=int(np.argmax(predicted))
    return CANDIDATES[selected],predicted.tolist()


def evaluate(model,pairs,blinded):
    rows=[]
    for w in pairs["worlds"]:
        action,q=choose(model,w,blinded)
        r=w["r0"] if action==0 else w["r2"]
        best=max(w["r0"],w["r2"])
        rows.append({"chosen_action":action,"realized_return":r,
                     "oracle_return":best,"regret":best-r,
                     "cold":w["cold"],"candidate_predictions":q})
    reward=np.array([d["realized_return"] for d in rows],np.float64)
    regret=np.array([d["regret"] for d in rows],np.float64)
    cold=np.array([d["cold"] for d in rows],bool)
    if cold.sum()<1:raise RuntimeError("R82_COLD_RETURN_WORLDS_MISSING")
    return {"mean_realized_return":float(reward.mean()),
            "mean_regret":float(regret.mean()),
            "mean_cold_return":float(reward[cold].mean()),
            "chosen_optimal_fraction":float(np.mean(regret<=1e-8)),
            "chosen_calibrate_fraction":float(np.mean([d["chosen_action"]==0 for d in rows])),
            "per_world_realized_return":reward.tolist(),
            "per_world_regret":regret.tolist(),
            "cold_mask":cold.tolist(),
            "world_count":len(rows)}


def constants(pairdata):
    worlds=pairdata["worlds"]
    c=np.array([x["r0"] for x in worlds])
    e=np.array([x["r2"] for x in worlds])
    oracle=np.maximum(c,e)
    coin=(c+e)/2
    cold=np.array([x["cold"] for x in worlds],bool)
    return {
        "ALWAYS_CALIBRATE":{"mean_realized_return":float(c.mean()),
                            "mean_cold_return":float(c[cold].mean())},
        "ALWAYS_EXPLOIT_POS":{"mean_realized_return":float(e.mean()),
                              "mean_cold_return":float(e[cold].mean())},
        "COIN_TOSS_EXPECTED":{"mean_realized_return":float(coin.mean()),
                               "mean_cold_return":float(coin[cold].mean()),
                               "mean_regret":float((oracle-coin).mean())},
        "ORACLE_EVALUATOR_ONLY":{"mean_realized_return":float(oracle.mean())}
    }


def lower_paired(base,treated):
    a=np.asarray(base,dtype=np.float64)
    b=np.asarray(treated,dtype=np.float64)
    if a.shape!=b.shape or len(a)!=COUNT["DEV"]:
        raise ValueError("R82_PAIRED_BOOTSTRAP_MISMATCH")
    sample=np.random.default_rng(25230555).integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((b-a)[sample].mean(axis=1),.025))


def admit(seed):
    frozen()
    if (os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")
        or os.getenv("R82_SEED")!=str(seed)):
        raise PermissionError("R82_UNAPPROVED_FIRST_USE_RUNNER")
    candidate=os.getenv("EXPECTED_CANDIDATE_SHA","")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,capture_output=True,text=True,timeout=10,check=False)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,capture_output=True,timeout=10,check=False)
    if (len(candidate)!=40 or git.returncode!=0 or git.stdout.strip()!=candidate
        or dirty.returncode!=0):
        raise PermissionError("R82_SOURCE_CANDIDATE_OR_CLEAN_HEAD_DRIFT")
    return candidate


def run(seed):
    candidate=admit(seed)
    torch.set_num_threads(1)
    receipt=ROOT/f"local_runs/r82_active_policy_seed{seed}.json"
    if receipt.exists():
        raise FileExistsError("R82_PREVIOUS_FIRST_USE_OR_UNCERTAIN_REPLAY_DENIED")
    start=time.perf_counter()
    f=frozen()
    result={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "candidate_SHA":candidate,"seed":seed,
            "arms_completed":{},"status":"INCOMPLETE",
            "DEV_OOD_first_use":False,"scientific_pass":False,
            "torch":str(torch.__version__),"platform":platform.platform()}
    try:
        fit=factual_worlds("FIT")
        valid=factual_worlds("FIT_VALIDATION")
        assert fit["returns_only_factual"] and valid["returns_only_factual"]
        arms=models(seed)
        alpha=dict(arms["NATIVE_FULL_VALUE"].named_parameters())
        beta=dict(arms["NATIVE_NO_HISTORY_VALUE"].named_parameters())
        if not all(torch.equal(alpha[n],beta[n]) for n in alpha):
            raise RuntimeError("R82_NATIVE_INITIALIZATION_CUSTODY_FAIL")
        stream=np.random.default_rng(seed+25230611)
        indices=stream.integers(0,fit["count"],size=(STEPS,BATCH))
        for arm in ARMS:
            result["arms_completed"][arm]=train_one(arms[arm],fit,valid,indices,arm==BLIND)
        if any(v["updates"]!=STEPS for v in result["arms_completed"].values()):
            raise RuntimeError("R82_INCOMPLETE_ALL_TRAINED_ARMS")
        result["weights_frozen_before_first_use"]=True
        result["DEV_OOD_first_use"]=True
        dev=factual_worlds("DEV")
        ood=factual_worlds("OOD")
        dev_scores={arm:evaluate(model,dev,arm==BLIND) for arm,model in arms.items()}
        ood_scores={arm:evaluate(model,ood,arm==BLIND) for arm,model in arms.items()}
        dev_constants=constants(dev)
        ood_constants=constants(ood)
        result["DEV"]={arm:{k:v for k,v in x.items() if k not in
            ("per_world_realized_return","per_world_regret","cold_mask")}
            for arm,x in dev_scores.items()}
        result["OOD"]={arm:{k:v for k,v in x.items() if k not in
            ("per_world_realized_return","per_world_regret","cold_mask")}
            for arm,x in ood_scores.items()}
        result["DEV_baselines"]=dev_constants
        result["OOD_baselines"]=ood_constants
        native=dev_scores["NATIVE_FULL_VALUE"]
        native_ood=ood_scores["NATIVE_FULL_VALUE"]
        blind=dev_scores[BLIND]
        trained=("GRU256_FULL_VALUE","MLP256_FULL_VALUE")
        best=max(trained,key=lambda a:dev_scores[a]["mean_realized_return"])
        conventional=dev_scores[best]
        conventional_ood=ood_scores[best]
        strong_constant=max(dev_constants[z]["mean_realized_return"]
                            for z in ("ALWAYS_CALIBRATE","ALWAYS_EXPLOIT_POS"))
        lower=lower_paired(conventional["per_world_realized_return"],
                           native["per_world_realized_return"])
        thresholds=f["evaluation"]["thresholds"]
        checks={
            "vs_best_conventional_DEV":native["mean_realized_return"]-conventional["mean_realized_return"]>=thresholds["native_DEV_reward_minus_best_trained_control_at_least"],
            "vs_best_constant_DEV":native["mean_realized_return"]-strong_constant>=thresholds["native_DEV_reward_minus_best_constant_action_at_least"],
            "vs_blind_native_DEV":native["mean_realized_return"]-blind["mean_realized_return"]>=thresholds["native_DEV_reward_minus_blind_native_at_least"],
            "vs_same_control_OOD":native_ood["mean_realized_return"]-conventional_ood["mean_realized_return"]>=thresholds["native_OOD_reward_minus_best_trained_control_at_least"],
            "vs_control_cold_DEV":native["mean_cold_return"]-conventional["mean_cold_return"]>=thresholds["native_cold_DEV_reward_minus_best_trained_control_at_least"],
            "regret_beats_coin":dev_constants["COIN_TOSS_EXPECTED"]["mean_regret"]-native["mean_regret"]>=thresholds["native_mean_regret_less_than_coin_expected_by"],
            "paired_bootstrap_lower95_positive":lower>0}
        result["comparisons"]={
            "strongest_conventional_policy_DEV":best,
            "native_minus_best_control_DEV":native["mean_realized_return"]-conventional["mean_realized_return"],
            "native_minus_best_constant_DEV":native["mean_realized_return"]-strong_constant,
            "native_minus_blind_DEV":native["mean_realized_return"]-blind["mean_realized_return"],
            "native_minus_best_control_OOD":native_ood["mean_realized_return"]-conventional_ood["mean_realized_return"],
            "native_minus_best_control_cold_DEV":native["mean_cold_return"]-conventional["mean_cold_return"],
            "coin_regret_minus_native_regret":dev_constants["COIN_TOSS_EXPECTED"]["mean_regret"]-native["mean_regret"],
            "paired_world_bootstrap_lower95_native_minus_strongest_conventional_DEV":lower}
        result["frozen_checks"]=checks
        result["status"]=(thresholds["on_success"] if all(checks.values())
                          else thresholds["on_fail"])
    except BaseException as e:
        result["status"]="INCOMPLETE_CI_OR_RESOURCE"
        result["error_class"]=type(e).__name__
        raise
    finally:
        result["wall_seconds"]=time.perf_counter()-start
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open("x",encoding="utf-8") as out:
            json.dump(result,out,sort_keys=True,indent=2,allow_nan=False)
            out.write("\n")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    args=p.parse_args()
    report=run(args.seed)
    print(json.dumps({"status":report["status"],
                      "seed":args.seed,
                      "trained_arms":len(report["arms_completed"]),
                      "DEV_OOD_first_use":report["DEV_OOD_first_use"],
                      "scientific_pass":False},sort_keys=True))
