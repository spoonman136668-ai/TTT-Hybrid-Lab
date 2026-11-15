"""R81 fresh paired factual intervention learning at max native CPU width.

Each paired physical trajectory is independently realized under one actually
selected intervention (CALIBRATE versus PROBE), with common latent law,
innovations and later exploitation actions. Neither model sees hidden state
or an unselected outcome. No R80 scored world, R78 trained weight, or R79
surrogate is read. DEV/OOD only first-generated after all weights are frozen.

This is selected-outcome prediction and observable-history diagnosis, NOT
autonomous choice of interventions or independent intelligence qualification.
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
from torch import nn

from experiments import r74_capability_ceiling as r74
from experiments import r75_cumulative_training as trainer
from experiments import r75_triangular_fastweight as r75

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r81_history_intervention_freeze.json"
FREEZE_BLOB="a65fc70ac0dcdd7b409a0164630f4b884cbdb0c4"
SCHEMA="ttt-hybrid.r81-paired-intervention-history-value-dev.v1"
PARENT="e191369166393b9f55e2108ab224399433411d39"
BRANCH="research/r81-paired-intervention-history-dependence-20261115"
SEEDS=(998431,998443)
ARMS=("NATIVE_FULL_HISTORY","NATIVE_NO_HISTORY",
      "GRU64_FULL_HISTORY","GRU256_FULL_HISTORY",
      "GRU256_NO_HISTORY","MLP256_FULL_HISTORY")
SALT={"FIT":25210111,"FIT_VALIDATION":25210211,
      "DEV":25210311,"OOD":25210411}
PAIRS={"FIT":192,"FIT_VALIDATION":32,"DEV":64,"OOD":64}
WIDTH=1024
LENGTH=64
STEPS=512
BATCH=2
PIVOT={"FIT":8,"FIT_VALIDATION":8,"DEV":8,"OOD":12}
BLIND={"NATIVE_NO_HISTORY","GRU256_NO_HISTORY"}


def frozen():
    if r75.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R81_ORIGINAL_FROZEN_GIT_BLOB_CHANGED")
    obj=json.loads(FREEZE.read_text(encoding="utf-8"))
    m=obj["models"]; w=obj["worlds"]
    if (obj["schema"]!=SCHEMA
        or obj["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R81_SOURCE_AND_FIRST_USE"
        or obj["parent"]["R80_exact_SHA"]!=PARENT
        or obj["parent"]["R80_result"]!="DEV_ROUTER_NO_GAIN_OR_NO_GO_both_seeds"
        or m["width"]!=WIDTH or m["sequence"]!=LENGTH
        or m["batch"]!=BATCH or m["seeds"]!=list(SEEDS)
        or m["arms"]!=list(ARMS)
        or w["salts"]!=SALT or w["pair_counts"]!=PAIRS
        or obj["training"]["steps"]!=STEPS
        or obj["evaluation"]["paired_bootstrap"]["resamples"]!=2000
        or any(obj["authority"].values())):
        raise PermissionError("R81_FROZEN_SOURCE_OR_SCOPE_CHANGED")
    r75.frozen()
    return obj


def paired_worlds(panel,count_override=None):
    """Factual physical paired rollout with common law/noise and chosen actions.

    Both trajectories are factual: the only difference is the action ACTUALLY
    chosen at the pivot, which changes actuator state for later exploitation.
    Only previous chosen observation and previous chosen action enter x[t].
    """
    frozen()
    if panel not in SALT:
        raise PermissionError("R81_PROTECTED_OR_SPENT_SOURCE_DENIED")
    pairs=PAIRS[panel] if count_override is None else int(count_override)
    if not 1<=pairs<=PAIRS[panel]:
        raise ValueError("R81_UNFROZEN_SOURCE_PAIR_COUNT")
    rng=np.random.default_rng(SALT[panel])
    count=2*pairs
    x=np.zeros((count,LENGTH,13),np.float32)
    y=np.zeros((count,LENGTH,1),np.float32)
    cold=np.zeros((count,LENGTH),bool)
    window=np.zeros((pairs,LENGTH),bool)
    switches=np.zeros(pairs,bool)
    pivot=PIVOT[panel]
    for k in range(pairs):
        sign=1 if k%2 else -1
        switched=k%4>=2
        switches[k]=switched
        b=int(rng.integers(22,30)) if panel!="OOD" else int(rng.integers(17,25))
        a=int(rng.integers(45,54)) if panel!="OOD" else int(rng.integers(39,48))
        innovations=rng.standard_normal((LENGTH,4))
        actions=rng.choice(4,size=LENGTH,p=[.12,.1,.39,.39]).astype(np.int64)
        # The actuator begins unset in BOTH trajectories. Before the branch,
        # block random calibration to preserve a well-defined intervention.
        actions[:pivot]=rng.choice([1,2,3],size=pivot,p=[.2,.4,.4])
        # Common downstream exploitation preserves the state distinction.
        actions[pivot+1:pivot+9]=rng.choice([2,3],size=8,p=[.5,.5])
        window[k,pivot+1:pivot+9]=True
        for arm in (0,1):
            row_id=2*k+arm
            chosen=actions.copy()
            chosen[pivot]=0 if arm==0 else 1
            state=0
            previous_action=None
            observed=np.zeros(4,np.float32)
            for t in range(LENGTH):
                law=-sign if switched and b<=t<a else sign
                action=int(chosen[t])
                row=x[row_id,t]
                row[action]=1.0
                if previous_action is not None:
                    row[4+previous_action]=1.0
                row[8:12]=observed
                row[12]=t/(LENGTH-1)
                eps=innovations[t]
                if action==0:
                    actual=np.array([-.28+.02*eps[0],eps[1],eps[2],eps[3]],
                                    dtype=np.float32)
                    state=law
                elif action==1:
                    cue=law if panel!="OOD" else -law
                    actual=np.array([-.16+.02*eps[0],eps[1],
                        cue+.14*eps[2],eps[3]],dtype=np.float32)
                else:
                    orient=1 if action==2 else -1
                    if state==law:
                        amp,penalty=1.0,0.0
                    elif state==0:
                        amp,penalty=.4,-.2
                    else:
                        amp,penalty=.4,-.7
                    noise=(.55 if panel!="OOD" else .85)*eps[0]
                    actual=np.array([orient*law*amp+penalty+noise,
                        eps[1],eps[2],eps[3]],dtype=np.float32)
                y[row_id,t,0]=actual[0]
                observed=actual
                previous_action=action
                cold[row_id,t]=bool(switched and t>=a)
    return {"input":x,"selected_reward":y,"cold_A_return_mask":cold,
            "pair_window":window,"switched_pairs":switches,
            "pairs":pairs,"source_panel":panel,"source_count":count}


def blinded_worlds(source):
    """Only prior public observation/action feature fields erased.

    Native recurrent states may still remember prior CURRENT actions. Thus
    this is input-history-feature ablation, NOT a complete state reset or
    proof of pure causal reasoning.
    """
    result={**source}
    x=source["input"].copy()
    x[:,:,4:12]=0
    result["input"]=x
    return result


class GRU(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.gru=nn.GRU(13,width,batch_first=True)
        self.head=nn.Linear(width,1)
    def forward(self,x):
        return self.head(self.gru(x)[0])


class MLP256(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(13,256),nn.Tanh(),
            nn.Linear(256,256),nn.Tanh(),nn.Linear(256,1))
    def forward(self,x):
        return self.net(x)


def models(seed):
    frozen()
    if seed not in SEEDS:
        raise ValueError("R81_UNFROZEN_TRAIN_SEED")
    torch.manual_seed(seed)
    model=r74.make_model("HybridPilot_FULL",WIDTH)
    old=model.ttt
    new=r75.TriangularFastWeightMemory(WIDTH,learning_rate=old.eta)
    new.load_state_dict(old.state_dict(),strict=True)
    model.ttt=new
    blind_native=copy.deepcopy(model)
    native_params=dict(model.named_parameters())
    blind_params=dict(blind_native.named_parameters())
    if native_params.keys()!=blind_params.keys() or any(
        not torch.equal(native_params[k],blind_params[k]) for k in native_params
    ):
        raise RuntimeError("R81_INITIAL_NATIVE_TENSORS_DIFFER")
    torch.manual_seed(seed+101)
    gru64=GRU(64)
    torch.manual_seed(seed+102)
    gru256=GRU(256)
    gru256_blind=copy.deepcopy(gru256)
    torch.manual_seed(seed+103)
    mlp=MLP256()
    return dict(zip(ARMS,(model,blind_native,gru64,gru256,
                          gru256_blind,mlp)))


def prefix_negative(model,source):
    model.eval()
    inp=torch.from_numpy(source["input"][:2])
    with torch.no_grad():
        original=model(inp)
        altered=inp.clone()
        mid=LENGTH//2
        altered[:,mid:]+=11
        changed=model(altered)
        shorter=model(inp[:,:mid])
        err=max(float((original[:,:mid]-changed[:,:mid]).abs().max()),
                float((original[:,:mid]-shorter).abs().max()))
    model.train()
    return err


def metrics(model,source,blind):
    model.eval()
    inp=blinded_worlds(source)["input"] if blind else source["input"]
    x=torch.from_numpy(inp)
    y=torch.from_numpy(source["selected_reward"])
    pred=np.empty_like(source["selected_reward"])
    with torch.no_grad():
        for i in range(0,len(x),2):
            out=model(x[i:i+2])
            if not bool(torch.isfinite(out).all()):
                raise RuntimeError("R81_NONFINITE_PREDICTION")
            pred[i:i+2]=out.detach().cpu().numpy()
    model.train()
    errs=(pred[:,:,0]-source["selected_reward"][:,:,0])**2
    per_world=errs.mean(axis=1).astype(float)
    pair_means=per_world.reshape(-1,2).mean(axis=1)
    cold=source["cold_A_return_mask"]
    if int(cold.sum())<=0:
        raise RuntimeError("R81_MISSING_COLD_RETURN")
    cold_mse=float(errs[cold].mean())
    p=source["pairs"]
    branch=source["pair_window"]
    predicted_delta=pred.reshape(p,2,LENGTH,1)[:,0,:,0]-pred.reshape(p,2,LENGTH,1)[:,1,:,0]
    target_delta=source["selected_reward"].reshape(p,2,LENGTH,1)[:,0,:,0]-source["selected_reward"].reshape(p,2,LENGTH,1)[:,1,:,0]
    contrast_error=(predicted_delta-target_delta)**2
    pair_contrast=contrast_error[branch].reshape(p,8).mean(axis=1)
    zero_baseline=float((target_delta[branch]**2).mean())
    return {"MSE":float(per_world.mean()),
            "cold_A_return_MSE":cold_mse,
            "paired_intervention_contrast_MSE":float(pair_contrast.mean()),
            "zero_contrast_baseline_MSE":zero_baseline,
            "per_pair_MSE":pair_means.tolist(),
            "worlds":len(per_world),"pairs":p}


def bootstrap_lower(base,treated):
    a=np.asarray(base,dtype=np.float64)
    b=np.asarray(treated,dtype=np.float64)
    if a.shape!=b.shape or len(a)!=PAIRS["DEV"]:
        raise ValueError("R81_BOOTSTRAP_PAIRED_WORLD_MISMATCH")
    rng=np.random.default_rng(25210555)
    sample=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[sample].mean(axis=1),.025))


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
        or os.getenv("R81_SEED")!=str(seed)):
        raise PermissionError("R81_ONLY_FIRST_EXACT_BRANCH_HOSTED_CPU")
    candidate=os.getenv("EXPECTED_CANDIDATE_SHA","")
    h=subprocess.run(["git","rev-parse","HEAD"],cwd=str(ROOT),
        text=True,capture_output=True,check=False,timeout=10)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=str(ROOT),
        capture_output=True,check=False,timeout=10)
    if len(candidate)!=40 or h.returncode!=0 or h.stdout.strip()!=candidate or dirty.returncode!=0:
        raise PermissionError("R81_EXACT_CLEAN_GIT_HEAD_MISMATCH")
    return candidate


def run(seed):
    """Train all 6 models on fresh paired FIT then FIRST generate new DEV/OOD."""
    expected=host_admit(seed)
    torch.set_num_threads(1)
    receipt=ROOT/f"local_runs/r81_paired_history_seed{seed}.json"
    if receipt.exists():
        raise FileExistsError("R81_ALREADY_SCORED_OR_UNCERTAIN_REPLAY_DENIED")
    t0=time.perf_counter()
    spec=frozen()
    result={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "candidate_SHA":expected,"seed":seed,"train_updates":STEPS,
        "classification":"INCOMPLETE","arms_completed":{},
        "DEV_OOD_firstuse":False,"scientific_pass":False,
        "platform":platform.platform(),"torch":str(torch.__version__)}
    try:
        fit=paired_worlds("FIT")
        val=paired_worlds("FIT_VALIDATION")
        blind_fit=blinded_worlds(fit)
        blind_val=blinded_worlds(val)
        components=models(seed)
        # Same initial native full and blind (only observable data differs).
        original=dict(components["NATIVE_FULL_HISTORY"].named_parameters())
        compare=dict(components["NATIVE_NO_HISTORY"].named_parameters())
        if any(not torch.equal(original[k],compare[k]) for k in original):
            raise RuntimeError("R81_NATIVE_WEIGHT_EQUALITY_NEGATIVE")
        prefix={}
        for arm,model in components.items():
            prefix[arm]=prefix_negative(model,blind_fit if arm in BLIND else fit)
        if any(v>1e-5 for v in prefix.values()):
            raise RuntimeError("R81_CAUSAL_PREFIX_NEGATIVE")
        result["causal_prefix_max_abs"]=prefix
        rng=np.random.default_rng(seed+25210611)
        selected=rng.integers(0,fit["source_count"],size=(STEPS,BATCH))
        for arm,model in components.items():
            blinded=arm in BLIND
            src=blind_fit if blinded else fit
            valid=blind_val if blinded else val
            start_score=trainer.score_fit_validation(model,valid)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            begin=time.perf_counter()
            for i in range(STEPS):
                trainer.one_fit_update(model,opt,src,selected[i])
            elapsed=time.perf_counter()-begin
            after_score=trainer.score_fit_validation(model,valid)
            if (not math.isfinite(after_score)
                or not all(bool(torch.isfinite(v).all()) for v in model.parameters())):
                raise ValueError("R81_NONFINITE_MODEL_OR_FIT_VALIDATION")
            result["arms_completed"][arm]={
                "updates":STEPS,"selected_factual_events":STEPS*BATCH*LENGTH,
                "parameters":sum(p.numel() for p in model.parameters()),
                "FIT_VALIDATION_initial":start_score,
                "FIT_VALIDATION_final":after_score,
                "trainer_seconds":elapsed,
                "history_features_erased":blinded}
        # First holdout access only after all model optimizer steps complete.
        result["all_model_weights_frozen_before_DEV_OOD"]=True
        result["DEV_OOD_firstuse"]=True
        dev=paired_worlds("DEV")
        ood=paired_worlds("OOD")
        raw={}
        for arm,model in components.items():
            raw[arm]={
                "DEV":metrics(model,dev,arm in BLIND),
                "OOD":metrics(model,ood,arm in BLIND)}
        result["scores"]={arm:{p:{k:v for k,v in s.items() if k!="per_pair_MSE"}
                               for p,s in scans.items()} for arm,scans in raw.items()}
        full=raw["NATIVE_FULL_HISTORY"]
        no=raw["NATIVE_NO_HISTORY"]
        conventional=("GRU64_FULL_HISTORY","GRU256_FULL_HISTORY",
                      "GRU256_NO_HISTORY","MLP256_FULL_HISTORY")
        best=min(conventional,key=lambda a:raw[a]["DEV"]["MSE"])
        base=raw[best]
        def gain(panel,key,control):
            ref=control[panel][key]
            val=full[panel][key]
            if not math.isfinite(ref) or ref<=0 or not math.isfinite(val):
                raise ValueError("R81_INVALID_HOLDOUT_LOSS")
            return (ref-val)/ref
        lower=bootstrap_lower(no["DEV"]["per_pair_MSE"],full["DEV"]["per_pair_MSE"])
        gate=spec["frozen_gates"]
        comparisons={
            "full_vs_no_history_DEV":gain("DEV","MSE",no),
            "full_vs_no_history_OOD":gain("OOD","MSE",no),
            "full_vs_no_history_cold_DEV":gain("DEV","cold_A_return_MSE",no),
            "full_vs_no_history_paired_contrast_DEV":gain("DEV","paired_intervention_contrast_MSE",no),
            "full_vs_zero_contrast_DEV":(full["DEV"]["zero_contrast_baseline_MSE"]-
                 full["DEV"]["paired_intervention_contrast_MSE"])/full["DEV"]["zero_contrast_baseline_MSE"],
            "full_vs_strongest_conventional_DEV":gain("DEV","MSE",base),
            "paired_bootstrap_lower95_full_vs_blind_DEV":lower}
        checks={
            "better_DEV_than_blind":comparisons["full_vs_no_history_DEV"]>=gate["NATIVE_FULL_vs_NO_HISTORY_DEV_relative_MSE_gain_min"],
            "better_OOD_than_blind":comparisons["full_vs_no_history_OOD"]>=gate["NATIVE_FULL_vs_NO_HISTORY_OOD_relative_MSE_gain_min"],
            "better_cold_than_blind":comparisons["full_vs_no_history_cold_DEV"]>=gate["NATIVE_FULL_vs_NO_HISTORY_cold_return_relative_MSE_gain_min"],
            "better_intervention_contrast_than_blind":comparisons["full_vs_no_history_paired_contrast_DEV"]>=gate["NATIVE_FULL_vs_NO_HISTORY_paired_intervention_contrast_MSE_gain_min"],
            "better_intervention_contrast_than_zero":comparisons["full_vs_zero_contrast_DEV"]>=gate["NATIVE_FULL_vs_zero_contrast_baseline_gain_min"],
            "better_DEV_than_best_conventional":comparisons["full_vs_strongest_conventional_DEV"]>=gate["NATIVE_FULL_vs_strongest_trained_conventional_DEV_MSE_gain_min"],
            "bootstrap_lower95_positive":lower>0}
        result["comparisons"]=comparisons
        result["best_trained_conventional_DEV"]=best
        result["frozen_checks"]=checks
        result["classification"]=(gate["classification_if_pass"]
            if all(checks.values()) else gate["classification_if_fail"])
    except BaseException as exc:
        result["classification"]="INCOMPLETE_CI_OR_RESOURCE"
        result["exception_type"]=type(exc).__name__
        raise
    finally:
        result["total_wall_seconds"]=time.perf_counter()-t0
        receipt.parent.mkdir(exist_ok=True,parents=True)
        with receipt.open("x",encoding="utf-8") as handle:
            json.dump(result,handle,indent=2,sort_keys=True,allow_nan=False)
            handle.write("\n")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,choices=SEEDS,required=True)
    args=parser.parse_args()
    result=run(args.seed)
    print(json.dumps({"seed":args.seed,"status":result["classification"],
                      "completed_arms":len(result["arms_completed"]),
                      "DEV_OOD_firstuse":result["DEV_OOD_firstuse"],
                      "scientific_pass":False},sort_keys=True))
