"""R84: frozen max-width native factual-action value learning on R83-identifiable physics.

This is a new first-use development comparison, NOT official fused Mamba2
CUDA training or evidence of general intelligence. The source-neutral R83
CALIBRATION is spent, and original R83 reserved QUAL/DEV/OOD salts are
sealed. All four NEW R84 salts are independent and predeclared.

FIT / FIT_VALIDATION expose exactly ONE genuine logged factual choice and
return per world; never supply the unchosen branch, oracle or hidden sign to
the learner. The new paired DEV/OOD evaluator is constructed ONLY after
all 5 original models finished their frozen budget; the model observes
only its legal public prefix, chooses an action, and receives the actual
realized consequence of that chosen physical intervention.
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
from torch.nn import functional as F

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r81_paired_history as r81
from experiments import r83_identifiability as r83

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r84_native_value_freeze.json"
FREEZE_BLOB="39c1866ea3949f1f317c35967f1bfd2afbb1ab92"
SCHEMA="ttt-hybrid.r84-fresh-native-learned-action-value-v1"
PARENT="7bbece7433ec04c8fc5781535930a44809ae3737"
BRANCH="research/r84-fresh-max-width-cue-value-20261115"
SEEDS=(999211,999223)
ARMS=("NATIVE_FULL_VALUE","NATIVE_OBSERVATION_BLIND",
      "GRU64_FULL_VALUE","GRU256_FULL_VALUE","MLP256_FULL_VALUE")
BLIND="NATIVE_OBSERVATION_BLIND"
SALT={"FIT":25260111,"FIT_VALIDATION":25260211,
      "DEV":25260311,"OOD_SHIFT":25260411}
COUNT={"FIT":1024,"FIT_VALIDATION":128,"DEV":128,"OOD_SHIFT":128}
LENGTH=64
MAX_PIVOT=48
BATCH=2
WIDTH=1024
NATIVE_UPDATES=1024
CONTROL_UPDATES=4096


def frozen():
    if r83.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R84_ORIGINAL_PROSPECTIVE_PROTOCOL_GIT_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p=f["parent"];s=f["source"];m=f["models"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="IMMUTABLE_PROSPECTIVE_FREEZE_BEFORE_R84_SOURCE_TESTS_CI_OR_FIRST_USE"
        or p["R83_exact_SHA"]!=PARENT
        or p["R83_freeze_git_blob"]!=r83.FREEZE_BLOB
        or p["R83_RESERVED_salts_still_sealed"]!=[25250211,25250311,25250411]
        or s["new_independent_salts"]!=SALT
        or s["world_counts"]!=COUNT
        or s["seeds"]!=list(SEEDS)
        or s["length"]!=LENGTH or s["batch"]!=BATCH
        or s["early_pivot"]!=16 or s["cold_pivot"]!=48
        or s["no_R83_CALIBRATION_or_reserved_QUAL_DEV_OOD_access"] is not True
        or m["arms"]!=list(ARMS)
        or m["native_parameter_count"]!=16818178
        or m["both_native_train_updates"]!=NATIVE_UPDATES
        or m["grus_and_mlp_train_updates"]!=CONTROL_UPDATES
        or f["evaluation"]["bootstrap"]["resamples"]!=2000
        or any(f["authority"].values())):
        raise PermissionError("R84_FROZEN_SEMANTIC_OR_SOURCE_DRIFT")
    r83.frozen()
    return f


def _world(panel,seed,index):
    if panel not in SALT:
        raise PermissionError("R84_OLD_SPENT_OR_R83_SEALED_SOURCE_DENIED")
    if (seed not in SEEDS or not isinstance(index,int)
        or not 0<=index<COUNT[panel]):
        raise ValueError("R84_NONFROZEN_WORLD_INDEX_OR_SEED")
    rng=np.random.default_rng(
        np.random.SeedSequence([SALT[panel],seed,index]))
    shift=panel=="OOD_SHIFT"
    pivot=(20 if shift else r83.EARLY) if index%2==0 else (
        44 if shift else r83.COLD)
    initial=1 if rng.integers(0,2) else -1
    start=int(rng.integers(24,30)) if shift else int(rng.integers(26,32))
    restore=int(rng.integers(36,42)) if shift else int(rng.integers(38,44))
    actions=rng.choice([1,2,3],size=LENGTH,p=[.20,.40,.40]).astype(np.int64)
    actions[pivot-1]=1
    actions[pivot+1:pivot+9]=r83.EXPLOIT
    innovations=rng.standard_normal((LENGTH,4))
    if shift:
        innovations=innovations*1.35
    w={"initial":initial,"start":start,"restore":restore,
       "pivot":pivot,"actions":actions,"innovations":innovations}
    # Private action logging coin is independent of hidden law, probe cue,
    # best future branch, public time and return. Exactly one factual label.
    logged=int(rng.choice([r83.CALIBRATE,r83.EXPLOIT]))
    return w,logged


def _factual_data(panel,seed,count_override=None):
    frozen()
    if panel not in ("FIT","FIT_VALIDATION"):
        raise PermissionError("R84_ONLY_FIT_SOURCE_IN_TESTS_AND_TRAINING")
    n=COUNT[panel] if count_override is None else count_override
    if not isinstance(n,int) or not 1<=n<=COUNT[panel]:
        raise ValueError("R84_NONFROZEN_FIT_SAMPLE_COUNT")
    x=np.zeros((n,MAX_PIVOT+1,13),np.float32)
    value=np.zeros(n,np.float32)
    pivot=np.zeros(n,np.int64)
    chosen=np.zeros(n,np.int64)
    for i in range(n):
        w,action=_world(panel,seed,i)
        legal,ret,_=r83._public_rollout(w,action)
        p=w["pivot"]
        if not np.array_equal(legal[p,:4],np.eye(4,dtype=np.float32)[action]):
            raise RuntimeError("R84_CURRENT_ACTUAL_ACTION_INPUT_INVALID")
        x[i,:p+1]=legal
        value[i]=ret
        pivot[i]=p
        chosen[i]=action
    return {"input":x,"target":value,"pivot":pivot,"logged_action":chosen,
            "count":n,"panel":panel,"only_one_factual_return":True}


def _evaluator_data(panel,seed):
    """Called ONLY after all training states frozen, never in any unit test."""
    if panel not in ("DEV","OOD_SHIFT"):
        raise PermissionError("R84_SCORING_PANEL_NOT_AUTHORIZED")
    frozen()
    worlds=[]
    for i in range(COUNT[panel]):
        w,_=_world(panel,seed,i)
        x0,r0,_=r83._public_rollout(w,r83.CALIBRATE)
        x2,r2,_=r83._public_rollout(w,r83.EXPLOIT)
        p=w["pivot"]
        if not np.array_equal(x0[:p],x2[:p]):
            raise RuntimeError("R84_PUBLIC_PAST_DIFFERED_BETWEEN_REALIZED_BRANCHES")
        if not np.array_equal(x0[p,4:],x2[p,4:]):
            raise RuntimeError("R84_CHOICE_COMPARED_WITH_DIFFERENT_INFORMATION")
        worlds.append({"x0":x0,"x2":x2,"r0":float(r0),"r2":float(r2),
                       "pivot":p,"cold":bool(i%2==1),
                       "public_probe":float(x0[p,10])})
    return {"panel":panel,"seed":seed,"worlds":worlds,
            "count":len(worlds),"paired_evaluator_only":True}


def _blind(inp):
    result=inp.copy()
    result[...,4:12]=0
    return result


def models(seed):
    frozen()
    if seed not in SEEDS:
        raise ValueError("R84_UNFROZEN_MODEL_SEED")
    torch.manual_seed(seed)
    native=r74.make_model("HybridPilot_FULL",WIDTH)
    old=native.ttt
    optimized=r75.TriangularFastWeightMemory(WIDTH,learning_rate=old.eta)
    optimized.load_state_dict(old.state_dict(),strict=True)
    native.ttt=optimized
    blind=copy.deepcopy(native)
    before=dict(native.named_parameters())
    after=dict(blind.named_parameters())
    if before.keys()!=after.keys() or any(
        not torch.equal(before[k],after[k]) for k in before):
        raise RuntimeError("R84_IDENTICAL_INITIAL_NATIVE_PARAMETER_FENCE_FAIL")
    torch.manual_seed(seed+64)
    gru64=r81.GRU(64)
    torch.manual_seed(seed+256)
    gru256=r81.GRU(256)
    torch.manual_seed(seed+1024)
    mlp=r81.MLP256()
    return dict(zip(ARMS,(native,blind,gru64,gru256,mlp)))


def _score_fit(model,data,blind):
    inp=_blind(data["input"]) if blind else data["input"]
    yy=data["target"]
    p=data["pivot"]
    model.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(inp),BATCH):
            n=min(BATCH,len(inp)-i)
            y=model(torch.from_numpy(inp[i:i+n]))
            prediction=y[torch.arange(n),torch.from_numpy(p[i:i+n]),0]
            out.extend((prediction-torch.from_numpy(yy[i:i+n])).square().tolist())
    model.train()
    return float(np.mean(out))


def _train_one(arm,model,fit,valid,selected):
    blind=(arm==BLIND)
    x=torch.from_numpy(_blind(fit["input"]) if blind else fit["input"])
    y=torch.from_numpy(fit["target"])
    p=torch.from_numpy(fit["pivot"])
    steps=NATIVE_UPDATES if arm.startswith("NATIVE") else CONTROL_UPDATES
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    checkpoints={0:_score_fit(model,valid,blind)}
    began=time.perf_counter()
    for update in range(1,steps+1):
        ix=selected[update-1]
        estimate=model(x[ix])[torch.arange(BATCH),p[ix],0]
        loss=F.mse_loss(estimate,y[ix])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if update in (512,1024,2048,3072,4096):
            checkpoints[update]=_score_fit(model,valid,blind)
    elapsed=time.perf_counter()-began
    if (any(not math.isfinite(v) for v in checkpoints.values())
        or not all(bool(torch.isfinite(v).all()) for v in model.parameters())):
        raise RuntimeError("R84_NONFINITE_OR_INCOMPLETE_ORIGINAL_TRAINING")
    return {"updates_completed":steps,"parameter_count":sum(p.numel() for p in model.parameters()),
            "real_selected_FIT_event_exposures":steps*BATCH,
            "same_logged_FIT_worlds":fit["count"],
            "FIT_validation_MSE_at_updates":{str(k):v for k,v in checkpoints.items()},
            "training_wall_seconds":elapsed,
            "original_hard_router_only":arm.startswith("NATIVE"),
            "previous_observation_action_erased":blind}


def _choose(model,world,blind=False):
    x=np.stack([world["x0"],world["x2"]]).astype(np.float32)
    if blind:x=_blind(x)
    model.eval()
    with torch.no_grad():
        values=model(torch.from_numpy(x))[:,-1,0].cpu().numpy()
    model.train()
    if not np.isfinite(values).all():
        raise RuntimeError("R84_NONFINITE_ACTUAL_ACTION_VALUES")
    return (r83.CALIBRATE if values[0]>values[1] else r83.EXPLOIT)


def _realized(choices,worlds):
    r0=np.array([z["r0"] for z in worlds],np.float64)
    r2=np.array([z["r2"] for z in worlds],np.float64)
    choice=np.asarray(choices,np.int64)
    if choice.shape!=r0.shape:
        raise RuntimeError("R84_ILLEGAL_POLICY_ACTION_COUNT")
    ret=np.where(choice==r83.CALIBRATE,r0,r2)
    regret=np.maximum(r0,r2)-ret
    cold=np.array([z["cold"] for z in worlds],bool)
    if cold.sum()!=len(worlds)//2:
        raise RuntimeError("R84_EARLY_COLD_WORLD_COUNT_DRIFT")
    return {"mean_reward":float(ret.mean()),
            "cold_reward":float(ret[cold].mean()),
            "early_reward":float(ret[~cold].mean()),
            "oracle_mean_regret":float(regret.mean()),
            "chosen_calibrate_fraction":float((choice==r83.CALIBRATE).mean()),
            "per_world_reward":ret.tolist()}


def _shuffled_rule(worlds,seed,panel):
    index=np.arange(len(worlds))
    rng=np.random.default_rng(np.random.SeedSequence(
        [25260555,seed,1 if panel=="DEV" else 2]))
    for pivot in set(z["pivot"] for z in worlds):
        where=np.array([i for i,z in enumerate(worlds) if z["pivot"]==pivot])
        index[where]=rng.permutation(where)
    cues=np.array([z["public_probe"] for z in worlds])
    return [r83.CALIBRATE if cues[index[i]]>0 else r83.EXPLOIT
            for i in range(len(worlds))]


def _evaluate(arm_models,source,seed):
    worlds=source["worlds"]
    raw={}
    choices={}
    for arm,model in arm_models.items():
        choose=[_choose(model,w,arm==BLIND) for w in worlds]
        choices[arm]=choose
        raw[arm]=_realized(choose,worlds)
    cue=[r83.CALIBRATE if w["public_probe"]>0 else r83.EXPLOIT for w in worlds]
    rules={
        "PUBLIC_PROBE_SIGN_RULE":cue,
        "SHUFFLED_PROBE_RULE":_shuffled_rule(worlds,seed,source["panel"]),
        "ALWAYS_CALIBRATE":[r83.CALIBRATE]*len(worlds),
        "ALWAYS_EXPLOIT_POS":[r83.EXPLOIT]*len(worlds)}
    for name,actions in rules.items():
        raw[name]=_realized(actions,worlds)
    oracle=[r83.CALIBRATE if w["r0"]>=w["r2"] else r83.EXPLOIT for w in worlds]
    raw["EVALUATOR_ONLY_ORACLE"]=_realized(oracle,worlds)
    return raw,choices,cue


def _bootstrap_lower(best,treated,seed):
    a=np.asarray(best,dtype=np.float64)
    b=np.asarray(treated,dtype=np.float64)
    if a.shape!=(COUNT["DEV"],) or b.shape!=a.shape:
        raise ValueError("R84_BOOTSTRAP_NOT_PAIRED_ON_FROZEN_DEV")
    rng=np.random.default_rng(np.random.SeedSequence([25260555,seed]))
    ids=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((b-a)[ids].mean(axis=1),.025))


def admit(seed):
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")
        or os.getenv("R84_SEED")!=str(seed)):
        raise PermissionError("R84_FIRST_HOSTED_PR_CPU_ONLY_ADMISSION_DENIED")
    candidate=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,text=True,
        capture_output=True,timeout=10,check=False)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
        capture_output=True,timeout=10,check=False)
    if (len(candidate)!=40 or head.returncode!=0 or head.stdout.strip()!=candidate
        or dirty.returncode!=0):
        raise PermissionError("R84_EXACT_SOURCE_CUSTODY_OR_DIRTY_CHECKOUT")
    return candidate


def original_ci(seed):
    candidate=admit(seed)
    torch.set_num_threads(1)
    path=ROOT/f"local_runs/r84_learned_value_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R84_SCORING_FIRST_USE_REPLAY_REFUSED")
    started=time.perf_counter()
    result={
        "schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
        "candidate_SHA":candidate,"seed":seed,
        "original_parent_SHA":PARENT,"device":"CPU",
        "status":"INCOMPLETE_ENGINEERING_OR_RESOURCE",
        "completed_arms":{},"DEV_OOD_FIRST_USE":False,
        "all_weights_frozen_before_scoring":False,
        "R83_reserved_science_panels_accessed":False,
        "scientific_pass":False,"torch":str(torch.__version__),
        "platform":platform.platform()}
    try:
        fit=_factual_data("FIT",seed)
        val=_factual_data("FIT_VALIDATION",seed)
        if not fit["only_one_factual_return"] or not val["only_one_factual_return"]:
            raise RuntimeError("R84_ILLEGAL_COUNTERFACTUAL_FIT_LABEL")
        ms=models(seed)
        original=dict(ms["NATIVE_FULL_VALUE"].named_parameters())
        blinded=dict(ms[BLIND].named_parameters())
        if original.keys()!=blinded.keys() or any(
            not torch.equal(original[k],blinded[k]) for k in original):
            raise RuntimeError("R84_NATIVE_SAME_INITIAL_STATE_MISMATCH")
        rnd=np.random.default_rng(np.random.SeedSequence([25260611,seed]))
        selected=rnd.integers(0,len(fit["input"]),size=(CONTROL_UPDATES,BATCH))
        for arm in ARMS:
            result["completed_arms"][arm]=_train_one(arm,ms[arm],fit,val,selected)
        if set(result["completed_arms"])!=set(ARMS):
            raise RuntimeError("R84_TRAINED_CONTROLS_MISSING")
        # The point of no return: no more parameter updates or hyperparameter
        # selection after first independent DEV/OOD evaluation generation.
        result["all_weights_frozen_before_scoring"]=True
        result["DEV_OOD_FIRST_USE"]=True
        dev=_evaluator_data("DEV",seed)
        ood=_evaluator_data("OOD_SHIFT",seed)
        dev_raw,dev_choices,rule_choice=_evaluate(ms,dev,seed)
        ood_raw,_,_=_evaluate(ms,ood,seed)
        best_conv=max(
            ("GRU64_FULL_VALUE","GRU256_FULL_VALUE","MLP256_FULL_VALUE"),
            key=lambda arm:dev_raw[arm]["mean_reward"])
        strong_const=max(
            dev_raw["ALWAYS_CALIBRATE"]["mean_reward"],
            dev_raw["ALWAYS_EXPLOIT_POS"]["mean_reward"])
        native=dev_raw["NATIVE_FULL_VALUE"]
        rule=dev_raw["PUBLIC_PROBE_SIGN_RULE"]
        blind=dev_raw[BLIND]
        conv=dev_raw[best_conv]
        control_ood=ood_raw[best_conv]
        native_ood=ood_raw["NATIVE_FULL_VALUE"]
        gain_over_fixed=rule["mean_reward"]-strong_const
        if gain_over_fixed<=0:
            raise RuntimeError("R84_BENCHMARK_RULE_IS_NOT_INFORMATIVE_ON_FRESH_DEV")
        fraction=(native["mean_reward"]-strong_const)/gain_over_fixed
        agree=float(np.mean(
            np.asarray(dev_choices["NATIVE_FULL_VALUE"])==np.asarray(rule_choice)))
        lower=_bootstrap_lower(conv["per_world_reward"],native["per_world_reward"],seed)
        comparison={
            "best_trained_conventional_DEV":best_conv,
            "native_minus_blinded_DEV":native["mean_reward"]-blind["mean_reward"],
            "native_minus_strongest_conventional_DEV":native["mean_reward"]-conv["mean_reward"],
            "native_minus_best_fixed_DEV":native["mean_reward"]-strong_const,
            "native_minus_shuffled_rule_DEV":native["mean_reward"]-dev_raw["SHUFFLED_PROBE_RULE"]["mean_reward"],
            "native_action_agreement_with_public_rule_DEV":agree,
            "native_fraction_of_public_rule_improvement":fraction,
            "paired_bootstrap_lower95_native_minus_best_conventional_DEV":lower,
            "native_minus_best_conventional_shift_OOD":native_ood["mean_reward"]-control_ood["mean_reward"],
            "native_minus_best_conventional_shift_OOD_cold":native_ood["cold_reward"]-control_ood["cold_reward"],
            "native_DEV_regret_to_evaluator_oracle":native["oracle_mean_regret"]}
        t=frozen()["evaluation"]["thresholds"]
        checks={
            "native_outperforms_blind_DEV":comparison["native_minus_blinded_DEV"]>=t["DEV_reward_native_minus_blind_min"],
            "native_outperforms_strongest_conventional_DEV":comparison["native_minus_strongest_conventional_DEV"]>=t["DEV_reward_native_minus_strongest_trained_conventional_min"],
            "native_outperforms_strongest_fixed_DEV":comparison["native_minus_best_fixed_DEV"]>=t["DEV_reward_native_minus_best_fixed_min"],
            "native_outperforms_shuffled_probe_DEV":comparison["native_minus_shuffled_rule_DEV"]>=t["DEV_reward_native_minus_shuffled_rule_min"],
            "native_matches_rule_actions":agree>=t["DEV_public_rule_agreement_min"],
            "native_recovers_rule_benefit":fraction>=t["DEV_native_reward_as_fraction_of_rule_gain_over_best_fixed_min"],
            "paired_bootstrap_lower95_positive":lower>0,
            "native_outperforms_conventional_OOD":comparison["native_minus_best_conventional_shift_OOD"]>=t["OOD_reward_native_minus_best_conventional_min"],
            "native_outperforms_conventional_cold_OOD":comparison["native_minus_best_conventional_shift_OOD_cold"]>=t["OOD_cold_reward_native_minus_best_conventional_min"],
            "native_regret_not_too_high":comparison["native_DEV_regret_to_evaluator_oracle"]<=t["DEV_oracle_regret_native_max"]}
        result["comparisons"]=comparison
        result["frozen_checks"]={key:bool(value) for key,value in checks.items()}
        result["DEV"]={k:{n:v for n,v in metric.items() if n!="per_world_reward"}
                       for k,metric in dev_raw.items()}
        result["OOD_SHIFT"]={k:{n:v for n,v in metric.items() if n!="per_world_reward"}
                              for k,metric in ood_raw.items()}
        result["status"]=(t["on_positive"] if all(checks.values()) else t["on_negative"])
    except BaseException as exc:
        result["status"]="INCOMPLETE_ENGINEERING_OR_RESOURCE"
        result["error_class"]=type(exc).__name__
        raise
    finally:
        result["total_wall_seconds"]=time.perf_counter()-started
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as h:
            json.dump(result,h,sort_keys=True,indent=2,allow_nan=False)
            h.write("\n")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seed",required=True,type=int,choices=SEEDS)
    args=p.parse_args()
    result=original_ci(args.seed)
    print(json.dumps({"seed":args.seed,"status":result["status"],
        "completed_arms":len(result["completed_arms"]),
        "scored_new_DEV_and_OOD":result["DEV_OOD_FIRST_USE"],
        "science_qualification":False},sort_keys=True))
