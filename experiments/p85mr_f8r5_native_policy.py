"""F8-R5: actually train original native Hybrid, depth/bypass and strong controls.

Factual actions/effects are generated solely by the fixed R3 mathematical
two-world recurrence. Models select later toy actions after a selected probe.
No original simulator, F7 source, CAL, OOD, merge or external first-use.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch

from experiments import p85mr_f8_depth_mechanism as f8
from experiments import p85mr_f8r3_counterpolicy_identifiability as r3
from experiments import p85mr_f5_native_vs_controls as f5
from experiments import p85mr_f7_native_candidate as f7
from experiments import p85mr_f4_train_controls as f4
from p85mr_dynamics_baselines import new_baseline,prequential as baseline_predict
from p80mr_causalgym_trace_adapter import _action_key,_observed_effect

ROOT=Path(__file__).resolve().parents[1]
FREEZE_BLOB="1514b278aa28d60f62e8e359c832c591a926b763"
PARENT_SHA="d1b24351ed8f3c9c8fd233d75dc0ad120a3c91cb"
ARMS=("native_hard","native_surrogate","bypass_hard","bypass_surrogate","linear7","ssm256")
SEEDS=(985201,985203,985207)
SPLITS={"fit":(19140000,24),"diagnostic":(19160000,12),"score":(19180000,12)}
STATES=(-.8,-.6,-.4,-.2,0.,.2,.4,.6,.8)
GOALS=(-.2,0.,.2)
UPDATES=96


def locked():
    raw=(ROOT/"experiments/p85mr_f8r5_native_policy_freeze.json").read_bytes()
    if f8.p7.git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R5_FROZEN_SOURCE_DRIFT")
    f=json.loads(raw);src=f["source"];spec=f["models"];auth=f["authority"]
    if (f["schema"]!="ttt-hybrid.f8r5-native-trained-active-toy-policy.v1"
        or f["parent"]["source_sha"]!=PARENT_SHA
        or f["parent"]["original_model_blob"]!=f8.MODEL_BLOB
        or spec["arms"]!=list(ARMS) or spec["seeds"]!=list(SEEDS)
        or spec["updates_each"]!=UPDATES or spec["batch_size"]!=8
        or spec["sampler_seed"]!=985299 or spec["weight_decay"]!=.01
        or spec["lr_native"]!=.002 or spec["lr_linear"]!=.001
        or spec["lr_ssm"]!=.001
        or src["train_worlds"]!=24 or src["fit_dev_worlds"]!=12
        or src["score_worlds"]!=12 or src["world_episode_actions"]!=12
        or [src[k] for k in ("train_seed_base","fit_dev_seed_base","score_seed_base")]!=
           [19140000,19160000,19180000]
        or f["evaluation"]["score_states"]!=list(STATES)
        or f["evaluation"]["score_goals"]!=list(GOALS)
        or f["evaluation"]["probe_state"]!=.2 or f["evaluation"]["probe_action"]!=0
        or auth["no_original_external_firstuse"] is not True
        or auth["no_spent_F7_source_replay"] is not True
        or auth["no_new_workflow"] is not True
        or auth["independent_scientific_PASS"] is not False):
        raise ValueError("F8R5_FROZEN_SCOPE_DRIFT")
    f8.locked();r3.locked()
    return f


def toy_episode(split,index):
    if split not in SPLITS or type(index) is not int or not 0<=index<SPLITS[split][1]:
        raise ValueError("F8R5_UNKNOWN_WORLD")
    seed=SPLITS[split][0]+index
    hidden=1 if index%2==0 else -1
    state=.42*math.sin(seed*.013)+.19*math.cos(seed*.021)
    actions=[];effects=[]
    for step in range(12):
        action=1-r3.behavior(state) if step==0 else (
            r3.behavior(state)^int(((seed+step*7)%5)<2))
        observed_next=r3.synthetic_effect(state,action,hidden)
        before=(state,0.,0.,0.);after=(observed_next,0.,0.,0.)
        actions.append(_action_key(before,action))
        effects.append(_observed_effect(before,after,
          1.-min(1.,(observed_next-.2)**2),False,step==11))
        state=observed_next
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32)}


def cohort(split):
    locked()
    return [toy_episode(split,i) for i in range(SPLITS[split][1])]


def model_for(arm,seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("F8R5_UNKNOWN_ARM_SEED")
    if arm in ("linear7","ssm256"):
        return new_baseline(arm,seed)
    return f8.model_for(
        "surrogate_hard_forward" if arm.endswith("surrogate") else "hard_original",
        seed,bypass=arm.startswith("bypass"))


def native_kind(arm):
    return f7.K_BYPASS if arm.startswith("bypass") else f7.K_NATIVE


def objective(arm,model,data,valid):
    if arm in ("linear7","ssm256"):
        return f4._physical_loss(model,data,valid)[0]
    return f7.loss_for_arm(native_kind(arm),model,data,valid)


def predict(arm,model,data):
    return (baseline_predict(model,data) if arm in ("linear7","ssm256")
            else f7.prequential(native_kind(arm),model,data))


def fit_score(arm,model,episodes):
    return (float(f4.calibration_mse(model,episodes)["cal_delta_mse"])
            if arm in ("linear7","ssm256")
            else f7.score_worlds(native_kind(arm),model,episodes))


def choose(arm,model,hidden,state,goal,leak_probe=False):
    # Only one selected real toy action receives a factual effect. No sign is
    # passed to the model; hidden is evaluator-owned after selected actions.
    outcome=r3.synthetic_effect(.2,0,hidden)
    key=_action_key((.2,0.,0.,0.),0)
    effect=_observed_effect((.2,0.,0.,0.),(outcome,0.,0.,0.),
        1.-min(1.,(outcome-.2)**2),False,False)
    estimates=[]
    for action in (0,1):
        current=_action_key((state,0.,0.,0.),action)
        fact={"action":torch.tensor([[key,current]],dtype=torch.float32),
              "effect":torch.tensor([[effect,[0.]*16]],dtype=torch.float32)}
        with torch.no_grad():
            value=float(predict(arm,model,fact)[0,1,0])
        if leak_probe:
            contaminated={k:v.clone() for k,v in fact.items()}
            contaminated["effect"][0,1,0]=.47
            with torch.no_grad():
                no_leak=float(predict(arm,model,contaminated)[0,1,0])
            if no_leak!=value:
                raise ValueError("F8R5_FUTURE_LABEL_LEAK")
        if not math.isfinite(value):
            raise ValueError("F8R5_NONFINITE_PREDICTION")
        estimates.append(state+2.4*math.atanh(max(-.98,min(.98,value))))
    selected=min((0,1),key=lambda a:((estimates[a]-goal)**2,a))
    # External to the policy, calculate the selected effect and an oracle
    # bound AFTER choosing; oracle never trains or guides the model.
    actual=r3.synthetic_effect(state,selected,hidden)
    oracle=min((r3.synthetic_effect(state,a,hidden)-goal)**2 for a in (0,1))
    return selected,max(0.,(actual-goal)**2-oracle)


def policy_score(arm,model):
    model.eval();started=time.perf_counter();regrets=[];correct=0
    for hidden in (-1,1):
        for state in STATES:
            for goal in GOALS:
                _,regret=choose(arm,model,hidden,state,goal)
                regrets.append(regret);correct+=int(regret<=1e-12)
    return {"mean_action_regret":math.fsum(regrets)/len(regrets),
            "optimal_actions":correct,"selected_toy_effects":len(regrets),
            "inference_wall_s":time.perf_counter()-started}


def train_one(arm,seed,fit,dev,score,sampler):
    model=model_for(arm,seed).cpu()
    original=f5.hash_weights(model)
    rate=.002 if arm not in ("linear7","ssm256") else .001
    opt=torch.optim.AdamW(model.parameters(),lr=rate,weight_decay=.01)
    start=time.perf_counter()
    off=0;positive_router=0
    for indices in sampler:
        data,valid=f4._padded([fit[i] for i in indices])
        model.train();opt.zero_grad(set_to_none=True)
        loss=objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R5_NONFINITE_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters()]
        if any(v is not None and not bool(torch.isfinite(v).all())
               for v in gradients) or not any(
                   v is not None and bool((v!=0).any()) for v in gradients):
            raise ValueError("F8R5_NONFINITE_OR_MISSING_GRADIENT")
        if arm not in ("linear7","ssm256"):
            depth=model.original.depth if arm.startswith("bypass") else model.depth
            off+=int(depth.last_utilization==0.)
            positive_router+=int(
                depth.router.weight.grad is not None
                and bool((depth.router.weight.grad!=0).any()))
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    training_wall=time.perf_counter()-start
    if f5.hash_weights(model)==original:
        raise ValueError("F8R5_WEIGHTS_NOT_TRAINED")
    model.eval()
    dev_mse=fit_score(arm,model,dev)
    scored_mse=fit_score(arm,model,score)
    if not math.isfinite(dev_mse) or not math.isfinite(scored_mse):
        raise ValueError("F8R5_NONFINITE_HELDOUT")
    cold=model_for(arm,seed).cpu()
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    cold.eval()
    if (f5.hash_weights(cold)!=f5.hash_weights(model)
        or abs(fit_score(arm,cold,score)-scored_mse)>1e-10
        or choose(arm,model,-1,.2,-.2,True)!=choose(arm,cold,-1,.2,-.2,True)):
        raise ValueError("F8R5_COLD_RESTORE_OR_CAUSAL_PARITY_FAIL")
    performance=policy_score(arm,cold)
    return {**performance,"arm":arm,"seed":seed,
       "parameters":sum(p.numel() for p in model.parameters()),
       "updates":UPDATES,"train_wall_s":training_wall,
       "dev_mse":dev_mse,"heldout_toy_mse":scored_mse,
       "initial_hash":original,"trained_hash":f5.hash_weights(model),
       "inactive_depth_updates":off if arm not in ("linear7","ssm256") else None,
       "positive_router_gradient_updates":positive_router if arm not in ("linear7","ssm256") else None,
       "cold_restore":True}


def run():
    cfg=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,dev,score=cohort("fit"),cohort("diagnostic"),cohort("score")
    g=torch.Generator(device="cpu");g.manual_seed(985299)
    indices=torch.randint(0,len(fit),(UPDATES,8),generator=g).tolist()
    results=[train_one(arm,seed,fit,dev,score,indices)
             for arm in ARMS for seed in SEEDS]
    by={(x["arm"],x["seed"]):x for x in results}
    blind=math.fsum(
        (r3.synthetic_effect(s,0,w)-goal)**2-min(
           (r3.synthetic_effect(s,a,w)-goal)**2 for a in (0,1))
        for w in (-1,1) for s in STATES for goal in GOALS)/54
    comparisons=[]
    for seed in SEEDS:
        vals={arm:by[(arm,seed)]["mean_action_regret"] for arm in ARMS}
        candidate=vals["bypass_surrogate"]
        comparisons.append({"seed":seed,"regret":vals,
          "candidate_ratio_to_best_classical":candidate/max(
             min(vals["linear7"],vals["ssm256"]),1e-20),
          "candidate_win":bool(candidate<=.9*min(vals["linear7"],vals["ssm256"])
                               and candidate<vals["bypass_hard"])})
    wins=sum(x["candidate_win"] for x in comparisons)
    control_competent=any(
        min(by[("linear7",seed)]["mean_action_regret"],
            by[("ssm256",seed)]["mean_action_regret"])<=.9*blind
        for seed in SEEDS)
    result={"schema":"ttt-hybrid.f8r5-native-vs-conventional-policy-toy-dev.v1",
        "status":"F8R5_TRAINED_TOY_DEV_COMPLETED","models":results,
        "comparison":comparisons,"candidate_wins":wins,
        "classical_competent":control_competent,
        "dev_pass":bool(wins>=2 and control_competent),
        "blind_action0_regret":blind,
        "oracle_mechanism_regret":0.,
        "sampler_sha256":hashlib.sha256(json.dumps(indices,separators=(",",":")).encode()).hexdigest(),
        "model_count":len(results),"total_optimizer_updates":len(results)*UPDATES,
        "total_training_s":sum(x["train_wall_s"] for x in results),
        "total_inference_s":sum(x["inference_wall_s"] for x in results),
        "real_external_source_actions":0,"F7_replayed":False,
        "independent_scientific_pass":False,"CAL_OOD":False,"deployment":False}
    if len(results)!=18 or result["total_optimizer_updates"]!=1728:
        raise ValueError("F8R5_MODEL_MATRIX_INCOMPLETE")
    return result


def save_github_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":return
    output=ROOT/"local_runs/p85mr_f8r5_native_policy_dev.json"
    output.parent.mkdir(exist_ok=True)
    if output.exists():raise FileExistsError("F8R5_REFUSE_REPLAY_OVERWRITE")
    output.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external(*args,**kwargs):
    raise PermissionError("F8R5_ORIGINAL_FIRSTUSE_NOT_AUTHORIZED")
