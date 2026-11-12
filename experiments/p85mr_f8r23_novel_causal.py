"""F8-R23 prospectively frozen cross-law and intervening-B memory falsifier.

Source-neutral tests ONLY. Retain original R22 training and frozen native/
GRU/SSM learning objectives. Both new-law panels are scored without fitting
on them or making a protected third-party simulator call. Not SCIENTIFIC_PASS.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from experiments import p85mr_f8r22_objective_matched as r22

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r23_novel_causal_freeze.json"
FREEZE_BLOB="b47b3a143f123b2337f97fdb79ed5dfb66a2f836"
SCHEMA="ttt-hybrid.f8r23-new-physics-interference-cold-memory-dev.v1"
ARMS=("native_focus20","gru_focus20","ssm_focus20")
SEEDS=(985201,985203,985207)
PANELS={"nonlinear_opposing_actions":19810000,
        "two_B_regime_interference":19820000}
GOALS=(-.2,.2)
N_GROUPS=12
B_STEPS=16
BATCH=8
UPDATES=384
BOOT=10000
BOOT_SEED=19829999


def blob_sha(data:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked()->dict:
    raw=FREEZE.read_bytes()
    if blob_sha(raw)!=FREEZE_BLOB:
        raise ValueError("F8R23_PRECOMMITTED_PROTOCOL_DRIFT")
    p=json.loads(raw)
    if (p["schema"]!=SCHEMA
        or p["status"]!="FROZEN_BEFORE_NEW_IMPLEMENTATION_AND_NUMERIC_RESULTS"
        or p["classification"]!="SOURCE_NEUTRAL_PROSPECTIVE_DEVELOPMENT_NOT_SCIENTIFIC_ACCEPTANCE"
        or p["parent"]["head"]!="3172595b954f0770176e0848ab537460df79bd91"
        or p["parent"]["r22_source_blob"]!="57e5977555f3c0afee7a987a4aee80be994cc082"
        or p["parent"]["r22_freeze_blob"]!="c7c5cd9bb3e3590a58ccff92439c763305659651"
        or p["world"]["panel_episodes_per_law"]!=48
        or p["world"]["selected_decisions_per_model_per_law"]!=96
        or [x["name"] for x in p["world"]["test_cases"]]!=list(PANELS)
        or [x["group_seed_base"] for x in p["world"]["test_cases"]]!=list(PANELS.values())
        or any(x["steps_B"]!=B_STEPS or x["B_groups"]!=N_GROUPS
               for x in p["world"]["test_cases"])
        or p["world"]["goals"]!=list(GOALS)
        or p["models"]["arms"]!=list(ARMS)
        or p["models"]["seeds"]!=list(SEEDS)
        or p["models"]["total_fits"]!=9
        or p["models"]["updates_each"]!=UPDATES
        or p["models"]["total_optimizer_updates"]!=3456
        or p["models"]["batch_worlds"]!=BATCH
        or p["models"]["sampler_seed"]!=985299
        or p["thresholds"]["bootstrap_draws"]!=BOOT
        or p["thresholds"]["bootstrap_seed"]!=BOOT_SEED
        or p["thresholds"]["scientific_pass"] is not False
        or any(x is not False for x in p["authority"].values())):
        raise ValueError("F8R23_PARENT_OR_OWNER_SCOPE_DRIFT")
    r22.locked()
    return p


def law(panel:str,state:float,action:int,sign:int)->float:
    if (panel not in PANELS or type(action) is not int or action not in (0,1)
        or type(sign) is not int or sign not in (-1,1)
        or not math.isfinite(state)):
        raise ValueError("F8R23_INVALID_LAW_INPUT")
    if panel=="nonlinear_opposing_actions":
        return .31*math.tanh(1.7*state)+.24*sign*(2*action-1)
    return r22.r20.r3.synthetic_effect(state,action,sign)


def episode(panel:str,group:int,z_b:int,z_a:int)->dict:
    if (panel not in PANELS or type(group) is not int
        or not 0<=group<N_GROUPS
        or type(z_b) is not int or z_b not in (-1,1)
        or type(z_a) is not int or z_a not in (-1,1)):
        raise ValueError("F8R23_UNREGISTERED_PANEL_GROUP_OR_SIGN")
    seed=PANELS[panel]+group
    actions,effects=[],[]
    def record(before,act,after):
        actions.append(r22.r18.r5._action_key((before,0.,0.,0.),act))
        effects.append(r22.r18._observed(before,after))
    record(.2,0,law(panel,.2,0,z_a))
    state=0.
    for i in range(B_STEPS):
        sign=z_b if i<8 else (-z_b if panel=="two_B_regime_interference" else z_b)
        if i==8 and panel=="two_B_regime_interference":
            state=0. # controlled new independent B phase, no A information
        action=r22.r20.r3.behavior(state)^int(
            (seed+5*i+int(sign==1))%3==0)
        after=law(panel,state,action,sign)
        record(state,action,after)
        state=after
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32),
            "panel":panel,"group":group,"z_a":z_a,"z_b":z_b,
            "return_state_evaluator_only":0.}


def cohort(panel:str)->list[dict]:
    locked()
    if panel not in PANELS:
        raise ValueError("F8R23_UNKNOWN_EVALUATION_PANEL")
    return [episode(panel,group,b,a) for group in range(N_GROUPS)
            for b in (-1,1) for a in (-1,1)]


def identity(eps:list[dict],panel:str)->dict:
    if len(eps)!=48 or panel not in PANELS:
        raise ValueError("F8R23_INCOMPLETE_PANEL")
    for i in range(0,len(eps),2):
        a,b=eps[i:i+2]
        if (a["panel"]!=panel or b["panel"]!=panel
           or a["group"]!=b["group"] or a["z_b"]!=b["z_b"]
           or a["z_a"]!=-1 or b["z_a"]!=1
           or not torch.equal(a["action"],b["action"])
           or not torch.equal(a["effect"][1:],b["effect"][1:])
           or torch.equal(a["effect"][0],b["effect"][0])
           or a["action"].shape!=(17,16)
           or a["effect"].shape!=(17,16)):
            raise ValueError("F8R23_OLD_A_LEAK_OR_UNMATCHED_B")
    return {"matched_A_pairs":24,"B_actions_effects_identical":True,
            "first_return_effect_observed":False,
            "no_original_source_effects":True}


def regret(panel:str,z_a:int,goal:float,act:int)->float:
    if goal not in GOALS:
        raise ValueError("F8R23_UNKNOWN_GOAL")
    actual=(law(panel,0.,act,z_a)-goal)**2
    oracle=min((law(panel,0.,a,z_a)-goal)**2 for a in (0,1))
    return max(0.,actual-oracle)


def analytic(eps:list[dict],panel:str)->dict:
    identity(eps,panel)
    pergroup={k:[] for k in ("fixed0","fixed1","A_identity_oracle")}
    for g in range(N_GROUPS):
        examples=eps[4*g:4*g+4]
        for control in pergroup:
            scores=[]
            for ep in examples:
                for goal in GOALS:
                    act=(0 if control=="fixed0" else 1) if control!="A_identity_oracle" else min(
                        (0,1),key=lambda a:(law(panel,0.,a,ep["z_a"])-goal)**2)
                    scores.append(regret(panel,ep["z_a"],goal,act))
            pergroup[control].append(float(np.mean(scores)))
    means={k:float(np.mean(v)) for k,v in pergroup.items()}
    expected=.096 if panel=="nonlinear_opposing_actions" else .015
    actual=min(means["fixed0"],means["fixed1"])
    if (not math.isclose(actual,expected,abs_tol=1e-12)
        or means["A_identity_oracle"]!=0):
        raise ValueError("F8R23_MISREGISTERED_INFORMATION_BOUND")
    fixed=min(("fixed0","fixed1"),key=lambda k:(means[k],k))
    return {"means":means,"pergroup":pergroup,
            "best_no_history":fixed,"no_history_mean_regret":actual,
            "oracle_regret":0.}


def hypothetical(arm:str,model,eps:list[dict],erase_early=False)->np.ndarray:
    inputs,observed=[],[]
    for ep in eps:
        for act in (0,1):
            key=torch.tensor(r22.r18.r5._action_key((0.,0.,0.,0.),act),
                             dtype=torch.float32).unsqueeze(0)
            action=torch.cat((ep["action"],key),dim=0)
            effects=torch.cat((ep["effect"],
                               torch.zeros((1,16),dtype=torch.float32)),dim=0)
            if erase_early:
                effects[0]=0.
            inputs.append(action)
            observed.append(effects)
    model.eval()
    with torch.no_grad():
        x=r22.predict(arm,model,
                      {"action":torch.stack(inputs),
                       "effect":torch.stack(observed)})
    if x.shape!=(2*len(eps),B_STEPS+2,16) or not bool(torch.isfinite(x).all()):
        raise ValueError("F8R23_INVALID_PREACTION_MODEL_PREDICTION")
    return np.asarray([2.4*math.atanh(max(-.98,min(.98,float(v))))
              for v in x[:,-1,0].detach().cpu().numpy()],
              dtype=np.float64).reshape(len(eps),2)


def score(arm:str,model,eps:list[dict],panel:str)->dict:
    identity(eps,panel)
    t0=time.perf_counter()
    original=hypothetical(arm,model,eps)
    erased=hypothetical(arm,model,eps,True)
    actions,losses,erased_losses=[],[],[]
    for i,ep in enumerate(eps):
        for goal in GOALS:
            selected=min((0,1),key=lambda q:((original[i,q]-goal)**2,q))
            no_early=min((0,1),key=lambda q:((erased[i,q]-goal)**2,q))
            actions.append(selected)
            losses.append(regret(panel,ep["z_a"],goal,selected))
            erased_losses.append(regret(panel,ep["z_a"],goal,no_early))
    contrasts=sum(actions[2*(4*g+2*b)+j]!=actions[2*(4*g+2*b)+j+2]
                  for g in range(N_GROUPS) for b in range(2) for j in range(2))
    return {"mean_regret":float(np.mean(losses)),
            "probe_erased_mean_regret":float(np.mean(erased_losses)),
            "paired_group_regret":[float(np.mean(losses[8*g:8*g+8]))
                                   for g in range(N_GROUPS)],
            "A_sign_action_contrasts":int(contrasts),
            "contrast_opportunities":48,
            "selected_actions":actions,
            "selected_decisions":len(actions),
            "two_action_predictions":2*len(eps),
            "evaluation_seconds":time.perf_counter()-t0}


def ridge_score(model:dict,eps:list[dict],panel:str)->dict:
    beta=np.asarray(model["coef"],dtype=np.float64)
    losses=[]
    for ep in eps:
        first=r22.r19._physical_delta(ep["effect"][0])
        delta=( [1.,first] if model["use_early"]
                 else [1.]+[r22.r19._physical_delta(x)
                            for x in ep["effect"][-3:]] )
        action0=float(beta@np.asarray(delta))
        action1=float(model["action1_mean"])
        for goal in GOALS:
            selected=min((0,1),key=lambda a:(
                ((action0 if a==0 else action1)-goal)**2,a))
            losses.append(regret(panel,ep["z_a"],goal,selected))
    return {"mean_regret":float(np.mean(losses)),
            "uses_early_factual_A_memory":bool(model["use_early"]),
            "uses_no_original_external_effects":True}


def train(arm:str,seed:int,fit:list[dict],dev:list[dict],
          panels:dict[str,list[dict]], batches:list[list[int]])->dict:
    model=r22.model_for(arm,seed).cpu()
    initial=r22.r18.r5.f5.hash_weights(model)
    opt=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm=="native_focus20" else .001,weight_decay=.01)
    start=time.perf_counter()
    for indices in batches:
        data,valid=r22.r18.r5.f4._padded([r22.factual(fit[i]) for i in indices])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=r22.objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R23_NONFINITE_SOURCE_NEUTRAL_LOSS")
        loss.backward()
        if (not any(t.grad is not None and bool((t.grad.abs()>0).any())
                    for t in model.parameters())
            or any(t.grad is not None and not bool(torch.isfinite(t.grad).all())
                   for t in model.parameters())):
            raise ValueError("F8R23_BAD_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    elapsed=time.perf_counter()-start
    digest=r22.r18.r5.f5.hash_weights(model)
    if digest==initial:
        raise ValueError("F8R23_NO_WEIGHT_CHANGE")
    model.eval()
    dev_original=r22.score(arm,model,dev,"dev")
    results={name:score(arm,model,eps,name) for name,eps in panels.items()}
    cold=r22.model_for(arm,seed).cpu()
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    cold.eval()
    if (r22.r18.r5.f5.hash_weights(cold)!=digest
        or any(results[name]["selected_actions"]!=
               score(arm,cold,eps,name)["selected_actions"]
               for name,eps in panels.items())):
        raise ValueError("F8R23_COLD_RESTORE_ACTION_MISMATCH")
    return {"arm":arm,"seed":seed,
        "parameter_count":sum(t.numel() for t in model.parameters()),
        "steps":UPDATES,"initial_sha":initial,"trained_sha":digest,
        "train_seconds":elapsed,
        "original_DEV_regret":dev_original["mean_regret"],
        "cold_restored_actions_equal":True,
        "panels":results}


def decisions(rows:list[dict],controls:dict)->dict:
    baseline=min((r for r in rows if r["arm"]!="native_focus20"),
                 key=lambda r:(r["original_DEV_regret"],r["arm"],r["seed"]))
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,N_GROUPS,size=(BOOT,N_GROUPS))
    results=[]
    for row in rows:
        if row["arm"]!="native_focus20":
            continue
        detail={}
        all_win=True
        for panel in PANELS:
            score_row=row["panels"][panel]
            fixed=controls[panel]["analytic"]
            reference=np.asarray(fixed["pergroup"][fixed["best_no_history"]])
            diffs=reference-np.asarray(score_row["paired_group_regret"])
            lower=float(np.percentile(diffs[draws].mean(axis=1),2.5))
            gain=score_row["probe_erased_mean_regret"]-score_row["mean_regret"]
            contrast=score_row["A_sign_action_contrasts"]
            cap=float(np.mean(reference))*.90
            win=bool(score_row["mean_regret"]<=cap
                 and gain>=.01 and contrast>=12 and lower>0)
            detail[panel]={
                "candidate_regret":score_row["mean_regret"],
                "reference_fixed_no_history":float(np.mean(reference)),
                "DEV_selected_GRU_SSM_regret":baseline["panels"][panel]["mean_regret"],
                "FIT_only_memory_ridge_regret":controls[panel]["early_A_ridge"]["mean_regret"],
                "FIT_only_recent_B_ridge_regret":controls[panel]["recent_B_ridge"]["mean_regret"],
                "probe_erasure_regret_increase":gain,
                "A_sign_action_contrasts":contrast,
                "bootstrap_2p5_lower_absolute_gain":lower,
                "developer_retention_win":win}
            all_win=all_win and win
        results.append({"seed":row["seed"],
                        "both_novel_worlds_dev_win":all_win,"panels":detail})
    return {"DEV_selected_conventional":{"arm":baseline["arm"],
               "seed":baseline["seed"],
               "original_DEV_regret":baseline["original_DEV_regret"]},
            "native_seed_decisions":results,
            "retention_dev_gate_pass":sum(int(x["both_novel_worlds_dev_win"])
                                       for x in results)>=2,
            "scientific_pass":False}


def run()->dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r22.cohort("fit")
    dev=r22.cohort("dev")
    if len(fit)!=256 or len(dev)!=32:
        raise ValueError("F8R23_R22_FIT_OR_DEV_DRIFT")
    panels={name:cohort(name) for name in PANELS}
    ctl={}
    early=r22.fit_ridge(fit,use_early=True)
    late=r22.fit_ridge(fit,use_early=False)
    for name,eps in panels.items():
        ctl[name]={"identity":identity(eps,name),
                   "analytic":analytic(eps,name),
                   "early_A_ridge":ridge_score(early,eps,name),
                   "recent_B_ridge":ridge_score(late,eps,name)}
    g=torch.Generator(device="cpu");g.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=g).tolist()
    start=time.perf_counter()
    rows=[train(arm,seed,fit,dev,panels,batches)
          for arm in ARMS for seed in SEEDS]
    result=decisions(rows,ctl)
    return {
        "schema":SCHEMA,"classification":
            "SOURCE_NEUTRAL_PROSPECTIVE_DEVELOPMENT_NOT_SCIENTIFIC_ACCEPTANCE",
        "frozen_protocol_blob":FREEZE_BLOB,
        "original_parent_sha":"3172595b954f0770176e0848ab537460df79bd91",
        "trained_original_R22_FIT_sessions":len(fit),
        "original_R22_FIT_synthetic_factual_effects":sum(len(x["effect"]) for x in fit),
        "new_panel_cases_each":48,
        "new_panel_selected_actions_each_per_model":96,
        "models":rows,"model_count":len(rows),
        "total_optimizer_updates":len(rows)*UPDATES,
        "controls":ctl,"decisions":result,
        "conventional_independently_convergence_qualified":False,
        "independent_scientific_pass":False,
        "original_CausalGym_external_effects":0,
        "F7_original_replay":False,
        "deployment":"NONE",
        "whole_fit_score_seconds":time.perf_counter()-start}


def retain_ci_receipt(r:dict)->None:
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r23_new_physics_interference_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R23_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(r,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_external_source(*_a,**_k):
    raise PermissionError("F8R23_EXTERNAL_SCIENTIFIC_FIRSTUSE_NOT_AUTHORIZED")
