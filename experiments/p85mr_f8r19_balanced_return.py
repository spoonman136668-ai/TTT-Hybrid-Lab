"""F8-R19: balanced factual return-action support and native causal-memory credit.

Source-neutral author-generated mathematical worlds only. Learners receive the
early A *factual* probe effect and actual selected FIT return consequences,
never hidden signs, current future labels, or unchosen action outcomes.
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

from experiments import p85mr_f8r18_matched_aba as r18
from experiments import p85mr_f8r3_counterpolicy_identifiability as r3
from experiments import p85mr_f8r5_native_policy as r5

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r19_balanced_return_freeze.json"
FREEZE_BLOB="3a6e96789b81d51ca41c429198d47229368271f5"
SCHEMA="ttt-hybrid.f8r19-balanced-factual-return-credit-dev.v1"
ARMS=("native_uniform","native_return_focus",
      "linear_return_focus","mlp_return_focus","ssm_return_focus")
UNDERLYING={"native_uniform":"bypass_surrogate",
            "native_return_focus":"bypass_surrogate",
            "linear_return_focus":"linear7",
            "mlp_return_focus":"mlp7",
            "ssm_return_focus":"ssm256"}
SEEDS=(985201,985203,985207)
COHORTS={"fit":(19550000,32),"dev":(19560000,12),
         "score":(19570000,16)}
GOALS=(-.2,.2)
UPDATES=160
BATCH=8
RETURN_WEIGHT=20.
BOOT=10000
BOOT_SEED=19579999
RIDGE=1e-9


def locked():
    raw=FREEZE.read_bytes()
    blob=hashlib.sha1(b"blob "+str(len(raw)).encode("ascii")+b"\0"+raw).hexdigest()
    if blob!=FREEZE_BLOB:
        raise ValueError("F8R19_FREEZE_BLOB_DRIFT")
    cfg=json.loads(raw)
    w,t,s,a=cfg["source"],cfg["training"],cfg["evaluation"],cfg["authority"]
    if (cfg["schema"]!=SCHEMA
        or cfg["stage"]!="PRECOMMITTED_BEFORE_CODE_TRAINING_AND_RESULTS"
        or cfg["classification"]!=
           "SOURCE_NEUTRAL_NATIVE_TRAINING_DEVELOPMENT_NOT_INDEPENDENT_SCIENCE"
        or cfg["parent"]["sha"]!="81fa6bf3c6d224e0c03d7b5b64dae91056a3ca7e"
        or cfg["parent"]["r18_implementation_blob"]!=
           "affdbf248d114cad5a012cfc07c60308a525665f"
        or cfg["parent"]["r18_freeze_blob"]!=
           "2b82e54e5be7f515fd75fc659b2074b4e42db270"
        or tuple(t["arms"])!=ARMS
        or tuple(t["underlying_models"])!=tuple(UNDERLYING.values())
        or tuple(t["seeds"])!=SEEDS
        or t["fixed_updates_per_model"]!=UPDATES
        or t["shared_batch_size_episodes"]!=BATCH
        or t["total_models"]!=15
        or t["total_optimizer_updates"]!=2400
        or t["return_FIT_weight_focused"]!=20
        or t["native_uniform_weight_focused"]!=0
        or tuple(t["update_checkpoints"])!=(80,160)
        or t["sampler_seed"]!=985299
        or t["no_score_training"] is not True
        or any(w["cohorts"][name]["group_seed_base"]!=start
               or w["cohorts"][name]["pair_groups"]!=n
               for name,(start,n) in COHORTS.items())
        or tuple(w["return_A"]["actions_FIT"])!=(0,1)
        or w["return_A"]["balanced_four_way_FIT"] is not True
        or w["return_A"]["no_dev_score_return_effect_before_decision"] is not True
        or tuple(w["return_A"]["cross_both_goals"])!=GOALS
        or s["score_pairs"]!=16
        or s["bootstrap_samples"]!=BOOT
        or s["bootstrap_seed"]!=BOOT_SEED
        or cfg["learned_memory_control"]["ridge_lambda"]!=RIDGE
        or s["scientific_pass"] is not False
        or any(value is not False for value in a.values())):
        raise ValueError("F8R19_FROZEN_SCOPE_OR_AUTHORITY_DRIFT")
    r18.locked()
    return cfg


def episode(split:str,group:int,sign_a:int,return_action:int|None=None):
    if (split not in COHORTS or type(group) is not int
        or not 0<=group<COHORTS[split][1]
        or type(sign_a) is not int or sign_a not in (-1,1)
        or (split=="fit" and
            (type(return_action) is not int or return_action not in (0,1)))
        or (split!="fit" and return_action is not None)):
        raise ValueError("F8R19_UNREGISTERED_WORLD_OR_RETURN_ACTION")
    seed=COHORTS[split][0]+group
    sign_b=1 if group%2==0 else -1
    s=.2
    first_action=0
    first_effect=r3.synthetic_effect(s,first_action,sign_a)
    actions=[r5._action_key((s,0.,0.,0.),first_action)]
    effects=[r18._observed(s,first_effect)]
    # B always starts at exactly zero, independent of which A was observed.
    s=0.
    for j in range(8):
        selected=r3.behavior(s)^int((seed+7*j)%3==0)
        observed=r3.synthetic_effect(s,selected,sign_b)
        actions.append(r5._action_key((s,0.,0.,0.),selected))
        effects.append(r18._observed(s,observed))
        s=observed
    if split=="fit":
        observed=r3.synthetic_effect(0.,return_action,sign_a)
        actions.append(r5._action_key((0.,0.,0.,0.),return_action))
        effects.append(r18._observed(0.,observed,final=True))
    return {"action":torch.tensor(actions,dtype=torch.float32),
            "effect":torch.tensor(effects,dtype=torch.float32),
            # Only evaluator may use signs; factual() excludes all metadata.
            "sign_a":sign_a,"sign_b":sign_b,"group":group,
            "return_action":return_action,"seed":seed,"return_state":0.}


def cohort(split:str):
    locked()
    if split=="fit":
        return [episode(split,g,a,act)
                for g in range(COHORTS[split][1])
                for a in (-1,1) for act in (0,1)]
    return [episode(split,g,a)
            for g in range(COHORTS[split][1]) for a in (-1,1)]


def factual(ep):
    return {"action":ep["action"],"effect":ep["effect"]}


def audit_fit_support(fit:list[dict]) -> dict:
    if len(fit)!=128:
        raise ValueError("F8R19_INCOMPLETE_FACTORIAL_FIT")
    cells={}
    for ep in fit:
        b,a,z=ep["sign_b"],ep["return_action"],ep["sign_a"]
        key=(b,a,z)
        cells[key]=cells.get(key,0)+1
    if (len(cells)!=8 or any(v!=16 for v in cells.values())):
        raise ValueError("F8R19_B_A_RETURN_FIT_SUPPORT_GAP")
    for k in range(0,len(fit),4):
        subset=fit[k:k+4]
        if ([(e["sign_a"],e["return_action"]) for e in subset]
            != [(-1,0),(-1,1),(1,0),(1,1)]
            or any(not torch.equal(subset[0]["action"][:9],
                                   e["action"][:9]) for e in subset[1:])
            or not torch.equal(subset[0]["effect"][1:9],
                               subset[2]["effect"][1:9])
            or not torch.equal(subset[0]["effect"][:9],
                               subset[1]["effect"][:9])):
            raise ValueError("F8R19_CORRUPT_MATCHED_FACTUAL_SUPPORT")
    return {"factorial_cells":len(cells),"count_per_B_A_return_cell":16,
            "count_per_B_return_cell":32,
            "FIT_source_neutral_selected_effects":1280}


def _model(arm,seed):
    if arm not in ARMS or type(seed) is not int or seed not in SEEDS:
        raise ValueError("F8R19_UNKNOWN_ARM_SEED")
    return r18._model(UNDERLYING[arm],seed).cpu()


def _pred(arm,model,data):
    return r18._predict(UNDERLYING[arm],model,data)


def _objective(arm,model,data,valid):
    base=r18._objective(UNDERLYING[arm],model,data,valid)
    if arm=="native_uniform":
        return base
    preds=_pred(arm,model,data)
    if (preds.shape!=data["effect"].shape
        or preds.shape[1]!=10
        or not bool(valid[:,-1].all())):
        raise ValueError("F8R19_MISSING_FACTUAL_RETURN_FIT_LABEL")
    # This label is the actually SELECTED action's observed effect at t=9,
    # with its own current effect shifted/masked from the model's input.
    return base+RETURN_WEIGHT*(preds[:,-1,:4]-data["effect"][:,-1,:4]).square().mean()


def _physical_delta(effect_row):
    norm=float(effect_row[0])
    if not math.isfinite(norm) or not -1.<norm<1.:
        raise ValueError("F8R19_INVALID_FACTUAL_PHYSICAL_DELTA")
    return 2.4*math.atanh(norm)


def learn_factual_memory(fit:list[dict]) -> dict:
    # Fully observed action0 FIT examples: 32 groups ×2 signed A outcomes.
    # Fit [intercept, early A factual delta] -> actual action0 return delta.
    xs,ys,action1=[],[],[]
    for ep in fit:
        y=_physical_delta(ep["effect"][-1])
        if ep["return_action"]==0:
            x=_physical_delta(ep["effect"][0])
            xs.append((1.,x));ys.append(y)
        elif ep["return_action"]==1:
            action1.append(y)
        else:
            raise ValueError("F8R19_BAD_LEARNED_FACTUAL_ACTION")
    if len(xs)!=64 or len(action1)!=64:
        raise ValueError("F8R19_INSUFFICIENT_FACTUAL_MEMORY_SUPPORT")
    X=np.asarray(xs,dtype=np.float64)
    Y=np.asarray(ys,dtype=np.float64)
    theta=np.linalg.solve(X.T@X+RIDGE*np.eye(2),X.T@Y)
    return {"coef":theta.tolist(),"action1_mean":float(np.mean(action1)),
            "action0_FIT_examples":len(xs),"action1_FIT_examples":len(action1),
            "uses_hidden_regime_labels":False,"uses_counterfactual_effects":False}


def score_factual_memory(fitted:dict,eps:list[dict]) -> dict:
    if len(eps)%2:
        raise ValueError("F8R19_INCOMPLETE_LEARNED_MEMORY_PAIR")
    weights=np.asarray(fitted["coef"],dtype=np.float64)
    sums=[];actions=[];contrasts=0
    for i in range(0,len(eps),2):
        minus,plus=eps[i:i+2]
        if (minus["sign_a"]!=-1 or plus["sign_a"]!=1
            or minus["sign_b"]!=plus["sign_b"]
            or not torch.equal(minus["action"],plus["action"])
            or not torch.equal(minus["effect"][1:],plus["effect"][1:])):
            raise ValueError("F8R19_LEARNED_MEMORY_PAIR_UNMATCHED")
        pairacts=[]
        regrets=[]
        for ep in (minus,plus):
            prior=_physical_delta(ep["effect"][0])
            proposed=(float(weights@np.asarray([1.,prior])),
                      fitted["action1_mean"])
            selections=[]
            for goal in GOALS:
                action=min((0,1),key=lambda a:((proposed[a]-goal)**2,a))
                regrets.append(r18._regret(ep["sign_a"],goal,action))
                selections.append(action)
                actions.append(action)
            pairacts.append(selections)
        contrasts+=sum(pairacts[0][k]!=pairacts[1][k] for k in (0,1))
        sums.append(float(math.fsum(regrets)/4))
    return {"mean_regret":float(np.mean(sums)),
            "per_group_regret":sums,
            "A_sign_action_contrast_count":contrasts,
            "A_sign_action_contrast_pairs":len(eps),
            "selected_actions":actions,
            "model_selected_synthetic_actions":len(actions)}


def train_one(arm,seed,fit,dev,score_eps,batches):
    model=_model(arm,seed)
    initial=r5.f5.hash_weights(model)
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("native_") else .001,weight_decay=.01)
    losses=[];dev_half=None
    t0=time.perf_counter()
    for update,indices in enumerate(batches,1):
        data,valid=r5.f4._padded([factual(fit[i]) for i in indices])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=_objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R19_NONFINITE_FIT_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(p is not None and bool((p!=0).any()) for p in grads)
            or any(p is not None and not bool(torch.isfinite(p).all()) for p in grads)):
            raise ValueError("F8R19_INVALID_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if update==80:
            model.eval()
            dev_half=r18.score(UNDERLYING[arm],model,dev)["mean_regret"]
    elapsed=time.perf_counter()-t0
    trained=r5.f5.hash_weights(model)
    if trained==initial:
        raise ValueError("F8R19_NO_MODEL_WEIGHT_CHANGE")
    model.eval()
    dev_score=r18.score(UNDERLYING[arm],model,dev)
    test_score=r18.score(UNDERLYING[arm],model,score_eps)
    cold=_model(arm,seed)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    cold.eval()
    repeated=r18.score(UNDERLYING[arm],cold,score_eps)
    if (r5.f5.hash_weights(cold)!=trained
        or test_score["selected_actions"]!=repeated["selected_actions"]
        or not np.allclose(test_score["per_group_regret"],
                           repeated["per_group_regret"],rtol=0,atol=1e-12)):
        raise ValueError("F8R19_COLD_RESTORED_ACTION_DRIFT")
    return {"arm":arm,"seed":seed,
       "parameters":sum(p.numel() for p in model.parameters()),
       "updates":UPDATES,"initial_hash":initial,"trained_hash":trained,
       "train_seconds":elapsed,
       "first16_loss":float(np.mean(losses[:16])),
       "last16_loss":float(np.mean(losses[-16:])),
       "dev_regret_at_update80":dev_half,
       "dev":dev_score,"test":test_score,
       "cold_restore_action_parity":True}


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,dev,score_eps=cohort("fit"),cohort("dev"),cohort("score")
    support=audit_fit_support(fit)
    control=r18._fixed_controls(score_eps)
    learned=learn_factual_memory(fit)
    learned_test=score_factual_memory(learned,score_eps)
    if (learned_test["mean_regret"]>=.001
        or learned_test["A_sign_action_contrast_count"]!=32):
        raise ValueError("F8R19_FIT_CONVENTIONAL_MEMORY_NOT_IDENTIFIABLE")
    sampler=torch.Generator(device="cpu")
    sampler.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=sampler).tolist()
    start=time.perf_counter()
    rows=[train_one(arm,seed,fit,dev,score_eps,batches)
          for arm in ARMS for seed in SEEDS]
    baselines=[r for r in rows if r["arm"].endswith("_return_focus")
               and r["arm"] not in ("native_return_focus",)]
    strongest=min(baselines,key=lambda x:
                  (x["dev"]["mean_regret"],x["arm"],x["seed"]))
    base_fixed=min(control["means"].values())
    histfree=control["per_group"][control["best_fixed"]]
    draws=np.random.default_rng(BOOT_SEED).integers(0,16,size=(BOOT,16))
    uniform_by_seed={r["seed"]:r for r in rows if r["arm"]=="native_uniform"}
    candidates=[]
    for row in rows:
        if row["arm"]!="native_return_focus": continue
        sc=row["test"]
        improve=np.asarray(histfree)-np.asarray(sc["per_group_regret"])
        lower=float(np.percentile(improve[draws].mean(axis=1),2.5))
        intact=float(sc["mean_regret"])
        erased=float(sc["erased_initial_A_mean_regret"])
        conv=float(strongest["test"]["mean_regret"])
        passed=bool(intact<=.9*base_fixed
           and intact<=conv+1e-4
           and sc["A_sign_action_contrast_count"]>=24
           and erased-intact>=.005
           and lower>0 and learned_test["mean_regret"]<.001)
        candidates.append({"seed":row["seed"],"mean_regret":intact,
           "erased_A_regret":erased,
           "contrast_pairs":sc["A_sign_action_contrast_count"],
           "uniform_native_same_seed_regret":
              uniform_by_seed[row["seed"]]["test"]["mean_regret"],
           "best_DEV_selected_conventional_TEST_regret":conv,
           "bootstrap_2p5_improvement_over_historyfree":lower,
           "candidate_dev_win":passed})
    return {"schema":SCHEMA,
      "classification":"SOURCE_NEUTRAL_NATIVE_TRAINING_DEVELOPMENT_NOT_INDEPENDENT_SCIENCE",
      "freeze_blob":FREEZE_BLOB,
      "source_neutral_fit_support":support,
      "source_neutral_factual_FIT_effects":len(fit)*10,
      "source_neutral_DEV_factual_effects":len(dev)*9,
      "source_neutral_SCORE_factual_effects":len(score_eps)*9,
      "models":rows,"model_count":len(rows),
      "total_optimizer_updates":len(rows)*UPDATES,
      "score_model_selected_toy_decisions":len(rows)*len(score_eps)*2,
      "fixed_controls":control,
      "learned_memory_coefficients_from_factual_FIT":learned,
      "learned_memory_score":learned_test,
      "best_DEV_selected_conventional":{
          "arm":strongest["arm"],"seed":strongest["seed"],
          "dev_regret":strongest["dev"]["mean_regret"],
          "score_regret":strongest["test"]["mean_regret"]},
      "native_focus_candidate_decisions":candidates,
      "native_focus_dev_gate_pass":sum(z["candidate_dev_win"] for z in candidates)>=2,
      "controls_independently_convergence_qualified":False,
      "external_source_effects":0,"F7_original_replay":False,
      "scientific_pass":False,"deployment":"NONE",
      "process_training_evaluation_wall_seconds":time.perf_counter()-start}


def save_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r19_balanced_return_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R19_EVIDENCE_OVERWRITE_REFUSED")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_source_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R19_ORIGINAL_F8_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
