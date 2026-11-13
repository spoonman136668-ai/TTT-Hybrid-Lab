"""F8-R33: prospectively frozen FIT-only probe-indicator/centering falsifier.

The R32 mathematical law is reused, but FIT return states and scored worlds
are independent. Only the actually selected factual return labels enter FIT.
An unselected return outcome is computed exclusively in evaluate(). Handcoded
nine-feature interactions are a conventional explanatory baseline, not evidence
that the neural Hybrid learned any of those interactions.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics

import torch
from experiments import p85mr_f8r29b_neural_information as b
from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r32_novel_quartic as r32

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r33_probe_calibration_freeze.json"
FREEZE_BLOB="94e7f63370143e7423a96ecd94937d9a5a31b097"
SCHEMA="ttt-hybrid.f8r33-probe-calibration-v1"
FIT_STATES=(.54,.62,.70,.78)
PANELS=("dev","ood")
POLICIES=b.POLICIES
GROUPS=12
GOAL=1.
PROBE=.42
FIT_COUNT=3072
FIT_GROUP_SIZE=48
RIDGE=.00001
SALT={"fit":20730000,"dev":20740000,"ood":20750000}
SEED_BOOT=20769999
DRAWS=10000
BUCKETS=("safe","informed","uncertain")
MODELS=("legacy","indicator","centered")


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    data=FREEZE.read_bytes()
    if _blob(data)!=FREEZE_BLOB:
        raise ValueError("F8R33_FREEZE_BLOB_DRIFT")
    f=json.loads(data)
    t,m,s,a=f["task"],f["methods"],f["precommitted_score"],f["authority"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_OR_NUMERIC_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_FACTUAL_CALIBRATION_DEV_ONLY" or
        f["parent"]["head"]!="8f0b42075784509fe0616e8c3a3577ac38e5947d" or
        f["parent"]["r32_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r32_novel_quartic.py").read_bytes()) or
        f["parent"]["r31_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r31_factual_conditional_risk.py").read_bytes()) or
        f["parent"]["r32_freeze_blob"]!=r32.FREEZE_BLOB or
        t["fit_groups"]!=64 or t["fit_episodes_per_group"]!=FIT_GROUP_SIZE or
        t["return_state_by_group_mod4"]!=list(FIT_STATES) or
        t["independent_B_salts"]!=SALT or
        t["fit_training_episodes"]!=2304 or
        t["fit_validation_episodes"]!=768 or
        t["score_groups_per_panel"]!=GROUPS or
        t["score_worlds_per_panel"]!=48 or
        t["dev"]!={"return_states_even_odd":[.6,.7],"q_abs":1.1,"B_length":8} or
        t["ood"]!={"return_states_even_odd":[.58,.74],"q_abs":.9,"B_length":16} or
        t["policies"]!=list(POLICIES) or
        m["ridge_lambda"]!=RIDGE or
        m["fit_groups_curve"]!=[12,24,48] or
        m["zero_neural_updates"] is not True or
        s["bootstrap_draws"]!=DRAWS or s["bootstrap_seed"]!=SEED_BOOT or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in a.values())):
        raise ValueError("F8R33_PROTOCOL_OR_AUTHORITY_DRIFT")
    r32.locked()
    return f


def _B(panel,group):
    if panel not in ("fit","dev","ood") or type(group) is not int:
        raise ValueError("F8R33_INVALID_B_GROUP")
    hi=64 if panel=="fit" else GROUPS
    if not 0<=group<hi:
        raise ValueError("F8R33_INVALID_B_GROUP")
    steps=16 if panel=="ood" else 8
    salt=SALT[panel]
    return tuple((t%2,round(.019*math.sin((salt+group)*.31+t*.47)
                 +.013*((t+group)%3)-.017,8)) for t in range(steps))


def _episode(group,context,sign,mag,fifth,ret):
    if (type(group) is not int or not 0<=group<64 or
        context not in (0,1) or sign not in (-1,1) or
        mag not in (.7,1.,1.2) or fifth not in (0,1) or ret not in (0,1)):
        raise ValueError("F8R33_UNREGISTERED_FIT")
    q=sign*mag
    state=FIT_STATES[group%4]
    rows=[b._record(s,a,r32.after(s,a,context,q),context) for s,a in r32.EARLY]
    rows.append(b._record(PROBE,fifth,r32.after(PROBE,fifth,context,q),context))
    rows.extend(b._record(0.,a,e,context) for a,e in _B("fit",group))
    actual=r32.after(state,ret,context,q)
    rows.append(b._record(state,ret,actual,context))
    actions=torch.stack([r[0] for r in rows])
    effects=torch.stack([r[1] for r in rows])
    if actions.shape!=(14,16) or effects.shape!=(14,16):
        raise ValueError("F8R33_FACTUAL_FIT_SHAPE")
    return {"action":actions,"effect":effects,
            "physical_target":effects[-1,0].clone(),
            "direct_target":torch.tensor(.1*(actual-GOAL)**2,
                                         dtype=torch.float32)}


def fit_cohort():
    locked()
    result=[_episode(g,c,sign,mag,fifth,ret)
            for g in range(64) for c in (0,1) for sign in (-1,1)
            for mag in (.7,1.,1.2) for fifth in (0,1) for ret in (0,1)]
    if len(result)!=FIT_COUNT:
        raise ValueError("F8R33_INCOMPLETE_FACTORIAL_FIT")
    return result


def world(panel,group,context,sign):
    if (panel not in PANELS or type(group) is not int or
        not 0<=group<GROUPS or context not in (0,1) or sign not in (-1,1)):
        raise ValueError("F8R33_INVALID_SCORE_WORLD")
    params={"dev":([.6,.7],1.1),"ood":([.58,.74],.9)}
    state=params[panel][0][group%2]
    q=sign*params[panel][1]
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":q,"return_state":state,"goal":GOAL,
            "b_history":_B(panel,group)}


def worlds(panel):
    return [world(panel,g,c,z) for g in range(GROUPS)
            for c in (0,1) for z in (-1,1)]


def _probe(policy,panel,group,context):
    if policy not in POLICIES:
        raise ValueError("F8R33_UNREGISTERED_POLICY")
    if policy=="active_information":
        return context
    if policy=="fixed_probe0":
        return 0
    if policy=="fixed_probe1":
        return 1
    return hashlib.sha256(
        f"f8r33/{panel}/{group}/{context}".encode()).digest()[0]&1


def candidate_inputs(w,policy,erase=False):
    context,q=w["public_context"],w["q_evaluator_only"]
    fifth=_probe(policy,w["panel"],w["group"],context)
    # Simulator delivers just the SELECTED fifth consequence, no other action.
    selected=r32.after(PROBE,fifth,context,q)
    rows=[b._record(s,a,r32.after(s,a,context,q),context)
          for s,a in r32.EARLY]
    rows.append(b._record(PROBE,fifth,selected,context))
    rows.extend(b._record(0.,a,e,context) for a,e in w["b_history"])
    actions=torch.stack([r[0] for r in rows])
    effects=torch.stack([r[1] for r in rows])
    if erase:
        effects[4].zero_()
    choices=[]
    for action in (0,1):
        key=b._record(w["return_state"],action,w["return_state"],context)[0]
        choices.append((torch.cat((actions,key.unsqueeze(0))),
                        torch.cat((effects,torch.zeros((1,16))))))
    return torch.stack([x[0] for x in choices]),torch.stack([x[1] for x in choices])


def _fit_safe_probe_offset(rows):
    # FIT-only factual non-informative (selected fifth action != public ctx).
    data=[]
    for row in rows:
        _,ctx,probe=r31._decode_key(row["action"][4])
        if probe!=ctx:
            data.append(r31._physical_delta(row["effect"][4,0]))
    if len(data)*2!=len(rows):
        raise ValueError("F8R33_UNBALANCED_FACTUAL_BASELINE")
    result=statistics.fmean(data)
    if not math.isfinite(result):
        raise ValueError("F8R33_INVALID_FACTUAL_PROBE_REFERENCE")
    return result


def _features(actions,effects,*,method,probe_offset):
    if method not in ("indicator","centered"):
        raise ValueError("F8R33_UNREGISTERED_FEATURE_METHOD")
    state,ctx,ret=r31._decode_key(actions[-1])
    probe_state,pctx,fifth=r31._decode_key(actions[4])
    if abs(probe_state-PROBE)>1e-4 or pctx!=ctx:
        raise ValueError("F8R33_BAD_FACTUAL_PROBE")
    measured=bool(torch.count_nonzero(effects[4]).item())
    active=int(ret==ctx)
    informed=int(active and fifth==ctx and measured)
    delta=(r31._physical_delta(effects[4,0]) if informed else 0.)
    if method=="centered":
        delta-=probe_offset if informed else 0.
    curvature=state**4-.15**4
    return ([1.,state,math.sin(3*state),float(active),
             active*curvature,float(informed),informed*curvature,
             informed*delta,informed*delta*curvature],
            "safe" if not active else ("informed" if informed else "uncertain"))


def _fit_model(fit,groups,method):
    if groups not in (12,24,48) or len(fit)!=FIT_COUNT or method not in (
        "indicator","centered"):
        raise ValueError("F8R33_UNREGISTERED_FIT_CONFIGURATION")
    rows=fit[:groups*FIT_GROUP_SIZE]
    offset=_fit_safe_probe_offset(rows)
    decoded=[(_features(r["action"],r["effect"],method=method,
                        probe_offset=offset),
              r31._physical_delta(r["physical_target"])) for r in rows]
    X=torch.tensor([r[0][0] for r in decoded],dtype=torch.float64)
    Y=torch.tensor([r[1] for r in decoded],dtype=torch.float64)
    reg=RIDGE*torch.eye(X.shape[1],dtype=torch.float64)
    reg[0,0]=0.
    weights=torch.linalg.solve(X.T@X+reg,X.T@Y)
    residual=X@weights-Y
    gradient=X.T@residual+reg@weights
    variance={bucket:statistics.fmean(float(residual[i]**2)
              for i,row in enumerate(decoded) if row[0][1]==bucket)
              for bucket in BUCKETS}
    validation=[]
    for row in fit[48*FIT_GROUP_SIZE:]:
        x,_=_features(row["action"],row["effect"],method=method,
                      probe_offset=offset)
        truth=r31._physical_delta(row["physical_target"])
        validation.append((math.fsum(v*w for v,w in zip(x,weights.tolist()))
                           -truth)**2)
    if (not bool(torch.isfinite(weights).all()) or
        not all(math.isfinite(v) and v>=0 for v in variance.values()) or
        len(validation)!=768):
        raise ValueError("F8R33_INVALID_FACTUAL_FIT")
    return {"method":method,"groups":groups,"factual_examples":len(rows),
            "fit_only_probe_offset":offset,"weights":weights.tolist(),
            "variance":variance,"fit_mse":float((residual**2).mean()),
            "fit_validation_mse":statistics.fmean(validation),
            "normal_gradient_abs_max":float(gradient.abs().max()),
            "ridge_lambda":RIDGE}


def _legacy(fit):
    model=r32.fit_factual_control(fit)
    return {"method":"legacy","groups":48,"factual_examples":2304,
            "weights":model["weights"],"variance":model["variance"],
            "fit_mse":model["fit_mse"],
            "fit_validation_mse":model["validation_mse"],
            "normal_gradient_abs_max":model["normal_gradient_max"],
            "ridge_lambda":RIDGE}


def _estimate(method,model,actions,effects,risk_aware):
    if method=="legacy":
        x,bucket=r32._ridge_features(actions,effects)
    else:
        x,bucket=_features(actions,effects,method=method,
                    probe_offset=model["fit_only_probe_offset"])
    pred=math.fsum(v*w for v,w in zip(x,model["weights"]))
    state=r31._decode_key(actions[-1])[0]
    var=model["variance"][bucket] if risk_aware else 0.
    return (state+pred,var,bucket)


def _choose(model,w,policy,risk_aware,erase):
    acts,obs=candidate_inputs(w,policy,erase)
    estimates=[_estimate(model["method"],model,acts[a],obs[a],risk_aware)
               for a in (0,1)]
    risks=[(a[0]-w["goal"])**2+a[1] for a in estimates]
    return min((0,1),key=lambda a:(risks[a],a))


def evaluate_regret_only(worlds_,choices):
    """Unselected return physical outcomes are legal ONLY in evaluator."""
    if len(worlds_)!=48 or len(choices)!=48:
        raise ValueError("F8R33_UNBALANCED_EVALUATION")
    values=[]
    for w,a in zip(worlds_,choices):
        losses=[(r32.after(w["return_state"],chosen,w["public_context"],
                            w["q_evaluator_only"])-w["goal"])**2
                for chosen in (0,1)]
        values.append(max(0.,losses[a]-min(losses)))
    return values


def _score(model,panel,policy,*,risk=True,erase=False):
    ww=worlds(panel)
    choices=[_choose(model,w,policy,risk,erase) for w in ww]
    regrets=evaluate_regret_only(ww,choices)
    return {"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[g*4:(g+1)*4])
                              for g in range(GROUPS)],
            "choices":choices,"chosen_action1":sum(choices),
            "selected_fifth_actual_effects":48}


def _score_all(model):
    result={}
    for panel in PANELS:
        controls={policy:_score(model,panel,policy) for policy in POLICIES}
        er=_score(model,panel,"active_information",erase=True)
        zero_risk=_score(model,panel,"active_information",risk=False)
        result[panel]={"policies":controls,"erased_active":er,
             "no_risk_active":zero_risk,
             "erasure_regret_penalty":er["mean_regret"]-
                       controls["active_information"]["mean_regret"],
             "erasure_action_flips":sum(a!=bb for a,bb in zip(
                      er["choices"],controls["active_information"]["choices"]))}
    return result


def _bootstrap_lower(current,previous,seed,draws):
    rng=random.Random(seed)
    gains=[]
    for _ in range(draws):
        chosen=[rng.randrange(GROUPS) for j in range(GROUPS)]
        gains.append(statistics.fmean(previous[i]-current[i] for i in chosen))
    return sorted(gains)[int(.025*draws)]


def _judge(scores,curve,freeze):
    threshold=freeze["precommitted_score"]
    results={}
    diff=math.sqrt(math.fsum((a-bb)**2 for a,bb in zip(
         curve["centered"][1]["weights"],curve["centered"][2]["weights"])))
    stable=diff<=threshold["curve_coeff_L2_diff_24_48_max"]
    grads=all(m["normal_gradient_abs_max"]<=
              threshold["fit_normal_equation_gradient_abs_max"]
              for seq in curve.values() for m in seq)
    for panel in PANELS:
        active=scores["centered"][panel]["policies"]["active_information"]
        legacy=scores["legacy"][panel]["policies"]["active_information"]
        baseline=min(scores["centered"][panel]["policies"][k]["mean_regret"]
             for k in ("fixed_probe0","fixed_probe1"))
        er=scores["centered"][panel]["erasure_regret_penalty"]
        gain=legacy["mean_regret"]-active["mean_regret"]
        lower=_bootstrap_lower(active["pergroup_regret"],
                     legacy["pergroup_regret"],
                     threshold["bootstrap_seed"],threshold["bootstrap_draws"])
        passed=bool(
            active["mean_regret"]<=threshold["baseline_mean_regret_centered_active_max_each"]
            and gain>=threshold["improvement_centered_over_legacy_active_min_each"]
            and er>=threshold["erasure_regret_penalty_centered_active_min_each"]
            and baseline-active["mean_regret"]>=
                threshold["active_gain_over_best_fixed_min_each"]
            and lower>threshold["paired_group_bootstrap_centered_over_legacy_lower_2p5_gt"])
        results[panel]={"centered_active_regret":active["mean_regret"],
            "legacy_active_regret":legacy["mean_regret"],
            "gain_over_legacy":gain,"erasure_regret_penalty":er,
            "gain_over_best_fixed":baseline-active["mean_regret"],
            "paired_bootstrap_lower":lower,"panel_development_pass":passed}
    return {"centered_coefficient_drift_24_to_48":diff,
            "curve_coefficient_stability_pass":stable,
            "fit_normal_equations_pass":grads,
            "panels":results,
            "source_neutral_development_pass":
                stable and grads and all(
                    p["panel_development_pass"] for p in results.values()),
            "independent_scientific_pass":False}


def run():
    protocol=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=fit_cohort()
    curves={method:[_fit_model(fit,n,method) for n in (12,24,48)]
            for method in ("indicator","centered")}
    models={"legacy":_legacy(fit),
            "indicator":curves["indicator"][-1],
            "centered":curves["centered"][-1]}
    scores={method:_score_all(model) for method,model in models.items()}
    restored=json.loads(json.dumps(models,sort_keys=True,allow_nan=False))
    if any(_score_all(restored[name])[panel]["policies"][policy]["choices"] !=
           scores[name][panel]["policies"][policy]["choices"]
           for name in MODELS for panel in PANELS for policy in POLICIES):
        raise ValueError("F8R33_COLD_RESTORE_ACTION_DRIFT")
    judgment=_judge(scores,curves,protocol)
    return {"schema":SCHEMA,
       "classification":"SOURCE_NEUTRAL_FACTUAL_CALIBRATION_DEV_ONLY",
       "freeze_blob":FREEZE_BLOB,
       "frozen_R32_parent":"8f0b42075784509fe0616e8c3a3577ac38e5947d",
       "factual_FIT_episodes":len(fit),
       "factual_FIT_effect_records":len(fit)*14,
       "factual_train_episodes":2304,
       "factual_validation_episodes":768,
       "fit_state_groups_per_state":{str(s):sum(g%4==i for g in range(64))
                                      for i,s in enumerate(FIT_STATES)},
       "fit_curve":curves,"models":models,"scores":scores,
       "judgment":judgment,"cold_restore_actions_equal":True,
       "neural_optimizer_updates":0,
       "independent_neural_convergence_qualified":False,
       "capacity_FLOPs_matched":False,
       "independent_scientific_pass":False,
       "original_external_effects":0,"spent_F7_replay":False,
       "deployment":"NONE"}


def retain_ci_receipt(data):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r33_probe_calibration_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R33_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(data,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R33_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
