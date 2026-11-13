"""R32: unseen quartic factual intervention; neural transfer and interference.

New source-neutral mathematical law, not original CausalGym/F7 external science.
R31's positive synthetic result and R29B's neural negatives stay unchanged.
Oracle alternate-action effects are computed ONLY by the evaluator.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time
import torch

from experiments import p85mr_f8r29b_neural_information as b
from experiments import p85mr_f8r31_factual_conditional_risk as r31

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r32_novel_quartic_freeze.json"
FREEZE_BLOB="38f4bc3bd5fd12ab50ed9b6c6420a585b255ed73"
SCHEMA="ttt-hybrid.f8r32-novel-quartic-intervention.v1"
EARLY=((-0.15,0),(.15,0),(-.15,1),(.15,1))
PANELS=("dev","ood")
POLICIES=b.POLICIES
GROUPS=12
RETURN={"dev":.78,"ood":.72}
QABS={"dev":1.,"ood":.85}
BSTEPS={"dev":8,"ood":16}
PROBE=.42
GOAL=1.
FIT_GROUPS=64
FIT_MAG=(.7,1.,1.2)
BATCH=8
UPDATES=256
SAMPLES_SEED=20649999
BOOT_SEED=20679999
BOOT_DRAWS=10000
ARMS=tuple((arm,seed) for arm in b.ARMS
           for seed in (b.SEEDS if arm=="native_dynamics" else b.SEEDS[:1]))


def _blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R32_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    d,t,s,a=f["law"],f["train"],f["score"],f["authority"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_TRAINING_AND_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_UNSEEN_QUARTIC_CAUSAL_DEV_ONLY" or
        f["parent"]["head"]!="784f068d9db28778556101c8471c66d589e65a6e" or
        f["parent"]["r31_freeze_blob"]!="d14dcad1a36bb472dc3b3b70b94f9e1b1b4ad1e0" or
        f["parent"]["r29b_source_blob"]!=b._blob((ROOT/"experiments/p85mr_f8r29b_neural_information.py").read_bytes()) or
        d["early_states"]!=[-.15,.15] or
        d["early_actions"]!=[0,1] or
        d["fifth_state"]!=PROBE or
        d["return_state"]!=RETURN or
        d["hidden_q_abs"]!={"fit":[.7,1,1.2],"dev":1,"ood":.85} or
        d["B_steps"]!={"fit":8,"dev":8,"ood":16} or
        d["independent_groups"]!={"fit":64,"dev":12,"ood":12} or
        t["checkpoints"]!=[128,256] or
        t["final_updates_each"]!=UPDATES or
        t["total_final_updates"]!=UPDATES*len(ARMS) or
        t["fit_group_train"]!=48 or t["fit_group_validation"]!=16 or
        t["fit_episodes_per_group"]!=48 or
        t["model_arms"]!=list(b.ARMS) or t["native_seeds"]!=list(b.SEEDS) or
        t["capacity_FLOPs_matched"] is not False or
        s["worlds_per_panel"]!=48 or s["panels"]!=list(PANELS) or
        s["pair_bootstrap_draws"]!=BOOT_DRAWS or
        s["pair_bootstrap_seed"]!=BOOT_SEED or
        s["independent_scientific_pass"] is not False or
        any(v is not False for v in a.values())):
        raise ValueError("F8R32_PROTOCOL_OR_AUTHORITY_DRIFT")
    b.locked()
    r31.locked()
    return f


def after(state,action,context,q):
    if action not in (0,1) or context not in (0,1) or not all(
        math.isfinite(z) for z in (state,q)):
        raise ValueError("F8R32_INVALID_PHYSICS")
    curvature=state**4 - .15**4
    if abs(curvature)<1e-14:
        curvature=0.
    return .25*state-.08+.05*math.sin(3*state)+(
        3.5*q*curvature if action==context else 0.)


def B_history(group,n):
    fit=20640000<=group<20640064 and n==8
    dev=20650000<=group<20650012 and n==8
    ood=20660000<=group<20660012 and n==16
    if type(group) is not int or not (fit or dev or ood):
        raise ValueError("F8R32_INVALID_B_STREAM")
    return tuple((t%2,round(.031*math.sin((group+47)*.23+t*.51)
                             +.017*(t%3)-.02,8)) for t in range(n))


def _fit_episode(group,context,sign,mag,fifth,ret):
    if (type(group) is not int or not 0<=group<64 or
        context not in (0,1) or sign not in (-1,1) or
        mag not in FIT_MAG or fifth not in (0,1) or ret not in (0,1)):
        raise ValueError("F8R32_UNREGISTERED_FIT")
    q=sign*mag  # generator only; never a model-visible label.
    rec=[b._record(s,a,after(s,a,context,q),context) for s,a in EARLY]
    rec.append(b._record(PROBE,fifth,after(PROBE,fifth,context,q),context))
    rec.extend(b._record(0.,a,y,context) for a,y in B_history(20640000+group,8))
    actual_return=after(RETURN["dev"],ret,context,q)
    rec.append(b._record(RETURN["dev"],ret,actual_return,context))
    actions=torch.stack([r[0] for r in rec])
    effects=torch.stack([r[1] for r in rec])
    if actions.shape!=(14,16) or effects.shape!=(14,16):
        raise ValueError("F8R32_UNQUALIFIED_FIT_SHAPE")
    return {"action":actions,"effect":effects,
            "physical_target":effects[-1,0].clone(),
            "direct_target":torch.tensor(.1*(actual_return-GOAL)**2,
                                          dtype=torch.float32)}


def fit_cohort():
    locked()
    return [_fit_episode(g,c,z,m,fifth,ret) for g in range(FIT_GROUPS)
            for c in (0,1) for z in (-1,1) for m in FIT_MAG
            for fifth in (0,1) for ret in (0,1)]


def world(panel,group,context,sign):
    if (panel not in PANELS or type(group) is not int or
        not 0<=group<GROUPS or context not in (0,1) or
        sign not in (-1,1)):
        raise ValueError("F8R32_UNREGISTERED_EVAL_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":sign*QABS[panel],
            "return_state":RETURN[panel],"goal":GOAL,
            "b_history":B_history((20650000 if panel=="dev" else 20660000)+group,BSTEPS[panel])}


def worlds(panel):
    return [world(panel,g,c,z) for g in range(GROUPS)
            for c in (0,1) for z in (-1,1)]


def probe_action(policy,panel,group,context):
    if (policy not in POLICIES or panel not in PANELS or
        group not in range(GROUPS) or context not in (0,1)):
        raise ValueError("F8R32_INVALID_PROBE_POLICY")
    if policy=="active_information":
        return context
    if policy=="fixed_probe0":
        return 0
    if policy=="fixed_probe1":
        return 1
    raw=f"f8r32/{panel}/{group}/{context}".encode()
    return hashlib.sha256(raw).digest()[0]&1


def candidate_inputs(w,policy,erase=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    fifth=probe_action(policy,w["panel"],w["group"],c)
    # Exactly one selected real fifth effect; no counterfactual fifth effect.
    obs=after(PROBE,fifth,c,q)
    rec=[b._record(s,a,after(s,a,c,q),c) for s,a in EARLY]
    rec.append(b._record(PROBE,fifth,obs,c))
    rec.extend(b._record(0.,a,y,c) for a,y in w["b_history"])
    actions=torch.stack([v[0] for v in rec])
    effects=torch.stack([v[1] for v in rec])
    if erase:
        effects[4].zero_()
    two=[]
    for a in (0,1):
        key=b._record(w["return_state"],a,w["return_state"],c)[0]
        two.append((torch.cat([actions,key.unsqueeze(0)]),
                    torch.cat([effects,torch.zeros((1,16))])))
    return torch.stack([v[0] for v in two]),torch.stack([v[1] for v in two])


def evaluator_regret(w,chosen):
    """The only function permitted to compute UNCHOSEN return consequences."""
    if chosen not in (0,1):
        raise ValueError("F8R32_INVALID_RETURN")
    costs=[(after(w["return_state"],a,w["public_context"],
                  w["q_evaluator_only"])-w["goal"])**2 for a in (0,1)]
    return max(0.,costs[chosen]-min(costs))


def _ridge_features(actions,effects):
    state,ctx,action=r31._decode_key(actions[-1])
    probe_state,probe_context,fifth=r31._decode_key(actions[4])
    if abs(probe_state-PROBE)>1e-4 or probe_context!=ctx:
        raise ValueError("F8R32_INCORRECT_PREACTION_PROBE")
    measured=bool(torch.count_nonzero(effects[4]).item())
    active=float(action==ctx)
    informative=bool(active and fifth==ctx and measured)
    delta=r31._physical_delta(effects[4,0]) if informative else 0.
    return ([1.,state,math.sin(3*state),active,active*state**4,
             float(informative)*delta,float(informative)*delta*state**4],
            "safe" if not active else ("informed" if informative else "uncertain"))


def fit_factual_control(fit):
    if len(fit)!=3072:
        raise ValueError("F8R32_FIT_COHORT_SIZE")
    records=[(_ridge_features(x["action"],x["effect"]),
              r31._physical_delta(x["physical_target"])) for x in fit[:2304]]
    X=torch.tensor([v[0][0] for v in records],dtype=torch.float64)
    Y=torch.tensor([v[1] for v in records],dtype=torch.float64)
    reg=.00001*torch.eye(X.shape[1],dtype=torch.float64)
    reg[0,0]=0.
    w=torch.linalg.solve(X.T@X+reg,X.T@Y)
    residual=(X@w-Y).tolist()
    variance={bucket:statistics.fmean(residual[i]**2
        for i,x in enumerate(records) if x[0][1]==bucket)
        for bucket in ("safe","informed","uncertain")}
    validation=[(_ridge_features(x["action"],x["effect"]),
                 r31._physical_delta(x["physical_target"])) for x in fit[2304:]]
    mse=statistics.fmean((sum(a*v for a,v in zip(x[0],w.tolist()))-y)**2
                          for x,y in validation)
    grad=(X.T@(X@w-Y)+reg@w).abs().max().item()
    return {"weights":w.tolist(),"variance":variance,
            "validation_mse":mse,"fit_mse":statistics.fmean(x*x for x in residual),
            "normal_gradient_max":grad,"fit_examples":2304,
            "validation_examples":768,"feature_count":len(w)}


def _choices(arm,model,panel,policy,erase=False):
    ww=worlds(panel)
    tensors=[candidate_inputs(w,policy,erase) for w in ww]
    actions=torch.cat([z[0] for z in tensors])
    effects=torch.cat([z[1] for z in tensors])
    result=[]
    if arm=="factual_control":
        for i,w in enumerate(ww):
            risks=[]
            for a in (0,1):
                x,bucket=_ridge_features(actions[i*2+a],effects[i*2+a])
                normalized=sum(v*u for v,u in zip(x,model["weights"]))
                expected=w["return_state"]+normalized
                risks.append((expected-w["goal"])**2+model["variance"][bucket])
            result.append(min((0,1),key=lambda a:(risks[a],a)))
    else:
        instance=model
        instance.eval()
        with torch.no_grad():
            pred=b._predict(arm,instance,{"action":actions,"effect":effects})
        if pred.shape!=effects.shape or not bool(torch.isfinite(pred).all()):
            raise ValueError("F8R32_INVALID_PREACTION_MODEL_OUTPUT")
        vals=pred[:,-1,0].detach().tolist()
        for i,w in enumerate(ww):
            if arm=="native_direct_value":
                costs=[vals[2*i],vals[2*i+1]]
            else:
                costs=[(w["return_state"]+2.4*math.atanh(
                         max(-.98,min(.98,vals[2*i+a])))-w["goal"])**2
                       for a in (0,1)]
            result.append(min((0,1),key=lambda a:(costs[a],a)))
    return result


def score_policy(arm,model,panel,policy,erase=False):
    ww=worlds(panel)
    choices=_choices(arm,model,panel,policy,erase)
    regrets=[evaluator_regret(w,a) for w,a in zip(ww,choices)]
    return {"regret":statistics.fmean(regrets),
            "pergroup":[statistics.fmean(regrets[g*4:(g+1)*4])
                        for g in range(GROUPS)],
            "choices":choices,"action1_count":sum(choices)}


def score_arm(arm,model):
    out={}
    for panel in PANELS:
        pol={policy:score_policy(arm,model,panel,policy) for policy in POLICIES}
        erased=score_policy(arm,model,panel,"active_information",erase=True)
        out[panel]={"policies":pol,
                    "erased_active":erased,
                    "erasure_regret_delta":
                        erased["regret"]-pol["active_information"]["regret"],
                    "erasure_decision_flips":sum(a!=b for a,b in zip(
                        erased["choices"],pol["active_information"]["choices"]))}
    return out


def _batch(fit,ids):
    rows=[fit[i] for i in ids]
    return {key:torch.stack([row[key] for row in rows])
            for key in ("action","effect","physical_target","direct_target")}


def _validation_loss(arm,model,fit):
    accum=[]
    model.eval()
    with torch.no_grad():
        for at in range(2304,len(fit),64):
            batch=_batch(fit,list(range(at,min(at+64,len(fit)))))
            out=b._predict(arm,model,{"action":batch["action"],
                                       "effect":batch["effect"]})
            target=(batch["direct_target"] if arm=="native_direct_value"
                    else batch["physical_target"])
            accum.extend((out[:,-1,0]-target).square().tolist())
    return statistics.fmean(accum)


def train_arm(arm,seed,fit,samples):
    model=b.model_for(arm,seed)
    digest=b.r25.r22.r18.r5.f5.hash_weights
    initial=digest(model)
    opt=torch.optim.AdamW(model.parameters(),lr=.002 if arm.startswith(
        "native") else .001,weight_decay=.01)
    start=time.perf_counter()
    trace=[]
    for step,ids in enumerate(samples,1):
        batch=_batch(fit,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        out=b._predict(arm,model,{"action":batch["action"],
                                   "effect":batch["effect"]})
        if out.shape!=batch["effect"].shape:
            raise ValueError("F8R32_UNQUALIFIED_TRAIN_SHAPE")
        target=(batch["direct_target"] if arm=="native_direct_value"
                else batch["physical_target"])
        loss=(out[:,-1,0]-target).square().mean()
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R32_NONFINITE_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in gradients)
            or any(g is not None and not bool(torch.isfinite(g).all())
                   for g in gradients)):
            raise ValueError("F8R32_ZERO_OR_INVALID_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if step in (128,256):
            trace.append({"updates":step,"last_batch_loss":float(loss.detach()),
                          "heldout_fit_validation_mse":
                              _validation_loss(arm,model,fit)})
    final=digest(model)
    if final==initial or len(trace)!=2:
        raise ValueError("F8R32_MODEL_DID_NOT_TRAIN")
    panels=score_arm(arm,model)
    restored=b.model_for(arm,seed)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (digest(restored)!=final or any(
        _choices(arm,restored,p,"active_information") !=
        panels[p]["policies"]["active_information"]["choices"]
        for p in PANELS)):
        raise ValueError("F8R32_COLD_RESTORE_DRIFT")
    return {"arm":arm,"seed":seed,"optimizer_updates":UPDATES,
            "parameter_count":sum(p.numel() for p in model.parameters()),
            "initial_hash":initial,"trained_hash":final,
            "wall_seconds":time.perf_counter()-start,
            "validation_trace":trace,"cold_restore_actions_equal":True,
            "panels":panels}


def _paired_lower(active,other):
    rng=random.Random(BOOT_SEED)
    result=[]
    for _ in range(BOOT_DRAWS):
        ix=[rng.randrange(GROUPS) for __ in range(GROUPS)]
        result.append(statistics.fmean(other[i]-active[i] for i in ix))
    return sorted(result)[int(.025*BOOT_DRAWS)]


def _judge(rows,freeze):
    s=freeze["score"]
    diagnostics=[]
    for row in rows:
        by_panel={}
        for panel in PANELS:
            p=row["panels"][panel]
            pol=p["policies"]
            active=pol["active_information"]
            fixed=min(pol["fixed_probe0"]["regret"],pol["fixed_probe1"]["regret"])
            other=(pol["fixed_probe0"] if pol["fixed_probe0"]["regret"]<=
                   pol["fixed_probe1"]["regret"] else pol["fixed_probe1"])
            gain=fixed-active["regret"]
            low=_paired_lower(active["pergroup"],other["pergroup"])
            er=p["erasure_regret_delta"]
            ok=(gain>=s["development_native_active_vs_best_fixed_gain_min"]
                and er>=s["development_native_erasure_regret_increase_min"]
                and low>s["development_native_bootstrap_lower_gt"])
            by_panel[panel]={"active_regret":active["regret"],
                "gain_over_best_fixed":gain,"erasure_regret_penalty":er,
                "paired_2p5_lower":low,"dev_win":bool(ok)}
        diagnostics.append({"arm":row["arm"],"seed":row["seed"],
            "panels":by_panel,"both_panel_win":all(
                z["dev_win"] for z in by_panel.values())})
    natives=sum(x["both_panel_win"] for x in diagnostics
                if x["arm"]=="native_dynamics")
    return {"native_winning_seeds":natives,
            "source_neutral_native_development_pass":
                natives>=s["development_native_win_both_panels_at_least"],
            "judgments":diagnostics,"independent_scientific_pass":False}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=fit_cohort()
    if len(fit)!=3072:
        raise ValueError("F8R32_FACTORIAL_FIT_GAP")
    control=fit_factual_control(fit)
    control_panels=score_arm("factual_control",control)
    generator=torch.Generator(device="cpu")
    generator.manual_seed(SAMPLES_SEED)
    samples=torch.randint(0,2304,(UPDATES,BATCH),
                          generator=generator).tolist()
    rows=[train_arm(arm,seed,fit,samples) for arm,seed in ARMS]
    judgement=_judge(rows,f)
    return {"schema":SCHEMA,
        "classification":"SOURCE_NEUTRAL_UNSEEN_QUARTIC_CAUSAL_DEV_ONLY",
        "freeze_blob":FREEZE_BLOB,
        "source_parent":"784f068d9db28778556101c8471c66d589e65a6e",
        "factual_fit_episodes":len(fit),
        "factual_fit_effects":len(fit)*14,
        "train_factual_episodes":2304,
        "validation_factual_episodes":768,
        "factual_control":{"fit":control,"panels":control_panels},
        "neural_model_instances":len(rows),
        "optimizer_updates":len(rows)*UPDATES,
        "models":rows,"judgement":judgement,
        "all_new_synthetic_score_worlds":2*GROUPS*4,
        "capacity_FLOPs_matched":False,
        "conventional_independent_convergence_qualified":False,
        "independent_scientific_pass":False,
        "original_external_effects":0,"spent_F7_replay":False,
        "deployment":"NONE"}


def retain_ci_receipt(data):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r32_novel_quartic_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R32_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(data,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R32_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
