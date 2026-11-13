"""F8-R26: real B-gradient interference vs A-factual gradient protection.

All source effects are generated from precommitted mathematical laws.
Both adaptation methods compute identical factual B and A gradients on
identical source-neutral batches; only gradient projection differs.
No external scientific source first-use, F7 spent-source reuse or deployment.
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

from experiments import p85mr_f8r25_multimechanism as r25
from experiments import p85mr_f8r24_online_interference as r24
from experiments import p85mr_f8r22_objective_matched as r22

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r26_gradient_conflict_freeze.json"
FREEZE_BLOB="10ed62c741c419f03f3be233188a5e1ff97c82c1"
SCHEMA="ttt-hybrid.f8r26-factual-gradient-conflict.v1"
MODEL_SPECS=(("native_focus20",985201),("native_focus20",985203),
             ("native_focus20",985207),("gru_focus20",985201),
             ("ssm_focus20",985201))
CONDITIONS=("no_update","plain_B_update","orthogonal_B_update")
SPLITS={"B_FIT":(20110000,16),"B_HOLD":(20120000,8),
        "A_SCORE":(20130000,16)}
A_B0=(-.34,-.14,.06,.26)
A_B1=(.34,.14,-.06,-.26)
A_K0=(-.22,.22)
A_K1=(.22,-.22)
RETURN_STATES=(-.15,.15)
GOALS=(-.2,.2)
B_STEPS=16
PRETRAIN=384
ADAPT=24
BATCH=8
B_LR=.0005
BOOT=10000
BOOT_SEED=20139999


def git_blob(raw:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R26_PRECOMMITTED_BLOB_DRIFT")
    f=json.loads(raw)
    w,t,s,a=f["world"],f["training"],f["evaluation"],f["authority"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_BEFORE_IMPLEMENTATION_AND_R26_NUMERICAL_RESULTS"
        or f["classification"]!="AUTHOR_GENERATED_SOURCE_NEUTRAL_DEVELOPMENT_ONLY"
        or f["parent"]["head"]!="582483c38516aaeb88e737345be0ee2d4b52528d"
        or f["parent"]["r25_code_blob"]!="ef488956741b339eb68c786432c48fefb6526165"
        or f["parent"]["r24_code_blob"]!="3fb48eab94ba333f4433bfc163bfabc5e1db9477"
        or git_blob((ROOT/"experiments/p85mr_f8r25_multimechanism.py").read_bytes())!="ef488956741b339eb68c786432c48fefb6526165"
        or git_blob((ROOT/"experiments/p85mr_f8r24_online_interference.py").read_bytes())!="3fb48eab94ba333f4433bfc163bfabc5e1db9477"
        or w["cohorts"]["B_FIT"]!={"seed_base":20110000,"groups":16,"bc_combinations":4,"episodes":64,"selected_effects":1024}
        or w["cohorts"]["B_HOLD"]!={"seed_base":20120000,"groups":8,"bc_combinations":4,"episodes":32,"selected_effects":512}
        or w["cohorts"]["A_SCORE"]!={"seed_base":20130000,"groups":16,"twins":2,"episodes":32,"factual_effects":640}
        or tuple(w["score_return_public_states"])!=RETURN_STATES
        or tuple(w["score_goals"])!=GOALS
        or w["B_policy"]!="R3 public behavior(s) XOR int((split_seed+7*t)%3==0) no hidden sign in selected behavior"
        or [(z["arm"],z["seed"]) for z in t["models"]]!=list(MODEL_SPECS)
        or t["pretrained_updates_total"]!=1920
        or t["B_updates_total"]!=240
        or t["conditions"]!=list(CONDITIONS)
        or t["B_updates_each_adaptive_condition"]!=ADAPT
        or t["B_batch_size"]!=BATCH
        or t["A_replay_batch_size"]!=BATCH
        or t["B_sampler_seed"]!=20119999
        or t["A_sampler_seed"]!=20118888
        or t["B_lr"]!=B_LR
        or s["score_clusters"]!=16
        or s["bootstrap_samples"]!=BOOT
        or s["bootstrap_seed"]!=BOOT_SEED
        or s["native_seed_threshold"]!=2
        or s["independent_scientific_pass"] is not False
        or any(z is not False for z in a.values())):
        raise ValueError("F8R26_PRECOMMITTED_PROTOCOL_OR_SCOPE_DRIFT")
    r25.locked()
    r24.locked()
    return f


def b_after(state:float,action:int,b:int,c:int)->float:
    if (not math.isfinite(state) or type(action) is not int or action not in (0,1)
        or type(b) is not int or b not in (-1,1)
        or type(c) is not int or c not in (-1,1)):
        raise ValueError("F8R26_BAD_INTERACTING_CAUSAL_B_LAW")
    return .29*math.tanh(1.2*state)+.15*b*(2*action-1)+.08*b*c*action


def _record(actions,effects,state,action,after):
    actions.append(r22.r18.r5._action_key((state,0.,0.,0.),action))
    effects.append(r22.r18._observed(state,after))


def episode(split:str,group:int,b:int,c:int,*,twin:int|None=None):
    if (split not in SPLITS or type(group) is not int
        or not 0<=group<SPLITS[split][1]
        or type(b) is not int or b not in (-1,1)
        or type(c) is not int or c not in (-1,1)
        or (split=="A_SCORE" and (type(twin) is not int or twin not in (0,1)))
        or (split!="A_SCORE" and twin is not None)):
        raise ValueError("F8R26_UNKNOWN_REGISTERED_WORLD")
    seed=SPLITS[split][0]+group
    acts,effs=[],[]
    params=None
    if split=="A_SCORE":
        b0,b1=A_B0[group%4],A_B1[(group//4)%4]
        k0,k1=A_K0[(group//2)%2],A_K1[(group//4)%2]
        intercepts=(b0,b1) if twin==0 else (b1,b0)
        slopes=(k0,k1) if twin==0 else (k1,k0)
        params=(intercepts,slopes)
        for state,selected in r25.PROBES:
            after=r25.A_after(state,selected,intercepts,slopes)
            _record(acts,effs,state,selected,after)
    state=0.
    for t in range(B_STEPS):
        if t==8:
            state=0.
        sign=b if t<8 else -b
        selected=r22.r20.r3.behavior(state)^int((seed+7*t)%3==0)
        after=b_after(state,selected,sign,c)
        _record(acts,effs,state,selected,after)
        state=after
    return {"action":torch.tensor(acts,dtype=torch.float32),
            "effect":torch.tensor(effs,dtype=torch.float32),
            "split":split,"group":group,"b":b,"c":c,"twin":twin,
            "params_evaluator_only":params}


def cohort(split):
    locked()
    if split not in SPLITS:
        raise ValueError("F8R26_UNKNOWN_SPLIT")
    if split=="A_SCORE":
        return [episode(split,g,b,c,twin=twin)
                for g in range(SPLITS[split][1])
                for b,c in [((-1,1) if g%2==0 else (1,-1))]
                for twin in (0,1)]
    return [episode(split,g,b,c) for g in range(SPLITS[split][1])
            for b in (-1,1) for c in (-1,1)]


def audit(train,holdout,score):
    if (len(train),len(holdout),len(score))!=(64,32,32):
        raise ValueError("F8R26_FACTUAL_COHORT_COUNT_DRIFT")
    for split,eps,n in (("B_FIT",train,16),("B_HOLD",holdout,16),
                        ("A_SCORE",score,20)):
        for e in eps:
            if (e["action"].shape!=(n,16) or e["effect"].shape!=(n,16)
                or not bool(torch.isfinite(e["effect"]).all())):
                raise ValueError("F8R26_FACTUAL_TENSOR_DRIFT")
    for arr in (train,holdout):
        if {(x["b"],x["c"]) for x in arr}!={(b,c) for b in (-1,1) for c in (-1,1)}:
            raise ValueError("F8R26_BC_CROSSING_INCOMPLETE")
    for i in range(0,32,2):
        a,b=score[i:i+2]
        if (a["twin"]!=0 or b["twin"]!=1 or a["group"]!=b["group"]
            or a["params_evaluator_only"][0]!=b["params_evaluator_only"][0][::-1]
            or a["params_evaluator_only"][1]!=b["params_evaluator_only"][1][::-1]
            or not torch.equal(a["action"],b["action"])
            or not torch.equal(a["effect"][4:],b["effect"][4:])
            or torch.equal(a["effect"][:4],b["effect"][:4])):
            raise ValueError("F8R26_NONMATCHED_INTERFERENCE_TWINS")
    return {"B_train_FIT_episodes":64,"B_train_FIT_effects":1024,
            "B_holdout_episodes":32,"B_holdout_effects":512,
            "A_score_episodes":32,"A_score_observed_effects":640,
            "paired_twins":16}


def factual(e):
    return {"action":e["action"],"effect":e["effect"]}


def actual_regret(ep,state,goal,selected):
    if (state not in RETURN_STATES or goal not in GOALS
        or type(selected) is not int or selected not in (0,1)):
        raise ValueError("F8R26_UNREGISTERED_RETURN_ACTION")
    b,k=ep["params_evaluator_only"]
    true=[r25.A_after(state,q,b,k) for q in (0,1)]
    losses=[(z-goal)**2 for z in true]
    return max(0.,losses[selected]-min(losses))


def _predict_returns(arm,model,eps,erase_probes=False):
    a,e=[],[]
    for ep in eps:
        for state in RETURN_STATES:
            for selected in (0,1):
                key=r22.r18.r5._action_key((state,0.,0.,0.),selected)
                acts=torch.cat([ep["action"],
                     torch.tensor([key],dtype=torch.float32)],dim=0)
                effects=torch.cat([ep["effect"],
                     torch.zeros((1,16),dtype=torch.float32)],dim=0)
                if erase_probes:
                    effects[:4]=0.
                a.append(acts);e.append(effects)
    model.eval()
    with torch.no_grad():
        pred=r22.predict(arm,model,{"action":torch.stack(a),
                                     "effect":torch.stack(e)})
    if (pred.shape!=(len(eps)*4,21,16)
        or not bool(torch.isfinite(pred).all())):
        raise ValueError("F8R26_MODEL_RETURN_QUERY_SHAPE")
    delta=np.asarray([2.4*math.atanh(max(-.98,min(.98,float(v))))
                      for v in pred[:,-1,0].detach().cpu().numpy()],dtype=np.float64)
    return delta.reshape(len(eps),2,2)


def a_score(arm,model,eps):
    normal=_predict_returns(arm,model,eps)
    erased=_predict_returns(arm,model,eps,erase_probes=True)
    chosen,losses,erased_losses=[],[],[]
    for i,ep in enumerate(eps):
        for j,state in enumerate(RETURN_STATES):
            for goal in GOALS:
                def select(pred):
                    return min((0,1),key=lambda q:((state+pred[i,j,q]-goal)**2,q))
                act=select(normal);ab=select(erased)
                chosen.append(act)
                losses.append(actual_regret(ep,state,goal,act))
                erased_losses.append(actual_regret(ep,state,goal,ab))
    groups=[float(np.mean(losses[8*g:8*g+8])) for g in range(16)]
    contrast=sum(chosen[8*g+j]!=chosen[8*g+4+j] for g in range(16) for j in range(4))
    return {"mean_regret":float(np.mean(losses)),
            "erased_A_mean_regret":float(np.mean(erased_losses)),
            "paired_group_regret":groups,"actions":chosen,
            "A_twin_action_contrasts":contrast,"action_count":len(chosen)}


def controls(score):
    fixed={a:[] for a in (0,1)}
    explicit=[]
    for ep in score:
        intercept0,intercept1,slope0,slope1=r25.four_probe_predictions(ep)
        for state in RETURN_STATES:
            for goal in GOALS:
                for a in (0,1):
                    fixed[a].append(actual_regret(ep,state,goal,a))
                outcome=[intercept0+slope0*state,intercept1+slope1*state]
                choice=min((0,1),key=lambda q:((outcome[q]-goal)**2,q))
                explicit.append(actual_regret(ep,state,goal,choice))
    best=min(float(np.mean(v)) for v in fixed.values())
    learned=float(np.mean(explicit))
    if best<=0. or learned>=1e-8:
        raise ValueError("F8R26_SYSTEM_ID_CONTROL_INVALID")
    return {"fixed0":float(np.mean(fixed[0])),"fixed1":float(np.mean(fixed[1])),
            "best_fixed":best,"observed_four_probe_linear_memory":learned}


def _state_model(arm,seed,model):
    clone=r22.model_for(arm,seed).cpu()
    clone.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    return clone


def _pretrain(arm,seed,fit,sample):
    model=r22.model_for(arm,seed).cpu()
    original=r22.r18.r5.f5.hash_weights(model)
    optimizer=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm.startswith("native") else .001,weight_decay=.01)
    start=time.perf_counter()
    for ids in sample:
        data,valid=r22.r18.r5.f4._padded([r25.r22.factual(fit[i]) for i in ids])
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=r25.objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R26_PRETRAIN_NONFINITE_LOSS")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    after=r22.r18.r5.f5.hash_weights(model)
    if after==original:
        raise ValueError("F8R26_NO_A_PRETRAINING")
    return model,after,time.perf_counter()-start


def adapt(arm,model,condition,b_fit,a_fit,b_batches,a_batches):
    if condition not in CONDITIONS:
        raise ValueError("F8R26_UNKNOWN_ADAPTATION_CONDITION")
    before=r22.r18.r5.f5.hash_weights(model)
    if condition=="no_update":
        return {"pretrained_hash":before,"adapted_hash":before,"updates":0,
                "negative_gradient_conflicts":0,"projected_steps":0,
                "first4_B_loss":None,"last4_B_loss":None,"gradient_forward_seconds":0.}
    params=list(model.parameters())
    optimizer=torch.optim.AdamW(params,lr=B_LR,weight_decay=.01)
    conflicts,projections,losses=0,0,[]
    start=time.perf_counter()
    for bidx,aidx in zip(b_batches,a_batches):
        b_data,b_valid=r22.r18.r5.f4._padded([factual(b_fit[i]) for i in bidx])
        a_data,a_valid=r22.r18.r5.f4._padded([r25.r22.factual(a_fit[i]) for i in aidx])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        B_loss=r24._delta_mse(arm,model,b_data,b_valid)
        A_loss=r25.objective(arm,model,a_data,a_valid)
        if not bool(torch.isfinite(B_loss)) or not bool(torch.isfinite(A_loss)):
            raise ValueError("F8R26_NONFINITE_FACTUAL_GRADIENT_LOSS")
        B_grads=torch.autograd.grad(B_loss,params,allow_unused=True)
        A_grads=torch.autograd.grad(A_loss,params,allow_unused=True)
        b=[torch.zeros_like(p) if g is None else g.detach().clone()
           for p,g in zip(params,B_grads)]
        a=[torch.zeros_like(p) if g is None else g.detach().clone()
           for p,g in zip(params,A_grads)]
        if any(not bool(torch.isfinite(g).all()) for g in (*a,*b)):
            raise ValueError("F8R26_NONFINITE_FACTUAL_GRADIENT")
        dot=sum(torch.sum(g*h) for g,h in zip(b,a))
        denom=sum(torch.sum(g*g) for g in a)
        if not bool(denom>0):
            raise ValueError("F8R26_NO_A_FACTUAL_PROTECTION_GRADIENT")
        if bool(dot<0):
            conflicts+=1
            if condition=="orthogonal_B_update":
                fac=dot/denom
                b=[g-fac*h for g,h in zip(b,a)]
                projections+=1
                if float(sum(torch.sum(g*h) for g,h in zip(b,a))) < -1e-5:
                    raise ValueError("F8R26_PROJECTION_NOT_ORTHOGONAL")
        for p,g in zip(params,b):
            p.grad=g
        torch.nn.utils.clip_grad_norm_(params,1.)
        optimizer.step()
        losses.append(float(B_loss.detach()))
    after=r22.r18.r5.f5.hash_weights(model)
    if after==before or len(losses)!=ADAPT:
        raise ValueError("F8R26_ADAPTATION_NOT_EFFECTIVE")
    return {"pretrained_hash":before,"adapted_hash":after,"updates":ADAPT,
            "negative_gradient_conflicts":conflicts,"projected_steps":projections,
            "first4_B_loss":float(np.mean(losses[:4])),
            "last4_B_loss":float(np.mean(losses[-4:])),
            "gradient_forward_seconds":time.perf_counter()-start}


def fit_one(arm,seed,a_fit,b_fit,b_hold,a_score_eps,pre_batches,b_batches,a_batches):
    pretrained,pre_hash,elapsed=_pretrain(arm,seed,a_fit,pre_batches)
    states={}
    for condition in CONDITIONS:
        model=_state_model(arm,seed,pretrained)
        if r22.r18.r5.f5.hash_weights(model)!=pre_hash:
            raise ValueError("F8R26_DIFFERENT_CONTROL_START")
        outcome=adapt(arm,model,condition,b_fit,a_fit,b_batches,a_batches)
        A=a_score(arm,model,a_score_eps)
        B=r24.b_mse(arm,model,b_hold)
        copy_model=_state_model(arm,seed,model).eval()
        replay=a_score(arm,copy_model,a_score_eps)
        if (r22.r18.r5.f5.hash_weights(copy_model)!=outcome["adapted_hash"]
            or replay["actions"]!=A["actions"]
            or replay["paired_group_regret"]!=A["paired_group_regret"]):
            raise ValueError("F8R26_COLD_RESTORE_ACTION_DRIFT")
        states[condition]={**outcome,"A_first_return":A,"B_holdout_MSE":B,
                           "cold_replay_action_parity":True}
    return {"arm":arm,"seed":seed,
        "parameters":sum(p.numel() for p in pretrained.parameters()),
        "pretrain_updates":PRETRAIN,"pretrained_hash":pre_hash,
        "pretrain_wall_seconds":elapsed,"conditions":states}


def decide(models,ctrl):
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,16,size=(BOOT,16))
    result=[]
    for model in models:
        if model["arm"]!="native_focus20":
            continue
        states=model["conditions"]
        orig=states["no_update"]
        plain=states["plain_B_update"]
        guarded=states["orthogonal_B_update"]
        A0=orig["A_first_return"]["mean_regret"]
        Ap=plain["A_first_return"]["mean_regret"]
        Ag=guarded["A_first_return"]["mean_regret"]
        Bi=orig["B_holdout_MSE"];Bg=guarded["B_holdout_MSE"]
        delta=(np.asarray(plain["A_first_return"]["paired_group_regret"])
               -np.asarray(guarded["A_first_return"]["paired_group_regret"]))
        lower=float(np.percentile(delta[draws].mean(axis=1),2.5))
        passed=bool(Bg<=.95*Bi and Ag<=.75*ctrl["best_fixed"]
                    and Ag<=A0+.01 and Ag<=Ap-.01 and lower>0.
                    and guarded["projected_steps"]>0
                    and guarded["cold_replay_action_parity"])
        result.append({"seed":model["seed"],"A_before":A0,
            "A_plain_after":Ap,"A_projected_after":Ag,
            "B_MSE_before":Bi,"B_MSE_projected_after":Bg,
            "bootstrap_plain_minus_projection_lower2p5":lower,
            "B_update_gradient_conflicts":guarded["negative_gradient_conflicts"],
            "actual_projections":guarded["projected_steps"],
            "frozen_seed_win":passed})
    return {"native_seed_outcomes":result,
            "frozen_development_pass":sum(int(z["frozen_seed_win"]) for z in result)>=2,
            "independent_scientific_pass":False}


def run():
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    fit=r25.cohort("fit")
    b_fit,b_hold,score=(cohort(k) for k in SPLITS)
    identity=audit(b_fit,b_hold,score)
    ctrl=controls(score)
    pre=torch.Generator(device="cpu");pre.manual_seed(985299)
    B=torch.Generator(device="cpu");B.manual_seed(20119999)
    A=torch.Generator(device="cpu");A.manual_seed(20118888)
    pre_batches=torch.randint(0,len(fit),(PRETRAIN,BATCH),generator=pre).tolist()
    b_batches=torch.randint(0,len(b_fit),(ADAPT,BATCH),generator=B).tolist()
    a_batches=torch.randint(0,len(fit),(ADAPT,BATCH),generator=A).tolist()
    models=[fit_one(arm,seed,fit,b_fit,b_hold,score,pre_batches,b_batches,a_batches)
            for arm,seed in MODEL_SPECS]
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "classification":"AUTHOR_GENERATED_SOURCE_NEUTRAL_DEVELOPMENT_ONLY",
            "source_audit":identity,"controls":ctrl,
            "model_count":len(models),
            "actual_A_pretraining_optimizer_updates":len(models)*PRETRAIN,
            "actual_B_adaptation_optimizer_updates":len(models)*2*ADAPT,
            "models":models,"decision":decide(models,ctrl),
            "B_training_data_source_neutral":True,
            "original_external_effects":0,"F7_original_effect_replay":False,
            "conventional_independently_convergence_qualified":False,
            "independent_scientific_pass":False,
            "deployment":"NONE",
            "total_wall_seconds":time.perf_counter()-start}


def retain_ci_receipt(result):
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    dest=ROOT/"local_runs/p85mr_f8r26_gradient_conflict_dev.json"
    dest.parent.mkdir(exist_ok=True)
    if dest.exists():
        raise FileExistsError("F8R26_REFUSE_EVIDENCE_OVERWRITE")
    dest.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_original_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R26_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
