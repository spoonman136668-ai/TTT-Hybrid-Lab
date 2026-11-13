"""R25 independently varying action-specific dynamics; source-neutral development.

This extends R23's binary-sign toy task to four independently varying,
real-valued intercepts/slopes, with four *factual* early probes. Models see no
hidden parameters, goal labels are only scored after the 8/16-step B phase,
and return effects are never revealed before selection.
No original simulator, spent F7, merge, owner dispatch or science pass.
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
from experiments import p85mr_f8r24_online_interference as prior
from experiments import p85mr_f8r23_novel_causal as r23

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r25_multimechanism_freeze.json"
FREEZE_BLOB="3821a5599f544e61a27ce3a3804142d4a030e6f5"
SCHEMA="ttt-hybrid.f8r25-multimechanism-factual-probes.v1"
ARMS=("native_focus20","gru_focus20","ssm_focus20")
SEEDS=(985201,985203,985207)
B_VALUES=(-.3,-.1,.1,.3)
K_VALUES=(-.25,.25)
PROBES=((-0.2,0),(0.2,0),(-0.2,1),(0.2,1))
GOALS=(-.2,.2)
RETURNS=(-.2,.2)
B_STEPS={"dev":8,"id":8,"ood":16}
GROUPS=16
FIT_EPISODES=256
UPDATES=384
BATCH=8
BOOT=10000
BOOT_SEED=20049999
SPLIT_SEEDS={"fit":20010000,"dev":20020000,
             "id":20030000,"ood":20040000}


def git_blob(raw:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked()->dict:
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R25_FROZEN_PROTOCOL_BLOB_DRIFT")
    f=json.loads(raw)
    w,t,s,a=f["world"],f["train"],f["score"],f["authority"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R25_LINKED_IMPLEMENTATION_OR_OUTCOMES"
        or f["classification"]!="SOURCE_NEUTRAL_DEV_ONLY_NOT_EXTERNAL_SCIENCE"
        or f["parent"]["head"]!="832f090bece9e4c42c2cac710294c7eed08c5af7"
        or f["parent"]["r23_source_blob"]!="5ca960ea653aa2200e5e1ec76f04844c3ac77b86"
        or f["parent"]["r22_source_blob"]!="57e5977555f3c0afee7a987a4aee80be994cc082"
        or f["parent"]["r24_online_source_blob"]!="3fb48eab94ba333f4433bfc163bfabc5e1db9477"
        or f["parent"]["r24_online_freeze_blob"]!="fd51e60133969db2d778cd1db5272ae4e1cb483b"
        or git_blob((ROOT/"experiments/p85mr_f8r24_online_interference.py").read_bytes())!="3fb48eab94ba333f4433bfc163bfabc5e1db9477"
        or w["A_intercepts_FIT"]!=list(B_VALUES)
        or w["A_slopes_FIT"]!=list(K_VALUES)
        or w["early_factual_probe_sequence"]!=[{"s":s,"a":a} for s,a in PROBES]
        or w["FIT_sessions"]!=FIT_EPISODES
        or w["FIT_factual_effects"]!=3328
        or w["FIT_B_steps"]!=8
        or w["FIT_recorded_steps"]!=13
        or w["B_signs"]!=[-1,1]
        or w["score"]["B_steps"]!=B_STEPS
        or w["score"]["group_count_each"]!=GROUPS
        or w["score"]["distinct_seeds"]!=SPLIT_SEEDS
        or w["score"]["return_public_states"]!=list(RETURNS)
        or w["score"]["return_goals"]!=list(GOALS)
        or w["score"]["score_worlds_each"]!=32
        or w["score"]["score_actions_per_model_each"]!=128
        or w["score"]["score_model_queries_per_model_each"]!=256
        or t["arms"]!=list(ARMS) or t["seeds"]!=list(SEEDS)
        or t["models"]!=9 or t["updates_each"]!=UPDATES
        or t["total_updates"]!=3456 or t["sampler_seed"]!=985299
        or t["batch_size"]!=BATCH
        or s["paired_bootstrap_resamples"]!=BOOT
        or s["paired_bootstrap_seed"]!=BOOT_SEED
        or s["seed_threshold_pass_at_least"]!=2
        or s["independent_scientific_pass"] is True
        or any(v is not False for v in a.values())):
        raise ValueError("F8R25_PRECOMMITTED_PROTOCOL_OR_AUTHORITY_DRIFT")
    prior.locked()
    return f


def A_after(s:float,action:int,b:tuple[float,float],k:tuple[float,float])->float:
    if (type(action) is not int or action not in (0,1)
        or not math.isfinite(s) or len(b)!=2 or len(k)!=2
        or not all(math.isfinite(z) for z in (*b,*k))):
        raise ValueError("F8R25_INVALID_ACTION_SPECIFIC_PHYSICS")
    return .4*s+b[action]+k[action]*s


def B_after(s:float,action:int,sign:int)->float:
    if type(action) is not int or action not in (0,1) or sign not in (-1,1):
        raise ValueError("F8R25_INVALID_INDEPENDENT_B")
    return .35*math.tanh(s)+.12*sign*(2*action-1)


def fit_episode(b:tuple[float,float],k:tuple[float,float],
                sign_B:int,actual_return_action:int,fit_group:int)->dict:
    if (any(x not in B_VALUES for x in b)
        or any(x not in K_VALUES for x in k)
        or sign_B not in (-1,1)
        or type(actual_return_action) is not int or actual_return_action not in (0,1)
        or type(fit_group) is not int or not 0<=fit_group<128):
        raise ValueError("F8R25_UNREGISTERED_FACTUAL_FIT")
    return _generate("fit",fit_group,b,k,sign_B,actual_return_action)


def _generate(split:str,group:int,b:tuple[float,float],k:tuple[float,float],
              sign_B:int,actual_return_action:int|None=None,twin:int=0)->dict:
    if (split not in SPLIT_SEEDS or sign_B not in (-1,1)
        or type(group) is not int or group<0
        or split!="fit" and group>=GROUPS
        or split=="fit" and group>=128
        or split!="fit" and actual_return_action is not None
        or split=="fit" and actual_return_action not in (0,1)):
        raise ValueError("F8R25_UNREGISTERED_WORLD")
    actions,effects=[],[]
    def record(before,action,after,final=False):
        actions.append(r22.r18.r5._action_key((before,0.,0.,0.),action))
        effects.append(r22.r18._observed(before,after,final=final))
    for state,action in PROBES:
        record(state,action,A_after(state,action,b,k))
    state=0.
    n=8 if split=="fit" else B_STEPS[split]
    for step in range(n):
        sign=sign_B if step<8 else -sign_B
        if step==8 and split=="ood":
            state=0.
        action=r22.r20.r3.behavior(state)^int(
            (SPLIT_SEEDS[split]+group+7*step+int(sign==1))%3==0)
        after=B_after(state,action,sign)
        record(state,action,after)
        state=after
    if split=="fit":
        record(0.,actual_return_action,
               A_after(0.,actual_return_action,b,k),True)
    return {
      "action":torch.tensor(actions,dtype=torch.float32),
      "effect":torch.tensor(effects,dtype=torch.float32),
      "split":split,"group":group,"twin":twin,
      "b_evaluator_only":b,"k_evaluator_only":k,
      "z_B_evaluator_only":sign_B,"selected_return":actual_return_action
    }


def cohort(split:str)->list[dict]:
    locked()
    if split=="fit":
        eps=[]
        group=0
        for b0 in B_VALUES:
            for b1 in B_VALUES:
                for k0 in K_VALUES:
                    for k1 in K_VALUES:
                        for sign in (-1,1):
                            for selected in (0,1):
                                eps.append(fit_episode((b0,b1),(k0,k1),
                                                       sign,selected,group))
                            group+=1
        if group!=128 or len(eps)!=FIT_EPISODES:
            raise ValueError("F8R25_FIT_NOT_FULL_FACTORIAL")
        return eps
    if split not in ("dev","id","ood"):
        raise ValueError("F8R25_UNKNOWN_SPLIT")
    b_vals=B_VALUES if split!="ood" else (-.25,-.05,.05,.25)
    k_vals=K_VALUES if split!="ood" else (-.2,.2)
    eps=[]
    for g in range(GROUPS):
        q=g%4
        b0,b1=b_vals[q],b_vals[(q+1+(g//4)%2)%4]
        k0,k1=k_vals[(g//4)%2],k_vals[(g//8)%2]
        sign=1 if g%2==0 else -1
        eps.append(_generate(split,g,(b0,b1),(k0,k1),sign,None,0))
        eps.append(_generate(split,g,(b1,b0),(k1,k0),sign,None,1))
    return eps


def audit(eps:list[dict],split:str)->dict:
    if split=="fit":
        if len(eps)!=FIT_EPISODES:
            raise ValueError("F8R25_FIT_COUNT_DRIFT")
        cells={}
        for e in eps:
            if e["action"].shape!=(13,16):
                raise ValueError("F8R25_FIT_SHAPE_DRIFT")
            key=(*e["b_evaluator_only"],*e["k_evaluator_only"],
                 e["z_B_evaluator_only"],e["selected_return"])
            cells[key]=cells.get(key,0)+1
        if len(cells)!=FIT_EPISODES or set(cells.values())!={1}:
            raise ValueError("F8R25_MISSING_FACTUAL_ACTION_MECHANISM")
        return {"factorial_factual_cells":256,"selected_effects":3328}
    if split not in ("dev","id","ood") or len(eps)!=32:
        raise ValueError("F8R25_SCORE_PANEL_DRIFT")
    expected_length=4+B_STEPS[split]
    for i in range(0,len(eps),2):
        original,swap=eps[i:i+2]
        if (original["group"]!=swap["group"]
            or original["twin"]!=0 or swap["twin"]!=1
            or original["b_evaluator_only"]!=swap["b_evaluator_only"][::-1]
            or original["k_evaluator_only"]!=swap["k_evaluator_only"][::-1]
            or not torch.equal(original["action"],swap["action"])
            or not torch.equal(original["effect"][4:],swap["effect"][4:])
            or torch.equal(original["effect"][:4],swap["effect"][:4])
            or original["action"].shape!=(expected_length,16)):
            raise ValueError("F8R25_SWAPPED_MECHANISM_PAIR_OR_B_DRIFT")
    return {"matched_twin_groups":16,"identical_B_pair":True,
            "measured_four_early_factual_probes":True,
            "score_sessions":32,"steps_before_return":expected_length}


def actual_regret(ep:dict,s:float,goal:float,selected:int)->float:
    if s not in RETURNS or goal not in GOALS or selected not in (0,1):
        raise ValueError("F8R25_UNREGISTERED_SCORE_QUERY")
    outcomes=[A_after(s,q,ep["b_evaluator_only"],
                      ep["k_evaluator_only"]) for q in (0,1)]
    errors=[(z-goal)**2 for z in outcomes]
    return max(0.,errors[selected]-min(errors))


def analytic_controls(eps:list[dict],split:str)->dict:
    audit(eps,split)
    raw={k:[] for k in ("fixed0","fixed1","full_information_oracle")}
    for g in range(GROUPS):
        worlds=eps[2*g:2*g+2]
        for arm in raw:
            vals=[]
            for ep in worlds:
                for s in RETURNS:
                    for goal in GOALS:
                        q=0 if arm=="fixed0" else 1
                        if arm=="full_information_oracle":
                            q=min((0,1),key=lambda a:(
                                (A_after(s,a,ep["b_evaluator_only"],
                                         ep["k_evaluator_only"])-goal)**2,a))
                        vals.append(actual_regret(ep,s,goal,q))
            raw[arm].append(float(np.mean(vals)))
    mean={k:float(np.mean(v)) for k,v in raw.items()}
    if not mean["fixed0"]>0 or not mean["fixed1"]>0 or mean["full_information_oracle"]!=0.:
        raise ValueError("F8R25_NO_CAUSAL_INFORMATION_GAP")
    reference=min(("fixed0","fixed1"),key=lambda arm:(mean[arm],arm))
    return {"mean":mean,"pergroup":raw,"best_no_early_A":reference}


def four_probe_predictions(ep:dict)->tuple[float,float,float,float]:
    # Exact FIT-like linear identification using only four actually observed
    # A effect rows. Do NOT feed evaluator-only b/k, unselected outcomes or
    # the later score decision to this stronger explicit memory baseline.
    pairs=[]
    for i in (0,2):
        lo_s,hi_s=PROBES[i][0],PROBES[i+1][0]
        lo=lo_s+r22.r19._physical_delta(ep["effect"][i])
        hi=hi_s+r22.r19._physical_delta(ep["effect"][i+1])
        slope=(hi-lo)/(hi_s-lo_s)
        intercept=(hi+lo)/2.
        pairs.append((float(intercept),float(slope)))
    return pairs[0][0],pairs[1][0],pairs[0][1],pairs[1][1]


def explicit_memory_score(eps:list[dict],split:str)->dict:
    audit(eps,split)
    losses=[]
    for ep in eps:
        b0,b1,t0,t1=four_probe_predictions(ep)
        for state in RETURNS:
            for goal in GOALS:
                pred=(b0+t0*state,b1+t1*state)
                q=min((0,1),key=lambda a:((pred[a]-goal)**2,a))
                losses.append(actual_regret(ep,state,goal,q))
    return {"regret":float(np.mean(losses)),
            "inputs":"four_original_factual_probes_only",
            "evaluator_hidden_latents_used":False,
            "parameter_count":4}


def predicted(arm:str,model,eps:list[dict],erase_A=False)->np.ndarray:
    acts,effs=[],[]
    for ep in eps:
        for state in RETURNS:
            for action in (0,1):
                key=torch.tensor(
                    r22.r18.r5._action_key((state,0.,0.,0.),action),
                    dtype=torch.float32).unsqueeze(0)
                aa=torch.cat((ep["action"],key),0)
                ee=torch.cat((ep["effect"],torch.zeros((1,16))),0)
                if erase_A:
                    ee[:4]=0.
                acts.append(aa);effs.append(ee)
    model.eval()
    with torch.no_grad():
        out=r22.predict(arm,model,{"action":torch.stack(acts),
                                   "effect":torch.stack(effs)})
    expected=(len(eps)*len(RETURNS)*2,5+B_STEPS[eps[0]["split"]],16)
    if out.shape!=expected or not bool(torch.isfinite(out).all()):
        raise ValueError("F8R25_UNOBSERVED_CURRENT_EFFECT_OR_SCORE_SHAPE")
    physical=np.asarray([2.4*math.atanh(max(-.98,min(.98,float(z))))
                        for z in out[:,-1,0].detach().cpu().numpy()],
                        dtype=np.float64)
    return physical.reshape(len(eps),len(RETURNS),2)


def model_score(arm:str,model,eps:list[dict],split:str)->dict:
    audit(eps,split)
    start=time.perf_counter()
    normal=predicted(arm,model,eps)
    erased=predicted(arm,model,eps,True)
    choices,regrets,masked=[],[],[]
    for i,ep in enumerate(eps):
        for j,s in enumerate(RETURNS):
            for goal in GOALS:
                q=min((0,1),key=lambda a:(
                    (s+normal[i,j,a]-goal)**2,a))
                m=min((0,1),key=lambda a:(
                    (s+erased[i,j,a]-goal)**2,a))
                choices.append(q)
                regrets.append(actual_regret(ep,s,goal,q))
                masked.append(actual_regret(ep,s,goal,m))
    twin_contrasts=sum(
        choices[8*g+j]!=choices[8*g+4+j]
        for g in range(GROUPS) for j in range(4))
    pergroup=[float(np.mean(regrets[8*g:8*g+8])) for g in range(GROUPS)]
    return {"mean_regret":float(np.mean(regrets)),
            "mean_erased_four_A_regret":float(np.mean(masked)),
            "paired_group_regret":pergroup,
            "twin_mechanism_action_contrasts":int(twin_contrasts),
            "contrast_opportunities":GROUPS*4,
            "actual_selected_actions":len(choices),
            "hypothetical_model_queries":2*len(eps)*len(RETURNS)*2,
            "actions":choices,"score_seconds":time.perf_counter()-start}


def objective(arm:str,model,data:dict,valid:torch.Tensor)->torch.Tensor:
    prediction=r22.predict(arm,model,data)
    if (prediction.shape!=data["effect"].shape
        or prediction.shape[1]!=13 or valid.shape!=(BATCH,13)
        or not bool(valid[:,-1].all())):
        raise ValueError("F8R25_SELECTED_FACTUAL_RETURN_SHAPE_DRIFT")
    selected=(prediction[:,-1,:4]-data["effect"][:,-1,:4]).square().mean()
    if arm=="gru_focus20":
        physical=r22.r18.r5.f4._physical_loss(model,data,valid)[0]
    else:
        physical=r22.r18._objective(
            "bypass_surrogate" if arm=="native_focus20" else "ssm256",
            model,data,valid)
    return physical+20.*selected


def fit_one(arm:str,seed:int,fit:list[dict],dev:list[dict],
            panels:dict[str,list[dict]],sampler:list[list[int]])->dict:
    model=r22.model_for(arm,seed).cpu()
    before=r22.r18.r5.f5.hash_weights(model)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002 if arm=="native_focus20" else .001,
                               weight_decay=.01)
    start=time.perf_counter()
    last_loss=None
    curve=[]
    for u,ids in enumerate(sampler,1):
        data,valid=r22.r18.r5.f4._padded([r22.factual(fit[i]) for i in ids])
        model.train();optimizer.zero_grad(set_to_none=True)
        loss=objective(arm,model,data,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R25_NONFINITE_PHYSICAL_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(g is not None and bool((g.abs()>0).any()) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads)):
            raise ValueError("F8R25_MISSING_OR_BAD_TRAINING_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        last_loss=float(loss.detach())
        if u in (128,256,384):
            d=model_score(arm,model,dev,"dev")
            curve.append({"updates":u,"dev_action_regret":d["mean_regret"]})
    fit_seconds=time.perf_counter()-start
    digest=r22.r18.r5.f5.hash_weights(model)
    if digest==before:
        raise ValueError("F8R25_UNTRAINED_MODEL")
    evaluations={name:model_score(arm,model,ep,name)
                 for name,ep in panels.items()}
    cold=r22.model_for(arm,seed).cpu()
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r22.r18.r5.f5.hash_weights(cold)!=digest
        or any(model_score(arm,cold,eps,name)["actions"]!=evaluations[name]["actions"]
               for name,eps in panels.items())):
        raise ValueError("F8R25_COLD_WEIGHT_CHECKPOINT_DRIFT")
    return {"arm":arm,"seed":seed,"parameters":sum(p.numel() for p in model.parameters()),
            "steps":UPDATES,"training_seconds":fit_seconds,
            "last_train_loss":last_loss,"dev_curve":curve,
            "initial_weights":before,"final_weights":digest,
            "cold_checkpoint_actions_identical":True,
            "panels":evaluations}


def decision(rows:list[dict],controls:dict)->dict:
    conventional=[r for r in rows if r["arm"]!="native_focus20"]
    selected=min(conventional,key=lambda r:(
        r["dev_curve"][-1]["dev_action_regret"],r["arm"],r["seed"]))
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,GROUPS,size=(BOOT,GROUPS))
    candidate=[]
    for r in rows:
        if r["arm"]!="native_focus20":
            continue
        halves={}
        both=True
        for split in ("id","ood"):
            a=r["panels"][split]
            c=controls[split]["analytic"]
            fixed=np.asarray(c["pergroup"][c["best_no_early_A"]])
            gain=fixed-np.asarray(a["paired_group_regret"])
            lower=float(np.percentile(gain[draws].mean(axis=1),2.5))
            cost=a["mean_regret"]
            base=float(np.mean(fixed))
            erased=a["mean_erased_four_A_regret"]-cost
            ok=bool(cost<=.03 and cost<=.8*base and lower>0.
                and erased>=.015 and a["twin_mechanism_action_contrasts"]>=12)
            halves[split]={
              "regret":cost,"fixed_reference_regret":base,
              "DEV_selected_recurrent_regret":selected["panels"][split]["mean_regret"],
              "explicit_factual_four_probe_regret":controls[split]["explicit_four_probe"]["regret"],
              "paired_bootstrap_2p5_gain":lower,
              "erase_A_regret_increase":erased,
              "twin_action_contrasts":a["twin_mechanism_action_contrasts"],
              "passes_prespecified_DEV":ok}
            both=both and ok
        candidate.append({"seed":r["seed"],"both_horizons_DEV_win":both,"panels":halves})
    return {"DEV_selected_conventional":{"arm":selected["arm"],"seed":selected["seed"],
                 "DEV_regret":selected["dev_curve"][-1]["dev_action_regret"]},
            "candidate_seeds":candidate,
            "frozen_development_pass":sum(x["both_horizons_DEV_win"] for x in candidate)>=2,
            "independent_scientific_pass":False}


def run()->dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=cohort("fit")
    dev=cohort("dev")
    panel={name:cohort(name) for name in ("id","ood")}
    audits={name:audit(ep,name) for name,ep in
            (("fit",fit),("dev",dev),*panel.items())}
    controls={name:{"analytic":analytic_controls(ep,name),
                    "explicit_four_probe":explicit_memory_score(ep,name)}
              for name,ep in panel.items()}
    gen=torch.Generator(device="cpu");gen.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=gen).tolist()
    start=time.perf_counter()
    models=[fit_one(arm,seed,fit,dev,panel,batches)
            for arm in ARMS for seed in SEEDS]
    d=decision(models,controls)
    return {"schema":SCHEMA,"classification":"SOURCE_NEUTRAL_DEV_ONLY_NOT_EXTERNAL_SCIENCE",
            "freeze_blob":FREEZE_BLOB,"original_R24_online_parent_sha":
                "832f090bece9e4c42c2cac710294c7eed08c5af7",
            "factual_sessions":len(fit),"factual_selected_effects":
                sum(len(ep["effect"]) for ep in fit),
            "model_count":len(models),"trained_optimizer_updates":len(models)*UPDATES,
            "source_audits":audits,"controls":controls,
            "models":models,"decision":d,
            "conventional_convergence_independently_qualified":False,
            "param_compute_matched":False,
            "external_original_source_effects":0,"F7_replay":False,
            "independent_scientific_pass":False,"deployment":"NONE",
            "training_and_scoring_seconds":time.perf_counter()-start}


def retain_ci_receipt(result:dict)->None:
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r25_multimechanism_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R25_RECEIPT_OVERWRITE_DENIED")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")


def refuse_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R25_ORIGINAL_FIRSTUSE_NOT_AUTHORIZED")
