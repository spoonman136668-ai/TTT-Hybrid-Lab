"""F8-R28: actual factual nonlinear intervention training and cold transfer.

R27 established an information-value ceiling but native did NOT exploit the
fifth causal intervention. Here the model receives new author-generated
q-conditional factual training, then acts without training on entirely new
DEV/ID/OOD groups. Every learner sees only selected factual consequences.
Original external simulator first use and spent F7 data are prohibited.
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
from experiments import p85mr_f8r27_active_interventions as r27

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r28_factual_curvature_freeze.json"
FREEZE_BLOB="0ab949d433a4405435aa72612a17f28859ce2315"
SCHEMA="ttt-hybrid.f8r28-factual-curvature-intervention-transfer.v1"
ARMS=("native_focus20","gru_focus20","ssm_focus20")
NATIVE_SEEDS=(985201,985203,985207)
CONVENTIONAL_SEEDS=(985201,)
PANELS=("dev","id","ood")
BASES={"dev":20320000,"id":20330000,"ood":20340000}
B_STEPS={"dev":8,"id":8,"ood":16}
Q_TRAIN=(-.55,.55)
GROUPS=12
PROBES=((-0.2,0),(.2,0),(-.2,1),(.2,1))
FIFTH=.7
STATES=(-.7,.7)
GOALS=(-.2,.2)
BATCH=8
UPDATES=384
BOOTS=10000
BOOT_SEED=20349999
POLICIES=("active_information","fixed_action0","fixed_action1","seeded_random")


def _blob(raw:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked()->dict:
    data=FREEZE.read_bytes()
    if _blob(data)!=FREEZE_BLOB:
        raise ValueError("F8R28_FROZEN_PROTOCOL_DRIFT")
    f=json.loads(data);w=f["world"];t=f["training"];s=f["score"];a=f["authority"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_PRE_IMPLEMENTATION_AND_NUMERIC_OUTCOME"
        or f["classification"]!="SOURCE_NEUTRAL_TRAINED_ACTIVE_CAUSAL_TRANSFER_DEVELOPMENT_ONLY"
        or f["parent"]["head"]!="a4913f86921d6758770098fcf7a688bf48cb6806"
        or f["parent"]["r27_source_blob"]!="270f61969f8a1c7c6174cd3492d6169cd2f3a514"
        or f["parent"]["r27_freeze_blob"]!="237aa9f50aaaab9232caaa9a93e3ab67a755becf"
        or f["parent"]["r25_source_blob"]!="ef488956741b339eb68c786432c48fefb6526165"
        or _blob((ROOT/"experiments/p85mr_f8r27_active_interventions.py").read_bytes())!="270f61969f8a1c7c6174cd3492d6169cd2f3a514"
        or _blob((ROOT/"experiments/p85mr_f8r25_multimechanism.py").read_bytes())!="ef488956741b339eb68c786432c48fefb6526165"
        or w["FIT_sessions"]!=4096 or w["FIT_per_base_expansions"]!=16
        or w["FIT_expected_effects"]!=57344
        or w["seeds"]!=BASES or w["B_steps"]!=B_STEPS
        or w["panels"]!=list(PANELS)
        or w["goals"]!=list(GOALS) or w["target_states"]!=list(STATES)
        or w["four_early_factual_probes"]!=[list(z) for z in PROBES]
        or w["fifth_factual_probe"]["state"]!=FIFTH
        or w["q_train"]!=list(Q_TRAIN)
        or w["policies"]!=list(POLICIES)
        or t["arms"]!=list(ARMS)
        or t["seeds_native"]!=list(NATIVE_SEEDS)
        or t["seeds_conventional"]!=list(CONVENTIONAL_SEEDS)
        or t["actual_fit_models"]!=5
        or t["total_optimizer_updates"]!=1920
        or t["updates_each"]!=UPDATES or t["batch"]!=BATCH
        or t["sampler_seed"]!=20319999
        or s["paired_bootstrap_draws"]!=BOOTS
        or s["paired_bootstrap_seed"]!=BOOT_SEED
        or s["native_min_seed_wins_both_ID_OOD"]!=2
        or s["independent_scientific_pass"] is not False
        or any(v is not False for v in a.values())):
        raise ValueError("F8R28_FROZEN_PARENT_OR_AUTHORITY_DRIFT")
    r27.locked()
    return f


def _record(s:float,action:int,effect:float)->tuple[list,list]:
    return r27._record(s,action,effect)


def fit_cohort()->list[dict]:
    locked()
    originals=r25.cohort("fit")
    if len(originals)!=256:
        raise ValueError("F8R28_BASE_FIT_DRIFT")
    rows=[]
    for base in originals:
        b=base["b_evaluator_only"]
        k=base["k_evaluator_only"]
        first=[_record(state,act,
                r27.after_A(state,act,b,k,(0.,0.)))
               for state,act in PROBES]
        B_actions=base["action"][4:12].tolist()
        B_effects=base["effect"][4:12].tolist()
        for q0 in Q_TRAIN:
            for q1 in Q_TRAIN:
                q=(q0,q1)
                for fifth_action in (0,1):
                    fifth=_record(FIFTH,fifth_action,
                        r27.after_A(FIFTH,fifth_action,b,k,q))
                    for state in STATES:
                        act=base["selected_return"]
                        final=_record(state,act,
                            r27.after_A(state,act,b,k,q))
                        first_and_fifth=first+[fifth]
                        aa=[x[0] for x in first_and_fifth]+B_actions+[final[0]]
                        ee=[x[1] for x in first_and_fifth]+B_effects+[final[1]]
                        if len(aa)!=14 or len(ee)!=14:
                            raise ValueError("F8R28_FIT_SEQUENCE_DRIFT")
                        rows.append({
                            "action":torch.tensor(aa,dtype=torch.float32),
                            "effect":torch.tensor(ee,dtype=torch.float32),
                            "b_evaluator_only":b,"k_evaluator_only":k,
                            "q_evaluator_only":q,
                            "selected_fifth":fifth_action,
                            "selected_return":act,
                            "return_state":state
                        })
    if len(rows)!=4096:
        raise ValueError("F8R28_FIT_NOT_FULL_FACTORIZED")
    return rows


def fit_audit(rows:list[dict])->dict:
    if len(rows)!=4096:
        raise ValueError("F8R28_FIT_ROW_COUNT_DRIFT")
    cells=set()
    for e in rows:
        if (e["action"].shape!=(14,16)
            or e["effect"].shape!=(14,16)
            or not bool(torch.isfinite(e["effect"]).all())):
            raise ValueError("F8R28_INVALID_FIT_SEQUENCE")
        key=(*e["b_evaluator_only"],*e["k_evaluator_only"],
             e["z_B_evaluator_only"],*e["q_evaluator_only"],e["selected_fifth"],
             e["return_state"],e["selected_return"])
        cells.add(key)
    if len(cells)!=4096:
        raise ValueError("F8R28_TRAINING_SUPPORT_CONFOUND")
    return {"full_factored_factual_sessions":4096,
            "actual_selected_factual_effects":57344,
            "each_episode_has_only_one_fifth_action":True,
            "each_episode_has_only_one_return_action":True}


def new_world(panel:str,group:int,twin:int)->dict:
    if panel not in PANELS or type(group) is not int or not 0<=group<GROUPS or twin not in (0,1):
        raise ValueError("F8R28_UNREGISTERED_NEW_WORLD")
    rng=np.random.default_rng(BASES[panel]+group)
    b=tuple(float(x) for x in rng.choice(r27.BVAL,size=2))
    k=tuple(float(x) for x in rng.choice(r27.KVAL,size=2))
    q=tuple(float(x) for x in rng.choice(Q_TRAIN if panel!="ood" else (-.45,.45),size=2))
    bs=1 if group%2==0 else -1
    if twin:
        b=b[::-1];k=k[::-1];q=q[::-1]
    early=[_record(state,act,r27.after_A(state,act,b,k,q))
           for state,act in PROBES]
    history=[]
    state=0.
    for step in range(B_STEPS[panel]):
        if step==8 and panel=="ood":
            state=0.
        sign=bs if step<8 else -bs
        act=r25.r22.r20.r3.behavior(state)^int(
            (BASES[panel]+group+7*step+int(sign==1))%3==0)
        nxt=r25.B_after(state,act,sign)
        history.append(_record(state,act,nxt))
        state=nxt
    return {"split":panel,"group":group,"twin":twin,"b":b,"k":k,"q":q,
            "Bsign":bs,"early":early,"B":history}


def worlds(panel:str)->list[dict]:
    locked()
    return [new_world(panel,g,t) for g in range(GROUPS) for t in (0,1)]


def audit_worlds(rows:list[dict],panel:str)->dict:
    if panel not in PANELS or len(rows)!=2*GROUPS:
        raise ValueError("F8R28_PANEL_DRIFT")
    for g in range(GROUPS):
        left,right=rows[2*g:2*g+2]
        if (left["group"]!=right["group"] or left["twin"]!=0 or right["twin"]!=1
            or left["q"]!=right["q"][::-1]
            or left["b"]!=right["b"][::-1] or left["k"]!=right["k"][::-1]
            or left["B"]!=right["B"]
            or len(left["early"])!=4 or len(left["B"])!=B_STEPS[panel]):
            raise ValueError("F8R28_B_LEAK_OR_UNMATCHED_TWINS")
    return {"paired_new_groups":GROUPS,"real_new_worlds":2*GROUPS,
            "B_history_identical_within_swapped_A_twins":True,
            "B_steps_each":B_STEPS[panel],
            "first_return_effect_unobserved":True}


def _choices(arm:str,model,plans:list[dict],erase:bool)->list[int]:
    a,e=[],[]
    for plan in plans:
        for action in (0,1):
            this=_record(plan["state"],action,plan["state"])
            rows=plan["history"]+[this]
            keys=torch.tensor([x for x,_ in rows],dtype=torch.float32)
            observed=torch.tensor([y for _,y in rows],dtype=torch.float32)
            observed[-1]=0.
            if erase:
                observed[4]=0.
            a.append(keys);e.append(observed)
    model.eval()
    with torch.no_grad():
        prediction=r25.r22.predict(arm,model,{
            "action":torch.stack(a),"effect":torch.stack(e)})
    n=2*len(plans)
    expected=(n,len(plans[0]["history"])+1,16)
    if prediction.shape!=expected or not bool(torch.isfinite(prediction).all()):
        raise ValueError("F8R28_INVALID_PREACTION_MODEL_PREDICTION")
    vals=prediction[:,-1,0].detach().cpu().numpy()
    actual=np.asarray([
        plans[i//2]["state"]+2.4*math.atanh(
          max(-.98,min(.98,float(v)))) for i,v in enumerate(vals)
    ],dtype=np.float64).reshape(len(plans),2)
    return [int(min((0,1),key=lambda action:(
        (actual[i,action]-plan["goal"])**2,action)))
        for i,plan in enumerate(plans)]


def score_model(arm:str,model,plans:list[dict])->dict:
    return r27._score_choices(plans,_choices(arm,model,plans,False),
                             _choices(arm,model,plans,True))


def _objective(arm:str,model,data:dict,valid:torch.Tensor)->torch.Tensor:
    pred=r25.r22.predict(arm,model,data)
    if (pred.shape!=data["effect"].shape or pred.shape[1]!=14
        or valid.shape!=(BATCH,14) or not bool(valid[:,-1].all())):
        raise ValueError("F8R28_FIT_SHAPE_OR_PREQUENTIAL_MASK_DRIFT")
    last=(pred[:,-1,:4]-data["effect"][:,-1,:4]).square().mean()
    if arm=="gru_focus20":
        physical=r25.r22.r18.r5.f4._physical_loss(model,data,valid)[0]
    else:
        physical=r25.r22.r18._objective(
          "bypass_surrogate" if arm=="native_focus20" else "ssm256",
          model,data,valid)
    return physical+20.*last


def train(arm:str,seed:int,fit:list[dict],
          plans:dict[str,dict[str,list[dict]]],
          batches:list[list[int]])->dict:
    model=r25.r22.model_for(arm,seed).cpu()
    initial=r25.r22.r18.r5.f5.hash_weights(model)
    opt=torch.optim.AdamW(model.parameters(),
        lr=.002 if arm=="native_focus20" else .001,weight_decay=.01)
    start=time.perf_counter()
    for ix in batches:
        batch,valid=r25.r22.r18.r5.f4._padded([
           {"action":fit[i]["action"],"effect":fit[i]["effect"]} for i in ix])
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=_objective(arm,model,batch,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R28_NONFINITE_PHYSICAL_FIT")
        loss.backward()
        grads=[p.grad for p in model.parameters()]
        if (not any(x is not None and bool(torch.any(x!=0)) for x in grads)
            or any(x is not None and not bool(torch.isfinite(x).all()) for x in grads)):
            raise ValueError("F8R28_INSUFFICIENT_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    elapsed=time.perf_counter()-start
    digest=r25.r22.r18.r5.f5.hash_weights(model)
    if digest==initial:
        raise ValueError("F8R28_UNTRAINED_MODEL")
    outcomes={}
    for panel,policies in plans.items():
        outcomes[panel]={policy:score_model(arm,model,p)
                         for policy,p in policies.items()}
    cold=r25.r22.model_for(arm,seed).cpu()
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r25.r22.r18.r5.f5.hash_weights(cold)!=digest
        or any(_choices(arm,cold,plans[panel]["active_information"],False)
              !=outcomes[panel]["active_information"]["choices"]
              for panel in ("id","ood"))):
        raise ValueError("F8R28_COLD_RESTORE_ACTION_DRIFT")
    return {"arm":arm,"seed":seed,"updates":UPDATES,
            "parameters":sum(p.numel() for p in model.parameters()),
            "initial_hash":initial,"trained_hash":digest,
            "train_wall_seconds":elapsed,"cold_checkpoint_actions_equal":True,
            "panels":outcomes}


def _judge(arms:dict,seed:int|None,kind:str,draws)->dict:
    active=arms["active_information"]
    best=min(("fixed_action0","fixed_action1"),
             key=lambda p:(arms[p]["mean_regret"],p))
    baseline=arms[best]
    diffs=np.asarray(baseline["pergroup_regret"])-np.asarray(
        active["pergroup_regret"])
    lower=float(np.percentile(diffs[draws].mean(axis=1),2.5))
    gain=float(baseline["mean_regret"]-active["mean_regret"])
    erasure=float(active["probe_information_penalty"])
    meets=bool(gain>=.005 and lower>0 and erasure>=.005)
    return {"seed":seed,"kind":kind,
            "active_regret":active["mean_regret"],
            "best_fixed_policy":best,
            "best_fixed_regret":baseline["mean_regret"],
            "seeded_random_regret":arms["seeded_random"]["mean_regret"],
            "fifth_effect_erasure_regret_increase":erasure,
            "paired_bootstrap_2p5_lower_absolute_gain":lower,
            "development_win":meets}


def decision(rows:list[dict],scripted:dict)->dict:
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,GROUPS,size=(BOOTS,GROUPS))
    native=[]
    for row in rows:
        if row["arm"]!="native_focus20":
            continue
        judged={panel:_judge(row["panels"][panel],row["seed"],"native",draws)
                for panel in ("id","ood")}
        native.append({"seed":row["seed"],"panels":judged,
                       "both_panel_win":all(q["development_win"] for q in judged.values())})
    conventional=[x for x in rows if x["arm"]!="native_focus20"]
    selected=min(conventional,
        key=lambda x:(x["panels"]["dev"]["active_information"]["mean_regret"],
                      x["arm"],x["seed"]))
    script={panel:_judge(scripted[panel],None,"scripted_prior",draws)
            for panel in ("id","ood")}
    return {"native_seed_decisions":native,
            "native_DEV_pass":sum(x["both_panel_win"] for x in native)>=2,
            "DEV_selected_conventional":{"arm":selected["arm"],"seed":selected["seed"],
                "dev_active_regret":selected["panels"]["dev"]["active_information"]["mean_regret"],
                "ID_active_regret":selected["panels"]["id"]["active_information"]["mean_regret"],
                "OOD_active_regret":selected["panels"]["ood"]["active_information"]["mean_regret"]},
            "scripted_information_value":script,
            "scripted_both_panels_pass":all(s["development_win"] for s in script.values()),
            "independent_scientific_pass":False}


def run()->dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=fit_cohort()
    fit_support=fit_audit(fit)
    panels={p:worlds(p) for p in PANELS}
    panel_checks={p:audit_worlds(panels[p],p) for p in PANELS}
    plans={panel:{policy:r27._plans(world,policy)
                   for policy in POLICIES}
           for panel,world in panels.items()}
    scripted={panel:{policy:r27.scripted_score(p)
                     for policy,p in choices.items()}
              for panel,choices in plans.items()}
    rng=torch.Generator(device="cpu")
    rng.manual_seed(20319999)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=rng).tolist()
    start=time.perf_counter()
    models=[train(arm,seed,fit,plans,batches)
            for arm in ARMS
            for seed in (NATIVE_SEEDS if arm=="native_focus20" else CONVENTIONAL_SEEDS)]
    d=decision(models,scripted)
    return {"schema":SCHEMA,
            "classification":"SOURCE_NEUTRAL_TRAINED_ACTIVE_CAUSAL_TRANSFER_DEVELOPMENT_ONLY",
            "freeze_blob":FREEZE_BLOB,"parent_head":"a4913f86921d6758770098fcf7a688bf48cb6806",
            "actual_training_models":len(models),
            "total_optimizer_updates":len(models)*UPDATES,
            "factual_fifth_and_return_selected_fit":fit_support,
            "new_world_panel_audits":panel_checks,
            "trained_models":models,"scripted_comparators":scripted,
            "decision":d,
            "conventional_independently_convergence_qualified":False,
            "matched_model_capacity_and_compute":False,
            "original_external_effects":0,"spent_F7_replay":False,
            "scientific_pass":False,"deployment":"NONE",
            "train_and_evaluate_seconds":time.perf_counter()-start}


def retain_ci_receipt(receipt:dict)->None:
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r28_factual_curvature_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R28_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_original_external_firstuse(*_args,**_kwargs):
    raise PermissionError("F8R28_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
