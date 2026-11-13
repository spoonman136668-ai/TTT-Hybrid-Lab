"""R26: source-neutral nonlinear intervention choice, trained native Hybrid vs Bayesian controls.

All actions/effects are from new author-generated mathematical worlds. The
protected original CausalGym/F7 sources are never executed or replayed.
The fifth probe's action is selected *before* its actual outcome is known.
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
from experiments import p85mr_f8r26_gradient_conflict as parent

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r27_active_interventions_freeze.json"
FREEZE_BLOB="237aa9f50aaaab9232caaa9a93e3ab67a755becf"
PARENT_BLOB="ef488956741b339eb68c786432c48fefb6526165"
SCHEMA="ttt-hybrid.f8r27-active-nonlinear-factual-interventions.v1"
SEEDS=(985201,985203,985207)
POLICIES=("active_information","fixed_action0","fixed_action1","seeded_random")
BVAL=(-.3,-.1,.1,.3)
KVAL=(-.25,.25)
Q_ID=(-.55,.55)
Q_OOD=(-.45,.45)
PROBES=((-0.2,0),(0.2,0),(-0.2,1),(0.2,1))
STATES=(-.7,.7)
GOALS=(-.2,.2)
FIFTH_STATE=.7
B_STEPS={"id":8,"ood":16}
BASES={"id":20230000,"ood":20240000}
GROUPS=12
UPDATES=384
BATCH=8
BOOT=10000
BOOT_SEED=20249999
PRIOR_Q=.55
PRIOR_PAIRS=tuple((x,y) for x in Q_ID for y in Q_ID)


def blob_sha(raw:bytes)->str:
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked()->dict:
    raw=FREEZE.read_bytes()
    if blob_sha(raw)!=FREEZE_BLOB:
        raise ValueError("F8R27_FROZEN_PROTOCOL_DRIFT")
    f=json.loads(raw);w=f["world"];t=f["training"];s=f["scoring"]
    if (f["schema"]!=SCHEMA or f["stage"]!="FROZEN_BEFORE_LINKED_R27_CODE_OR_RESULTS"
        or f["classification"]!="SOURCE_NEUTRAL_ACTIVE_INTERVENTION_DEVELOPMENT_ONLY"
        or f["parent"]["head"]!="ab9b7503ae45f48c959a8e365c77f407d327dff2"
        or f["parent"]["r25_source_blob"]!=PARENT_BLOB
        or f["parent"]["r26_gradient_source_blob"]!="35402a765caf1781b50a283a83589ba82ed36481"
        or f["parent"]["r26_gradient_freeze_blob"]!="10ed62c741c419f03f3be233188a5e1ff97c82c1"
        or blob_sha((ROOT/"experiments/p85mr_f8r26_gradient_conflict.py").read_bytes())!="35402a765caf1781b50a283a83589ba82ed36481"
        or blob_sha((ROOT/"experiments/p85mr_f8r25_multimechanism.py").read_bytes())!=PARENT_BLOB
        or w["b_values"]!=list(BVAL) or w["k_values"]!=list(KVAL)
        or w["q_ID"]!=list(Q_ID) or w["q_OOD"]!=list(Q_OOD)
        or w["original_four_factual_probes"]!=[list(p) for p in PROBES]
        or w["fifth_factual_probe_state"]!=FIFTH_STATE
        or w["B_ID_steps"]!=8 or w["B_OOD_steps"]!=16
        or w["panel_seed_bases"]!=BASES or w["panels"]!=list(B_STEPS)
        or w["groups_per_panel"]!=GROUPS or w["plans_per_panel"]!=96
        or w["return_states"]!=list(STATES) or w["goals"]!=list(GOALS)
        or f["policies"]["variants"]!=list(POLICIES)
        or t["model"]!="native_focus20" or t["seeds"]!=list(SEEDS)
        or t["models"]!=3 or t["updates_each"]!=UPDATES
        or t["total_optimizer_updates"]!=1152 or t["batch_size"]!=BATCH
        or t["actual_batch_seed"]!=985299
        or s["bootstrap_draws"]!=BOOT or s["bootstrap_seed"]!=BOOT_SEED
        or s["native_min_seeds_passing_ID_and_OOD"]!=2
        or s["independent_scientific_pass"] is not False
        or any(x is not False for x in f["authority"].values())):
        raise ValueError("F8R27_OWNER_SOURCE_OR_SCOPE_DRIFT")
    r25.locked()
    parent.locked()
    return f


def after_A(state:float,action:int,b:tuple,k:tuple,q:tuple)->float:
    if (type(action) is not int or action not in (0,1)
        or not math.isfinite(state) or len(b)!=2 or len(k)!=2 or len(q)!=2):
        raise ValueError("F8R27_INVALID_ACTION_SPECIFIC_PHYSICS")
    return .4*state+b[action]+k[action]*state+q[action]*(state**2-.04)


def _record(state:float,action:int,following:float)->tuple[list,list]:
    return (r25.r22.r18.r5._action_key((state,0.,0.,0.),action),
            r25.r22.r18._observed(state,following))


def _world(split:str,group:int,twin:int)->dict:
    if (split not in B_STEPS or type(group) is not int or not 0<=group<GROUPS
        or type(twin) is not int or twin not in (0,1)):
        raise ValueError("F8R27_UNREGISTERED_WORLD")
    seed=BASES[split]+group
    rng=np.random.default_rng(seed)
    b=tuple(float(z) for z in rng.choice(BVAL,size=2))
    k=tuple(float(z) for z in rng.choice(KVAL,size=2))
    q=tuple(float(z) for z in rng.choice(Q_ID if split=="id" else Q_OOD,size=2))
    Bsign=1 if group%2==0 else -1
    if twin:
        b=b[::-1];k=k[::-1];q=q[::-1]
    early=[_record(s,a,after_A(s,a,b,k,q)) for s,a in PROBES]
    B=[]
    state=0.
    for i in range(B_STEPS[split]):
        sign=Bsign if i<8 else -Bsign
        if i==8 and split=="ood":
            state=0.
        action=r25.r22.r20.r3.behavior(state)^int(
            (seed+7*i+int(sign==1))%3==0)
        following=r25.B_after(state,action,sign)
        B.append(_record(state,action,following))
        state=following
    return {"split":split,"group":group,"twin":twin,"b":b,"k":k,"q":q,
            "Bsign":Bsign,"early":early,"B":B}


def worlds(split:str)->list[dict]:
    locked()
    return [_world(split,g,t) for g in range(GROUPS) for t in (0,1)]


def audit(ws:list[dict],split:str)->dict:
    if split not in B_STEPS or len(ws)!=24:
        raise ValueError("F8R27_SCORE_PANEL_COUNT_DRIFT")
    for g in range(GROUPS):
        a,b=ws[2*g:2*g+2]
        if (a["group"]!=b["group"] or a["twin"]!=0 or b["twin"]!=1
            or a["q"]!=b["q"][::-1] or a["b"]!=b["b"][::-1]
            or a["k"]!=b["k"][::-1] or a["B"]!=b["B"]
            or len(a["early"])!=4 or len(a["B"])!=B_STEPS[split]):
            raise ValueError("F8R27_INCONSISTENT_SWAPPED_MECHANISM_B_TWIN")
        for w in (a,b):
            # The quadratic term vanishes at original ±.2 probes, so no
            # factual early observation identifies q0 or q1.
            for idx,(s,act) in enumerate(PROBES):
                true_no_q=after_A(s,act,w["b"],w["k"],(0.,0.))
                if abs(after_A(s,act,w["b"],w["k"],w["q"])-true_no_q)>1e-12:
                    raise ValueError("F8R27_PREMATURE_Q_INFORMATION")
    return {"matched_B_twin_groups":GROUPS,"worlds":len(ws),
            "A_early_curvature_identifiable":False,
            "B_factual_steps_each":B_STEPS[split],
            "oracle_action_labels_excluded_from_inputs":True}


def factual_b_and_slope(w:dict)->tuple[tuple,tuple]:
    ep={"effect":torch.tensor([e for _,e in w["early"]],dtype=torch.float32)}
    b0,b1,slope0,slope1=r25.four_probe_predictions(ep)
    return (b0,b1),(slope0,slope1)


def regret(w:dict,state:float,goal:float,act:int)->float:
    outcomes=[after_A(state,q,w["b"],w["k"],w["q"]) for q in (0,1)]
    errs=[(v-goal)**2 for v in outcomes]
    return max(0.,errs[act]-min(errs))


def posterior_act(b:tuple,slope:tuple,state:float,goal:float,
                  measured_action:int,measured_q:float)->int:
    dim=state*state-.04
    posterior=[]
    for a in (0,1):
        point=b[a]+slope[a]*state
        if a==measured_action:
            point+=measured_q*dim
            var=0.
        else:
            var=(PRIOR_Q*dim)**2
        posterior.append((point-goal)**2+var)
    return min((0,1),key=lambda a:(posterior[a],a))


def active_probe(b:tuple,slope:tuple,state:float,goal:float)->int:
    """Select the *next* action by prospective Bayes decision information."""
    expected=[]
    for probe_act in (0,1):
        total=0.
        for qa,qb in PRIOR_PAIRS:
            q=(qa,qb)
            chosen=posterior_act(b,slope,state,goal,probe_act,q[probe_act])
            outcomes=[b[a]+slope[a]*state+q[a]*(state*state-.04)
                      for a in (0,1)]
            errs=[(v-goal)**2 for v in outcomes]
            total+=errs[chosen]-min(errs)
        expected.append(total/len(PRIOR_PAIRS))
    return min((0,1),key=lambda a:(expected[a],a))


def probe_action(policy:str,b:tuple,slope:tuple,state:float,
                 goal:float,group:int,si:int,gi:int)->int:
    if policy=="active_information":
        return active_probe(b,slope,state,goal)
    if policy=="fixed_action0":
        return 0
    if policy=="fixed_action1":
        return 1
    if policy=="seeded_random":
        return (group*7+si*3+gi*5+1)%2
    raise ValueError("F8R27_UNREGISTERED_INTERVENTION_POLICY")


def _plans(ws:list[dict],policy:str)->list[dict]:
    plans=[]
    for w in ws:
        b,slope=factual_b_and_slope(w)
        for si,state in enumerate(STATES):
            for gi,goal in enumerate(GOALS):
                a=probe_action(policy,b,slope,state,goal,w["group"],si,gi)
                extra=_record(FIFTH_STATE,a,
                    after_A(FIFTH_STATE,a,w["b"],w["k"],w["q"]))
                plans.append({"w":w,"state":state,"goal":goal,
                              "fifth_action":a,"fifth":extra,
                              "history":w["early"]+[extra]+w["B"]})
    if len(plans)!=96:
        raise ValueError("F8R27_PLAN_COUNT_DRIFT")
    return plans


def scripted_choice(p:dict,erased=False)->int:
    w=p["w"];b,slope=factual_b_and_slope(w)
    if erased:
        # Identical missing q information for both actions; no hidden q.
        expected=[(b[a]+slope[a]*p["state"]-p["goal"])**2
                  +(PRIOR_Q*(p["state"]**2-.04))**2 for a in (0,1)]
        return min((0,1),key=lambda a:(expected[a],a))
    fifth_s=FIFTH_STATE
    factual_delta=r25.r22.r19._physical_delta(
        torch.tensor(p["fifth"][1],dtype=torch.float32))
    after=fifth_s+factual_delta
    probed=p["fifth_action"]
    qhat=(after-b[probed]-slope[probed]*fifth_s)/(fifth_s**2-.04)
    return posterior_act(b,slope,p["state"],p["goal"],probed,qhat)


def _score_choices(plans:list[dict],choices:list[int],
                   erased_choices:list[int])->dict:
    if len(plans)!=96 or len(choices)!=96 or len(erased_choices)!=96:
        raise ValueError("F8R27_BAD_ACTION_COUNT")
    vals=[regret(p["w"],p["state"],p["goal"],int(a))
          for p,a in zip(plans,choices)]
    erased=[regret(p["w"],p["state"],p["goal"],int(a))
            for p,a in zip(plans,erased_choices)]
    groups=[float(np.mean(vals[8*g:8*g+8])) for g in range(GROUPS)]
    return {"mean_regret":float(np.mean(vals)),
            "erased_fifth_mean_regret":float(np.mean(erased)),
            "probe_information_penalty":float(np.mean(erased)-np.mean(vals)),
            "pergroup_regret":groups,
            "choices":choices,"actual_selected_actions":len(choices),
            "factual_fifth_probe_effects":len(plans),
            "oracle_mean_regret":0.}


def scripted_score(plans:list[dict])->dict:
    return _score_choices(plans,[scripted_choice(p) for p in plans],
                          [scripted_choice(p,True) for p in plans])


def _native_predictions(model,plans:list[dict],erased:bool)->list[int]:
    action_rows,effect_rows=[],[]
    for p in plans:
        base=p["history"]
        for action in (0,1):
            current=_record(p["state"],action,p["state"]) # effect ignored
            rows=base+[current]
            aa=torch.tensor([a for a,_ in rows],dtype=torch.float32)
            ee=torch.tensor([e for _,e in rows],dtype=torch.float32)
            ee[-1]=0.
            if erased:
                ee[4]=0.
            action_rows.append(aa);effect_rows.append(ee)
    inputs={"action":torch.stack(action_rows),"effect":torch.stack(effect_rows)}
    model.eval()
    with torch.no_grad():
        result=r25.r22.predict("native_focus20",model,inputs)
    expected=(2*len(plans),len(plans[0]["history"])+1,16)
    if result.shape!=expected or not bool(torch.isfinite(result).all()):
        raise ValueError("F8R27_NATIVE_HYPOTHETICAL_SHAPE_OR_NAN")
    arr=result[:,-1,0].detach().cpu().numpy()
    # Only the predicted consequence of the hypothetical action can
    # drive selection, never the evaluator's true hidden mechanism.
    after=np.asarray([plans[i//2]["state"]+2.4*math.atanh(
        max(-.98,min(.98,float(v)))) for i,v in enumerate(arr)],
        dtype=np.float64)
    vals=after.reshape(len(plans),2)
    return [int(min((0,1),key=lambda a:(
        (vals[i,a]-p["goal"])**2,a)))
        for i,p in enumerate(plans)]


def native_score(model,plans:list[dict])->dict:
    return _score_choices(plans,
      _native_predictions(model,plans,False),
      _native_predictions(model,plans,True))


def evaluate_models(model,panels:dict)->dict:
    result={}
    for split,ws in panels.items():
        case={}
        for name in POLICIES:
            plans=_plans(ws,name)
            case[name]={"native":native_score(model,plans),
                        "scripted":scripted_score(plans)}
        result[split]=case
    return result


def _decision(results:list[dict],panels:dict)->dict:
    rng=np.random.default_rng(BOOT_SEED)
    draws=rng.integers(0,GROUPS,size=(BOOT,GROUPS))
    def judged(arms:dict,who:str)->dict:
        current=arms["active_information"][who]
        f=min(("fixed_action0","fixed_action1"),
              key=lambda name:(arms[name][who]["mean_regret"],name))
        fixed=arms[f][who]
        diff=np.asarray(fixed["pergroup_regret"])-np.asarray(
             current["pergroup_regret"])
        lower=float(np.percentile(diff[draws].mean(axis=1),2.5))
        gain=fixed["mean_regret"]-current["mean_regret"]
        info=current["probe_information_penalty"]
        required= (.005 if who=="native" else .005)
        success=bool(gain>=required and lower>0
                  and (who!="native" or info>=.005))
        return {"active_regret":current["mean_regret"],
                "strongest_fixed_policy":f,
                "strongest_fixed_regret":fixed["mean_regret"],
                "seeded_random_regret":arms["seeded_random"][who]["mean_regret"],
                "active_fifth_probe_erasure_penalty":info,
                "active_minus_best_fixed_gain":gain,
                "bootstrap_2p5_lower_gain":lower,
                "development_win":success}
    per_seed=[]
    for row in results:
        parts={split:judged(row["panels"][split],"native")
               for split in B_STEPS}
        per_seed.append({"seed":row["seed"],"panels":parts,
                         "both_panel_win":all(v["development_win"]
                                              for v in parts.values())})
    scripted={split:judged(
        {p:{"scripted":scripted_score(_plans(ws,p))}
         for p in POLICIES},"scripted") for split,ws in panels.items()}
    return {"native_seed_decisions":per_seed,
            "native_development_pass":sum(int(x["both_panel_win"])
                     for x in per_seed)>=2,
            "scripted_decisions":scripted,
            "scripted_both_panel_pass":all(
                d["development_win"] for d in scripted.values()),
            "independent_scientific_pass":False}


def run()->dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit=r25.cohort("fit")
    if len(fit)!=256:
        raise ValueError("F8R27_R25_FIT_CUSTODY_DRIFT")
    panels={p:worlds(p) for p in B_STEPS}
    audits={p:audit(ws,p) for p,ws in panels.items()}
    g=torch.Generator(device="cpu");g.manual_seed(985299)
    batches=torch.randint(0,len(fit),(UPDATES,BATCH),generator=g).tolist()
    start=time.perf_counter()
    rows=[]
    for seed in SEEDS:
        model=r25.r22.model_for("native_focus20",seed).cpu()
        original=r25.r22.r18.r5.f5.hash_weights(model)
        optim=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.01)
        train_start=time.perf_counter()
        for indices in batches:
            data,valid=r25.r22.r18.r5.f4._padded(
                [r25.r22.factual(fit[i]) for i in indices])
            model.train();optim.zero_grad(set_to_none=True)
            loss=r25.objective("native_focus20",model,data,valid)
            if not bool(torch.isfinite(loss)):
                raise ValueError("F8R27_NONFINITE_FIT_LOSS")
            loss.backward()
            grads=[p.grad for p in model.parameters()]
            if (not any(v is not None and bool((v.abs()>0).any()) for v in grads)
                or any(v is not None and not bool(torch.isfinite(v).all()) for v in grads)):
                raise ValueError("F8R27_MISSING_OR_INVALID_NATIVE_GRAD")
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optim.step()
        train_s=time.perf_counter()-train_start
        digest=r25.r22.r18.r5.f5.hash_weights(model)
        if digest==original:
            raise ValueError("F8R27_NO_ACTUAL_TRAINING")
        scores=evaluate_models(model,panels)
        cold=r25.r22.model_for("native_focus20",seed).cpu()
        cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
        if (r25.r22.r18.r5.f5.hash_weights(cold)!=digest
            or any(_native_predictions(cold,_plans(panels[split],"active_information"),False)
                   !=scores[split]["active_information"]["native"]["choices"]
                   for split in B_STEPS)):
            raise ValueError("F8R27_COLD_CHECKPOINT_ACTION_PARITY_FAILED")
        rows.append({"seed":seed,"arm":"native_focus20","steps":UPDATES,
                     "parameters":sum(p.numel() for p in model.parameters()),
                     "initial_hash":original,"trained_hash":digest,
                     "native_fit_wall_seconds":train_s,
                     "cold_checkpoint_action_parity":True,
                     "panels":scores})
    decision=_decision(rows,panels)
    return {"schema":SCHEMA,
            "classification":"SOURCE_NEUTRAL_ACTIVE_INTERVENTION_DEVELOPMENT_ONLY",
            "original_R25_source_blob":PARENT_BLOB,
            "frozen_protocol_blob":FREEZE_BLOB,
            "train_factual_R25_sessions":len(fit),
            "train_factual_R25_effects":len(fit)*13,
            "model_count":len(rows),
            "actual_optimizer_updates":len(rows)*UPDATES,
            "ID_OOD_panel_audits":audits,
            "registered_policies":list(POLICIES),
            "model_results":rows,"decision":decision,
            "scripted_bayesian_memory_uses_no_hidden_test_latents":True,
            "native_conventional_convergence_qualified":False,
            "whole_process_wall_seconds":time.perf_counter()-start,
            "original_external_source_effects":0,"F7_replay":False,
            "independent_scientific_pass":False,"deployment":"NONE"}


def retain_ci_receipt(data:dict)->None:
    if os.environ.get("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r27_active_intervention_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R27_RECEIPT_OVERWRITE_DENIED")
    path.write_text(json.dumps(data,sort_keys=True,indent=2,
                                allow_nan=False)+"\n")


def refuse_original_scientific_source(*_args,**_kwargs):
    raise PermissionError("F8R27_EXTERNAL_SCIENTIFIC_FIRSTUSE_NOT_AUTHORIZED")
