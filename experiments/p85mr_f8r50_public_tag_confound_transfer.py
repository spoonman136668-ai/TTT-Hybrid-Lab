"""F8-R50: Frozen R49 policy under new public-tag confound transfer.

Source-neutral, synthetic, prospective, no retraining on new OOD, no oracle
to model. Source and acceptance thresholds were frozen *before* this file.
"""
from __future__ import annotations
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

import torch
import torch.nn.functional as F

from experiments import p85mr_f8r49_public_event_selection as r49

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r50_public_tag_confound_freeze.json"
FREEZE_BLOB="6339b5eab5ca6b146902ace371511b0d28098684"
SCHEMA="ttt-hybrid.f8r50-public-tag-confound-transfer-freeze.v1"
PARENT="2b297ba7316c0c552522ecd2ae1a753934e88174"
SALT=22490011
GROUPS=8
B_STEPS=32
SEEDS=(985201,985207)
ARMS=("event_context_selector","raw_context_selector",
      "event_context_native","gru_context_control")
CAUSAL_MAG=.12
SPOOFER_TAGS=(.88,.77)
SPOOFER_DELTAS=(.015,-.015)
CUE_TAGS=(.42,.18)
THRESHOLD=.06
ZERO_RETURN_EFFECT=torch.zeros((1,16),dtype=torch.float32)


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R50_FREEZE_CHANGED_AFTER_PRECOMMIT")
    f=json.loads(raw)
    d,m,t=f["source"],f["methods"],f["frozen_decision"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_GENERATOR_CODE_OR_ANY_R50_SCORE" or
        f["parent"]["source_head"]!=PARENT or
        f["parent"]["r49_control_blob"]!=r49.FREEZE_BLOB or
        d["new_ood_salt"]!=SALT or d["eval_groups"]!=GROUPS or
        d["selected_B_length"]!=B_STEPS or
        d["causal_physical_delta_magnitude"]!=CAUSAL_MAG or
        d["confounding_nuisance"]["visible_tags"]!=list(SPOOFER_TAGS) or
        d["confounding_nuisance"]["selected_physical_deltas"]!=list(SPOOFER_DELTAS) or
        m["updates_per_arm"]!=r49.UPDATES or
        m["batch_pairs"]!=r49.BATCH or
        m["original_training_seed"]!=r49.DRAW_SEED or
        m["total_replayed_training_updates"]!=len(ARMS)*len(SEEDS)*r49.UPDATES or
        tuple((a["name"],tuple(a["seeds"])) for a in m["arm_instances"])
           !=tuple((name,SEEDS) for name in ARMS) or
        t["independently_qualified_scientific_gain"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R50_SOURCE_OR_AUTHORITY_DRIFT")
    r49.locked()
    return f


def positions(group):
    if type(group) is not int or not 0<=group<GROUPS:
        raise ValueError("F8R50_UNREGISTERED_GROUP")
    # Four unique positions are allocated from a prospective source salt.
    ranks=sorted(range(B_STEPS),key=lambda t:
                 hashlib.sha256(f"F8-R50/{SALT}/{group}/{t}".encode()).digest())
    return tuple(ranks[:4])


def world(group,ctx,q_sign,gate):
    positions(group)
    if ctx not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R50_UNREGISTERED_FACTOR")
    # Return dynamics are the previous synthetic task; new OOD HISTORY is not.
    return r49.world("ood",group,ctx,q_sign,gate)


def worlds():
    return [world(g,c,q,gate)
            for g in range(GROUPS)
            for c in (0,1) for q in (-1,1) for gate in (-1,1)]


def B_selected(w):
    g,c,gate=w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,high1,high2=positions(g)
    facts=[]
    for t in range(B_STEPS):
        if t==cue:
            tag,delta=(.42 if c==0 else .18),CAUSAL_MAG*gate
        elif t==decoy:
            tag,delta=(.18 if c==0 else .42),-CAUSAL_MAG*gate
        elif t==high1:
            tag,delta=SPOOFER_TAGS[0],SPOOFER_DELTAS[0]
        elif t==high2:
            tag,delta=SPOOFER_TAGS[1],SPOOFER_DELTAS[1]
        else:
            tag=.03 if t%3==0 else 0.
            delta=round(.007*math.sin((SALT+g)*.137+t*.271)
                +.004*(((t+g)%4)-1.5),8)
        facts.append((tag,t%2,delta))
    return facts


def observed(w,*,erase_B=False,erase_tag=False,erase_fifth=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    early=[r49.r33.b._record(s,act,r49.r33.r32.after(s,act,c,q),c)
           for s,act in r49.r33.r32.EARLY]
    early.append(r49.r33.b._record(r49.PROBE,c,
              r49.r33.r32.after(r49.PROBE,c,c,q),c))
    for tag,action,delta in B_selected(w):
        key,eff=r49.r33.b._record(
            0. if erase_tag else tag,action,
            (0. if erase_tag else tag)+delta,c)
        eff=eff.clone()
        eff[4]=0.
        early.append((key,eff))
    a=torch.stack([x for x,_ in early])
    e=torch.stack([y for _,y in early])
    if erase_B:e[5:].zero_()
    if erase_fifth:e[4].zero_()
    return a,e


def candidates(w,**opts):
    a,e=observed(w,**opts)
    aa=[];ee=[]
    for action in (0,1):
        key=r49.r33.b._record(
            w["return_state"],action,w["return_state"],w["public_context"])[0]
        aa.append(torch.cat((a,key[None])))
        ee.append(torch.cat((e,ZERO_RETURN_EFFECT)))
    return torch.stack(aa),torch.stack(ee)


def source_parity():
    highest=0.
    for group in range(GROUPS):
        for ctx in (0,1):
            for q in (-1,1):
                plus=world(group,ctx,q,1)
                minus=world(group,ctx,q,-1)
                a1,e1=candidates(plus)
                a2,e2=candidates(minus)
                if not torch.equal(a1,a2):
                    raise ValueError("F8R50_GATE_ACTION_KEY_LEAK")
                distance=(e1[:,5:-1,:].mean(1)-
                          e2[:,5:-1,:].mean(1)).abs().max().item()
                highest=max(highest,distance)
                if distance>1e-7:
                    raise ValueError("F8R50_GATE_MEAN_EFFECT_LEAK")
                if torch.equal(e1[:,5:-1],e2[:,5:-1]):
                    raise ValueError("F8R50_GATE_FACTS_INSEPARABLE")
    return {"groups":GROUPS,"contexts":2,
            "maximum_opposite_gate_B_mean_difference":highest,
            "all_keys_equal_opposite_gate":True}


def positive_reference(offset):
    mistakes=0
    for w in worlds():
        actions,effects=candidates(w)
        a,e=actions[0],effects[0]
        ctx=r49.r31._decode_key(a[-1])[1]
        fifth=r49.PROBE+r49.r31._physical_delta(e[4,0])-offset
        facts=[(r49.r31._decode_key(a[t])[0],
                r49.r31._physical_delta(e[t,0]))
               for t in range(5,a.shape[0]-1)]
        strongest=sorted(facts,key=lambda x:abs(x[1]),reverse=True)[:2]
        if (len(strongest)!=2 or
            min(abs(effect) for _,effect in strongest)<=THRESHOLD or
            any(abs(effect)>THRESHOLD for _,effect in
                sorted(facts,key=lambda x:abs(x[1]),reverse=True)[2:])):
            raise ValueError("F8R50_OBSERVABLE_ELIGIBILITY_NOT_IDENTIFIABLE")
        informative=(max(strongest,key=lambda x:x[0])[1]
                     if ctx==0 else min(strongest,key=lambda x:x[0])[1])
        picked=ctx if fifth*informative>0 else 1-ctx
        mistakes+=int(r49.regret(w,picked)>1e-10)
    if mistakes:
        raise ValueError("F8R50_PUBLIC_REFERENCE_NOT_FEASIBLE")
    return {"worlds":GROUPS*8,"mistakes":mistakes,
            "uses_hidden_cue_indices":False,
            "handcoded_physical_effect_magnitude_threshold":THRESHOLD}


def original_draws():
    gen=torch.Generator(device="cpu")
    gen.manual_seed(r49.DRAW_SEED)
    return torch.randint(0,r49.TRAIN_PAIRS,
            (r49.UPDATES,r49.BATCH),generator=gen).tolist()


def reproduce_R49_frozen(arm,seed,offset,records,labels,draws,f):
    model=r49.Policy(arm,seed,offset)
    groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            groups.append({"params":[p],
                 "lr":.02 if name in ("slopes","slope","gain","native_gain")
                      else .001})
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    start=time.perf_counter()
    productive=0
    for ids in draws:
        a,e,y=r49.batch(records,labels,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R50_NONFINITE_REPRODUCTION_LOSS")
        loss.backward()
        if r49.productive_gradient(
            [p.grad for p in model.parameters() if p.requires_grad]):
            productive+=1
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
    if arm=="event_context_selector":
        target=f["source"]["original_R49_seed_slope_repro_target"][
            f"seed{seed}"]
        got=model.slopes.detach().tolist()
        tol=f["source"]["original_R49_seed_slope_repro_target"][
            "max_absolute_tolerance"]
        if max(abs(a-b) for a,b in zip(target,got))>tol:
            raise ValueError("F8R50_ORIGINAL_R49_LEARNED_WEIGHT_DRIFT")
        if productive!=r49.UPDATES:
            raise ValueError("F8R50_ORIGINAL_R49_NO_GENUINE_SLOPE_TRAINING")
    params=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return model,{"arm":arm,"seed":seed,"reproduced_R49_FIT_updates":len(draws),
                   "productive_gradient_steps":productive,
                   "trained_parameters":params,
                   "R49_original_heldout_FIT":r49.heldout(model,records,labels),
                   "context_slopes":model.slopes.detach().tolist()
                      if hasattr(model,"slopes") else None,
                   "training_wall_seconds":time.perf_counter()-start,
                   "trained_weights_sha256":r49.r40._hash(model)}


def evaluate(model,*,erase_B=False,erase_tag=False,erase_fifth=False,
             zero_native=False):
    ww=worlds()
    inputs=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                       erase_fifth=erase_fifth) for w in ww]
    a=torch.cat([pair[0] for pair in inputs])
    e=torch.cat([pair[1] for pair in inputs])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,zero_native=zero_native)
    if costs.shape!=(len(ww),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R50_INVALID_MODEL_PREACTION_COST")
    actions=costs.argmin(-1).tolist()
    regrets=[r49.regret(w,action) for w,action in zip(ww,actions)]
    return {"choices":actions,"mean_regret":statistics.fmean(regrets),
            "per_context_regret":{
                str(c):statistics.fmean(regrets[i] for i,w in enumerate(ww)
                                      if w["public_context"]==c)
                for c in (0,1)},
            "per_group_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(GROUPS)],
            "gate_pair_flips":sum(actions[i]!=actions[i+1]
                                  for i in range(0,len(ww),2))}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity=source_parity()
    fit,labels=r49.factual_fit()
    offset=r49.calibration(fit)
    # Check eligibility from observed measurements before model scoring.
    control=positive_reference(offset)
    draws=original_draws()
    all_records=[]
    for arm in ARMS:
        for seed in SEEDS:
            model,training=reproduce_R49_frozen(
                arm,seed,offset,fit,labels,draws,f)
            started=time.perf_counter()
            panels={
                "active":evaluate(model),
                "erase_B":evaluate(model,erase_B=True),
                "erase_tag":evaluate(model,erase_tag=True),
                "erase_fifth":evaluate(model,erase_fifth=True),
                "zero_native":evaluate(model,zero_native=True)
            }
            cold=r49.Policy(arm,seed,offset)
            cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
            if r49.r40._hash(cold)!=training["trained_weights_sha256"]:
                raise ValueError("F8R50_COLD_STATE_DRIFT")
            if evaluate(cold)["choices"]!=panels["active"]["choices"]:
                raise ValueError("F8R50_COLD_POLICY_DRIFT")
            all_records.append({**training,"transfer":panels,
                "OOD_score_wall_seconds":time.perf_counter()-started,
                "cold_restore":True})
    gates=[]
    for seed in SEEDS:
        a=next(row for row in all_records
               if row["arm"]=="event_context_selector" and row["seed"]==seed)
        scored=a["transfer"]["active"]
        thresholds=f["frozen_decision"]["main_transfer_success_thresholds"]
        passes=bool(
            scored["mean_regret"]<=thresholds["max_ood_mean_regret"] and
            all(x<=thresholds["max_each_public_context_regret"] for
                x in scored["per_context_regret"].values()) and
            scored["gate_pair_flips"]>=thresholds["min_gate_pair_flips"])
        gates.append({"seed":seed,"transfer_conditions_pass":passes,
                "OOD_regret":scored["mean_regret"],
                "OOD_per_context_regret":scored["per_context_regret"],
                "OOD_gate_pair_flips":scored["gate_pair_flips"],
                "R49_original_FIT_heldout":a["R49_original_heldout_FIT"],
                "R49_original_reproduced_slopes":a["context_slopes"]})
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "source_neutral_new_panel_only":True,
        "original_R49_FIT_only_training":True,
        "original_R49_FIT_factual_selected_returns":len(fit),
        "original_R49_FIT_only_offset":offset,
        "new_R50_synthetic_OOD_worlds":len(worlds()),
        "observational_source_parity":parity,
        "observable_positive_reference":control,
        "models":all_records,
        "original_R49_reproduction_optimizer_updates":
            len(all_records)*r49.UPDATES,
        "judgment":{"both_seeds_transfer_pass":
            all(x["transfer_conditions_pass"] for x in gates),
            "seeds":gates,
            "independent_scientific_pass":False,
            "Native_specific_gain_verified":False,
            "source_neutral_synthetic_only":True},
        "independent_scientific_pass":False,
        "external_original_firstuse":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    path=ROOT/"local_runs/p85mr_f8r50_public_tag_confound_transfer_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R50_REFUSE_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                              allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R50_EXTERNAL_FIRSTUSE_UNAUTHORIZED")
