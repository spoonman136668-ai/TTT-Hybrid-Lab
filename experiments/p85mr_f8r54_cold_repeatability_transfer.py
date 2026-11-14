"""F8-R54 prospective synthetic cold transfer of unmodified R53 trained policies.

Test whether a learned variance-weighted selector survives unseen stationary
distractor action-contrasts (contrary to R53 FIT's oscillatory ones).
No new OOD training, no original external first-use, no oracle return effect.
All reproducibility and acceptance criteria were frozen before this code.
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
from torch.nn import functional as F
from experiments import p85mr_f8r53_repeated_intervention_stability as r53

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r54_cold_repeatability_transfer_freeze.json"
FREEZE_BLOB="e88e70a4945dff1021873d9f91d44be7b4488861"
SCHEMA="ttt-hybrid.f8r54-cold-repeatability-transfer-freeze.v1"
PARENT="289f44df8b8c7217f01c40afa646c2bbd2a18272"
SALT={"oscillating":22810000,"stationary_confounder":22820000}
PANELS=("oscillating","stationary_confounder")
SEEDS=r53.SEEDS
ARMS=r53.ARMS
N_SOURCE=16
REPS=3
GROUPS=8
EXPECTED_WORLD_COUNT=64
B_RECORDS=96
SPURIOUS=.45
CAUSAL=.12


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R54_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    src=f["source"]
    replay=f["original_replay"]
    gate=f["frozen_decision"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_NEW_GENERATOR_OR_ANY_R54_SCORE" or
        f["parent"]["qualified_R53_SHA"]!=PARENT or
        f["parent"]["r53_freeze_blob"]!=r53.FREEZE_BLOB or
        src["new_salts"]!=SALT or
        src["worlds_each"]!=EXPECTED_WORLD_COUNT or
        src["public_sources"]!=N_SOURCE or
        src["selected_repetitions_per_source"]!=REPS or
        src["actually_selected_interventions_per_repetition"]!=2 or
        src["selected_B_events"]!=B_RECORDS or
        src["full_history_events"]!=102 or
        src["causal_action_contrast_magnitude"]!=CAUSAL or
        src["spurious_action_contrast_magnitude"]!=SPURIOUS or
        replay["original_seeds"]!=list(SEEDS) or
        replay["arms"]!=list(r53.ARM_NAMES) or
        replay["original_fit_updates_per_arm"]!=r53.UPDATES or
        replay["training_draws"]!=r53.BATCH or
        replay["total_replayed_fit_steps"]!=len(ARMS)*r53.UPDATES or
        gate["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R54_PROTOCOL_OR_AUTHORITY_CHANGED")
    r53.locked()
    return f


def positions(panel,group):
    if (panel not in SALT or type(group) is not int
        or not 0<=group<GROUPS):
        raise ValueError("F8R54_UNREGISTERED_SHIFT_GROUP")
    order=sorted(range(N_SOURCE),key=lambda s:hashlib.sha256(
        f"F8-R54/{panel}/{SALT[panel]}/{group}/{s}".encode()).digest())
    return tuple(order[:4])


def world(panel,group,ctx,q_sign,gate):
    positions(panel,group)
    if ctx not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R54_INVALID_FACTOR")
    # The evaluator return physics uses exactly original R53 OOD.
    # r54_panel is for NEW source generation only, not an action/model input.
    return {"panel":"ood","r54_panel":panel,"group":group,
            "public_context":ctx,"q_evaluator_only":q_sign,
            "gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R54_UNREGISTERED_SHIFT_PANEL")
    return [world(panel,g,c,q,gate)
            for g in range(GROUPS)
            for c in (0,1)
            for q in (-1,1)
            for gate in (-1,1)]


def selected_history(w):
    panel=w["r54_panel"]
    g,ctx,gate=w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,s1,s2=positions(panel,g)
    rows=[]
    for source in range(N_SOURCE):
        if source==cue: tag=.42 if ctx==0 else .18
        elif source==decoy: tag=.18 if ctx==0 else .42
        elif source==s1: tag=.88
        elif source==s2: tag=.77
        else:tag=.03 if source%3==0 else 0.
        for repetition in range(REPS):
            if source==cue:
                contrast=CAUSAL*gate
            elif source==decoy:
                contrast=-CAUSAL*gate
            elif source in (s1,s2):
                sign=1 if source==s1 else -1
                pattern=1 if panel=="stationary_confounder" else (
                    1 if repetition!=1 else -1)
                contrast=SPURIOUS*sign*pattern
            else:
                contrast=0.
            baseline=(round(.014*math.sin(
                 (SALT[panel]+g)*.137+source*.271)+
                 .004*(((source+g)%4)-1.5),8)
                 if source not in (cue,decoy,s1,s2) else 0.)
            rows.append((tag,0,baseline+contrast))
            rows.append((tag,1,baseline-contrast))
    return rows


def observed_history(w,*,erase_B=False,erase_tag=False,erase_fifth=False):
    ctx,q=w["public_context"],w["q_evaluator_only"]
    records=[r53.r52.r51.r33.b._record(
             s,action,r53.r52.r51.r33.r32.after(s,action,ctx,q),ctx)
             for s,action in r53.r52.r51.r33.r32.EARLY]
    records.append(r53.r52.r51.r33.b._record(
        r53.PROBE,ctx,
        r53.r52.r51.r33.r32.after(r53.PROBE,ctx,ctx,q),ctx))
    for tag,action,delta in selected_history(w):
        keytag=0. if erase_tag else tag
        key,eff=r53.r52.r51.r33.b._record(
            keytag,action,keytag+delta,ctx)
        eff=eff.clone()
        eff[4].zero_()
        records.append((key,eff))
    a=torch.stack([a for a,_ in records])
    e=torch.stack([e for _,e in records])
    if erase_B:e[5:].zero_()
    if erase_fifth:e[4].zero_()
    return a,e


def candidates(w,**opts):
    a,e=observed_history(w,**opts)
    actions=[];effects=[]
    for choice in (0,1):
        key=r53.r52.r51.r33.b._record(
            w["return_state"],choice,w["return_state"],
            w["public_context"])[0]
        actions.append(torch.cat((a,key[None])))
        effects.append(torch.cat((
            e,torch.zeros((1,16),dtype=e.dtype))))
    return torch.stack(actions),torch.stack(effects)


def source_parity(panel):
    biggest=0.
    for group in range(GROUPS):
        for ctx in (0,1):
            for q in (-1,1):
                a,e=candidates(world(panel,group,ctx,q,1))
                aa,ee=candidates(world(panel,group,ctx,q,-1))
                if not torch.equal(a,aa):
                    raise ValueError("F8R54_GATE_ACTION_KEY_LEAK")
                difference=(e[:,5:-1].mean(1)-
                            ee[:,5:-1].mean(1)).abs().max().item()
                biggest=max(biggest,difference)
                if difference>1e-7:
                    raise ValueError("F8R54_FULL_MEAN_GATE_LEAK")
                if torch.equal(e[:,5:-1],ee[:,5:-1]):
                    raise ValueError("F8R54_GATE_SIGNAL_IDENTICAL")
    return {"panel":panel,"groups":GROUPS,
            "maximum_paired_full_B_mean_difference":biggest}


def public_tag_reference(offset):
    controls={}
    for panel in PANELS:
        errors=0
        for w in worlds(panel):
            a,e=candidates(w)
            tags,mean,variance,reliable,magnitude,ratio,fifth,ctx,obs=(
                r53.public_repeated_features(
                    a,e,torch.tensor(offset,dtype=torch.float32)))
            # A weak FIT-informed transparent source-specific control.
            # No hidden causal source index, private q/gate or OOD labels.
            hits=((tags-.42).abs()<.005)|((tags-.18).abs()<.005)
            if int(hits.sum())!=2:
                raise ValueError("F8R54_PUBLIC_TAG_IDENTIFIABILITY_FAILED")
            ix=torch.nonzero(hits[0],as_tuple=False).flatten()
            cues=tags[0,ix]
            contrasts=mean[0,ix]
            chosen_idx=int(torch.argmax(cues) if w["public_context"]==0
                           else torch.argmin(cues))
            pred=(w["public_context"] if
                  float(fifth[0]*contrasts[chosen_idx])>0
                  else 1-w["public_context"])
            errors+=int(r53.regret(w,pred)>1e-10)
        if errors:
            raise ValueError("F8R54_PUBLIC_REFERENCE_CANNOT_SOLVE_NEW_PANEL")
        controls[panel]={"worlds":EXPECTED_WORLD_COUNT,
                         "mistakes":errors,
                         "handcoded_public_tag_identity":True}
    return controls


def original_draws():
    rng=torch.Generator(device="cpu")
    rng.manual_seed(r53.DRAW_SEED)
    return torch.randint(0,r53.TRAIN_PAIRS,
                         (r53.UPDATES,r53.BATCH),generator=rng).tolist()


def reproduce_original_FIT(arm,seed,offset,records,labels,draws,f):
    model=r53.Policy(arm,seed,offset)
    param_groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            param_groups.append({"params":[p],
                "lr":.02 if name in ("slopes","beta","gain","native_gain")
                     else .001})
    optimizer=torch.optim.AdamW(param_groups,weight_decay=0.)
    productive=0
    before=time.perf_counter()
    for ids in draws:
        a,e,y=r53.batch(records,labels,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R54_NONFINITE_ORIGINAL_FIT_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in grads):
            raise ValueError("F8R54_NONFINITE_ORIGINAL_FIT_GRADIENT")
        productive+=int(any(g is not None and bool(torch.any(g!=0))
                           for g in grads))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
    target=f["original_replay"]["reproduction_targets"]
    if arm=="learned_repeatability":
        spec=target[f"seed{seed}"]
        beta=float(model.beta.detach())
        slopes=model.slopes.detach().tolist()
        if (min(abs(beta-spec["beta_windows"]),
                abs(beta-spec["beta_ubuntu"]))>
                      target["tolerance_beta"] or
            any(min(abs(slopes[i]-
                spec["context_slopes_windows"][i]),
                abs(slopes[i]-
                spec["context_slopes_ubuntu"][i]))>
                target["tolerance_slopes"] for i in (0,1))
            or productive!=r53.UPDATES):
            raise ValueError("F8R54_ORIGINAL_R53_WEIGHT_REPRODUCTION_FAILED")
    heldout=r53.heldout(model,records,labels)
    original=r53.score(model,"ood")
    if (arm=="learned_repeatability" and
        (heldout!=1.0 or original["mean_regret"]!=0.)):
        raise ValueError("F8R54_ORIGINAL_R53_POLICY_NOT_REPRODUCED")
    return model,{"arm":arm,"seed":seed,
        "original_FIT_heldout":heldout,
        "original_R53_OOD_regret":original["mean_regret"],
        "original_R53_OOD_choices":original["choices"],
        "trainable_parameters":sum(p.numel()
                                   for p in model.parameters() if p.requires_grad),
        "original_FIT_replayed_optimizer_updates":len(draws),
        "productive_gradient_steps":productive,
        "replayed_beta":float(model.beta.detach())
           if hasattr(model,"beta") else None,
        "replayed_context_slopes":model.slopes.detach().tolist()
           if hasattr(model,"slopes") else None,
        "replayed_weight_hash":r53.r52.r51.r40._hash(model),
        "original_FIT_train_seconds":time.perf_counter()-before}


def score(model,panel,*,zero_beta=False,zero_native=False,erase_tag=False):
    ww=worlds(panel)
    data=[candidates(w,erase_tag=erase_tag) for w in ww]
    a=torch.cat([x[0] for x in data])
    e=torch.cat([x[1] for x in data])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,zero_beta=zero_beta,zero_native=zero_native)
    if (costs.shape!=(len(ww),2) or
        not bool(torch.isfinite(costs).all())):
        raise ValueError("F8R54_INVALID_COLD_POLICY_OUTPUT")
    choices=costs.argmin(-1).tolist()
    regrets=[r53.regret(w,action) for w,action in zip(ww,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "context_regrets":{str(ctx):statistics.fmean(
                regrets[i] for i,w in enumerate(ww)
                if w["public_context"]==ctx) for ctx in (0,1)},
            "per_group_regret":[statistics.fmean(
                regrets[g*8:(g+1)*8]) for g in range(GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(ww),2))}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity={p:source_parity(p) for p in PANELS}
    original_fit,labels=r53.factual_fit()
    offset=r53.calibration(original_fit)
    feasibility=public_tag_reference(offset)
    draws=original_draws()
    trained=[]
    for arm,seed in ARMS:
        m,base=reproduce_original_FIT(
            arm,seed,offset,original_fit,labels,draws,f)
        panels={}
        start=time.perf_counter()
        for panel in PANELS:
            panels[panel]={
                "active":score(m,panel),
                "zero_beta":score(m,panel,zero_beta=True),
                "zero_native":score(m,panel,zero_native=True),
                "erase_tag":score(m,panel,erase_tag=True)}
        restored=r53.Policy(arm,seed,offset)
        restored.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
        if r53.r52.r51.r40._hash(restored)!=base["replayed_weight_hash"]:
            raise ValueError("F8R54_COLD_RESTORE_WEIGHT_DRIFT")
        if any(score(restored,p)["choices"]!=panels[p]["active"]["choices"]
                for p in PANELS):
            raise ValueError("F8R54_COLD_RESTORE_ACTION_DRIFT")
        trained.append({**base,"new_panels":panels,
            "new_OOD_scoring_seconds":time.perf_counter()-start,
            "cold_restore":True})
    frozen=f["frozen_decision"]
    verdict=[]
    for seed in SEEDS:
        row=next(x for x in trained if x["arm"]=="learned_repeatability"
                 and x["seed"]==seed)
        criteria={}
        for panel in PANELS:
            active=row["new_panels"][panel]["active"]
            criteria[panel]=bool(
                active["mean_regret"]<=frozen["max_new_each_panel_mean_regret"]
                and all(v<=frozen["max_new_each_public_context_regret"]
                        for v in active["context_regrets"].values())
                and active["gate_pair_flips"]>=
                       frozen["min_new_gate_pair_flips"])
        verdict.append({"seed":seed,"panels_pass":criteria,
            "full_cold_transfer_pass":all(criteria.values()),
            "oscillating_OOD":row["new_panels"]["oscillating"]["active"],
            "stationary_OOD":row["new_panels"]["stationary_confounder"]["active"],
            "original_R53_OOD_regret":row["original_R53_OOD_regret"]})
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "source_neutral_synthetic_cold_transfer_only":True,
        "original_R53_FIT_selected_factual_returns":len(original_fit),
        "original_R53_FIT_train_pairs":r53.TRAIN_PAIRS,
        "original_R53_FIT_heldout_pairs":r53.HELDOUT_PAIRS,
        "original_FIT_only_fifth_offset":offset,
        "new_panel_source_parity":parity,
        "positive_observation_reference":feasibility,
        "original_R53_FIT_replayed_optimizer_steps":len(ARMS)*r53.UPDATES,
        "original_R53_models":trained,
        "judgment":{"two_seed_transfer_pass":all(x["full_cold_transfer_pass"]
                                                  for x in verdict),
                    "seeds":verdict,
                    "independent_scientific_pass":False,
                    "Native_specific_gain_verified":False,
                    "full_capacity_compute_matched":False},
        "independent_scientific_pass":False,
        "external_original_first_use":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r54_cold_repeatability_transfer_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R54_REFUSE_RESULT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R54_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
