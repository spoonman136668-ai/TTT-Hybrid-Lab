"""F8-R56 cold transfer: genuinely selected upstream interventions and their
observed immediate and delayed goal-proxy sensor effects on TWO entirely new
salts; original R55 FIT model weights are reproduced and never updated on
new OOD. Goal proxy magnitude is not itself proof of goal causation.
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
from experiments import p85mr_f8r55_goal_mediated_interventions as r55

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r56_cold_delayed_proxy_shift_freeze.json"
FREEZE_BLOB="e7ca7b6f1d40fd8afe72735643e73cc4c6a1ee81"
SCHEMA="ttt-hybrid.f8r56-cold-delayed-proxy-shift-freeze.v1"
PARENT="540bc40b62ce852946402a179a91f9cbbe0ea0c0"
SALT={"control":23010000,"delayed_proxy_confounder":23020000}
PANELS=("control","delayed_proxy_confounder")
SEEDS=r55.SEEDS
ARMS=r55.ARMS
N_SOURCE=16
RECORDS_PER_ACTION=3
GROUPS=8
EXPECTED_WORLD_COUNT=64
B_RECORDS=96
UP_SPURIOUS=.45
IMMEDIATE_SPURIOUS=.52
DELAY_SPURIOUS=(.50,-.48)
CAUSAL=.12


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R56_FROZEN_BLOB_DRIFT")
    f=json.loads(raw)
    src,replay,decision=f["source"],f["replay"],f["frozen_decision"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R56_GENERATOR_OR_SCORED_RESULTS" or
        f["parent"]["qualified_R55_SHA"]!=PARENT or
        f["parent"]["r55_freeze_blob"]!=r55.FREEZE_BLOB or
        src["salts"]!=SALT or
        src["worlds_each"]!=EXPECTED_WORLD_COUNT or
        src["public_sources"]!=N_SOURCE or
        src["selected_upstream_actions_per_source"]!=2 or
        src["observed_records_per_selected_action"]!=RECORDS_PER_ACTION or
        src["selected_B_record_count"]!=B_RECORDS or
        src["full_preaction_record_length"]!=102 or
        src["causal_upstream_immediate_delayed_action_contrast"]!=CAUSAL or
        src["spurious_upstream_action_contrast"]!=UP_SPURIOUS or
        src["spurious_immediate_action_contrast"]!=IMMEDIATE_SPURIOUS or
        src["spurious_delayed_action_contrast_shift"]!=list(DELAY_SPURIOUS) or
        replay["original_seeds"]!=list(SEEDS) or
        replay["arms"]!=list(r55.ARM_NAMES) or
        replay["optimizer_updates_per_arm"]!=r55.UPDATES or
        replay["original_draw_seed"]!=r55.DRAW_SEED or
        replay["batch_pairs"]!=r55.BATCH or
        replay["total_replayed_optimizer_attempts"]!=len(ARMS)*r55.UPDATES or
        decision["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R56_FROZEN_PROTOCOL_OR_AUTHORITY_CHANGED")
    r55.locked()
    return f

def positions(panel,group):
    if (panel not in SALT or type(group) is not int
        or not 0<=group<GROUPS):
        raise ValueError("F8R56_UNREGISTERED_SHIFT_GROUP")
    order=sorted(range(N_SOURCE),key=lambda s:hashlib.sha256(
        f"F8-R56/{panel}/{SALT[panel]}/{group}/{s}".encode()).digest())
    return tuple(order[:4])


def world(panel,group,ctx,q_sign,gate):
    positions(panel,group)
    if ctx not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R56_INVALID_FACTOR")
    # The evaluator return physics uses exactly original R53 OOD.
    # r56_panel is for NEW source generation only, not an action/model input.
    return {"panel":"ood","r56_panel":panel,"group":group,
            "public_context":ctx,"q_evaluator_only":q_sign,
            "gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R56_UNREGISTERED_SHIFT_PANEL")
    return [world(panel,g,c,q,gate)
            for g in range(GROUPS)
            for c in (0,1)
            for q in (-1,1)
            for gate in (-1,1)]


def selected_history(w):
    """All six public factual records per source selected from reset state."""
    panel=w["r56_panel"]
    group,ctx,gate=w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,s1,s2=positions(panel,group)
    rows=[]
    for source in range(N_SOURCE):
        if source==cue:tag=.42 if ctx==0 else .18
        elif source==decoy:tag=.18 if ctx==0 else .42
        elif source==s1:tag=.88
        elif source==s2:tag=.77
        else:tag=.03 if source%3==0 else 0.
        for action in (0,1):
            sign=1. if action==0 else -1.
            if source==cue:
                upstream=immediate=delayed=CAUSAL*gate*sign
            elif source==decoy:
                upstream=immediate=delayed=-CAUSAL*gate*sign
            elif source in (s1,s2):
                direction=1. if source==s1 else -1.
                upstream=UP_SPURIOUS*direction*sign
                immediate=IMMEDIATE_SPURIOUS*direction*sign
                delayed=(0. if panel=="control" else
                         DELAY_SPURIOUS[0 if source==s1 else 1]*sign)
            else:
                nuisance=round(.009*math.sin(
                    (SALT[panel]+group)*.137+source*.271)+
                    .004*(((source+group)%4)-1.5),8)
                upstream=immediate=nuisance
                delayed=0.
            rows.extend(((tag,action,upstream),
                         (r55.NOW_STATE,action,immediate),
                         (r55.DELAY_STATE,action,delayed)))
    return rows

def observed_history(w,*,erase_B=False,erase_tag=False,
                     erase_fifth=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    rows=[r55.r53.r52.r51.r33.b._record(
        state,action,r55.r53.r52.r51.r33.r32.after(state,action,c,q),c)
        for state,action in r55.r53.r52.r51.r33.r32.EARLY]
    rows.append(r55.r53.r52.r51.r33.b._record(
        r55.PROBE,c,r55.r53.r52.r51.r33.r32.after(r55.PROBE,c,c,q),c))
    for idx,(state,action,delta) in enumerate(selected_history(w)):
        visible=0. if erase_tag and idx%3==0 else state
        key,effect=r55.r53.r52.r51.r33.b._record(
            visible,action,visible+delta,c)
        effect=effect.clone()
        effect[4]=0.
        rows.append((key,effect))
    actions=torch.stack([x for x,_ in rows])
    effects=torch.stack([e for _,e in rows])
    if erase_B:effects[5:].zero_()
    if erase_fifth:effects[4].zero_()
    return actions,effects

def candidates(w,**opts):
    a,e=observed_history(w,**opts)
    actions=[];effects=[]
    for choice in (0,1):
        key=r55.r53.r52.r51.r33.b._record(
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
                    raise ValueError("F8R56_GATE_ACTION_KEY_LEAK")
                difference=(e[:,5:-1].mean(1)-
                            ee[:,5:-1].mean(1)).abs().max().item()
                biggest=max(biggest,difference)
                if difference>1e-7:
                    raise ValueError("F8R56_FULL_MEAN_GATE_LEAK")
                if torch.equal(e[:,5:-1],ee[:,5:-1]):
                    raise ValueError("F8R56_GATE_SIGNAL_IDENTICAL")
    return {"panel":panel,"groups":GROUPS,
            "maximum_paired_full_B_mean_difference":biggest}


def public_tag_reference(offset):
    out={}
    for panel in PANELS:
        failures=0
        for w in worlds(panel):
            a,e=candidates(w)
            tag,up,immediate,delayed,fifth,ctx,obs=(
                r55.observed_goal_features(a,e,torch.tensor(offset)))
            mask=((tag-.42).abs()<.005)|((tag-.18).abs()<.005)
            if int(mask.sum())!=2:
                raise ValueError("F8R56_NO_PUBLIC_IDENTIFIABILITY")
            idx=torch.nonzero(mask[0],as_tuple=False).flatten()
            tags,delta=tag[0,idx],delayed[0,idx]
            role=int(tags.argmax() if w["public_context"]==0
                     else tags.argmin())
            pred=(w["public_context"] if float(fifth[0]*delta[role])>0
                  else 1-w["public_context"])
            failures+=int(r55.regret(w,pred)>1e-10)
        if failures:
            raise ValueError("F8R56_SOURCE_ONLY_REFERENCE_FAILED")
        out[panel]={"worlds":EXPECTED_WORLD_COUNT,"mistakes":failures,
                    "handcoded_FIT_known_tag_windows":True}
    return out

def original_draws():
    rng=torch.Generator(device="cpu")
    rng.manual_seed(r55.DRAW_SEED)
    return torch.randint(0,r55.TRAIN_PAIRS,
                         (r55.UPDATES,r55.BATCH),generator=rng).tolist()


def reproduce_original_FIT(arm,seed,offset,records,labels,draws,f):
    model=r55.Policy(arm,seed,offset)
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
        a,e,y=r55.batch(records,labels,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R56_NONFINITE_ORIGINAL_FIT_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in grads):
            raise ValueError("F8R56_NONFINITE_ORIGINAL_FIT_GRADIENT")
        productive+=int(any(g is not None and bool(torch.any(g!=0))
                           for g in grads))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
    target=f["replay"]["reproduction_targets"]
    if arm=="learned_repeatability":
        spec=target[str(seed)]
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
            or productive!=r55.UPDATES):
            raise ValueError("F8R56_ORIGINAL_R55_WEIGHT_REPRODUCTION_FAILED")
    heldout=r55.heldout(model,records,labels)
    original=r55.score(model,"ood")
    if (arm=="learned_repeatability" and
        (heldout!=1.0 or original["mean_regret"]!=0.)):
        raise ValueError("F8R56_ORIGINAL_R55_POLICY_NOT_REPRODUCED")
    return model,{"arm":arm,"seed":seed,
        "original_FIT_heldout":heldout,
        "original_R55_OOD_regret":original["mean_regret"],
        "original_R55_OOD_choices":original["choices"],
        "trainable_parameters":sum(p.numel()
                                   for p in model.parameters() if p.requires_grad),
        "original_FIT_replayed_optimizer_updates":len(draws),
        "productive_gradient_steps":productive,
        "replayed_beta":float(model.beta.detach())
           if hasattr(model,"beta") else None,
        "replayed_context_slopes":model.slopes.detach().tolist()
           if hasattr(model,"slopes") else None,
        "replayed_weight_hash":r55.r53.r52.r51.r40._hash(model),
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
        raise ValueError("F8R56_INVALID_COLD_POLICY_OUTPUT")
    choices=costs.argmin(-1).tolist()
    regrets=[r55.regret(w,action) for w,action in zip(ww,choices)]
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
    original_fit,labels=r55.factual_fit()
    offset=r55.calibration(original_fit)
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
        restored=r55.Policy(arm,seed,offset)
        restored.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
        if r55.r53.r52.r51.r40._hash(restored)!=base["replayed_weight_hash"]:
            raise ValueError("F8R56_COLD_RESTORE_WEIGHT_DRIFT")
        if any(score(restored,p)["choices"]!=panels[p]["active"]["choices"]
                for p in PANELS):
            raise ValueError("F8R56_COLD_RESTORE_ACTION_DRIFT")
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
                active["mean_regret"]<=frozen["maximum_each_panel_regret"]
                and all(v<=frozen["maximum_each_public_context_regret"]
                        for v in active["context_regrets"].values())
                and active["gate_pair_flips"]>=
                       frozen["minimum_each_panel_gate_reversals"])
        verdict.append({"seed":seed,"panels_pass":criteria,
            "full_cold_transfer_pass":all(criteria.values()),
            "oscillating_OOD":row["new_panels"]["oscillating"]["active"],
            "stationary_OOD":row["new_panels"]["stationary_confounder"]["active"],
            "original_R55_OOD_regret":row["original_R55_OOD_regret"]})
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
        "source_neutral_synthetic_cold_transfer_only":True,
        "original_R55_FIT_selected_factual_returns":len(original_fit),
        "original_R55_FIT_train_pairs":r55.TRAIN_PAIRS,
        "original_R55_FIT_heldout_pairs":r55.HELDOUT_PAIRS,
        "original_FIT_only_fifth_offset":offset,
        "new_panel_source_parity":parity,
        "positive_observation_reference":feasibility,
        "original_R55_FIT_replayed_optimizer_steps":len(ARMS)*r55.UPDATES,
        "original_R55_models":trained,
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
    path=ROOT/"local_runs/p85mr_f8r56_cold_delayed_proxy_shift_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R56_REFUSE_RESULT_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R56_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
