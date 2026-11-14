"""F8-R52: prospective *selected* action-pair contrast eligibility.

Source-neutral synthetic intervention data. Two actually selected action
consequences per public state are observed. No original external environment,
evaluator labels, unselected return effects or hidden source positions enter a
model. A fixed pairing/contrast feature is NOT learned causal event discovery.
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
from torch import nn
from torch.nn import functional as F

from experiments import p85mr_f8r51_learned_eligibility as r51

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r52_action_contrast_freeze.json"
FREEZE_BLOB="466819fe0780298279935c1e15fa06a04653ec36"
SCHEMA="ttt-hybrid.f8r52-selected-intervention-contrast-freeze.v1"
PARENT="84b3a78a661c0f747982a9e3d0ba232f41094807"
SALT={"fit":22610000,"cal":22620000,"ood":22630000}
PAIRS={"fit":12,"cal":16,"ood":20}
AMP={"fit":.24,"cal":.18,"ood":.12}
SPUR={"fit":(.30,-.27),"cal":(.34,-.29),"ood":(.38,-.32)}
NOISE={"fit":.006,"cal":.009,"ood":.014}
SEEDS=(985201,985207)
ARM_NAMES=("learned_action_contrast","raw_allpair_contrast",
           "tag_top2_contrast","raw_effect_magnitude_top2",
           "fixed_action_contrast_top2","native_action_contrast",
           "gru_public_interventions")
ARMS=tuple((name,seed) for name in ARM_NAMES for seed in SEEDS)
FIT_GROUPS,TRAIN_GROUPS,EVAL_GROUPS=24,16,8
TRAIN_PAIRS,HELDOUT_PAIRS,BATCH,UPDATES=128,64,16,256
DRAW_SEED,BOOT_SEED,BOOT_DRAWS=22649999,22669999,10000
CHECKPOINTS=(1,32,64,128,192,256)
PANELS=("cal","ood")
PROBE=.42


def git_blob(data):
    return hashlib.sha1(
        b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R52_PREREGISTERED_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    s,m,j=f["source"],f["methods"],f["frozen_score"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_GENERATOR_MODEL_OR_SCORED_RESULTS" or
        f["parent"]["qualified_R51_HEAD"]!=PARENT or
        f["parent"]["r51_freeze_blob"]!=r51.FREEZE_BLOB or
        s["new_salts"]!=SALT or
        s["paired_selected_intervention_sources_per_world"]!=PAIRS or
        s["causal_observed_action_consequence_delta_by_panel"]!=AMP or
        s["nuisance_action_invariant_delta_amplitude_by_panel"]!=NOISE or
        m["seeds"]!=list(SEEDS) or m["arms"]!=list(ARM_NAMES) or
        m["train_updates_per_arm"]!=UPDATES or
        m["sampler_seed"]!=DRAW_SEED or
        m["train_batch_pairs"]!=BATCH or
        m["total_training_optimizer_steps"]!=len(ARMS)*UPDATES or
        j["paired_bootstrap"]["seed"]!=BOOT_SEED or
        j["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R52_PROTOCOL_OR_AUTHORITY_DRIFT")
    r51.locked()
    return f


def positions(panel,group):
    if (panel not in SALT or type(group) is not int or
        not 0<=group<(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)):
        raise ValueError("F8R52_UNREGISTERED_GROUP")
    ordered=sorted(range(PAIRS[panel]),key=lambda n:hashlib.sha256(
        f"F8-R52/{panel}/{SALT[panel]}/{group}/{n}".encode()).digest())
    return tuple(ordered[:4])


def world(panel,group,ctx,q_sign,gate):
    positions(panel,group)
    if ctx not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R52_INVALID_FACTOR")
    return {"panel":panel,"group":group,"public_context":ctx,
            "q_evaluator_only":q_sign,"gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R52_UNREGISTERED_PANEL")
    n=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    return [world(panel,g,c,q,gate) for g in range(n)
            for c in (0,1) for q in (-1,1) for gate in (-1,1)]


def selected_B_interventions(w):
    """Two genuinely recorded action consequences per public state.

    The causal pair reverses response under action intervention. Distractors
    have higher raw physical magnitude but action-invariant effects.
    """
    p,g,ctx,gate=w["panel"],w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,high1,high2=positions(p,g)
    out=[]
    for pair in range(PAIRS[p]):
        if pair==cue:
            tag=.42 if ctx==0 else .18
            delta0,delta1=+AMP[p]*gate,-AMP[p]*gate
        elif pair==decoy:
            tag=.18 if ctx==0 else .42
            delta0,delta1=-AMP[p]*gate,+AMP[p]*gate
        elif pair in (high1,high2):
            ix=0 if pair==high1 else 1
            tag=(.88,.77)[ix]
            delta0=delta1=SPUR[p][ix]
        else:
            tag=.03 if pair%3==0 else 0.
            delta0=delta1=round(NOISE[p]*math.sin(
                (SALT[p]+g)*.137+pair*.271)+.004*(((pair+g)%4)-1.5),8)
        out.extend(((tag,0,delta0),(tag,1,delta1)))
    return out


def following(w,choice):
    s,c=w["return_state"],w["public_context"]
    q=w["q_evaluator_only"]*w["gate_evaluator_only"]
    return (r51.r41._law(s,choice,c,q,"ood")
            if w["panel"]=="ood" else r51.r33.r32.after(s,choice,c,q))


def observed_history(w,*,erase_B=False,erase_tag=False,erase_fifth=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    facts=[r51.r33.b._record(s,act,r51.r33.r32.after(s,act,c,q),c)
           for s,act in r51.r33.r32.EARLY]
    facts.append(r51.r33.b._record(
        PROBE,c,r51.r33.r32.after(PROBE,c,c,q),c))
    for tag,selected_action,delta in selected_B_interventions(w):
        visible=0. if erase_tag else tag
        key,effect=r51.r33.b._record(visible,selected_action,
                                     visible+delta,c)
        effect=effect.clone()
        effect[4]=0.
        facts.append((key,effect))
    actions=torch.stack([a for a,_ in facts])
    effects=torch.stack([e for _,e in facts])
    if erase_B:effects[5:].zero_()
    if erase_fifth:effects[4].zero_()
    return actions,effects


def candidates(w,**opts):
    a,e=observed_history(w,**opts)
    actions=[];effects=[]
    for choice in (0,1):
        key=r51.r33.b._record(
            w["return_state"],choice,w["return_state"],w["public_context"])[0]
        actions.append(torch.cat((a,key[None])))
        effects.append(torch.cat((e,torch.zeros((1,16),dtype=e.dtype))))
    return torch.stack(actions),torch.stack(effects)


def source_parity(panel):
    groups=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    worst=0.
    for g in range(groups):
        for ctx in (0,1):
            for q in (-1,1):
                a,e=candidates(world(panel,g,ctx,q,1))
                aa,ee=candidates(world(panel,g,ctx,q,-1))
                if not torch.equal(a,aa):
                    raise ValueError("F8R52_GATE_ACTION_KEY_LEAK")
                err=float((e[:,5:-1].mean(1)-
                           ee[:,5:-1].mean(1)).abs().max())
                worst=max(worst,err)
                if err>1e-7 or torch.equal(e[:,5:-1],ee[:,5:-1]):
                    raise ValueError("F8R52_GATE_MEAN_LEAK_OR_NO_CONSEQUENCE")
    return {"panel":panel,"groups":groups,"contexts":2,
            "max_paired_full_B_mean_difference":worst}


def factual_fit():
    locked()
    records=[];labels=[]
    for w in worlds("fit"):
        outcomes=[]
        for choice in (0,1):
            a,e=observed_history(w)
            after=following(w,choice)
            key,eff=r51.r33.b._record(
                w["return_state"],choice,after,w["public_context"])
            aa=torch.cat((a,key[None]))
            ee=torch.cat((e,eff[None]))
            recovered=w["return_state"]+r51.r31._physical_delta(
                ee[-1,0])
            outcomes.append((recovered-1.)**2)
            records.append((aa,ee))
        labels.append(0 if outcomes[0]<=outcomes[1] else 1)
    if len(records)!=384 or len(labels)!=192:
        raise ValueError("F8R52_BAD_FACTUAL_FIT_COHORT")
    return records,labels


def calibration(records):
    if len(records)!=384:
        raise ValueError("F8R52_UNQUALIFIED_FIT_CALIBRATION")
    after=[PROBE+r51.r31._physical_delta(records[2*i][1][4,0])
           for i in range(TRAIN_PAIRS)]
    offset=statistics.fmean(after)
    if not math.isfinite(offset):
        raise ValueError("F8R52_INVALID_CALIBRATION")
    return offset


def regret(w,choice):
    if choice not in (0,1):
        raise ValueError("F8R52_UNREGISTERED_RETURN_CHOICE")
    losses=[(following(w,a)-1.)**2 for a in (0,1)]
    return max(0.,losses[choice]-min(losses))


def public_pair_features(actions,effects,offset):
    if (actions.ndim!=3 or actions.shape!=effects.shape or
        actions.shape[-1]!=16 or actions.shape[1] not in (30,38,46)
        or actions.shape[0]%2):
        raise ValueError("F8R52_INVALID_SELECTED_PAIRED_HISTORY")
    obs=effects.clone()
    obs[:,-1,:].zero_()
    aa=actions[::2]
    e0=obs[::2,5:-1:2,0]
    e1=obs[::2,6:-1:2,0]
    action0=aa[:,5:-1:2]
    action1=aa[:,6:-1:2]
    t0=2.4*torch.atanh(
        (action0[:,:,1]/action0[:,:,0]).clamp(-.98,.98))
    t1=2.4*torch.atanh(
        (action1[:,:,1]/action1[:,:,0]).clamp(-.98,.98))
    if not bool(torch.allclose(t0,t1,atol=1e-6,rtol=0)):
        raise ValueError("F8R52_MISMATCHED_PUBLIC_PAIR")
    d0=2.4*torch.atanh(e0.clamp(-.98,.98))
    d1=2.4*torch.atanh(e1.clamp(-.98,.98))
    # Causal contrast is an observed, fixed feature transform; its
    # relevance and public tag role weights are trained and ablated.
    contrast=(d0-d1)*.5
    magnitude=(d0.abs()+d1.abs())*.5
    fifth=PROBE+2.4*torch.atanh(
        obs[::2,4,0].clamp(-.98,.98))-offset
    ctx=(3.*torch.atanh(
        (aa[:,-1,2]/aa[:,-1,0]).clamp(-.98,.98))
        ).round().long()
    return t0,contrast,magnitude,d0,d1,fifth,ctx,obs


def positive_observation_reference(offset):
    counts={}
    for panel in ("fit","cal","ood"):
        failures=0
        cohort=worlds(panel)
        for w in cohort:
            acts,effects=candidates(w)
            tag,diff,mag,d0,d1,fifth,ctx,obs=public_pair_features(
                acts,effects,torch.tensor(offset))
            selected=torch.argsort(diff.abs(),dim=1,
                                   descending=True,stable=True)[:,:2]
            ts=torch.gather(tag,1,selected)[0].tolist()
            ds=torch.gather(diff,1,selected)[0].tolist()
            # Upper bound on public identifiability, not a learned selector.
            if (min(abs(x) for x in ds)<.10 or
                sum(float(abs(x)>.05) for x in diff[0])!=2):
                raise ValueError("F8R52_PUBLIC_PAIR_IDENTIFIABILITY_FAILED")
            idx=(ts.index(max(ts)) if w["public_context"]==0
                 else ts.index(min(ts)))
            pred=w["public_context"] if float(fifth[0])*ds[idx]>0 else (
                1-w["public_context"])
            failures+=int(regret(w,pred)>1e-10)
        if failures:
            raise ValueError("F8R52_POSITIVE_CONTROL_FAILED")
        counts[panel]={"worlds":len(cohort),"mistakes":failures}
    return counts


class R52FullHistoryFactualMemory(r51.r34.FactualMemory):
    """Local adapter only, all selected records retained, no shared mutation."""
    def forward(self,actions,effects,*,ablate_backbone=False):
        if (actions.ndim!=3 or effects.shape!=actions.shape
            or actions.shape[-1]!=16 or actions.shape[1] not in (30,38,46)):
            raise ValueError("F8R52_UNREGISTERED_NATIVE_HISTORY")
        rec=torch.cat((actions[:,:-1],effects[:,:-1]),dim=-1)
        _,h=self.memory(rec)
        if self.backbone is None or ablate_backbone:
            native=torch.zeros((actions.shape[0],1),
                dtype=actions.dtype,device=actions.device)
        else:
            pred=r51.r34.b._predict(self.base_arm,self.backbone,
                {"action":actions,"effect":effects})
            if pred.shape!=effects.shape:
                raise ValueError("F8R52_INVALID_NATIVE_PREDICTION")
            native=pred[:,-1,0:1]
        return self.readout(torch.cat((h[-1],actions[:,-1],native),dim=-1)
                            ).squeeze(-1)


class Policy(nn.Module):
    def __init__(self,arm,seed,offset):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R52_INVALID_ARM")
        self.arm,self.seed=arm,seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm!="gru_public_interventions":
            self.slopes=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        if arm in ("learned_action_contrast","native_action_contrast"):
            self.salience=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.gain=nn.Parameter(torch.tensor(3.+(seed-985201)*.01))
        if arm=="native_action_contrast":
            self.native=R52FullHistoryFactualMemory(
                "native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.))
        if arm=="gru_public_interventions":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22680000+seed)
                self.gru=nn.GRU(4,24,batch_first=True)
                self.readout=nn.Linear(24,1)

    def forward(self,a,e,*,erase_salience=False,force_tag_top2=False,
                force_raw_magnitude_top2=False,force_uniform=False,
                zero_native=False,zero_contrast=False):
        tag,contrast,magnitude,d0,d1,fifth,ctx,obs=public_pair_features(
            a,e,self.fit_offset)
        if self.arm=="gru_public_interventions":
            x=torch.stack((tag,d0,d1,
                           ctx[:,None].float().expand_as(tag)),dim=-1)
            _,h=self.gru(x)
            signal=torch.tanh(self.readout(h[-1]).squeeze(-1))
        else:
            if (self.arm=="tag_top2_contrast" or force_tag_top2):
                ids=r51.r49.stable_public_top_two(tag)
            elif (self.arm=="raw_effect_magnitude_top2" or
                  force_raw_magnitude_top2):
                ids=torch.argsort(magnitude,descending=True,
                                  stable=True,dim=-1)[:,:2]
            elif self.arm=="fixed_action_contrast_top2":
                ids=torch.argsort(contrast.abs(),descending=True,
                                  stable=True,dim=-1)[:,:2]
            else:
                ids=None
            if ids is None:
                active_tag,active_contrast=tag,contrast
            else:
                active_tag=torch.gather(tag,1,ids)
                active_contrast=torch.gather(contrast,1,ids)
            logits=self.slopes[ctx,None]*active_tag
            if (self.arm in ("learned_action_contrast","native_action_contrast")
                and not erase_salience and ids is None):
                logits=logits+self.salience*12.*active_contrast.abs()
            weights=(torch.full_like(logits,1./logits.shape[1])
                     if force_uniform else
                     torch.softmax(logits,dim=-1))
            signal=(weights*active_contrast).sum(-1)
        if zero_contrast:
            signal=torch.zeros_like(signal)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*signal)
        if self.arm=="native_action_contrast":
            native=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref=torch.tanh(torch.atanh(pref.clamp(-.98,.98))+
                            torch.tanh(self.native_gain)*
                            (native[:,1]-native[:,0]))
        return r51.r47._choice_cost(ctx,pref)


def batch(records,labels,indices):
    if (len(indices)!=BATCH or
        any(type(i) is not int or not 0<=i<TRAIN_PAIRS for i in indices)):
        raise ValueError("F8R52_BAD_TRAIN_DRAW")
    rows=[records[2*i+j] for i in indices for j in (0,1)]
    return (torch.stack([x[0] for x in rows]),
            torch.stack([x[1] for x in rows]),
            torch.tensor([labels[i] for i in indices],dtype=torch.long))


def heldout(model,records,labels):
    model.eval()
    correct=0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            subset=list(range(start,start+BATCH))
            rows=[records[2*i+j] for i in subset for j in (0,1)]
            a=torch.stack([x[0] for x in rows])
            e=torch.stack([x[1] for x in rows])
            y=torch.tensor([labels[i] for i in subset])
            correct+=int((model(a,e).argmin(-1)==y).sum())
    return correct/HELDOUT_PAIRS


def score(model,panel,*,erase_B=False,erase_tag=False,erase_fifth=False,
          erase_salience=False,force_tag_top2=False,
          force_raw_magnitude_top2=False,force_uniform=False,
          zero_native=False,zero_contrast=False):
    if panel not in PANELS:
        raise ValueError("F8R52_INVALID_SCORE_PANEL")
    cohort=worlds(panel)
    raw=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                    erase_fifth=erase_fifth) for w in cohort]
    aa=torch.cat([x[0] for x in raw])
    ee=torch.cat([x[1] for x in raw])
    model.eval()
    with torch.no_grad():
        costs=model(aa,ee,erase_salience=erase_salience,
                    force_tag_top2=force_tag_top2,
                    force_raw_magnitude_top2=force_raw_magnitude_top2,
                    force_uniform=force_uniform,zero_native=zero_native,
                    zero_contrast=zero_contrast)
    if costs.shape!=(len(cohort),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R52_INVALID_POLICY_OUTPUT")
    choices=costs.argmin(-1).tolist()
    regrets=[regret(w,c) for w,c in zip(cohort,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "context_regrets":{
                str(ctx):statistics.fmean(regrets[i] for i,w in enumerate(cohort)
                if w["public_context"]==ctx) for ctx in (0,1)},
            "per_group_regret":[statistics.fmean(regrets[g*8:(g+1)*8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(choices),2))}


def train_one(arm,seed,offset,records,labels,draws):
    model=Policy(arm,seed,offset)
    params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            lr=.02 if name in ("slopes","salience","gain","native_gain") else .001
            params.append({"params":[p],"lr":lr})
    optimizer=torch.optim.AdamW(params,weight_decay=0.)
    productive=0
    slope_grads=[False,False]
    salience_grads=False
    trace=[]
    start=time.perf_counter()
    for update,indices in enumerate(draws,1):
        a,e,y=batch(records,labels,indices)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R52_NONFINITE_TRAINING_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in gradients):
            raise ValueError("F8R52_NONFINITE_GRADIENT")
        productive+=int(any(g is not None and bool(torch.any(g!=0))
                            for g in gradients))
        if arm=="learned_action_contrast":
            salience_grads |= (model.salience.grad is not None and
                bool(model.salience.grad!=0))
            for i in (0,1):
                slope_grads[i] |= (
                    model.slopes.grad is not None and
                    bool(model.slopes.grad[i]!=0))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if update in CHECKPOINTS:
            trace.append({"step":update,"heldout_accuracy":heldout(
                model,records,labels),"loss":float(loss.detach()),
                "slopes":model.slopes.detach().tolist()
                  if hasattr(model,"slopes") else None,
                "salience":float(model.salience.detach())
                  if hasattr(model,"salience") else None})
    if arm=="learned_action_contrast" and (
        productive==0 or not salience_grads or not all(slope_grads)):
        raise ValueError("F8R52_MAIN_MISSING_LEARNED_INTERVENTION_GRADIENT")
    panel_scores={p:{"active":score(model,p)} for p in PANELS}
    for title,kwargs in [
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("zero_salience",{"erase_salience":True}),
        ("force_tag_top2",{"force_tag_top2":True}),
        ("force_raw_magnitude_top2",{"force_raw_magnitude_top2":True}),
        ("force_uniform",{"force_uniform":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_contrast",{"zero_contrast":True})]:
        panel_scores["ood"][title]=score(model,"ood",**kwargs)
    cold=Policy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r51.r40._hash(cold)!=r51.r40._hash(model) or
        any(score(cold,p)["choices"]!=panel_scores[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R52_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"actual_updates":UPDATES,
            "productive_gradient_steps":productive,
            "salience_received_real_gradient":salience_grads
              if arm=="learned_action_contrast" else None,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                      if p.requires_grad),
            "train_seconds":time.perf_counter()-start,
            "traces":trace,"scores":panel_scores,"cold_restore":True}


def bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED)
    vals=[]
    for _ in range(BOOT_DRAWS):
        draws=[rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        vals.append(statistics.fmean(b[i]-a[i] for i in draws))
    return sorted(vals)[int(.025*BOOT_DRAWS)]


def judge(models,f):
    limits=f["frozen_score"]
    by={(row["arm"],row["seed"]):row for row in models}
    out=[]
    for seed in SEEDS:
        main=by["learned_action_contrast",seed]
        raw=by["raw_allpair_contrast",seed]
        heuristic=by["fixed_action_contrast_top2",seed]
        active=main["scores"]["ood"]["active"]
        cal=main["scores"]["cal"]["active"]
        penalties={key:main["scores"]["ood"][key]["mean_regret"]-
                   active["mean_regret"] for key in (
                "erase_B","erase_tag","erase_fifth","zero_salience",
                "force_tag_top2","force_raw_magnitude_top2","force_uniform",
                "zero_native","zero_contrast")}
        raw_ood=raw["scores"]["ood"]["active"]
        gain=raw_ood["mean_regret"]-active["mean_regret"]
        lower=bootstrap_lower(active["per_group_regret"],
                              raw_ood["per_group_regret"])
        slopes=main["traces"][-1]["slopes"]
        salience=main["traces"][-1]["salience"]
        passed=bool(
            main["traces"][-1]["heldout_accuracy"]>=
                 limits["min_heldout_FIT_accuracy"]
            and cal["mean_regret"]<=limits["max_CAL_mean_regret"]
            and active["mean_regret"]<=limits["max_OOD_mean_regret"]
            and all(v<=limits["max_OOD_each_context_regret"]
                    for v in active["context_regrets"].values())
            and cal["gate_pair_flips"]>=limits["min_CAL_gate_flips"]
            and active["gate_pair_flips"]>=limits["min_OOD_gate_flips"]
            and penalties["erase_B"]>=limits["min_OOD_erase_B_regret_penalty"]
            and penalties["erase_fifth"]>=
                limits["min_OOD_erase_fifth_regret_penalty"]
            and penalties["zero_salience"]>=
                limits["min_OOD_zero_salience_penalty"]
            and penalties["force_tag_top2"]>=
                limits["min_OOD_force_tag_top2_penalty"]
            and penalties["force_raw_magnitude_top2"]>=
                limits["min_OOD_force_raw_magnitude_top2_penalty"]
            and gain>=limits["min_gain_over_raw_allpair_contrast"]
            and lower>0 and slopes[0]>=limits["min_context0_slope"]
            and slopes[1]<=limits["max_context1_slope"] and salience>0
            and main["salience_received_real_gradient"])
        out.append({"seed":seed,"frozen_DEV_pass":passed,
            "heldout_FIT":main["traces"][-1]["heldout_accuracy"],
            "CAL_regret":cal["mean_regret"],"OOD_regret":active["mean_regret"],
            "OOD_gate_flips":active["gate_pair_flips"],
            "OOD_context_regret":active["context_regrets"],
            "learned_slopes":slopes,"learned_action_contrast_salience":salience,
            "OOD_ablation_penalties":penalties,
            "OOD_gain_vs_raw_allpair":gain,"bootstrap_lower_gain":lower,
            "fixed_contrast_heuristic_OOD_regret":
                heuristic["scores"]["ood"]["active"]["mean_regret"]})
    return {"two_seed_frozen_DEV_pass":all(v["frozen_DEV_pass"] for v in out),
            "seeds":out,"independent_scientific_pass":False,
            "Native_specific_gain_verified":False,
            "whole_process_capacity_compute_matched":False}


def run():
    freeze=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity={p:source_parity(p) for p in ("fit",)+PANELS}
    records,labels=factual_fit()
    offset=calibration(records)
    observable=positive_observation_reference(offset)
    rng=torch.Generator(device="cpu")
    rng.manual_seed(DRAW_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(UPDATES,BATCH),generator=rng).tolist()
    models=[train_one(name,seed,offset,records,labels,draws)
            for name,seed in ARMS]
    result=judge(models,freeze)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "source_neutral_synthetic_DEV_only":True,
            "selected_FIT_factual_return_effects":len(records),
            "FIT_training_pairs":TRAIN_PAIRS,"FIT_heldout_pairs":HELDOUT_PAIRS,
            "FIT_only_selected_fifth_offset":offset,
            "source_parity":parity,"observable_positive_reference":observable,
            "neural_optimizer_updates":len(ARMS)*UPDATES,
            "models":models,"judgment":result,
            "independent_scientific_pass":False,
            "external_original_first_use":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r52_action_contrast_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R52_REFUSE_RESULT_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,
                               allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R52_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
