"""F8-R55: learn from selected upstream interventions and observed downstream goal proxy.

All upstream actions are actually selected. Both immediate and delayed target
consequences are actually observed; the future return action is never observed
by the learner. A transparent sensor grouping and time ordering is PROVIDED,
not a claim of autonomous causal structure discovery. New scientific criteria
were frozen in a separate commit BEFORE this generator/model existed.
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
from experiments import p85mr_f8r53_repeated_intervention_stability as r53

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r55_goal_mediated_interventions_freeze.json"
FREEZE_BLOB="57cbbada3b9e98e512149b508fa9313563a44ad6"
SCHEMA="ttt-hybrid.f8r55-goal-mediated-selected-interventions-freeze.v1"
PARENT="174e8dc796f192e2f5517c0b445c545574af74fe"
SALT={"fit":22910000,"cal":22920000,"ood":22930000}
NSOURCE={"fit":8,"cal":10,"ood":16}
AMP={"fit":.24,"cal":.18,"ood":.12}
UP_SPUR={"fit":.56,"cal":.50,"ood":.45}
NOW_SPUR={"fit":.62,"cal":.57,"ood":.52}
SEEDS=(985201,985207)
ARM_NAMES=("learned_goal_mediation","raw_delayed_context",
           "public_tag_top2","upstream_magnitude_top2",
           "immediate_magnitude_top2","fixed_delayed_goal_top2",
           "native_goal_mediation","gru_goal_context")
ARMS=tuple((arm,seed) for arm in ARM_NAMES for seed in SEEDS)
FIT_GROUPS,TRAIN_GROUPS,EVAL_GROUPS=24,16,8
TRAIN_PAIRS,HELDOUT_PAIRS,BATCH,UPDATES=128,64,16,256
DRAW_SEED,BOOT_SEED,BOOT_DRAWS=22949999,22969999,10000
CHECKPOINTS=(1,32,64,128,192,256)
PANELS=("cal","ood")
PROBE=.42
NOW_STATE=.55
DELAY_STATE=.73


def git_blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R55_FROZEN_SOURCE_BLOB_DRIFT")
    f=json.loads(raw)
    s,t,d=f["source"],f["training"],f["frozen_score"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_NEW_GENERATOR_MODEL_OR_SCORE" or
        f["parent"]["r54_head"]!=PARENT or
        f["parent"]["r54_freeze_blob"]!="e88e70a4945dff1021873d9f91d44be7b4488861" or
        s["salts"]!=SALT or
        s["public_sources"]!=NSOURCE or
        s["causal_delayed_goal_proxy_contrast"]!=AMP or
        s["distractor_upstream_contrast"]!=UP_SPUR or
        s["distractor_immediate_target_contrast"]!=NOW_SPUR or
        s["full_candidate_events"]!={"fit":54,"cal":66,"ood":102} or
        t["seeds"]!=list(SEEDS) or t["arms"]!=list(ARM_NAMES) or
        t["updates_per_arm"]!=UPDATES or
        t["batch_pairs"]!=BATCH or
        t["draw_seed"]!=DRAW_SEED or
        t["optimizer_attempts_total"]!=len(ARMS)*UPDATES or
        d["group_bootstrap"]["seed"]!=BOOT_SEED or
        d["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R55_FROZEN_PROTOCOL_OR_AUTHORITY_CHANGED")
    r53.locked()
    return f


def positions(panel,group):
    if (panel not in SALT or type(group) is not int or
        not 0<=group<(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)):
        raise ValueError("F8R55_UNREGISTERED_GROUP")
    ranked=sorted(range(NSOURCE[panel]),key=lambda x:hashlib.sha256(
        f"F8-R55/{panel}/{SALT[panel]}/{group}/{x}".encode()).digest())
    return tuple(ranked[:4])


def world(panel,group,context,q_sign,gate):
    positions(panel,group)
    if context not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R55_INVALID_WORLD_FACTOR")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":q_sign,"gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R55_UNREGISTERED_PANEL")
    count=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    return [world(panel,g,c,q,gate) for g in range(count)
            for c in (0,1) for q in (-1,1) for gate in (-1,1)]


def selected_upstream_downstream(w):
    """6 actual factual records per source:
       source-action0, immediate-target-action0, delayed-target-action0;
       source-action1, immediate-target-action1, delayed-target-action1.
    Each action is selected from a fresh same-state reset, never an unselected
    future RETURN action. Target sensor/time grouping is public and fixed.
    """
    p,g,c,gate=w["panel"],w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,sp1,sp2=positions(p,g)
    rows=[]
    for source in range(NSOURCE[p]):
        if source==cue:tag=.42 if c==0 else .18
        elif source==decoy:tag=.18 if c==0 else .42
        elif source==sp1:tag=.88
        elif source==sp2:tag=.77
        else:tag=.03 if source%3==0 else 0.
        for action in (0,1):
            sign=1. if action==0 else -1.
            if source==cue:
                upstream=delayed=AMP[p]*gate*sign
                immediate=AMP[p]*gate*sign
            elif source==decoy:
                upstream=delayed=-AMP[p]*gate*sign
                immediate=-AMP[p]*gate*sign
            elif source in (sp1,sp2):
                orientation=1. if source==sp1 else -1.
                upstream=UP_SPUR[p]*orientation*sign
                immediate=NOW_SPUR[p]*orientation*sign
                delayed=0.
            else:
                offset=round(.009*math.sin(
                    (SALT[p]+g)*.137+source*.271)+
                    .004*(((source+g)%4)-1.5),8)
                upstream=immediate=offset
                delayed=0.
            rows.extend((
                (tag,action,upstream),
                (NOW_STATE,action,immediate),
                (DELAY_STATE,action,delayed)))
    return rows


def following(w,action):
    return r53.following(w,action)


def observed_history(w,*,erase_B=False,erase_tag=False,
                     erase_fifth=False,erase_immediate=False,
                     erase_delayed=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    rows=[r53.r52.r51.r33.b._record(
        s,action,r53.r52.r51.r33.r32.after(s,action,c,q),c)
        for s,action in r53.r52.r51.r33.r32.EARLY]
    rows.append(r53.r52.r51.r33.b._record(
        PROBE,c,r53.r52.r51.r33.r32.after(PROBE,c,c,q),c))
    for idx,(state,action,delta) in enumerate(selected_upstream_downstream(w)):
        visible=(0. if erase_tag and idx%3==0 else state)
        key,effect=r53.r52.r51.r33.b._record(
            visible,action,visible+delta,c)
        effect=effect.clone()
        effect[4]=0.
        if (erase_immediate and idx%3==1 or
            erase_delayed and idx%3==2):
            effect.zero_()
        rows.append((key,effect))
    aa=torch.stack([x[0] for x in rows])
    ee=torch.stack([x[1] for x in rows])
    if erase_B:ee[5:].zero_()
    if erase_fifth:ee[4].zero_()
    return aa,ee


def candidates(w,**opts):
    a,e=observed_history(w,**opts)
    aa=[];ee=[]
    for action in (0,1):
        key=r53.r52.r51.r33.b._record(
            w["return_state"],action,w["return_state"],
            w["public_context"])[0]
        aa.append(torch.cat((a,key[None])))
        ee.append(torch.cat((e,torch.zeros((1,16),dtype=e.dtype))))
    return torch.stack(aa),torch.stack(ee)


def source_parity(panel):
    largest=0.
    groups=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    for g in range(groups):
        for c in (0,1):
            for q in (-1,1):
                a,e=candidates(world(panel,g,c,q,1))
                aa,ee=candidates(world(panel,g,c,q,-1))
                if not torch.equal(a,aa):
                    raise ValueError("F8R55_HIDDEN_GATE_PUBLIC_ACTION_KEY_LEAK")
                diff=float((e[:,5:-1].mean(1)-
                            ee[:,5:-1].mean(1)).abs().max())
                largest=max(largest,diff)
                if diff>1e-7 or torch.equal(e[:,5:-1],ee[:,5:-1]):
                    raise ValueError("F8R55_GATE_MEAN_LEAK_OR_NO_FACTUAL_SIGNAL")
    return {"panel":panel,"groups":groups,
            "max_paired_full_B_mean_difference":largest}


def factual_fit():
    locked()
    records=[];labels=[]
    for w in worlds("fit"):
        losses=[]
        for action in (0,1):
            a,e=observed_history(w)
            y=following(w,action)
            key,eff=r53.r52.r51.r33.b._record(
                w["return_state"],action,y,w["public_context"])
            aa=torch.cat((a,key[None]))
            ee=torch.cat((e,eff[None]))
            reconstructed=w["return_state"]+r53.r52.r51.r31._physical_delta(
                ee[-1,0])
            losses.append((reconstructed-1.)**2)
            records.append((aa,ee))
        labels.append(0 if losses[0]<=losses[1] else 1)
    if len(records)!=384 or len(labels)!=192:
        raise ValueError("F8R55_UNQUALIFIED_FACTUAL_FIT")
    return records,labels


def calibration(records):
    if len(records)!=384:
        raise ValueError("F8R55_BAD_FIT_CALIBRATION")
    measures=[PROBE+r53.r52.r51.r31._physical_delta(
        records[2*i][1][4,0]) for i in range(TRAIN_PAIRS)]
    offset=statistics.fmean(measures)
    if not math.isfinite(offset):
        raise ValueError("F8R55_INVALID_FIT_ONLY_CENTER")
    return offset


def regret(w,choice):
    if choice not in (0,1):
        raise ValueError("F8R55_INVALID_RETURN_CHOICE")
    errors=[(following(w,a)-1.)**2 for a in (0,1)]
    return max(0.,errors[choice]-min(errors))


def observed_goal_features(a,e,offset):
    if (a.ndim!=3 or e.shape!=a.shape or a.shape[-1]!=16 or
        a.shape[1] not in (54,66,102) or a.shape[0]%2):
        raise ValueError("F8R55_INVALID_SELECTED_GOAL_HISTORY")
    observed=e.clone()
    observed[:,-1].zero_()  # No hypothetical future return consequence.
    batch=a.shape[0]//2
    n={54:8,66:10,102:16}[a.shape[1]]
    keys=a[::2,5:-1].reshape(batch,n,2,3,16)
    vals=observed[::2,5:-1,0].reshape(batch,n,2,3)
    state=2.4*torch.atanh(
        (keys[:,:,:,0,1]/keys[:,:,:,0,0]).clamp(-.98,.98))
    if not bool(torch.allclose(state[:,:,0],state[:,:,1],
                               atol=1e-6,rtol=0)):
        raise ValueError("F8R55_DIFFERENT_PUBLIC_SOURCE_STATES_FOR_ACTIONS")
    tag=state[:,:,0]
    now_state=2.4*torch.atanh(
        (keys[:,:,:,1,1]/keys[:,:,:,1,0]).clamp(-.98,.98))
    delayed_state=2.4*torch.atanh(
        (keys[:,:,:,2,1]/keys[:,:,:,2,0]).clamp(-.98,.98))
    if (not bool(torch.allclose(
          now_state,torch.full_like(now_state,NOW_STATE),atol=1e-6,rtol=0))
        or not bool(torch.allclose(
          delayed_state,torch.full_like(delayed_state,DELAY_STATE),
          atol=1e-6,rtol=0))):
        raise ValueError("F8R55_TARGET_SENSOR_PUBLIC_EVENT_ORDER_MISMATCH")
    effects=2.4*torch.atanh(vals.clamp(-.98,.98))
    upstream=(effects[:,:,0,0]-effects[:,:,1,0])*.5
    immediate=(effects[:,:,0,1]-effects[:,:,1,1])*.5
    delayed=(effects[:,:,0,2]-effects[:,:,1,2])*.5
    fifth=PROBE+2.4*torch.atanh(
        observed[::2,4,0].clamp(-.98,.98))-offset
    ctx=(3.*torch.atanh(
        (a[::2,-1,2]/a[::2,-1,0]).clamp(-.98,.98))).round().long()
    return tag,upstream,immediate,delayed,fifth,ctx,observed


def positive_reference(offset):
    out={}
    for panel in ("fit",)+PANELS:
        mistakes=0
        for w in worlds(panel):
            a,e=candidates(w)
            tag,u,now,delay,fifth,ctx,obs=observed_goal_features(
                a,e,torch.tensor(offset))
            top=torch.argsort(delay.abs(),descending=True,
                               stable=True,dim=-1)[:,:2]
            top_tag=torch.gather(tag,1,top)[0].tolist()
            top_delay=torch.gather(delay,1,top)[0].tolist()
            if (len(top_tag)!=2 or
                min(abs(x) for x in top_delay)<.1):
                raise ValueError("F8R55_SELECTED_TARGET_PROBE_NOT_IDENTIFIABLE")
            ix=top_tag.index(max(top_tag)) if w["public_context"]==0 else (
                top_tag.index(min(top_tag)))
            chosen=w["public_context"] if float(fifth[0])*top_delay[ix]>0 else (
                1-w["public_context"])
            mistakes+=int(regret(w,chosen)>1e-10)
        out[panel]={"worlds":len(worlds(panel)),"mistakes":mistakes}
        if mistakes:
            raise ValueError("F8R55_OBSERVED_ONLY_REFERENCE_FAILED")
    return out


class Policy(nn.Module):
    def __init__(self,arm,seed,offset):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R55_UNREGISTERED_ARM_SEED")
        self.arm,self.seed=arm,seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm!="gru_goal_context":
            self.slopes=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        if arm in ("learned_goal_mediation","native_goal_mediation"):
            self.beta=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.gain=nn.Parameter(torch.tensor(3.+(seed-985201)*.01))
        if arm=="native_goal_mediation":
            self.native=r53.LongNative("native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.))
        if arm=="gru_goal_context":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22980000+seed)
                self.gru=nn.GRU(5,24,batch_first=True)
                self.readout=nn.Linear(24,1)

    def forward(self,a,e,*,zero_beta=False,force_tag_top2=False,
                force_upstream_top2=False,force_immediate_top2=False,
                zero_native=False,zero_signal=False):
        tag,up,now,delay,fifth,ctx,obs=observed_goal_features(
            a,e,self.fit_offset)
        if self.arm=="gru_goal_context":
            x=torch.stack((tag,up,now,delay,
                ctx[:,None].float().expand_as(tag)),dim=-1)
            _,h=self.gru(x)
            signal=torch.tanh(self.readout(h[-1]).squeeze(-1))
        else:
            if self.arm=="public_tag_top2" or force_tag_top2:
                ids=r53.r52.r51.r49.stable_public_top_two(tag)
            elif self.arm=="upstream_magnitude_top2" or force_upstream_top2:
                ids=torch.argsort(up.abs(),descending=True,
                                  stable=True,dim=-1)[:,:2]
            elif self.arm=="immediate_magnitude_top2" or force_immediate_top2:
                ids=torch.argsort(now.abs(),descending=True,
                                  stable=True,dim=-1)[:,:2]
            elif self.arm=="fixed_delayed_goal_top2":
                ids=torch.argsort(delay.abs(),descending=True,
                                  stable=True,dim=-1)[:,:2]
            else:
                ids=None
            if ids is None:
                selected_tag,selected_delayed=tag,delay
            else:
                selected_tag=torch.gather(tag,1,ids)
                selected_delayed=torch.gather(delay,1,ids)
            logits=self.slopes[ctx,None]*selected_tag
            if (self.arm in ("learned_goal_mediation","native_goal_mediation")
                and not zero_beta and ids is None):
                logits=logits+self.beta*12.*selected_delayed.abs()
            weights=torch.softmax(logits,dim=-1)
            signal=(weights*selected_delayed).sum(-1)
            if self.arm in ("learned_goal_mediation","native_goal_mediation"):
                signal=torch.where((tag==0).all(-1),
                    torch.zeros_like(signal),signal)
        if zero_signal:
            signal=torch.zeros_like(signal)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*signal)
        if self.arm=="native_goal_mediation":
            native=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref=torch.tanh(torch.atanh(pref.clamp(-.98,.98))+
                torch.tanh(self.native_gain)*(native[:,1]-native[:,0]))
        return r53.r52.r51.r47._choice_cost(ctx,pref)


def batch(facts,labels,ids):
    if (len(ids)!=BATCH or any(type(i) is not int or
                              not 0<=i<TRAIN_PAIRS for i in ids)):
        raise ValueError("F8R55_INVALID_FIT_DRAW")
    records=[facts[2*i+j] for i in ids for j in (0,1)]
    return (torch.stack([x[0] for x in records]),
            torch.stack([x[1] for x in records]),
            torch.tensor([labels[i] for i in ids],dtype=torch.long))


def heldout(model,facts,labels):
    model.eval()
    correct=0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            ids=list(range(start,start+BATCH))
            rows=[facts[2*i+j] for i in ids for j in (0,1)]
            a=torch.stack([x[0] for x in rows])
            e=torch.stack([x[1] for x in rows])
            y=torch.tensor([labels[i] for i in ids])
            correct+=int((model(a,e).argmin(-1)==y).sum())
    return correct/HELDOUT_PAIRS


def score(model,panel,*,erase_B=False,erase_tag=False,erase_fifth=False,
          erase_immediate=False,erase_delayed=False,
          zero_beta=False,force_tag_top2=False,
          force_upstream_top2=False,force_immediate_top2=False,
          zero_native=False,zero_signal=False):
    cohort=worlds(panel)
    inputs=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
        erase_fifth=erase_fifth,erase_immediate=erase_immediate,
        erase_delayed=erase_delayed) for w in cohort]
    a=torch.cat([x[0] for x in inputs])
    e=torch.cat([x[1] for x in inputs])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,zero_beta=zero_beta,
            force_tag_top2=force_tag_top2,
            force_upstream_top2=force_upstream_top2,
            force_immediate_top2=force_immediate_top2,
            zero_native=zero_native,zero_signal=zero_signal)
    if costs.shape!=(len(cohort),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R55_INVALID_PREACTION_COST")
    choices=costs.argmin(-1).tolist()
    regrets=[regret(w,c) for w,c in zip(cohort,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "context_regrets":{str(c):statistics.fmean(
                regrets[i] for i,w in enumerate(cohort)
                if w["public_context"]==c) for c in (0,1)},
            "per_group_regret":[statistics.fmean(
                regrets[g*8:g*8+8]) for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(choices),2))}


def train_one(arm,seed,offset,facts,labels,draws):
    model=Policy(arm,seed,offset)
    groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            groups.append({"params":[p],
                "lr":.02 if name in ("slopes","beta","gain","native_gain")
                     else .001})
    opt=torch.optim.AdamW(groups,weight_decay=0.)
    productive=0
    beta_gradient=False
    slopes_gradient=[False,False]
    trace=[]
    began=time.perf_counter()
    for step,ids in enumerate(draws,1):
        a,e,y=batch(facts,labels,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R55_INVALID_FIT_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in gradients):
            raise ValueError("F8R55_NONFINITE_GRADIENT")
        productive+=int(any(g is not None and bool(torch.any(g!=0))
                            for g in gradients))
        if arm=="learned_goal_mediation":
            beta_gradient|=(model.beta.grad is not None and
                             bool(model.beta.grad!=0))
            for c in (0,1):
                slopes_gradient[c]|=(
                    model.slopes.grad is not None and
                    bool(model.slopes.grad[c]!=0))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
        if step in CHECKPOINTS:
            trace.append({"step":step,"fit_loss":float(loss.detach()),
                          "heldout_fit":heldout(model,facts,labels),
                          "slopes":model.slopes.detach().tolist()
                           if hasattr(model,"slopes") else None,
                          "beta":float(model.beta.detach())
                           if hasattr(model,"beta") else None})
    if arm=="learned_goal_mediation" and (
        productive==0 or not beta_gradient or not all(slopes_gradient)):
        raise ValueError("F8R55_MAIN_MISSING_REAL_BETA_GRADIENT")
    scores={p:{"active":score(model,p)} for p in PANELS}
    for name,kwargs in (
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("erase_immediate",{"erase_immediate":True}),
        ("erase_delayed",{"erase_delayed":True}),
        ("zero_beta",{"zero_beta":True}),
        ("force_tag_top2",{"force_tag_top2":True}),
        ("force_upstream_top2",{"force_upstream_top2":True}),
        ("force_immediate_top2",{"force_immediate_top2":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_signal",{"zero_signal":True})):
        scores["ood"][name]=score(model,"ood",**kwargs)
    cold=Policy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r53.r52.r51.r40._hash(cold)!=r53.r52.r51.r40._hash(model) or
        any(score(cold,p)["choices"]!=scores[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R55_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"optimizer_steps":UPDATES,
            "productive_steps":productive,
            "real_beta_gradient":beta_gradient
              if arm=="learned_goal_mediation" else None,
            "trainable_parameters":sum(p.numel() for p in model.parameters()
                                      if p.requires_grad),
            "training_seconds":time.perf_counter()-began,
            "trace":trace,"scores":scores,"cold_restore":True}


def bootstrap_lower(main,control):
    rng=random.Random(BOOT_SEED)
    values=[]
    for _ in range(BOOT_DRAWS):
        idx=[rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        values.append(statistics.fmean(control[i]-main[i] for i in idx))
    return sorted(values)[int(.025*BOOT_DRAWS)]


def judge(models,freeze):
    limits=freeze["frozen_score"]
    by={(m["arm"],m["seed"]):m for m in models}
    outcomes=[]
    for seed in SEEDS:
        main=by["learned_goal_mediation",seed]
        raw=by["raw_delayed_context",seed]
        fixed=by["fixed_delayed_goal_top2",seed]
        active=main["scores"]["ood"]["active"]
        cal=main["scores"]["cal"]["active"]
        raw_ood=raw["scores"]["ood"]["active"]
        penalties={name:main["scores"]["ood"][name]["mean_regret"]-
                   active["mean_regret"] for name in (
                    "erase_B","erase_tag","erase_fifth",
                    "erase_immediate","erase_delayed","zero_beta",
                    "force_tag_top2","force_upstream_top2",
                    "force_immediate_top2","zero_native","zero_signal")}
        gain=raw_ood["mean_regret"]-active["mean_regret"]
        lower=bootstrap_lower(active["per_group_regret"],
                              raw_ood["per_group_regret"])
        slopes=main["trace"][-1]["slopes"]
        beta=main["trace"][-1]["beta"]
        passed=bool(
          main["trace"][-1]["heldout_fit"]>=limits["heldout_FIT_accuracy_min"]
          and cal["mean_regret"]<=limits["CAL_mean_regret_max"]
          and active["mean_regret"]<=limits["OOD_mean_regret_max"]
          and all(x<=limits["OOD_each_context_regret_max"]
                  for x in active["context_regrets"].values())
          and cal["gate_pair_flips"]>=limits["CAL_gate_flips_min"]
          and active["gate_pair_flips"]>=limits["OOD_gate_flips_min"]
          and penalties["erase_B"]>=limits["OOD_B_erase_penalty_min"]
          and penalties["erase_fifth"]>=limits["OOD_fifth_erase_penalty_min"]
          and penalties["erase_delayed"]>=limits["OOD_delayed_erase_penalty_min"]
          and penalties["zero_beta"]>=limits["OOD_beta_zero_penalty_min"]
          and penalties["force_tag_top2"]>=
              limits["OOD_force_public_tag_top2_penalty_min"]
          and penalties["force_upstream_top2"]>=
              limits["OOD_force_upstream_top2_penalty_min"]
          and penalties["force_immediate_top2"]>=
              limits["OOD_force_immediate_top2_penalty_min"]
          and gain>=limits["OOD_gain_over_raw_delayed_context_min"]
          and lower>0 and slopes[0]>=limits["context0_slope_min"]
          and slopes[1]<=limits["context1_slope_max"] and beta>0
          and main["real_beta_gradient"])
        outcomes.append({"seed":seed,"frozen_DEV_pass":passed,
            "FIT_heldout":main["trace"][-1]["heldout_fit"],
            "CAL_regret":cal["mean_regret"],"OOD_regret":active["mean_regret"],
            "OOD_gate_pair_flips":active["gate_pair_flips"],
            "OOD_context_regrets":active["context_regrets"],
            "slopes":slopes,"beta":beta,
            "OOD_ablation_penalties":penalties,
            "raw_delayed_OOD_regret":raw_ood["mean_regret"],
            "gain_over_raw":gain,"bootstrap_lower_gain":lower,
            "fixed_delayed_goal_OOD_regret":
                fixed["scores"]["ood"]["active"]["mean_regret"]})
    return {"two_seed_frozen_DEV_pass":all(x["frozen_DEV_pass"]
                                           for x in outcomes),
            "seeds":outcomes,"independent_scientific_pass":False,
            "Native_specific_gain_verified":False,
            "fully_matched_capacity_compute":False}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity={p:source_parity(p) for p in ("fit",)+PANELS}
    facts,labels=factual_fit()
    offset=calibration(facts)
    reference=positive_reference(offset)
    rng=torch.Generator(device="cpu")
    rng.manual_seed(DRAW_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(UPDATES,BATCH),
                        generator=rng).tolist()
    models=[train_one(arm,seed,offset,facts,labels,draws)
            for arm,seed in ARMS]
    result=judge(models,f)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "source_neutral_synthetic_DEV_only":True,
            "FIT_selected_factual_returns":len(facts),
            "FIT_train_pairs":TRAIN_PAIRS,"FIT_heldout_pairs":HELDOUT_PAIRS,
            "FIT_only_calibration":offset,
            "source_mean_parity":parity,"observable_reference":reference,
            "optimizer_attempts":len(ARMS)*UPDATES,
            "models":models,"judgment":result,
            "independent_scientific_pass":False,
            "external_original_firstuse":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    path=ROOT/"local_runs/p85mr_f8r55_goal_mediated_interventions_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError("F8R55_RESULT_ALREADY_EXISTS")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def reject_external_original_firstuse(*args,**kwargs):
    raise PermissionError("F8R55_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
