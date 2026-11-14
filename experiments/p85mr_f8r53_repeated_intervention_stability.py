"""R53 prospective synthetic repeatability of actually selected interventions.

The model sees three repetitions of factual effects for actions 0 and 1 at
each public source state. Public source-conditional contrast mean/variance is
a FIXED transparent feature transform, not discovered causal structure.
Real learned coefficients and full original strong controls are evaluated.
No original external first-use, private generator cues, or oracle return effect.
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

from experiments import p85mr_f8r52_action_contrast as r52

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r53_repeated_intervention_stability_freeze.json"
FREEZE_BLOB="2ae097f98fd9953181c4d6d57349cf7a23bd727d"
SCHEMA="ttt-hybrid.f8r53-repeated-selected-intervention-stability-freeze.v1"
PARENT="316563614bd0d2fdaf56b027358ef57dc0ebb4a8"
SALT={"fit":22710000,"cal":22720000,"ood":22730000}
NSOURCE={"fit":8,"cal":10,"ood":16}
AMP={"fit":.24,"cal":.18,"ood":.12}
SPUR={"fit":.8,"cal":.62,"ood":.45}
NUIS={"fit":.006,"cal":.009,"ood":.014}
SEEDS=(985201,985207)
ARM_NAMES=("learned_repeatability","raw_mean_context","public_tag_top2",
           "mean_magnitude_top2","fixed_stability_top2",
           "native_repeatability","gru_repetition_control")
ARMS=tuple((name,seed) for name in ARM_NAMES for seed in SEEDS)
FIT_GROUPS,TRAIN_GROUPS,EVAL_GROUPS=24,16,8
TRAIN_PAIRS,HELDOUT_PAIRS,BATCH,UPDATES=128,64,16,256
DRAW_SEED,BOOT_SEED,BOOT_DRAWS=22749999,22769999,10000
CHECKPOINTS=(1,32,64,128,192,256)
PANELS=("cal","ood")
PROBE=.42
REPS=3


def git_blob(data):
    return hashlib.sha1(
        b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R53_FROZEN_BLOB_CHANGED")
    f=json.loads(raw)
    so,me,sc=f["source"],f["methods"],f["frozen_score"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_GENERATOR_MODEL_OR_SCORE" or
        f["parent"]["qualified_r52_head"]!=PARENT or
        f["parent"]["r52_freeze_blob"]!=r52.FREEZE_BLOB or
        so["fresh_salts"]!=SALT or
        so["public_source_count"]!=NSOURCE or
        so["replications_per_source"]!=REPS or
        so["causal_persistent_action_contrast"]!=AMP or
        so["distractor_alternating_action_contrast_amplitude"]!=SPUR or
        so["full_candidate_record_length"]!={"fit":54,"cal":66,"ood":102} or
        me["original_seeds"]!=list(SEEDS) or
        me["arms"]!=list(ARM_NAMES) or
        me["train_updates_per_arm"]!=UPDATES or
        me["train_batch_pairs"]!=BATCH or
        me["draw_seed"]!=DRAW_SEED or
        me["total_optimizer_attempts"]!=len(ARMS)*UPDATES or
        sc["paired_group_bootstrap"]["seed"]!=BOOT_SEED or
        sc["independent_scientific_pass"] is not False or
        any(f["authority"].values())):
        raise ValueError("F8R53_PROTOCOL_OR_AUTHORITY_CHANGED")
    r52.locked()
    return f


def positions(panel,group):
    if (panel not in SALT or type(group) is not int or
        not 0<=group<(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)):
        raise ValueError("F8R53_UNREGISTERED_GROUP")
    ranked=sorted(range(NSOURCE[panel]),key=lambda s:hashlib.sha256(
        f"F8-R53/{panel}/{SALT[panel]}/{group}/{s}".encode()).digest())
    return tuple(ranked[:4])


def world(panel,group,context,q_sign,gate):
    positions(panel,group)
    if context not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R53_INVALID_WORLD_FACTOR")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":q_sign,"gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R53_UNREGISTERED_PANEL")
    g=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    return [world(panel,i,c,q,h) for i in range(g)
            for c in (0,1) for q in (-1,1) for h in (-1,1)]


def selected_repeated_B(w):
    """Each (public source,repetition) observes two independently selected actions.

    The causal source has stationary action contrast across all 3 repeats.
    Larger-mean, higher-tag confounds alternate contrasts +A,-A,+A
    (or -A,+A,-A) and are independent of the hidden causal gate.
    """
    p,g,c,gate=w["panel"],w["group"],w["public_context"],w["gate_evaluator_only"]
    cue,decoy,s1,s2=positions(p,g)
    rows=[]
    for source in range(NSOURCE[p]):
        if source==cue:
            tag=.42 if c==0 else .18
        elif source==decoy:
            tag=.18 if c==0 else .42
        elif source==s1:
            tag=.88
        elif source==s2:
            tag=.77
        else:
            tag=.03 if source%3==0 else 0.
        for rep in range(REPS):
            if source==cue:
                contrast=AMP[p]*gate
            elif source==decoy:
                contrast=-AMP[p]*gate
            elif source==s1:
                contrast=SPUR[p]*(1 if rep!=1 else -1)
            elif source==s2:
                contrast=-SPUR[p]*(1 if rep!=1 else -1)
            else:
                contrast=0.
            # B-only action-invariant nuisance: identical across selected
            # actions, not the causal contrast. It is selected and observed.
            if source not in (cue,decoy,s1,s2):
                offset=round(NUIS[p]*math.sin(
                    (SALT[p]+g)*.137+source*.271)+
                    .004*(((source+g)%4)-1.5),8)
            else:
                offset=0.
            rows.append((tag,0,offset+contrast))
            rows.append((tag,1,offset-contrast))
    return rows


def following(w,choice):
    # The R52 return law uses public state plus evaluator-private q×gate,
    # not the R52 B generator. The new source salt is independent.
    return r52.following(w,choice)


def observed_history(w,*,erase_B=False,erase_tag=False,erase_fifth=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    records=[r52.r51.r33.b._record(
        s,action,r52.r51.r33.r32.after(s,action,c,q),c)
        for s,action in r52.r51.r33.r32.EARLY]
    records.append(r52.r51.r33.b._record(
        PROBE,c,r52.r51.r33.r32.after(PROBE,c,c,q),c))
    for tag,action,delta in selected_repeated_B(w):
        shown=0. if erase_tag else tag
        key,eff=r52.r51.r33.b._record(
            shown,action,shown+delta,c)
        eff=eff.clone()
        eff[4].zero_()
        records.append((key,eff))
    actions=torch.stack([a for a,_ in records])
    effects=torch.stack([e for _,e in records])
    if erase_B:effects[5:].zero_()
    if erase_fifth:effects[4].zero_()
    return actions,effects


def candidates(w,**kwargs):
    a,e=observed_history(w,**kwargs)
    aa=[];ee=[]
    for action in (0,1):
        k=r52.r51.r33.b._record(
            w["return_state"],action,w["return_state"],
            w["public_context"])[0]
        aa.append(torch.cat((a,k[None])))
        ee.append(torch.cat((e,torch.zeros((1,16),dtype=e.dtype))))
    return torch.stack(aa),torch.stack(ee)


def source_parity(panel):
    n=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    largest=0.
    for g in range(n):
        for c in (0,1):
            for q in (-1,1):
                a,e=candidates(world(panel,g,c,q,1))
                aa,ee=candidates(world(panel,g,c,q,-1))
                if not torch.equal(a,aa):
                    raise ValueError("F8R53_GATE_ACTION_KEYS_LEAK")
                err=(e[:,5:-1].mean(1)-ee[:,5:-1].mean(1)).abs().max().item()
                largest=max(largest,err)
                if err>1e-7 or torch.equal(e[:,5:-1],ee[:,5:-1]):
                    raise ValueError("F8R53_GATE_MEAN_LEAK_OR_NO_SELECTED_SIGNAL")
    return {"panel":panel,"groups":n,"contexts":2,
            "max_gate_full_B_mean_difference":largest}


def factual_fit():
    locked()
    factual=[];labels=[]
    for w in worlds("fit"):
        costs=[]
        for action in (0,1):
            a,e=observed_history(w)
            after=following(w,action)
            key,effect=r52.r51.r33.b._record(
                w["return_state"],action,after,w["public_context"])
            aa=torch.cat((a,key[None]))
            ee=torch.cat((e,effect[None]))
            physical=w["return_state"]+r52.r51.r31._physical_delta(ee[-1,0])
            costs.append((physical-1.)**2)
            factual.append((aa,ee))
        labels.append(0 if costs[0]<=costs[1] else 1)
    if len(factual)!=384 or len(labels)!=192:
        raise ValueError("F8R53_BAD_FACTUAL_FIT")
    return factual,labels


def calibration(records):
    if len(records)!=384:
        raise ValueError("F8R53_BAD_FIT_CALIBRATION")
    fifth=[PROBE+r52.r51.r31._physical_delta(records[2*i][1][4,0])
           for i in range(TRAIN_PAIRS)]
    center=statistics.fmean(fifth)
    if not math.isfinite(center):
        raise ValueError("F8R53_NONFINITE_FIFTH_CENTER")
    return center


def regret(w,chosen):
    if chosen not in (0,1):
        raise ValueError("F8R53_INVALID_ACTION")
    costs=[(following(w,act)-1.)**2 for act in (0,1)]
    return max(0.,costs[chosen]-min(costs))


def public_repeated_features(a,e,offset):
    if (a.ndim!=3 or e.shape!=a.shape or a.shape[-1]!=16 or
        a.shape[1] not in (54,66,102) or a.shape[0]%2):
        raise ValueError("F8R53_INVALID_PUBLIC_HISTORY")
    obs=e.clone()
    obs[:,-1].zero_()
    action_keys=a[::2,5:-1]
    effects=obs[::2,5:-1,0]
    batch=action_keys.shape[0]
    nsource={54:8,66:10,102:16}[a.shape[1]]
    keys=action_keys.reshape(batch,nsource,REPS,2,16)
    selected=effects.reshape(batch,nsource,REPS,2)
    # All 6 observed records at a public source have the same state tag.
    tags=2.4*torch.atanh(
        (keys[...,1]/keys[...,0]).clamp(-.98,.98))
    if not bool(torch.allclose(tags,tags[...,0,0,None,None],
                               atol=1e-6,rtol=0)):
        raise ValueError("F8R53_PUBLIC_SOURCE_TAG_CHANGED_WITHIN_REPEAT")
    tag=tags[:,:,0,0]
    physical=2.4*torch.atanh(selected.clamp(-.98,.98))
    if not bool(torch.isfinite(physical).all()):
        raise ValueError("F8R53_NONFINITE_FACTUAL_EFFECT")
    contrasts=(physical[:,:,:,0]-physical[:,:,:,1])*.5
    mean=contrasts.mean(-1)
    variance=((contrasts-mean[:,:,None])**2).mean(-1)
    # Public continuous features available to both learned and fixed arms.
    reliable=12.*mean.abs()/(1.+20.*variance)
    magnitude=mean.abs()
    ratio=mean.abs()/(variance.sqrt()+.02)
    fifth=PROBE+2.4*torch.atanh(obs[::2,4,0].clamp(-.98,.98))-offset
    ctx=(3.*torch.atanh(
        (a[::2,-1,2]/a[::2,-1,0]).clamp(-.98,.98))).round().long()
    return tag,mean,variance,reliable,magnitude,ratio,fifth,ctx,obs


def positive_reference(offset):
    outcome={}
    for panel in ("fit",)+PANELS:
        mistakes=0
        for w in worlds(panel):
            a,e=candidates(w)
            tag,mean,var,rel,mag,ratio,fifth,ctx,obs=(
                public_repeated_features(a,e,torch.tensor(offset)))
            ids=torch.argsort(ratio,dim=-1,descending=True,
                               stable=True)[:,:2]
            ts=torch.gather(tag,1,ids)[0].tolist()
            mu=torch.gather(mean,1,ids)[0].tolist()
            if min(abs(x) for x in mu)<.10:
                raise ValueError("F8R53_PUBLIC_REFERENCE_NOT_IDENTIFIABLE")
            chosen_idx=ts.index(max(ts)) if w["public_context"]==0 else (
                ts.index(min(ts)))
            chosen=w["public_context"] if float(fifth[0])*mu[chosen_idx]>0 else (
                1-w["public_context"])
            mistakes+=int(regret(w,chosen)>1e-10)
        if mistakes:
            raise ValueError("F8R53_POSITIVE_CONTROL_FAILED")
        outcome[panel]={"worlds":len(worlds(panel)),"mistakes":mistakes}
    return outcome


class LongNative(r52.r51.r34.FactualMemory):
    """R53-only full selected-input adapter, shared R34 untouched."""
    def forward(self,a,e,*,ablate_backbone=False):
        if (a.ndim!=3 or e.shape!=a.shape or
            a.shape[-1]!=16 or a.shape[1] not in (54,66,102)):
            raise ValueError("F8R53_NATIVE_UNREGISTERED_LENGTH")
        rec=torch.cat((a[:,:-1],e[:,:-1]),dim=-1)
        _,h=self.memory(rec)
        if self.backbone is None or ablate_backbone:
            native=torch.zeros((a.shape[0],1),dtype=a.dtype,device=a.device)
        else:
            pred=r52.r51.r34.b._predict(
                self.base_arm,self.backbone,{"action":a,"effect":e})
            if pred.shape!=e.shape:
                raise ValueError("F8R53_INVALID_NATIVE_OUTPUT")
            native=pred[:,-1,0:1]
        return self.readout(torch.cat((h[-1],a[:,-1],native),dim=-1)
                            ).squeeze(-1)


class Policy(nn.Module):
    def __init__(self,arm,seed,offset):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R53_UNREGISTERED_ARM")
        self.arm,self.seed=arm,seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm!="gru_repetition_control":
            self.slopes=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        if arm in ("learned_repeatability","native_repeatability"):
            self.beta=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.gain=nn.Parameter(torch.tensor(3.+(seed-985201)*.01))
        if arm=="native_repeatability":
            self.native=LongNative("native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.))
        if arm=="gru_repetition_control":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22780000+seed)
                self.gru=nn.GRU(4,24,batch_first=True)
                self.readout=nn.Linear(24,1)

    def forward(self,a,e,*,zero_beta=False,force_tag_top2=False,
                force_mean_top2=False,force_uniform=False,
                zero_native=False,zero_signal=False):
        tag,mean,var,rel,mag,ratio,fifth,ctx,obs=(
            public_repeated_features(a,e,self.fit_offset))
        if self.arm=="gru_repetition_control":
            feats=torch.stack((tag,mean,var,
                               ctx[:,None].float().expand_as(tag)),dim=-1)
            _,h=self.gru(feats)
            signal=torch.tanh(self.readout(h[-1]).squeeze(-1))
        else:
            if self.arm=="public_tag_top2" or force_tag_top2:
                ids=r52.r51.r49.stable_public_top_two(tag)
            elif self.arm=="mean_magnitude_top2" or force_mean_top2:
                ids=torch.argsort(mag,dim=-1,descending=True,stable=True)[:,:2]
            elif self.arm=="fixed_stability_top2":
                ids=torch.argsort(ratio,dim=-1,descending=True,stable=True)[:,:2]
            else:
                ids=None
            if ids is None:
                selected_tag,selected_mean,selected_rel=tag,mean,rel
            else:
                selected_tag=torch.gather(tag,1,ids)
                selected_mean=torch.gather(mean,1,ids)
                selected_rel=torch.gather(rel,1,ids)
            logits=self.slopes[ctx,None]*selected_tag
            if (self.arm in ("learned_repeatability","native_repeatability")
                and not zero_beta and ids is None):
                logits=logits+self.beta*selected_rel
            weights=(torch.full_like(logits,1./logits.shape[1])
                     if force_uniform else torch.softmax(logits,dim=-1))
            signal=(weights*selected_mean).sum(-1)
        if zero_signal:signal=torch.zeros_like(signal)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*signal)
        if self.arm=="native_repeatability":
            native=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref=torch.tanh(torch.atanh(pref.clamp(-.98,.98))+
                torch.tanh(self.native_gain)*(native[:,1]-native[:,0]))
        return r52.r51.r47._choice_cost(ctx,pref)


def batch(records,labels,ids):
    if len(ids)!=BATCH or any(type(i) is not int or
                               not 0<=i<TRAIN_PAIRS for i in ids):
        raise ValueError("F8R53_BAD_TRAIN_BATCH")
    rows=[records[2*i+j] for i in ids for j in (0,1)]
    return (torch.stack([x[0] for x in rows]),
            torch.stack([x[1] for x in rows]),
            torch.tensor([labels[i] for i in ids],dtype=torch.long))


def heldout(model,records,labels):
    model.eval()
    good=0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            ids=list(range(start,start+BATCH))
            rows=[records[2*i+j] for i in ids for j in (0,1)]
            aa=torch.stack([x[0] for x in rows])
            ee=torch.stack([x[1] for x in rows])
            yy=torch.tensor([labels[i] for i in ids])
            good+=int((model(aa,ee).argmin(-1)==yy).sum())
    return good/HELDOUT_PAIRS


def score(model,panel,*,erase_B=False,erase_tag=False,erase_fifth=False,
          zero_beta=False,force_tag_top2=False,force_mean_top2=False,
          force_uniform=False,zero_native=False,zero_signal=False):
    if panel not in PANELS:
        raise ValueError("F8R53_UNREGISTERED_SCORE_PANEL")
    cohort=worlds(panel)
    input_rows=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                           erase_fifth=erase_fifth) for w in cohort]
    aa=torch.cat([x[0] for x in input_rows])
    ee=torch.cat([x[1] for x in input_rows])
    model.eval()
    with torch.no_grad():
        losses=model(aa,ee,zero_beta=zero_beta,
            force_tag_top2=force_tag_top2,
            force_mean_top2=force_mean_top2,
            force_uniform=force_uniform,zero_native=zero_native,
            zero_signal=zero_signal)
    if losses.shape!=(len(cohort),2) or not bool(torch.isfinite(losses).all()):
        raise ValueError("F8R53_INVALID_RETURN_POLICY")
    choices=losses.argmin(-1).tolist()
    regrets=[regret(w,action) for w,action in zip(cohort,choices)]
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "context_regrets":{str(c):statistics.fmean(
                regrets[i] for i,w in enumerate(cohort)
                if w["public_context"]==c) for c in (0,1)},
            "per_group_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(cohort),2))}


def train_one(arm,seed,offset,records,labels,draws):
    model=Policy(arm,seed,offset)
    params=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            rate=.02 if name in ("slopes","beta","gain","native_gain") else .001
            params.append({"params":[p],"lr":rate})
    opt=torch.optim.AdamW(params,weight_decay=0.)
    actual=0
    beta_grad=False
    context_grad=[False,False]
    trace=[]
    start=time.perf_counter()
    for step,ids in enumerate(draws,1):
        aa,ee,y=batch(records,labels,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(aa,ee),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R53_NONFINITE_TRAINING_LOSS")
        loss.backward()
        gradients=[p.grad for p in model.parameters() if p.requires_grad]
        if any(g is not None and not bool(torch.isfinite(g).all())
               for g in gradients):
            raise ValueError("F8R53_NONFINITE_GRADIENT")
        actual+=int(any(g is not None and bool(torch.any(g!=0))
                        for g in gradients))
        if arm=="learned_repeatability":
            beta_grad|=(model.beta.grad is not None and
                        bool(model.beta.grad!=0))
            for k in (0,1):
                context_grad[k]|=(model.slopes.grad is not None and
                                  bool(model.slopes.grad[k]!=0))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
        if step in CHECKPOINTS:
            trace.append({"step":step,"FIT_loss":float(loss.detach()),
                "FIT_heldout":heldout(model,records,labels),
                "context_slopes":model.slopes.detach().tolist()
                  if hasattr(model,"slopes") else None,
                "learned_beta":float(model.beta.detach())
                  if hasattr(model,"beta") else None})
    if arm=="learned_repeatability" and (
        actual==0 or not beta_grad or not all(context_grad)):
        raise ValueError("F8R53_MISSING_REAL_RELIABILITY_GRADIENT")
    results={p:{"active":score(model,p)} for p in PANELS}
    for name,options in (
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("zero_beta",{"zero_beta":True}),
        ("force_tag_top2",{"force_tag_top2":True}),
        ("force_mean_top2",{"force_mean_top2":True}),
        ("force_uniform",{"force_uniform":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_signal",{"zero_signal":True})):
        results["ood"][name]=score(model,"ood",**options)
    cold=Policy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r52.r51.r40._hash(cold)!=r52.r51.r40._hash(model) or
        any(score(cold,p)["choices"]!=results[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R53_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,
            "optimizer_attempts":UPDATES,
            "productive_gradient_steps":actual,
            "real_beta_gradient":beta_grad if arm=="learned_repeatability" else None,
            "parameter_count":sum(p.numel() for p in model.parameters()
                                  if p.requires_grad),
            "train_seconds":time.perf_counter()-start,
            "traces":trace,"scores":results,"cold_restore":True}


def bootstrap_lower(main,raw):
    rng=random.Random(BOOT_SEED)
    scores=[]
    for _ in range(BOOT_DRAWS):
        ids=[rng.randrange(EVAL_GROUPS) for __ in range(EVAL_GROUPS)]
        scores.append(statistics.fmean(raw[i]-main[i] for i in ids))
    return sorted(scores)[int(.025*BOOT_DRAWS)]


def judge(models,freeze):
    limits=freeze["frozen_score"]
    by={(m["arm"],m["seed"]):m for m in models}
    out=[]
    for seed in SEEDS:
        main=by["learned_repeatability",seed]
        raw=by["raw_mean_context",seed]
        fixed=by["fixed_stability_top2",seed]
        active=main["scores"]["ood"]["active"]
        cal=main["scores"]["cal"]["active"]
        raw_ood=raw["scores"]["ood"]["active"]
        gains={k:main["scores"]["ood"][k]["mean_regret"]-
                active["mean_regret"] for k in (
                "erase_B","erase_tag","erase_fifth","zero_beta",
                "force_tag_top2","force_mean_top2","force_uniform",
                "zero_native","zero_signal")}
        gain=raw_ood["mean_regret"]-active["mean_regret"]
        low=bootstrap_lower(active["per_group_regret"],
                            raw_ood["per_group_regret"])
        beta=main["traces"][-1]["learned_beta"]
        s0,s1=main["traces"][-1]["context_slopes"]
        qualified=bool(
            main["traces"][-1]["FIT_heldout"]>=limits["min_heldout_fit_accuracy"]
            and cal["mean_regret"]<=limits["max_cal_regret"]
            and active["mean_regret"]<=limits["max_ood_regret"]
            and all(v<=limits["max_ood_regret_each_context"]
                    for v in active["context_regrets"].values())
            and cal["gate_pair_flips"]>=limits["min_cal_gate_flips"]
            and active["gate_pair_flips"]>=limits["min_ood_gate_flips"]
            and gains["erase_B"]>=limits["min_ood_erase_B_penalty"]
            and gains["erase_fifth"]>=limits["min_ood_erase_fifth_penalty"]
            and gains["zero_beta"]>=limits["min_ood_zero_beta_penalty"]
            and gains["force_tag_top2"]>=limits["min_ood_force_tag_top2_penalty"]
            and gains["force_mean_top2"]>=limits["min_ood_force_mean_top2_penalty"]
            and gain>=limits["min_ood_gain_vs_raw_context"]
            and low>0 and s0>=limits["context0_slope_min"]
            and s1<=limits["context1_slope_max"] and beta>0
            and main["real_beta_gradient"])
        out.append({"seed":seed,"frozen_DEV_pass":qualified,
            "FIT_heldout":main["traces"][-1]["FIT_heldout"],
            "CAL_regret":cal["mean_regret"],
            "OOD_regret":active["mean_regret"],
            "OOD_each_context":active["context_regrets"],
            "OOD_gate_pair_flips":active["gate_pair_flips"],
            "learned_slopes":[s0,s1],"learned_beta":beta,
            "OOD_ablation_penalties":gains,
            "gain_vs_raw":gain,"bootstrap_lower_gain":low,
            "fixed_stability_OOD_regret":
                fixed["scores"]["ood"]["active"]["mean_regret"]})
    return {"two_seed_frozen_DEV_pass":all(x["frozen_DEV_pass"] for x in out),
            "seeds":out,"independent_scientific_pass":False,
            "Native_specific_gain_verified":False,
            "whole_process_capacity_matched":False}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parity={p:source_parity(p) for p in ("fit",)+PANELS}
    facts,labels=factual_fit()
    offset=calibration(facts)
    reference=positive_reference(offset)
    gen=torch.Generator(device="cpu")
    gen.manual_seed(DRAW_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(UPDATES,BATCH),
                        generator=gen).tolist()
    models=[train_one(arm,seed,offset,facts,labels,draws)
            for arm,seed in ARMS]
    j=judge(models,f)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,"parent":PARENT,
            "source_neutral_synthetic_DEV_only":True,
            "FIT_factual_selected_returns":len(facts),
            "FIT_train_pairs":TRAIN_PAIRS,"FIT_heldout_pairs":HELDOUT_PAIRS,
            "FIT_only_selected_fifth_offset":offset,
            "source_parity":parity,"observable_reference":reference,
            "optimizer_attempts":len(ARMS)*UPDATES,
            "models":models,"judgment":j,
            "independent_scientific_pass":False,
            "external_original_first_use":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    path=ROOT/"local_runs/p85mr_f8r53_repeated_intervention_stability_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R53_REFUSE_OVERWRITE")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def forbid_original_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R53_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
