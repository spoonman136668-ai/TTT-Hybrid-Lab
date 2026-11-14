"""F8-R51 learned effect-magnitude eligibility plus context-conditioned role: prospective DEV.

An independently frozen, source-neutral synthetic test of conditional causal
memory. The public context controls which observable tag encodes the causal
B consequence. This never uses generator hidden q/gate/positions as model input.
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

from experiments import p85mr_f8r31_factual_conditional_risk as r31
from experiments import p85mr_f8r33_probe_calibration as r33
from experiments import p85mr_f8r34_learned_factual_memory as r34
from experiments import p85mr_f8r40_native_fusion as r40
from experiments import p85mr_f8r41_history_conditional as r41
from experiments import p85mr_f8r47_cue_key_contrast as r47
from experiments import p85mr_f8r48_context_role_switch as r48
from experiments import p85mr_f8r49_public_event_selection as r49

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r51_learned_eligibility_freeze.json"
FREEZE_BLOB="ef2905101d4dd7f2086b6759e8a496f2e3e3629e"
SCHEMA="ttt-hybrid.f8r51-learned-effect-magnitude-eligibility-freeze.v1"
PARENT="2b297ba7316c0c552522ecd2ae1a753934e88174"
SALT={"fit":22510000,"cal":22520000,"ood":22530000}
AMP={"fit":.24,"cal":.18,"ood":.12}
NUIS={"fit":.006,"cal":.009,"ood":.014}
SEEDS=(985201,985207)
ARM_NAMES=("learned_magnitude_context","raw_context_selector",
           "tag_top_two_context","fixed_magnitude_top_two_context",
           "learned_magnitude_native","gru_context_control")
ARMS=tuple((name,seed) for name in ARM_NAMES for seed in SEEDS)
FIT_GROUPS, TRAIN_GROUPS, EVAL_GROUPS=24,16,8
TRAIN_PAIRS, HELDOUT_PAIRS, BATCH, UPDATES=128,64,16,256
DRAW_SEED,BOOT_SEED,BOOT_DRAWS=22549999,22569999,10000
CHECKPOINTS=(1,32,64,128,192,256)
PANELS=("cal","ood")
PROBE,CUE_TAG,DECOY_TAG=.42,.42,.18


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R51_PRE_REGISTERED_SOURCE_DRIFT")
    f=json.loads(raw)
    d,m,s=f["source"],f["training"],f["frozen_score"]
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R51_MODEL_GENERATOR_OR_RESULTS" or
        f["parent"]["qualified_R49_head"]!=PARENT or
        f["parent"]["r49_freeze_blob"]!=r49.FREEZE_BLOB or
        d["panel_salts"]!=SALT or
        d["causal_B_delta_abs"]!=AMP or
        d["other_delta_amplitudes"]!=NUIS or
        d["selected_B_history_length"]!={"fit":12,"cal":20,"ood":40} or
        d["original_selected_return_factual_labels"]!=384 or
        m["seeds"]!=list(SEEDS) or
        m["arm_names"]!=list(ARM_NAMES) or
        m["train_updates_per_arm"]!=UPDATES or
        m["training_sample_seed"]!=DRAW_SEED or
        m["balanced_paired_batch"]!=BATCH or
        s["bootstrap"]["seed"]!=BOOT_SEED or
        s["scientific_pass"] is not False or
        any(v is not True for v in f["authority"].values())):
        raise ValueError("F8R51_FREEZE_PROTOCOL_OR_AUTHORITY_DRIFT")
    r49.locked()
    return f

def positions(panel, group):
    if (panel not in SALT or type(group) is not int or
        not 0<=group<(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)):
        raise ValueError("F8R51_UNREGISTERED_GROUP")
    length={"fit":12,"cal":20,"ood":40}[panel]
    rank=sorted(range(length),key=lambda step:
        hashlib.sha256(f"F8-R51/{panel}/{SALT[panel]}/{group}/{step}".encode()
        ).digest())
    return tuple(rank[:4])

def world(panel,group,context,q_sign,gate):
    positions(panel,group)
    if context not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R51_INVALID_FACTORIAL_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":q_sign,"gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R51_UNREGISTERED_PANEL")
    return [world(panel,g,c,q,h)
            for g in range(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)
            for c in (0,1) for q in (-1,1) for h in (-1,1)]


def B_selected(w):
    p,g,c=w["panel"],w["group"],w["public_context"]
    cue,decoy,high1,high2=positions(p,g)
    confounded=p!="fit" or g%2==0
    length={"fit":12,"cal":20,"ood":40}[p]
    rows=[]
    for t in range(length):
        if t==cue:
            tag,delta=(CUE_TAG if c==0 else DECOY_TAG),AMP[p]*w["gate_evaluator_only"]
        elif t==decoy:
            tag,delta=(DECOY_TAG if c==0 else CUE_TAG),-AMP[p]*w["gate_evaluator_only"]
        elif confounded and t==high1:
            tag,delta=.88,.015
        elif confounded and t==high2:
            tag,delta=.77,-.015
        else:
            tag=.03 if t%3==0 else 0.
            delta=round(NUIS[p]*math.sin((SALT[p]+g)*.137+t*.271)
                +.004*(((t+g)%4)-1.5),8)
        rows.append((tag,t%2,delta))
    return rows

def following(w,choice):
    s,c=w["return_state"],w["public_context"]
    q=w["q_evaluator_only"]*w["gate_evaluator_only"]
    return (r41._law(s,choice,c,q,"ood")
            if w["panel"]=="ood" else r33.r32.after(s,choice,c,q))


def observed_history(w,*,erase_B=False,erase_tag=False,
                     erase_fifth=False,swap_B=False):
    c,q=w["public_context"],w["q_evaluator_only"]
    records=[r33.b._record(s,choice,r33.r32.after(s,choice,c,q),c)
             for s,choice in r33.r32.EARLY]
    records.append(r33.b._record(PROBE,c,
                   r33.r32.after(PROBE,c,c,q),c))
    for tag,choice,delta in B_selected(w):
        if swap_B and tag>=.08:
            delta=-delta
        public_tag=(0. if erase_tag and tag>0 else tag)
        key,effect=r33.b._record(public_tag,choice,
                                public_tag+delta,c)
        effect=effect.clone()
        effect[4]=0.
        records.append((key,effect))
    a=torch.stack([pair[0] for pair in records])
    e=torch.stack([pair[1] for pair in records])
    if erase_B:
        e[5:].zero_()
    if erase_fifth:
        e[4].zero_()
    return a,e


def candidates(w,**kwargs):
    a,e=observed_history(w,**kwargs)
    keys=[]; effects=[]
    for choice in (0,1):
        key=r33.b._record(w["return_state"],choice,
             w["return_state"],w["public_context"])[0]
        keys.append(torch.cat((a,key[None])))
        effects.append(torch.cat((e,torch.zeros((1,16)))))
    return torch.stack(keys),torch.stack(effects)


def parity(panel):
    n=FIT_GROUPS if panel=="fit" else EVAL_GROUPS
    worst=0.
    for g in range(n):
        for c in (0,1):
            a1,e1=candidates(world(panel,g,c,1,1))
            a2,e2=candidates(world(panel,g,c,1,-1))
            if not torch.equal(a1,a2):
                raise ValueError("F8R51_PUBLIC_ACTION_KEY_LEAK")
            delta=(e1[:,5:-1,:].mean(1)-
                   e2[:,5:-1,:].mean(1)).abs().max().item()
            worst=max(worst,delta)
            if delta>1e-7:
                raise ValueError("F8R51_FULL_B_MEAN_LEAK")
            if torch.equal(e1[:,5:-1,:],e2[:,5:-1,:]):
                raise ValueError("F8R51_B_OBSERVATIONS_INSEPARABLE")
    return {"panel":panel,"groups":n,"contexts":2,
            "max_paired_full_B_mean_difference":worst}


def factual_fit():
    locked()
    records=[]; labels=[]
    for w in worlds("fit"):
        costs=[]
        for choice in (0,1):
            a,e=observed_history(w)
            after=following(w,choice)
            key,eff=r33.b._record(w["return_state"],choice,
                                   after,w["public_context"])
            aa=torch.cat((a,key[None]))
            ee=torch.cat((e,eff[None]))
            # Decoded physical state, not misinterpreted delta.
            actual=w["return_state"]+r31._physical_delta(ee[-1,0])
            costs.append((actual-1.)**2)
            records.append((aa,ee))
        labels.append(min((0,1),key=lambda idx:(costs[idx],idx)))
    if len(records)!=384 or len(labels)!=192 or labels.count(0)!=96:
        raise ValueError("F8R51_FACTUAL_SOURCE_IMBALANCE")
    return records,labels


def calibration(records):
    if len(records)!=384:
        raise ValueError("F8R51_CALIBRATION_UNQUALIFIED")
    after=[PROBE+r31._physical_delta(records[2*i][1][4,0])
           for i in range(TRAIN_PAIRS)]
    offset=statistics.fmean(after)
    if not math.isfinite(offset):
        raise ValueError("F8R51_CALIBRATION_INVALID")
    return offset


def regret(w,choice):
    if choice not in (0,1):
        raise ValueError("F8R51_INVALID_RETURN_ACTION")
    costs=[(following(w,a)-1.)**2 for a in (0,1)]
    return max(0.,costs[choice]-min(costs))


def positive_observation_reference(offset):
    out={}
    for panel in ("fit",)+PANELS:
        ww=worlds(panel)
        errors=0
        for w in ww:
            a,e=candidates(w)
            a,e=a[0],e[0]
            ctx=r31._decode_key(a[-1])[1]
            fifth=PROBE+r31._physical_delta(e[4,0])-offset
            tagged=[(r31._decode_key(a[i])[0],
                      r31._physical_delta(e[i,0]))
                    for i in range(5,a.shape[0]-1)]
            magnitude_sorted=sorted(tagged,key=lambda row:
                                    abs(row[1]),reverse=True)
            largest=magnitude_sorted[:2]
            if (len(largest)!=2 or
                min(abs(effect) for _,effect in largest)<=.06 or
                any(abs(effect)>.03 for _,effect in magnitude_sorted[2:])):
                raise ValueError("F8R51_SOURCE_ELIGIBILITY_NOT_OBSERVABLE")
            causal=(max(largest,key=lambda x:x[0])[1]
                    if ctx==0 else min(largest,key=lambda x:x[0])[1])
            choice=ctx if fifth*causal>0 else 1-ctx
            errors+=int(regret(w,choice)>1e-10)
        out[panel]={"worlds":len(ww),"mistakes":errors}
        if errors:
            raise ValueError("F8R51_SOURCE_POSITIVE_REFERENCE_FAILED")
    return out

def components(a,e,offset):
    if (a.ndim!=3 or e.shape!=a.shape or a.shape[-1]!=16
        or a.shape[1] not in (18,26,46) or a.shape[0]%2):
        raise ValueError("F8R51_INVALID_PAIRED_HISTORY")
    obs=e.clone()
    obs[:,-1,:].zero_()
    tag=2.4*torch.atanh(
        (a[::2,5:-1,1]/a[::2,5:-1,0]).clamp(-.98,.98))
    normalized_delta=obs[::2,5:-1,0]
    measured_delta=2.4*torch.atanh(normalized_delta.clamp(-.98,.98))
    fifth=PROBE+2.4*torch.atanh(
        obs[::2,4,0].clamp(-.98,.98))-offset
    ctx=(3.0*torch.atanh(
        (a[::2,-1,2]/a[::2,-1,0]).clamp(-.98,.98))
        ).round().long()
    return tag,normalized_delta,measured_delta,fifth,ctx,obs


class R51FullHistoryMemory(r34.FactualMemory):
    """R51-only full-length adapter; original shared R34 remains unchanged."""
    def forward(self,actions,effects,*,ablate_backbone=False):
        if (actions.ndim!=3 or actions.shape[-1]!=16 or
            effects.shape!=actions.shape):
            raise ValueError("F8R51_INVALID_NATIVE_HISTORY")
        if actions.shape[1] in (14,22):
            return super().forward(actions,effects,ablate_backbone=ablate_backbone)
        if actions.shape[1] not in (18,26,46):
            raise ValueError("F8R51_NATIVE_UNREGISTERED_HISTORY")
        record=torch.cat((actions[:,:-1,:],effects[:,:-1,:]),dim=-1)
        _,memory=self.memory(record)
        context=memory[-1]
        if self.backbone is None or ablate_backbone:
            native=torch.zeros((actions.shape[0],1),
                dtype=actions.dtype,device=actions.device)
        else:
            pred=r34.b._predict(self.base_arm,self.backbone,
                {"action":actions,"effect":effects})
            if pred.shape!=effects.shape:
                raise ValueError("F8R51_NATIVE_BACKBONE_PREDICTION_SHAPE")
            native=pred[:,-1,0:1]
        return self.readout(torch.cat((context,actions[:,-1,:],native),dim=-1)
                            ).squeeze(-1)


class Policy(nn.Module):
    def __init__(self,arm,seed,offset):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R51_UNREGISTERED_ARM_OR_SEED")
        self.arm,self.seed=arm,seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm!="gru_context_control":
            self.slopes=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        if arm in ("learned_magnitude_context","learned_magnitude_native"):
            self.salience=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.gain=nn.Parameter(torch.tensor(3.+(seed-985201)*.01))
        if arm=="learned_magnitude_native":
            self.native=R51FullHistoryMemory("native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.))
        if arm=="gru_context_control":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22580000+seed)
                self.gru=nn.GRU(3,24,batch_first=True)
                self.readout=nn.Linear(24,1)

    def forward(self,a,e,*,zero_salience=False,force_uniform=False,
                force_top_two_tag=False,zero_native=False,
                zero_causal=False):
        tag,delta,physical,fifth,ctx,obs=components(a,e,self.fit_offset)
        if self.arm=="gru_context_control":
            x=torch.stack((tag,delta,ctx[:,None].float().expand_as(tag)),dim=-1)
            _,hidden=self.gru(x)
            b_signal=torch.tanh(self.readout(hidden[-1]).squeeze(-1))
        else:
            slope=self.slopes[ctx]
            use_top=self.arm=="tag_top_two_context" or force_top_two_tag
            use_magnitude=self.arm=="fixed_magnitude_top_two_context"
            if use_top:
                ids=r49.stable_public_top_two(tag)
                active_tag=torch.gather(tag,1,ids)
                active_delta=torch.gather(delta,1,ids)
                active_physical=torch.gather(physical,1,ids)
            elif use_magnitude:
                ids=torch.argsort(physical.abs(),descending=True,
                                  stable=True,dim=-1)[:,:2]
                active_tag=torch.gather(tag,1,ids)
                active_delta=torch.gather(delta,1,ids)
                active_physical=torch.gather(physical,1,ids)
            else:
                active_tag,active_delta,active_physical=tag,delta,physical
            logits=slope[:,None]*active_tag
            if (self.arm in ("learned_magnitude_context","learned_magnitude_native")
                and not zero_salience and not force_top_two_tag):
                logits=logits+self.salience*12.*active_physical.abs()
            weights=(torch.full_like(active_tag,1./active_tag.shape[1])
                     if force_uniform else torch.softmax(logits,dim=-1))
            b_signal=(weights*active_delta).sum(-1)
        if zero_causal:
            b_signal=torch.zeros_like(b_signal)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*b_signal)
        if self.arm=="learned_magnitude_native":
            native=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref=torch.tanh(torch.atanh(pref.clamp(-.98,.98))+
                torch.tanh(self.native_gain)*(native[:,1]-native[:,0]))
        return r47._choice_cost(ctx,pref)

def batch(records,labels,indices):
    if len(indices)!=BATCH or any(type(i) is not int or
                                  not 0<=i<TRAIN_PAIRS for i in indices):
        raise ValueError("F8R51_INVALID_TRAIN_INDEX")
    rows=[records[2*i+j] for i in indices for j in (0,1)]
    a=torch.stack([x[0] for x in rows])
    e=torch.stack([x[1] for x in rows])
    y=torch.tensor([labels[i] for i in indices],dtype=torch.long)
    return a,e,y


def heldout(model,records,labels):
    model.eval()
    correct=0
    with torch.no_grad():
        for start in range(TRAIN_PAIRS,TRAIN_PAIRS+HELDOUT_PAIRS,BATCH):
            ids=list(range(start,start+BATCH))
            rows=[records[2*i+j] for i in ids for j in (0,1)]
            a=torch.stack([x[0] for x in rows])
            e=torch.stack([x[1] for x in rows])
            y=torch.tensor([labels[i] for i in ids],dtype=torch.long)
            correct+=int((model(a,e).argmin(-1)==y).sum())
    return correct/HELDOUT_PAIRS


def score(model,panel,*,erase_B=False,erase_tag=False,
          erase_fifth=False,zero_salience=False,force_uniform=False,
          force_top_two_tag=False,zero_native=False,zero_causal=False):
    if panel not in PANELS:
        raise ValueError("F8R51_UNREGISTERED_EVALUATION_PANEL")
    ww=worlds(panel)
    raw=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                    erase_fifth=erase_fifth) for w in ww]
    a=torch.cat([x[0] for x in raw])
    e=torch.cat([x[1] for x in raw])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,zero_salience=zero_salience,
            force_uniform=force_uniform,force_top_two_tag=force_top_two_tag,
            zero_native=zero_native,zero_causal=zero_causal)
    if costs.shape!=(len(ww),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R51_NONFINITE_OR_INVALID_SCORE")
    choices=costs.argmin(-1).tolist()
    regrets=[regret(w,c) for w,c in zip(ww,choices)]
    subgroup={str(c):statistics.fmean(regrets[i]
        for i,w in enumerate(ww) if w["public_context"]==c)
        for c in (0,1)}
    return {"choices":choices,"mean_regret":statistics.fmean(regrets),
            "context_regrets":subgroup,
            "per_group_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(choices),2))}

def productive_gradient(grads):
    """An exact zero-gradient batch is valid for a negative control, not learning.

    Reject all nonfinite gradients regardless of arm, and retain a separate
    count of optimizer steps with an actual nonzero derivative.
    """
    if any(g is not None and not bool(torch.isfinite(g).all())
           for g in grads):
        raise ValueError("F8R51_NONFINITE_GRADIENT")
    return any(g is not None and bool(torch.any(g!=0))
               for g in grads)


def train_arm(arm,seed,offset,records,labels,draws):
    model=Policy(arm,seed,offset)
    groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            lr=.02 if name in ("slopes","salience","gain","native_gain") else .001
            groups.append({"params":[p],"lr":lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=0.)
    productive_steps=0
    magnitude_grad_ever=False
    both_context_grad_ever=[False,False]
    traces=[]
    started=time.perf_counter()
    for step,ids in enumerate(draws,1):
        a,e,y=batch(records,labels,ids)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R51_INVALID_TRAINING_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        productive=productive_gradient(grads)
        productive_steps+=int(productive)
        mg=(float(model.salience.grad.detach())
            if hasattr(model,"salience") and model.salience.grad is not None
            else None)
        if arm=="learned_magnitude_context":
            if mg is not None and mg!=0:
                magnitude_grad_ever=True
            for k in (0,1):
                both_context_grad_ever[k] |= (
                    model.slopes.grad is not None and
                    float(model.slopes.grad[k])!=0.)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if step in CHECKPOINTS:
            traces.append({"step":step,"loss":float(loss.detach()),
                "heldout_accuracy":heldout(model,records,labels),
                "slopes":model.slopes.detach().tolist()
                  if hasattr(model,"slopes") else None,
                "salience":float(model.salience.detach())
                  if hasattr(model,"salience") else None,
                "magnitude_gradient":mg})
    if arm=="learned_magnitude_context" and (
        not magnitude_grad_ever or not all(both_context_grad_ever)
        or productive_steps==0):
        raise ValueError("F8R51_MAIN_DID_NOT_LEARN_FROM_REAL_GRADIENT")
    panels={p:{"active":score(model,p)} for p in PANELS}
    for label,kwargs in (
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("zero_salience",{"zero_salience":True}),
        ("force_uniform",{"force_uniform":True}),
        ("force_top_two_tag",{"force_top_two_tag":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_causal",{"zero_causal":True})):
        panels["ood"][label]=score(model,"ood",**kwargs)
    cold=Policy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r40._hash(cold)!=r40._hash(model) or
        any(score(cold,p)["choices"]!=panels[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R51_COLD_MODEL_DRIFT")
    return {"arm":arm,"seed":seed,"actual_updates":UPDATES,
        "productive_gradient_steps":productive_steps,
        "main_salience_received_gradient":magnitude_grad_ever
           if arm=="learned_magnitude_context" else None,
        "trainable_parameters":sum(p.numel() for p in model.parameters()
                                   if p.requires_grad),
        "train_seconds":time.perf_counter()-started,
        "traces":traces,"scores":panels,"cold_restore":True}

def bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED)
    vals=[]
    for _ in range(BOOT_DRAWS):
        ids=[rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        vals.append(statistics.fmean(b[i]-a[i] for i in ids))
    return sorted(vals)[int(.025*BOOT_DRAWS)]


def judge(models,freeze):
    limits=freeze["frozen_score"]
    lookup={(m["arm"],m["seed"]):m for m in models}
    judged=[]
    for seed in SEEDS:
        main=lookup["learned_magnitude_context",seed]
        tag=lookup["tag_top_two_context",seed]
        magnitude=lookup["fixed_magnitude_top_two_context",seed]
        active=main["scores"]["ood"]["active"]
        cal=main["scores"]["cal"]["active"]
        tag_ood=tag["scores"]["ood"]["active"]
        mag_ood=magnitude["scores"]["ood"]["active"]
        ablations={key:main["scores"]["ood"][key]["mean_regret"]-
                   active["mean_regret"] for key in
                   ("erase_B","erase_tag","erase_fifth","zero_salience",
                    "force_uniform","force_top_two_tag","zero_native","zero_causal")}
        gain=tag_ood["mean_regret"]-active["mean_regret"]
        lower=bootstrap_lower(active["per_group_regret"],
                              tag_ood["per_group_regret"])
        s0,s1=main["traces"][-1]["slopes"]
        final_mag=main["traces"][-1]["salience"]
        pass_now=bool(
            main["traces"][-1]["heldout_accuracy"]>=limits["min_fit_heldout_accuracy"]
            and cal["mean_regret"]<=limits["max_cal_regret"]
            and active["mean_regret"]<=limits["max_ood_regret"]
            and all(v<=limits["max_each_ood_context_regret"]
                    for v in active["context_regrets"].values())
            and cal["gate_pair_flips"]>=limits["min_cal_gate_pair_flips"]
            and active["gate_pair_flips"]>=limits["min_ood_gate_pair_flips"]
            and ablations["erase_B"]>=limits["min_ood_penalty_B_erase"]
            and ablations["erase_tag"]>=limits["min_ood_penalty_tag_erase"]
            and ablations["erase_fifth"]>=limits["min_ood_penalty_fifth_erase"]
            and ablations["zero_salience"]>=limits["min_ood_penalty_magnitude_zero"]
            and ablations["force_top_two_tag"]>=limits["min_ood_penalty_force_tag_top_two"]
            and gain>=limits["min_ood_gain_vs_trained_public_tag_top_two"]
            and lower>0 and s0>=limits["context0_slope_min"]
            and s1<=limits["context1_slope_max"] and final_mag>0
            and main["main_salience_received_gradient"])
        judged.append({"seed":seed,"frozen_DEV_pass":pass_now,
            "heldout_FIT":main["traces"][-1]["heldout_accuracy"],
            "CAL_regret":cal["mean_regret"],"OOD_regret":active["mean_regret"],
            "OOD_context_regret":active["context_regrets"],
            "OOD_flips":active["gate_pair_flips"],
            "learned_slopes":[s0,s1],"learned_salience":final_mag,
            "OOD_gain_vs_tag_top_two":gain,"bootstrap_lower_gain":lower,
            "magnitude_heuristic_OOD_regret":mag_ood["mean_regret"],
            "OOD_ablation_penalties":ablations})
    return {"two_seed_frozen_DEV_pass":all(x["frozen_DEV_pass"] for x in judged),
            "seeds":judged,"positive_Native_backbone_gain_verified":False,
            "independent_scientific_pass":False,
            "full_capacity_compute_matched":False,
            "source_neutral_synthetic_only":True}

def run():
    frozen=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    provenance={panel:parity(panel) for panel in ("fit",)+PANELS}
    records,labels=factual_fit()
    offset=calibration(records)
    ceiling=positive_observation_reference(offset)
    rng=torch.Generator(device="cpu")
    rng.manual_seed(DRAW_SEED)
    draws=torch.randint(0,TRAIN_PAIRS,(UPDATES,BATCH),
                         generator=rng).tolist()
    models=[train_arm(arm,seed,offset,records,labels,draws)
            for arm,seed in ARMS]
    status=judge(models,frozen)
    return {"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "parent":PARENT,"source_neutral_DEV_only":True,
            "factual_FIT_selected_returns":len(records),
            "FIT_training_pairs":TRAIN_PAIRS,
            "FIT_heldout_pairs":HELDOUT_PAIRS,
            "FIT_only_calibrated_probe_offset":offset,
            "B_parity":provenance,
            "observable_reference":ceiling,
            "neural_optimizer_updates":len(ARMS)*UPDATES,
            "models":models,"judgment":status,
            "independent_scientific_pass":False,
            "external_original_first_use":False,"deployment":"NONE"}


def retain_ci_receipt(result):
    if os.getenv("GITHUB_ACTIONS")!="true":return
    path=ROOT/"local_runs/p85mr_f8r51_learned_eligibility_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError("F8R51_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def deny_external_original_first_use(*args,**kwargs):
    raise PermissionError("F8R51_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
