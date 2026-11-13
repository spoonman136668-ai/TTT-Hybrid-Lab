"""F8-R48 contextual cue-role reversal: prospective selected-observation DEV.

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

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r48_context_role_switch_freeze.json"
FREEZE_BLOB="3578147275084ad9dc2d488f5b958897da2d994a"
SCHEMA="ttt-hybrid.f8r48-contextual-cue-role-switch-freeze.v1"
PARENT="e629ffd9e44d118251739ec4cbc9b33512333c3f"
SALT={"fit":22210000,"cal":22220000,"ood":22230000}
AMP={"fit":.24,"cal":.20,"ood":.16}
NUIS={"fit":.006,"cal":.006,"ood":.012}
SEEDS=(985201,985207)
ARM_NAMES=("context_key_selector","global_key_selector",
           "uniform_key_selector","context_key_native","gru_context_control")
ARMS=tuple((name,seed) for name in ARM_NAMES for seed in SEEDS)
FIT_GROUPS, TRAIN_GROUPS, EVAL_GROUPS=24,16,8
TRAIN_PAIRS, HELDOUT_PAIRS, BATCH, UPDATES=128,64,16,256
DRAW_SEED,BOOT_SEED,BOOT_DRAWS=22249999,22269999,10000
CHECKPOINTS=(1,32,64,128,192,256)
PANELS=("cal","ood")
PROBE,CUE_TAG,DECOY_TAG=.42,.42,.18


def git_blob(raw):
    return hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if git_blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R48_FREEZE_BLOB_DRIFT")
    f=json.loads(raw)
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R48_SOURCE_BIRTH_TRAINING_OR_SCORING" or
        f["parent"]["head"]!=PARENT or
        f["source"]["salts"]!=SALT or
        f["source"]["amplitude_by_panel"]!=AMP or
        f["source"]["nuisance_by_panel"]!=NUIS or
        f["source"]["cohorts"]["fit"]["selected_factual_returns"]!=384 or
        f["methods"]["train_updates_each"]!=UPDATES or
        f["methods"]["sample_seed"]!=DRAW_SEED or
        f["frozen_dev_score"]["bootstrap"]["seed"]!=BOOT_SEED or
        f["frozen_dev_score"]["independent_scientific_pass"] is not False or
        tuple(f["methods"]["original_seeds"])!=SEEDS or
        tuple((row["name"],tuple(row["seeds"]))
              for row in f["methods"]["arms"])!=tuple((name,SEEDS) for name in ARM_NAMES) or
        any(f["authority"][k] for k in ("new_workflow","new_runner",
            "new_controller","paid_api","original_external_first_use",
            "previously_scored_original_source_first_use","GPU_training",
            "merge","promotion","deployment"))):
        raise ValueError("F8R48_PROTOCOL_OR_AUTHORITY_DRIFT")
    r47.locked()
    return f


def positions(panel, group):
    if (panel not in SALT or type(group) is not int or
        not 0<=group<(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)):
        raise ValueError("F8R48_UNREGISTERED_GROUP")
    length=16 if panel=="ood" else 8
    raw=hashlib.sha256(f"F8-R48/{panel}/{group}/{SALT[panel]}".encode()).digest()
    cue=int.from_bytes(raw[:4],"big")%length
    decoy=int.from_bytes(raw[4:8],"big")%(length-1)
    if decoy>=cue:decoy+=1
    return cue,decoy


def world(panel,group,context,q_sign,gate):
    positions(panel,group)
    if context not in (0,1) or q_sign not in (-1,1) or gate not in (-1,1):
        raise ValueError("F8R48_INVALID_FACTORIAL_WORLD")
    return {"panel":panel,"group":group,"public_context":context,
            "q_evaluator_only":q_sign,"gate_evaluator_only":gate,
            "return_state":(.61,.73)[group%2],"goal":1.}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R48_UNREGISTERED_PANEL")
    return [world(panel,g,c,q,h)
            for g in range(FIT_GROUPS if panel=="fit" else EVAL_GROUPS)
            for c in (0,1) for q in (-1,1) for h in (-1,1)]


def B_selected(w):
    p,g,c=w["panel"],w["group"],w["public_context"]
    cue,decoy=positions(p,g)
    length=16 if p=="ood" else 8
    rows=[]
    for t in range(length):
        if t==cue:
            tag,delta=(CUE_TAG if c==0 else DECOY_TAG),AMP[p]*w["gate_evaluator_only"]
        elif t==decoy:
            tag,delta=(DECOY_TAG if c==0 else CUE_TAG),-AMP[p]*w["gate_evaluator_only"]
        else:
            tag=0.
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
        if swap_B and tag>0:
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
                raise ValueError("F8R48_PUBLIC_ACTION_KEY_LEAK")
            delta=(e1[:,5:-1,:].mean(1)-
                   e2[:,5:-1,:].mean(1)).abs().max().item()
            worst=max(worst,delta)
            if delta>1e-7:
                raise ValueError("F8R48_FULL_B_MEAN_LEAK")
            if torch.equal(e1[:,5:-1,:],e2[:,5:-1,:]):
                raise ValueError("F8R48_B_OBSERVATIONS_INSEPARABLE")
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
        raise ValueError("F8R48_FACTUAL_SOURCE_IMBALANCE")
    return records,labels


def calibration(records):
    if len(records)!=384:
        raise ValueError("F8R48_CALIBRATION_UNQUALIFIED")
    after=[PROBE+r31._physical_delta(records[2*i][1][4,0])
           for i in range(TRAIN_PAIRS)]
    offset=statistics.fmean(after)
    if not math.isfinite(offset):
        raise ValueError("F8R48_CALIBRATION_INVALID")
    return offset


def regret(w,choice):
    if choice not in (0,1):
        raise ValueError("F8R48_INVALID_RETURN_ACTION")
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
            tagged=[row for row in tagged if row[0]>.08]
            if len(tagged)!=2:
                raise ValueError("F8R48_PUBLIC_ROLE_IDENTIFICATION_FAILED")
            causal=(max(tagged,key=lambda x:x[0])[1]
                if ctx==0 else min(tagged,key=lambda x:x[0])[1])
            choice=ctx if fifth*causal>0 else 1-ctx
            errors+=int(regret(w,choice)>1e-10)
        out[panel]={"worlds":len(ww),"mistakes":errors}
        if errors:
            raise ValueError("F8R48_SOURCE_FEASIBILITY_FAILED_BEFORE_TRAINING")
    return out


def components(a,e,offset):
    tag,delta,fifth,context,observed=r47._observed_components(a,e,offset)
    return tag,delta,fifth,context,observed


class Policy(nn.Module):
    def __init__(self,arm,seed,offset):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R48_UNREGISTERED_MODEL")
        self.arm,self.seed=arm,seed
        self.register_buffer("fit_offset",torch.tensor(offset,dtype=torch.float32))
        if arm in ("context_key_selector","context_key_native"):
            self.slopes=nn.Parameter(torch.zeros(2,dtype=torch.float32))
        if arm=="global_key_selector":
            self.slope=nn.Parameter(torch.tensor(0.,dtype=torch.float32))
        self.gain=nn.Parameter(torch.tensor(3.+(seed-985201)*.01))
        if arm=="context_key_native":
            self.native=r34.FactualMemory("native_factual_memory",seed)
            self.native_gain=nn.Parameter(torch.tensor(0.))
        if arm=="gru_context_control":
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(22280000+seed)
                self.gru=nn.GRU(3,24,batch_first=True)
                self.readout=nn.Linear(24,1)

    def forward(self,a,e,*,force_uniform=False,
                force_context0_slope=False,
                zero_native=False,zero_key=False):
        tag,delta,fifth,ctx,obs=components(a,e,self.fit_offset)
        if self.arm=="gru_context_control":
            x=torch.stack((tag,delta,
                           ctx[:,None].float().expand_as(tag)),dim=-1)
            _,hidden=self.gru(x)
            b_signal=torch.tanh(self.readout(hidden[-1]).squeeze(-1))
        else:
            if self.arm in ("context_key_selector","context_key_native"):
                contexts=ctx*int(not force_context0_slope)
                slope=self.slopes[contexts]
            elif self.arm=="global_key_selector":
                slope=self.slope.expand_as(fifth)
            else:
                slope=torch.zeros_like(fifth)
            weights=(torch.full_like(tag,1./tag.shape[1]) if force_uniform
                     else torch.softmax(slope[:,None]*tag,dim=-1))
            b_signal=(weights*delta).sum(-1)
        if zero_key:
            b_signal=torch.zeros_like(b_signal)
        pref=torch.tanh(F.softplus(self.gain)*40.*fifth*b_signal)
        if self.arm=="context_key_native":
            native=self.native(a,obs,ablate_backbone=zero_native).reshape(-1,2)
            pref=torch.tanh(torch.atanh(pref.clamp(-.98,.98))+
                            torch.tanh(self.native_gain)*
                            (native[:,1]-native[:,0]))
        return r47._choice_cost(ctx,pref)


def batch(records,labels,indices):
    if len(indices)!=BATCH or any(type(i) is not int or
                                  not 0<=i<TRAIN_PAIRS for i in indices):
        raise ValueError("F8R48_INVALID_TRAIN_INDEX")
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
          erase_fifth=False,swap_B=False,force_uniform=False,
          force_context0_slope=False,zero_native=False,zero_key=False):
    if panel not in PANELS:
        raise ValueError("F8R48_UNREGISTERED_SCORE_PANEL")
    ww=worlds(panel)
    raw=[candidates(w,erase_B=erase_B,erase_tag=erase_tag,
                    erase_fifth=erase_fifth,swap_B=swap_B) for w in ww]
    a=torch.cat([x[0] for x in raw]);e=torch.cat([x[1] for x in raw])
    model.eval()
    with torch.no_grad():
        costs=model(a,e,force_uniform=force_uniform,
                    force_context0_slope=force_context0_slope,
                    zero_native=zero_native,zero_key=zero_key)
    if costs.shape!=(len(ww),2) or not bool(torch.isfinite(costs).all()):
        raise ValueError("F8R48_INVALID_SCORE")
    choices=costs.argmin(-1).tolist()
    regrets=[regret(w,c) for w,c in zip(ww,choices)]
    subgroup={}
    for ctx in (0,1):
        selected=[i for i,w in enumerate(ww) if w["public_context"]==ctx]
        subgroup[str(ctx)]=statistics.fmean(regrets[i] for i in selected)
    return {"choices":choices,
            "mean_regret":statistics.fmean(regrets),
            "context_regrets":subgroup,
            "per_group_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(EVAL_GROUPS)],
            "gate_pair_flips":sum(choices[i]!=choices[i+1]
                                  for i in range(0,len(choices),2))}


def train_arm(arm,seed,offset,records,labels,draws):
    model=Policy(arm,seed,offset)
    param_groups=[]
    for name,p in model.named_parameters():
        if p.requires_grad:
            lr=.02 if name in ("slopes","slope","gain","native_gain") else .001
            param_groups.append({"params":[p],"lr":lr})
    opt=torch.optim.AdamW(param_groups,weight_decay=0.)
    traces=[]
    start=time.perf_counter()
    for step,indices in enumerate(draws,1):
        a,e,y=batch(records,labels,indices)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss=F.cross_entropy(-model(a,e),y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R48_NONFINITE_LOSS")
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad]
        if (not any(g is not None and bool(torch.any(g!=0)) for g in grads)
            or any(g is not None and not bool(torch.isfinite(g).all()) for g in grads)):
            raise ValueError("F8R48_NO_VALID_REAL_GRADIENT")
        g0=(model.slopes.grad.detach().tolist()
            if hasattr(model,"slopes") and model.slopes.grad is not None
            else None)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],1.)
        opt.step()
        if step in CHECKPOINTS:
            traces.append({"step":step,
                           "FIT_cross_entropy":float(loss.detach()),
                           "heldout_accuracy":heldout(model,records,labels),
                           "context_slopes":model.slopes.detach().tolist()
                             if hasattr(model,"slopes") else None,
                           "global_slope":float(model.slope.detach())
                             if hasattr(model,"slope") else None,
                           "context_slope_gradients":g0})
    results={p:{"active":score(model,p)} for p in PANELS}
    for name,kwargs in (
        ("erase_B",{"erase_B":True}),
        ("erase_tag",{"erase_tag":True}),
        ("erase_fifth",{"erase_fifth":True}),
        ("swap_B",{"swap_B":True}),
        ("force_uniform",{"force_uniform":True}),
        ("force_context0_slope",{"force_context0_slope":True}),
        ("zero_native",{"zero_native":True}),
        ("zero_key",{"zero_key":True})):
        results["ood"][name]=score(model,"ood",**kwargs)
    cold=Policy(arm,seed,offset)
    cold.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (r40._hash(cold)!=r40._hash(model) or
        any(score(cold,p)["choices"]!=results[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R48_COLD_RESTORE_FAILED")
    return {"arm":arm,"seed":seed,"actual_updates":UPDATES,
            "trainable_parameters":sum(x.numel() for x in model.parameters()
                                       if x.requires_grad),
            "train_seconds":time.perf_counter()-start,
            "traces":traces,"scores":results,"cold_restore":True}


def bootstrap_lower(a,b):
    rng=random.Random(BOOT_SEED)
    vals=[]
    for _ in range(BOOT_DRAWS):
        ids=[rng.randrange(EVAL_GROUPS) for _ in range(EVAL_GROUPS)]
        vals.append(statistics.fmean(b[i]-a[i] for i in ids))
    return sorted(vals)[int(.025*BOOT_DRAWS)]


def judge(models,frozen):
    t=frozen["frozen_dev_score"]
    by={(x["arm"],x["seed"]):x for x in models}
    output=[]
    for seed in SEEDS:
        selected=by["context_key_selector",seed]
        global_model=by["global_key_selector",seed]
        active=selected["scores"]["ood"]["active"]
        cal=selected["scores"]["cal"]["active"]
        gc=global_model["scores"]["ood"]["active"]
        gain=gc["mean_regret"]-active["mean_regret"]
        lower=bootstrap_lower(active["per_group_regret"],gc["per_group_regret"])
        penalties={name:selected["scores"]["ood"][name]["mean_regret"]-
                    active["mean_regret"]
                   for name in ("erase_B","erase_tag","erase_fifth",
                     "swap_B","force_uniform","force_context0_slope",
                     "zero_native","zero_key")}
        slope0,slope1=selected["traces"][-1]["context_slopes"]
        pass_now=bool(
            selected["traces"][-1]["heldout_accuracy"]>=t["min_FIT_heldout_accuracy"]
            and cal["mean_regret"]<=t["max_CAL_mean_regret"]
            and active["mean_regret"]<=t["max_OOD_mean_regret"]
            and all(v<=t["max_each_context_OOD_regret"]
                    for v in active["context_regrets"].values())
            and cal["gate_pair_flips"]>=t["min_CAL_gate_flips"]
            and active["gate_pair_flips"]>=t["min_OOD_gate_flips"]
            and penalties["erase_B"]>=t["min_ood_penalty_B_erasure"]
            and penalties["erase_tag"]>=t["min_ood_penalty_tag_erasure"]
            and penalties["erase_fifth"]>=t["min_ood_penalty_fifth_erasure"]
            and penalties["force_uniform"]>=t["min_ood_penalty_uniform"]
            and penalties["force_context0_slope"]>=t["min_ood_penalty_context_slope_erasure"]
            and gain>=t["min_ood_gain_vs_best_single_global_slope"]
            and slope0>=t["min_learned_context_0_slope"]
            and slope1<=t["max_learned_context_1_slope"]
            and lower>0)
        output.append({"seed":seed,"frozen_dev_pass":pass_now,
            "heldout_FIT":selected["traces"][-1]["heldout_accuracy"],
            "CAL_regret":cal["mean_regret"],
            "OOD_regret":active["mean_regret"],
            "OOD_regret_by_context":active["context_regrets"],
            "OOD_gate_flips":active["gate_pair_flips"],
            "learned_context_slopes":[slope0,slope1],
            "global_control_OOD_regret":gc["mean_regret"],
            "gain_vs_global":gain,
            "bootstrap_lower_gain_vs_global":lower,
            "OOD_ablation_regret_penalties":penalties})
    return {"two_seed_DEV_pass":all(z["frozen_dev_pass"] for z in output),
            "seeds":output,"native_specific_gain_verified":False,
            "whole_process_capacity_compute_matched":False,
            "independent_scientific_pass":False}


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
    path=ROOT/"local_runs/p85mr_f8r48_context_role_switch_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():raise FileExistsError("F8R48_RECEIPT_ALREADY_EXISTS")
    path.write_text(json.dumps(result,indent=2,sort_keys=True,
                               allow_nan=False)+"\n")


def deny_external_original_first_use(*args,**kwargs):
    raise PermissionError("F8R48_EXTERNAL_ORIGINAL_FIRSTUSE_DENIED")
