"""F8-R41: frozen-binder Native/GRU/SSM history-conditional causal falsifier.

The independent B-history measurement controls a later intervention-sign
reversal. Early and fifth observations cannot distinguish gate ±1. Every FIT
label comes from two separately selected factual return outcomes. Scoring
counterfactual alternatives never enters model inputs or optimization.
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
from experiments import p85mr_f8r36_balanced_choice as r36
from experiments import p85mr_f8r37_informative_strata as r37
from experiments import p85mr_f8r38_learned_binding as r38
from experiments import p85mr_f8r39_binding_necessity as r39
from experiments import p85mr_f8r40_native_fusion as r40

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/p85mr_f8r41_history_conditional_freeze.json"
FREEZE_BLOB="6d1af794e5747768719549b108f788df475a233e"
SCHEMA="ttt-hybrid.f8r41-b-history-conditional-native-incremental.v1"
SEEDS=(985201,985207)
ARMS=tuple((arm,seed) for arm in ("native_history","history_only")
           for seed in SEEDS)+(("gru_history",SEEDS[0]),("ssm_history",SEEDS[0]))
STEPS=384
BATCH=16
TRAIN_PAIRS=256
VAL_PAIRS=128
PANELS=("dev","ood")
GROUPS=16
GOAL=1.
SALT={"fit":21510000,"dev":21520000,"ood":21530000}
STATES={"fit":(.61,.73),"dev":(.61,.73),"ood":(.60,.75)}
MAG={"fit":1.,"dev":1.02,"ood":.90}
B_STEPS={"fit":8,"dev":8,"ood":16}
B_MARK={"fit":2,"dev":2,"ood":5}
BOOT_SEED=21569999
BOOT_DRAW=10000


def _blob(data):
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def locked():
    raw=FREEZE.read_bytes()
    if _blob(raw)!=FREEZE_BLOB:
        raise ValueError("F8R41_PROTOCOL_BLOB_DRIFT")
    f=json.loads(raw)
    s,t,d,m,a=(f["source"],f["training"],f["preregistered_score"],
               f["models"],f["authority"])
    if (f["schema"]!=SCHEMA or
        f["stage"]!="FROZEN_BEFORE_R41_IMPLEMENTATION_TRAINING_AND_RESULTS" or
        f["classification"]!="SOURCE_NEUTRAL_HISTORICAL_CAUSAL_RECURRENT_NEURAL_DEVELOPMENT_ONLY" or
        f["parent"]["head"]!="c96316f33bcc0f89b6f032d87a9a390f9db6922a" or
        f["parent"]["r40_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r40_native_fusion.py").read_bytes()) or
        f["parent"]["r40_freeze_blob"]!=r40.FREEZE_BLOB or
        f["parent"]["r39_binder_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r39_binding_necessity.py").read_bytes()) or
        f["parent"]["r38_binder_source_blob"]!=_blob(
            (ROOT/"experiments/p85mr_f8r38_learned_binding.py").read_bytes()) or
        s["fit"]["episodes"]!=768 or s["fit"]["train_paired_worlds"]!=TRAIN_PAIRS or
        s["fit"]["validation_paired_worlds"]!=VAL_PAIRS or
        s["B_marker_location_fit"]!=B_MARK["fit"] or
        s["B_marker_location_dev"]!=B_MARK["dev"] or
        s["B_marker_location_ood"]!=B_MARK["ood"] or
        s["dev"]["worlds"]!=128 or s["ood"]["worlds"]!=128 or
        m["instances"]!=len(ARMS) or
        m["pretrained_binder_seeds"]!=list(SEEDS) or
        t["new_updates_each"]!=STEPS or
        t["total_new_updates"]!=STEPS*len(ARMS) or
        t["training_pairs_per_update"]!=BATCH or
        t["sampling_seed"]!=21549999 or
        t["checkpoint_steps"]!=[192,384] or
        d["native_seed_both_panels_pass_min"]!=2 or
        d["native_history_gate_pair_flips_min_each"]!=56 or
        d["bootstrap_seed"]!=BOOT_SEED or
        d["bootstrap_group_draws"]!=BOOT_DRAW or
        d["independent_scientific_pass"] is not False or
        any(v is not False for v in a.values())):
        raise ValueError("F8R41_FROZEN_SCOPE_DRIFT")
    r40.locked()
    return f


def _base(s):
    return .25*s-.08+.05*math.sin(3*s)


def _law(s,action,context,q,panel):
    """Generator-only law: never imported into a model or training loss."""
    if panel=="ood":
        return _base(s)+(2.7*q*s*(s*s-.15**2) if action==context else 0.)
    return r33.r32.after(s,action,context,q)


def _B(panel,g,gate):
    if (panel not in SALT or type(g) is not int or
        not 0<=g<(48 if panel=="fit" else GROUPS) or gate not in (-1,1)):
        raise ValueError("F8R41_UNREGISTERED_B_HISTORY")
    values=[]
    for t in range(B_STEPS[panel]):
        nuisance=.008*math.sin((SALT[panel]+g)*.19+t*.31)+.003*((t+g)%3)-.003
        v=nuisance+(.27*gate if t==B_MARK[panel] else 0.)
        values.append((t%2,round(v,8)))
    return tuple(values)


def world(panel,g,context,sign,gate):
    if (panel not in SALT or type(g) is not int or
        not 0<=g<(48 if panel=="fit" else GROUPS) or
        context not in (0,1) or sign not in (-1,1) or gate not in (-1,1)):
        raise ValueError("F8R41_UNREGISTERED_WORLD")
    return {"panel":panel,"group":g,"public_context":context,
        "q_evaluator_only":sign*MAG[panel],
        "gate_evaluator_only":gate,
        "return_state":STATES[panel][g%2],"goal":GOAL,
        "b_history":_B(panel,g,gate)}


def worlds(panel):
    if panel not in SALT:
        raise ValueError("F8R41_UNREGISTERED_PANEL")
    limit=48 if panel=="fit" else GROUPS
    return [world(panel,g,c,sign,gate) for g in range(limit)
            for c in (0,1) for sign in (-1,1) for gate in (-1,1)]


def _factual_history(w,*,erase_b=False,erase_fifth=False):
    p,c,q=w["panel"],w["public_context"],w["q_evaluator_only"]
    rows=[r33.b._record(s,a,_law(s,a,c,q,p),c) for s,a in r33.r32.EARLY]
    rows.append(r33.b._record(r33.PROBE,c,_law(r33.PROBE,c,c,q,p),c))
    for a,value in w["b_history"]:
        rows.append(r33.b._record(0.,a,value,c))
    a=torch.stack([x[0] for x in rows])
    e=torch.stack([x[1] for x in rows])
    if erase_b:
        e[5+B_MARK[p]].zero_()
    if erase_fifth:
        e[4].zero_()
    return a,e


def candidates(w,*,erase_b=False,erase_fifth=False,
               opposite_gate_measurement=False):
    a,e=_factual_history(w,erase_b=erase_b,erase_fifth=erase_fifth)
    if opposite_gate_measurement:
        opposite=_B(w["panel"],w["group"],-w["gate_evaluator_only"])
        e[5+B_MARK[w["panel"]]]=r33.b._record(
            0.,opposite[B_MARK[w["panel"]]][0],
            opposite[B_MARK[w["panel"]]][1],w["public_context"])[1]
    choices=[]
    for selected in (0,1):
        key=r33.b._record(w["return_state"],selected,w["return_state"],
                           w["public_context"])[0]
        choices.append((torch.cat((a,key.unsqueeze(0))),
                        torch.cat((e,torch.zeros((1,16))))))
    return (torch.stack([x[0] for x in choices]),
            torch.stack([x[1] for x in choices]))


def fit_cohort():
    locked()
    rows=[]
    correct=[]
    for w in worlds("fit"):
        # The two outcomes are from independently selected factual actions
        # in two equivalent generated worlds; neither is a model-visible
        # alternative-action oracle target.
        selected=[]
        for ret in (0,1):
            a,e=_factual_history(w)
            actual=_law(w["return_state"],ret,w["public_context"],
                        w["q_evaluator_only"]*w["gate_evaluator_only"],"fit")
            current=r33.b._record(w["return_state"],ret,actual,
                                  w["public_context"])
            action=torch.cat((a,current[0].unsqueeze(0)))
            effect=torch.cat((e,current[1].unsqueeze(0)))
            # Derive the factual cost from actual selected observed effect;
            # never use evaluator-only hidden generator q/gate as label.
            actual_from_effect=w["return_state"]+r31._physical_delta(effect[-1,0])
            selected.append((action,effect,(actual_from_effect-GOAL)**2))
        rows.extend({"action":a,"effect":e}
                    for a,e,_ in selected)
        correct.append(min((0,1),key=lambda i:(selected[i][2],i)))
    if len(rows)!=768 or len(correct)!=384:
        raise ValueError("F8R41_FACTUAL_FIT_COUNT")
    return rows,correct


def _training_batch(fit,labels,pair_ids):
    if len(pair_ids)!=BATCH or any(type(i) is not int or
                                  not 0<=i<TRAIN_PAIRS for i in pair_ids):
        raise ValueError("F8R41_INVALID_TRAIN_PAIRS")
    rows=[fit[2*i+j] for i in pair_ids for j in (0,1)]
    return (torch.stack([x["action"] for x in rows]),
            torch.stack([x["effect"] for x in rows]),
            torch.tensor([labels[i] for i in pair_ids],dtype=torch.long))


class HistoryAdapter(nn.Module):
    """A frozen binder with each R34 factual-memory backbone as residual."""

    def __init__(self,arm,seed,binder):
        super().__init__()
        if (arm,seed) not in ARMS:
            raise ValueError("F8R41_UNREGISTERED_MODEL")
        self.arm=arm
        self.binder=copy.deepcopy(binder)
        for v in self.binder.parameters():
            v.requires_grad_(False)
        self.binder.eval()
        if arm=="native_history":
            self.recurrent=r34.FactualMemory("native_factual_memory",seed)
        elif arm=="history_only":
            # Same exact recurrent memory/readout initialization as Native,
            # but remove the Native backbone BEFORE learning: source-neutral
            # equivalent of existing R34 history_only even for seed 985207.
            self.recurrent=r34.FactualMemory("native_factual_memory",seed)
            self.recurrent.backbone=None
            self.recurrent.base_arm=None
            self.recurrent.arm="history_only"
        else:
            self.recurrent=r34.FactualMemory(
                {"gru_history":"gru_factual_memory",
                 "ssm_history":"ssm_factual_memory"}[arm],seed)

    def train(self,mode=True):
        super().train(mode)
        self.binder.eval()
        return self

    def forward(self,a,e,*,zero_history=False,zero_backbone=False):
        if a.ndim!=3 or e.shape!=a.shape or a.shape[-1]!=16 or a.shape[1] not in (14,22):
            raise ValueError("F8R41_INVALID_FACTUAL_P80")
        observed=e.clone()
        observed[:,-1,:].zero_()
        base=r38.predict(self.binder,"binder",a,observed)
        if zero_history:
            return base
        return base+self.recurrent(a,observed,ablate_backbone=zero_backbone)


def _choices(model,ww,*,erase_b=False,erase_fifth=False,swap_b=False,
             zero_history=False,zero_backbone=False):
    packs=[candidates(w,erase_b=erase_b,erase_fifth=erase_fifth,
                      opposite_gate_measurement=swap_b) for w in ww]
    aa=torch.cat([a for a,e in packs])
    ee=torch.cat([e for a,e in packs])
    model.eval()
    with torch.no_grad():
        score=model(aa,ee,zero_history=zero_history,
                    zero_backbone=zero_backbone)
    values=score.detach().cpu().tolist()
    if len(values)!=len(ww)*2 or not all(math.isfinite(v) for v in values):
        raise ValueError("F8R41_INVALID_PREACTION_VALUES")
    return [min((0,1),key=lambda action:(values[2*i+action],action))
            for i in range(len(ww))]


def _regrets(ww,actions):
    if len(ww)!=len(actions):
        raise ValueError("F8R41_INVALID_EVALUATOR_ACTIONS")
    result=[]
    for w,chosen in zip(ww,actions):
        targets=[(_law(w["return_state"],i,w["public_context"],
                       w["q_evaluator_only"]*w["gate_evaluator_only"],
                       w["panel"])-GOAL)**2 for i in (0,1)]
        result.append(max(0.,targets[chosen]-min(targets)))
    return result


def _score(model,panel,**kwargs):
    ww=worlds(panel)
    decision=_choices(model,ww,**kwargs)
    regrets=_regrets(ww,decision)
    return {"choices":decision,"mean_regret":statistics.fmean(regrets),
            "pergroup_regret":[statistics.fmean(regrets[8*g:8*g+8])
                                for g in range(GROUPS)],
            "history_gate_pair_flips":sum(
               decision[k]!=decision[k+1] for k in range(0,len(decision),2))}


def _fitted_conventional_baseline(fit,ww,panel):
    # Explicitly hand-coded positive reference, NOT a trained neural result:
    # all parameters measured from factual FIT observations, no hidden labels.
    safe=[]
    for k in range(TRAIN_PAIRS):
        effect=fit[2*k]["effect"][4,0]
        safe.append(r31._physical_delta(effect))
    offset=statistics.fmean(safe)
    correct=[]
    for w in ww:
        a,e=candidates(w)
        fifth=r31._physical_delta(e[0,4,0])
        gate_measure=r31._physical_delta(e[0,5+B_MARK[panel],0])
        positive=(fifth-offset)*gate_measure>0.
        c=r31._decode_key(a[0,-1])[1]
        correct.append(c if positive else 1-c)
    regrets=_regrets(ww,correct)
    return {"choices":correct,"mean_regret":statistics.fmean(regrets),
            "history_gate_pair_flips":sum(
               correct[i]!=correct[i+1] for i in range(0,len(correct),2)),
            "factual_fit_probe_offset":offset}


def _validation(model,fit,labels):
    ids=list(range(TRAIN_PAIRS,TRAIN_PAIRS+VAL_PAIRS))
    model.eval()
    right=0
    losses=[]
    with torch.no_grad():
        for start in range(0,len(ids),16):
            chosen=ids[start:start+16]
            rows=[fit[2*i+j] for i in chosen for j in (0,1)]
            a=torch.stack([x["action"] for x in rows])
            e=torch.stack([x["effect"] for x in rows])
            target=torch.tensor([labels[i] for i in chosen],dtype=torch.long)
            pred=model(a,e).reshape(len(chosen),2)
            right+=int((pred.argmin(dim=1)==target).sum())
            losses.append((len(chosen),float(F.cross_entropy(-pred,target))))
    return {"pairs":len(ids),"choice_accuracy":right/len(ids),
            "choice_CE":sum(n*v for n,v in losses)/len(ids)}


def _weight_hash(model):
    return r40._hash(model)


def _train(arm,seed,binder,fit,labels,samples):
    model=HistoryAdapter(arm,seed,binder)
    before_native=_weight_hash(model.recurrent)
    frozen_sha=_weight_hash(model.binder)
    opt=torch.optim.AdamW([v for v in model.parameters()
                           if v.requires_grad],lr=.001,weight_decay=.01)
    logs=[]
    tic=time.perf_counter()
    for step,ids in enumerate(samples,1):
        a,e,y=_training_batch(fit,labels,ids)
        model.train()
        opt.zero_grad(set_to_none=True)
        pred=model(a,e).reshape(BATCH,2)
        loss=F.cross_entropy(-pred,y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R41_NONFINITE_FACTUAL_LOSS")
        loss.backward()
        grads=[v.grad for v in model.parameters() if v.requires_grad]
        if (not any(v is not None and bool(torch.any(v!=0)) for v in grads)
            or any(v is not None and not bool(torch.isfinite(v).all())
                   for v in grads)):
            raise ValueError("F8R41_INVALID_NEURAL_GRADIENT")
        nn.utils.clip_grad_norm_([v for v in model.parameters()
                                  if v.requires_grad],1.)
        opt.step()
        if step in (192,384):
            logs.append({"updates":step,"last_train_CE":float(loss.detach()),
                         "heldout":_validation(model,fit,labels)})
    if (_weight_hash(model.binder)!=frozen_sha or
        _weight_hash(model.recurrent)==before_native or len(logs)!=2):
        raise ValueError("F8R41_TRAINING_OR_BINDER_CUSTODY_FAIL")
    panels={}
    for p in PANELS:
        panels[p]={k:_score(model,p,**v) for k,v in {
            "active":{},"erase_B":{"erase_b":True},
            "swap_B":{"swap_b":True},"erase_fifth":{"erase_fifth":True},
            "zero_history":{"zero_history":True},
            "zero_backbone":{"zero_backbone":True}}.items()}
    restored=HistoryAdapter(arm,seed,binder)
    restored.load_state_dict(copy.deepcopy(model.state_dict()),strict=True)
    if (_weight_hash(restored)!=_weight_hash(model) or
        any(_score(restored,p)["choices"]!=panels[p]["active"]["choices"]
            for p in PANELS)):
        raise ValueError("F8R41_COLD_RESTORE_FAIL")
    return {"arm":arm,"seed":seed,"optimizer_updates":STEPS,
            "parameters_recurrent":sum(v.numel()
                for v in model.recurrent.parameters()),
            "binder_frozen_sha256":frozen_sha,
            "native_initial_sha256":before_native,
            "native_final_sha256":_weight_hash(model.recurrent),
            "cold_restore_action_identity":True,
            "train_wall_seconds":time.perf_counter()-tic,
            "validation_checkpoints":logs,"panels":panels}


def _bootstrap_lower(native,other):
    rng=random.Random(BOOT_SEED)
    values=[]
    for _ in range(BOOT_DRAW):
        sampled=[rng.randrange(GROUPS) for __ in range(GROUPS)]
        values.append(statistics.fmean(other[i]-native[i] for i in sampled))
    return sorted(values)[int(.025*BOOT_DRAW)]


def _judge(rows,control,freeze):
    t=freeze["preregistered_score"]
    index={(r["arm"],r["seed"]):r for r in rows}
    report=[]
    for seed in SEEDS:
        native=index["native_history",seed]
        historical=index["history_only",seed]
        diagnostics={}
        for panel in PANELS:
            n=native["panels"][panel]
            r=n["active"]
            h=historical["panels"][panel]["active"]
            gain=h["mean_regret"]-r["mean_regret"]
            gain_frozen=n["zero_history"]["mean_regret"]-r["mean_regret"]
            penaltyB=n["erase_B"]["mean_regret"]-r["mean_regret"]
            penaltyQ=n["erase_fifth"]["mean_regret"]-r["mean_regret"]
            backbone=n["zero_backbone"]["mean_regret"]-r["mean_regret"]
            lower=_bootstrap_lower(r["pergroup_regret"],h["pergroup_regret"])
            qualified=bool(
                r["mean_regret"]<=t["native_active_regret_max_each"] and
                r["history_gate_pair_flips"]>=t["native_history_gate_pair_flips_min_each"] and
                penaltyB>=t["native_B_marker_erasure_regret_penalty_min_each"] and
                backbone>=t["native_zero_backbone_regret_increase_min_each"] and
                gain>=t["native_gain_over_history_only_same_seed_min_each"] and
                gain_frozen>=t["native_gain_over_frozen_binder_only_min_each"] and
                penaltyQ>=t["native_fifth_erasure_regret_increase_min_each"] and
                lower>t["paired_bootstrap_lower_native_vs_history_only_gt"])
            diagnostics[panel]={"native_regret":r["mean_regret"],
                "history_only_regret":h["mean_regret"],
                "native_history_gate_flips":r["history_gate_pair_flips"],
                "B_erasure_penalty":penaltyB,"B_swap_penalty":
                    n["swap_B"]["mean_regret"]-r["mean_regret"],
                "fifth_erasure_penalty":penaltyQ,
                "zero_backbone_penalty":backbone,
                "zero_history_penalty":gain_frozen,
                "gain_vs_trained_history_only":gain,
                "bootstrap_lower_vs_history_only":lower,
                "frozen_pass":qualified}
        report.append({"seed":seed,"panels":diagnostics,
            "both_panel_pass":all(x["frozen_pass"] for x in diagnostics.values())})
    conventional=all(control[p]["history_gate_pair_flips"]>=
                     t["factual_conventional_positive_control_needs_gate_flips_min_each"]
                     and control[p]["mean_regret"]==0 for p in PANELS)
    count=sum(x["both_panel_pass"] for x in report)
    return {"native_both_panel_pass_seeds":count,
        "frozen_native_incremental_development_pass":
            count>=t["native_seed_both_panels_pass_min"] and conventional,
        "factual_conventional_positive_pass":conventional,
        "seed_diagnostics":report,
        "independent_scientific_pass":False}


def run():
    f=locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit,labels=fit_cohort()
    # Reuse original R39 source-qualified trained binder, one pretrain per seed.
    r33fit=r33.fit_cohort()
    winners=r36.fit_pairs(r33fit)
    pools=r37.selection_pools(r33fit,winners)
    g=torch.Generator(device="cpu")
    g.manual_seed(21249999)
    original=torch.randint(0,len(pools["informative"]),(512,16),
                           generator=g).tolist()
    saved={}
    for seed in SEEDS:
        binder,custody=r39.replay_original_seed(seed,r33fit,winners,pools,original)
        saved[seed]=binder
        if not custody["replayed_R38_decision_fingerprints_match"]:
            raise ValueError("F8R41_PRETRAIN_BINDER_DRIFT")
    generator=torch.Generator(device="cpu")
    generator.manual_seed(21549999)
    sample=torch.randint(0,TRAIN_PAIRS,(STEPS,BATCH),
                         generator=generator).tolist()
    rows=[_train(arm,seed,saved[seed],fit,labels,sample)
          for arm,seed in ARMS]
    control={p:_fitted_conventional_baseline(fit,worlds(p),p) for p in PANELS}
    return {"schema":SCHEMA,"classification":f["classification"],
        "freeze_blob":FREEZE_BLOB,"frozen_parent":f["parent"]["head"],
        "factual_fit_episodes":len(fit),"train_factual_pairs":TRAIN_PAIRS,
        "heldout_factual_pairs":VAL_PAIRS,"models":rows,
        "original_R38_pretrain_optimizer_updates":1024,
        "new_native_control_optimizer_updates":len(rows)*STEPS,
        "conventional_positive":control,"judgment":_judge(rows,control,f),
        "source_neutral_only":True,"independent_scientific_pass":False,
        "original_external_effects":0,"spent_F7_replay":False,
        "new_law_fitting":False,"deployment":"NONE"}


def retain_ci_receipt(x):
    if os.getenv("GITHUB_ACTIONS")!="true":
        return
    path=ROOT/"local_runs/p85mr_f8r41_history_conditional_dev.json"
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R41_REFUSE_RECEIPT_OVERWRITE")
    path.write_text(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+"\n")


def refuse_external_firstuse(*args,**kwargs):
    raise PermissionError("F8R41_EXTERNAL_FIRSTUSE_NOT_AUTHORIZED")
