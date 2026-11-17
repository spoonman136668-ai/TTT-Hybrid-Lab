"""R113 prospective UNIT_ONLY structured causal intervention-choice experiment.

All actions are genuinely selected before their simulator returns a factual
signed consequence. No actor sees the private compositional mechanism,
unoffered intervention outcomes, or reserved query. Synthetic-only source;
no original protected science, GPU, model promotion or FORGE admission.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r112_portable_probe_diagnostic as r112
from experiments import r113_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
SEEDS=custody.SEEDS
Q=384
STEPS=6
COUNT={"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
SALTS={"UNIT_FIT":27301101,"UNIT_VALID":27301107,
       "UNIT_UNSEEN":27301113,"UNIT_BATCH":27301119,
       "UNIT_CHOICES":27301123,"UNIT_OUTCOMES":27301129,
       "UNIT_QUERIES":27301131,"UNIT_COLD":27301137}
OPS=("AND","OR","XOR","NAND","NOR","XNOR")
UNSEEN=(("AND","XOR"),("OR","NAND"),("NAND","OR"),("NOR","AND"),
        ("XNOR","NOR"),("XOR","AND"),("NAND","NOR"),("NOR","NAND"))
ARMS=("R113_NATIVE_INTEGER","R113_MLP_TRAINED","R113_GRU_TRAINED",
      "R113_COMPOSITION_BAYES_IG","R113_LOW_COUNT_COVERAGE",
      "R113_MATCHED_RANDOM","R113_FIXED_FIRST","R113_NATIVE_FEEDBACK_BLIND")
NATIVE,MLP,GRU,BAYESIG,COUNTARM,RANDOM,FIXED,BLIND=ARMS
TRAINABLE=(NATIVE,MLP,GRU)
RECEIPT=ROOT/"local_runs/r113_unit_only_structured_active_causal_probes.json"
_ORIGINAL_EVALUATOR_ONLY=object()
EPS=.15
LOG_MATCH=math.log(1.-EPS)
LOG_MISMATCH=math.log(EPS)


def frozen():
    f=custody.audit_freeze()
    src=f["source"];train=f["training"]
    if (tuple(src["seeds"])!=SEEDS or src["panel_salts"]!=SALTS
        or tuple(src["operators"])!=OPS
        or tuple(tuple(x) for x in src["unseen_compositions"])!=UNSEEN
        or src["worlds_per_seed"]!=COUNT
        or src["chosen_action_budget_per_phase"]!=STEPS
        or src["outcome_flip_probability"]!=EPS
        or train["AdamW_updates_each"]!=512
        or train["batch_size"]!=16 or train["CPU_wall_budget_seconds_per_seed"]!=240):
        raise PermissionError("R113_FROZEN_SOURCE_PARAMETERS_DRIFT")
    return f


def _deadline(t):
    if t is not None and time.perf_counter()>=t:
        raise TimeoutError("R113_FROZEN_CPU_BUDGET_EXCEEDED")


def _op(name,a,b):
    if name=="AND": return a&b
    if name=="OR": return a|b
    if name=="XOR": return a^b
    if name=="NAND": return 1-(a&b)
    if name=="NOR": return 1-(a|b)
    if name=="XNOR": return 1-(a^b)
    raise ValueError("R113_UNKNOWN_STRUCTURAL_OPERATOR")


def truth_bit(pair,pattern):
    if type(pattern) is not int or not 0<=pattern<8:
        raise PermissionError("R113_UNREGISTERED_INTERVENTION")
    x=(pattern>>2)&1;y=(pattern>>1)&1;z=pattern&1
    return _op(pair[1],_op(pair[0],x,y),z)


def mask_of(pair):
    return sum(truth_bit(pair,p)<<p for p in range(8))


ALL=tuple((a,b) for a in OPS for b in OPS)
RESERVED_MASKS={mask_of(p) for p in UNSEEN}
FIT=tuple(p for p in ALL if mask_of(p) not in RESERVED_MASKS)
TRUTHS=np.asarray([[truth_bit(pair,p) for p in range(8)]
                   for pair in ALL],dtype=np.int8)


def admit_templates():
    f=frozen()
    if (len(ALL)!=36 or len(FIT)!=27 or len(UNSEEN)!=8 or
        len({mask_of(x) for x in FIT})!=22 or
        len(RESERVED_MASKS)!=8 or
        set(FIT)&set(UNSEEN) or
        any(mask_of(x) in RESERVED_MASKS for x in FIT) or
        [f"{mask_of(x):02x}" for x in UNSEEN]!=
            f["source"]["unseen_composition_truth_masks_hex"] or
        any(not any(p[0]==o for p in FIT) or not any(p[1]==o for p in FIT)
            for o in OPS)):
        raise PermissionError("R113_NOVEL_COMPOSITION_SOURCE_OR_COVERAGE_DRIFT")


def assign(panel,index):
    if panel not in COUNT or type(index) is not int or not 0<=index<COUNT[panel]:
        raise PermissionError("R113_ORIGINAL_OR_UNKNOWN_SOURCE_DENIED")
    catalog=UNSEEN if panel=="UNIT_UNSEEN" else FIT
    n=len(catalog)
    changed=bool((index//n)%2)
    old=catalog[index%n]
    new=catalog[(index+3)%n] if changed else old
    return old,new,changed


class PhysicalStructuredWorld:
    """Private two-stage SCM; one factual chosen outcome per permitted do()."""
    def __init__(self,panel,seed,index,phase,*,_token=None):
        if (panel not in COUNT or seed not in SEEDS or phase not in (0,1)
            or type(index) is not int or not 0<=index<COUNT.get(panel,0)):
            raise PermissionError("R113_PRIVATE_SOURCE_ADMISSION_DENIED")
        if panel=="UNIT_UNSEEN" and _token is not _ORIGINAL_EVALUATOR_ONLY:
            raise PermissionError("R113_SEALED_UNIT_UNSEEN_BEFORE_COMPLETED_TRAINING")
        if panel!="UNIT_UNSEEN":
            custody.guard_panel(panel)
        old,new,_=assign(panel,index)
        self._private_pair=old if phase==0 else new
        self._panel=panel;self._seed=seed;self._index=index;self._phase=phase
        self._step=0
        self.history=[]
        self.records=[]

    def offers(self):
        if self._step>=STEPS:
            raise PermissionError("R113_PHYSICAL_PROBE_BUDGET_EXHAUSTED")
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[self._panel],SALTS["UNIT_CHOICES"],self._seed,
            self._index,self._phase,self._step]))
        return tuple(sorted(int(v) for v in rng.choice(8,2,replace=False)))

    def intervene(self,selected):
        offers=self.offers()
        if type(selected) is not int or selected not in offers:
            raise PermissionError("R113_UNOFFERED_ACTION_OR_UNCHOSEN_RETURN_DENIED")
        # Private mediator Z computed *only* for selected public action.
        sign=1 if truth_bit(self._private_pair,selected) else -1
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[self._panel],SALTS["UNIT_OUTCOMES"],self._seed,
            self._index,self._phase,self._step,selected]))
        if bool(rng.random()<EPS):
            sign=-sign
        self.records.append({"chosen_do":selected,"observed_signed_effect":sign})
        self.history.append((selected,sign))
        self._step+=1
        return sign

    def sealed_query(self):
        if self._step!=STEPS:
            raise PermissionError("R113_UNSEEN_QUERY_BEFORE_ALL_ACTUAL_ACTIONS")
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS[self._panel],SALTS["UNIT_QUERIES"],
            self._seed,self._index,self._phase]))
        p=int(rng.integers(0,8))
        return p,1 if truth_bit(self._private_pair,p) else -1


def _posterior(history):
    if len(history)>STEPS:
        raise ValueError("R113_TOO_MANY_OBSERVED_ACTIONS")
    logs=np.zeros(len(ALL),np.float64)
    for p,r in history:
        if type(p) is not int or not 0<=p<8 or r not in (-1,1):
            raise ValueError("R113_NONPUBLIC_FACTUAL_HISTORY")
        logs+=np.where(TRUTHS[:,p]==int(r>0),LOG_MATCH,LOG_MISMATCH)
    w=np.exp(logs-logs.max())
    return w/w.sum()


def _entropy(weights):
    w=np.asarray(weights,np.float64)
    return float(-(w*np.log2(np.maximum(w,1e-30))).sum())


def information_gain(pattern,history):
    w=_posterior(history)
    p_plus=EPS+(1-2*EPS)*float(w@TRUTHS[:,pattern].astype(np.float64))
    plus_like=np.where(TRUTHS[:,pattern]>0,1-EPS,EPS)
    minus_like=1.-plus_like
    wplus=w*plus_like; wplus/=wplus.sum()
    wminus=w*minus_like; wminus/=wminus.sum()
    return max(0.,_entropy(w)-p_plus*_entropy(wplus)-(1-p_plus)*_entropy(wminus))


def fixed_readout(pattern,history):
    w=_posterior(history)
    probability=float(w@TRUTHS[:,pattern].astype(np.float64))
    return 1 if probability>=.5 else -1


def integer_native_features(history,offers,phase,*,blind=False):
    if (phase not in (0,1) or len(offers)!=2 or offers[0]==offers[1]
        or any(type(p) is not int or not 0<=p<8 for p in offers)
        or len(history)>STEPS):
        raise ValueError("R113_INVALID_PUBLIC_CANDIDATE_OBSERVATIONS")
    counts=[0]*8;signed=[0]*8
    for p,r in history:
        if type(p) is not int or not 0<=p<8 or r not in (-1,1):
            raise ValueError("R113_NONPUBLIC_EFFECT_IN_NATIVE_STATE")
        counts[p]+=1;signed[p]+=r
    if blind:signed=[0]*8
    u=Q//STEPS
    rows=[]
    for p in offers:
        bits=[(p>>2)&1,(p>>1)&1,p&1]
        row=([u*c for c in counts]+[u*x for x in signed]
             +[Q*b for b in bits]+[Q*phase,u*len(history),
                                     u*counts[p],u*signed[p],u*abs(signed[p])])
        if len(row)!=24:
            raise RuntimeError("R113_NATIVE_ADDRESS_FEATURE_DRIFT")
        rows.append(row)
    return np.asarray(rows,np.int64)


def features(history,offers,phase,*,blind=False):
    integer=integer_native_features(history,offers,phase,blind=blind)
    n=(integer.astype(np.float32)/Q)
    m=n.reshape(48)
    g=np.zeros((STEPS,10),np.float32)
    for j,(p,s) in enumerate(history):
        g[j,p]=1.;g[j,8]=s if not blind else 0.;g[j,9]=1.
    return n,m,g


class NativeProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(24,32),nn.GELU(),
                                  nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))
    def forward(self,n,m,g):
        if n.ndim!=3 or n.shape[1:]!=(2,24):
            raise RuntimeError("R113_NATIVE_FACTUAL_MEMORY_SHAPE")
        return self.scorer(n).squeeze(-1)


class MLPProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(48,64),nn.GELU(),
                                 nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,n,m,g):
        if m.ndim!=2 or m.shape[1]!=48:
            raise RuntimeError("R113_PUBLIC_MLP_SHAPE")
        return self.model(m)


class GRUProbe(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(10,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(40,48),nn.GELU(),nn.Linear(48,2))
    def forward(self,n,m,g):
        if g.ndim!=3 or g.shape[1:]!=(6,10) or n.shape!=(len(g),2,24):
            raise RuntimeError("R113_PUBLIC_GRU_SHAPE")
        full,_=self.rnn(g)
        ix=(g[:,:,9]>0).sum(dim=1).clamp(min=1).long()-1
        h=full[torch.arange(len(g)),ix]
        common=torch.cat((n[:,:,16:19].reshape(len(g),6),
                          n[:,0,19:21]),dim=-1)
        return self.head(torch.cat((h,common),dim=-1))


def new_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R113_UNREGISTERED_LEARNING_SEED")
    torch.manual_seed(seed+113);native=NativeProbe()
    torch.manual_seed(seed+213);mlp=MLPProbe()
    torch.manual_seed(seed+313);gru=GRUProbe()
    return {NATIVE:native,MLP:mlp,GRU:gru}


def fit_examples(seed,deadline=None):
    admit_templates()
    if seed not in SEEDS:
        raise PermissionError("R113_UNREGISTERED_FIT_SOURCE")
    xs=[];ms=[];gs=[];ys=[]
    for index in range(COUNT["UNIT_FIT"]):
        _deadline(deadline)
        for phase in (0,1):
            world=PhysicalStructuredWorld("UNIT_FIT",seed,index,phase)
            for step in range(STEPS):
                offers=world.offers()
                n,m,g=features(world.history,offers,phase)
                targets=[information_gain(p,world.history) for p in offers]
                xs.append(n);ms.append(m);gs.append(g);ys.append(targets)
                rng=np.random.default_rng(np.random.SeedSequence([
                    SALTS["UNIT_FIT"],SALTS["UNIT_BATCH"],seed,index,phase,step]))
                chosen=offers[int(rng.integers(0,2))]
                world.intervene(int(chosen))
    n=np.asarray(xs,np.float32);m=np.asarray(ms,np.float32)
    g=np.asarray(gs,np.float32);y=np.asarray(ys,np.float32)
    if (n.shape!=(2112,2,24) or m.shape!=(2112,48)
        or g.shape!=(2112,6,10) or y.shape!=(2112,2)
        or not bool(np.isfinite(y).all()) or not bool(np.all(y>=0))):
        raise RuntimeError("R113_COHERENT_PUBLIC_FIT_TRAJECTORIES_DRIFT")
    return tuple(torch.from_numpy(z) for z in (n,m,g,y))


def train_arm(model,name,fit,minibatches,deadline):
    f=frozen()["training"]
    if name not in TRAINABLE or minibatches.shape!=(512,16):
        raise PermissionError("R113_FROZEN_OPTIMIZER_BUDGET_DRIFT")
    optimizer=torch.optim.AdamW(model.parameters(),lr=f["learning_rate"],
                                weight_decay=f["weight_decay"])
    model.train();t0=time.perf_counter()
    n,m,g,targets=fit
    for ids in minibatches:
        _deadline(deadline)
        estimate=model(n[ids],m[ids],g[ids])
        if estimate.shape!=(16,2):
            raise RuntimeError("R113_NONMATCHED_PUBLIC_CANDIDATE_SHAPE")
        loss=F.mse_loss(estimate,targets[ids])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R113_NONFINITE_INFORMATION_TRAINING")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),f["gradient_clip"])
        optimizer.step()
    _deadline(deadline)
    return {"updates":512,"parameters":sum(p.numel() for p in model.parameters()),
            "fit_seconds":time.perf_counter()-t0}


def quantize_native(model):
    custody.audit_freeze()
    h=hashlib.sha256(b"R113_24_32_16_1_INTEGER_ACTIVE_CAUSAL_Q384")
    layers=[]
    for layer in (model.scorer[0],model.scorer[2],model.scorer[4]):
        values=[]
        for name in ("weight","bias"):
            p=getattr(layer,name).detach().cpu().numpy()
            if not np.isfinite(p).all():
                raise RuntimeError("R113_QUANTIZE_NONFINITE_WEIGHTS")
            q=np.rint(p.astype(np.float64)*Q).astype("<i8")
            if np.any(np.abs(q)>=2**20):
                raise OverflowError("R113_FIXED_POINT_OUT_OF_RANGE")
            h.update(name.encode());h.update(struct.pack("<I",q.ndim))
            for z in q.shape:h.update(struct.pack("<I",z))
            h.update(q.tobytes(order="C"))
            values.append(q.tolist())
        layers.append({"weight":values[0],"bias":values[1]})
    return layers,h.hexdigest()


def integer_scores(layers,history,offers,phase,*,blind=False):
    z=integer_native_features(history,offers,phase,blind=blind)
    dims=(24,32,16,1)
    if len(layers)!=3:
        raise PermissionError("R113_UNKNOWN_FIXED_POINT_LAYERS")
    for i,layer in enumerate(layers):
        w=np.asarray(layer["weight"],np.int64);b=np.asarray(layer["bias"],np.int64)
        if w.shape!=(dims[i+1],dims[i]) or b.shape!=(dims[i+1],):
            raise RuntimeError("R113_INTEGER_LAYER_INTERFACE_DRIFT")
        z=r112._rdiv(z@w.T,Q)+b[None,:]
        if bool(np.any(np.abs(z)>2**25)):
            raise OverflowError("R113_FIXED_POINT_INFERENCE_OVERFLOW")
        if i<2:z=r112._hard_swish(z)
    if z.shape!=(2,1):
        raise RuntimeError("R113_INTEGER_SCORE_SHAPE_DRIFT")
    return int(z[0,0]),int(z[1,0])


def choose(arm,models,layers,offers,history,phase,seed,index,step):
    if arm not in ARMS:
        raise PermissionError("R113_UNKNOWN_INTERVENTION_SELECTOR")
    if arm==FIXED:return int(offers[0])
    if arm==RANDOM:
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS["UNIT_CHOICES"],seed,index,phase,step,113]))
        return int(offers[int(rng.integers(0,2))])
    if arm==COUNTARM:
        n=[sum(p==x for p,_ in history) for x in offers]
        return int(offers[int(n[1]<n[0])])
    if arm==BAYESIG:
        g=[information_gain(p,history) for p in offers]
        return int(offers[int(g[1]>g[0]+1e-10)])
    if arm in (NATIVE,BLIND):
        s0,s1=integer_scores(layers,history,offers,phase,blind=arm==BLIND)
        return int(offers[int(s1>s0)])
    n,m,g=features(history,offers,phase)
    models[arm].eval()
    with torch.no_grad():
        score=models[arm](torch.from_numpy(n[None]),torch.from_numpy(m[None]),
                          torch.from_numpy(g[None]))[0]
    if score.shape!=(2,) or not bool(torch.isfinite(score).all()):
        raise RuntimeError("R113_NONFINITE_TRAINED_PROBE_DECISION")
    return int(offers[int(score[1].item()>score[0].item()+1e-10)])


def evaluate_seed(models,layers,seed,deadline,_token):
    if _token is not _ORIGINAL_EVALUATOR_ONLY or seed not in SEEDS:
        raise PermissionError("R113_UNSEEN_FIRST_USE_BEFORE_ALL_CONTROLS_TRAINED")
    metrics={};trace={}
    for arm in ARMS:
        correct={"old_before":0,"old_after":0,"new_after":0,
                 "changed_law_new_after":0,"unchanged_law_new_after":0}
        actions=[];positive=0;unique=0
        for index in range(COUNT["UNIT_UNSEEN"]):
            _deadline(deadline)
            _,_,changed=assign("UNIT_UNSEEN",index)
            old=PhysicalStructuredWorld("UNIT_UNSEEN",seed,index,0,_token=_token)
            new=PhysicalStructuredWorld("UNIT_UNSEEN",seed,index,1,_token=_token)
            for phase,world in ((0,old),(1,new)):
                for step in range(STEPS):
                    offers=world.offers()
                    chosen=choose(arm,models,layers,offers,world.history,
                                  phase,seed,index,step)
                    if chosen not in offers:
                        raise RuntimeError("R113_UNCHOSEN_OR_UNOFFERED_INTERVENTION")
                    got=world.intervene(chosen)
                    actions.append(chosen)
                    positive+=int(got==1)
                unique+=len(set(x for x,_ in world.history))
                query,truth=world.sealed_query()
                prediction=fixed_readout(query,world.history)
                if phase==0:
                    correct["old_before"]+=int(prediction==truth)
                else:
                    correct["new_after"]+=int(prediction==truth)
                    key=("changed_law_new_after" if changed else
                         "unchanged_law_new_after")
                    correct[key]+=int(prediction==truth)
            p,t=old.sealed_query()
            correct["old_after"]+=int(fixed_readout(p,old.history)==t)
        if len(actions)!=COUNT["UNIT_UNSEEN"]*2*STEPS:
            raise RuntimeError("R113_REAL_ACTION_BUDGET_NOT_FULLY_OBSERVED")
        trace[arm]=actions
        metrics[arm]={
            **{k+"_correct":int(v) for k,v in correct.items()},
            **{k+"_accuracy":v/(64 if k.startswith(("changed_","unchanged_"))
                                 else COUNT["UNIT_UNSEEN"])
               for k,v in correct.items()},
            "physically_chosen_probe_count":len(actions),
            "positive_observed_consequences":positive,
            "mean_distinct_selected_probes_per_phase":
                unique/(2*COUNT["UNIT_UNSEEN"]),
            "actions_sha256":hashlib.sha256(bytes(actions)).hexdigest()}
    divergence=sum(int(x!=y) for x,y in zip(trace[NATIVE],trace[RANDOM]))
    return metrics,divergence


def isolated_integer_cold_reload(layers):
    samples=[([], (0,3),0), ([(0,1),(3,-1)],(2,7),1),
             ([(1,1),(2,-1),(3,1),(1,-1)],(3,5),0),
             ([(2,-1),(3,1),(1,-1),(6,1),(0,1),(5,-1)],(1,6),1)]
    original=[integer_scores(layers,h,o,phase)
              for h,o,phase in samples]
    payload=json.dumps({"layers":layers,"samples":samples},sort_keys=True).encode()
    code=("import json,sys\n"
          "from experiments.r113_structured_active_probe_diagnostic import integer_scores\n"
          "x=json.load(sys.stdin)\n"
          "print(json.dumps([integer_scores(x['layers'],[tuple(y) for y in h],"
          "tuple(o),p) for h,o,p in x['samples']]))\n")
    p=subprocess.run([sys.executable,"-c",code],cwd=ROOT,input=payload,
                     capture_output=True,check=True,timeout=30)
    restored=json.loads(p.stdout)
    return int(sum(tuple(x)!=tuple(y) for x,y in zip(original,restored)))


def frozen_checks(results,divergence,cold):
    required=frozen()["evaluation"]["per_seed_criteria_all_three_required"]
    acc=lambda name,key="new_after_accuracy":results[name][key]
    values={
      "native_new_accuracy_min":acc(NATIVE),
      "native_minus_random_new_accuracy_min":acc(NATIVE)-acc(RANDOM),
      "native_minus_lowcount_new_accuracy_min":acc(NATIVE)-acc(COUNTARM),
      "native_minus_fixed_first_new_accuracy_min":acc(NATIVE)-acc(FIXED),
      "native_minus_best_conventional_new_accuracy_min":
          acc(NATIVE)-max(acc(MLP),acc(GRU)),
      "native_minus_exact_IG_new_accuracy_min":acc(NATIVE)-acc(BAYESIG),
      "native_minus_feedback_blind_new_accuracy_min":acc(NATIVE)-acc(BLIND),
      "native_changed_law_minus_random_min":
          acc(NATIVE,"changed_law_new_after_accuracy")-
          acc(RANDOM,"changed_law_new_after_accuracy"),
      "native_unchanged_law_minus_random_min":
          acc(NATIVE,"unchanged_law_new_after_accuracy")-
          acc(RANDOM,"unchanged_law_new_after_accuracy"),
      "native_old_before_accuracy_min":acc(NATIVE,"old_before_accuracy"),
      "old_before_minus_old_after_accuracy_max":
          acc(NATIVE,"old_before_accuracy")-acc(NATIVE,"old_after_accuracy"),
      "exact_IG_minus_random_new_accuracy_min":acc(BAYESIG)-acc(RANDOM),
      "native_random_action_divergence_min":divergence,
      "native_quantized_cold_reload_mismatches_max":cold,
      "all_arms_exact_12_chosen_probe_budget":
          all(results[a]["physically_chosen_probe_count"]==128*12 for a in ARMS)}
    if set(values)!=set(required):
        raise RuntimeError("R113_PRECOMMITTED_THRESHOLD_MAPPING_DRIFT")
    checks={k:(v<=required[k] if k.endswith("_max") else v>=required[k])
            for k,v in values.items()}
    return {"values":values,"checks":checks,
            "all_unit_only_checks_pass":all(checks.values()),
            "scientific_pass":False}


def run_seed(seed):
    admit_templates()
    if seed not in SEEDS:
        raise PermissionError("R113_UNREGISTERED_EVALUATION_SEED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    deadline=start+frozen()["training"]["CPU_wall_budget_seconds_per_seed"]
    fit=fit_examples(seed,deadline)
    rng=np.random.default_rng(np.random.SeedSequence([SALTS["UNIT_BATCH"],seed,113]))
    minibatches=rng.integers(0,len(fit[-1]),size=(512,16))
    models=new_models(seed)
    trained={arm:train_arm(models[arm],arm,fit,minibatches,deadline)
             for arm in TRAINABLE}
    if set(trained)!=set(TRAINABLE) or any(x["updates"]!=512 for x in trained.values()):
        raise PermissionError("R113_HELDOUT_PRIOR_TO_ALL_SCRATCH_CONTROLS")
    layers,weight_sha=quantize_native(models[NATIVE])
    cold=isolated_integer_cold_reload(layers)
    # FIRST and ONLY UNIT_ONLY heldout generation on this prospectively frozen SHA.
    evaluations,divergence=evaluate_seed(
        models,layers,seed,deadline,_ORIGINAL_EVALUATOR_ONLY)
    checks=frozen_checks(evaluations,divergence,cold)
    _deadline(deadline)
    return {"seed":seed,"status":(
              frozen()["evaluation"]["pass_status"] if checks["all_unit_only_checks_pass"]
              else frozen()["evaluation"]["fail_status"]),
            "trained":trained,"all_three_models_512_steps_completed":True,
            "quantized_native_weight_sha256":weight_sha,
            "native_vs_random_action_divergence":divergence,
            "evaluation":evaluations,"frozen_checks":checks,
            "cold_integer_score_mismatches":cold,
            "seconds":time.perf_counter()-start,
            "original_scientific_firstuse":False,"scientific_pass":False}


def run_all():
    frozen()
    if (ROOT/"experiments/r113_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R113_PROTECTED_FIRSTUSE_NOT_PERMITTED")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R113_EXACT_SCORED_SHA_DRIFT")
    by_seed={str(seed):run_seed(seed) for seed in SEEDS}
    passed=all(x["frozen_checks"]["all_unit_only_checks_pass"]
               for x in by_seed.values())
    return {"schema":"ttt-hybrid.r113.original-structured-causal-active-probe.unit-only.v1",
            "exact_source_sha":git,"freeze_blob":custody.FREEZE_BLOB,
            "status":(frozen()["evaluation"]["pass_status"] if passed
                      else frozen()["evaluation"]["fail_status"]),
            "by_seed":by_seed,"all_three_seed_criteria_pass":passed,
            "spent_prior_R111_cohort_reused":False,
            "original_scientific_firstuse":False,"GPU_firstuse":False,
            "scientific_pass":False,"promotion":False,"deployment":"HOLD"}


def main():
    result=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R113_IMMUTABLE_ORIGINAL_UNIT_ONLY_RECEIPT_OVERWRITE_DENIED")
    RECEIPT.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R113_UNIT_ONLY "+json.dumps({
      "sha":result["exact_source_sha"],"status":result["status"],
      "by_seed":{s:{"new_correct":{arm:d["new_after_correct"]
                                   for arm,d in v["evaluation"].items()},
                   "checks":v["frozen_checks"]["checks"],
                   "weight_sha":v["quantized_native_weight_sha256"],
                   "native_divergence":v["native_vs_random_action_divergence"],
                   "cold":v["cold_integer_score_mismatches"],
                   "seconds":round(v["seconds"],3)}
                 for s,v in result["by_seed"].items()},
      "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
