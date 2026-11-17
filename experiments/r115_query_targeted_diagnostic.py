"""R115: UNIT_ONLY query-targeted active causal learning with full controls.

Prospective frozen source, fresh four-factor compositions, training 3
matched-public-FIT policies, exact integer Q384 native intervention choices,
actual chosen-only intervention consequences and fixed exact all-216-rule
public Bayesian query decoder. Synthetic DEV ONLY, no protected science.
"""
from __future__ import annotations
import hashlib
import json
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
from experiments import r115_source_custody as custody
from experiments import r115_structured_query_source as source

ROOT=Path(__file__).resolve().parents[1]
SEEDS=custody.SEEDS
Q=384
STEPS=5
COUNT=source.PANEL
SALTS=source.SALTS
ARMS=("R115_NATIVE_QUERY_INTEGER","R115_MLP_TRAINED",
      "R115_GRU_TRAINED","R115_FULL_BAYES_TARGET_VOI",
      "R115_LOW_COUNT","R115_MATCHED_RANDOM","R115_FIRST",
      "R115_NATIVE_QUERY_BLIND","R115_NATIVE_FEEDBACK_BLIND")
NATIVE,MLP,GRU,BAYES,LOW,RANDOM,FIXED,QBLIND,FBLIND=ARMS
TRAINABLE=(NATIVE,MLP,GRU)
RECEIPT=ROOT/"local_runs/r115_query_targeted_four_factor_unit_only.json"
QFEATURES=45


def frozen():
    f=source.audit_novelty()
    t=f["training"]
    if (t["FIT_training_examples_per_seed"]!=2560
        or t["updates_per_arm"]!=512 or t["batch_size"]!=16
        or t["per_seed_CPU_wall_budget_seconds"]!=360
        or t["teacher_target_scale"]!=20
        or t["three_fresh_trainable_policies"]!=["NATIVE_PUBLIC_QUERY_INTEGER","MLP_PUBLIC_QUERY_FULL","GRU_PUBLIC_QUERY_SEQUENCE"]
        or f["source"]["candidate_actions_per_step"]!=2
        or f["source"]["query_public_before_support"] is not True):
        raise PermissionError("R115_FROZEN_TRAINING_AND_SOURCE_DRIFT")
    return f


def _deadline(stop):
    if stop is not None and time.perf_counter()>=stop:
        raise TimeoutError("R115_FROZEN_CPU_WALL_BUDGET")


def _norm(n):
    if type(n) is not int or not 0<=n<=5:
        raise ValueError("R115_NONPUBLIC_COUNT_RANGE")
    return (n*Q+2)//5


def native_integer_features(history,offers,query,phase,*,query_blind=False,feedback_blind=False):
    if (type(query) is not int or not 0<=query<16
        or phase not in (0,1) or len(offers)!=2 or offers[0]==offers[1]
        or any(type(p) is not int or not 0<=p<16 for p in offers)
        or len(history)>STEPS):
        raise ValueError("R115_PUBLIC_FEATURE_OR_QUERY_SHAPE")
    count=[0]*16;signed=[0]*16
    for p,r in history:
        if type(p) is not int or not 0<=p<16 or r not in (-1,1):
            raise ValueError("R115_UNCHOSE_NONFACTUAL_PUBLIC_HISTORY")
        count[p]+=1;signed[p]+=r
    if feedback_blind:
        signed=[0]*16
    query_bits=[0]*4 if query_blind else [(query>>i)&1 for i in (3,2,1,0)]
    arr=[]
    for p in offers:
        offered_bits=[(p>>i)&1 for i in (3,2,1,0)]
        row=([_norm(x) for x in count]
             +[int((v*Q+(2 if v>=0 else -2))//5)
               if v>=0 else -_norm(-v) for v in signed]
             +[Q*x for x in query_bits]+[Q*x for x in offered_bits]
             +[Q*phase,_norm(len(history)),
               _norm(count[p]),int(np.sign(signed[p]))*_norm(abs(signed[p])),
               _norm(abs(signed[p]))])
        if len(row)!=QFEATURES:
            raise RuntimeError("R115_PUBLIC_45_FEATURE_INTERFACE_DRIFT")
        arr.append(row)
    return np.asarray(arr,np.int64)


def public_inputs(history,offers,query,phase,*,query_blind=False,feedback_blind=False):
    integers=native_integer_features(history,offers,query,phase,
                                    query_blind=query_blind,feedback_blind=feedback_blind)
    n=integers.astype(np.float32)/Q
    m=n.reshape(90)
    g=np.zeros((STEPS,18),np.float32)
    for j,(p,s) in enumerate(history):
        g[j,p]=1.
        g[j,16]=s if not feedback_blind else 0.
        g[j,17]=1.
    return n,m,g


class NativePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(45,48),nn.GELU(),
                                  nn.Linear(48,24),nn.GELU(),nn.Linear(24,1))
    def forward(self,n,m,g):
        if n.ndim!=3 or n.shape[1:]!=(2,45):
            raise RuntimeError("R115_TOKENLESS_NATIVE_SHAPE_DRIFT")
        return self.scorer(n).squeeze(-1)


class MLPPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(90,64),nn.GELU(),
                                 nn.Linear(64,32),nn.GELU(),nn.Linear(32,2))
    def forward(self,n,m,g):
        if m.ndim!=2 or m.shape[1]!=90:
            raise RuntimeError("R115_STRONG_MLP_PUBLIC_INTERFACE")
        return self.model(m)


class GRUPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.GRU(18,32,batch_first=True)
        self.head=nn.Sequential(nn.Linear(46,40),nn.GELU(),nn.Linear(40,2))
    def forward(self,n,m,g):
        if g.ndim!=3 or g.shape[1:]!=(STEPS,18) or n.shape!=(len(g),2,45):
            raise RuntimeError("R115_STRONG_GRU_PUBLIC_INTERFACE")
        out,_=self.rnn(g)
        latest=(g[:,:,17]>0).sum(dim=1).clamp(min=1).long()-1
        h=out[torch.arange(len(g)),latest]
        # 4 query bits, 8 offered action bits, 1 phase, 1 history length.
        context=torch.cat((n[:,0,32:36],n[:,:,36:40].reshape(len(g),8),
                           n[:,0,40:42]),dim=-1)
        return self.head(torch.cat((h,context),dim=-1))


def new_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R115_UNREGISTERED_TRAINABLE_SEED")
    torch.manual_seed(seed+115);native=NativePolicy()
    torch.manual_seed(seed+215);mlp=MLPPolicy()
    torch.manual_seed(seed+315);gru=GRUPolicy()
    return {NATIVE:native,MLP:mlp,GRU:gru}


def fit_examples(seed,deadline=None):
    frozen()
    if seed not in SEEDS:
        raise PermissionError("R115_FORBIDDEN_FIT_SEED")
    nrows=[];mrows=[];grows=[];targets=[]
    for index in range(COUNT["UNIT_FIT"]):
        _deadline(deadline)
        for phase in (0,1):
            w=source.PhysicalCausalWorld("UNIT_FIT",seed,index,phase)
            q=w.public_query
            for step in range(STEPS):
                offers=w.offers()
                n,m,g=public_inputs(w.history,offers,q,phase)
                t=[source.public_query_value_for_fit(p,q,w.history) for p in offers]
                nrows.append(n);mrows.append(m);grows.append(g);targets.append(t)
                rng=np.random.default_rng(np.random.SeedSequence([
                    SALTS["UNIT_FIT"],SALTS["UNIT_BATCH"],
                    seed,index,phase,step]))
                chosen=offers[int(rng.integers(0,2))]
                w.intervene(int(chosen))
    examples=tuple(torch.from_numpy(np.asarray(z,np.float32))
                   for z in (nrows,mrows,grows,targets))
    if (tuple(z.shape for z in examples)!=
        ((2560,2,45),(2560,90),(2560,5,18),(2560,2))
        or not bool(torch.isfinite(examples[-1]).all())
        or float(examples[-1].min())<0):
        raise RuntimeError("R115_PHYSICAL_FIT_PUBLIC_TARGETS_INVALID")
    return examples


def train(model,arm,fit,indices,deadline):
    f=frozen()["training"]
    if arm not in TRAINABLE or indices.shape!=(512,16):
        raise PermissionError("R115_UNMATCHED_UPDATE_BUDGET")
    opt=torch.optim.AdamW(model.parameters(),lr=f["learning_rate"],
                          weight_decay=f["weight_decay"])
    n,m,g,target=fit
    t0=time.perf_counter();model.train()
    for ix in indices:
        _deadline(deadline)
        output=model(n[ix],m[ix],g[ix])
        if output.shape!=(16,2):
            raise RuntimeError("R115_UNMATCHED_CANDIDATE_OUTPUT")
        loss=F.mse_loss(output,target[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R115_NONFINITE_FIT_TRAINING")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),f["gradient_clip"])
        opt.step()
    return {"updates":len(indices),
            "trainable_parameters":sum(p.numel() for p in model.parameters()),
            "train_seconds":time.perf_counter()-t0}


def quantize_native(model):
    layers=[];h=hashlib.sha256(b"R115_NATIVE_PUBLIC_QUERY_45_48_24_1_Q384")
    for layer in (model.scorer[0],model.scorer[2],model.scorer[4]):
        tensors=[]
        for key in ("weight","bias"):
            v=getattr(layer,key).detach().cpu().numpy()
            if not np.isfinite(v).all():
                raise RuntimeError("R115_NONFINITE_TRAINED_NATIVE_WEIGHT")
            x=np.rint(v.astype(np.float64)*Q).astype("<i8")
            if np.any(np.abs(x)>2**20):
                raise OverflowError("R115_QUANTIZED_NATIVE_PARAMETER_OVERFLOW")
            h.update(key.encode());h.update(struct.pack("<I",x.ndim))
            for d in x.shape:h.update(struct.pack("<I",d))
            h.update(x.tobytes(order="C"))
            tensors.append(x.tolist())
        layers.append({"weight":tensors[0],"bias":tensors[1]})
    return layers,h.hexdigest()


def native_scores(layers,history,offers,query,phase,*,query_blind=False,feedback_blind=False):
    z=native_integer_features(history,offers,query,phase,
                              query_blind=query_blind,
                              feedback_blind=feedback_blind)
    dims=(45,48,24,1)
    if len(layers)!=3:
        raise RuntimeError("R115_INTEGER_NATIVE_UNKNOWN_LAYERS")
    for i,layer in enumerate(layers):
        weight=np.asarray(layer["weight"],np.int64)
        bias=np.asarray(layer["bias"],np.int64)
        if weight.shape!=(dims[i+1],dims[i]) or bias.shape!=(dims[i+1],):
            raise RuntimeError("R115_INTEGER_NATIVE_MATRIX_SHAPE")
        z=r112._rdiv(z@weight.T,Q)+bias[None,:]
        if bool(np.any(np.abs(z)>2**25)):
            raise OverflowError("R115_NATIVE_INTEGER_OVERFLOW")
        if i<2:z=r112._hard_swish(z)
    return int(z[0,0]),int(z[1,0])


def choose(arm,models,layers,offers,history,query,phase,seed,index,step):
    if arm not in ARMS:
        raise PermissionError("R115_UNAUTHORIZED_INTERVENTION_POLICY")
    if arm==FIXED:
        return int(offers[0])
    if arm==RANDOM:
        rng=np.random.default_rng(np.random.SeedSequence([
            SALTS["UNIT_CHOICES"],seed,index,phase,step,115]))
        return int(offers[int(rng.integers(0,2))])
    if arm==LOW:
        counts=[sum(p==x for p,_ in history) for x in offers]
        return int(offers[int(counts[1]<counts[0])])
    if arm==BAYES:
        scores=[source.target_query_information_value(p,query,history,full=True)
                for p in offers]
        # Denominators are identical on the same public history.
        return int(offers[int(scores[1][0]>scores[0][0])])
    if arm in (NATIVE,QBLIND,FBLIND):
        a,b=native_scores(layers,history,offers,query,phase,
                          query_blind=arm==QBLIND,feedback_blind=arm==FBLIND)
        return int(offers[int(b>a)])
    n,m,g=public_inputs(history,offers,query,phase)
    models[arm].eval()
    with torch.no_grad():
        z=models[arm](torch.from_numpy(n[None]),torch.from_numpy(m[None]),
                      torch.from_numpy(g[None]))[0]
    if z.shape!=(2,) or not bool(torch.isfinite(z).all()):
        raise RuntimeError("R115_NONFINITE_CONVENTIONAL_PREDICTION")
    return int(offers[int(z[1].item()>z[0].item()+1e-10)])


def integer_cold_restore(layers):
    samples=[([],(0,7),8,0), ([(0,1),(7,-1)],(3,9),5,1),
             ([(1,-1),(2,1),(15,1),(7,-1)],(0,13),11,1),
             ([(0,1),(3,-1),(5,1),(7,-1),(9,1)],(2,14),15,0)]
    before=[native_scores(layers,h,o,q,p) for h,o,q,p in samples]
    code=("import json,sys\n"
          "from experiments.r115_query_targeted_diagnostic import native_scores\n"
          "x=json.load(sys.stdin)\n"
          "print(json.dumps([native_scores(x['layers'],[tuple(z) for z in h],"
          "tuple(o),q,p) for h,o,q,p in x['samples']]))\n")
    p=subprocess.run([sys.executable,"-c",code],
                     input=json.dumps({"layers":layers,"samples":samples}).encode(),
                     capture_output=True,cwd=ROOT,check=True,timeout=30)
    after=json.loads(p.stdout)
    return sum(int(tuple(x)!=tuple(y)) for x,y in zip(before,after))


def _hash_octets(seq):
    return hashlib.sha256(bytes(seq)).hexdigest()


def evaluate(models,layers,seed,deadline,token):
    if token is not source._SCORER_ONLY or seed not in SEEDS:
        raise PermissionError("R115_HELDOUT_BEFORE_ALL_SCRATCH_ARMS_COMPLETE")
    metrics={};traces={}
    for arm in ARMS:
        actions=[];effects=[];correctness=[];predicted=[]
        tally={"old_before":0,"old_after":0,"new_after":0,
               "changed_law_new_after":0,"unchanged_law_new_after":0}
        ones=0;untied=0
        for index in range(COUNT["UNIT_UNSEEN"]):
            _deadline(deadline)
            _,_,changed=source.assigned("UNIT_UNSEEN",index)
            old=source.PhysicalCausalWorld(
                "UNIT_UNSEEN",seed,index,0,_token=token)
            new=source.PhysicalCausalWorld(
                "UNIT_UNSEEN",seed,index,1,_token=token)
            for phase,w in ((0,old),(1,new)):
                q=w.public_query
                for step in range(STEPS):
                    offers=w.offers()
                    selected=choose(arm,models,layers,offers,w.history,
                                    q,phase,seed,index,step)
                    if selected not in offers:
                        raise PermissionError("R115_FUTURE_OR_UNOFFERED_ACTION")
                    sign=w.intervene(selected)
                    actions.append(selected);effects.append(int(sign>0))
                    ones+=int(sign>0)
                observed,truth=w.sealed_query()
                if observed!=q:
                    raise RuntimeError("R115_PUBLIC_QUERY_FUTURE_DRIFT")
                guess,tie=source.posterior_prediction(q,w.history)
                correct=int(guess==truth)
                correctness.append(correct);predicted.append(int(guess>0))
                untied+=int(not tie)
                if phase==0:
                    tally["old_before"]+=correct
                else:
                    tally["new_after"]+=correct
                    kind="changed_law_new_after" if changed else "unchanged_law_new_after"
                    tally[kind]+=correct
            # Immutable old context reload; never a trained retention claim.
            p,t=old.sealed_query()
            g,_=source.posterior_prediction(p,old.history)
            tally["old_after"]+=int(g==t)
        if len(actions)!=COUNT["UNIT_UNSEEN"]*2*STEPS or len(correctness)!=256:
            raise RuntimeError("R115_UNMATCHED_REAL_FACTUAL_PROBE_COUNT")
        traces[arm]=actions
        metrics[arm]={
            **{k+"_correct":int(v) for k,v in tally.items()},
            **{k+"_accuracy":v/(64 if k.startswith(("changed_","unchanged_"))
                                 else COUNT["UNIT_UNSEEN"])
               for k,v in tally.items()},
            "actual_chosen_probe_count":len(actions),
            "positive_observed_effect_count":ones,
            "nontied_exact_query_reads":untied,
            "actions_sha256":_hash_octets(actions),
            "signed_effects_sha256":_hash_octets(effects),
            "query_correctness_sha256":_hash_octets(correctness),
            "query_predicted_sign_sha256":_hash_octets(predicted)}
    div=sum(x!=y for x,y in zip(traces[NATIVE],traces[RANDOM]))
    return metrics,div


def frozen_checks(evals,divergence,cold):
    required=frozen()["evaluation"]["precommitted_each_seed_all_required"]
    acc=lambda a,key="new_after_accuracy":evals[a][key]
    values={
        "native_new_accuracy_min":acc(NATIVE),
        "native_minus_random_new_accuracy_min":acc(NATIVE)-acc(RANDOM),
        "native_minus_lowcount_new_accuracy_min":acc(NATIVE)-acc(LOW),
        "native_minus_fixed_first_new_accuracy_min":acc(NATIVE)-acc(FIXED),
        "native_minus_best_fully_trained_conventional_new_accuracy_min":
             acc(NATIVE)-max(acc(MLP),acc(GRU)),
        "native_minus_query_blind_new_accuracy_min":acc(NATIVE)-acc(QBLIND),
        "native_minus_feedback_blind_new_accuracy_min":acc(NATIVE)-acc(FBLIND),
        "native_minus_full_Bayes_VOI_new_accuracy_min":acc(NATIVE)-acc(BAYES),
        "full_Bayes_VOI_minus_random_new_accuracy_min":acc(BAYES)-acc(RANDOM),
        "native_changed_law_minus_random_min":
             acc(NATIVE,"changed_law_new_after_accuracy")-
             acc(RANDOM,"changed_law_new_after_accuracy"),
        "native_unchanged_law_minus_random_min":
             acc(NATIVE,"unchanged_law_new_after_accuracy")-
             acc(RANDOM,"unchanged_law_new_after_accuracy"),
        "native_old_before_accuracy_min":acc(NATIVE,"old_before_accuracy"),
        "old_before_minus_old_after_accuracy_max":
             acc(NATIVE,"old_before_accuracy")-acc(NATIVE,"old_after_accuracy"),
        "native_vs_random_probe_divergence_min":divergence,
        "native_integer_independent_cold_reload_mismatches_max":cold,
        "all_policy_actual_chosen_probe_budget_exact":
             all(evals[a]["actual_chosen_probe_count"]==128*10 for a in ARMS)}
    if set(values)!=set(required):
        raise RuntimeError("R115_FROZEN_SCIENCE_METRIC_MAPPING_DRIFT")
    outcome={k:(x is True if type(required[k]) is bool else
                x<=required[k] if k.endswith("_max") else x>=required[k])
             for k,x in values.items()}
    return {"values":values,"checks":outcome,
            "all_unit_only_checks_pass":all(outcome.values()),
            "scientific_pass":False}


def run_seed(seed):
    if seed not in SEEDS:
        raise PermissionError("R115_UNREGISTERED_FIRST_USE_SEED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    stop=start+frozen()["training"]["per_seed_CPU_wall_budget_seconds"]
    fit=fit_examples(seed,stop)
    rng=np.random.default_rng(np.random.SeedSequence([
        SALTS["UNIT_BATCH"],seed,115]))
    indices=rng.integers(0,len(fit[-1]),size=(512,16))
    models=new_models(seed)
    trained={arm:train(models[arm],arm,fit,indices,stop)
             for arm in TRAINABLE}
    if (set(trained)!=set(TRAINABLE)
        or any(t["updates"]!=512 for t in trained.values())):
        raise PermissionError("R115_HELDOUT_NOT_ADMITTED_BEFORE_ALL_CONTROLS")
    layers,sha=quantize_native(models[NATIVE])
    cold=integer_cold_restore(layers)
    evals,divergence=evaluate(models,layers,seed,stop,source._SCORER_ONLY)
    checks=frozen_checks(evals,divergence,cold)
    _deadline(stop)
    return {"seed":seed,"status":(
                frozen()["evaluation"]["pass_status"] if checks["all_unit_only_checks_pass"]
                else frozen()["evaluation"]["fail_status"]),
            "all_three_fresh_policies_512_updates_completed":True,
            "training":trained,"native_quantized_weight_sha256":sha,
            "evaluation":evals,"native_random_action_divergence":divergence,
            "cold_independent_integer_score_mismatch_count":cold,
            "frozen_checks":checks,
            "process_wall_seconds":time.perf_counter()-start,
            "original_science_firstuse":False,"scientific_pass":False}


def run_all():
    frozen()
    if (ROOT/"experiments/r115_protected_original_scientific_firstuse_authorization.json").exists():
        raise PermissionError("R115_PROTECTED_ORIGINAL_SOURCE_NOT_AUTHORIZED")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R115_EXACT_CANDIDATE_SOURCE_DRIFT")
    scores={str(seed):run_seed(seed) for seed in SEEDS}
    ok=all(v["frozen_checks"]["all_unit_only_checks_pass"] for v in scores.values())
    return {"schema":"ttt-hybrid.r115.original-query-targeted-active-causal.unit-only.v1",
            "candidate_exact_sha":git,"preimplementation_freeze_blob":custody.FREEZE_BLOB,
            "status":(frozen()["evaluation"]["pass_status"] if ok else
                      frozen()["evaluation"]["fail_status"]),
            "by_seed":scores,"all_3_frozen_criteria_pass":ok,
            "protected_scientific_firstuse":False,"GPU_firstuse":False,
            "scientific_pass":False,"model_promotion":False,
            "deployment":"HOLD"}


def main():
    d=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R115_ORIGINAL_UNIT_ONLY_RECEIPT_REPLAY_DENIED")
    RECEIPT.write_text(json.dumps(d,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R115_UNIT_ONLY "+json.dumps({
        "sha":d["candidate_exact_sha"],"status":d["status"],
        "seeds":{seed:{
                   "new_correct":{a:row["new_after_correct"]
                                  for a,row in v["evaluation"].items()},
                   "criteria":v["frozen_checks"]["checks"],
                   "native_weight_sha":v["native_quantized_weight_sha256"],
                   "divergence":v["native_random_action_divergence"],
                   "cold":v["cold_independent_integer_score_mismatch_count"],
                   "seconds":round(v["process_wall_seconds"],2)}
                 for seed,v in d["by_seed"].items()},
        "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
