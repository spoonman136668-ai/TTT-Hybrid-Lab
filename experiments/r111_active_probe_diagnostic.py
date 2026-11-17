"""R111: real chosen-action causal probe policies in novel physical Boolean worlds.

UNIT_ONLY development. The returned signed outcome is computed only AFTER a
public do(role_factors=chosen_pattern) intervention. No counterfactual outcomes,
hidden changed-law labels, heldout query, or evaluator mask is passed to a
policy. Shared fixed Bayesian readout isolates action choice from prediction.
This is a small synthetic substrate, NOT qualified native Hybrid intelligence.
"""
from __future__ import annotations

import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r111_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
SEEDS=(1011011,1011023,1011037)
TRAIN=tuple(int(x,16) for x in ("ca","96","80","fe","7f","e8","16","68","81","87","e1"))
UNSEEN=tuple(int(x,16) for x in ("04","0b","32","3d","51","5e","ae","cd"))
SALTS={"UNIT_FIT":27101101,"UNIT_VALID":27101107,
       "UNIT_UNSEEN":27101113,"UNIT_BATCH":27101119,
       "UNIT_COLD":27101123,"UNIT_CHOICES":27101129,
       "UNIT_OUTCOMES":27101131,"UNIT_QUERIES":27101137}
COUNTS={"UNIT_FIT":176,"UNIT_VALID":88,"UNIT_UNSEEN":128}
STEPS=6
ARMS=("R111_NATIVE_LEARNED","R111_MLP_FULLY_TRAINED","R111_GRU_FULLY_TRAINED",
      "R111_EXACT_PUBLIC_BAYES_IG","R111_LOW_COUNT",
      "R111_RANDOM_MATCHED","R111_FIXED_FIRST",
      "R111_NATIVE_FEEDBACK_BLIND")
NATIVE,MLP,GRU,BAYESIG,COUNT,RANDOM,FIXED,BLIND=ARMS
TRAINABLE=(NATIVE,MLP,GRU)
RECEIPT=ROOT/"local_runs/r111_unit_only_active_causal_probes.json"
EPS=.15
LOGIT_STEP=math.log((1.-EPS)/EPS)


def frozen():
    custody.audit_freeze()
    f=json.loads(custody.FREEZE.read_text(encoding="utf-8"))
    src=f["source"];t=f["training"]
    if (tuple(src["seeds"])!=SEEDS
        or tuple(int(x,16) for x in src["train_masks_hex"])!=TRAIN
        or tuple(int(x,16) for x in src["fresh_unseen_masks_hex"])!=UNSEEN
        or src["panel_salts"]!=SALTS
        or src["worlds_per_seed"]!=COUNTS
        or src["support_probe_budget_per_phase"]!=STEPS
        or src["outcome_noise_flip_probability"]!=EPS
        or t["exact_training_steps_each"]!=512
        or t["batch_size"]!=16
        or t["maximum_CPU_seconds_per_seed"]!=240):
        raise PermissionError("R111_FROZEN_SOURCE_OR_TRAIN_BUDGET_DRIFT")
    return f


def deadline_check(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError("R111_240S_PER_SEED_CPU_BUDGET_EXCEEDED")


def assign_masks(panel,index):
    if panel not in COUNTS or type(index) is not int or not 0<=index<COUNTS[panel]:
        raise PermissionError("R111_INVALID_SOURCE_PANEL_OR_EPISODE")
    masks=UNSEEN if panel=="UNIT_UNSEEN" else TRAIN
    n=len(masks)
    old=masks[index%n]
    changed=(index//n)%2==1
    new=masks[(index+3)%n] if changed else old
    return old,new,bool(changed)


def entropy(p):
    p=max(1e-9,min(1.-1e-9,float(p)))
    return -p*math.log2(p)-(1.-p)*math.log2(1.-p)


def posterior(pattern,history):
    if type(pattern) is not int or not 0<=pattern<8:
        raise ValueError("R111_UNKNOWN_INTERVENTION_PATTERN")
    signed=sum(int(reward) for role,reward in history if role==pattern)
    logit=LOGIT_STEP*signed
    return 1./(1.+math.exp(-logit))


def information_gain(pattern,history):
    p=posterior(pattern,history)
    prob_plus=EPS+(1.-2*EPS)*p
    p_if_plus=(1.-EPS)*p/prob_plus
    p_if_minus=EPS*p/(1.-prob_plus)
    return max(0.,entropy(p)-prob_plus*entropy(p_if_plus)
               -(1.-prob_plus)*entropy(p_if_minus))


def fixed_readout(pattern,history):
    # Common Bayes-optimal decision under independent 256-rule prior.
    # Ties use +1 throughout; no method sees an unchosen factual return.
    return 1 if posterior(pattern,history)>=.5 else -1


class PhysicalCausalWorld:
    """The only portal allowed to produce a factual intervention outcome.

    A world exposes public candidate actions, then consumes exactly one
    selected intervention at each step; neither an oracle-label nor an
    unchosen-action reward accessor is available to a policy.
    """
    def __init__(self,panel,seed,index,phase):
        frozen()
        if seed not in SEEDS or phase not in (0,1):
            raise PermissionError("R111_UNREGISTERED_SEED_OR_PHASE")
        old,new,changed=assign_masks(panel,index)
        self._mask=old if phase==0 else new
        self._panel=panel
        self._seed=seed
        self._index=index
        self._phase=phase
        self._position=0
        self.history=[]
        self.records=[]
        self._changed=changed

    def offers(self):
        if self._position>=STEPS:
            raise PermissionError("R111_PUBLIC_PROBES_FINISHED")
        rng=np.random.default_rng(np.random.SeedSequence(
            [SALTS[self._panel],SALTS["UNIT_CHOICES"],self._seed,
             self._index,self._phase,self._position]))
        return tuple(sorted(int(x) for x in rng.choice(8,size=2,replace=False)))

    def intervene(self,chosen):
        options=self.offers()
        if type(chosen) is not int or chosen not in options:
            raise PermissionError("R111_UNOFFERED_OR_UNCHOSEN_PROBE_DENIED")
        # The world's hidden Boolean law is evaluated for ONLY this chosen do()
        # intervention; no paired counterfactual outcome is generated.
        result=1 if (self._mask>>chosen)&1 else -1
        rng=np.random.default_rng(np.random.SeedSequence(
            [SALTS[self._panel],SALTS["UNIT_OUTCOMES"],self._seed,
             self._index,self._phase,self._position,chosen]))
        if bool(rng.random()<EPS):
            result=-result
        self.history.append((chosen,result))
        self.records.append({"chosen":chosen,"observed_sign":result})
        self._position+=1
        return result

    def sealed_query(self):
        if self._position!=STEPS or len(self.records)!=STEPS:
            raise PermissionError("R111_QUERY_BEFORE_ALL_CHOSEN_FACTUAL_PROBES")
        rng=np.random.default_rng(np.random.SeedSequence(
            [SALTS[self._panel],SALTS["UNIT_QUERIES"],self._seed,
             self._index,self._phase]))
        pattern=int(rng.integers(0,8))
        # Evaluator-only truth is returned AFTER all chosen actions; never
        # present the label, query or private mask to the acting controller.
        answer=1 if (self._mask>>pattern)&1 else -1
        return pattern,answer


def features(history,offers,phase,*,blind=False):
    if phase not in (0,1) or len(offers)!=2 or offers[0]==offers[1] or any(
            type(p) is not int or not 0<=p<8 for p in offers):
        raise ValueError("R111_INVALID_PUBLIC_CANDIDATE_PROBES")
    if len(history)>STEPS:
        raise ValueError("R111_TOO_MANY_PUBLIC_PROBE_OUTCOMES")
    counts=np.zeros(8,np.float32)
    signed=np.zeros(8,np.float32)
    seq=np.zeros((STEPS,10),np.float32)
    for i,(pattern,reward) in enumerate(history):
        if type(pattern) is not int or not 0<=pattern<8 or reward not in (-1,1):
            raise ValueError("R111_INVALID_CHOSEN_FACTUAL_HISTORY")
        counts[pattern]+=1
        signed[pattern]+=reward
        seq[i,pattern]=1.
        seq[i,8]=reward
        seq[i,9]=1.
    policy_signed=np.zeros_like(signed) if blind else signed
    native=np.array([[counts[p]/STEPS,policy_signed[p]/STEPS,
                      abs(policy_signed[p])/STEPS,float(phase),len(history)/STEPS]
                     for p in offers],np.float32)
    offers_hot=np.eye(8,dtype=np.float32)[list(offers)]
    mlp=np.concatenate((counts/STEPS,policy_signed/STEPS,
                        offers_hot.reshape(-1),[float(phase),len(history)/STEPS]))
    if mlp.shape!=(34,) or native.shape!=(2,5) or seq.shape!=(6,10):
        raise RuntimeError("R111_PUBLIC_TOKENLESS_FEATURE_SHAPE_FAILURE")
    if blind:
        seq[:,8]=0.
    return native,np.asarray(mlp,np.float32),seq


class NativeProbePolicy(nn.Module):
    """Eight-address factual memory + parameter-shared physical probe scorer."""
    def __init__(self):
        super().__init__()
        self.scorer=nn.Sequential(nn.Linear(5,32),nn.GELU(),
                                  nn.Linear(32,16),nn.GELU(),nn.Linear(16,1))

    def forward(self,native,mlp,seq):
        if native.ndim!=3 or native.shape[1:]!=(2,5):
            raise RuntimeError("R111_NATIVE_MEMORY_FEATURE_SHAPE")
        return self.scorer(native).squeeze(-1)


class MLPProbePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(nn.Linear(34,48),nn.GELU(),
                                 nn.Linear(48,32),nn.GELU(),nn.Linear(32,2))

    def forward(self,native,mlp,seq):
        if mlp.ndim!=2 or mlp.shape[-1]!=34:
            raise RuntimeError("R111_MLP_HISTORY_FEATURE_SHAPE")
        return self.model(mlp)


class GRUProbePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.memory=nn.GRU(10,32,batch_first=True)
        self.readout=nn.Sequential(nn.Linear(32+18,48),nn.GELU(),
                                   nn.Linear(48,2))

    def forward(self,native,mlp,seq):
        if seq.ndim!=3 or seq.shape[1:]!=(6,10) or mlp.shape!=(len(seq),34):
            raise RuntimeError("R111_GRU_HISTORY_FEATURE_SHAPE")
        # Using the last valid factual event maintains strict observed-order.
        full,_=self.memory(seq)
        lengths=(seq[:,:,9]>0).sum(dim=1).clamp(min=1).long()-1
        h=full[torch.arange(len(seq)),lengths]
        return self.readout(torch.cat((h,mlp[:,16:]),dim=-1))


def new_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R111_UNREGISTERED_TRAIN_SEED")
    torch.manual_seed(seed+111)
    models={NATIVE:NativeProbePolicy()}
    torch.manual_seed(seed+211)
    models[MLP]=MLPProbePolicy()
    torch.manual_seed(seed+311)
    models[GRU]=GRUProbePolicy()
    return models


def fit_examples(seed,deadline):
    frozen()
    if seed not in SEEDS:
        raise PermissionError("R111_UNREGISTERED_FIT_SEED")
    feat_n=[];feat_m=[];feat_g=[];targets=[]
    for world_index in range(COUNTS["UNIT_FIT"]):
        deadline_check(deadline)
        for phase in (0,1):
            world=PhysicalCausalWorld("UNIT_FIT",seed,world_index,phase)
            for step in range(STEPS):
                options=world.offers()
                n,m,g=features(world.history,options,phase)
                target=[information_gain(p,world.history) for p in options]
                feat_n.append(n);feat_m.append(m);feat_g.append(g)
                targets.append(target)
                # Public deterministic randomized FIT-only exploration; action
                # chosen before (and without) either option's outcome.
                rng=np.random.default_rng(np.random.SeedSequence(
                    [SALTS["UNIT_FIT"],SALTS["UNIT_BATCH"],seed,world_index,phase,step]))
                choice=int(options[int(rng.integers(0,2))])
                world.intervene(choice)
            if len(world.records)!=STEPS:
                raise RuntimeError("R111_FIT_CHOSEN_ONLY_BUDGET_NOT_CONSUMED")
    n=np.asarray(feat_n,np.float32);m=np.asarray(feat_m,np.float32)
    g=np.asarray(feat_g,np.float32);y=np.asarray(targets,np.float32)
    if (n.shape!=(2112,2,5) or m.shape!=(2112,34) or
        g.shape!=(2112,6,10) or y.shape!=(2112,2) or
        not np.isfinite(y).all() or not bool(np.all((y>=0)&(y<=1)))):
        raise RuntimeError("R111_FIT_EXAMPLES_PUBLIC_SOURCE_DRIFT")
    return (torch.from_numpy(n),torch.from_numpy(m),torch.from_numpy(g),
            torch.from_numpy(y))


def train_model(model,name,train_data,minibatches,deadline):
    f=frozen()["training"]
    if name not in TRAINABLE or minibatches.shape!=(512,16):
        raise PermissionError("R111_UNREGISTERED_TRAINABLE_OR_OPTIMIZER_BUDGET")
    optimizer=torch.optim.AdamW(model.parameters(),lr=f["learning_rate"],
                                 weight_decay=f["weight_decay"])
    start=time.perf_counter()
    model.train()
    n,m,g,y=train_data
    for ix in minibatches:
        deadline_check(deadline)
        estimates=model(n[ix],m[ix],g[ix])
        if estimates.shape!=(16,2):
            raise RuntimeError("R111_TRAINED_VALUE_OUTPUT_SHAPE_INVALID")
        loss=F.mse_loss(estimates,y[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R111_NONFINITE_PROBE_VALUE_TRAINING")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),f["gradient_clip"])
        optimizer.step()
    deadline_check(deadline)
    return {"updates":512,
            "parameters":sum(p.numel() for p in model.parameters()),
            "fit_seconds":time.perf_counter()-start}


def choose(arm,models,offers,history,phase,seed,index,step):
    if arm not in ARMS:
        raise PermissionError("R111_UNREGISTERED_POLICY_ARM")
    if arm==FIXED:
        return int(offers[0])
    if arm==RANDOM:
        rng=np.random.default_rng(np.random.SeedSequence(
            [SALTS["UNIT_CHOICES"],seed,index,phase,step,111]))
        return int(offers[int(rng.integers(0,2))])
    if arm==COUNT:
        counts=[sum(p==q for p,_ in history) for q in offers]
        return int(offers[int(counts[1]<counts[0])])
    if arm==BAYESIG:
        gains=[information_gain(p,history) for p in offers]
        return int(offers[int(gains[1]>gains[0]+1e-10)])
    n,m,g=features(history,offers,phase,blind=arm==BLIND)
    model=models[NATIVE if arm==BLIND else arm]
    model.eval()
    with torch.no_grad():
        scores=model(torch.from_numpy(n[None]),
                     torch.from_numpy(m[None]),
                     torch.from_numpy(g[None]))[0]
    if scores.shape!=(2,) or not bool(torch.isfinite(scores).all()):
        raise RuntimeError("R111_NONFINITE_NATIVE_EXPERIMENT_VALUE")
    return int(offers[int(scores[1].item()>scores[0].item()+1e-10)])


def evaluate_seed(models,seed,deadline):
    # The FIRST point where any R111 unseen world/mask/query is generated.
    # Fit of ALL scratch native and conventional models MUST be completed first.
    if seed not in SEEDS:
        raise PermissionError("R111_UNKNOWN_SEED_BEFORE_HELDOUT")
    by_arm={};trace={}
    for arm in ARMS:
        deadline_check(deadline)
        correct={"old_before":0,"old_after":0,"new_after":0,
                 "changed_law_new_after":0,"unchanged_law_new_after":0}
        unique=0;positives=0;observations=0;choices=[]
        for index in range(COUNTS["UNIT_UNSEEN"]):
            _,_,changed=assign_masks("UNIT_UNSEEN",index)
            world0=PhysicalCausalWorld("UNIT_UNSEEN",seed,index,0)
            world1=PhysicalCausalWorld("UNIT_UNSEEN",seed,index,1)
            for phase,world in ((0,world0),(1,world1)):
                for step in range(STEPS):
                    deadline_check(deadline)
                    offers=world.offers()
                    action=choose(arm,models,offers,world.history,phase,seed,index,step)
                    if action not in offers:
                        raise RuntimeError("R111_ILLEGAL_CHOSEN_CAUSAL_ACTION")
                    result=world.intervene(action)
                    choices.append(action)
                    positives+=int(result==1)
                    observations+=1
                unique+=len(set(pattern for pattern,_ in world.history))
                pattern,truth=world.sealed_query()
                estimate=fixed_readout(pattern,world.history)
                if phase==0:
                    correct["old_before"]+=int(estimate==truth)
                else:
                    correct["new_after"]+=int(estimate==truth)
                    key="changed_law_new_after" if changed else "unchanged_law_new_after"
                    correct[key]+=int(estimate==truth)
            # Return to the original phase AFTER learning/observing B; do not
            # modify old memory. Frozen history is same, hence real retention
            # metric is a state-integrity test, not evidence of lifelong learning.
            old_pattern,old_truth=world0.sealed_query()
            correct["old_after"]+=int(fixed_readout(old_pattern,world0.history)==old_truth)
        if observations!=COUNTS["UNIT_UNSEEN"]*12 or len(choices)!=observations:
            raise RuntimeError("R111_OBSERVATION_BUDGET_OR_PROBE_LOG_MISSING")
        by_arm[arm]={
            **{key+"_correct":v for key,v in correct.items()},
            **{key+"_accuracy":v/(64 if key.startswith(("changed_","unchanged_")) else 128)
               for key,v in correct.items()},
            "mean_distinct_probes_per_phase":unique/(2*COUNTS["UNIT_UNSEEN"]),
            "positive_factual_signs":positives,
            "physically_chosen_probe_count":observations}
        trace[arm]=choices
    divergence=sum(int(x!=y) for x,y in zip(trace[NATIVE],trace[RANDOM]))
    return by_arm,divergence


def independent_cold_reload(model,train_data,deadline):
    deadline_check(deadline)
    model.eval()
    n,m,g,_=train_data
    sample=(n[:8],m[:8],g[:8])
    with torch.no_grad():
        original=model(*sample)
    stream=io.BytesIO()
    torch.save({"state":model.state_dict(),"samples":sample},stream)
    code=("import io,sys,torch\n"
          "from experiments.r111_active_probe_diagnostic import NativeProbePolicy\n"
          "torch.set_num_threads(1)\n"
          "p=torch.load(io.BytesIO(sys.stdin.buffer.read()),map_location='cpu',weights_only=True)\n"
          "model=NativeProbePolicy(); model.load_state_dict(p['state']); model.eval()\n"
          "with torch.no_grad(): out=model(*p['samples'])\n"
          "torch.save(out,sys.stdout.buffer)\n")
    proc=subprocess.run([sys.executable,"-c",code],input=stream.getvalue(),
                         capture_output=True,cwd=ROOT,check=True,timeout=30)
    child=torch.load(io.BytesIO(proc.stdout),map_location="cpu",weights_only=True)
    if child.shape!=original.shape:
        raise RuntimeError("R111_INDEPENDENT_COLD_SHAPE_DRIFT")
    deadline_check(deadline)
    return int(torch.count_nonzero(original!=child))


def frozen_checks(scores,divergence,cold):
    threshold=frozen()["evaluation"]["frozen_checks_all_seeds"]
    acc=lambda a,k="new_after_accuracy":scores[a][k]
    values={
      "native_unfamiliar_new_accuracy_min":acc(NATIVE),
      "native_minus_random_unfamiliar_accuracy_min":acc(NATIVE)-acc(RANDOM),
      "native_minus_lowcount_unfamiliar_accuracy_min":acc(NATIVE)-acc(COUNT),
      "native_minus_fixed_first_unfamiliar_accuracy_min":acc(NATIVE)-acc(FIXED),
      "native_minus_strongest_trained_conventional_min":
          acc(NATIVE)-max(acc(MLP),acc(GRU)),
      "native_minus_information_gain_oracle_min":acc(NATIVE)-acc(BAYESIG),
      "native_minus_feedback_blind_unfamiliar_min":acc(NATIVE)-acc(BLIND),
      "native_changed_law_minus_random_min":
          acc(NATIVE,"changed_law_new_after_accuracy")-
          acc(RANDOM,"changed_law_new_after_accuracy"),
      "native_unchanged_law_minus_random_min":
          acc(NATIVE,"unchanged_law_new_after_accuracy")-
          acc(RANDOM,"unchanged_law_new_after_accuracy"),
      "native_old_before_accuracy_min":acc(NATIVE,"old_before_accuracy"),
      "old_before_minus_old_after_accuracy_max":
          acc(NATIVE,"old_before_accuracy")-acc(NATIVE,"old_after_accuracy"),
      "information_gain_oracle_minus_random_min":acc(BAYESIG)-acc(RANDOM),
      "native_vs_random_probe_choice_divergence_count_min":divergence,
      "independent_process_cold_restore_mismatch_count_max":cold,
      "every_policy_executed_exact_chosen_physical_probe_budget":
          int(all(scores[a]["physically_chosen_probe_count"]==128*12 for a in ARMS))}
    if set(values)!=set(threshold):
        raise RuntimeError("R111_FROZEN_THRESHOLD_NAMES_DRIFT")
    passed={k:(v<=threshold[k] if k.endswith("_max") else v>=threshold[k])
            for k,v in values.items()}
    return {"values":values,"checks":passed,
            "all_unit_only_checks_pass":all(passed.values()),"scientific_pass":False}


def run_seed(seed):
    if seed not in SEEDS:
        raise PermissionError("R111_UNREGISTERED_FRESH_SEED")
    frozen()
    torch.set_num_threads(1)
    start=time.perf_counter()
    deadline=start+frozen()["training"]["maximum_CPU_seconds_per_seed"]
    fit=fit_examples(seed,deadline)
    rng=np.random.default_rng(np.random.SeedSequence([SALTS["UNIT_BATCH"],seed,111]))
    minibatches=rng.integers(0,len(fit[3]),size=(512,16))
    models=new_models(seed)
    trained={a:train_model(models[a],a,fit,minibatches,deadline) for a in TRAINABLE}
    if set(trained)!=set(TRAINABLE) or any(x["updates"]!=512 for x in trained.values()):
        raise RuntimeError("R111_HELDOUT_BEFORE_ALL_TRAINABLE_CONTROLS_FINISH")
    # Source first-use is only now permitted to the bounded UNIT_ONLY scorer.
    scores,divergence=evaluate_seed(models,seed,deadline)
    cold=independent_cold_reload(models[NATIVE],fit,deadline)
    criteria=frozen_checks(scores,divergence,cold)
    deadline_check(deadline)
    return {"seed":seed,
            "status":(frozen()["evaluation"]["status_dev_pass"]
                      if criteria["all_unit_only_checks_pass"]
                      else frozen()["evaluation"]["status_dev_fail"]),
            "fit_public_training_examples":len(fit[3]),
            "all_trainable_arms_completed":True,
            "trained_models":trained,"evaluations":scores,
            "probe_choice_divergence_vs_random":divergence,
            "frozen_checks":criteria,"cold_reload_mismatch_count":cold,
            "whole_process_seconds":time.perf_counter()-start,
            "original_scientific_firstuse":False,"scientific_pass":False}


def run_all():
    frozen()
    if (ROOT/"experiments/r111_protected_original_firstuse_authorization.json").exists():
        raise PermissionError("R111_PROTECTED_SCIENCE_NOT_AUTHORIZED")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R111_CANDIDATE_SHA_DRIFT")
    scored={str(seed):run_seed(seed) for seed in SEEDS}
    passed=all(r["frozen_checks"]["all_unit_only_checks_pass"] for r in scored.values())
    return {"schema":"ttt-hybrid.r111.unit-only-active-chosen-causal-probes.v1",
            "candidate_exact_sha":git,"freeze_blob":custody.FREEZE_BLOB,
            "status":(frozen()["evaluation"]["status_dev_pass"]
                      if passed else frozen()["evaluation"]["status_dev_fail"]),
            "by_seed":scored,"all_three_unit_only_criteria_pass":passed,
            "original_scientific_firstuse":False,"GPU_firstuse":False,
            "scientific_pass":False,"promotion":False,"deployment":"HOLD"}


def main():
    data=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R111_IMMUTABLE_UNIT_ONLY_RECEIPT_OVERWRITE_DENIED")
    RECEIPT.write_text(json.dumps(data,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R111_UNIT_ONLY "+json.dumps({
      "sha":data["candidate_exact_sha"],"status":data["status"],
      "by_seed":{k:{"actions":{a:v["new_after_correct"]
                            for a,v in r["evaluations"].items()},
                    "criteria":r["frozen_checks"]["checks"],
                    "divergence":r["probe_choice_divergence_vs_random"],
                    "seconds":round(r["whole_process_seconds"],3)}
                 for k,r in data["by_seed"].items()},
      "scientific_pass":False},sort_keys=True))


if __name__=="__main__":
    main()
