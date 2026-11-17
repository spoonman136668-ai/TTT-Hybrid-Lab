"""R116 frozen matched-sign causal-feedback probe test (UNIT_ONLY DEV).

Three truly trained, FIT-only scratch policies choose the fourth physical
intervention after three observed chosen-action consequences. Matched paired
unfamiliar SCMs give identical public query, offers and first-two responses,
and opposite diagnostic response. Original protected/scored cohorts denied.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r115_query_targeted_diagnostic as r115
from experiments import r116_source_custody as custody
from experiments import r116_pair_source as source

ROOT=Path(__file__).resolve().parents[1]
SEEDS=custody.SEEDS
ARMS=("R116_NATIVE_SIGN_INTEGER","R116_CONVENTIONAL_MLP","R116_CONVENTIONAL_GRU",
      "R116_EXACT_KNOWN_CLASS_BAYES_VOI","R116_SIGN_HANDCODED",
      "R116_FIXED_DO12","R116_FIXED_DO14","R116_SEEDED_RANDOM",
      "R116_NATIVE_FEEDBACK_BLIND","R116_NATIVE_DIAGNOSTIC_SIGN_INVERT",
      "R116_STATIC_COUNT_ONLY")
NATIVE,MLP,GRU,BAYES,RULE,F12,F14,RANDOM,BLIND,INVERT,COUNT=ARMS
TRAINABLE=(NATIVE,MLP,GRU)
RECEIPT=ROOT/"local_runs/r116_factual_pair_original_unit_only.json"


def freeze():
    f=source.admit()
    t=f["training"]
    if (t["updates_per_trainable_arm"]!=512 or
        t["batch_size"]!=16 or t["fit_samples_per_seed"]!=256 or
        t["optimizer"]!="AdamW" or t["learning_rate"]!=.002 or
        t["weight_decay"]!=.0001 or t["gradient_clip"]!=1 or
        t["torch_CPU_threads"]!=1 or
        t["trainable_policies"]!=list(TRAINABLE) or
        f["source"]["per_world_total_physical_actions"]!=4):
        raise PermissionError("R116_TRAINING_FREEZE_OR_ACTION_BUDGET_DRIFT")
    return f


def new_models(seed):
    if seed not in SEEDS:
        raise PermissionError("R116_UNREGISTERED_MODEL_SEED")
    torch.manual_seed(seed+116)
    n=r115.NativePolicy()
    torch.manual_seed(seed+216)
    m=r115.MLPPolicy()
    torch.manual_seed(seed+316)
    g=r115.GRUPolicy()
    return {NATIVE:n,MLP:m,GRU:g}


def make_fit(seed):
    if seed not in SEEDS:
        raise PermissionError("R116_UNREGISTERED_FIT_WORLD")
    source.admit()
    inputs=[];mlps=[];seqs=[];targets=[]
    plus=minus=0
    for index in range(256):
        w=source.PhysicalPairWorld("UNIT_FIT",seed,index)
        for required in source.ACTIONS:
            if w.offers()!=(required,):
                raise RuntimeError("R116_PHYSICAL_FIXED_CALIBRATION_DRIFT")
            w.intervene(required)
        plus+=int(w.history[2][1]>0)
        minus+=int(w.history[2][1]<0)
        n,m,g=r115.public_inputs(w.history,source.FINAL,source.QUERY,0)
        target=source.public_observation_target(w.history)
        inputs.append(n);mlps.append(m);seqs.append(g);targets.append(target)
    d=tuple(torch.from_numpy(np.asarray(x,np.float32))
            for x in (inputs,mlps,seqs,targets))
    if (tuple(x.shape for x in d)!=
        ((256,2,45),(256,90),(256,5,18),(256,2))
        or not bool(torch.isfinite(d[-1]).all()) or
        not bool((d[-1]>=0).all()) or plus<90 or minus<90 or
        not bool(torch.any(d[-1][:,0]!=d[-1][:,1]))):
        raise RuntimeError("R116_NOT_MATCHED_FIT_FACTUAL_SIGN_CLASSES")
    return d,{"factual_positive_diagnostic":plus,
              "factual_negative_diagnostic":minus}


def train_arm(model,arm,fit,draws):
    f=freeze()["training"]
    if arm not in TRAINABLE or draws.shape!=(512,16):
        raise PermissionError("R116_FROZEN_FIT_STEP_BUDGET")
    n,m,g,target=fit
    opt=torch.optim.AdamW(model.parameters(),lr=f["learning_rate"],
                          weight_decay=f["weight_decay"])
    model.train()
    start=time.perf_counter()
    for ix in draws:
        estimate=model(n[ix],m[ix],g[ix])
        if estimate.shape!=(16,2):
            raise RuntimeError("R116_TRAINED_ACTION_PAIR_INTERFACE")
        loss=F.mse_loss(estimate,target[ix])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R116_NONFINITE_FIT_OBJECTIVE")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),f["gradient_clip"])
        opt.step()
    return {"updates":512,
            "parameters":sum(p.numel() for p in model.parameters()),
            "FIT_only_training_seconds":time.perf_counter()-start}


def choose(arm,models,layers,history,seed,index):
    if arm not in ARMS:
        raise PermissionError("R116_UNREGISTERED_NEXT_ACTION_POLICY")
    source.public_exact_voi_for_next(history)
    signed=history[2][1]
    if arm in (F12,COUNT):return 12
    if arm==F14:return 14
    if arm==RANDOM:
        rng=np.random.default_rng(np.random.SeedSequence([
            27601123,seed,index,116]))
        return int(source.FINAL[int(rng.integers(0,2))])
    if arm==RULE:
        return 14 if signed<0 else 12
    if arm==BAYES:
        gains=source.public_exact_voi_for_next(history)
        return int(source.FINAL[int(gains[1][0]>gains[0][0])])
    if arm in (NATIVE,BLIND,INVERT):
        facts=([(p,-s if p==7 else s) for p,s in history]
               if arm==INVERT else history)
        score=r115.native_scores(layers,facts,source.FINAL,source.QUERY,0,
                                 feedback_blind=arm==BLIND)
        return int(source.FINAL[int(score[1]>score[0])])
    n,m,g=r115.public_inputs(history,source.FINAL,source.QUERY,0)
    models[arm].eval()
    with torch.no_grad():
        z=models[arm](torch.from_numpy(n[None]),torch.from_numpy(m[None]),
                      torch.from_numpy(g[None]))[0]
    if z.shape!=(2,) or not bool(torch.isfinite(z).all()):
        raise RuntimeError("R116_CONVENTIONAL_PUBLIC_ACTION_SCORE_DRIFT")
    return int(source.FINAL[int(z[1].item()>z[0].item()+1e-10)])


def _digest(values):
    if not values or any(type(v) is not int or v<0 or v>255 for v in values):
        raise ValueError("R116_NONCANONICAL_PHYSICAL_HASH")
    return hashlib.sha256(bytes(values)).hexdigest()


def evaluate(models,layers,seed,_token):
    if _token is not source._SCORER_ONLY or seed not in SEEDS:
        raise PermissionError("R116_HELDOUT_PRIOR_TO_ALL_512_UPDATES")
    by_arm={};actions={};native_correctness=None
    for arm in ARMS:
        chosen=[];physical=[];predictions=[];correctness=[]
        calibrated=0;correct=0;negcorrect=0;poscorrect=0
        for index in range(128):
            w=source.PhysicalPairWorld("UNIT_UNSEEN",seed,index,_token=_token)
            for p in source.ACTIONS:
                if w.offers()!=(p,):
                    raise RuntimeError("R116_PREPARATORY_INTERVENTION_CHANGED")
                sign=w.intervene(p)
                chosen.append(p);physical.append(int(sign>0))
            calibrated+=int([(p,s) for p,s in w.history[:2]]==[(8,1),(9,-1)])
            a=choose(arm,models,layers,w.history,seed,index)
            if a not in source.FINAL:
                raise PermissionError("R116_UNOFFERED_ADAPTIVE_ACTION")
            sign=w.intervene(a)
            chosen.append(a);physical.append(int(sign>0))
            query,truth=w.sealed_query()
            if query!=source.QUERY:
                raise RuntimeError("R116_DIFFERENT_PUBLIC_TARGET_QUERY")
            guess,_=source.public_exact_prediction(w.history)
            ok=int(guess==truth)
            correct+=ok
            diag=w.history[2][1]
            negcorrect+=ok if diag<0 else 0
            poscorrect+=ok if diag>0 else 0
            correctness.append(ok);predictions.append(int(guess>0))
        actions[arm]=[chosen[i*4+3] for i in range(128)]
        if arm==NATIVE:
            native_correctness=tuple(correctness)
        by_arm[arm]={
            "correct":int(correct),"query_accuracy":correct/128,
            "negative_branch_correct":int(negcorrect),
            "positive_branch_correct":int(poscorrect),
            "same_public_calibration_actions":calibrated==128,
            "actual_four_chosen_actions":len(chosen),
            "action_history_sha256":_digest(chosen),
            "factual_observed_sign_sha256":_digest(physical),
            "query_prediction_sha256":_digest(predictions),
            "query_correctness_sha256":_digest(correctness)}
    flipped=both_correct=0
    native_actions=actions[NATIVE]
    # index = 8*cycle+mask_id; compare paired minus mask k vs plus k+4.
    for cycle in range(16):
        for pair in range(4):
            left=8*cycle+pair;right=left+4
            flipped+=int(native_actions[left]!=native_actions[right])
            both_correct+=int(native_correctness[left] and native_correctness[right])
    return by_arm,actions,flipped,both_correct


def evaluate_with_pairs(models,layers,seed):
    # Evaluate only after all trainables are frozen and cold restored.
    metrics,actions,flipped,both=evaluate(models,layers,seed,source._SCORER_ONLY)
    return metrics,actions,flipped,both


def frozen_checks(metrics,flips,both,cold,trained):
    t=freeze()["evaluation"]["all_3_seed_checks_frozen"]
    acc=lambda name:metrics[name]["query_accuracy"]
    values={
      "native_new_query_accuracy_min":acc(NATIVE),
      "native_minus_feedback_blind_query_accuracy_min":acc(NATIVE)-acc(BLIND),
      "native_minus_diagnostic_sign_inverted_accuracy_min":acc(NATIVE)-acc(INVERT),
      "native_minus_random_query_accuracy_min":acc(NATIVE)-acc(RANDOM),
      "native_minus_best_fixed_query_accuracy_min":
         acc(NATIVE)-max(acc(F12),acc(F14),acc(COUNT)),
      "native_minus_best_conventional_query_accuracy_min":
         acc(NATIVE)-max(acc(MLP),acc(GRU)),
      "native_minus_exact_bayes_oracle_query_accuracy_min":
         acc(NATIVE)-acc(BAYES),
      "native_minus_hardcoded_factual_rule_query_accuracy_min":
         acc(NATIVE)-acc(RULE),
      "oracle_minus_random_accuracy_min":acc(BAYES)-acc(RANDOM),
      "native_next_action_changes_for_opposite_diagnostic_sign_pairs_min":flips,
      "native_correct_on_both_members_of_pair_count_min":both,
      "native_same_public_calibration_actions":
         metrics[NATIVE]["same_public_calibration_actions"],
      "all_policies_exact_four_chosen_actions":
         all(m["actual_four_chosen_actions"]==512 for m in metrics.values()),
      "native_cold_integer_score_mismatches_max":cold,
      "all_training_arms_fully_trained_512_steps":
         set(trained)==set(TRAINABLE) and all(m["updates"]==512 for m in trained.values())}
    if set(values)!=set(t):
        raise RuntimeError("R116_PRECOMMITTED_SCIENCE_METRIC_MAPPING_DRIFT")
    checks={k:(v is True if type(t[k]) is bool else
               v<=t[k] if k.endswith("_max") else v>=t[k])
            for k,v in values.items()}
    return {"values":values,"checks":checks,
            "all_unit_only_checks_pass":all(checks.values()),
            "scientific_pass":False}


def run_seed(seed):
    torch.set_num_threads(1)
    start=time.perf_counter()
    fit,counts=make_fit(seed)
    rng=np.random.default_rng(np.random.SeedSequence([
        27601119,seed,116]))
    draws=rng.integers(0,len(fit[-1]),size=(512,16))
    models=new_models(seed)
    trained={arm:train_arm(models[arm],arm,fit,draws) for arm in TRAINABLE}
    if set(trained)!=set(TRAINABLE) or any(d["updates"]!=512 for d in trained.values()):
        raise PermissionError("R116_ORIGINAL_HELDOUT_NOT_ADMITTED")
    layers,weight_sha=r115.quantize_native(models[NATIVE])
    cold=r115.integer_cold_restore(layers)
    metrics,actions,flips,paired_both=evaluate_with_pairs(models,layers,seed)
    # This is paired query correctness on a single heldout scoring pass;
    # it is not a memory-retention or continual-learning metric.
    checks=frozen_checks(metrics,flips,paired_both,cold,trained)
    return {"seed":seed,"status":(
        freeze()["evaluation"]["pass_status"] if checks["all_unit_only_checks_pass"]
        else freeze()["evaluation"]["fail_status"]),
        "native_quantized_weight_sha256":weight_sha,
        "native_cold_integer_mismatch_count":cold,
        "FIT_diagnostic_sign_balance":counts,
        "trained":trained,"evaluation":metrics,
        "paired_opposite_sign_next_action_flips":flips,
        "paired_both_correct":paired_both,
        "frozen_checks":checks,
        "seconds":time.perf_counter()-start,
        "protected_scientific_first_use":False,"scientific_pass":False}


def run_all():
    freeze()
    if (ROOT/"experiments/r116_original_protected_firstuse_authorization.json").exists():
        raise PermissionError("R116_PROTECTED_ORIGINAL_FIRSTUSE_NOT_AUTHORIZED")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,check=True,
                       capture_output=True,text=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and expected!=git:
        raise PermissionError("R116_SCORED_EXACT_SHA_DRIFT")
    result={str(seed):run_seed(seed) for seed in SEEDS}
    ok=all(s["frozen_checks"]["all_unit_only_checks_pass"]
           for s in result.values())
    return {"schema":"ttt-hybrid.r116.unit-only-causal-feedback-sign-conditioned.v1",
            "source_sha":git,"preimplementation_freeze_blob":custody.FREEZE_BLOB,
            "status":(freeze()["evaluation"]["pass_status"] if ok
                      else freeze()["evaluation"]["fail_status"]),
            "by_seed":result,"original_scientific_first_use":False,
            "scientific_pass":False,"GPU_firstuse":False,
            "promotion":False,"deployment":"HOLD"}


def main():
    r=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R116_ORIGINAL_UNIT_ONLY_RECEIPT_REPLAY_DENIED")
    RECEIPT.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R116_UNIT_ONLY "+json.dumps({
        "sha":r["source_sha"],"status":r["status"],
        "seeds":{k:{"new_correct":{a:d["correct"] for a,d in v["evaluation"].items()},
                    "criteria":v["frozen_checks"]["checks"],
                    "native_weight_sha":v["native_quantized_weight_sha256"],
                    "opposite_sign_pair_action_flips":v["paired_opposite_sign_next_action_flips"],
                    "paired_both_correct":v["paired_both_correct"],
                    "cold":v["native_cold_integer_mismatch_count"],
                    "seconds":round(v["seconds"],3)}
                 for k,v in r["by_seed"].items()},
        "protected_scientific_first_use":False},sort_keys=True))


if __name__=="__main__":
    main()
