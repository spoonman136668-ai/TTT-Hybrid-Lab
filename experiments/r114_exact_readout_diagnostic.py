"""R114 FIT/VALID-only integer Bayesian readout portability diagnosis.

No R113 UNIT_UNSEEN generated, touched, scored or replayed. Exact integer
Bayes reference comparison is over a previously disclosed 36-rule hypothesis
class, not independent causal discovery, GPU science, or model admission.
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

from experiments import r113_structured_active_probe_diagnostic as r113
from experiments import r114_source_custody as custody

ROOT=Path(__file__).resolve().parents[1]
RECEIPT=ROOT/"local_runs/r114_exact_integer_bayes_fit_valid_only.json"
SEEDS=custody.SEEDS
ARMS=r113.ARMS


def exact_weights_from_public(history):
    """36 exact positive integer unnormalized posterior weights.

    All hypotheses start with equal priors; feedback signs are the only
    evidence. Equal denominator 20**len(history) cancels for every template.
    Python integer products are platform-independent even at exact ties.
    """
    custody.audit_freeze()
    if len(history)>r113.STEPS:
        raise ValueError("R114_TOO_MANY_FACTUAL_SUPPORT_OBSERVATIONS")
    counts=[]
    for pattern,sign in history:
        if type(pattern) is not int or not 0<=pattern<8 or sign not in (-1,1):
            raise ValueError("R114_UNCHOSEN_OR_NONPUBLIC_FACTUAL_HISTORY")
    for pair in r113.ALL:
        correct=sum(int(r113.truth_bit(pair,p)==int(sign==1))
                    for p,sign in history)
        counts.append(17**correct*3**(len(history)-correct))
    if len(counts)!=36 or not all(type(v) is int and v>0 for v in counts):
        raise RuntimeError("R114_EXACT_POSTERIOR_INTEGER_CUSTODY_FAILURE")
    return counts


def integer_prediction(query,history):
    if type(query) is not int or not 0<=query<8:
        raise ValueError("R114_QUERY_PATTERN_OUTSIDE_PUBLIC_THREE_FACTOR_SPACE")
    w=exact_weights_from_public(history)
    positive=sum(w[k] for k,pair in enumerate(r113.ALL)
                 if r113.truth_bit(pair,query)==1)
    total=sum(w)
    if not 0<=positive<=total:
        raise RuntimeError("R114_INVALID_INTEGER_POSTERIOR")
    return (1 if 2*positive>=total else -1),int(2*positive==total)


def _hash_octets(values):
    vals=[int(x) for x in values]
    if not vals or any(x<0 or x>255 for x in vals):
        raise RuntimeError("R114_NONCANONICAL_EVENT_HASH_INPUT")
    return hashlib.sha256(bytes(vals)).hexdigest()


def _actor(seed,arm,models,layers):
    custody.allow("UNIT_VALID")
    if seed not in SEEDS or arm not in ARMS:
        raise PermissionError("R114_UNREGISTERED_FIT_VALID_ACTOR")
    actions=[];effects=[];correctness=[];signs=[];before=after=new=0
    float_correct=0;float_disagreements=ties=positive_count=feedback_effect=0
    for index in range(88):
        old=r113.PhysicalStructuredWorld("UNIT_VALID",seed,index,0)
        late=r113.PhysicalStructuredWorld("UNIT_VALID",seed,index,1)
        for phase,world in ((0,old),(1,late)):
            for step in range(6):
                offers=world.offers()
                chosen=r113.choose(arm,models,layers,offers,world.history,
                                   phase,seed,index,step)
                blinded=[(p,1) for p,_ in world.history]
                blinded_choice=r113.choose(arm,models,layers,offers,blinded,
                                          phase,seed,index,step)
                feedback_effect+=int(chosen!=blinded_choice)
                if chosen not in offers:
                    raise RuntimeError("R114_UNCHOSEN_PUBLIC_ACTION_DENIED")
                effect=world.intervene(chosen)
                actions.append(chosen);effects.append(int(effect>0))
                positive_count+=int(effect>0)
            query,truth=world.sealed_query()
            predicted,is_tie=integer_prediction(query,world.history)
            old_float=r113.fixed_readout(query,world.history)
            correct=int(predicted==truth)
            ties+=is_tie
            float_disagreements+=int(old_float!=predicted)
            correctness.append(correct);signs.append(int(predicted>0))
            if phase:
                new+=correct
                float_correct+=int(old_float==truth)
            else:
                before+=correct
                after+=correct   # Cold old history retained, not continual learning.
    if len(actions)!=88*12 or len(correctness)!=88*2 or len(effects)!=len(actions):
        raise RuntimeError("R114_CHANGED_FACTUAL_ACTION_BUDGET")
    return {"old_before_correct":before,"old_after_correct":after,
            "new_after_correct":new,"new_after_accuracy":new/88,
            "float_original_new_correct":float_correct,
            "float_vs_integer_readout_disagreements":float_disagreements,
            "exact_half_posterior_tie_count":ties,
            "total_chosen_actions":len(actions),
            "positive_observed_effect_count":positive_count,
            "feedback_sensitivity_choices":feedback_effect,
            "chosen_actions_sha256":_hash_octets(actions),
            "observed_effects_sha256":_hash_octets(effects),
            "integer_prediction_signs_sha256":_hash_octets(signs),
            "integer_correctness_sha256":_hash_octets(correctness)}


def run_seed(seed):
    custody.audit_freeze()
    if seed not in SEEDS:
        raise PermissionError("R114_UNREGISTERED_NEW_SCIENCE_SOURCE_DENIED")
    torch.set_num_threads(1)
    start=time.perf_counter()
    train=r113.fit_examples(seed,None)  # R113 FIT ONLY, not R113 unseen.
    rng=np.random.default_rng(np.random.SeedSequence([
        r113.SALTS["UNIT_BATCH"],seed,113]))
    minibatches=rng.integers(0,len(train[-1]),size=(512,16))
    models=r113.new_models(seed)
    trained={a:r113.train_arm(models[a],a,train,minibatches,None)
             for a in r113.TRAINABLE}
    if (set(trained)!=set(r113.TRAINABLE) or
        any(d["updates"]!=512 for d in trained.values())):
        raise PermissionError("R114_VALIDATION_BEFORE_FULL_FIT_TRAINING")
    layers,weight_sha=r113.quantize_native(models[r113.NATIVE])
    cold=r113.isolated_integer_cold_reload(layers)
    if cold:
        raise RuntimeError("R114_SCRATCH_INTEGER_NATIVE_RESTORE_MISMATCH")
    scores={arm:_actor(seed,arm,models,layers) for arm in ARMS}
    limit=custody.audit_freeze()["evaluation"]["local_quality_checks"][
        "integer_vs_float_VALID_new_accuracy_drop_max"]
    drop=max((s["float_original_new_correct"]-s["new_after_correct"])/88
             for s in scores.values())
    local=(drop<=limit and all(s["total_chosen_actions"]==1056
                              for s in scores.values()))
    return {"seed":seed,
            "all_three_trainable_arms_512_FIT_steps":True,
            "trained":trained,"native_quantized_weight_sha256":weight_sha,
            "cold_integer_native_reload_differences":cold,
            "by_arm":scores,"worst_integer_new_accuracy_drop_from_float":drop,
            "local_quality_gate_pass":local,
            "wall_seconds":time.perf_counter()-start,
            "R113_original_UNSEEN_replayed":False,
            "scientific_pass":False}


def run_all():
    freeze=custody.audit_freeze()
    if (ROOT/"experiments/r114_original_protected_firstuse_authorization.json").exists():
        raise PermissionError("R114_PROTECTED_ORIGINAL_SCIENCE_NOT_ALLOWED")
    sha=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,check=True,timeout=10).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and sha!=expected:
        raise PermissionError("R114_EXACT_SOURCE_CANDIDATE_SHA_DRIFT")
    results={str(seed):run_seed(seed) for seed in SEEDS}
    ok=all(x["local_quality_gate_pass"] for x in results.values())
    return {"schema":"ttt-hybrid.r114.fit-valid-exact-integer-bayes-posterior.v1",
            "source_exact_sha":sha,"freeze_blob":custody.FREEZE_BLOB,
            "status":("R114_LOCAL_FIT_VALID_ENGINEERING_PASS_CROSS_OS_UNVERIFIED"
                      if ok else freeze["evaluation"]["fail_status"]),
            "all_three_seed_local_quality_pass":ok,
            "cross_OS_exact_parity_requires_original_artifact_comparison":True,
            "by_seed":results,"R113_spent_original_UNSEEN_replayed":False,
            "scientific_pass":False,"GPU_firstuse":False,
            "merge_or_promotion":False,"deployment":"HOLD"}


def main():
    r=run_all()
    RECEIPT.parent.mkdir(parents=True,exist_ok=True)
    if RECEIPT.exists():
        raise FileExistsError("R114_ORIGINAL_RECEIPT_ALREADY_EXISTS_REPLAY_DENIED")
    RECEIPT.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n",encoding="utf-8")
    print("R114_FIT_VALID_INTEGER_READOUT "+json.dumps({
        "sha":r["source_exact_sha"],"status":r["status"],
        "seeds":{seed:{"quality_pass":v["local_quality_gate_pass"],
                        "native_weight_sha":v["native_quantized_weight_sha256"],
                        "native_exact_new_correct":v["by_arm"][r113.NATIVE]["new_after_correct"],
                        "native_float_new_correct":v["by_arm"][r113.NATIVE]["float_original_new_correct"],
                        "native_posterior_ties":v["by_arm"][r113.NATIVE]["exact_half_posterior_tie_count"],
                        "float_readout_disagreements":v["by_arm"][r113.NATIVE]["float_vs_integer_readout_disagreements"],
                        "cold_mismatches":v["cold_integer_native_reload_differences"]}
                 for seed,v in r["by_seed"].items()},
        "protected_science":False,"R113_UNSEEN_replay":False},sort_keys=True))


if __name__=="__main__":
    main()
