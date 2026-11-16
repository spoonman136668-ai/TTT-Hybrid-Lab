"""R104: new UNIT_ONLY 256-rule public factual causal identifiability.

The eight heldout truth functions are *evaluator-only*. The diagnostic
learner uses ALL 256 possible 3-input Boolean operators with uniform
prior, 16 genuinely chosen-action physical rewards, and noisy PUBLIC
role/operand observations. No R101/R102/R103 original scored sample,
trained checkpoint, or protected FIT/DEV source is ever accessed.

Engineering diagnostic results DO NOT establish model learning, broad
intelligence, protected scientific acceptance or deployment readiness.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r88_causal_event_access as r88

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r104_operator_identifiability_freeze.json"
FREEZE_BLOB="8e2a7d4bc1a4e0ef32ba79859464c41a5bf270e8"
SCHEMA="ttt-hybrid.r104.operator-identifiability.unit-only-prospective-freeze.v1"
UNIT_SALT=26349991
SEEDS=(1004011,1004023,1004037,1004049)
MASKS=(0x1b,0x66,0x2d,0x55,0x8e,0xa3,0xb4,0xc9)
SUPPORT=16
N_PER_SEED=128
HYPOTHESES=np.arange(256,dtype=np.uint16)
RECEIPT=ROOT/"local_runs/r104_unit_only_identifiability_audit.json"


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R104_PREIMPLEMENTATION_FREEZE_BLOB_DRIFT")
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FROZEN_BEFORE_R104_IMPLEMENTATION_OR_OBSERVATION"
        or f["source"]["UNIT_ONLY"] is not True
        or f["source"]["unit_only_salt"]!=UNIT_SALT
        or tuple(f["source"]["seeds"])!=SEEDS
        or f["source"]["units_per_seed"]!=N_PER_SEED
        or tuple(int(x,16) for x in
                 f["source"]["hidden_evaluator_only_truth_masks_hex"])!=MASKS
        or f["source"]["supports"]!=SUPPORT
        or f["source"]["max_unit_only_probe_wall_seconds_per_seed"]!=180
        or f["authority"]["protected_original_scientific_first_use"] is not False
        or any(bool(x) for k,x in f["authority"].items()
               if k!="unit_only_CPU_software_engineering")):
        raise PermissionError("R104_FROZEN_SOURCE_OR_AUTHORITY_DRIFT")
    return f


def _deadline(deadline):
    if deadline is not None and time.perf_counter()>=deadline:
        raise TimeoutError("R104_FROZEN_UNIT_ONLY_180_SECONDS_PER_SEED_EXCEEDED")


def all256_pairwise():
    """Noiseless combinatorial bound; NOT real physics or model accuracy."""
    f=frozen()
    distances={str(d):0 for d in range(1,9)}
    for a in range(256):
        for b in range(a+1,256):
            distances[str((a^b).bit_count())]+=1
    if sum(distances.values())!=256*255//2:
        raise RuntimeError("R104_ALL_256_OPERATOR_PAIR_COVERAGE_INVALID")
    probabilities={
        str(d):{"six":((8-d)/8)**6,"sixteen":((8-d)/8)**16}
        for d in range(1,9)
    }
    if any(probabilities[k]["sixteen"]>probabilities[k]["six"]
           for k in probabilities):
        raise RuntimeError("R104_COMBINATORIAL_CEILING_INVALID")
    nearest={f"{m:02x}":min((m^other).bit_count()
                           for other in MASKS if other!=m)
             for m in MASKS}
    if any(x<4 for x in nearest.values()):
        raise RuntimeError("R104_FROZEN_HELDOUT_OPERATOR_DISTINCTNESS_INVALID")
    return {"all_candidate_operator_count":len(HYPOTHESES),
            "all_unordered_pairs":256*255//2,
            "pair_counts_by_hamming_distance":distances,
            "chance_no_distinguishing_uniform_pattern":probabilities,
            "eight_selected_evaluator_masks_min_pair_distance":nearest,
            "original_scored_cohorts_opened":False,
            "scientific_pass":False}


def _private_truth(factors,roles,mask):
    """Hidden truth solely for world physics and final evaluation."""
    if not (isinstance(mask,(int,np.integer)) and 0<=mask<256
            and len(roles)==3 and len(set(roles))==3):
        raise ValueError("R104_PRIVATE_TRUTH_SHAPE_OR_MASK_INVALID")
    pattern=0
    for i in roles:
        if not 0<=int(i)<4:
            raise ValueError("R104_PRIVATE_FACTORS_OR_ROLE_INVALID")
        pattern=(pattern<<1)|int(factors[int(i)]>0)
    return 1 if (int(mask)>>pattern)&1 else -1


def _unit_world(seed,episode,event,mask):
    """NEW independent R104 salt. Pure old simulator, NOT any old source."""
    if (seed not in SEEDS or not isinstance(episode,int)
        or not 0<=episode<N_PER_SEED
        or not isinstance(event,int) or not 0<=event<=SUPPORT
        or mask not in MASKS):
        raise PermissionError("R104_UNIT_ONLY_GENERATOR_OUT_OF_SCOPE")
    rng=np.random.default_rng(np.random.SeedSequence([
        UNIT_SALT,seed,episode,event]))
    # Every hidden mask has exactly 8 EARLY and 8 COLD episodes.
    pivot=(episode//len(MASKS))%2
    physical_index=(2*(episode%(r85.N//2))+pivot)%r85.N
    w=r85._make_synthetic_world(physical_index,rng)
    roles=tuple(int(i) for i in rng.permutation(4)[:3])
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    w["factors_a"]=factors
    w["r104_private_roles"]=roles
    w["law_a"]=_private_truth(factors,roles,mask)
    return w


def _public_view(world,action):
    """Only physically observed pre-action slots plus actual chosen reward."""
    prefix,reward=r86._view(world,action)
    public=r88._public_slots(prefix).copy()
    tags=np.zeros(4,np.float32)
    for position,role in enumerate(world["r104_private_roles"],start=1):
        tags[role]=position/4.
    public[:4,8]=tags
    if public.shape!=(5,13) or not np.isfinite(public).all():
        raise RuntimeError("R104_PUBLIC_OBSERVATION_INVALID")
    return public,float(reward)


def unit_episode(seed,index,*,deadline=None):
    """Simulated episode with strict PUBLIC/PRIVATE object separation."""
    frozen()
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<N_PER_SEED:
        raise PermissionError("R104_UNIT_EPISODE_OR_SEED_NOT_FROZEN")
    mask=MASKS[index%len(MASKS)]
    public=[];actual=[];actions=[]
    for j in range(SUPPORT):
        _deadline(deadline)
        action=r85.CALIBRATE if j%2==0 else r85.EXPLOIT_POS
        world=_unit_world(seed,index,j,mask)
        x,reward=_public_view(world,action)
        public.append(x);actual.append(reward);actions.append(action)
    _deadline(deadline)
    query_world=_unit_world(seed,index,SUPPORT,mask)
    q0,r0=_public_view(query_world,r85.CALIBRATE)
    q2,r2=_public_view(query_world,r85.EXPLOIT_POS)
    if (not np.array_equal(q0[:4],q2[:4])
        or not np.array_equal(q0[-1,4:],q2[-1,4:])):
        raise RuntimeError("R104_UNCHOSEN_REWARD_LEAKED_IN_PUBLIC_QUERY")
    optimal=r85.CALIBRATE if r0>r2 else r85.EXPLOIT_POS
    true_optimal=(r85.CALIBRATE if query_world["law_a"]>0
                  else r85.EXPLOIT_POS)
    if optimal!=true_optimal:
        raise RuntimeError("R104_PRIVATE_ACTION_LAW_OR_REWARD_INVALID")
    return {"public":{
        "support_slots":np.stack(public).astype(np.float32),
        "support_actions":np.asarray(actions,dtype=np.int8),
        "actual_factual_rewards":np.asarray(actual,dtype=np.float32),
        "query_slots":q0,
    },"evaluator_only":{
        "hidden_mask":mask,
        "query_law":int(query_world["law_a"]),
        "query_return_calibrate":r0,
        "query_return_exploit":r2,
        "optimal_action":true_optimal,
        "pivot":"EARLY" if (index//8)%2==0 else "COLD",
    }}


def _patterns_from_public(slots,missing_roles=False):
    """No world IDs, formula labels, noise innovations or hidden truth."""
    if slots.ndim!=3 or slots.shape[1:]!=(5,13):
        raise ValueError("R104_PUBLIC_ROLE_AND_FACTOR_SHAPES_INVALID")
    tags=np.rint(slots[:,:4,8].astype(np.float64)*4).astype(np.int16)
    output=np.empty(len(slots),dtype=np.uint8)
    for j in range(len(slots)):
        if missing_roles:
            indices=(0,1,2)
        else:
            indices=[]
            for mark in (1,2,3):
                hits=np.flatnonzero(tags[j]==mark)
                if len(hits)!=1:
                    raise RuntimeError("R104_PUBLIC_ROLE_BINDING_MISSING")
                indices.append(int(hits[0]))
        pattern=0
        for physical_role in indices:
            pattern=(pattern<<1)|int(slots[j,int(physical_role),10]>=0)
        output[j]=pattern
    return output


def _shuffle_action_strata(actual,actions):
    """Deterministic non-self reward derangement within physical action."""
    if actual.shape!=(SUPPORT,) or actions.shape!=(SUPPORT,):
        raise ValueError("R104_PUBLIC_SUPPORT_OUTCOME_SHAPE_INVALID")
    outcome=np.empty(SUPPORT,np.float32)
    for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
        group=np.flatnonzero(actions==action)
        if len(group)!=8:
            raise RuntimeError("R104_UNBALANCED_PUBLIC_INTERVENTIONS")
        outcome[group]=actual[np.roll(group,1)]
    return outcome


def posterior_decision(public,*,supports=SUPPORT,mode="factual"):
    """Inferred operator may be ANY of 256; no private shortlist supplied.

    Receives PUBLIC dict ONLY. It cannot access evaluator_only or hidden mask.
    """
    frozen()
    if supports not in (6,16) or mode not in (
        "factual","no_feedback","shuffled","role_missing"):
        raise ValueError("R104_NONFROZEN_CONTROL_OR_SUPPORT_COUNT")
    if set(public)!={"support_slots","support_actions",
                    "actual_factual_rewards","query_slots"}:
        raise PermissionError("R104_PRIVATE_SOURCE_FIELD_VISIBLE_TO_INFERENCE")
    x=public["support_slots"];actions=public["support_actions"]
    rewards=public["actual_factual_rewards"]
    query=public["query_slots"]
    if (x.shape!=(SUPPORT,5,13) or actions.shape!=(SUPPORT,)
        or rewards.shape!=(SUPPORT,) or query.shape!=(5,13)
        or not np.isfinite(x).all() or not np.isfinite(rewards).all()):
        raise RuntimeError("R104_FROZEN_PUBLIC_INPUT_SHAPE_INVALID")
    p=_patterns_from_public(x[:supports],mode=="role_missing")
    q=int(_patterns_from_public(query[None],mode=="role_missing")[0])
    likelihood=np.zeros(256,np.float64)
    if mode!="no_feedback":
        labels=(_shuffle_action_strata(rewards,actions) if mode=="shuffled"
                else rewards)[:supports]>0
        guesses=((HYPOTHESES[:,None] >> p[None,:])&1)>0
        likelihood=np.where(guesses==labels[None,:],
                            math.log(.85),math.log(.15)).sum(axis=1)
    likelihood-=np.max(likelihood)
    w=np.exp(likelihood)
    w/=w.sum()
    predicted_law=float(np.dot(w,2*((HYPOTHESES>>q)&1).astype(np.float64)-1))
    if abs(predicted_law)<=1e-12:
        chosen=(r85.CALIBRATE if q%2==0 else r85.EXPLOIT_POS)
    else:
        chosen=(r85.CALIBRATE if predicted_law>0 else r85.EXPLOIT_POS)
    return int(chosen)


def run_seed(seed,*,deadline=None):
    """Independent new unit-only cohort; DOES NOT constitute science PASS."""
    f=frozen()
    if seed not in SEEDS:
        raise PermissionError("R104_UNIT_ONLY_SEED_NOT_REGISTERED")
    counts={k:0 for k in ("factual16","first6","shuffled","no_feedback","role_missing")}
    represented={(mask,pivot):0 for mask in MASKS for pivot in ("EARLY","COLD")}
    for index in range(N_PER_SEED):
        _deadline(deadline)
        episode=unit_episode(seed,index,deadline=deadline)
        p,e=episode["public"],episode["evaluator_only"]
        represented[(e["hidden_mask"],e["pivot"])]+=1
        opt=e["optimal_action"]
        for name,method,n in (
            ("factual16","factual",16),("first6","factual",6),
            ("shuffled","shuffled",16),("no_feedback","no_feedback",16),
            ("role_missing","role_missing",16)):
            counts[name]+=int(posterior_decision(p,supports=n,mode=method)==opt)
    if any(count!=8 for count in represented.values()):
        raise RuntimeError("R104_PRIVATE_MASK_VS_PUBLIC_PIVOT_LEAK")
    accuracy={k:v/N_PER_SEED for k,v in counts.items()}
    gains={
        "factual_minus_no_feedback_accuracy_min":
            accuracy["factual16"]-accuracy["no_feedback"],
        "factual_minus_action_stratum_shuffled_accuracy_min":
            accuracy["factual16"]-accuracy["shuffled"],
        "factual_minus_role_missing_accuracy_min":
            accuracy["factual16"]-accuracy["role_missing"],
        "factual_16_minus_first6_accuracy_min":
            accuracy["factual16"]-accuracy["first6"],
        "mean_16_factual_accuracy_min":accuracy["factual16"],
        "prior_or_no_feedback_chance_accuracy_max":accuracy["no_feedback"]
    }
    thresholds=f["prespecified_unit_only_checks"]
    checks={
        key:(gains[key] <= threshold if key.endswith("_max")
             else gains[key]>=threshold)
        for key,threshold in thresholds.items() if isinstance(threshold,(float,int))
    }
    return {"seed":seed,"episodes":N_PER_SEED,
            "counts_correct":counts,"accuracy":accuracy,"contrasts":gains,
            "prespecified_checks":checks,
            "passed_all_unit_only_checks":all(checks.values()),
            "status":("R104_UNIT_ONLY_INFORMATION_SIGNAL" if all(checks.values())
                      else "R104_NO_UNIT_ONLY_IDENTIFIABILITY_SIGNAL"),
            "protected_scored_data_accessed":False,
            "scientific_pass":False}


def run_all():
    """No protected first-use, GPU, scheduled work or trained model."""
    f=frozen()
    prior=ROOT/"experiments/r104_cpu_firstuse_authorization.json"
    if prior.exists():
        raise PermissionError("R104_PROTECTED_SOURCE_AUTHORIZATION_IN_UNIT_DIAGNOSTIC")
    git=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                       capture_output=True,text=True,timeout=10,check=True).stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected and git!=expected:
        raise PermissionError("R104_UNIT_ONLY_EXACT_SOURCE_SHA_DRIFT")
    audit=all256_pairwise()
    started=time.perf_counter()
    results={}
    for seed in SEEDS:
        deadline=time.perf_counter()+f["source"]["max_unit_only_probe_wall_seconds_per_seed"]
        results[str(seed)]=run_seed(seed,deadline=deadline)
    overall=all(x["passed_all_unit_only_checks"] for x in results.values())
    return {"schema":"ttt-hybrid.r104.unit-only-identifiability-result.v1",
            "status":("R104_UNIT_ONLY_ALL_SEEDS_DIAGNOSTIC_SIGNAL" if overall
                      else "R104_UNIT_ONLY_NEGATIVE_OR_MIXED_DIAGNOSTIC"),
            "source_sha":git,"frozen_blob":FREEZE_BLOB,"source_type":"UNIT_ONLY",
            "original_scored_source_used":False,
            "R101_R102_R103_original_scored_replay":False,
            "no_prior_trained_weights":True,
            "all_operator_pairwise_analytic":audit,
            "per_seed":results,
            "all_four_UNIT_ONLY_seeds_pass":overall,
            "whole_unit_only_wall_seconds":time.perf_counter()-started,
            "scientific_pass":False,
            "deployment_status":"HOLD"}


def main():
    receipt=run_all()
    RECEIPT.parent.mkdir(exist_ok=True,parents=True)
    if RECEIPT.exists():
        raise FileExistsError("R104_DIAGNOSTIC_RECEIPT_ALREADY_EXISTS")
    RECEIPT.write_text(json.dumps(receipt,sort_keys=True,indent=2)+"\n",
                       encoding="utf-8")
    print("R104_UNIT_ONLY_AUDIT "+json.dumps({
        "source_sha":receipt["source_sha"],
        "status":receipt["status"],
        "per_seed":{k:{"accuracy":v["accuracy"],"contrasts":v["contrasts"],
                        "checks":v["prespecified_checks"]} for k,v in
                    receipt["per_seed"].items()},
        "scientific_pass":False,
        "whole_unit_only_wall_seconds":receipt["whole_unit_only_wall_seconds"],
    },sort_keys=True))


if __name__=="__main__":
    main()
