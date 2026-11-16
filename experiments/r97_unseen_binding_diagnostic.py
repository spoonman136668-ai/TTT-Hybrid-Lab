"""R97: zero-shot transfer to unseen *public role binding* of a known mux operator.

FRESH TEST-only physics. No previously scored R85–R96 source, weights, target
teacher or unchosen FIT action is opened. Source generated after exact original
hosted G4 dual-platform qualification; new DEV/transfer/OOD worlds only AFTER
all nine model arms complete full factual-return training and freeze.
"""
from __future__ import annotations
import copy
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r96_role_mux_diagnostic as r96

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r97_unseen_binding_freeze.json"
FREEZE_BLOB="a73fdfbbad3a7659cc00ba98f37a01695a59154e"
SCHEMA="ttt-hybrid.r97-unseen-public-role-binding.testonly.v1"
PARENT="93b02e45a5d37299fa4edd6fb50d619ab129fa73"
BRANCH="research/r97-unseen-public-role-binding-20261116"
SEEDS=(1001711,1001723)
SALT={"FIT":25380111,"VALID":25380211,"DEV_SEEN":25380311,
      "DEV_HELDOUT":25380411,"OOD_SHIFT":25380511,
      "ROLE_REASSIGN":25380611}
COUNT={"FIT":2048,"VALID":256,"DEV_SEEN":256,"DEV_HELDOUT":256,
       "OOD_SHIFT":256,"ROLE_REASSIGN":256}
TAG_PERMUTE_SALT=25380711
BATCH_SALT=25380811
BOOTSTRAP_SALT=25380911
UNIT_SALT=25389991
TRAIN_BINDINGS=((0,1,2),(1,2,3),(2,3,0))
HELDOUT_BINDING=(3,0,1)
REASSIGN_BINDING=(0,1,2)
ARMS=("NATIVE32_PUBLIC_SLOTS5","NATIVE32_FLAT_TAGGED_HEAD",
      "NATIVE32_SET_TAGGED_HEAD","MLP64_TAGGED_RAW",
      "MLP_TAGGED_SET","MLP64_ROLE_ONLY_TAG_BLIND",
      "MLP_TAGGED_SET_PERMUTED_NEGATIVE","GRU64_PUBLIC_SLOTS5",
      "MLP64_PUBLIC_CANONICAL_UPPER")
(NATIVE_RAW,NATIVE_FLAT,NATIVE_SET,MLP_RAW,MLP_SET,
 BLIND,NEGATIVE,GRU,UPPER)=ARMS
STEPS=4096
BATCH=16
MAX_SECONDS=260


def frozen()->dict:
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R97_PROSPECTIVE_FROZEN_GIT_BLOB_DRIFT")
    r96.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,t,e,x=f["parent"],f["source"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="FROZEN_BEFORE_R97_IMPLEMENTATION_TESTS_OR_SOURCE_FIRST_USE"
        or p["original_r96_final_sha"]!=PARENT
        or p["r96_freeze_git_blob"]!=r96.FREEZE_BLOB
        or p["r96_spent_sha"]!=r96.ORIGINAL_R96_SPENT_SHA
        or p["r96_spent_run"]!=r96.ORIGINAL_R96_SPENT_RUN_ID
        or p["original_r96_science_status"]!="R96_ROLE_CONDITIONAL_MECHANISM_NO_GO"
        or p["previous_training_or_scored_data_or_weights_reused"]
        or s["seeds"]!=list(SEEDS)
        or s["salts"]!={**SALT,"TAG_PERMUTE":TAG_PERMUTE_SALT,
                         "BATCH":BATCH_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "UNIT_ONLY":UNIT_SALT}
        or s["counts"]!=COUNT
        or s["training_bindings"]!=[list(z) for z in TRAIN_BINDINGS]
        or s["unseen_test_binding"]!=list(HELDOUT_BINDING)
        or s["role_reassign_counterfactual_binding"]!=list(REASSIGN_BINDING)
        or s["previous_scored_or_reserved_sources"]
        or s["negative_role_disruption_implied_law_agreement_expected"]!=.625
        or s["negative_implied_law_audit_lower"]!=.50
        or s["negative_implied_law_audit_upper"]!=.75
        or s["negative_role_disruption_not_causal_information_independence"] is not True
        or f["arms"]!=list(ARMS)
        or t["all_arm_steps"]!=STEPS or t["batch"]!=BATCH
        or t["whole_process_CPU_ceiling_seconds_per_seed"]!=MAX_SECONDS
        or e["bootstrap_samples"]!=2000
        or list(e["mechanism_checks"].values())!=[.75,.75,.15,.10,.30,.85,.50,.10,.20,.70,.85]
        or list(e["native_checks"].values())!=[.20,.50,.20,.30]
        or e["independent_scientific_pass"] is not False
        or x["new_workflow_file"]
        or not x["branch_scoped_original_two_seed_jobs"]
        or not f["authority"]["scoped_CPU_testonly_diagnostic_allowed"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="scoped_CPU_testonly_diagnostic_allowed")):
        raise PermissionError("R97_PROSPECTIVE_PROTOCOL_OR_AUTHORITY_DRIFT")
    salts=set(s["salts"].values())
    if len(salts)!=len(s["salts"]) or salts & set(r96.frozen()["source"]["salts"].values()):
        raise RuntimeError("R97_PRIOR_SPENT_OR_DUPLICATE_SALT")
    return f


def binding(panel:str,index:int)->tuple[int,int,int]:
    if panel not in SALT or not isinstance(index,int) or index<0 or index>=COUNT[panel]:
        raise PermissionError("R97_UNREGISTERED_SOURCE_PANEL_OR_INDEX")
    return TRAIN_BINDINGS[index%len(TRAIN_BINDINGS)] if panel in (
        "FIT","VALID","DEV_SEEN") else HELDOUT_BINDING


def mux(factors:np.ndarray,role_binding:tuple[int,int,int])->int:
    if (factors.shape!=(4,) or not np.all(np.isin(factors,(-1,1)))
        or len(set(role_binding))!=3
        or any(not 0<=r<4 for r in role_binding)):
        raise ValueError("R97_PHYSICAL_MUX_BAD_FACTORS_OR_ROLES")
    selector,yes,no=role_binding
    return int(factors[yes] if factors[selector]>0 else factors[no])


def tags_for(role_binding:tuple[int,int,int])->np.ndarray:
    if len(set(role_binding))!=3 or any(not 0<=i<4 for i in role_binding):
        raise ValueError("R97_INVALID_PUBLIC_ROLE_TAG_BINDING")
    tags=np.zeros(4,dtype=np.int64)
    for tag,role in enumerate(role_binding,1):
        tags[role]=tag
    return tags


def _chosen(seed:int,index:int)->int:
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R97_UNFROZEN_FACTUAL_ACTION")
    return (r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0
            else r85.EXPLOIT_POS)


def balance_preflight():
    """Only design indexes, NO physics/world creation or new score readback."""
    out={}
    for seed in SEEDS:
        out[str(seed)]={}
        for panel,n in COUNT.items():
            cell={}
            for i in range(n):
                key=(i%2,_chosen(seed,i))
                cell[key]=cell.get(key,0)+1
            if len(cell)!=4 or set(cell.values())!={n//4}:
                raise RuntimeError("R97_CHOSEN_ACTION_VS_COLD_SOURCE_DESIGN_CONFOUNDED")
            out[str(seed)][panel]={"four_action_cold_cells":4,"count_per_cell":n//4}
    return out


def _admit(seed:int)->str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r97-unseen-binding"
        or os.getenv("R97_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R97_ORIGINAL_CLEAN_HOSTED_G4_FIRST_ATTEMPT_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    actual=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                          capture_output=True,text=True,check=False,timeout=10)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,text=True,check=False,timeout=10)
    if len(sha)!=40 or actual.returncode or actual.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R97_EXACT_SHA_OR_CLEAN_TREE_DENIED")
    return sha


def unit_world(index:int):
    if not isinstance(index,int) or not 0<=index<8:
        raise PermissionError("R97_ONLY_INDEPENDENT_UNIT_FIXTURE_SOURCE")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,index]))
    world=r85._make_synthetic_world(index,rng)
    roles=TRAIN_BINDINGS[index%3] if index<6 else HELDOUT_BINDING
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["law_a"]=mux(factors,roles)
    world["r97_binding"]=roles
    return world,_chosen(SEEDS[0],index)


def _world(panel:str,seed:int,index:int,sha:str):
    if panel not in SALT:
        raise PermissionError("R97_PREVIOUS_SPENT_OR_PROTECTED_PANEL")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R97_NONFROZEN_SEED_OR_WORLD_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r97-unseen-binding"
        or os.getenv("R97_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R97_NEW_SCORING_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    roles=binding(panel,index)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["law_a"]=mux(factors,roles)
    world["r97_binding"]=roles
    if panel=="OOD_SHIFT":
        world=r86._shift_public_world(world,index)
    return world,_chosen(seed,index)


def _public_view(world:dict,action:int):
    """Create independent *public physical observation* tags before compression.

    R85 rollout already generates only pre-action selected observation rows.
    The role tags represent the environment's publicly stated instruction,
    NOT hidden law signs, actual returns, oracle, or unchosen branch.
    """
    x,return_value=r85._rollout(world,action)
    roles=world["r97_binding"]
    tags=tags_for(roles)
    for t,(phase,role,marker) in world["probe"].items():
        if phase=="A" and t+1<=world["pivot"]:
            x[t+1,8]=float(tags[role])/3.0
    probes=sorted(t for t in world["probe"] if t<world["pivot"])
    public=r86._compressed(x,probes)
    return public,float(return_value)


def _features(slots:np.ndarray)->np.ndarray:
    if slots.shape!=(5,13) or slots.dtype!=np.float32:
        raise ValueError("R97_BAD_LEGALLY_OBSERVED_A_EVENT_ROWS")
    tags=np.rint(slots[:4,8]*3).astype(np.int64)
    if (set(tags)!={0,1,2,3} or np.max(np.abs(slots[:4,8]-tags/3))>1.e-5
        or not np.array_equal(np.rint(slots[:4,9]*4).astype(int),np.arange(1,5))
        or not np.all(slots[:4,5]==1) or not np.all(slots[:4,11]==1)):
        raise RuntimeError("R97_MISSING_OR_AMBIGUOUS_PUBLIC_ROLE_INSTRUCTIONS")
    out=np.zeros(12,np.float32)
    out[:4]=slots[:4,10]
    out[4:8]=slots[:4,8]
    out[8]=slots[-1,r85.CALIBRATE]
    out[9]=slots[-1,r85.EXPLOIT_POS]
    out[10]=slots[-1,12]
    if (out[8],out[9]) not in ((1.,0.),(0.,1.)):
        raise RuntimeError("R97_ILLEGAL_FACTUAL_ACTION_FLAGS")
    return out


def _canonical(raw:np.ndarray)->np.ndarray:
    if raw.ndim!=2 or raw.shape[1]!=12:
        raise ValueError("R97_CANONICAL_PUBLIC_INPUT_NOT_12D")
    ids=np.rint(raw[:,4:8]*3).astype(int)
    if not all(set(row)=={0,1,2,3} for row in ids):
        raise RuntimeError("R97_NOT_ALL_PUBLIC_ROLE_INSTRUCTIONS_PRESENT")
    out=np.zeros((len(raw),8),np.float32)
    for tag in range(4):
        idx=np.argmax(ids==tag,axis=1)
        out[:,tag]=raw[np.arange(len(raw)),idx]
    # Explicitly engineered selector/yes/no/distractor mapping is an UPPER,
    # not an independently discovered causal binding by the native model.
    out[:,4:7]=raw[:,8:11]
    return out


def _blind(raw:np.ndarray)->np.ndarray:
    out=raw.copy()
    out[:,4:8]=0.
    return out


def _permuted(raw:np.ndarray,seed:int,panel:str):
    if panel not in COUNT or seed not in SEEDS or raw.shape!=(COUNT[panel],12):
        raise PermissionError("R97_TAG_NEGATIVE_ONLY_NEW_FROZEN_SOURCE")
    out=raw.copy()
    for i in range(len(raw)):
        rng=np.random.default_rng(np.random.SeedSequence(
            [TAG_PERMUTE_SALT,seed,SALT[panel],i]))
        out[i,4:8]=raw[i,4:8][rng.permutation(4)]
    if not np.array_equal(out[:,:4],raw[:,:4]) or not np.array_equal(
            out[:,8:],raw[:,8:]):
        raise RuntimeError("R97_PUBLIC_TAG_ONLY_NEGATIVE_CHANGED_CUES_OR_ACTION")
    return out


def _private_implied_law(factors:np.ndarray,tags:np.ndarray)->int:
    ids=np.rint(tags*3).astype(int)
    if set(ids)!={0,1,2,3}:
        raise ValueError("R97_NEGATIVE_PUBLIC_TAG_SET_CHANGED")
    selector=int(np.flatnonzero(ids==1)[0])
    yes=int(np.flatnonzero(ids==2)[0])
    no=int(np.flatnonzero(ids==3)[0])
    return mux(factors,(selector,yes,no))


def _audit(law:np.ndarray,raw:np.ndarray,negative:np.ndarray,
           true_factors:np.ndarray,chosen:np.ndarray|None):
    if (law.shape!=(len(raw),) or raw.shape!=(len(raw),12)
        or negative.shape!=raw.shape or true_factors.shape!=(len(raw),4)):
        raise ValueError("R97_PRIVATE_EVALUATOR_AUDIT_DIMENSIONS")
    rates={}
    for timepoint in np.unique(raw[:,10]):
        ids=np.flatnonzero(raw[:,10]==timepoint)
        rate=float(np.mean(law[ids]>0))
        rates[str(float(timepoint))]=rate
        if not .38<=rate<=.62:
            raise RuntimeError("R97_PRIVATE_LAW_TIME_STRATUM_OUT_OF_RANGE")
    if chosen is not None:
        for action in (r85.CALIBRATE,r85.EXPLOIT_POS):
            rate=float(np.mean(law[chosen==action]>0))
            if not .38<=rate<=.62:
                raise RuntimeError("R97_PRIVATE_LAW_ACTION_STRATUM_OUT_OF_RANGE")
    predicted=np.asarray([_private_implied_law(true_factors[i],negative[i,4:8])
                          for i in range(len(raw))],np.int8)
    match=float(np.mean(predicted==law))
    # The preimplementation frozen combinatorial expectation is 0.625,
    # NOT 0.5. This arm disrupts role binding, not all causal information.
    if not .50<=match<=.75:
        raise RuntimeError("R97_ROLE_DISRUPTION_NEGATIVE_CONTROL_INVALID")
    return {"private_law_positive_by_public_pivot":rates,
            "tag_permuted_implied_law_agreement":match,
            "combinatorial_expected_implied_law_agreement":.625,
            "role_disruption_not_causal_information_independence":True,
            "valid":True}
