"""R100 — factorized PUBLIC causal-program prior versus factual-return-sign feedback.

R100 TEST-only CPU research. The MUX/PARITY/NEGATION algebra and public role
canonicalization are EXPLICITLY ENGINEERED structural prior. This is NOT
new rule discovery or a production native-model promotion. The auxiliary
feedback comes solely from the ONE actual selected action's factual return.
Never reopen R85–R99 scored source, saved weights, hidden law or oracle.
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
from experiments import r88_causal_event_access as r88
from experiments import r95_learned_relation_diagnostic as r95
from experiments import r98_native_program_diagnostic as r98
from experiments import r99_microfeedback_diagnostic as r99

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"experiments/r100_factored_state_freeze.json"
FREEZE_BLOB="9c9ffd09fadc95d9f94662560480d0ba6e4be54b"
SCHEMA="ttt-hybrid.r100-factored-native-causal-state.cpu-testonly.v1"
PARENT="772217b0dbf750e0749999fc560fec8f4b4bb7f1"
BRANCH="research/r100-factored-native-causal-state-20261116"
SEEDS=(1002311,1002323)
SALT={"FIT":25410111,"VALID":25410211,"DEV_SEEN":25410311,
      "DEV_COMPOSE":25410411,"OOD_SHIFT":25410511,
      "PROGRAM_SWITCH":25410611}
COUNT={"FIT":2048,"VALID":256,"DEV_SEEN":256,"DEV_COMPOSE":256,
       "OOD_SHIFT":256,"PROGRAM_SWITCH":256}
RETURN_SHUFFLE_SALT=25410711
BATCH_SALT=25410811
BOOTSTRAP_SALT=25410911
UNIT_SALT=25419991
TRAIN_PROGRAMS=((0,0),(0,1),(1,0))
COMPOSE_PROGRAM=(1,1)
TRAIN_BINDINGS=((0,1,2),(1,2,3),(2,3,0))
HELDOUT_BINDING=(3,0,1)
ARMS=("NATIVE32_ORIGINAL","NATIVE32_UNSTRUCTURED_STEM",
      "NATIVE32_FACTORED_NO_AUX","NATIVE32_FACTORED_FACTUAL_SIGN_AUX",
      "NATIVE32_FACTORED_SHUFFLED_SIGN_AUX",
      "NATIVE32_FACTORED_OPCODE_MASK",
      "NATIVE32_FACTORED_EXTERNAL_VALUE",
      "MLP64_RAW_PUBLIC","ROLE_SET_PUBLIC","GRU64_PUBLIC_EVENTS",
      "MLP64_ENGINEERED_PUBLIC_RULE_UPPER")
(NATIVE_RAW,UNSTRUCTURED,NOAUX,AUX,SHUFFLED,OPMASK,EXTERNAL,
 MLP,SET,GRU,UPPER)=ARMS
FACTORED_INTERNAL_ARMS=(NOAUX,AUX,SHUFFLED,OPMASK)
STEPS=3072
BATCH=16
LAMBDA=.15
MAX_SECONDS=300


def frozen():
    if r85.git_blob(FREEZE.read_bytes())!=FREEZE_BLOB:
        raise RuntimeError("R100_PREIMPLEMENTATION_FROZEN_GIT_BLOB_DRIFT")
    r99.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,t,e,x=f["parent"],f["source"],f["training"],f["evaluation"],f["execution"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="PROSPECTIVE_FREEZE_BEFORE_R100_CODE_TESTS_OR_SOURCE_FIRST_USE"
        or p["qualified_r99_sealed_SHA"]!=PARENT
        or p["r99_freeze_blob"]!=r99.FREEZE_BLOB
        or p["r99_spent_SHA"]!=r99.ORIGINAL_R99_SPENT_SHA
        or p["r99_spent_run"]!=r99.ORIGINAL_R99_SPENT_RUN_ID
        or p["r99_science_status"]!="R99_INTERNAL_MICROFEEDBACK_NO_GO"
        or p["prior_scored_sources_or_weights_reused"]
        or s["seeds"]!=list(SEEDS)
        or s["salts"]!={**SALT,"RETURN_SHUFFLE":RETURN_SHUFFLE_SALT,
                         "BATCH":BATCH_SALT,"BOOTSTRAP":BOOTSTRAP_SALT,
                         "UNIT_ONLY":UNIT_SALT}
        or s["counts"]!=COUNT
        or s["train_programs"]!=[list(v) for v in TRAIN_PROGRAMS]
        or s["heldout_program"]!=list(COMPOSE_PROGRAM)
        or s["train_role_bindings"]!=[list(v) for v in TRAIN_BINDINGS]
        or s["heldout_roles"]!=list(HELDOUT_BINDING)
        or s["older_scored_sources_reused"]
        or f["arms"]!=list(ARMS)
        or t["steps_each_arm"]!=STEPS
        or t["batch"]!=BATCH
        or t["outcome_aux_weight"]!=LAMBDA
        or t["whole_CPU_budget_seconds_per_seed"]!=MAX_SECONDS
        or e["independent_seeds"]!=2 or e["bootstrap_draws"]!=2000
        or list(e["structural_checks"].values())!=[.5,.4,.75,.5,.8,.4,.1,.2,.7,.8]
        or list(e["feedback_checks"].values())!=[.2,.2,.08,.15]
        or list(e["native_specific_checks"].values())!=[.15,.15]
        or e["independent_scientific_pass"]
        or not x["only_two_branch_scoped_seed_jobs_appended"]
        or not x["requires_full_G4_Linux_Windows_before_fresh_scoring"]
        or not f["authority"]["CPU_TEST_ONLY_scoped_original_first_use"]
        or any(bool(v) for k,v in f["authority"].items()
               if k!="CPU_TEST_ONLY_scoped_original_first_use")):
        raise PermissionError("R100_FROZEN_HYPOTHESIS_SOURCE_OR_OWNER_SCOPE_DRIFT")
    new=set(s["salts"].values())
    older=set(r99.frozen()["source"]["salts"].values())
    if len(new)!=len(s["salts"]) or bool(new&older):
        raise RuntimeError("R100_SPENT_OR_DUPLICATE_SOURCE_SALT")
    return f


def program(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_PROGRAM_OUTSIDE_NEW_FROZEN_COHORT")
    return TRAIN_PROGRAMS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else COMPOSE_PROGRAM


def roles(panel,index):
    if panel not in SALT or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_ROLES_OUTSIDE_NEW_FROZEN_COHORT")
    return TRAIN_BINDINGS[index%3] if panel in ("FIT","VALID","DEV_SEEN") else HELDOUT_BINDING


def chosen(seed,index):
    if seed not in SEEDS or not isinstance(index,int) or index<0:
        raise PermissionError("R100_UNREGISTERED_FACTUAL_ACTION")
    return r85.CALIBRATE if (index//4+SEEDS.index(seed))%2==0 else r85.EXPLOIT_POS


def design_preflight():
    report={}
    for seed in SEEDS:
        report[str(seed)]={}
        for panel,n in COUNT.items():
            buckets={}
            for i in range(n):
                key=(i%2,chosen(seed,i))
                buckets[key]=buckets.get(key,0)+1
            if len(buckets)!=4 or set(buckets.values())!={n//4}:
                raise RuntimeError("R100_COLD_ACTION_STRATA_UNBALANCED")
            report[str(seed)][panel]={"four_action_time_cells":4,"per_cell":n//4}
    return report


def _admit(seed):
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r100-factored-native"
        or os.getenv("R100_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R100_GITHUB_HOSTED_CPU_ORIGINAL_FIRST_USE_ONLY")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
                        text=True,capture_output=True,timeout=10,check=False)
    clean=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
                         capture_output=True,timeout=10,check=False)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or clean.returncode:
        raise PermissionError("R100_EXACT_SHA_OR_CLEAN_SOURCE_MISMATCH")
    return sha


def unit_world(i):
    if not isinstance(i,int) or not 0<=i<12:
        raise PermissionError("R100_INDEPENDENT_UNIT_FIXTURE_ONLY")
    rng=np.random.default_rng(np.random.SeedSequence([UNIT_SALT,i]))
    world=r85._make_synthetic_world(i,rng)
    op=TRAIN_PROGRAMS[i%3] if i<9 else COMPOSE_PROGRAM
    bind=TRAIN_BINDINGS[i%3] if i<9 else HELDOUT_BINDING
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=bind
    world["law_a"]=r98.law_of(factors,bind,op)
    return world,chosen(SEEDS[0],i)


def _world(panel,seed,index,sha):
    if panel not in SALT:
        raise PermissionError("R100_OLD_SPENT_OR_PROTECTED_SOURCE_FORBIDDEN")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R100_UNFROZEN_SEED_OR_WORLD_INDEX")
    if (not sha or sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r100-factored-native"
        or os.getenv("R100_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"):
        raise PermissionError("R100_NEW_SCIENCE_SOURCE_NOT_ADMITTED")
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    world=r85._make_synthetic_world(index%r85.N,rng)
    factors=rng.choice([-1,1],size=4).astype(np.int64)
    op=program(panel,index)
    bind=roles(panel,index)
    world["factors_a"]=factors
    world["r98_op"]=op
    world["r98_roles"]=bind
    world["law_a"]=r98.law_of(factors,bind,op)
    if panel=="OOD_SHIFT":
        world=r86._shift_public_world(world,index)
    return world,chosen(seed,index)


def _public(world,action):
    """Only R98/R99 pure public observation, never old scored source."""
    return r99._observed(world,action)


def _mask_program(slots,raw):
    """Ablate public program bits BOTH in raw features and native event rows."""
    if raw.ndim!=2 or raw.shape[1]!=14 or slots.ndim!=3 or slots.shape[1:]!=(5,13):
        raise ValueError("R100_PROGRAM_ABLATION_INPUT_GEOMETRY")
    a=raw.copy();x=slots.copy()
    a[:,8:10]=0.
    tag=np.rint(raw[:,4:8]*3)
    x[:,:4,8]=tag/15.
    return x,a


def _derange(raw,action,seed,panel):
    if (panel not in ("FIT","VALID") or seed not in SEEDS
        or raw.shape!=(COUNT[panel],14)
        or action.shape!=(COUNT[panel],)):
        raise PermissionError("R100_AUX_RETURN_SHUFFLE_UNFROZEN_PANEL")
    rng=np.random.default_rng(np.random.SeedSequence([RETURN_SHUFFLE_SALT,seed,SALT[panel]]))
    perm=np.arange(len(raw))
    groups={}
    for i in range(len(raw)):
        groups.setdefault((float(raw[i,12]),int(action[i])),[]).append(i)
    for values in groups.values():
        g=np.asarray(values,np.int64)
        if len(g)<8:
            raise RuntimeError("R100_SHUFFLE_PUBLIC_STRATUM_TOO_SMALL")
        for trial in range(256):
            p=rng.permutation(g)
            if not np.any(p==g):
                perm[g]=p
                break
        else:
            raise RuntimeError("R100_INDEPENDENT_DERANGEMENT_NOT_FOUND")
    if (np.any(perm==np.arange(len(raw)))
        or not np.array_equal(raw[perm,12],raw[:,12])
        or not np.array_equal(action[perm],action)):
        raise RuntimeError("R100_RETURN_SHUFFLE_ACTION_TIME_OR_FIXED_POINTS")
    return perm


def _audit(law,raw,action=None,perm=None):
    if law.shape!=(len(raw),) or raw.shape!=(len(raw),14):
        raise ValueError("R100_PRIVATE_SOURCE_AUDIT_SHAPE")
    rates={}
    for pivot in np.unique(raw[:,12]):
        selected=raw[:,12]==pivot
        rate=float(np.mean(law[selected]>0))
        if not .35<=rate<=.65:
            raise RuntimeError("R100_PRIVATE_LAW_PUBLIC_PIVOT_SOURCE_INVALID")
        rates[str(float(pivot))]=rate
    if action is not None:
        for a in (r85.CALIBRATE,r85.EXPLOIT_POS):
            rate=float(np.mean(law[action==a]>0))
            if not .35<=rate<=.65:
                raise RuntimeError("R100_PRIVATE_LAW_CHOSEN_ACTION_SOURCE_INVALID")
    donor_match=None
    if perm is not None:
        donor_match=float(np.mean(law==law[perm]))
        if not .35<=donor_match<=.65:
            raise RuntimeError("R100_RETURN_SHUFFLE_PRIVATE_LAW_CONTROL_INVALID")
    return {"private_law_plus_by_public_pivot":rates,
            "donor_private_law_match":donor_match,
            "return_shuffle_not_independent_causal_null":True,
            "source_valid":True}
