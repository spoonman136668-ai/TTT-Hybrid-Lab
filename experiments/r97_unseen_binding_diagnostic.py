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


def _factual(panel:str,seed:int,sha:str)->dict:
    if panel not in ("FIT","VALID"):
        raise PermissionError("R97_TRAINING_CANNOT_OPEN_FRESH_SCORED_SOURCE")
    n=COUNT[panel]
    slots=np.empty((n,5,13),np.float32)
    raw=np.empty((n,12),np.float32)
    target=np.empty(n,np.float32)
    chosen=np.empty(n,np.int8)
    law=np.empty(n,np.int8)
    factors=np.empty((n,4),np.int8)
    for i in range(n):
        world,action=_world(panel,seed,i,sha)
        public,reward=_public_view(world,action)
        slot=r88._public_slots(public)
        slots[i],raw[i]=slot,_features(slot)
        target[i],chosen[i]=reward,action
        law[i],factors[i]=world["law_a"],world["factors_a"]
    neg=_permuted(raw,seed,panel)
    audit=_audit(law,raw,neg,factors,chosen)
    return {"slots":slots,"raw":raw,"canonical":_canonical(raw),
            "blind":_blind(raw),"negative":neg,"target":target,
            "chosen":chosen,"audit":audit,
            "only_one_factually_selected_action_return":True}


def _paired(panel:str,seed:int,sha:str)->dict:
    if panel not in ("DEV_SEEN","DEV_HELDOUT","OOD_SHIFT","ROLE_REASSIGN"):
        raise PermissionError("R97_EVALUATOR_FORBIDS_TRAINING_OR_OLD_SOURCE")
    n=COUNT[panel]
    slots0=np.empty((n,5,13),np.float32)
    slots2=np.empty_like(slots0)
    raw0=np.empty((n,12),np.float32)
    raw2=np.empty_like(raw0)
    outcomes0=np.empty(n,np.float64)
    outcomes2=np.empty(n,np.float64)
    law=np.empty(n,np.int8)
    factors=np.empty((n,4),np.int8)
    cold=np.empty(n,bool)
    counter=None
    if panel=="ROLE_REASSIGN":
        counter={"slots0":np.empty_like(slots0),"slots2":np.empty_like(slots2),
                 "raw0":np.empty_like(raw0),"raw2":np.empty_like(raw2),
                 "reward0":np.empty_like(outcomes0),"reward2":np.empty_like(outcomes2),
                 "law":np.empty_like(law),"cold":np.empty_like(cold)}
    for i in range(n):
        world,_=_world(panel,seed,i,sha)
        prefix0,r0=_public_view(world,r85.CALIBRATE)
        prefix2,r2=_public_view(world,r85.EXPLOIT_POS)
        if (not np.array_equal(prefix0[:-1],prefix2[:-1])
            or not np.array_equal(prefix0[-1,4:],prefix2[-1,4:])
            or prefix0[-1,0]!=1. or prefix2[-1,2]!=1.):
            raise RuntimeError("R97_PUBLIC_PREFIX_LEAKED_ALTERNATE_ACTION")
        slots0[i],slots2[i]=r88._public_slots(prefix0),r88._public_slots(prefix2)
        raw0[i],raw2[i]=_features(slots0[i]),_features(slots2[i])
        outcomes0[i],outcomes2[i]=r0,r2
        law[i],factors[i]=world["law_a"],world["factors_a"]
        cold[i]=bool(i%2)
        if counter is not None:
            reassigned=copy.deepcopy(world)
            reassigned["r97_binding"]=REASSIGN_BINDING
            reassigned["law_a"]=mux(reassigned["factors_a"],REASSIGN_BINDING)
            p0,q0=_public_view(reassigned,r85.CALIBRATE)
            p2,q2=_public_view(reassigned,r85.EXPLOIT_POS)
            if not np.array_equal(prefix0[:,9:12],p0[:,9:12]):
                # Roles/noisy factors of the real physical observations must
                # remain identical; only public instruction channel 8 changes.
                raise RuntimeError("R97_ROLE_REASSIGNMENT_CHANGED_CAUSAL_FACTOR_NOISE")
            counter["slots0"][i],counter["slots2"][i]=r88._public_slots(p0),r88._public_slots(p2)
            counter["raw0"][i],counter["raw2"][i]=(
                _features(counter["slots0"][i]),_features(counter["slots2"][i]))
            counter["reward0"][i],counter["reward2"][i]=q0,q2
            counter["law"][i]=reassigned["law_a"]
            counter["cold"][i]=bool(i%2)
    neg0=_permuted(raw0,seed,panel)
    neg2=_permuted(raw2,seed,panel)
    audit=_audit(law,raw0,neg0,factors,None)
    result={"slots0":slots0,"slots2":slots2,
            "raw0":raw0,"raw2":raw2,"blind0":_blind(raw0),
            "blind2":_blind(raw2),"canonical0":_canonical(raw0),
            "canonical2":_canonical(raw2),"negative0":neg0,"negative2":neg2,
            "reward0":outcomes0,"reward2":outcomes2,
            "law":law,"cold":cold,"audit":audit}
    if counter is not None:
        counter["blind0"]=_blind(counter["raw0"])
        counter["blind2"]=_blind(counter["raw2"])
        counter["canonical0"]=_canonical(counter["raw0"])
        counter["canonical2"]=_canonical(counter["raw2"])
        counter["negative0"]=_permuted(counter["raw0"],seed,panel)
        counter["negative2"]=_permuted(counter["raw2"],seed,panel)
        result["counter"]=counter
    return result


class RawTaggedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(12,64),nn.GELU(),
                               nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=12:
            raise RuntimeError("R97_RAW_MLP_EXPECTS_12D_PUBLIC_TAG_AND_CUES")
        return self.net(x)[:,0]


class TaggedSetValue(nn.Module):
    """Trainable shared-role set encoder, supplied ONLY public role tags/cues.

    One-hot interpretation of *observed task tag* and shared pooling are
    engineered inductive biases; the physical conditional law and its reward
    mapping must still be learned from one factual outcome per FIT episode.
    """
    def __init__(self):
        super().__init__()
        self.role=nn.Sequential(nn.Linear(5,32),nn.GELU(),
                                nn.Linear(32,32),nn.GELU())
        self.out=nn.Sequential(nn.Linear(35,64),nn.GELU(),
                               nn.Linear(64,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,x):
        if x.ndim!=2 or x.shape[1]!=12:
            raise RuntimeError("R97_TAGGED_SET_EXPECTS_12D_PUBLIC_TAG_AND_CUES")
        tags=torch.round(x[:,4:8]*3).long()
        if bool((tags<0).any()) or bool((tags>3).any()):
            raise RuntimeError("R97_UNKNOWN_PUBLIC_TASK_ROLE_TAG")
        onehot=F.one_hot(tags,num_classes=4).to(x.dtype)
        rows=torch.cat((x[:,:4].unsqueeze(2),onehot),dim=2)
        encoded=self.role(rows).sum(dim=1)
        return self.out(torch.cat((encoded,x[:,8:11]),dim=1))[:,0]


class NativeSidecar(nn.Module):
    """Only an additive experimental value head, never promoted core weights."""
    def __init__(self,core:nn.Module,head:nn.Module):
        super().__init__()
        self.core=core
        self.head=head
    def forward(self,slots,raw):
        return self.core(slots)[:,-1,0]+self.head(raw)


def _models(seed:int):
    if seed not in SEEDS:
        raise PermissionError("R97_UNFROZEN_MODEL_SEED")
    out={}
    out[NATIVE_RAW]=r95._native(seed)
    torch.manual_seed(seed+71)
    out[NATIVE_FLAT]=NativeSidecar(r95._native(seed),RawTaggedMLP())
    torch.manual_seed(seed+72)
    out[NATIVE_SET]=NativeSidecar(r95._native(seed),TaggedSetValue())
    for arm in (MLP_RAW,BLIND):
        torch.manual_seed(seed+73)
        out[arm]=RawTaggedMLP()
    for arm in (MLP_SET,NEGATIVE):
        torch.manual_seed(seed+74)
        out[arm]=TaggedSetValue()
    torch.manual_seed(seed+75)
    out[GRU]=r81.GRU(64)
    torch.manual_seed(seed+76)
    out[UPPER]=r87.PublicValueMLP()
    first,flat,setcore=[dict(z.named_parameters()) for z in (
        out[NATIVE_RAW],out[NATIVE_FLAT].core,out[NATIVE_SET].core)]
    if first.keys()!=flat.keys() or first.keys()!=setcore.keys() or any(
        not torch.equal(first[k],flat[k]) or not torch.equal(first[k],setcore[k])
        for k in first):
        raise RuntimeError("R97_NATIVE_INITIAL_CORES_NOT_EXACT_TENSORS")
    for pair in ((MLP_RAW,BLIND),(MLP_SET,NEGATIVE)):
        a,b=[dict(out[v].named_parameters()) for v in pair]
        if a.keys()!=b.keys() or any(not torch.equal(a[k],b[k]) for k in a):
            raise RuntimeError("R97_SAME_ARCHITECTURE_CONTROLS_NOT_EQUAL_INIT")
    return out


def _arrays(data,arm,suffix=""):
    field=("canonical" if arm==UPPER else "blind" if arm==BLIND else
           "negative" if arm==NEGATIVE else "raw")
    return data["slots"+suffix],data[field+suffix]


def _forward(model,arm,slots,raw):
    if arm in (NATIVE_RAW,GRU):
        output=model(slots)[:,-1,0]
    elif arm in (NATIVE_FLAT,NATIVE_SET):
        output=model(slots,raw)
    else:
        output=model(raw)
    if output.shape!=(len(slots),):
        raise RuntimeError("R97_VALUE_PREDICTION_SHAPE_DRIFT")
    return output


def _predict(model,arm,data,suffix=""):
    slots,raw=_arrays(data,arm,suffix)
    model.eval()
    output=[]
    with torch.no_grad():
        for k in range(0,len(raw),32):
            t=torch.from_numpy(slots[k:k+32])
            f=torch.from_numpy(raw[k:k+32])
            output.extend(_forward(model,arm,t,f).tolist())
    model.train()
    pred=np.asarray(output,np.float64)
    if pred.shape!=(len(raw),) or not np.isfinite(pred).all():
        raise RuntimeError("R97_NONFINITE_OR_INCOMPLETE_PREDICTIONS")
    return pred


def _train(model,arm,fit,valid,batches):
    slots,raw=_arrays(fit,arm)
    x=torch.from_numpy(slots);f=torch.from_numpy(raw)
    target=torch.from_numpy(fit["target"])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    before=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    started=time.perf_counter()
    for indices in batches:
        estimate=_forward(model,arm,x[indices],f[indices])
        loss=F.mse_loss(estimate,target[indices])
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("R97_NONFINITE_OBSERVED_FACTUAL_RETURN_TRAINING_LOSS")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
    elapsed=time.perf_counter()-started
    train_mse=float(np.mean((_predict(model,arm,fit)-fit["target"])**2))
    valid_mse=float(np.mean((_predict(model,arm,valid)-valid["target"])**2))
    if not all(math.isfinite(x) for x in (elapsed,before,train_mse,valid_mse)):
        raise RuntimeError("R97_NONFINITE_FULLY_TRAINED_ARM")
    return {"steps":len(batches),"factual_observed_exposures":len(batches)*BATCH,
            "parameters":sum(p.numel() for p in model.parameters()),
            "valid_initial_MSE":before,"fit_final_MSE":train_mse,
            "valid_final_MSE":valid_mse,"fit_wall_seconds":elapsed}


def _policy_from_public_instruction(raw:np.ndarray)->np.ndarray:
    tags=np.rint(raw[:,4:8]*3).astype(int)
    cues=np.where(raw[:,:4]>0,1,-1)
    signs=np.empty(len(raw),np.int8)
    for i in range(len(raw)):
        roles={tag:int(np.flatnonzero(tags[i]==tag)[0]) for tag in (1,2,3)}
        signs[i]=cues[i,roles[2]] if cues[i,roles[1]]>0 else cues[i,roles[3]]
    return np.where(signs>0,r85.CALIBRATE,r85.EXPLOIT_POS)


def _score(models,data):
    decisions={};metrics={};realized={}
    for name,model in models.items():
        v0=_predict(model,name,data,"0")
        v2=_predict(model,name,data,"2")
        decisions[name]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    public_instruction=_policy_from_public_instruction(data["raw0"])
    physical_law=np.where(data["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
    decisions["PUBLIC_INSTRUCTION_MUX_EVALUATOR_ONLY"]=public_instruction
    decisions["PRIVATE_TRUE_MUX_EVALUATOR_ONLY"]=physical_law
    decisions["ALWAYS_CALIBRATE"]=np.full(len(physical_law),r85.CALIBRATE)
    decisions["ALWAYS_EXPLOIT_POS"]=np.full(len(physical_law),r85.EXPLOIT_POS)
    oracle=np.maximum(data["reward0"],data["reward2"])
    for name,choice in decisions.items():
        ret=np.where(choice==r85.CALIBRATE,data["reward0"],data["reward2"])
        realized[name]=ret
        metrics[name]={"mean_reward":float(np.mean(ret)),
                       "cold_reward":float(np.mean(ret[data["cold"]])),
                       "oracle_regret":float(np.mean(oracle-ret)),
                       "public_instruction_agreement":float(np.mean(choice==public_instruction)),
                       "true_physical_law_agreement":float(np.mean(choice==physical_law))}
    metrics["PHYSICAL_PAIRED_ORACLE_EVALUATOR_ONLY"]={
        "mean_reward":float(np.mean(oracle)),
        "cold_reward":float(np.mean(oracle[data["cold"]])),
        "oracle_regret":0.}
    return {"metrics":metrics,"actions":decisions,"rewards":realized}


def _bootstrap_lower(a,b,seed):
    if a.shape!=(COUNT["DEV_HELDOUT"],) or b.shape!=a.shape:
        raise RuntimeError("R97_FROZEN_PAIRED_BOOTSTRAP_SHAPE")
    rng=np.random.default_rng(np.random.SeedSequence([BOOTSTRAP_SALT,seed]))
    index=rng.integers(0,len(a),size=(2000,len(a)))
    return float(np.quantile((a-b)[index].mean(axis=1),.025))


def original_ci(seed:int)->dict:
    sha=_admit(seed)
    preflight=balance_preflight()
    torch.set_num_threads(1)
    path=ROOT/f"local_runs/r97_binding_transfer_seed{seed}.json"
    if path.exists():
        raise FileExistsError("R97_FIRST_USE_SOURCE_ALREADY_CONSUMED_OR_UNCERTAIN")
    started=time.perf_counter()
    result={"schema":SCHEMA,"seed":seed,"source_sha":sha,"freeze_blob":FREEZE_BLOB,
            "R96_parent_SHA":PARENT,"platform":platform.platform(),
            "torch":str(torch.__version__),
            "source_preflight":preflight[str(seed)],
            "status":"R97_ENGINEERING_OR_RESOURCE_INCOMPLETE_NO_SCIENTIFIC_VERDICT",
            "training":{},"source_audits":{},
            "all_nine_weights_frozen_before_first_DEV":False,
            "DEV_HELDOUT_OOD_ROLE_REASSIGN_first_use":False,
            "older_scored_source_or_weights_used":False,"scientific_pass":False}
    try:
        fit=_factual("FIT",seed,sha)
        valid=_factual("VALID",seed,sha)
        if not fit["only_one_factually_selected_action_return"]:
            raise RuntimeError("R97_UNCHOSEN_BRANCH_SUPERVISION")
        result["source_audits"]["FIT"]=fit["audit"]
        result["source_audits"]["VALID"]=valid["audit"]
        batch_rng=np.random.default_rng(np.random.SeedSequence([BATCH_SALT,seed]))
        batches=batch_rng.integers(0,COUNT["FIT"],size=(STEPS,BATCH))
        models=_models(seed)
        for arm in ARMS:
            result["training"][arm]=_train(models[arm],arm,fit,valid,batches)
            if time.perf_counter()-started>MAX_SECONDS:
                raise TimeoutError("R97_FROZEN_WHOLE_CPU_PROCESS_BUDGET_EXCEEDED")
        if set(result["training"])!=set(ARMS):
            raise RuntimeError("R97_NINE_ARM_FULL_TRAINING_INCOMPLETE")
        result["all_nine_weights_frozen_before_first_DEV"]=True
        # ALL parameters frozen; no heldout world or paired unchosen branch
        # created or evaluated at any prior time.
        result["DEV_HELDOUT_OOD_ROLE_REASSIGN_first_use"]=True
        seen=_paired("DEV_SEEN",seed,sha)
        heldout=_paired("DEV_HELDOUT",seed,sha)
        shifted=_paired("OOD_SHIFT",seed,sha)
        reassigned=_paired("ROLE_REASSIGN",seed,sha)
        result["source_audits"].update({
            "DEV_SEEN":seen["audit"],"DEV_HELDOUT":heldout["audit"],
            "OOD_SHIFT":shifted["audit"],"ROLE_REASSIGN":reassigned["audit"]})
        sm=_score(models,seen)
        hm=_score(models,heldout)
        om=_score(models,shifted)
        rm=_score(models,reassigned)
        swapped=_score(models,reassigned["counter"])
        d=hm["metrics"];o=om["metrics"]
        original_choice=rm["actions"][MLP_SET]
        changed_choice=swapped["actions"][MLP_SET]
        private_changed=reassigned["law"]!=reassigned["counter"]["law"]
        if not private_changed.any():
            raise RuntimeError("R97_NO_GENUINELY_CHANGED_PHYSICAL_ROLE_LAWS")
        true_before=np.where(reassigned["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
        true_after=np.where(reassigned["counter"]["law"]>0,r85.CALIBRATE,r85.EXPLOIT_POS)
        measures={
            "HELDOUT_set_minus_blind_reward":d[MLP_SET]["mean_reward"]-d[BLIND]["mean_reward"],
            "HELDOUT_set_minus_tag_permuted_reward":d[MLP_SET]["mean_reward"]-d[NEGATIVE]["mean_reward"],
            "HELDOUT_set_minus_raw_MLP_reward":d[MLP_SET]["mean_reward"]-d[MLP_RAW]["mean_reward"],
            "HELDOUT_set_minus_GRU_reward":d[MLP_SET]["mean_reward"]-d[GRU]["mean_reward"],
            "HELDOUT_canonical_upper_minus_set_reward":d[UPPER]["mean_reward"]-d[MLP_SET]["mean_reward"],
            "HELDOUT_set_public_instruction_agreement":d[MLP_SET]["public_instruction_agreement"],
            "SHIFTED_COLD_set_minus_tag_permuted_reward":o[MLP_SET]["cold_reward"]-o[NEGATIVE]["cold_reward"],
            "HELDOUT_set_vs_tag_permuted_bootstrap_lower95":_bootstrap_lower(
                hm["rewards"][MLP_SET],hm["rewards"][NEGATIVE],seed),
            "ROLE_REASSIGN_private_law_change_fraction":float(private_changed.mean()),
            "ROLE_REASSIGN_set_action_flip_given_true_law_change":float(np.mean(
                original_choice[private_changed]!=changed_choice[private_changed])),
            "ROLE_REASSIGN_set_private_law_accuracy_both":float((
                (original_choice==true_before).sum()+
                (changed_choice==true_after).sum())/(2*len(private_changed))),
            "NATIVE_HELDOUT_set_minus_flat_reward":d[NATIVE_SET]["mean_reward"]-d[NATIVE_FLAT]["mean_reward"],
            "NATIVE_HELDOUT_set_minus_original_reward":d[NATIVE_SET]["mean_reward"]-d[NATIVE_RAW]["mean_reward"],
            "NATIVE_SHIFTED_COLD_set_minus_flat_reward":o[NATIVE_SET]["cold_reward"]-o[NATIVE_FLAT]["cold_reward"],
            "NATIVE_HELDOUT_conventional_set_minus_native_set_reward":d[MLP_SET]["mean_reward"]-d[NATIVE_SET]["mean_reward"]}
        e=frozen()["evaluation"]
        t=e["mechanism_checks"]
        nc=e["native_checks"]
        m={
          "heldout_set_beats_role_blind":measures["HELDOUT_set_minus_blind_reward"]>=t["heldout_tagged_set_minus_blind_MLP_reward_min"],
          "heldout_set_beats_tag_disrupted":measures["HELDOUT_set_minus_tag_permuted_reward"]>=t["heldout_tagged_set_minus_permuted_negative_reward_min"],
          "heldout_set_beats_strong_raw_MLP":measures["HELDOUT_set_minus_raw_MLP_reward"]>=t["heldout_tagged_set_minus_raw_MLP_reward_min"],
          "heldout_set_beats_GRU":measures["HELDOUT_set_minus_GRU_reward"]>=t["heldout_tagged_set_minus_GRU_reward_min"],
          "heldout_set_near_engineered_upper":measures["HELDOUT_canonical_upper_minus_set_reward"]<=t["heldout_canonical_upper_minus_tagged_set_reward_gap_max"],
          "heldout_set_public_instruction_agreement":measures["HELDOUT_set_public_instruction_agreement"]>=t["heldout_tagged_set_public_instruction_accuracy_min"],
          "shifted_cold_outperforms_tag_disruption":measures["SHIFTED_COLD_set_minus_tag_permuted_reward"]>=t["shifted_OOD_cold_tagged_set_minus_permuted_reward_min"],
          "heldout_paired_bootstrap":measures["HELDOUT_set_vs_tag_permuted_bootstrap_lower95"]>=t["heldout_tagged_set_vs_permuted_paired_bootstrap_lower95_min"],
          "physical_assignment_changes_law":measures["ROLE_REASSIGN_private_law_change_fraction"]>=t["role_reassignment_true_law_flip_fraction_min"],
          "model_changes_action_with_law":measures["ROLE_REASSIGN_set_action_flip_given_true_law_change"]>=t["role_reassignment_tagged_set_action_flip_on_law_change_min"],
          "correct_on_original_and_reassigned":measures["ROLE_REASSIGN_set_private_law_accuracy_both"]>=t["role_reassignment_tagged_set_private_law_accuracy_min"]}
        n={
          "native_set_beats_flat_head":measures["NATIVE_HELDOUT_set_minus_flat_reward"]>=nc["heldout_native_set_minus_native_flat_reward_min"],
          "native_set_beats_unmodified_core":measures["NATIVE_HELDOUT_set_minus_original_reward"]>=nc["heldout_native_set_minus_original_native_reward_min"],
          "native_set_beats_flat_shifted_cold":measures["NATIVE_SHIFTED_COLD_set_minus_flat_reward"]>=nc["shifted_OOD_cold_native_set_minus_flat_reward_min"],
          "native_set_near_conventional_set":measures["NATIVE_HELDOUT_conventional_set_minus_native_set_reward"]<=nc["heldout_native_set_vs_conventional_set_reward_gap_max"]}
        result["DEV_SEEN"]=sm["metrics"]
        result["DEV_HELDOUT"]=hm["metrics"]
        result["OOD_SHIFT"]=om["metrics"]
        result["ROLE_REASSIGN"]={"original":rm["metrics"],
            "counterfactual":swapped["metrics"],
            "law_changed_n":int(private_changed.sum())}
        result["measures"]=measures
        result["mechanism_checks"]={k:bool(v) for k,v in m.items()}
        result["native_checks"]={k:bool(v) for k,v in n.items()}
        status=e["statuses"]
        result["status"]=(status["both"] if all(m.values()) and all(n.values())
                          else status["conventional"] if all(m.values())
                          else status["no_go"])
    except BaseException as exc:
        result["error_type"]=type(exc).__name__
        result["error_message"]=str(exc)
        result["status"]=("R97_SOURCE_OR_CONTROL_INVALID_NO_SCIENTIFIC_VERDICT"
                          if "R97_PRIVATE_LAW_" in str(exc) or
                          "R97_ROLE_DISRUPTION_NEGATIVE_CONTROL_INVALID" in str(exc)
                          else "R97_ENGINEERING_OR_RESOURCE_INCOMPLETE_NO_SCIENTIFIC_VERDICT")
        raise
    finally:
        result["whole_process_seconds"]=time.perf_counter()-started
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open("x",encoding="utf-8") as fp:
            json.dump(result,fp,indent=2,sort_keys=True,allow_nan=False)
            fp.write("\n")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    args=p.parse_args()
    result=original_ci(args.seed)
    print(json.dumps({"seed":args.seed,"status":result["status"],
                      "trained":len(result["training"]),
                      "DEV_first_use":result["DEV_HELDOUT_OOD_ROLE_REASSIGN_first_use"],
                      "scientific_pass":False},sort_keys=True))
