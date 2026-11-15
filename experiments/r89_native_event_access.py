"""R89 frozen prospective native Hybrid role-event access scientific comparison.

Run only on original exact clean same-repo first-attempt hosted Ubuntu CPU job
AFTER full G4 Linux/Windows qualification. Native R74 width1024 original hard
router plus R75 equivalent triangular fastweight TTT; no fused GPU claim.
All new FIT labels are one true physically realized selected intervention
return. Never generate new evaluator DEV/OOD before all model weights freeze.
No R85-R88 previously scored data or model weights accessed.
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

from experiments import r74_capability_ceiling as r74
from experiments import r75_triangular_fastweight as r75
from experiments import r81_paired_history as r81
from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r88_causal_event_access as r88

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r89_native_event_access_freeze.json"
FREEZE_BLOB = "72ed933c253a6f447bca0647a149b75dde732db7"
SCHEMA = "ttt-hybrid.r89-native-role-event-access-first-study.v1"
PARENT = "698eb433c3b15b5fe84679f0fe40408d13daff03"
BRANCH = "research/r89-native-role-event-access-20261115"
SEEDS = (1000211, 1000223)
SALT = {"FIT":25310111, "FIT_VALIDATION":25310211,
        "DEV":25310311, "OOD_SHIFT":25310411}
COUNT = {"FIT":2048,"FIT_VALIDATION":256,"DEV":256,"OOD_SHIFT":256}
ARMS = ("NATIVE_FULL_RAW48","NATIVE_FULL_DETERMINISTIC_A_SLOTS5",
        "NATIVE_FULL_LEARNED_ROLE_ACCESS5",
        "GRU64_FULL_RAW48","GRU64_DETERMINISTIC_A_SLOTS5",
        "GRU64_SHUFFLED_A_SLOTS5",
        "GRU256_DETERMINISTIC_A_SLOTS5","MLP64_LEGAL_RAW_A_CUES")
RAW, HARD, LEARN = ARMS[:3]
GRU_RAW, GRU_HARD, GRU_SHUFFLE, GRU256, MLP = ARMS[3:]
NATIVE = (RAW,HARD,LEARN)
GRUS = (GRU_RAW,GRU_HARD,GRU_SHUFFLE,GRU256)
UPDATES = {a:(1024 if a in NATIVE else 4096 if a==MLP else 8192)
           for a in ARMS}
WIDTH = 1024
BATCH = 2
EVENTS = 5


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R89_FROZEN_PROSPECTIVE_SOURCE_BLOB_DRIFT")
    r88.frozen()
    f=json.loads(FREEZE.read_text(encoding="utf-8"))
    p,s,t,e=f["parent"],f["sources"],f["training"],f["evaluation"]
    if (f["schema"]!=SCHEMA
        or f["stage"]!="IMMUTABLE_PROSPECTIVE_FREEZE_ONLY_BEFORE_R89_IMPLEMENTATION_TEST_CI_OR_SOURCE_FIRST_USE"
        or p["R88_qualified_head"]!=PARENT
        or p["R88_freeze_blob"]!=r88.FREEZE_BLOB
        or p["R88_status"]!="R88_LEARNED_ROLE_EVENT_READOUT_DIAGNOSTIC_SIGNAL_ONLY_BOTH_SEEDS_13_OF_13"
        or s["independent_salts"]!=SALT or s["seeds"]!=list(SEEDS)
        or s["counts"]!=COUNT or s["public_selected_rows_per_episode"]!=48
        or f["arms"]!=list(ARMS)
        or t["native_width"]!=WIDTH or t["batch_size"]!=BATCH
        or t["native_updates_each"]!=1024
        or t["GRU64_and_GRU256_updates_each"]!=8192
        or t["MLP64_updates"]!=4096
        or t["original_native_core_parameter_count"]!=16818178
        or e["paired_bootstrap"]["resamples"]!=2000
        or e["thresholds"]["DEV_learned_native_minus_raw_native_min"]!=1.
        or e["thresholds"]["DEV_learned_native_minus_native_deterministic_slot_min"]!=-.25
        or e["thresholds"]["OOD_cold_native_learned_minus_raw_min"]!=.75
        or any(x for k,x in f["authority"].items()
               if k!="cpu_scoped_factual_training_allowed_without_merge")):
        raise PermissionError("R89_FROZEN_OR_AUTHORITY_DRIFT")
    forbidden=set(p["R88_spent_salts"])|set(p["R87_spent"])|set(p["R86_spent"])|set(p["R85_still_sealed"])|{p["R85_CALIBRATION_spent"]}
    if (set(SALT.values()) & forbidden
        or not f["authority"]["cpu_scoped_factual_training_allowed_without_merge"]):
        raise RuntimeError("R89_PREVIOUS_SOURCE_OR_AUTHORITY_FENCE_FAIL")
    return f


def _admit(seed: int) -> str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_EVENT_NAME")!="pull_request"
        or os.getenv("GITHUB_REPOSITORY")!="spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("GITHUB_JOB")!="r89-native-role-event-access"
        or os.getenv("R89_SEED")!=str(seed)
        or os.getenv("RUNNER_OS")!="Linux"
        or os.getenv("RUNNER_ENVIRONMENT")!="github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT")!="1"
        or "/merge" not in os.getenv("GITHUB_REF","")):
        raise PermissionError("R89_NOT_ORIGINAL_FIRST_ATTEMPT_CLEAN_HOSTED_JOB")
    sha=os.getenv("EXPECTED_CANDIDATE_SHA","")
    head=subprocess.run(["git","rev-parse","HEAD"],cwd=ROOT,
        capture_output=True,text=True,timeout=10,check=False)
    dirty=subprocess.run(["git","diff","--quiet","HEAD","--"],cwd=ROOT,
        capture_output=True,timeout=10,check=False)
    if len(sha)!=40 or head.returncode or head.stdout.strip()!=sha or dirty.returncode:
        raise PermissionError("R89_EXACT_SHA_OR_DIRTY_SOURCE_CUSTODY_FAIL")
    return sha


def _world(panel:str,seed:int,index:int,*,custody_sha:str|None=None):
    if panel not in SALT:
        raise PermissionError("R89_OLD_SPENT_OR_SEALED_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index,int) or not 0<=index<COUNT[panel]:
        raise PermissionError("R89_NONFROZEN_SEED_OR_INDEX")
    if (not custody_sha or custody_sha!=os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS")!="true"
        or os.getenv("GITHUB_JOB")!="r89-native-role-event-access"
        or os.getenv("GITHUB_HEAD_REF")!=BRANCH
        or os.getenv("R89_SEED")!=str(seed)):
        raise PermissionError("R89_ORIGINAL_FIRST_USE_SOURCE_NOT_ADMITTED")
    # Existing immutable physics only. Fresh independent RNG salt+seed+index.
    rng=np.random.default_rng(np.random.SeedSequence([SALT[panel],seed,index]))
    w=r85._make_synthetic_world(index%r85.N,rng)
    if panel=="OOD_SHIFT":
        w=r86._shift_public_world(w,index)
    chosen=int(rng.choice([r85.CALIBRATE,r85.EXPLOIT_POS]))
    return w,chosen


def _test_world(index:int)->dict:
    rng=np.random.default_rng(np.random.SeedSequence([25319991,index]))
    return r85._make_synthetic_world(index,rng)


def _view(w:dict,action:int):
    return r86._view(w,action)


def _slots(x:np.ndarray):
    return r88._public_slots(x)


def _shuffle(slots:np.ndarray,seed:int,panel:str)->np.ndarray:
    if panel not in COUNT or slots.shape!=(COUNT[panel],EVENTS,13):
        raise PermissionError("R89_SHUFFLE_ONLY_FROZEN_PANEL_GEOMETRY")
    times=slots[:,-1,12]
    if len(np.unique(times))!=2:
        raise RuntimeError("R89_SOURCE_STRATUM_MISSING")
    rng=np.random.default_rng(np.random.SeedSequence([25310555,seed,SALT[panel]]))
    index=np.arange(len(times))
    for t in np.unique(times):
        w=np.flatnonzero(times==t)
        index[w]=rng.permutation(w)
    result=slots.copy()
    result[:,:4]=slots[index,:4]
    return result


def _factual(panel:str,seed:int,*,custody_sha:str)->dict:
    if panel not in ("FIT","FIT_VALIDATION"):
        raise PermissionError("R89_NO_SCORED_PANEL_IN_FIT")
    n=COUNT[panel]
    raw=np.zeros((n,48,13),np.float32)
    slots=np.zeros((n,5,13),np.float32)
    feats=np.zeros((n,8),np.float32)
    target=np.zeros(n,np.float32)
    for i in range(n):
        w,act=_world(panel,seed,i,custody_sha=custody_sha)
        x,reward=_view(w,act)
        raw[i]=x
        slots[i]=_slots(x)
        feats[i]=r87._legal_features(x)[0]
        target[i]=reward
    return {"raw":raw,"slots":slots,"shuffled":_shuffle(slots,seed,panel),
            "features":feats,"target":target,"one_factual_action_return":True}


def _scored(panel:str,seed:int,*,custody_sha:str)->dict:
    if panel not in ("DEV","OOD_SHIFT"):
        raise PermissionError("R89_SCORING_ONLY_IN_NEW_DEV_OOD")
    n=COUNT[panel]
    raw0,raw2=np.zeros((n,48,13),np.float32),np.zeros((n,48,13),np.float32)
    slots0,slots2=np.zeros((n,5,13),np.float32),np.zeros((n,5,13),np.float32)
    feats0,feats2=np.zeros((n,8),np.float32),np.zeros((n,8),np.float32)
    r0,r2=np.zeros(n,np.float64),np.zeros(n,np.float64)
    cold,legal,private=np.zeros(n,bool),np.zeros(n,np.int64),np.zeros(n,np.int64)
    for i in range(n):
        w,_=_world(panel,seed,i,custody_sha=custody_sha)
        x0,v0=_view(w,r85.CALIBRATE)
        x2,v2=_view(w,r85.EXPLOIT_POS)
        if (not np.array_equal(x0[:-1],x2[:-1])
            or not np.array_equal(x0[-1,4:],x2[-1,4:])
            or x0[-1,0]!=1.0 or x2[-1,2]!=1.0):
            raise RuntimeError("R89_PROTECTED_EVALUATOR_CAUSAL_PREFIX_DRIFT")
        raw0[i],raw2[i]=x0,x2
        slots0[i],slots2[i]=_slots(x0),_slots(x2)
        feats0[i],feats2[i]=r87._legal_features(x0)[0],r87._legal_features(x2)[0]
        r0[i],r2[i]=v0,v2
        cold[i],legal[i],private[i]=bool(i%2),r85._policy(x0),w["law_a"]
    return {"raw0":raw0,"raw2":raw2,
            "slots0":slots0,"slots2":slots2,
            "shuffled0":_shuffle(slots0,seed,panel),
            "shuffled2":_shuffle(slots2,seed,panel),
            "features0":feats0,"features2":feats2,
            "r0":r0,"r2":r2,"cold":cold,
            "public_lawful_action_evaluator_only":legal,
            "private_law_evaluator_only":private}


class NativeLearnedRoleEventAccess(nn.Module):
    """Trainable public role-sensitive 48->5 row front-end and original native core.

    The phase mask and public role ids are priors, not autonomously discovered.
    No hidden A law, parity label or evaluator oracle is exposed.
    """
    def __init__(self,native:nn.Module):
        super().__init__()
        self.native=native
        self.key=nn.Sequential(nn.Linear(13,32),nn.GELU(),nn.Linear(32,32))
        self.query=nn.Parameter(torch.randn(4,32)*.02)
        self.role_strength_raw=nn.Parameter(torch.tensor(24.,dtype=torch.float32))

    def legal_rows(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim!=3 or x.shape[1:]!=(48,13):
            raise ValueError("R89_ADAPTER_ONLY_ORIGINAL_LEGAL_48x13")
        # Prior PROBE action onehot[5] and actually observed phase marker +1.
        mask=(x[:,:,5]>.5)&(x[:,:,11]>.5)
        if not bool((mask.sum(dim=1)==4).all()):
            raise RuntimeError("R89_MISSING_ACTUAL_A_PHASE_PROBE")
        roles=x[:,:,9]
        fixed=x.new_tensor([.25,.5,.75,1.])
        q=self.query
        key=self.key(x)
        scores=torch.einsum("brd,btd->brt",q[None].expand(len(x),-1,-1),key)/(32.**.5)
        scores=scores-F.softplus(self.role_strength_raw)*(roles[:,None,:]-fixed[None,:,None]).square()
        scores=scores.masked_fill(~mask[:,None,:],-1e4)
        selected=torch.bmm(scores.softmax(dim=-1),x)
        # No synthesized hidden factor or return: five legal public rows.
        return torch.cat([selected,x[:,-1:,:]],dim=1)

    def forward(self,x):
        return self.native(self.legal_rows(x))


def _models(seed:int)->dict:
    torch.manual_seed(seed)
    original=r74.make_model("HybridPilot_FULL",WIDTH)
    ttt_old=original.ttt
    ttt_new=r75.TriangularFastWeightMemory(WIDTH,learning_rate=ttt_old.eta)
    ttt_new.load_state_dict(ttt_old.state_dict(),strict=True)
    original.ttt=ttt_new
    hard=copy.deepcopy(original)
    access=copy.deepcopy(original)
    if sum(p.numel() for p in original.parameters())!=16818178:
        raise RuntimeError("R89_NATIVE_MAX_WIDTH_OR_PARAMETER_DRIFT")
    a,b,c=[dict(m.named_parameters()) for m in (original,hard,access)]
    if a.keys()!=b.keys() or a.keys()!=c.keys() or any(
        not torch.equal(a[k],b[k]) or not torch.equal(a[k],c[k]) for k in a):
        raise RuntimeError("R89_NATIVE_CORES_NOT_IDENTICAL_INITIAL_TENSORS")
    torch.manual_seed(seed+301)
    role=NativeLearnedRoleEventAccess(access)
    torch.manual_seed(seed+64)
    gru_raw=r81.GRU(64)
    torch.manual_seed(seed+65)
    gru_slots=r81.GRU(64)
    torch.manual_seed(seed+66)
    gru_shuffled=r81.GRU(64)
    torch.manual_seed(seed+256)
    gru256=r81.GRU(256)
    torch.manual_seed(seed+512)
    mlp=r87.PublicValueMLP()
    return dict(zip(ARMS,(original,hard,role,gru_raw,gru_slots,
                          gru_shuffled,gru256,mlp)))


def _input(source:dict,arm:str)->np.ndarray:
    return source[{RAW:"raw",HARD:"slots",LEARN:"raw",
                   GRU_RAW:"raw",GRU_HARD:"slots",GRU_SHUFFLE:"shuffled",
                   GRU256:"slots",MLP:"features"}[arm]]


def _prediction(m:nn.Module,x:torch.Tensor,arm:str)->torch.Tensor:
    y=m(x)
    return y if arm==MLP else y[:,-1,0]


def _valid(m,arm,v)->float:
    m.eval()
    x=torch.from_numpy(_input(v,arm))
    y=torch.from_numpy(v["target"])
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),BATCH):
            est=_prediction(m,x[i:i+BATCH],arm)
            out.extend((est-y[i:i+BATCH]).square().tolist())
    m.train()
    return float(np.mean(out))


def _train(m,arm,fit,valid,indices):
    x=torch.from_numpy(_input(fit,arm))
    y=torch.from_numpy(fit["target"])
    optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
    steps=UPDATES[arm]
    metrics={"0":_valid(m,arm,valid)}
    begin=time.perf_counter()
    for update in range(1,steps+1):
        ids=indices[update-1]
        est=_prediction(m,x[ids],arm)
        loss=F.mse_loss(est,y[ids])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
        optimizer.step()
        if update in (512,1024,2048,4096,8192):
            metrics[str(update)]=_valid(m,arm,valid)
    elapsed=time.perf_counter()-begin
    if any(not math.isfinite(q) for q in metrics.values()) or any(
        not bool(torch.isfinite(p).all()) for p in m.parameters()):
        raise RuntimeError("R89_NONFINITE_OR_INCOMPLETE_NATIVE_CONVENTIONAL_TRAINING")
    return {"updates":steps,"parameters":sum(p.numel() for p in m.parameters()),
            "same_FIT_worlds":COUNT["FIT"],"factual_selected_event_exposures":steps*BATCH,
            "FIT_validation_MSE":metrics,"train_wall_seconds":elapsed,
            "original_native_hard_router":arm in NATIVE,
            "recurrent_unroll":48 if arm in (RAW,GRU_RAW) else 5 if arm!=MLP else 0}


def _eval_input(source:dict,arm:str,act:int)->np.ndarray:
    tail="0" if act==r85.CALIBRATE else "2"
    base={RAW:"raw",HARD:"slots",LEARN:"raw",
          GRU_RAW:"raw",GRU_HARD:"slots",GRU_SHUFFLE:"shuffled",
          GRU256:"slots",MLP:"features"}[arm]
    return source[base+tail]


def _pred_all(m,arm,x)->np.ndarray:
    m.eval()
    out=[]
    with torch.no_grad():
        for i in range(0,len(x),BATCH):
            out.extend(_prediction(m,torch.from_numpy(x[i:i+BATCH]),arm).tolist())
    m.train()
    v=np.asarray(out,np.float64)
    if v.shape!=(len(x),) or not np.isfinite(v).all():
        raise RuntimeError("R89_INVALID_LEGAL_ACTION_VALUE_PREDICTIONS")
    return v


def _evaluate(models,source):
    actions={}
    for arm,m in models.items():
        v0=_pred_all(m,arm,_eval_input(source,arm,r85.CALIBRATE))
        v2=_pred_all(m,arm,_eval_input(source,arm,r85.EXPLOIT_POS))
        actions[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    actions["PUBLIC_A_PARITY_MEMORY_RULE_EVALUATOR_ONLY"]=source["public_lawful_action_evaluator_only"]
    actions["ALWAYS_CALIBRATE"]=np.full(COUNT["DEV"],r85.CALIBRATE)
    actions["ALWAYS_EXPLOIT_POS"]=np.full(COUNT["DEV"],r85.EXPLOIT_POS)
    actions["LAST_A_CUE"]=np.asarray(
        [r85._policy(x,"LAST_A") for x in source["raw0"]],np.int64)
    actions["RECENT_B_PARITY"]=np.asarray(
        [r85._policy(x,"RECENT_B") for x in source["raw0"]],np.int64)
    rewards={a:np.where(v==r85.CALIBRATE,source["r0"],source["r2"])
             for a,v in actions.items()}
    oracle=np.maximum(source["r0"],source["r2"])
    rewards["PAIRED_PHYSICAL_ORACLE_EVALUATOR_ONLY"]=oracle
    cold=source["cold"]
    metric={arm:{"mean_reward":float(reward.mean()),
                 "cold_reward":float(reward[cold].mean()),
                 "early_reward":float(reward[~cold].mean()),
                 "oracle_regret":float((oracle-reward).mean())}
            for arm,reward in rewards.items()}
    public_signs=np.asarray([r87._legal_features(row)[1] for row in source["raw0"]])
    return {"metrics":metric,"actions":actions,"rewards":rewards,
            "public_sign_accuracy_evaluator_only":float(
                np.mean(np.where(public_signs>0,1,-1)==source["private_law_evaluator_only"]))}


def _bootstrap(native:np.ndarray,baseline:np.ndarray,seed:int)->float:
    if native.shape!=(COUNT["DEV"],) or baseline.shape!=native.shape:
        raise RuntimeError("R89_BOOTSTRAP_COUNTS_DRIFT")
    rng=np.random.default_rng(np.random.SeedSequence([25310555,seed]))
    ix=rng.integers(0,len(native),(2000,len(native)))
    return float(np.quantile((native-baseline)[ix].mean(axis=1),.025))


def original_ci(seed:int):
    sha=_admit(seed)
    torch.set_num_threads(1)
    target=ROOT/f"local_runs/r89_native_event_seed{seed}.json"
    if target.exists():
        raise FileExistsError("R89_OLD_OR_UNCERTAIN_FIRST_USE_REPLAY_DENIED")
    begin=time.perf_counter()
    result={"schema":SCHEMA,"candidate_SHA":sha,"parent_R88_SHA":PARENT,
            "freeze_git_blob":FREEZE_BLOB,"seed":seed,
            "completed_arms":{},"all_model_weights_frozen_before_DEV_OOD":False,
            "DEV_OOD_FIRST_USE":False,"old_spent_or_reserved_source_used":False,
            "scientific_pass":False,
            "status":"R89_INCOMPLETE_ENGINEERING_OR_RESOURCE_NOT_SCIENTIFIC_NEGATIVE",
            "platform":platform.platform(),"torch":str(torch.__version__)}
    try:
        fit=_factual("FIT",seed,custody_sha=sha)
        valid=_factual("FIT_VALIDATION",seed,custody_sha=sha)
        if not fit["one_factual_action_return"]:
            raise RuntimeError("R89_NOT_SINGLE_LOGGED_ACTION_RETURN")
        models=_models(seed)
        rng=np.random.default_rng(np.random.SeedSequence([25310611,seed]))
        indices=rng.integers(0,COUNT["FIT"],size=(max(UPDATES.values()),BATCH))
        for arm in ARMS:
            result["completed_arms"][arm]=_train(models[arm],arm,fit,valid,indices)
        if set(result["completed_arms"])!=set(ARMS):
            raise RuntimeError("R89_UNFINISHED_MODEL_CONTROLS")
        result["all_model_weights_frozen_before_DEV_OOD"]=True
        result["DEV_OOD_FIRST_USE"]=True
        dev=_scored("DEV",seed,custody_sha=sha)
        ood=_scored("OOD_SHIFT",seed,custody_sha=sha)
        d=_evaluate(models,dev)
        o=_evaluate(models,ood)
        dm,om=d["metrics"],o["metrics"]
        ld,rd,hd=dm[LEARN],dm[RAW],dm[HARD]
        strongest_gru_slot=max(dm[GRU_HARD]["mean_reward"],dm[GRU256]["mean_reward"])
        fixed=max(dm["ALWAYS_CALIBRATE"]["mean_reward"],dm["ALWAYS_EXPLOIT_POS"]["mean_reward"])
        agreement=float(np.mean(d["actions"][LEARN]==dev["public_lawful_action_evaluator_only"]))
        lower=_bootstrap(d["rewards"][LEARN],d["rewards"][RAW],seed)
        scores={
          "DEV_learn_minus_raw_native":ld["mean_reward"]-rd["mean_reward"],
          "DEV_learn_minus_hard_native_slots":ld["mean_reward"]-hd["mean_reward"],
          "DEV_learn_minus_best_trained_slot_GRU":ld["mean_reward"]-strongest_gru_slot,
          "DEV_learn_minus_shuffled_slot_GRU":ld["mean_reward"]-dm[GRU_SHUFFLE]["mean_reward"],
          "DEV_learn_minus_best_fixed":ld["mean_reward"]-fixed,
          "DEV_cold_learn_minus_raw_native":ld["cold_reward"]-rd["cold_reward"],
          "DEV_learn_lawful_rule_agreement":agreement,
          "DEV_learn_physical_oracle_regret":ld["oracle_regret"],
          "DEV_paired_lower95_learn_minus_raw_native":lower,
          "OOD_learn_minus_raw_native":om[LEARN]["mean_reward"]-om[RAW]["mean_reward"],
          "OOD_cold_learn_minus_raw_native":om[LEARN]["cold_reward"]-om[RAW]["cold_reward"],
          "OOD_learn_minus_GRU256_slots":om[LEARN]["mean_reward"]-om[GRU256]["mean_reward"],
          "DEV_public_A_sign_accuracy_evaluator_only":d["public_sign_accuracy_evaluator_only"]}
        t=frozen()["evaluation"]["thresholds"]
        checks={
          "DEV_learn_vs_raw_native":scores["DEV_learn_minus_raw_native"]>=t["DEV_learned_native_minus_raw_native_min"],
          "DEV_learn_vs_deterministic_native_slots":scores["DEV_learn_minus_hard_native_slots"]>=t["DEV_learned_native_minus_native_deterministic_slot_min"],
          "DEV_learn_vs_best_slot_GRU":scores["DEV_learn_minus_best_trained_slot_GRU"]>=t["DEV_learned_native_minus_best_trained_slot_GRU_min"],
          "DEV_learn_vs_shuffled":scores["DEV_learn_minus_shuffled_slot_GRU"]>=t["DEV_learned_native_minus_shuffled_slot_GRU_min"],
          "DEV_learn_vs_fixed":scores["DEV_learn_minus_best_fixed"]>=t["DEV_learned_native_minus_best_fixed_min"],
          "DEV_cold_learn_vs_raw":scores["DEV_cold_learn_minus_raw_native"]>=t["DEV_cold_native_learned_minus_raw_min"],
          "DEV_lawful_rule_agreement":agreement>=t["DEV_lawful_parity_rule_action_agreement_min"],
          "DEV_oracle_regret":ld["oracle_regret"]<=t["DEV_learned_native_oracle_regret_max"],
          "DEV_paired_bootstrap":lower>0,
          "OOD_learn_vs_raw":scores["OOD_learn_minus_raw_native"]>=t["OOD_native_learned_minus_raw_min"],
          "OOD_cold_learn_vs_raw":scores["OOD_cold_learn_minus_raw_native"]>=t["OOD_cold_native_learned_minus_raw_min"],
          "OOD_learn_vs_strong_slot_GRU":scores["OOD_learn_minus_GRU256_slots"]>=t["OOD_native_learned_minus_trained_GRU256_slots_min"],
          "DEV_public_causal_sign_identifiable":scores["DEV_public_A_sign_accuracy_evaluator_only"]>=t["DEV_public_A_sign_identifiability_min"]}
        result["DEV"]=dm
        result["OOD_SHIFT"]=om
        result["comparisons"]=scores
        result["frozen_checks"]={k:bool(v) for k,v in checks.items()}
        result["status"]=t["on_positive"] if all(checks.values()) else t["on_negative"]
    except BaseException as e:
        result["error_type"]=type(e).__name__
        result["status"]="R89_INCOMPLETE_ENGINEERING_OR_RESOURCE_NOT_SCIENTIFIC_NEGATIVE"
        raise
    finally:
        result["total_process_wall_seconds"]=time.perf_counter()-begin
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open("x",encoding="utf-8") as o:
            json.dump(result,o,indent=2,sort_keys=True,allow_nan=False)
            o.write("\n")
    return result


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,choices=SEEDS,required=True)
    seed=p.parse_args().seed
    v=original_ci(seed)
    print(json.dumps({"seed":seed,"status":v["status"],"completed_arms":len(v["completed_arms"]),
        "DEV_OOD_first_use":v["DEV_OOD_FIRST_USE"],
        "scientific_pass":v["scientific_pass"]},sort_keys=True))
