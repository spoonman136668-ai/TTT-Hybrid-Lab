"""R88: original frozen public role-conditioned causal event access experiment.

Not a native Hybrid improvement. First-use independent source only in exact SHA,
first attempt, GitHub-hosted Ubuntu CPU job AFTER original G4 full Linux+Windows.
All training factual chosen-action returns only; DEV/OOD evaluator first-used
after ALL six models' optimizer updates are complete. Prior panels untouched.
"""
from __future__ import annotations

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

from experiments import r85_delayed_composition as r85
from experiments import r86_native_compositional_value as r86
from experiments import r87_public_composition_diagnostic as r87
from experiments import r81_paired_history as r81

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r88_causal_event_access_freeze.json"
FREEZE_BLOB = "bb465029814c567c7391f0be5259b046490fecf7"
SCHEMA = "ttt-hybrid.r88-public-causal-event-access-v1"
PARENT = "624c3f07a6e3bbb59007233aeeb3ac54fc6099d5"
BRANCH = "research/r88-fresh-causal-event-access-20261115"
SEEDS = (1000011, 1000023)
SALT = {"FIT": 25300111, "FIT_VALIDATION": 25300211,
        "DEV": 25300311, "OOD_SHIFT": 25300411}
COUNT = {"FIT": 2048, "FIT_VALIDATION": 256, "DEV": 256, "OOD_SHIFT": 256}
ARMS = ("GRU64_FULL_PUBLIC_RAW", "GRU64_A_PHASE_MASKED",
        "GRU64_PUBLIC_A_SLOTS", "GRU64_PUBLIC_A_SLOTS_SHUFFLED",
        "LEARNED_PUBLIC_ROLE_ATTENTION", "MLP64_LEGAL_A_FACTORS")
GRUS = ARMS[:4]
LEARNED = "LEARNED_PUBLIC_ROLE_ATTENTION"
BASE = "GRU64_FULL_PUBLIC_RAW"
MASK = "GRU64_A_PHASE_MASKED"
SLOTS = "GRU64_PUBLIC_A_SLOTS"
SHUFFLE = "GRU64_PUBLIC_A_SLOTS_SHUFFLED"
MLP = "MLP64_LEGAL_A_FACTORS"
BATCH = 16
STEPS = 4096


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R88_ORIGINAL_IMMUTABLE_FREEZE_BLOB_DRIFT")
    r87.frozen()
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    p, s, t, e = f["parent"], f["source"], f["training"], f["evaluation"]
    if (f["schema"] != SCHEMA
        or f["stage"] != "IMMUTABLE_PROSPECTIVE_FREEZE_ONLY_BEFORE_R88_IMPLEMENTATION_TEST_CI_OR_FRESH_SOURCE"
        or p["R87_source_sha"] != PARENT
        or p["R87_freeze_blob"] != r87.FREEZE_BLOB
        or p["R87_science"] != "R87_COMPOSITION_BOTTLENECK_NO_GO_BOTH_SEEDS_7_OF_10_FROZEN_CHECKS_PASS"
        or s["independent_salts"] != SALT or s["seeds"] != list(SEEDS)
        or s["counts"] != COUNT or s["full_public_history_selected_rows"] != 48
        or f["arms"] != list(ARMS)
        or t["all_arms_steps"] != STEPS or t["batch"] != BATCH
        or e["bootstrap"]["resamples"] != 2000
        or e["thresholds"]["DEV_attention_minus_raw_GRU_min"] != .75
        or e["thresholds"]["OOD_attention_cold_minus_raw_GRU_min"] != .50
        or any(f["authority"].values())):
        raise PermissionError("R88_FROZEN_SOURCE_MODELS_OR_THRESHOLDS_DRIFT")
    old = (set(p["R87_spent_salts"]) | set(p["R86_spent_salts"])
           | set(p["R85_spent"]) | set(p["R85_reserved_untouched"]))
    if set(SALT.values()) & old or len(set(SALT.values())) != len(SALT):
        raise RuntimeError("R88_ORIGINAL_SALT_OR_SEALED_SOURCE_OVERLAP")
    return f


def _admit(seed: int) -> str:
    frozen()
    if (seed not in SEEDS or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_EVENT_NAME") != "pull_request"
        or os.getenv("GITHUB_REPOSITORY") != "spoonman136668-ai/TTT-Hybrid-Lab"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("GITHUB_JOB") != "r88-causal-event-access"
        or os.getenv("R88_SEED") != str(seed)
        or os.getenv("RUNNER_OS") != "Linux"
        or os.getenv("RUNNER_ENVIRONMENT") != "github-hosted"
        or os.getenv("GITHUB_RUN_ATTEMPT") != "1"
        or "/merge" not in os.getenv("GITHUB_REF", "")):
        raise PermissionError("R88_FIRST_ATTEMPT_ORIGINAL_G4_HOSTED_JOB_REQUIRED")
    sha = os.getenv("EXPECTED_CANDIDATE_SHA", "")
    checked = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
        capture_output=True, text=True, check=False, timeout=10)
    dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--"], cwd=ROOT,
        capture_output=True, check=False, timeout=10)
    if len(sha) != 40 or checked.returncode or checked.stdout.strip() != sha or dirty.returncode:
        raise PermissionError("R88_EXACT_SHA_CLEAN_WORKTREE_CUSTODY_FAIL")
    return sha


def _world(panel: str, seed: int, index: int, *, custody_sha: str | None = None):
    if panel not in SALT:
        raise PermissionError("R88_OLD_OR_SEALED_PANEL_DENIED")
    if seed not in SEEDS or not isinstance(index, int) or not 0 <= index < COUNT[panel]:
        raise PermissionError("R88_NONFROZEN_OR_OUT_OF_RANGE_PANEL_INDEX")
    if (not custody_sha or custody_sha != os.getenv("EXPECTED_CANDIDATE_SHA")
        or os.getenv("GITHUB_ACTIONS") != "true"
        or os.getenv("GITHUB_JOB") != "r88-causal-event-access"
        or os.getenv("GITHUB_HEAD_REF") != BRANCH
        or os.getenv("R88_SEED") != str(seed)):
        raise PermissionError("R88_ORIGINAL_SOURCE_NOT_ADMITTED")
    # Pure original physical constructor, NEW RNG and NEW panels only.
    rng = np.random.default_rng(np.random.SeedSequence([SALT[panel], seed, index]))
    w = r85._make_synthetic_world(index % r85.N, rng)
    if panel == "OOD_SHIFT":
        w = r86._shift_public_world(w, index)
    return w, int(rng.choice([r85.CALIBRATE, r85.EXPLOIT_POS]))


def _test_only_world(index: int) -> dict:
    rng = np.random.default_rng(np.random.SeedSequence([25309991, index]))
    return r85._make_synthetic_world(index, rng)


def _view(w: dict, chosen: int):
    return r86._view(w, chosen)


def _public_slots(prefix: np.ndarray) -> np.ndarray:
    """Exactly four actually observed A PROBE *rows*, plus current action.

    The input includes public 13D raw observations and physical timestamps,
    not the hidden A law, unchosen result, or an engineered parity product.
    """
    if prefix.dtype != np.float32 or prefix.shape != (48, 13):
        raise ValueError("R88_ILLEGAL_PUBLIC_PREFIX_FORMAT")
    rows = prefix[(prefix[:, 5] == 1) & (prefix[:, 11] == 1)]
    if len(rows) != 4:
        raise RuntimeError("R88_FOUR_ACTUAL_A_PROBE_RESPONSES_REQUIRED")
    events = np.zeros((5, 13), dtype=np.float32)
    seen = set()
    for row in rows:
        role = int(round(float(row[9])*4))-1
        if role not in range(4) or role in seen:
            raise RuntimeError("R88_INVALID_OR_DUPLICATE_PUBLIC_ROLE")
        seen.add(role)
        events[role] = row
    if seen != {0,1,2,3}:
        raise RuntimeError("R88_MISSING_PUBLIC_ROLE")
    events[-1] = prefix[-1]
    return events


def _public_A_mask(prefix: np.ndarray) -> np.ndarray:
    if prefix.dtype != np.float32 or prefix.shape != (48,13):
        raise ValueError("R88_INVALID_OR_PRIVILEGED_PUBLIC_INPUT")
    a_probes = (prefix[:,5] == 1) & (prefix[:,11] == 1)
    if int(a_probes.sum()) != 4:
        raise RuntimeError("R88_INCOMPLETE_PUBLIC_A_SIGNAL")
    out = prefix.copy()
    out[~a_probes,8:12] = 0
    return out


def _shuffle_slots(slots: np.ndarray, seed: int, panel: str) -> np.ndarray:
    if panel not in SALT or slots.shape != (COUNT[panel],5,13):
        raise PermissionError("R88_SHUFFLE_ONLY_FROZEN_SOURCE_SHAPE")
    times = slots[:,4,12]
    if len(np.unique(times)) != 2:
        raise RuntimeError("R88_SHUFFLE_EARLY_COLD_PARITY_STRATA_LOST")
    rng = np.random.default_rng(np.random.SeedSequence([25300555,seed,SALT[panel]]))
    order = np.arange(len(slots))
    for tm in np.unique(times):
        group = np.flatnonzero(times == tm)
        order[group] = rng.permutation(group)
    out = slots.copy()
    out[:,:4,:] = slots[order,:4,:]
    return out


def _assemble(panel: str, seed: int, *, custody_sha: str):
    if panel not in ("FIT", "FIT_VALIDATION"):
        raise PermissionError("R88_TRAIN_DATA_CANNOT_INCLUDE_DEV_OOD")
    n = COUNT[panel]
    raw = np.zeros((n,48,13),np.float32)
    masked = np.zeros_like(raw)
    slots = np.zeros((n,5,13),np.float32)
    features = np.zeros((n,8),np.float32)
    targets = np.zeros(n,np.float32)
    for i in range(n):
        w, chosen = _world(panel, seed, i, custody_sha=custody_sha)
        x, ret = _view(w,chosen)
        raw[i], masked[i] = x, _public_A_mask(x)
        slots[i] = _public_slots(x)
        features[i] = r87._legal_features(x)[0]  # Public raw factor extraction; no parity.
        targets[i] = ret
    return {"raw":raw, "masked":masked, "slots":slots,
            "shuffled_slots":_shuffle_slots(slots,seed,panel),
            "features":features,"target":targets,"one_factual_return_only":True}


def _paired(panel: str, seed: int, *, custody_sha: str):
    if panel not in ("DEV","OOD_SHIFT"):
        raise PermissionError("R88_ONLY_NEW_HELDOUT_EVALUATOR_SOURCE")
    n = COUNT[panel]
    x0, x2 = np.zeros((n,48,13),np.float32),np.zeros((n,48,13),np.float32)
    s0, s2 = np.zeros((n,5,13),np.float32),np.zeros((n,5,13),np.float32)
    f0, f2 = np.zeros((n,8),np.float32),np.zeros((n,8),np.float32)
    r0,r2 = np.zeros(n,np.float64),np.zeros(n,np.float64)
    cold = np.zeros(n,bool)
    lawful = np.zeros(n,np.int64)
    private = np.zeros(n,np.int64)
    for i in range(n):
        w,_ = _world(panel,seed,i,custody_sha=custody_sha)
        a,ra = _view(w,r85.CALIBRATE)
        b,rb = _view(w,r85.EXPLOIT_POS)
        if not np.array_equal(a[:-1],b[:-1]) or not np.array_equal(a[-1,4:],b[-1,4:]):
            raise RuntimeError("R88_PAIRED_INTERVENTION_PUBLIC_PREFIX_DRIFT")
        if a[-1,0] != 1 or b[-1,2] != 1:
            raise RuntimeError("R88_CURRENT_INTERVENTION_NOT_REALIZED")
        x0[i],x2[i],s0[i],s2[i]=a,b,_public_slots(a),_public_slots(b)
        f0[i],f2[i]=r87._legal_features(a)[0],r87._legal_features(b)[0]
        r0[i],r2[i],cold[i]=ra,rb,bool(i%2)
        lawful[i],private[i]=r85._policy(a),w["law_a"]
    sh0,sh2 = _shuffle_slots(s0,seed,panel),_shuffle_slots(s2,seed,panel)
    return {"raw0":x0,"raw2":x2,"masked0":np.stack([_public_A_mask(x) for x in x0]),
            "masked2":np.stack([_public_A_mask(x) for x in x2]),
            "slots0":s0,"slots2":s2,"shuffled0":sh0,"shuffled2":sh2,
            "features0":f0,"features2":f2,
            "r0":r0,"r2":r2,"cold":cold,"lawful_actions":lawful,
            "private_A_law_evaluator_only":private}


class LearnedPublicRoleEventValue(nn.Module):
    """Public A-PROBE event-gated, differentiable role-query attention/value.

    The four public role ids and phase marker supply an architectural prior.
    The keys/queries/event content + downstream composition are trainable only
    against chosen factual returns. This is NOT discovery without structure.
    """
    def __init__(self):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(13,64),nn.GELU(),nn.Linear(64,64))
        self.queries = nn.Parameter(torch.randn(4,64)*0.02)
        self.role_penalty_raw = nn.Parameter(torch.tensor(64.0))
        self.pivot_projection = nn.Linear(13,64)
        self.gru = nn.GRU(64,64,batch_first=True)
        self.head = nn.Linear(64,1)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        if x.ndim != 3 or x.shape[1:]!=(48,13):
            raise ValueError("R88_LEARNED_ATTENTION_PUBLIC_SEQUENCE_WRONG_SHAPE")
        is_A=(x[:,:,5]>0.5)&(x[:,:,11]>0.5)
        if not bool((is_A.sum(dim=1)==4).all()):
            raise RuntimeError("R88_LEARNED_GATE_MUST_SEE_FOUR_PUBLIC_A_PROBES")
        h=self.embed(x)
        qlogits=torch.einsum("btd,rd->brt",h,self.queries)/(64**.5)
        role_ids=x[:,:,9]
        targets=x.new_tensor([.25,.5,.75,1.0])
        dist=(role_ids[:,None,:]-targets[None,:,None]).square()
        scores=qlogits-F.softplus(self.role_penalty_raw)*dist
        scores=scores.masked_fill(~is_A[:,None,:],-1e4)
        weights=scores.softmax(dim=2)
        slots=torch.bmm(weights,h)
        final=self.pivot_projection(x[:,-1])[:,None,:]
        seq=torch.cat([slots,final],dim=1)
        return self.head(self.gru(seq)[0][:,-1])[:,0]


def _models(seed:int)->dict:
    models={}
    for i,arm in enumerate(ARMS):
        torch.manual_seed(seed+200+i)
        if arm in GRUS:
            models[arm]=r81.GRU(64)
        elif arm==LEARNED:
            models[arm]=LearnedPublicRoleEventValue()
        else:
            models[arm]=r87.PublicValueMLP()
    counts=[sum(p.numel() for p in models[arm].parameters()) for arm in GRUS]
    if len(set(counts))!=1:
        raise RuntimeError("R88_FULL_MASKED_SLOT_AND_SHUFFLE_GRU_PARAM_MISMATCH")
    return models


def _data(source:dict,arm:str)->np.ndarray:
    return {"GRU64_FULL_PUBLIC_RAW":source["raw"],
            "GRU64_A_PHASE_MASKED":source["masked"],
            "GRU64_PUBLIC_A_SLOTS":source["slots"],
            "GRU64_PUBLIC_A_SLOTS_SHUFFLED":source["shuffled_slots"],
            LEARNED:source["raw"],MLP:source["features"]}[arm]


def _prediction(model,arm:str,x:torch.Tensor)->torch.Tensor:
    y=model(x)
    return y[:,-1,0] if arm in GRUS else y


def _val(model,arm,valid):
    model.eval()
    xx=torch.from_numpy(_data(valid,arm))
    yy=torch.from_numpy(valid["target"])
    errors=[]
    with torch.no_grad():
        for i in range(0,len(xx),BATCH):
            y=_prediction(model,arm,xx[i:i+BATCH])
            errors.extend((y-yy[i:i+BATCH]).square().tolist())
    model.train()
    return float(np.mean(errors))


def _train(model,arm,fit,valid,indexes):
    xx=torch.from_numpy(_data(fit,arm))
    yy=torch.from_numpy(fit["target"])
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    checkpoints={"0":_val(model,arm,valid)}
    began=time.perf_counter()
    for step in range(1,STEPS+1):
        ix=indexes[step-1]
        pred=_prediction(model,arm,xx[ix])
        loss=F.mse_loss(pred,yy[ix])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if step in (1024,2048,4096):
            checkpoints[str(step)]=_val(model,arm,valid)
    elapsed=time.perf_counter()-began
    if any(not math.isfinite(v) for v in checkpoints.values()) or any(
        not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise RuntimeError("R88_NONFINITE_OR_INCOMPLETE_ORIGINAL_TRAINING")
    return {"steps":STEPS,"parameters":sum(p.numel() for p in model.parameters()),
        "factual_selected_event_exposures":STEPS*BATCH,
        "same_factual_FIT_worlds":COUNT["FIT"],
        "validation_MSE":checkpoints,"training_wall_seconds":elapsed,
        "recurrent_unroll":48 if arm in (BASE,MASK) else 5 if arm in GRUS else None,
        "explicit_A_probe_gate":arm in (MASK,SLOTS,SHUFFLE,LEARNED)}


def _eval_data(paired:dict,arm:str,chosen:int)->np.ndarray:
    key0={"GRU64_FULL_PUBLIC_RAW":"raw","GRU64_A_PHASE_MASKED":"masked",
        "GRU64_PUBLIC_A_SLOTS":"slots","GRU64_PUBLIC_A_SLOTS_SHUFFLED":"shuffled",
        LEARNED:"raw",MLP:"features"}[arm]
    return paired[key0+("0" if chosen==r85.CALIBRATE else "2")]


def _predict_array(model,arm,x:np.ndarray)->np.ndarray:
    model.eval()
    values=[]
    with torch.no_grad():
        for i in range(0,len(x),BATCH):
            values.extend(_prediction(model,arm,torch.from_numpy(x[i:i+BATCH])).tolist())
    model.train()
    a=np.asarray(values,np.float64)
    if a.shape!=(len(x),) or not np.isfinite(a).all():
        raise RuntimeError("R88_NONFINITE_LEGAL_VALUE_AT_FIRSTUSE")
    return a


def _score(models:dict,p:dict)->dict:
    actions={}
    for arm,model in models.items():
        v0=_predict_array(model,arm,_eval_data(p,arm,r85.CALIBRATE))
        v2=_predict_array(model,arm,_eval_data(p,arm,r85.EXPLOIT_POS))
        actions[arm]=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
    actions["PUBLIC_A_PARITY_RULE_EVALUATOR_ONLY"]=p["lawful_actions"]
    actions["ALWAYS_CALIBRATE"]=np.full(len(p["r0"]),r85.CALIBRATE,np.int64)
    actions["ALWAYS_EXPLOIT_POS"]=np.full(len(p["r0"]),r85.EXPLOIT_POS,np.int64)
    actions["LAST_A_CUE"]=np.asarray([
        r85._policy(x,"LAST_A") for x in p["raw0"]],np.int64)
    actions["RECENT_B_PARITY"]=np.asarray([
        r85._policy(x,"RECENT_B") for x in p["raw0"]],np.int64)
    rewards={arm:np.where(a==r85.CALIBRATE,p["r0"],p["r2"])
             for arm,a in actions.items()}
    rewards["PHYSICAL_ORACLE_EVALUATOR_ONLY"]=np.maximum(p["r0"],p["r2"])
    cold=p["cold"]
    metrics={arm:{"mean_reward":float(v.mean()),
                  "cold_reward":float(v[cold].mean()),
                  "early_reward":float(v[~cold].mean()),
                  "oracle_regret":float((rewards["PHYSICAL_ORACLE_EVALUATOR_ONLY"]-v).mean())}
             for arm,v in rewards.items()}
    legal_A_accuracy=float(np.mean(
        np.where(np.asarray([r87._legal_features(x)[1] for x in p["raw0"]])>0,1,-1)
        ==p["private_A_law_evaluator_only"]))
    return {"metrics":metrics,"rewards":rewards,"actions":actions,
            "legal_A_sign_accuracy_evaluator_only":legal_A_accuracy}


def _bootstrap_lower(raw:np.ndarray,attn:np.ndarray,seed:int)->float:
    if raw.shape!=(COUNT["DEV"],) or attn.shape!=raw.shape:
        raise RuntimeError("R88_ORIGINAL_PAIRED_BOOTSTRAP_SHAPE_DRIFT")
    rng=np.random.default_rng(np.random.SeedSequence([25300555,seed]))
    idx=rng.integers(0,len(raw),(2000,len(raw)))
    return float(np.quantile((attn-raw)[idx].mean(axis=1),.025))


def original_ci(seed:int):
    sha=_admit(seed)
    torch.set_num_threads(1)
    receipt=ROOT/f"local_runs/r88_causal_event_seed{seed}.json"
    if receipt.exists():
        raise FileExistsError("R88_ORIGINAL_REPLAY_DENIED")
    started=time.perf_counter()
    result={"schema":SCHEMA,"source_SHA":sha,"freeze_blob":FREEZE_BLOB,
        "R87_parent_SHA":PARENT,"seed":seed,"completed_arms":{},
        "all_models_frozen_before_DEV_OOD":False,"DEV_OOD_FIRST_USE":False,
        "old_or_reserved_salts_used":False,"scientific_pass":False,
        "status":"R88_INCOMPLETE_ENGINEERING_OR_RESOURCE_NOT_SCIENTIFIC_NEGATIVE",
        "runner":platform.platform()}
    try:
        fit=_assemble("FIT",seed,custody_sha=sha)
        val=_assemble("FIT_VALIDATION",seed,custody_sha=sha)
        if not fit["one_factual_return_only"]:
            raise RuntimeError("R88_SELECTED_FIT_DATA_NOT_FACTUAL")
        models=_models(seed)
        rng=np.random.default_rng(np.random.SeedSequence([25300611,seed]))
        index=rng.integers(0,COUNT["FIT"],size=(STEPS,BATCH))
        for arm in ARMS:
            result["completed_arms"][arm]=_train(models[arm],arm,fit,val,index)
        if set(result["completed_arms"])!=set(ARMS):
            raise RuntimeError("R88_MISSING_FROZEN_TRAINED_ARM")
        result["all_models_frozen_before_DEV_OOD"]=True
        result["DEV_OOD_FIRST_USE"]=True
        dev=_paired("DEV",seed,custody_sha=sha)
        ood=_paired("OOD_SHIFT",seed,custody_sha=sha)
        d=_score(models,dev)
        o=_score(models,ood)
        md,mo=d["metrics"],o["metrics"]
        a,b,m,s,sh=md[LEARNED],md[BASE],md[MASK],md[SLOTS],md[SHUFFLE]
        agree=float(np.mean(d["actions"][LEARNED]==dev["lawful_actions"]))
        lower=_bootstrap_lower(d["rewards"][BASE],d["rewards"][LEARNED],seed)
        best_fixed=max(md["ALWAYS_CALIBRATE"]["mean_reward"],
                       md["ALWAYS_EXPLOIT_POS"]["mean_reward"])
        comparisons={
            "DEV_attention_minus_raw_GRU":a["mean_reward"]-b["mean_reward"],
            "DEV_attention_minus_masked_GRU":a["mean_reward"]-m["mean_reward"],
            "DEV_attention_minus_shuffled_slots":a["mean_reward"]-sh["mean_reward"],
            "DEV_attention_minus_best_fixed":a["mean_reward"]-best_fixed,
            "DEV_attention_cold_minus_raw_GRU":a["cold_reward"]-b["cold_reward"],
            "DEV_attention_agreement_with_public_rule":agree,
            "DEV_attention_oracle_regret":a["oracle_regret"],
            "DEV_paired_bootstrap_lower95_attention_minus_raw":lower,
            "DEV_deterministic_slots_minus_raw_GRU":s["mean_reward"]-b["mean_reward"],
            "DEV_deterministic_slots_minus_shuffled_slots":s["mean_reward"]-sh["mean_reward"],
            "OOD_attention_minus_raw_GRU":mo[LEARNED]["mean_reward"]-mo[BASE]["mean_reward"],
            "OOD_attention_cold_minus_raw_GRU":mo[LEARNED]["cold_reward"]-mo[BASE]["cold_reward"],
            "DEV_legal_A_sign_identifiability":d["legal_A_sign_accuracy_evaluator_only"],
            "DEV_MLP_extracted_A_baseline":md[MLP]["mean_reward"],
            "DEV_deterministic_slots_baseline":s["mean_reward"]}
        t=frozen()["evaluation"]["thresholds"]
        checks={
            "DEV_attn_beats_raw":comparisons["DEV_attention_minus_raw_GRU"]>=t["DEV_attention_minus_raw_GRU_min"],
            "DEV_attn_beats_phase_mask":comparisons["DEV_attention_minus_masked_GRU"]>=t["DEV_attention_minus_masked_GRU_min"],
            "DEV_attn_beats_shuffle":comparisons["DEV_attention_minus_shuffled_slots"]>=t["DEV_attention_minus_shuffled_A_slots_min"],
            "DEV_attn_beats_fixed":comparisons["DEV_attention_minus_best_fixed"]>=t["DEV_attn_minus_best_fixed_min"],
            "DEV_cold_attn_beats_raw":comparisons["DEV_attention_cold_minus_raw_GRU"]>=t["DEV_attn_cold_minus_raw_GRU_min"],
            "DEV_attn_matches_public_rule":agree>=t["DEV_attention_legal_rule_choice_agreement_min"],
            "DEV_attn_oracle_regret":a["oracle_regret"]<=t["DEV_attention_oracle_regret_max"],
            "DEV_bootstrap_attn_beats_raw":lower>0,
            "DEV_deterministic_slots_beats_raw":comparisons["DEV_deterministic_slots_minus_raw_GRU"]>=t["DEV_deterministic_slots_minus_raw_GRU_min"],
            "DEV_deterministic_slots_beats_shuffle":comparisons["DEV_deterministic_slots_minus_shuffled_slots"]>=t["DEV_deterministic_slots_minus_shuffled_slots_min"],
            "OOD_attn_beats_raw":comparisons["OOD_attention_minus_raw_GRU"]>=t["OOD_attention_minus_raw_GRU_min"],
            "OOD_cold_attn_beats_raw":comparisons["OOD_attention_cold_minus_raw_GRU"]>=t["OOD_attention_cold_minus_raw_GRU_min"],
            "DEV_legal_cue_identifiable":comparisons["DEV_legal_A_sign_identifiability"]>=t["DEV_legal_A_sign_identifiability_min"]}
        result["DEV"]=md
        result["OOD_SHIFT"]=mo
        result["comparison"]=comparisons
        result["frozen_checks"]={k:bool(v) for k,v in checks.items()}
        result["status"]=t["on_pass"] if all(checks.values()) else t["on_fail"]
    except BaseException as exc:
        result["error_class"]=type(exc).__name__
        result["status"]="R88_INCOMPLETE_ENGINEERING_OR_RESOURCE_NOT_SCIENTIFIC_NEGATIVE"
        raise
    finally:
        result["total_wall_seconds"]=time.perf_counter()-started
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return result


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--seed",type=int,required=True,choices=SEEDS)
    s=parser.parse_args().seed
    z=original_ci(s)
    print(json.dumps({"seed":s,"source_SHA":z["source_SHA"],
        "completed_arms":len(z["completed_arms"]),
        "DEV_OOD_FIRST_USE":z["DEV_OOD_FIRST_USE"],
        "scientific_pass":False,"status":z["status"]},sort_keys=True))
