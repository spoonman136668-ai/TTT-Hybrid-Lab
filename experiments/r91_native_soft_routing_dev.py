"""R91 native soft-routing multiworld TEST-only development comparison.

Development evidence, never independent science or official GPU training.
All worlds originate from immutable R89 TEST generator, indexes other than R90.
The evaluator's paired alternate action return is generated only after every
learner's train budget is complete. Never touch R89 original FIT/DEV/OOD.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiments import r85_delayed_composition as r85
from experiments import r89_native_event_access as r89
from experiments import r90_native_credit_path_diagnostic as r90

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/r91_native_soft_routing_dev_freeze.json"
FREEZE_BLOB = "75fe01dd3ee9fb9d3563d1e992282b9dd809fac3"
PARENT = "f3de743eb2fb11d21a4f6e6399d2353fbfb79194"
SEED = 1000211
FIT = tuple(range(64, 192))
DEV_TEST = tuple(range(256, 320))
BATCH = 2
ARMS = ("NATIVE_HARD_SLOTS5", "NATIVE_SOFT_SLOTS5",
        "NATIVE_SOFT_ROLE_READER48", "GRU64_SLOTS5",
        "GRU64_SHUFFLED_SLOTS5")
NATIVE = ARMS[:3]
UPDATES = {arm: (384 if arm in NATIVE else 4096) for arm in ARMS}
SCHEMA = "ttt-hybrid.r91-native-soft-routing-multiworld-dev-result.v1"
BRANCH = "research/r91-native-soft-routing-multiworld-dev-20261115"


def frozen() -> dict:
    if r85.git_blob(FREEZE.read_bytes()) != FREEZE_BLOB:
        raise RuntimeError("R91_FROZEN_DEVELOPMENT_SOURCE_DRIFT")
    r90.frozen()
    f = json.loads(FREEZE.read_text(encoding="utf-8"))
    src, tr, ev = f["sources"], f["training"], f["evaluation"]
    if (f["schema"] != "ttt-hybrid.r91-native-soft-routing-multiworld-dev-freeze.v1"
        or f["stage"] != "FROZEN_SOURCE_ONLY_BEFORE_R91_IMPLEMENTATION_TEST_OR_TEST_WORLD_FIRST_USE"
        or f["parent"]["R90_candidate_SHA"] != PARENT
        or f["parent"]["R90_freeze_blob"] != r90.FREEZE_BLOB
        or f["parent"]["R89_original_science"] != "R89_NATIVE_ROLE_ACCESS_NO_GO_BOTH_SEEDS"
        or src["test_train_indices"] != {"start":64,"stop_exclusive":192,"count":128}
        or src["test_dev_indices"] != {"start":256,"stop_exclusive":320,"count":64}
        or src["previous_R90_test_indexes"] != [0,1]
        or src["actions"] != [r85.CALIBRATE,r85.EXPLOIT_POS]
        or any(src[k] for k in ("protected_science_sources_used",
                               "prior_R85_to_R89_FIT_VALID_DEV_OOD_reuse",
                               "new_protected_independent_salts",
                               "hidden_law_or_oracle_training_labels",
                               "previous_trained_weights"))
        or f["models"]["native_reference_width"] != 1024
        or f["models"]["native_core_params"] != 16818178
        or f["models"]["arms"] != list(ARMS)
        or tr["seed"] != SEED or tr["mini_batch"] != BATCH
        or tr["native_updates_per_arm"] != 384
        or tr["GRU_updates_per_arm"] != 4096
        or tr["lr"] != .001 or tr["weight_decay"] != .0001
        or tr["grad_clip"] != 1 or tr["early_stopping"]
        or tr["existing_G4_full_regression_timeout_seconds"] != 600
        or ev["dev_worlds"] != len(DEV_TEST)
        or ev["paired_bootstrap_resamples"] != 1000
        or ev["bootstrap_seed_salt"] != 25329993
        or ev["soft_vs_hard_mean_reward_min"] != .5
        or ev["soft_vs_shuffled_mean_reward_min"] != .5
        or ev["soft_vs_GRU64_mean_reward_min"] != -.5
        or ev["soft_rule_agreement_min"] != .65
        or ev["soft_oracle_regret_max"] != 1.5
        or ev["independent_scientific_pass"]
        or not f["authority"]["existing_TEST_only_dev_training"]
        or any(v for k,v in f["authority"].items()
               if k != "existing_TEST_only_dev_training")):
        raise PermissionError("R91_SOURCE_OR_THRESHOLD_AUTHORITY_DRIFT")
    if (set(FIT) & set(DEV_TEST)) or set(FIT+DEV_TEST) & {0,1}:
        raise RuntimeError("R91_TEST_ONLY_WORLD_INDEX_OVERLAP")
    return f


def _chosen(index: int) -> int:
    if index not in FIT and index not in DEV_TEST:
        raise PermissionError("R91_UNREGISTERED_TEST_INDEX")
    rng = np.random.default_rng(np.random.SeedSequence([25329991,index]))
    return int(rng.choice([r85.CALIBRATE,r85.EXPLOIT_POS]))


def _shuffle_A(slots: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    if slots.shape != (len(indices),5,13) or len(indices) % 2:
        raise RuntimeError("R91_INVALID_SHUFFLE_GEOMETRY")
    # Deterministically replace only legal observed A rows, preserving
    # own chosen pivot action. Cross-world within each early/cold stratum.
    order = np.arange(len(indices))
    for parity in (0,1):
        eligible = np.asarray([i for i,n in enumerate(indices) if n % 2 == parity])
        if len(eligible) < 2:
            raise RuntimeError("R91_SHUFFLE_STRATUM_TOO_SMALL")
        order[eligible] = np.roll(eligible, 1)
    out = slots.copy()
    out[:,:4] = slots[order,:4]
    return out


def _fit() -> dict:
    slots, raw, rewards = [], [], []
    for index in FIT:
        world = r89._test_world(index)
        observed, actual_return = r89._view(world, _chosen(index))
        slots.append(r89._slots(observed))
        raw.append(observed)
        rewards.append(actual_return)
    s = np.stack(slots).astype(np.float32)
    return {"slots": s, "shuffled": _shuffle_A(s,FIT),
            "raw": np.stack(raw).astype(np.float32),
            "target": np.asarray(rewards,np.float32),
            "factual_chosen_only": True}


def _scored_test_after_freeze() -> dict:
    raw0,raw2,slots0,slots2,r0,r2,selected,truth = [],[],[],[],[],[],[],[]
    for index in DEV_TEST:
        world = r89._test_world(index)
        x0,v0 = r89._view(world,r85.CALIBRATE)
        x2,v2 = r89._view(world,r85.EXPLOIT_POS)
        if not np.array_equal(x0[:-1],x2[:-1]):
            raise RuntimeError("R91_EVALUATOR_PREFIX_CAUSAL_DRIFT")
        raw0.append(x0);raw2.append(x2)
        slots0.append(r89._slots(x0));slots2.append(r89._slots(x2))
        r0.append(v0);r2.append(v2)
        action = _chosen(index)
        selected.append(action)
        truth.append(v0 if action == r85.CALIBRATE else v2)
    s0 = np.stack(slots0).astype(np.float32)
    s2 = np.stack(slots2).astype(np.float32)
    return {
        "raw0":np.stack(raw0).astype(np.float32),
        "raw2":np.stack(raw2).astype(np.float32),
        "slots0":s0,"slots2":s2,
        "shuffled0":_shuffle_A(s0,DEV_TEST),
        "shuffled2":_shuffle_A(s2,DEV_TEST),
        "r0":np.asarray(r0,np.float64),
        "r2":np.asarray(r2,np.float64),
        "selected":np.asarray(selected,np.int64),
        "factual":np.asarray(truth,np.float32),
        "cold":np.asarray([i%2==1 for i in DEV_TEST],bool)}


def _models() -> dict[str,nn.Module]:
    original = r90._native(1024)
    if sum(p.numel() for p in original.parameters()) != 16818178:
        raise RuntimeError("R91_NATIVE_16M_ORIGINAL_CORE_DRIFT")
    hard = copy.deepcopy(original)
    soft = copy.deepcopy(original)
    soft.depth = r90.SoftDiagnosticRouter(soft.depth)
    soft_reader = copy.deepcopy(soft)
    torch.manual_seed(SEED + 301)
    soft_reader = r89.NativeLearnedRoleEventAccess(soft_reader)
    p0 = dict(original.named_parameters())
    for contender in (hard,soft,soft_reader.native):
        p = dict(contender.named_parameters())
        if p0.keys() != p.keys() or any(not torch.equal(t,p[k]) for k,t in p0.items()):
            raise RuntimeError("R91_NATIVE_CORE_INITIALIZATION_NOT_IDENTICAL")
    torch.manual_seed(SEED + 64)
    gru = r89.r81.GRU(64)
    shuffled = copy.deepcopy(gru)
    return dict(zip(ARMS,(hard,soft,soft_reader,gru,shuffled)))


def _source_x(src:dict,arm:str,suffix:str="") -> np.ndarray:
    field = ("raw" if arm==ARMS[2] else
             "shuffled" if arm==ARMS[4] else "slots")
    return src[field+suffix]


def _values(model:nn.Module,x:np.ndarray) -> np.ndarray:
    model.eval()
    arr=[]
    with torch.no_grad():
        for i in range(0,len(x),BATCH):
            y=model(torch.from_numpy(x[i:i+BATCH]))[:,-1,0]
            arr.extend(y.tolist())
    model.train()
    out=np.asarray(arr,np.float64)
    if out.shape != (len(x),) or not np.isfinite(out).all():
        raise RuntimeError("R91_INVALID_VALUE_PREDICTIONS")
    return out


def _factual_mse(model:nn.Module,src:dict,arm:str) -> float:
    y=src["target"]
    p=_values(model,_source_x(src,arm))
    return float(np.mean((p-y)**2))


def _train(model:nn.Module,arm:str,fit:dict,indices:np.ndarray) -> dict:
    x=torch.from_numpy(_source_x(fit,arm))
    y=torch.from_numpy(fit["target"])
    original_mse = _factual_mse(model,fit,arm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    steps=UPDATES[arm]
    began=time.perf_counter()
    for u in range(steps):
        pair=indices[u]
        estimate=model(x[pair])[:,-1,0]
        loss=F.mse_loss(estimate,y[pair])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    train_mse=_factual_mse(model,fit,arm)
    if not math.isfinite(train_mse):
        raise RuntimeError("R91_NONFINITE_FIT")
    return {"steps":steps,"parameter_count":sum(p.numel() for p in model.parameters()),
            "factual_selected_exposures":steps*BATCH,
            "initial_train_MSE":original_mse,"final_train_MSE":train_mse,
            "wall_seconds":round(time.perf_counter()-began,3)}


def _evaluate(models:dict,src:dict) -> tuple[dict,dict]:
    results, actions, chosen_reward = {},{},{}
    oracle=np.maximum(src["r0"],src["r2"])
    lawful=np.asarray([r85._policy(row) for row in src["raw0"]],np.int64)
    selected=src["selected"]
    for arm,model in models.items():
        v0=_values(model,_source_x(src,arm,"0"))
        v2=_values(model,_source_x(src,arm,"2"))
        selected_values=np.where(selected==r85.CALIBRATE,v0,v2)
        action=np.where(v0>v2,r85.CALIBRATE,r85.EXPLOIT_POS)
        realized=np.where(action==r85.CALIBRATE,src["r0"],src["r2"])
        actions[arm]=action
        chosen_reward[arm]=realized
        results[arm]={
            "DEV_TEST_mean_actual_reward":float(realized.mean()),
            "DEV_TEST_cold_actual_reward":float(realized[src["cold"]].mean()),
            "DEV_TEST_lawful_action_agreement":float(np.mean(action==lawful)),
            "DEV_TEST_oracle_regret":float((oracle-realized).mean()),
            "DEV_TEST_factual_MSE":float(np.mean((selected_values-src["factual"])**2))}
    for name,action in (("ALWAYS_CALIBRATE",r85.CALIBRATE),
                        ("ALWAYS_EXPLOIT_POS2",r85.EXPLOIT_POS)):
        vals=src["r0"] if action==r85.CALIBRATE else src["r2"]
        results[name]={"DEV_TEST_mean_actual_reward":float(vals.mean()),
                       "DEV_TEST_oracle_regret":float((oracle-vals).mean())}
    results["PUBLIC_A_RULE_EVALUATOR_ONLY"]={
        "DEV_TEST_mean_actual_reward":float(np.where(lawful==r85.CALIBRATE,src["r0"],src["r2"]).mean()),
        "DEV_TEST_lawful_action_agreement":1.0}
    results["PHYSICAL_ORACLE_EVALUATOR_ONLY"]={
        "DEV_TEST_mean_actual_reward":float(oracle.mean()),
        "DEV_TEST_oracle_regret":0.0}
    rng=np.random.default_rng(np.random.SeedSequence([25329993,SEED]))
    picks=rng.integers(0,len(DEV_TEST),(1000,len(DEV_TEST)))
    difference=chosen_reward[ARMS[1]]-chosen_reward[ARMS[0]]
    paired_lower=float(np.quantile(difference[picks].mean(axis=1),.025))
    s=results[ARMS[1]]
    checks={
        "soft_vs_hard_reward_at_least_0p5":s["DEV_TEST_mean_actual_reward"]-results[ARMS[0]]["DEV_TEST_mean_actual_reward"]>=.5,
        "soft_vs_shuffled_reward_at_least_0p5":s["DEV_TEST_mean_actual_reward"]-results[ARMS[4]]["DEV_TEST_mean_actual_reward"]>=.5,
        "soft_within_0p5_of_fully_trained_GRU64":s["DEV_TEST_mean_actual_reward"]-results[ARMS[3]]["DEV_TEST_mean_actual_reward"]>=-.5,
        "soft_lawful_agreement_at_least_65pct":s["DEV_TEST_lawful_action_agreement"]>=.65,
        "soft_oracle_regret_at_most_1p5":s["DEV_TEST_oracle_regret"]<=1.5}
    return results,{"checks":checks,
                   "soft_minus_hard_paired_bootstrap_lower95":paired_lower,
                   "soft_minus_hard_reward":s["DEV_TEST_mean_actual_reward"]-results[ARMS[0]]["DEV_TEST_mean_actual_reward"],
                   "soft_minus_GRU_reward":s["DEV_TEST_mean_actual_reward"]-results[ARMS[3]]["DEV_TEST_mean_actual_reward"]}


def development_ci() -> dict:
    frozen()
    torch.set_num_threads(1)
    start=time.perf_counter()
    receipt=ROOT/"local_runs/r91_native_soft_routing_dev.json"
    # Guard accidental reruns or overwritten receipts within the same checkout.
    if receipt.exists():
        raise FileExistsError("R91_EXISTING_DIAGNOSTIC_RECEIPT_REPLAY_DENIED")
    result={"schema":SCHEMA,"freeze_blob":FREEZE_BLOB,
            "parent_R90_SHA":PARENT,"candidate_SHA":os.getenv("EXPECTED_CANDIDATE_SHA"),
            "TEST_only_fit_indices":[FIT[0],FIT[-1]],
            "TEST_only_dev_indices":[DEV_TEST[0],DEV_TEST[-1]],
            "original_R89_spent_or_sealed_source_used":False,
            "real_PROTECTED_scientific_first_use":False,
            "all_weights_frozen_before_DEV_TEST":False,
            "completed_arms":{},"scientific_pass":False,
            "status":"R91_INCOMPLETE_ENGINEERING_RESOURCE_NOT_NEGATIVE",
            "platform":platform.platform(),"torch":str(torch.__version__)}
    try:
        fit=_fit()
        if not fit["factual_chosen_only"]:
            raise RuntimeError("R91_NOT_FACTUAL_SELECTED_RETURN")
        models=_models()
        gen=np.random.default_rng(np.random.SeedSequence([25329992,SEED]))
        indices=gen.integers(0,len(FIT),(max(UPDATES.values()),BATCH))
        for arm in ARMS:
            result["completed_arms"][arm]=_train(models[arm],arm,fit,indices)
        if set(result["completed_arms"]) != set(ARMS):
            raise RuntimeError("R91_INCOMPLETE_MODELS")
        result["all_weights_frozen_before_DEV_TEST"]=True
        # Test-only heldout worlds may be generated just once after weights freeze;
        # never accessed by the optimizer or used for post-score retuning.
        dev=_scored_test_after_freeze()
        metrics,qualification=_evaluate(models,dev)
        result["DEV_TEST"]=metrics
        result["comparisons"]=qualification
        result["status"]=("R91_SOFT_ROUTING_DEV_SIGNAL_ONLY"
                          if all(qualification["checks"].values())
                          else "R91_SOFT_ROUTING_DEV_NO_GO")
    except BaseException as exc:
        result["error_type"]=type(exc).__name__
        raise
    finally:
        result["process_wall_seconds"]=round(time.perf_counter()-start,3)
        receipt.parent.mkdir(parents=True,exist_ok=True)
        with receipt.open("x",encoding="utf-8") as f:
            json.dump(result,f,indent=2,sort_keys=True,allow_nan=False)
            f.write("\n")
    return result


if __name__ == "__main__":
    print(json.dumps({k:v for k,v in development_ci().items()
                      if k in ("schema","status","completed_arms","process_wall_seconds","scientific_pass")},
                     sort_keys=True,allow_nan=False))
