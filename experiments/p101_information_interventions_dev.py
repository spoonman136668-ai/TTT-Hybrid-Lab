"""P101 prospective information-directed intervention acquisition on causal worlds.

Author-synthetic DEV only; no OOD, external simulator or original-model edits.
Frozen *before* implementation at 2f27ebcf70b06a27b2182b11e3473fda1b62a55d.
Different from P97-P100 context-switch mechanisms: policies choose ACTIONS;
all policies share an exactly identical causal learner and 16-effect budget.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import time
from typing import NamedTuple

import torch
from torch.nn import functional as F

from experiments import p86_portable_causal_preflight as p86

D = 16
FREEZE_SHA = "2f27ebcf70b06a27b2182b11e3473fda1b62a55d"
FREEZE_BLOB = "17ec9ef4319b9274e8bd2d28c6a8743f168e1e5d"
FAMILIES = ("isotropic", "anisotropic")
POLICIES = ("max_variance", "hash_random", "cyclic", "max_norm")
PASSIVES = POLICIES[1:]
SEEDS = {
    "isotropic": {"FIT": (1010111, 1010113, 1010119),
                  "CAL": (1010127, 1010131, 1010133)},
    "anisotropic": {"FIT": (1010211, 1010217, 1010223),
                    "CAL": (1010227, 1010229, 1010233)},
}
RESERVED = (1010401, 1010403, 1010501, 1010503)
COUNTS = {"FIT": 24, "CAL": 12}
STEPS, CHOICES, PROBES = 8, 8, 12
RLS_PRIOR = 0.5
EXPECTED_BLOBS = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p97_recurrent_context_dev.py":
        "74030c3d7d2f5f51781c740493be21d3c769792d",
    "experiments/p98_blind_switch_diagnostics.py":
        "7cd9521c2ceaa6f84f2128ba16d37ceaae7ff006",
    "experiments/p99_event_checkpoint_dev.py":
        "23279971d2f7531716c349991d6fd56095656c9e",
}


def pinned_protocol() -> dict:
    root = Path(__file__).resolve().parents[1]
    raw = (root / "experiments/p101_information_interventions_frozen.json").read_bytes()
    h = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if h != FREEZE_BLOB:
        raise ValueError("P101_ORIGINAL_FROZEN_BLOB_DRIFT")
    data = json.loads(raw)
    if (data["schema"] !=
        "ttt-hybrid.p101-information-directed-causal-intervention-dev-freeze.v1"
        or data["identity"]["parent_sha"] !=
           "5f932c24f37c396e7d1d0f89f40321303a14be24"
        or data["source"]["seeds"] != {
           **{k:{s:list(v) for s,v in x.items()} for k,x in SEEDS.items()},
           "OOD_RESERVED":[1010401,1010403], "NEGATIVE_RESERVED":[1010501,1010503]}
        or data["source"]["worlds_per_seed"] != COUNTS
        or data["source"]["interventions_per_task"] != STEPS
        or data["source"]["candidate_actions_per_intervention"] != CHOICES
        or data["source"]["query_actions_per_task"] != PROBES
        or data["learner"]["prior_lambda"] != RLS_PRIOR
        or data["acceptance"]["candidate_vs_strongest_passive_primary_ratio_max"] != .85
        or data["acceptance"]["require_per_family_median_ratio_max"] != .85):
        raise ValueError("P101_PROTOCOL_DRIFT")
    for rel, expected in EXPECTED_BLOBS.items():
        if p86.blob(root / rel) != expected:
            raise ValueError("P101_P99_ORIGINAL_CUSTODY_DRIFT")
    return data


class World(NamedTuple):
    # These two matrices remain within the environment; never passed to chooser.
    maps: tuple[torch.Tensor, torch.Tensor]
    raw: torch.Tensor       # [2,8,8,16] action opportunity sets
    pool: torch.Tensor      # [2,8,8,16] normalized action opportunity sets
    queries: torch.Tensor   # [2,12,16] EVALUATOR ONLY, not visible to chooser


def build_world(family: str, split: str, index: int, world: int) -> World:
    if (family not in FAMILIES or split not in COUNTS
        or type(index) is not int or index not in (0,1,2)
        or type(world) is not int or world < 0 or world >= COUNTS[split]):
        raise ValueError("P101_PROTECTED_OR_UNKNOWN_SOURCE_DENIED")
    seed = SEEDS[family][split][index]
    m = []
    for p in range(2):
        u = torch.tensor(p86.basis(seed, world, "p101-u" + str(p)),
                         dtype=torch.float64)
        v = torch.tensor(p86.basis(seed, world, "p101-v" + str(p)),
                         dtype=torch.float64)
        m.append((v @ u.T) / 256.)
    pools = []
    questions = []
    for task in range(2):
        choices = []
        for t in range(STEPS):
            actions = []
            for option in range(CHOICES):
                coords = []
                for dim in range(D):
                    value = p86.digest_int(
                        seed, world, "p101-action-"+str(task),
                        t*CHOICES+option, dim, 193) - 96
                    if family == "anisotropic":
                        value = value // (2 ** (dim % 4))
                    coords.append(value/128.)
                actions.append(coords)
            choices.append(actions)
        raw = torch.tensor(choices, dtype=torch.float64)
        if not bool((raw.norm(dim=-1) > 1e-8).all()):
            raise ValueError("P101_ZERO_ACTION_CANDIDATE")
        pools.append(raw)
        qa = []
        for q in range(PROBES):
            coords = []
            for dim in range(D):
                value = p86.digest_int(
                    seed, world, "p101-probe-"+str(task),q,dim,193)-96
                if family == "anisotropic":
                    value = value // (2 ** (dim % 4))
                coords.append(value/128.)
            qa.append(coords)
        questions.append(qa)
    raw_tensor = torch.stack(pools)
    return World(tuple(m), raw_tensor,
                 F.normalize(raw_tensor,dim=-1),
                 F.normalize(torch.tensor(questions,dtype=torch.float64),dim=-1))


def world_hash(family: str, split: str, index: int) -> str:
    h = hashlib.sha256()
    for w in range(COUNTS[split]):
        world = build_world(family, split, index, w)
        for value in (*world.maps, world.raw, world.pool, world.queries):
            h.update(value.contiguous().cpu().numpy().tobytes())
    return h.hexdigest()


class RLS(NamedTuple):
    weights: torch.Tensor
    precision: torch.Tensor


def initial() -> RLS:
    return RLS(torch.zeros((D,D),dtype=torch.float64),
               torch.eye(D,dtype=torch.float64)/RLS_PRIOR)


def read(state: RLS, action: torch.Tensor) -> torch.Tensor:
    if (action.shape != (D,) or action.dtype != torch.float64
        or not bool(torch.isfinite(action).all())):
        raise ValueError("P101_INVALID_ACTION_READ")
    return state.weights @ action


def confirmed_write(state: RLS, action: torch.Tensor,
                    effect: torch.Tensor, confirmed=True) -> RLS:
    if not confirmed:
        if (action.shape!=(D,) or effect.shape!=(D,)
            or action.dtype!=torch.float64 or effect.dtype!=torch.float64):
            raise ValueError("P101_INVALID_UNCONFIRMED_SHAPE")
        return state
    if (action.shape!=(D,) or effect.shape!=(D,)
        or action.dtype!=torch.float64 or effect.dtype!=torch.float64
        or not bool(torch.isfinite(action).all())
        or not bool(torch.isfinite(effect).all())):
        raise ValueError("P101_UNQUALIFIED_CONFIRMED_EFFECT")
    g=state.precision@action
    den=1.+action@g
    if not bool(torch.isfinite(den)) or float(den)<=0:
        raise ValueError("P101_INVALID_PRECISION")
    residual=effect-(state.weights@action)
    return RLS(state.weights+torch.outer(residual,g)/den,
               state.precision-torch.outer(g,g)/den)


def choose(policy: str, state: RLS, pool: torch.Tensor,
           raw: torch.Tensor, seed: int, w: int, task: int, t: int) -> int:
    # Deliberately NO World/latent map/outcome/query parameter: no oracle input.
    if (policy not in POLICIES or pool.shape!=(CHOICES,D)
        or raw.shape!=pool.shape or not bool(torch.isfinite(pool).all())):
        raise ValueError("P101_UNQUALIFIED_CHOICE_SET")
    if policy=="max_variance":
        value=((pool@state.precision)*pool).sum(dim=-1)
        return int(torch.argmax(value))
    if policy=="hash_random":
        return p86.digest_int(seed,w,"p101-random",task*STEPS+t,0,CHOICES)
    if policy=="cyclic":
        return t%CHOICES
    return int(torch.argmax(raw.square().sum(dim=-1)))


def run_world(world: World, policy: str, seed: int, wid: int,
              wrong_credit=False):
    states=[]
    receipts=[]
    all_journal=[]
    task0_prediction_before_task1 = None
    task0_snapshot = None
    for task in range(2):
        state=initial()
        journal=[]
        for t in range(STEPS):
            available=world.pool[task,t]
            raw=world.raw[task,t]
            selected=choose(policy,state,available,raw,seed,wid,task,t)
            action=available[selected].detach().clone()
            # The environment computes the effect ONLY AFTER the chosen action
            # is committed, no counterfactual return for the other 7 choices.
            effect=(world.maps[task] @ action).detach().clone()
            receipts.append((task,t,selected))
            journal.append((action,effect))
            state=confirmed_write(state,action,effect)
        states.append(RLS(state.weights.detach().clone(),
                          state.precision.detach().clone()))
        all_journal.append(journal)
        if task == 0:
            task0_snapshot = RLS(states[0].weights.clone(),
                                 states[0].precision.clone())
            task0_prediction_before_task1 = world.queries[0]@states[0].weights.T
    # Both tasks' read-only outputs, and cold Task0 after Task1.
    scores=[]
    for task in range(2):
        q=world.queries[task]
        predicted=q@states[task].weights.T
        expected=q@world.maps[task].T
        scores.append(float((predicted-expected).square().mean()))
    cold0=world.queries[0]@states[0].weights.T
    reloaded0=world.queries[0]@task0_snapshot.weights.T
    if (not torch.equal(cold0,task0_prediction_before_task1)
        or not torch.equal(cold0,reloaded0)
        or not torch.equal(states[0].weights,task0_snapshot.weights)
        or not torch.equal(states[0].precision,task0_snapshot.precision)):
        raise ValueError("P101_COLD_RETENTION_CHANGED")
    counterfactual_wrong=None
    if wrong_credit:
        wrong_scores=[]
        for task in range(2):
            state=initial()
            previous=all_journal[task][-1][0]
            for action,effect in all_journal[task]:
                # Assign consequence to a different confirmed action.
                state=confirmed_write(state,previous,effect)
                previous=action
            q=world.queries[task]
            wrong_scores.append(float(((q@state.weights.T)-(q@world.maps[task].T)).square().mean()))
        counterfactual_wrong=sum(wrong_scores)/len(wrong_scores)
    journal_digest=hashlib.sha256(json.dumps(receipts,separators=(",",":")).encode()).hexdigest()
    if len(receipts)!=2*STEPS:
        raise ValueError("P101_EFFECT_BUDGET_MISMATCH")
    return {
        "mse_task0":scores[0],"mse_task1":scores[1],
        "mse_all":sum(scores)/len(scores),
        "cold_task0_bitwise":True,
        "n_confirmed_effects":len(receipts),
        "selected_indices_sha256":journal_digest,
        "wrong_credit_mse":counterfactual_wrong,
        "persistent_floats":2*(D*D+D*D),
    }


def evaluate(family: str, split: str, seed_index: int, policy: str,
             negative=False, count: int|None=None):
    n=COUNTS[split] if count is None else count
    if n<1 or n>COUNTS[split]:
        raise ValueError("P101_NONFROZEN_WORLD_SCOPE")
    seed=SEEDS[family][split][seed_index]
    rows=[run_world(build_world(family,split,seed_index,w),policy,seed,w,
                    wrong_credit=negative) for w in range(n)]
    return {
        "mse_task0":sum(x["mse_task0"] for x in rows)/n,
        "mse_task1":sum(x["mse_task1"] for x in rows)/n,
        "mse_all":sum(x["mse_all"] for x in rows)/n,
        "world_count":n,
        "total_confirmed_outcomes":sum(x["n_confirmed_effects"] for x in rows),
        "cold_restore_bitwise":all(x["cold_task0_bitwise"] for x in rows),
        "persistent_floats":rows[0]["persistent_floats"],
        "journal_sha256":hashlib.sha256("".join(
            x["selected_indices_sha256"] for x in rows).encode()).hexdigest(),
        "wrong_credit_mse": (sum(x["wrong_credit_mse"] for x in rows)/n
                             if negative else None),
    }


def bench(family: str, seed_index: int, policy: str):
    seed=SEEDS[family]["CAL"][seed_index]
    worlds=[build_world(family,"CAL",seed_index,w) for w in range(8)]
    def run_batch():
        return [run_world(x,policy,seed,w) for w,x in enumerate(worlds)]
    for _ in range(2):
        run_batch()
    values=[]
    for _ in range(7):
        t=time.perf_counter()
        run_batch()
        values.append((time.perf_counter()-t)*1000)
    return statistics.median(values)


def run() -> dict:
    freeze=pinned_protocol()  # source-free custody BEFORE any new source birth
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started=time.perf_counter()
    out={
        "schema":"ttt-hybrid.p101-intervention-value-dev-result.v1",
        "freeze_commit":FREEZE_SHA,
        "freeze_blob":FREEZE_BLOB,
        "status":"UNVERIFIED",
        "scientific_pass":False,"external_simulator_used":False,
        "reserved_OOD_generated":False,"reserved_negative_generated":False,
        "original_model_modified":False,"native_hybrid_trained":False,
        "source_sha256":{},"families":{},
    }
    for family in FAMILIES:
        for split in COUNTS:
            for index in range(3):
                label=f"{family}_{split}_{index}"
                out["source_sha256"][label]=world_hash(family,split,index)
    if len(set(out["source_sha256"].values()))!=12:
        raise ValueError("P101_SOURCE_COLLISION")
    for family in FAMILIES:
        result={"seeds":[]}
        for i in range(3):
            fit={p:evaluate(family,"FIT",i,p) for p in PASSIVES}
            selected=min(PASSIVES,key=lambda p:fit[p]["mse_all"])
            # Single prospective CAL score per policy, per seed.
            cal={p:evaluate(family,"CAL",i,p,negative=(p=="max_variance"))
                 for p in POLICIES}
            active=cal["max_variance"]
            baseline=cal[selected]
            ratios={k:active["mse_"+k]/max(baseline["mse_"+k],1e-18)
                    for k in ("all","task0","task1")}
            negative_ratio=active["wrong_credit_mse"]/max(active["mse_all"],1e-18)
            cpu={p:bench(family,i,p) for p in (selected,"max_variance")}
            runtime_ratio=cpu["max_variance"]/max(cpu[selected],1e-12)
            criteria={
                "aggregate_gain":ratios["all"]<=.85,
                "task1_gain":ratios["task1"]<=.95,
                "cold_task0_gain":ratios["task0"]<=.95,
                "wrong_credit_negative":negative_ratio>=1.05,
                "whole_episode_CPU":runtime_ratio<=3,
                "cold_restore":all(x["cold_restore_bitwise"] for x in cal.values()),
                "equal_outcome_calls":all(x["total_confirmed_outcomes"]==
                     COUNTS["CAL"]*2*STEPS for x in cal.values()),
                "equal_state_capacity":all(x["persistent_floats"]==1024 for x in cal.values()),
            }
            result["seeds"].append({
                "index":i,"fit_selection":selected,"FIT_passives":fit,
                "CAL_policies":cal,"quality_ratio_active_over_control":ratios,
                "wrong_credit_ratio":negative_ratio,"cpu_median_ms":cpu,
                "cpu_ratio":runtime_ratio,"criteria":criteria,
                "seed_pass":all(criteria.values())
            })
        rows=result["seeds"]
        result["pass_2_of_3"]=sum(row["seed_pass"] for row in rows)>=2
        result["median_primary_ratio"]=statistics.median(
            row["quality_ratio_active_over_control"]["all"] for row in rows)
        result["family_pass"]=result["pass_2_of_3"] and result["median_primary_ratio"]<=.85
        out["families"][family]=result
    out["status"]=("DEV_CAL_ONLY_PASS_OOD_STILL_SEALED"
                   if all(x["family_pass"] for x in out["families"].values())
                   else "DEV_FAIL_INTERVENTION_VALUE_OR_COST")
    out["process_wall_seconds"]=time.perf_counter()-started
    return out


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    report=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({
        "status":report["status"],
        "family_pass":{k:v["family_pass"] for k,v in report["families"].items()},
        "scientific_pass":False,
    }))


if __name__=="__main__":
    main()
