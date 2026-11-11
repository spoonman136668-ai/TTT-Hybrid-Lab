"""P103: fresh goal-conditioned causal two-step control (author-synthetic DEV only).

Freeze committed before code/source birth: 40a70dca2120a96404eaf10e0bc7538ec39fbf20.
Uses original immutable P101 confirmed-effect RLS. Never passes latent world maps
or counterfactual outcomes to action chooser. No external/OOD first use.
"""
from __future__ import annotations

import argparse
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
from experiments import p101_information_interventions_dev as p101

FREEZE_COMMIT = "40a70dca2120a96404eaf10e0bc7538ec39fbf20"
FREEZE_BLOB = "bbeefd335a96e8f7ea6d54362d7986a2132ec0b6"
ROOT = Path(__file__).resolve().parents[1]
PARENT = "52553526ceea2ce01f26cc035bb486b351974fd8"
D, CALIBRATION, CONTROL, CHOICES, PROBES = 16, 8, 8, 8, 12
FAMILIES = ("isotropic", "anisotropic")
SEEDS = {
    "isotropic": {"FIT": (1030111, 1030113, 1030119),
                  "CAL": (1030127, 1030131, 1030133)},
    "anisotropic": {"FIT": (1030211, 1030217, 1030223),
                    "CAL": (1030227, 1030229, 1030233)},
}
COUNT = {"FIT": 16, "CAL": 12}
BETAS = (.02, .08, .20)
CONVENTIONAL = ("pure_mpc", "greedy", "geometry", "random", "cyclic")
ARMS = ("epistemic_mpc",) + CONVENTIONAL + ("goal_sham",)
PINNED = {
    "model.py": "5d9ea57b827aa1796548447868fa5ee14c251b49",
    "experiments/p86_portable_causal_preflight.py":
        "008492665ee9090db5791963f2e2336ea283ee62",
    "experiments/p101_information_interventions_dev.py":
        "a4375102e64492f179826b8775855b63b1c315fd",
    "experiments/p102_consequence_interventions_frozen.json":
        "4f50b8bcfbc6429a675765b89a3b33abbc56d1dc",
}


def freeze() -> dict:
    raw = (ROOT / "experiments/p103_goal_mpc_freeze.json").read_bytes()
    blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("P103_FROZEN_BLOB_DRIFT")
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.p103-goal-conditioned-epistemic-mpc-dev-freeze.v1"
        or f["parent"]["sha"] != PARENT
        or f["source"]["seeds"] != {
            **{fam:{sp:list(x) for sp,x in cfg.items()}
               for fam,cfg in SEEDS.items()},
            "reserved_OOD":[1030401,1030403],
            "reserved_NEGATIVE":[1030501,1030503]}
        or f["source"]["fit_worlds_per_seed"] != COUNT["FIT"]
        or f["source"]["cal_worlds_per_seed"] != COUNT["CAL"]
        or f["source"]["calibration"] !=
           "8 initial actual interventions per task selected with identical fixed P101 geometric variance policy for all arms; exact same confirmed RLS updates and returned outcome; common state at control start"
        or f["source"]["control"] !=
           "8 more real goal-directed interventions per task with action candidate pools identical across all policies, 8 alternatives per step"
        or f["policies"]["beta_grid"] != list(BETAS)
        or f["learner"]["same_action_budget"] !=
           "8 calibrate + 8 control = 16 actual confirmed outcomes per task = 32/world per policy"
        or f["acceptance"]["median_primary_ratio_each_family_max"] != .90
        or f["acceptance"]["whole_episode_CPU_candidate_vs_mpc_max"] != 2.):
        raise ValueError("P103_PROTOCOL_DRIFT")
    for path,blob in PINNED.items():
        if p86.blob(ROOT / path) != blob:
            raise ValueError("P103_PARENT_MODEL_BLOB_DRIFT")
    return f


class World(NamedTuple):
    maps: tuple[torch.Tensor,torch.Tensor]  # environment ONLY
    calibrate: torch.Tensor  # [2,8,8,16]
    pool: torch.Tensor       # [2,8,8,16]
    goals: torch.Tensor      # [2,16] public task goals
    queries: torch.Tensor    # [2,12,16] evaluator ONLY


def _action(seed: int, world: int, tag: str, n: int, family: str) -> torch.Tensor:
    coords=[]
    for j in range(D):
        x=p86.digest_int(seed,world,tag,n,j,193)-96
        if family == "anisotropic":
            x=x//(2**(j%4))
        coords.append(x/128.)
    v=torch.tensor(coords,dtype=torch.float64)
    if v.norm()<1e-12:
        raise ValueError("P103_ZERO_ACTION")
    return F.normalize(v,dim=0)


def build_world(family: str, split: str, i: int, wid: int) -> World:
    if (family not in FAMILIES or split not in COUNT
        or type(i) is not int or i not in (0,1,2)
        or type(wid) is not int or not 0<=wid<COUNT[split]):
        raise ValueError("P103_PROTECTED_SOURCE_REFUSAL")
    seed=SEEDS[family][split][i]
    maps,cal,pools,goals,queries=[],[],[],[],[]
    for task in (0,1):
        U=torch.tensor(p86.basis(seed,wid,"p103-u"+str(task)),
                       dtype=torch.float64)
        V=torch.tensor(p86.basis(seed,wid,"p103-v"+str(task)),
                       dtype=torch.float64)
        M=(V@U.T)/256.
        maps.append(M)
        cal.append(torch.stack([
            torch.stack([_action(seed,wid,"p103-cal-"+str(task),
                                 t*CHOICES+j,family)
                         for j in range(CHOICES)]) for t in range(CALIBRATION)]))
        pools.append(torch.stack([
            torch.stack([_action(seed,wid,"p103-ctrl-"+str(task),
                                 t*CHOICES+j,family)
                         for j in range(CHOICES)]) for t in range(CONTROL)]))
        hidden_target=_action(seed,wid,"p103-goal-"+str(task),0,family)
        goals.append(M@(2.5*hidden_target))
        queries.append(torch.stack([
            _action(seed,wid,"p103-query-"+str(task),q,family)
            for q in range(PROBES)]))
    return World(tuple(maps),torch.stack(cal),torch.stack(pools),
                 torch.stack(goals),torch.stack(queries))


def fingerprint(family: str, split: str, i: int) -> str:
    h=hashlib.sha256()
    for wid in range(COUNT[split]):
        w=build_world(family,split,i,wid)
        for tensor in (*w.maps,w.calibrate,w.pool,w.goals,w.queries):
            h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def chooser(arm: str, learned: p101.RLS, physical: torch.Tensor,
            goal: torch.Tensor, options: torch.Tensor,
            next_options: torch.Tensor|None,
            seed: int, wid: int, task: int, step: int,
            beta: float=.02) -> int:
    if (arm not in ARMS or options.shape!=(CHOICES,D)
        or options.dtype!=torch.float64 or physical.shape!=(D,)
        or goal.shape!=(D,) or beta not in BETAS
        or (next_options is not None and next_options.shape!=(CHOICES,D))
        or not bool(torch.isfinite(options).all())
        or not bool(torch.isfinite(physical).all())
        or not bool(torch.isfinite(goal).all())):
        raise ValueError("P103_CHOOSER_BOUNDARY_VIOLATION")
    if arm=="random":
        return int(p86.digest_int(seed,wid,"p103-random",task*CONTROL+step,0,CHOICES))
    if arm=="cyclic":
        return step%CHOICES
    risk=((options@learned.precision)*options).sum(-1)
    if arm=="geometry":
        return int(torch.argmax(risk))
    W=learned.weights
    expected_effect=options@W.T
    target = -goal if arm == "goal_sham" else goal
    predicted_state=physical[None,:]+expected_effect
    if arm=="greedy" or next_options is None:
        predicted_cost=(predicted_state-target).square().mean(-1)
    else:
        # Evaluate all candidate i and next candidate j, using MODEL effects.
        # None of these predicted future outcomes is admitted as actual data.
        next_pred=next_options@W.T
        paths=predicted_state[:,None,:]+next_pred[None,:,:]
        predicted_cost=(paths-target).square().mean(-1).min(dim=1).values
    if arm in ("epistemic_mpc","goal_sham"):
        norm_risk=risk/risk.mean().clamp_min(1e-12)
        predicted_cost=predicted_cost-beta*norm_risk*goal.square().mean()
    return int(torch.argmin(predicted_cost))


def one_world(world: World, arm: str, seed: int, wid: int,
              beta: float=.02, wrong_credit=False):
    journal=[]
    task_snapshots=[]
    task0_before=None
    task0_state=None
    tracking=[]
    task0_query_before=None
    for task in (0,1):
        model=p101.initial()
        events=[]
        # Every arm gets EXACTLY the same 8 factual calibration interventions.
        # Physical state is reset to zero after calibration for ALL arms.
        for t in range(CALIBRATION):
            choices=world.calibrate[task,t]
            index=int(torch.argmax(((choices@model.precision)*choices).sum(-1)))
            action=choices[index]
            events.append((action.clone(),(world.maps[task]@action).clone()))
            model=p101.confirmed_write(model,*events[-1])
            journal.append((task,0,t,index))
        phys=torch.zeros(D,dtype=torch.float64)
        actual_goal=world.goals[task]  # visible to each policy
        per_step=[]
        for t in range(CONTROL):
            options=world.pool[task,t]
            next_options=world.pool[task,t+1] if t+1<CONTROL else None
            j=chooser(arm,model,phys,actual_goal,options,next_options,
                      seed,wid,task,t,beta)
            action=options[j].clone()
            journal.append((task,1,t,j))
            # Only NOW is the chosen physical action's factual consequence born.
            consequence=(world.maps[task]@action).clone()
            phys=phys+consequence
            events.append((action,consequence))
            model=p101.confirmed_write(model,action,consequence)
            per_step.append(float((phys-actual_goal).square().mean()))
        tracking.append(per_step)
        saved=p101.RLS(model.weights.clone(),model.precision.clone())
        task_snapshots.append(saved)
        task_snapshots[task].weights  # intentional no-op: never mutate
        if task==0:
            task0_state=p101.RLS(saved.weights.clone(),saved.precision.clone())
            task0_before=world.queries[0]@saved.weights.T
        if task==1:
            read_again=world.queries[0]@task0_state.weights.T
            if (not torch.equal(read_again,task0_before)
                or not torch.equal(task_snapshots[0].weights,task0_state.weights)
                or not torch.equal(task_snapshots[0].precision,task0_state.precision)):
                raise ValueError("P103_COLD_TASK0_DRIFT")
    q_mse=[]
    for task in (0,1):
        q=world.queries[task]
        predict=q@task_snapshots[task].weights.T
        true=q@world.maps[task].T
        q_mse.append(float((predict-true).square().mean()))
    wrong=None
    if wrong_credit:
        errs=[]
        # Reconstruct both task models from same exact 16 actual effects
        # but attach effects to the wrong chronological factual action.
        for task in (0,1):
            # Rebuild actual journal from frozen interventions chosen in run,
            # without any new simulator calls or choosing new actions.
            actions=[]
            for logged_task,phase,t,j in journal:
                if logged_task==task:
                    a=(world.calibrate if phase==0 else world.pool)[task,t,j]
                    actions.append(a)
            truths=[world.maps[task]@a for a in actions]
            replay=p101.initial()
            for t,effect in enumerate(truths):
                replay=p101.confirmed_write(replay,actions[(t-1)%len(actions)],effect)
            q=world.queries[task]
            errs.append(float(((q@replay.weights.T)-(q@world.maps[task].T)).square().mean()))
        wrong=statistics.mean(errs)
    if len(journal)!=2*(CALIBRATION+CONTROL):
        raise ValueError("P103_ACTION_BUDGET_DRIFT")
    return {
        "tracking_all":statistics.mean(sum(tracking,[])),
        "tracking_task0":statistics.mean(tracking[0]),
        "tracking_task1":statistics.mean(tracking[1]),
        "tracking_final_task0":tracking[0][-1],
        "tracking_final_task1":tracking[1][-1],
        "query_mse":statistics.mean(q_mse),
        "wrong_credit_query_mse":wrong,
        "confirmed_effects":len(journal),
        "cold_bitwise":True,
        "persistent_float64_state":1024,
        "journal_sha256":hashlib.sha256(json.dumps(journal,separators=(",",":")).encode()).hexdigest(),
    }


def evaluate(fam: str, split: str, i: int, arm: str,
             beta: float=.02, wrong_credit=False, count=None):
    if fam not in FAMILIES or split not in COUNT or type(i) is not int or i not in (0,1,2):
        raise ValueError("P103_UNQUALIFIED_SOURCE")
    n=COUNT[split] if count is None else count
    if type(n) is not int or not 1<=n<=COUNT[split]:
        raise ValueError("P103_NONFROZEN_WORLD_COUNT")
    seed=SEEDS[fam][split][i]
    rows=[one_world(build_world(fam,split,i,w),arm,seed,w,beta,wrong_credit)
          for w in range(n)]
    cols=("tracking_all","tracking_task0","tracking_task1",
          "tracking_final_task0","tracking_final_task1","query_mse")
    out={key:statistics.mean(z[key] for z in rows) for key in cols}
    out.update({
        "worlds":n,"confirmed_outcomes":sum(z["confirmed_effects"] for z in rows),
        "cold_bitwise":all(z["cold_bitwise"] for z in rows),
        "persistent_float64_state":1024,
        "journal_sha256":hashlib.sha256("".join(z["journal_sha256"] for z in rows).encode()).hexdigest(),
        "wrong_credit_query_mse":(
            statistics.mean(z["wrong_credit_query_mse"] for z in rows)
            if wrong_credit else None),
    })
    return out


def cpu_bench(fam: str,i: int,arm: str,beta: float):
    seed=SEEDS[fam]["CAL"][i]
    batch=[build_world(fam,"CAL",i,w) for w in range(8)]
    def f():
        return [one_world(w,arm,seed,j,beta) for j,w in enumerate(batch)]
    for _ in range(2):f()
    durations=[]
    for _ in range(7):
        start=time.perf_counter();f()
        durations.append(1000*(time.perf_counter()-start))
    return statistics.median(durations)


def study():
    freeze()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    result={
        "schema":"ttt-hybrid.p103-goal-conditioned-epistemic-mpc-result.v1",
        "status":"UNVERIFIED",
        "freeze_commit":FREEZE_COMMIT,"freeze_blob":FREEZE_BLOB,
        "scientific_pass":False,"external_source_used":False,
        "OOD_generated":False,"NEGATIVE_reserved_generated":False,
        "native_model_trained":False,"controller_modified":False,
        "source_sha256":{},"families":{},
    }
    for fam in FAMILIES:
        for split in COUNT:
            for i in range(3):
                result["source_sha256"][f"{fam}_{split}_{i}"]=fingerprint(fam,split,i)
    if len(set(result["source_sha256"].values())) != 12:
        raise ValueError("P103_COHORT_COLLISION")
    for fam in FAMILIES:
        family={"seeds":[]}
        for i in range(3):
            beta_fit={str(beta):evaluate(fam,"FIT",i,"epistemic_mpc",beta)
                      for beta in BETAS}
            chosen_beta=min(BETAS,key=lambda b:beta_fit[str(b)]["tracking_all"])
            controls_fit={a:evaluate(fam,"FIT",i,a) for a in CONVENTIONAL}
            best=min(CONVENTIONAL,key=lambda a:controls_fit[a]["tracking_all"])
            # CAL measured once per registered arm and frozen beta, no tuning.
            candidate=evaluate(fam,"CAL",i,"epistemic_mpc",chosen_beta,True)
            conventional={a:evaluate(fam,"CAL",i,a) for a in CONVENTIONAL}
            sham=evaluate(fam,"CAL",i,"goal_sham",chosen_beta)
            selected=conventional[best]
            pure=conventional["pure_mpc"]
            ratios={target:{
                key:candidate[key]/max(base[key],1e-18)
                for key in ("tracking_all","tracking_task0","tracking_task1")
            } for target,base in (("FIT_strongest",selected),("pure_mpc",pure))}
            goal_sham_ratio=sham["tracking_all"]/max(candidate["tracking_all"],1e-18)
            wrong_ratio=(candidate["wrong_credit_query_mse"]/
                         max(candidate["query_mse"],1e-18))
            cpu={a:cpu_bench(fam,i,a,chosen_beta)
                 for a in ("epistemic_mpc","pure_mpc")}
            cpu_ratio=cpu["epistemic_mpc"]/max(cpu["pure_mpc"],1e-12)
            gates={
                "primary_vs_strongest":ratios["FIT_strongest"]["tracking_all"]<=.90,
                "primary_vs_pure_mpc":ratios["pure_mpc"]["tracking_all"]<=.90,
                "task0_vs_both":all(x["tracking_task0"]<=.95 for x in ratios.values()),
                "task1_vs_both":all(x["tracking_task1"]<=.95 for x in ratios.values()),
                "goal_sham_degrades":goal_sham_ratio>=1.05,
                "wrong_credit_degrades":wrong_ratio>=1.05,
                "cpu_vs_pure_mpc":cpu_ratio<=2.0,
                "equal_action_budget":all(z["confirmed_outcomes"]==
                    COUNT["CAL"]*2*(CALIBRATION+CONTROL)
                    for z in (candidate,sham,*conventional.values())),
                "same_capacity":all(z["persistent_float64_state"]==1024
                    for z in (candidate,sham,*conventional.values())),
                "cold_task0":all(z["cold_bitwise"]
                    for z in (candidate,sham,*conventional.values())),
            }
            family["seeds"].append({
                "seed_index":i,"FIT_beta_grid":beta_fit,"FIT_beta":chosen_beta,
                "FIT_controls":controls_fit,"FIT_strongest":best,
                "CAL_candidate":candidate,"CAL_controls":conventional,
                "CAL_goal_sham":sham,"ratios":ratios,"goal_sham_ratio":goal_sham_ratio,
                "wrong_credit_ratio":wrong_ratio,"CPU_median_ms":cpu,
                "CPU_ratio":cpu_ratio,"gates":gates,"seed_pass":all(gates.values()),
            })
        family["pass_2of3"]=sum(x["seed_pass"] for x in family["seeds"])>=2
        family["median_primary_ratio"]={
            name:statistics.median(x["ratios"][name]["tracking_all"]
                   for x in family["seeds"])
            for name in ("FIT_strongest","pure_mpc")
        }
        family["family_pass"]=family["pass_2of3"] and all(
            x<=.90 for x in family["median_primary_ratio"].values())
        result["families"][fam]=family
    result["status"]=(
        "DEV_CAL_GOAL_MPC_POSITIVE_NOT_SCIENTIFIC_PASS"
        if all(x["family_pass"] for x in result["families"].values())
        else "DEV_FAIL_GOAL_MPC_VS_CONVENTIONAL_CONTROLS")
    result["study_wall_seconds"]=time.perf_counter()-start
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--receipt",type=Path,required=True)
    args=ap.parse_args()
    report=study()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],
                      "families":{x:y["family_pass"] for x,y in report["families"].items()},
                      "scientific_pass":False}))


if __name__=="__main__":
    main()
