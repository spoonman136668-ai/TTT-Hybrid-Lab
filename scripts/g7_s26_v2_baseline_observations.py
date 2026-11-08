"""S26-on-V2 baseline *input* adapter: development evidence, NEVER model fitting.

Only seed-approved S25 paired worlds with their already-public WorldStep
sensor/action/target interface are accepted. This creates correctly padded
causal input batches for the existing conventional GRU and RBF-KRR control
implementations; the fitter, kernel calibrator, and science world generator
are deliberately not imported or invoked.

IMPORTANT: same-state paired clone observation is simulated, not real-world
causal identification. Development input readout is not scientific evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable

# -I omits the script directory. Admit only this exact checkout's scripts.
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch
from torch import Tensor
from g7_s25_numeric_ttt import causal_history32, WorldStep
from g7_s25_paired_interventions import (
    ENGINEERING_SEEDS, PAIR_ACTIONS, MAX_STEPS, PairedS20World,
)

DEV_SEEDS=(3101,3109,3119)
DEV_REGIME="source_stable"
BATCH_WORLDS=32
HISTORY_LEN=32
OBSERVED_PAIR_WIDTH=2
QUERY_FEATURES=3
FROZEN_S25_BLOB="31633b341403729b7ffbb7d73f3a04098ad44380"
FROZEN_S26_BLOB="34251b2cde6373ed484dbc21e4617c9165047e93"
V2_S20_SOURCE_BLOB="bc3e9f255a4d1b967252ada8782ac9f8c000f7b7"


class BaselineReadinessBlocked(ValueError):
    pass


def require(ok:bool,why:str)->None:
    if not ok:
        raise BaselineReadinessBlocked(why)


@dataclass(frozen=True)
class DevExampleBatch:
    """Read-only factual inputs; copied in causal order before any learner fit."""
    history:Tensor
    query_pair:Tensor
    observed_pair:Tensor
    realized_action:Tensor
    stream_sha256:str
    seed:int
    regime:str
    worlds:int
    steps:int


def fingerprint_step(sha,step:int,item:WorldStep,history:Tensor)->None:
    """Hash inputs, labels and realized-action selectors with shape and dtype."""
    sha.update(step.to_bytes(2,"little"))
    for name,tensor in (
        ("causal_history",history),
        ("query_pair",item.query_pair),
        ("observed_pair",item.observed_pair),
        ("realized_action",item.realized_action_index),
    ):
        value=tensor.detach().cpu().contiguous()
        sha.update(name.encode("ascii"))
        sha.update(json.dumps({
            "shape":list(value.shape),"dtype":str(value.dtype)
        },sort_keys=True,separators=(",",":")).encode("ascii"))
        sha.update(value.numpy().tobytes())


@torch.no_grad()
def collect_development_observations(
    seed:int,batch:int=BATCH_WORLDS,regime:str=DEV_REGIME,
    *,world_factory:Callable=PairedS20World,
)->DevExampleBatch:
    """Complete same-seed S25 dev-only prequential stream, no model updates.

    The only observer API is WorldStep.emit(), never world._x/_z/_old/_new.
    All examples use 32-frame right-padded *prior* history. Query/label
    pairing is retained so either conventional model can later use identical
    information without oracle hindsight at prediction time.
    """
    require(type(seed) is int and seed in DEV_SEEDS and
            DEV_SEEDS==ENGINEERING_SEEDS,
            "S26_BASELINE_DEV_ONLY_SEED_REQUIRED")
    require(type(batch) is int and batch==BATCH_WORLDS,
            "S26_BASELINE_FROZEN_WORLD_BATCH_REQUIRED")
    require(regime==DEV_REGIME,"S26_BASELINE_SOURCE_WORLD_ONLY")
    require(world_factory is PairedS20World,
            "S26_BASELINE_UNREGISTERED_WORLD_FACTORY")
    world=world_factory(seed,batch,regime)
    hist=[];queries=[];targets=[];selectors=[]
    h=hashlib.sha256()
    prev_selected=None
    previous_history_length=0
    for step in range(MAX_STEPS):
        item=world.emit()
        require(isinstance(item,WorldStep) and
                item.history.shape==(batch,1 if step==0 else step,4) and
                item.query_pair.shape==(batch,OBSERVED_PAIR_WIDTH,QUERY_FEATURES) and
                item.observed_pair.shape==(batch,OBSERVED_PAIR_WIDTH) and
                item.realized_action_index.shape==(batch,),
                "S26_BASELINE_PREQUENTIAL_PAIR_SHAPE_DRIFT")
        prefix=causal_history32(item.history)
        require(prefix.shape==(batch,HISTORY_LEN,4) and
                item.query_pair.device.type=="cpu" and
                item.observed_pair.device.type=="cpu" and
                item.realized_action_index.device.type=="cpu",
                "S26_BASELINE_NUMERIC_CPU_CONTEXT_REQUIRED")
        require(bool(torch.isfinite(prefix).all() and
                     torch.isfinite(item.query_pair).all() and
                     torch.isfinite(item.observed_pair).all()),
                "S26_BASELINE_NONFINITE_SOURCE")
        require(bool(((item.realized_action_index==0)|
                      (item.realized_action_index==1)).all()),
                "S26_BASELINE_EXECUTED_SELECTOR_NOT_BINARY")
        require(bool((item.query_pair[:,:,1]==
                      torch.tensor(PAIR_ACTIONS,device="cpu").reshape(1,2)).all()),
                "S26_BASELINE_ACTION_ORDER_DRIFT")
        require(bool((item.query_pair[:,0,0]==item.query_pair[:,1,0]).all()
                     and (item.query_pair[:,0,2]==item.query_pair[:,1,2]).all()),
                "S26_BASELINE_PAIRED_PROBES_NOT_SAME_STATE_TIME")
        # Actual realized action at step k must be the last available
        # sensor in the NEXT step's history, never the CURRENT label.
        if prev_selected is not None:
            require(bool(torch.equal(item.history[:,-1,3],prev_selected)),
                    "S26_BASELINE_FUTURE_LABEL_LEAKAGE")
        previous_history_length=item.history.shape[1]
        prev_selected=item.observed_pair.gather(
            1,item.realized_action_index[:,None]).squeeze(1).clone()
        fingerprint_step(h,step,item,prefix)
        hist.append(prefix.clone())
        queries.append(item.query_pair.clone())
        targets.append(item.observed_pair.clone())
        selectors.append(item.realized_action_index.clone())
    closed=world.close()
    require(closed["steps"]==MAX_STEPS and closed["worlds"]==batch and
            closed["paired_observations"]==MAX_STEPS*batch*2 and
            closed["realized_transitions"]==MAX_STEPS*batch and
            previous_history_length==MAX_STEPS-1,
            "S26_BASELINE_WORLD_NOT_CLOSED_EXACTLY")
    result=DevExampleBatch(
        torch.cat(hist,0).detach(),
        torch.cat(queries,0).detach(),
        torch.cat(targets,0).detach(),
        torch.cat(selectors,0).detach(),
        h.hexdigest(),seed,regime,batch,MAX_STEPS,
    )
    require(result.history.shape==(MAX_STEPS*batch,32,4) and
            result.query_pair.shape==(MAX_STEPS*batch,2,3) and
            result.observed_pair.shape==(MAX_STEPS*batch,2) and
            result.realized_action.shape==(MAX_STEPS*batch,),
            "S26_BASELINE_AGGREGATE_BATCH_DRIFT")
    return result


def audit_development_example_inputs()->dict:
    """Read-only exposure/seed fingerprints; does not fit or validate a model."""
    rows=[]
    for seed in DEV_SEEDS:
        item=collect_development_observations(seed)
        rows.append({
            "seed":seed,"regime":item.regime,
            "observation_stream_sha256":item.stream_sha256,
            "paired_step_examples":int(item.history.shape[0]),
            "paired_sensor_targets":int(item.observed_pair.numel()),
            "history_shape":list(item.history.shape),
            "action_query_shape":list(item.query_pair.shape),
            "paired_targets_shape":list(item.observed_pair.shape),
            "target_read_after_prior_history":True,
            "oracle_state_or_coefficients_exposed":False,
        })
    require(len({r["observation_stream_sha256"] for r in rows})==3,
            "S26_BASELINE_DEV_STREAM_COLLISION")
    return {
        "schema":"ttt-hybrid.g7-s26.v2.dev-only-conventional-input-readiness.v1",
        "status":"PASS_DEVELOPMENT_PREQUENTIAL_INPUTS_NO_MODEL_FIT",
        "rows":rows,
        "paired_world_steps":sum(r["paired_step_examples"] for r in rows),
        "actual_observations":sum(r["paired_sensor_targets"] for r in rows),
        "frozen_s25_blob":FROZEN_S25_BLOB,
        "frozen_s26_blob":FROZEN_S26_BLOB,
        "composite_v2_s20_blob":V2_S20_SOURCE_BLOB,
        "conventional_baseline_fit":"NOT_RUN",
        "krr_calibration":"NOT_RUN",
        "scientific_first_use":"NOT_AUTHORIZED_NOT_RUN",
        "frozen_fit_validation_and_heldout_seeds":"NOT_CONSUMED",
        "scientific_learning_conclusion":"UNVERIFIED",
        "gpu":"NOT_USED",
    }


def git(root:Path,*args:str)->str:
    p=subprocess.run(["git","-C",str(root),*args],
                     text=True,capture_output=True,check=False,timeout=12)
    require(p.returncode==0,"S26_BASELINE_GIT_SOURCE_NOT_AVAILABLE")
    return p.stdout.strip()


def main()->int:
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--expected-sha",required=True)
    p.add_argument("--receipt",type=Path,required=True)
    args=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    require(os.environ.get("CUDA_VISIBLE_DEVICES")=="-1",
            "S26_BASELINE_CUDA_NOT_ISOLATED")
    require(len(args.expected_sha)==40 and
            git(root,"rev-parse","HEAD")==args.expected_sha and
            git(root,"status","--porcelain","--untracked-files=all")=="",
            "S26_BASELINE_SOURCE_SHA_OR_CLEANLINESS_DRIFT")
    for path,expected in (
        ("experiments/g7_s25_frozen_contradiction_ttt.json",FROZEN_S25_BLOB),
        ("experiments/g7_s26_frozen_quarter_reserve.json",FROZEN_S26_BLOB),
        ("scripts/g7_s20_hidden_multicycle.py",V2_S20_SOURCE_BLOB),
    ):
        require(git(root,"hash-object","--",path)==expected,
                "S26_BASELINE_SCIENCE_PROTOCOL_OR_PHYSICS_DRIFT")
    torch.set_num_threads(1)
    result=audit_development_example_inputs()
    result["candidate_sha"]=args.expected_sha
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print("S26_CONVENTIONAL_INPUTS="+result["status"])
    print("S26_TRAINED_GRU_KRR=NOT_RUN")
    print("S26_SCIENCE=NOT_RUN")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
