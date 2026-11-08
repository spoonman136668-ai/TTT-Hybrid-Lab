"""Integration engineering: checkpoint/resume ACTUAL optimizer training and factual fast memory.

Unchanged native P0/P1/P2 and checkpoint implementations; no sci world/seed use.
The analytic intervention reverses at event 40. The *only* training label
is the factual outcome returned after the prequential prediction is recorded.
This diagnostic tests cumulative training/optimizer/fast-memory exact resume,
not scientific intelligence, transfer or accepted persistent owner custody.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from native_continuous_state import NativeContinuousHybrid, ContinuousState
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts.native_p2_synthetic_adaptation import analytic_consequence
from native_training_checkpoint import (
    CheckpointRejected, load_checkpoint, save_checkpoint, validate_chain)

SCHEMA="ttt-hybrid.native-p3.checkpoint-reversal-dev.v1"
MODEL_INITIALIZATIONS=(46111,46117,46129)  # DEV model seeds, never science world seeds
TOTAL_EVENTS=80
CHECKPOINT_AT=40
MODEL_WIDTH=12
MAX_WALL_SECONDS=60
DEV_LEARNING_RATE=.003


def source_sha() -> str:
    p=subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],
                     capture_output=True,text=True,timeout=10,check=False)
    if p.returncode!=0 or len(p.stdout.strip())!=40:
        raise RuntimeError("P3_EXACT_GIT_SOURCE_UNAVAILABLE")
    sha=p.stdout.strip()
    expected=os.getenv("EXPECTED_CANDIDATE_SHA")
    if expected is not None and sha != expected:
        raise RuntimeError("P3_SOURCE_SHA_DRIFT")
    return sha


def fingerprint(*parts: bytes) -> str:
    h=hashlib.sha256()
    for p in parts:
        h.update(p)
    return h.hexdigest()


def identity() -> dict:
    """Source- and environment-specific local integrity, not an owner signature."""
    import platform
    return {
        "source_sha": source_sha(),
        "architecture_sha256": fingerprint((ROOT/"native_continuous_state.py").read_bytes()),
        "interface_sha256": fingerprint((ROOT/"scripts/native_prequential_gate.py").read_bytes(),
                                        (ROOT/"native_training_checkpoint.py").read_bytes()),
        "protocol_sha256": fingerprint((ROOT/"experiments/native_p3_continuity_engineering_frozen.json").read_bytes()),
        "data_split_sha256": fingerprint((ROOT/"scripts/native_p2_synthetic_adaptation.py").read_bytes(),
                                        b"analytic-action-reversal-80-checkpoint-at-40"),
        "harness_sha256": fingerprint((ROOT/"scripts/native_p3_continuity.py").read_bytes()),
        "environment_sha256": fingerprint(json.dumps({
            "python":platform.python_version(),
            "torch":torch.__version__,
            "platform":platform.platform(),
            "device":"cpu","threads":1,
        },sort_keys=True).encode()),
    }


def new_model_and_optimizer(initialization: int):
    if type(initialization) is not int or initialization not in MODEL_INITIALIZATIONS:
        raise ValueError("P3_UNREGISTERED_DEV_INITIALIZATION")
    torch.manual_seed(initialization)
    model=NativeContinuousHybrid(1,1,width=MODEL_WIDTH,history_limit=4,
                                 fast_rate=.25).cpu().eval()
    optim=torch.optim.AdamW(model.parameters(),lr=DEV_LEARNING_RATE)
    return model,optim


def state_dict(episode:NativeCausalEpisode,model:NativeContinuousHybrid):
    state=episode.snapshot()
    if state is None:
        state=model.initial_state(torch.zeros((1,1),dtype=torch.float32))
    return {"recurrent":state.recurrent,"fast_weights":state.fast_weights,
            "history":state.history}


def save_completed(directory:Path,identity_map:dict,model,optimizer,episode,
                   cursor:int,parent:str|None)->str:
    if episode._pending is not None:
        raise CheckpointRejected("P3_PENDING_CAUSAL_EVENT_NOT_COMPLETED")
    if type(cursor) is not int or cursor<0 or cursor>TOTAL_EVENTS:
        raise CheckpointRejected("P3_CURSOR_INVALID")
    if episode._event_index!=cursor:
        raise CheckpointRejected("P3_EVENT_INDEX_CURSOR_MISMATCH")
    return save_checkpoint(directory,model=model,optimizer=optimizer,
                           identity=identity_map,native_state=state_dict(episode,model),
                           sampler_state={"position":cursor,"event_index":cursor,
                                          "pending":False},
                           steps=cursor,consumed_examples=cursor,
                           parent_manifest_sha256=parent)


def restore_completed(directory:Path,digest:str,identity_map:dict,model,optimizer):
    verified=validate_chain(directory,digest,identity_map)
    native_state,sampler=load_checkpoint(directory,digest,model=model,
                                        optimizer=optimizer,identity=identity_map)
    if set(sampler)!={"position","event_index","pending"} or sampler["pending"] is not False:
        raise CheckpointRejected("P3_PENDING_OR_INCOMPLETE_SAMPLER")
    cursor=sampler["position"]
    if (type(cursor) is not int or cursor<0 or cursor>TOTAL_EVENTS or
        type(sampler["event_index"]) is not int or sampler["event_index"]!=cursor or
        verified["steps"]!=cursor or verified["consumed_examples"]!=cursor):
        raise CheckpointRejected("P3_SAMPLER_REPLAY_OR_STEP_MISMATCH")
    state=ContinuousState(native_state["recurrent"],native_state["fast_weights"],
                          native_state["history"])
    model._validate_state(state,torch.zeros((1,1),dtype=torch.float32))
    episode=NativeCausalEpisode(model)
    # Only executed following hash-verified checkpoint load and strict
    # completed-event cursor validation. No pending ticket crosses the boundary.
    episode._state=NativeCausalEpisode._copy_state(state)
    episode._event_index=cursor
    return episode,cursor


def transact(model,optimizer,episode:NativeCausalEpisode,t:int)->float:
    if not 0<=t<TOTAL_EVENTS or episode._event_index!=t or episode._pending is not None:
        raise RuntimeError("P3_REPLAY_OR_PENDING_EVENT")
    action=torch.tensor([[1.0 if t%2 else -1.0]],dtype=torch.float32)
    sensor=torch.zeros((1,1),dtype=torch.float32)
    elapsed=torch.tensor([[.2]],dtype=torch.float32)
    ticket=episode.predict(sensor,action,elapsed)
    # Obtain the factual consequence AFTER prediction, never counterfactual.
    observed=analytic_consequence(action,t)
    prequential=float((ticket.prediction-observed).square().item())
    episode.observe(ticket,observed,factual=True)
    # A real AdamW gradient step from *factual* measured output. The DEV-only
    # sensor jitter probes exact torch RNG replay as well as optimizer moments.
    jittered=sensor+.0005*torch.randn_like(sensor)
    trained_prediction=model.step(jittered,action,elapsed)[0]
    loss=F.mse_loss(trained_prediction,observed)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return prequential


def _exact_equal(left,right)->bool:
    if isinstance(left,torch.Tensor):
        return isinstance(right,torch.Tensor) and torch.equal(left,right)
    if isinstance(left,dict):
        return isinstance(right,dict) and left.keys()==right.keys() and all(
            _exact_equal(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return type(left) is type(right) and len(left)==len(right) and all(
            _exact_equal(a,b) for a,b in zip(left,right))
    return left==right


def one_initialization(initialization:int,root_dir:Path)->dict:
    model,optim=new_model_and_optimizer(initialization)
    episode=NativeCausalEpisode(model)
    ident=identity()
    root=save_completed(root_dir,ident,model,optim,episode,0,None)
    prediction_errors=[]
    for t in range(CHECKPOINT_AT):
        prediction_errors.append(transact(model,optim,episode,t))
    assert any(bool(torch.count_nonzero(s)) for s in
               (episode.snapshot().fast_weights,))
    original_at_checkpoint=state_dict(episode,model)
    tip=save_completed(root_dir,ident,model,optim,episode,CHECKPOINT_AT,root)
    assert validate_chain(root_dir,tip,ident)["steps"]==CHECKPOINT_AT
    saved_next_rng=torch.get_rng_state().clone()
    for t in range(CHECKPOINT_AT,TOTAL_EVENTS):
        prediction_errors.append(transact(model,optim,episode,t))
    full_memory=state_dict(episode,model)
    full_model={k:v.detach().clone() for k,v in model.state_dict().items()}
    full_optimizer=optim.state_dict()

    # Genuine cold model/optimizer object reconstruction; no reuse of live
    # module, optimizer, or sampled data cursor.
    cold_model,cold_opt=new_model_and_optimizer(initialization)
    cold_ep,position=restore_completed(root_dir,tip,ident,cold_model,cold_opt)
    cold_at_checkpoint=state_dict(cold_ep,cold_model)
    state_exact=_exact_equal(cold_at_checkpoint,original_at_checkpoint)
    rng_exact=bool(torch.equal(saved_next_rng,torch.get_rng_state()))
    resumed_errors=[]
    for t in range(position,TOTAL_EVENTS):
        resumed_errors.append(transact(cold_model,cold_opt,cold_ep,t))
    checks={
        "checkpoint_source_and_ancestry_verified": True,
        "sampler_cold_resume_no_reconsume": position==CHECKPOINT_AT,
        "factual_fast_memory_after_checkpoint_nonzero": bool(torch.count_nonzero(
            cold_at_checkpoint["fast_weights"])),
        "native_memory_cold_checkpoint_bit_identical": state_exact,
        "full_prediction_path_postcheckpoint_bit_identical": prediction_errors[CHECKPOINT_AT:]==resumed_errors,
        "final_model_parameters_bit_identical": _exact_equal(full_model,cold_model.state_dict()),
        "final_optimizer_moments_bit_identical": _exact_equal(full_optimizer,cold_opt.state_dict()),
        "final_recurrent_fast_history_bit_identical": _exact_equal(full_memory,state_dict(cold_ep,cold_model)),
        "torch_rng_before_next_step_bit_identical": rng_exact,
        "immutable_root_and_tip_no_accepted_latest_pointer": (
            root!=tip and not (root_dir/"latest").exists() and
            not (root_dir/"accepted").exists()),
    }
    return {
        "initialization":initialization,
        "status":"PASS_EXACT_CONTINUATION_ENGINEERING_ONLY" if all(checks.values())
                 else "FAIL_EXACT_CONTINUATION_ENGINEERING_ONLY",
        "checks":checks,
        "checkpoint_root_sha256":root,
        "checkpoint_tip_sha256":tip,
        "cumulative_optimizer_steps":TOTAL_EVENTS,
        "checkpoint_at_completed_step":CHECKPOINT_AT,
        "final_consumed_factual_examples":TOTAL_EVENTS,
        "post_switch_uninterrupted_mse":sum(prediction_errors[CHECKPOINT_AT:])/(
            TOTAL_EVENTS-CHECKPOINT_AT),
        "post_switch_resumed_mse":sum(resumed_errors)/len(resumed_errors),
    }


def run()->dict:
    if os.environ.get("NATIVE_P3_ENGINEERING_ONLY","1")!="1":
        raise RuntimeError("P3_SCIENCE_FIRST_USE_FORBIDDEN")
    torch.set_num_threads(1)
    start=time.perf_counter()
    records=[]
    with tempfile.TemporaryDirectory(prefix="native-p3-dev-") as td:
        for init in MODEL_INITIALIZATIONS:
            directory=Path(td)/str(init)
            directory.mkdir()
            records.append(one_initialization(init,directory))
    result={
        "schema":SCHEMA,
        "status":"PASS_COLD_RESUME_INTEGRATION_ENGINEERING_ONLY" if all(
            r["status"]=="PASS_EXACT_CONTINUATION_ENGINEERING_ONLY"
            for r in records) else "FAIL_COLD_RESUME_INTEGRATION_ENGINEERING_ONLY",
        "exact_source_sha":source_sha(),
        "identity_sha256":fingerprint(json.dumps(identity(),sort_keys=True).encode()),
        "science_first_use":"NOT_RUN",
        "scientific_seeds_consumed":[],
        "scientific_worlds_constructed":0,
        "training_type":"PUBLIC_ANALYTIC_DEV_FACTUAL_GRADIENT_AND_FAST_MEMORY",
        "scope":"LOCAL_CANDIDATE_PERSISTENCE_NOT_AUTHENTICATED_OWNER_STORAGE_OR_ACCEPTED_SCIENCE",
        "checkpoint_artifact_scope":"EPHEMERAL_DEV_FILES_PLUS_HASH_RECEIPTS",
        "record_count":len(records),
        "records":records,
        "wall_seconds":time.perf_counter()-start,
    }
    return result


def main()->int:
    parser=argparse.ArgumentParser()
    parser.add_argument("--receipt",type=Path,required=True)
    args=parser.parse_args()
    r=run()
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    args.receipt.write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
    print(json.dumps({
        "status":r["status"],"exact_source_sha":r["exact_source_sha"],
        "record_count":r["record_count"],"wall_seconds":r["wall_seconds"],
        "scoped_result":[{"init":x["initialization"],"status":x["status"],
                          "all_checks":all(x["checks"].values()),
                          "post_switch_mse":x["post_switch_resumed_mse"]}
                         for x in r["records"]],
        "science_first_use":r["science_first_use"]
    },sort_keys=True))
    return 0 if r["status"]=="PASS_COLD_RESUME_INTEGRATION_ENGINEERING_ONLY" else 1

if __name__=="__main__":
    raise SystemExit(main())
