"""F7 first genuine training from a separately collected, single-use FIT128 source.

Original native Hybrid, zero-initialized raw public action-key bypass and
fully-trained 256-state SSM receive exactly the same 96 FIT train worlds.
The 32 remaining FIT worlds are consumed for development checkpoint choice;
they are NOT independent CAL or OOD. All results retain science_pass=false.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"experiments"))

import numpy as np
import torch
from experiments import p85mr_f7_protocol as cfg
from experiments import p85mr_f7_native_candidate as candidate
from experiments import p85mr_f5_native_vs_controls as f5
from experiments import p85mr_f4_train_controls as f4
from p80mr_causalgym_trace_adapter import adapt_capture

MAX_SECONDS=7200
ARM_LIMIT=2400
FREEZE_SAMPLER_SEED=985299
OWNER_REF="refs/heads/research/p85mr-f7-native-raw-bypass-freshfit-20261112"
SOURCE_ROOT_ALIAS="f7-freshfit-1821000000-once-v1"


def _canon(obj):
    return (json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False)+"\n").encode()


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def _read_immutable(path):
    if path.is_symlink():
        raise ValueError("F7_TRAIN_SYMLINK_SOURCE")
    raw=path.read_bytes()
    obj=json.loads(raw)
    if _canon(obj)!=raw:
        raise ValueError("F7_TRAIN_NONCANONICAL_SOURCE")
    return obj


def _owner():
    cfg.frozen()
    checks={
      "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
      "GITHUB_ACTOR":"spoonman136668-ai",
      "GITHUB_EVENT_NAME":"push",
      "GITHUB_REF":OWNER_REF,
      "GITHUB_RUN_NUMBER":"1",
      "GITHUB_RUN_ATTEMPT":"1",
      "RUNNER_NAME":"TTT-HYBRID-GPU-WIN",
      "P85MR_F7_FIT_TRAIN_ONLY":"1",
    }
    if any(os.environ.get(k)!=v for k,v in checks.items()):
        raise PermissionError("F7_TRAIN_UNQUALIFIED_OWNER_OR_ATTEMPT")
    if (os.name!="posix" or os.getuid()!=1000
        or sys.version_info[:2]!=(3,11)
        or "microsoft" not in os.uname().release.lower()):
        raise PermissionError("F7_TRAIN_NOT_OWNER_WSL")
    me=Path.home().name
    intended=f"/home/{me}/.local/share/ttt-hybrid/g7-s25-cpu311-r1/venv/bin/python"
    if Path(sys.executable).resolve()!=Path(intended).resolve():
        raise PermissionError("F7_TRAIN_NOT_QUALIFIED_CPU_RUNTIME")
    run=os.environ.get("GITHUB_RUN_ID","")
    sha=os.environ.get("GITHUB_SHA","")
    if not re.fullmatch(r"[0-9]{6,20}",run) or not re.fullmatch(r"[0-9a-f]{40}",sha):
        raise PermissionError("F7_TRAIN_INVALID_SOURCE_OR_RUN")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="-1":
        raise PermissionError("F7_TRAIN_CPU_ONLY")
    return {"run_id":run,"sha":sha,"actor":"spoonman136668-ai",
            "runner":"TTT-HYBRID-GPU-WIN","frozen_blob":cfg.FREEZE_BLOB}


def audit_capture(source,source_summary,owner):
    freeze=cfg.frozen()
    capture_path=Path(source)
    summary=_read_immutable(Path(source_summary))
    capture=_read_immutable(capture_path)
    result=summary["result"]
    if (summary["owner"]!=owner or result["owner"]!=owner
        or summary["external_actions"]!=result["source_effects"]
        or result["source_sessions"]!=cfg.FIT_WORLDS
        or result["source_effects"]>cfg.MAX_ACTIONS
        or result["CAL_OOD_NEGATIVE_created"] is not False
        or result["scientific_pass"] is not False
        or capture["upstream_commit"]!=freeze["scope"]["external_sha"]
        or capture["source_id"]!=f"p85mr-f7-fit-{owner['run_id']}"
        or capture["custody_ref"]!="ghrun:"+owner["run_id"]
        or capture["producer_id"]!="ttt-hybrid-f7-owner"
        or _sha(capture_path.read_bytes())!=result["capture_sha256"]
        or len(capture["worlds"])!=cfg.FIT_WORLDS):
        raise ValueError("F7_TRAIN_SOURCE_PROVENANCE_OR_SPLIT_DENIED")
    root=Path.home()/".local"/"share"/"ttt-hybrid"/SOURCE_ROOT_ALIAS
    if root.is_symlink() or not root.is_dir() or not root.resolve().is_relative_to(Path.home().resolve()):
        raise ValueError("F7_TRAIN_NATIVE_EXT4_JOURNAL_MISSING")
    first=_read_immutable(root/"ATTEMPT.json")
    last=_read_immutable(root/"RESULT.json")
    if (first["run"]!=owner or last!=result
        or first["replay_permitted"] is not False
        or last["replay_allowed"] is not False
        or _sha((root/"FIT_CAPTURE.json").read_bytes())!=result["capture_sha256"]):
        raise ValueError("F7_TRAIN_EXT4_OWNER_RECEIPT_DRIFT")
    count=0; receipts=set()
    for idx,world in enumerate(capture["worlds"]):
        seed=cfg.seed(idx)
        session=root/f"fit-{idx:03d}-{seed}"
        if session.is_symlink() or not session.is_dir():
            raise ValueError("F7_TRAIN_MISSING_EXT4_SESSION")
        if (world["world_id"]!=f"f7-fit-{idx:03d}-{seed}"
            or world["episode_seed"]!=seed
            or not 1<=len(world["transitions"])<=cfg.HORIZON
            or world["probe_event_ids"]!=list(range(2,len(world["transitions"])+1))
            or _read_immutable(session/"world.json")!=world):
            raise ValueError("F7_TRAIN_WRONG_OR_SHORT_WORLD")
        claim=_read_immutable(session/"claim.json")
        if (claim["run"]!=owner["run_id"] or claim["seed"]!=seed
            or claim["index"]!=idx or claim["no_retry"] is not True):
            raise ValueError("F7_TRAIN_SESSION_OWNER_MISMATCH")
        assert _read_immutable(session/"reset.outcome.json")["seed"]==seed
        previous=None
        for step,event in enumerate(world["transitions"],1):
            if set(event)!={"event_id","state_before","executed_action",
                            "state_after","reward","terminated","truncated",
                            "mode","receipt_ref"}:
                raise ValueError("F7_TRAIN_NONFACTUAL_EFFECT_FIELDS")
            if (event["event_id"]!=step or event["mode"]!="do"
                or event["executed_action"]!=cfg.factual_policy(
                    idx,step,event["state_before"])
                or event["receipt_ref"]!=f"ghrun:{owner['run_id']}:F7-fit:{idx}:{step}"
                or event["receipt_ref"] in receipts
                or type(event["terminated"]) is not bool
                or type(event["truncated"]) is not bool
                or type(event["reward"]) not in (int,float)
                or not math.isfinite(event["reward"])):
                raise ValueError("F7_TRAIN_ACTION_IDENTITY_OR_NONFINITE")
            receipts.add(event["receipt_ref"])
            before=cfg.state4(event["state_before"])
            after=cfg.state4(event["state_after"])
            if previous is not None and previous!=before:
                raise ValueError("F7_TRAIN_BROKEN_CAUSAL_CHAIN")
            previous=after
            if step<len(world["transitions"]) and (
                event["terminated"] or event["truncated"]):
                raise ValueError("F7_TRAIN_TERMINATED_WORLD_REPLAY")
            intent=_read_immutable(session/f"{step:03d}.intent.json")
            outcome=_read_immutable(session/f"{step:03d}.outcome.json")
            if (outcome!=event or intent["action"]!=event["executed_action"]
                or cfg.state4(intent["state_before"])!=before
                or intent["step"]!=step
                or intent["unselected_effect"]!="UNOBSERVED"):
                raise ValueError("F7_TRAIN_WRONG_CREDIT_OR_MISSING_INTENT")
            count+=1
        if len(world["transitions"])<2:
            # Do not silently discard a terminal episode to manufacture
            # 96/32 full-world split; source remains consumed.
            raise ValueError("F7_TRAIN_SHORT_WORLD_ABORT_NO_REPLAY")
    if count!=result["source_effects"] or len(receipts)!=count:
        raise ValueError("F7_TRAIN_GLOBAL_ACTION_RECEIPT_COUNT")
    material=adapt_capture(capture)
    if (material["custody"]["scientific_pass"] is not False
        or len(material["worlds"])!=128):
        raise ValueError("F7_TRAIN_UNEXPECTED_SCIENCE_PROMOTION")
    worlds=[]
    for item in material["worlds"]:
        events=item["events"]
        worlds.append({
            "action":torch.tensor([e["action"] for e in events],dtype=torch.float32),
            "effect":torch.tensor([e["effect"] for e in events],dtype=torch.float32),
        })
    train,val=worlds[:96],worlds[96:]
    if len(train)!=96 or len(val)!=32:
        raise ValueError("F7_TRAIN_HOLDOUT_OR_TRAIN_SPLIT")
    return train,val,{"authentic_source_actions":count,
        "source_worlds":128,"source_capture_sha256":result["capture_sha256"],
        "train_worlds":96,"validation_worlds":32,
        "native_owner_journal_audit":True,
        "independent_CAL_evaluation":False}


def _atomic_json(path,obj):
    temp=path.with_suffix(".partial")
    if temp.exists():
        raise ValueError("F7_TRAIN_UNCERTAIN_RECEIPT_OVERWRITE")
    data=json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+"\n"
    with temp.open("x",encoding="utf-8") as out:
        out.write(data);out.flush();os.fsync(out.fileno())
    os.replace(temp,path)


def _zero_baseline(worlds):
    accum=[]
    for w in worlds:
        y=w["effect"][1:,:4]
        if not len(y): raise ValueError("F7_MISSING_NONZERO_VALID_STEPS")
        accum.append(float(y.square().mean().item()))
    return math.fsum(accum)/len(accum)


def qualified_gradients(kind,model):
    """Accept genuinely dormant original hard-depth router; never hide failed learning.

    The original CausalDepthRouter intentionally evaluates no FFN when its
    hard threshold routes zero tokens. In that case ONLY its depth parameters
    may lack an autograd path. All reachable gradients must remain finite,
    the stem/head and non-depth native blocks must receive learning signal,
    and a conventional SSM must have no missing gradients.
    """
    if kind not in cfg.ARMS:
        raise ValueError("F7_UNKNOWN_FROZEN_GRADIENT_ARM")
    missing=[];connected={}
    for name,param in model.named_parameters():
        if not param.requires_grad:
            raise ValueError("F7_FROZEN_MODEL_PARAM_NOT_TRAINABLE")
        grad=param.grad
        if grad is None:
            missing.append(name)
            continue
        if grad.shape!=param.shape or not bool(torch.isfinite(grad).all()):
            raise ValueError("F7_NONFINITE_OR_WRONG_SHAPE_GRADIENT")
        connected[name]=bool((grad!=0).any())
    if not connected or not any(connected.values()):
        raise ValueError("F7_NO_REAL_MODEL_GRADIENT")
    if kind==candidate.K_SSM:
        if missing:
            raise ValueError("F7_UNEXPECTED_SSM_DISCONNECTED_GRADIENT")
        return {"dormant_depth_count":0,"connected":len(connected)}
    prefix="depth." if kind==candidate.K_NATIVE else "original.depth."
    depth=model.depth if kind==candidate.K_NATIVE else model.original.depth
    if missing and (float(depth.last_utilization)!=0.0 or
                    any(not name.startswith(prefix) for name in missing)):
        raise ValueError("F7_UNQUALIFIED_NONDEPTH_MODEL_GRADIENT")
    core="" if kind==candidate.K_NATIVE else "original."
    for component in ("stem.","ssm.","attn.","ttt.","head."):
        if not any(name.startswith(core+component) and has_grad
                   for name,has_grad in connected.items()):
            raise ValueError("F7_CORE_LEARNING_GRADIENT_MISSING_"+component)
    if kind==candidate.K_BYPASS and not any(
            name.startswith("raw_key.") and has_grad
            for name,has_grad in connected.items()):
        raise ValueError("F7_RAW_ACTION_KEY_NOT_TRAINABLE")
    return {"dormant_depth_count":len(missing),"connected":len(connected)}


def _train_arm(kind,seed,train,val,indices,output,global_start):
    model=candidate.model_from_arm(kind,seed).cpu()
    started=time.monotonic()
    before=f5.hash_weights(model)
    optimizer=torch.optim.AdamW(
        model.parameters(),lr=.001 if kind==candidate.K_SSM else .002,
        weight_decay=.01)
    best=float("inf");best_state=None;best_step=None;trace=[]
    dormant_depth_updates=0
    for step in range(1,801):
        if time.monotonic()-global_start>MAX_SECONDS:
            raise TimeoutError("F7_FIT_OWNER_GLOBAL_TRAINING_BUDGET")
        if time.monotonic()-started>ARM_LIMIT or f5.rss()>3.0:
            raise TimeoutError("F7_FIT_ARM_CPU_MEMORY_BUDGET")
        picked=[train[i] for i in indices[step-1].tolist()]
        facts,valid=f4._padded(picked)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss=candidate.loss_for_arm(kind,model,facts,valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F7_INVALID_REAL_FIT_LOSS")
        loss.backward()
        gradient_receipt=qualified_gradients(kind,model)
        if gradient_receipt["dormant_depth_count"]:
            dormant_depth_updates+=1
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%50==0:
            model.eval()
            mse=candidate.score_worlds(kind,model,val)
            if not math.isfinite(mse):
                raise ValueError("F7_NONFINITE_FIT_HELDOUT")
            trace.append({"step":step,"FIT_holdout_delta_MSE":mse})
            if mse<best:
                best=mse;best_step=step
                best_state=copy.deepcopy(model.state_dict())
    if best_state is None:raise ValueError("F7_MISSING_SELECTED_FIT_CHECKPOINT")
    model.load_state_dict(best_state);model.eval()
    chosen=candidate.score_worlds(kind,model,val)
    if abs(chosen-best)>1e-8:
        raise ValueError("F7_COLD_FIT_HELDOUT_SCORE_PARITY")
    if f5.hash_weights(model)==before:
        raise ValueError("F7_NO_REAL_FIT_WEIGHT_UPDATE")
    matrices={name:v.detach().cpu().contiguous().numpy()
              for name,v in model.state_dict().items()}
    safe_name=kind.replace("original_four_block_with_16D_raw_action_key_bypass","hybrid-raw-key").replace("original_four_block_HybridPilot","native-hybrid").replace("original_256state_SSM","ssm256")
    filename=f"{safe_name}-{seed}.npz"
    path=output/filename
    if path.exists():raise FileExistsError("F7_REAL_FIT_MODEL_ALREADY_USED")
    np.savez_compressed(path,**matrices)
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=set(matrices) or any(
              not np.array_equal(archive[k],matrices[k]) for k in matrices):
            raise ValueError("F7_NONPICKLE_CHECKPOINT_READBACK_FAIL")
    cold=candidate.model_from_arm(kind,seed)
    new={k:torch.from_numpy(v.copy()) for k,v in matrices.items()}
    cold.load_state_dict(new,strict=True)
    cold.eval()
    if (f5.hash_weights(cold)!=f5.hash_weights(model)
        or abs(candidate.score_worlds(kind,cold,val)-chosen)>1e-10):
        raise ValueError("F7_RESTORED_NATIVE_MODEL_NOT_EQUIVALENT")
    negative=None
    if kind!=candidate.K_SSM:
        negative=f5.cal_mse(model,val,corrupted=True)
    return {
        "arm":kind,"model_seed":seed,"started_weights":before,
        "selected_weights":f5.hash_weights(model),
        "selected_checkpoint_update":best_step,
        "updates_executed":800,
        "dormant_original_depth_updates":dormant_depth_updates,
        "FIT_holdout_delta_MSE":chosen,
        "wrong_credit_FIT_holdout_MSE":negative,
        "checkpoints":trace,
        "parameters_static":sum(p.numel() for p in model.parameters()),
        "per_world_memory_floats":256 if kind==candidate.K_SSM else None,
        "train_wall_s":time.monotonic()-started,
        "process_peak_rss_gib":f5.rss(),
        "safe_checkpoint":filename,
        "NPZ_sha256":_sha(path.read_bytes()),
        "SCIENCE_PASS":False,
    }


def run(*,source,source_summary,output,receipt):
    owner=_owner()
    train,holdout,source_meta=audit_capture(source,source_summary,owner)
    return train_admitted_fit(train=train,holdout=holdout,
                             source_meta=source_meta,owner=owner,
                             output=output,receipt=receipt)


def train_admitted_fit(*,train,holdout,source_meta,owner,output,receipt):
    """Unchanged frozen F7 optimizer/selection; caller must admit source first.

    The existing one-shot source runner calls run(), which enforces its
    original owner. A distinct training-only owner gate may call this seam
    ONLY after authentic immutable F7 previous-run journal verification.
    No simulator code, source constructor, source action, or effect exists here.
    """
    if (not isinstance(source_meta,dict)
        or source_meta.get("source_worlds")!=128
        or source_meta.get("authentic_source_actions")!=12288
        or source_meta.get("source_capture_sha256")!=
           "7df8e2a2dc276cb90fbcac91619083dec7911453d5af10d2ccbdcfd45e5007d4"
        or source_meta.get("native_owner_journal_audit") is not True
        or len(train)!=96 or len(holdout)!=32):
        raise PermissionError("F7_TRAIN_UNQUALIFIED_OR_NONAUTHENTIC_FIT_SOURCE")
    outdir=Path(output)
    outdir.mkdir(mode=0o700,parents=True,exist_ok=False)
    result_path=Path(receipt)
    result_path.parent.mkdir(exist_ok=True,parents=True)
    if result_path.exists():
        raise FileExistsError("F7_TRAIN_RECEIPT_NO_REPLAY")
    data={"schema":"ttt-hybrid.f7-authentic-FIT-train-validation.v1",
          "status":"F7_FIT_TRAINING_IN_PROGRESS_DEV_ONLY",
          "source":source_meta,"owner":owner,"results":[],
          "original_source_actions_executed_by_this_trainer":0,
          "source_FIT_training":True,
          "new_CAL_OOD_NEGATIVE":False,
          "scientific_pass":False,"deployment":False}
    _atomic_json(result_path,data)
    start=time.monotonic()
    sampler=f4.frozen_indices(len(train))
    sampler_sha=_sha(sampler.contiguous().cpu().numpy().tobytes())
    data["sampler_sha256"]=sampler_sha
    zero=_zero_baseline(holdout)
    data["zero_FIT_holdout_delta_MSE"]=zero
    try:
        for arm in cfg.ARMS:
            for seed in cfg.MODEL_SEEDS:
                one=_train_arm(arm,seed,train,holdout,sampler,outdir,start)
                data["results"].append(one)
                _atomic_json(result_path,data)
        by_seed={(x["arm"],x["model_seed"]):x for x in data["results"]}
        wins=0;competent=True
        if not math.isfinite(zero) or zero<=0:
            raise ValueError("F7_UNQUALIFIED_ZERO_CONTROL")
        comparisons=[]
        for seed in cfg.MODEL_SEEDS:
            native=by_seed[(candidate.K_NATIVE,seed)]
            bypass=by_seed[(candidate.K_BYPASS,seed)]
            ssm=by_seed[(candidate.K_SSM,seed)]
            ratio_vs_native=bypass["FIT_holdout_delta_MSE"]/native["FIT_holdout_delta_MSE"]
            ratio_vs_ssm=bypass["FIT_holdout_delta_MSE"]/ssm["FIT_holdout_delta_MSE"]
            ssm_competent=ssm["FIT_holdout_delta_MSE"]<=.75*zero
            competent=competent and ssm_competent
            wins+=(ratio_vs_native<=.90 and ratio_vs_ssm<=.90)
            comparisons.append({"seed":seed,
                "ratio_bypass_to_native":ratio_vs_native,
                "ratio_bypass_to_SSM":ratio_vs_ssm,
                "ssm_control_FIT_competent":ssm_competent})
        data["comparisons"]=comparisons
        data["FIT_development_mechanism_candidate"]=bool(wins>=2 and competent)
        data["FIT_bypass_wins_seeds"]=wins
        data["SSM_all_seeds_competent"]=bool(competent)
        data["status"]="F7_FIT_DEV_TEST_COMPLETED_CAL_NOT_TOUCHED"
        data["whole_training_wall_s"]=time.monotonic()-start
    except BaseException as e:
        data["status"]="F7_FIT_TRAINING_STOPPED_NO_CAL_NO_REPLAY"
        data["failure_class"]=type(e).__name__
        _atomic_json(result_path,data)
        raise
    _atomic_json(result_path,data)
    return data


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",type=Path,required=True)
    ap.add_argument("--source-summary",type=Path,required=True)
    ap.add_argument("--models",type=Path,required=True)
    ap.add_argument("--receipt",type=Path,required=True)
    a=ap.parse_args()
    data=run(source=a.source,source_summary=a.source_summary,
             output=a.models,receipt=a.receipt)
    print(json.dumps({"status":data["status"],
          "FIT_candidate":data["FIT_development_mechanism_candidate"],
          "scientific_pass":False,"CAL_used":False},sort_keys=True))


if __name__=="__main__":
    main()
