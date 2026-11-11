"""F5: preregistered real FIT/CAL original HybridPilot versus F4 competent controls.

Owner-only, no external simulator execution or OOD. Original model.py unchanged.
Continuous factual event vectors, no text tokenizer or autoregressive label leak.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
try:
    import resource
except ImportError:
    resource = None
import statistics
import subprocess
import sys
import time
import zipfile

# -I does not add script directory: use only immutable project directories.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import numpy as np
import torch
from model import HybridPilot, Configuration
from p85mr_f4_train_controls import read_source_zip, _padded, frozen_indices, _check_fit_cal
from p85mr_dynamics_baselines import (
    SEEDS, new_baseline, prequential, component_errors, _validate_factual
)

PROJECT = "spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH = "research/p85mr-f5-native-hybrid-vs-competent-controls-20261111"
WORKFLOW = ".github/workflows/p85mr-f5-native-fitcal-once.yml"
FREEZE_BLOB = "ba24ca8b9f3146c4c1be55b37d980d044b123676"
PARENT_SHA = "00196c634509c7d04962e78988f4da97227f38f6"
CONTROLS_SHA = "0ce28fcf2976d4a97e922998dbc8696e483db1cfbf1a01bfdd5cb77052ecc56d"
CONTROLS_ZIP_SHA = "ac93a3616619bf4376023e18cfc2ab32757a1a89715fbc02394164682c0159b5"
CONTROL_GH_RUN_ID = "43080774054"
SOURCE_F3_ZIP_SHA = "a04697bdfba1eb93548244329f4f76fb807489aa7766f4a343de897a56caceb5"
SAMPLE_SHA = "212ce45052b6095056238381491417d965eec23733e8ce7190033eed65ab93e8"
BATCH = 16
MAX_UPDATES = 800
CHECK = 50
NATIVE_LR = .002
MAX_RSS_GIB = 3.0
MAX_ARM_SECONDS = 3600
EXPECTED_SSM = {
    985201: (1.328794543553613e-5, "9ff9bbad4d545780cf1d5c6108075b2720aa51032ba3c2c70773abf6a6e2cec4"),
    985203: (1.2177126073709132e-5, "c6af43defa91f8cf308845d72e9b6acf15008761d88d3ede054029affcce4cc5"),
    985207: (1.2313318364931547e-5, "0836105936aba355ad08102075209828f748040d85d50a827648d6166ea64c45"),
}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def locked() -> dict:
    data = (ROOT / "experiments/p85mr_f5_native_vs_controls_freeze.json").read_bytes()
    blob = hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
    if blob != FREEZE_BLOB:
        raise ValueError("F5_FROZEN_SCIENCE_DRIFT")
    d = json.loads(data)
    if (d["schema"] != "ttt-hybrid.p85mr-f5-real-fitcal-native-hybrid-vs-strong-controls-freeze.v1"
        or d["parent"]["source_sha"] != PARENT_SHA
        or d["parent"]["F4_controls_result_sha256"] != CONTROLS_SHA
        or d["parent"]["F4_controls_archive_sha256"] != CONTROLS_ZIP_SHA
        or d["parent"]["source_zip_sha256"] != SOURCE_F3_ZIP_SHA
        or d["training"]["max_updates"] != MAX_UPDATES
        or d["training"]["sampler_seed"] != 985299
        or d["training"]["seeds"] != list(SEEDS)
        or d["native_model"]["input_dim"] != 32
        or d["native_model"]["output_dim"] != 16
        or d["training"]["batch_size"] != BATCH
        or d["training"]["lr"] != NATIVE_LR
        or d["execution"]["branch"] != BRANCH
        or d["execution"]["workflow"] != WORKFLOW
        or d["evaluation"]["quality_max_ratio_vs_strongest"] != .90):
        raise ValueError("F5_FROZEN_PROTOCOL_MISMATCH")
    if d["controls"]["SSM_strongest_CAL"] != {str(k): v[0] for k,v in EXPECTED_SSM.items()}:
        raise ValueError("F5_STRONGEST_LOCK_MISMATCH")
    return d


def owner_admission() -> dict:
    required = {
        "GITHUB_REPOSITORY": PROJECT, "GITHUB_ACTOR": "spoonman136668-ai",
        "GITHUB_EVENT_NAME": "push", "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_WORKFLOW_REF": PROJECT + "/" + WORKFLOW + "@refs/heads/" + BRANCH,
        "GITHUB_RUN_NUMBER": "1", "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "TTT-HYBRID-GPU-WIN", "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64", "P85MR_F5_NATIVE_FITCAL_ONLY": "1",
    }
    if any(os.getenv(k) != v for k,v in required.items()):
        raise PermissionError("F5_OWNER_ROUTE_DENIED")
    if (os.name != "posix" or os.getuid() != 1000
        or "microsoft" not in os.uname().release.lower()
        or sys.version_info[:2] != (3,11)
        or not torch.__version__.startswith("2.5.1")):
        raise PermissionError("F5_ORIGINAL_OWNER_WSL_CPU_ONLY")
    sha_id = os.getenv("GITHUB_SHA","")
    run_id = os.getenv("GITHUB_RUN_ID","")
    if not re.fullmatch("[a-f0-9]{40}",sha_id) or not re.fullmatch("[0-9]{1,20}",run_id):
        raise PermissionError("F5_OWNER_SHA_OR_RUN_MISSING")
    def git(*args):
        v=subprocess.run(["git","-C",str(ROOT),*args],capture_output=True,text=True,timeout=15)
        if v.returncode: raise PermissionError("F5_GIT_CHECK_FAILED")
        return v.stdout.strip()
    if (git("rev-parse","HEAD") != sha_id
        or git("status","--porcelain","--untracked-files=all")
        or git("remote","get-url","origin") not in (
            "https://github.com/"+PROJECT, "https://github.com/"+PROJECT+".git")):
        raise PermissionError("F5_UNTRUSTED_CHECKOUT")
    locked()
    return {"sha":sha_id, "run_id":run_id, "runner":"TTT-HYBRID-GPU-WIN"}


def original_native(seed: int):
    if seed not in SEEDS: raise ValueError("F5_SEED_NOT_FROZEN")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=HybridPilot(32,16,Configuration(
            width=16,heads=2,ssm=True,attention=True,ttt=True,depth=True))
    return model


def factual_inputs(facts: dict, *, corrupted=False) -> torch.Tensor:
    actions,effects=_validate_factual(facts)
    historical=torch.zeros_like(effects)
    if corrupted:
        # Wrong-credit negative uses ONLY two-step-old effects, never future.
        historical[:,2:]=effects[:,:-2]
    else:
        historical[:,1:]=effects[:,:-1]
    return torch.cat((actions,historical),dim=-1)


def native_predict(model, facts, *, corrupted=False) -> torch.Tensor:
    return model(factual_inputs(facts,corrupted=corrupted))


def training_loss(model, facts, valid):
    a,y=_validate_factual(facts)
    if valid.shape != a.shape[:2] or valid.dtype != torch.bool or bool(valid[:,0].any()):
        raise ValueError("F5_INVALID_FACTUAL_MASK")
    pred=native_predict(model,facts)
    if pred.shape != y.shape:
        raise ValueError("F5_ORIGINAL_OUTPUT_SHAPE")
    terms=component_errors(pred,y)
    mask=valid.to(dtype=pred.dtype)
    den=mask.sum()
    if not bool(den>0):
        raise ValueError("F5_NO_VALID_OBSERVATIONS")
    loss=(
        (terms["deltas"]*mask).sum()/den
        +.2*(terms["reward"]*mask).sum()/den
        +.1*(terms["terminated"]*mask).sum()/den
        +.1*(terms["truncated"]*mask).sum()/den
    )
    if not bool(torch.isfinite(loss)):raise ValueError("F5_NONFINITE_LOSS")
    return loss


def cal_mse(model, worlds, *, corrupted=False) -> float:
    model.eval()
    scores=[]
    with torch.no_grad():
        for start in range(0,len(worlds),BATCH):
            group,valid=_padded(worlds[start:start+BATCH])
            preds=native_predict(model,group,corrupted=corrupted)
            mse=component_errors(preds,group["effect"])["deltas"]
            den=valid.sum(1).clamp_min(1)
            scores.extend(((mse*valid).sum(1)/den).tolist())
    return sum(scores)/len(scores)


def hash_weights(model) -> str:
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def rss() -> float:
    if resource is None:raise PermissionError("F5_RSS_REQUIRES_POSIX")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024)


def source_controls(archive: Path):
    if not archive.is_file() or sha(archive.read_bytes())!=CONTROLS_ZIP_SHA:
        raise ValueError("F5_CONTROL_ARCHIVE_ZIP_SHA_DRIFT")
    with zipfile.ZipFile(archive) as z:
        raw=z.read("p85mr_f4_controls_result.json")
        if sha(raw)!=CONTROLS_SHA:raise ValueError("F5_F4_NUMERICAL_RECEIPT_DRIFT")
        result=json.loads(raw)
        if (result["status"]!="FIT_CAL_CONVENTIONAL_CONTROLS_COMPETENT_OOD_SEALED"
            or result["control_competence_pass"] is not True
            or result["source_owner"]["run_id"]!=CONTROL_GH_RUN_ID
            or result["source_owner"]["sha"]!=PARENT_SHA
            or result["source_artifact_sha256"]!=SOURCE_F3_ZIP_SHA
            or result["sampler_sha256"]!=SAMPLE_SHA
            or result["scientific_pass"] is not False
            or len(result["models"])!=12):
            raise ValueError("F5_UNQUALIFIED_OR_TAMPERED_F4_CONTROLS")
        grouped={}
        for row in result["models"]:
            kind,seed=row["kind"],row["model_seed"]
            key=(kind,seed)
            if key in grouped or not row["competent"] or row["sampler_sha256"]!=SAMPLE_SHA:
                raise ValueError("F5_DUPLICATED_OR_FAILED_CONTROL")
            name="p85mr_f4_models/"+row["checkpoint_name"]
            saved=z.read(name)
            if sha(saved)!=row["npz_sha256"]:
                raise ValueError("F5_CONTROL_CHECKPOINT_SHA_DRIFT")
            grouped[key]=row
        assert len(grouped)==12
        models={}
        strongest={}
        for seed in SEEDS:
            winner=min((grouped[(kind,seed)] for kind in (
                "linear7","mlp7","gru256","ssm256")),key=lambda t:t["cal_delta_mse"])
            if winner["kind"]!="ssm256" or not math.isclose(
                winner["cal_delta_mse"],EXPECTED_SSM[seed][0],rel_tol=1e-9):
                raise ValueError("F5_PRIOR_STRONGEST_CONVENTIONAL_DRIFT")
            strong=grouped[("ssm256",seed)]
            if strong["npz_sha256"]!=EXPECTED_SSM[seed][1]:
                raise ValueError("F5_SSM_CHECKPOINT_DRIFT")
            b=z.read("p85mr_f4_models/"+strong["checkpoint_name"])
            with np.load(io.BytesIO(b),allow_pickle=False) as npz:
                arrays={k:torch.from_numpy(npz[k].copy()) for k in npz.files}
            model=new_baseline("ssm256",seed)
            model.load_state_dict(arrays,strict=True)
            model.eval()
            models[seed]=model
            strongest[seed]=strong
    return models,strongest


def median_ms(fn):
    for _ in range(3):
        with torch.no_grad():fn()
    values=[]
    for _ in range(10):
        t=time.perf_counter()
        with torch.no_grad():fn()
        values.append((time.perf_counter()-t)*1000)
    return statistics.median(values)


def atomic_json(path: Path, obj: dict):
    tmp=path.with_suffix(".tmp")
    with tmp.open("x") as f:
        json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def train_one(seed, fit, cal, indices, strong_model, strong_meta, out_dir):
    m=original_native(seed)
    params=list(m.parameters())
    initial=hash_weights(m)
    opt=torch.optim.AdamW(params,lr=NATIVE_LR,weight_decay=.01)
    best=float("inf");best_step=None;best_state=None;no_gain=0;trace=[]
    started=time.perf_counter()
    for t in range(1,MAX_UPDATES+1):
        if time.perf_counter()-started>MAX_ARM_SECONDS or rss()>MAX_RSS_GIB:
            raise TimeoutError("F5_OWNER_CPU_OR_MEMORY_CAP")
        select=[fit[i] for i in indices[t-1].tolist()]
        data,valid=_padded(select)
        m.train();opt.zero_grad(set_to_none=True)
        loss=training_loss(m,data,valid)
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(params,1.)
        if not bool(torch.isfinite(grad)):raise ValueError("F5_NONFINITE_GRADIENT")
        opt.step()
        if t%CHECK==0:
            current=cal_mse(m,cal)
            if not math.isfinite(current):raise ValueError("F5_NONFINITE_CAL")
            trace.append({"step":t,"cal_delta_mse":current})
            if current<best:
                best=current;best_step=t;best_state=copy.deepcopy(m.state_dict());no_gain=0
            else:no_gain+=1
            if (t>=400 and no_gain>=6 and len(trace)>=3
                and max(v["cal_delta_mse"] for v in trace[-3:])-
                    min(v["cal_delta_mse"] for v in trace[-3:])<=.005):
                break
    updates=t
    if best_state is None: raise ValueError("F5_NO_SELECTED_WEIGHTS")
    m.load_state_dict(best_state);m.eval()
    restored=cal_mse(m,cal)
    if abs(restored-best)>1e-8:raise ValueError("F5_CHECKPOINT_RESTORE_DRIFT")
    wrong=cal_mse(m,cal,corrupted=True)
    # Withhold future effect when evaluating current prediction.
    sample,valid=_padded(cal[:4])
    with torch.no_grad():
        orig=native_predict(m,sample)
        modified={k:v.clone() for k,v in sample.items()}
        modified["effect"][:,3,:7]+=0.17
        altered=native_predict(m,modified)
    if not torch.equal(orig[:,:4],altered[:,:4]):
        raise ValueError("F5_FUTURE_TARGET_LEAKAGE")
    compare,_=_padded(cal[:8])
    ssm_ms=median_ms(lambda:prequential(strong_model,compare))
    hybrid_ms=median_ms(lambda:native_predict(m,compare))
    sec=time.perf_counter()-started
    weight_hash=hash_weights(m)
    filename=f"native-hybrid-{seed}.npz"
    target=out_dir/filename
    arrays={name:p.detach().cpu().numpy() for name,p in m.state_dict().items()}
    np.savez_compressed(target,**arrays)
    with np.load(target,allow_pickle=False) as v:
        if set(v.files)!=set(arrays) or any(not np.array_equal(v[k],arrays[k]) for k in arrays):
            raise ValueError("F5_CHECKPOINT_NONPICKLE_RESTORE_FAIL")
    cal_ratio=restored/strong_meta["cal_delta_mse"]
    wrong_ratio=wrong/restored
    wall_ratio=sec/strong_meta["cpu_wall_seconds"]
    inference_ratio=hybrid_ms/ssm_ms
    # No source OOD; CAL is *development* and never scientific acceptance.
    eligible=(cal_ratio<=.9 and wrong_ratio>=1.05
              and wall_ratio<=3 and inference_ratio<=3)
    return {
        "seed":seed,"original_model":"HybridPilot original 4 blocks width16 continuous events",
        "initial_sha256":initial,"selected_sha256":weight_hash,
        "checkpoint_file":filename,"npz_sha256":sha(target.read_bytes()),
        "params_static":sum(p.numel() for p in params),
        "updates_completed":updates,"selected_step":best_step,
        "cal_delta_mse":restored,"wrong_credit_mse":wrong,
        "strongest_kind":"ssm256","strongest_cal_delta_mse":strong_meta["cal_delta_mse"],
        "quality_ratio":cal_ratio,"wrong_credit_ratio":wrong_ratio,
        "training_wall_seconds":sec,"strongest_train_seconds":strong_meta["cpu_wall_seconds"],
        "training_wall_ratio":wall_ratio,"native_inference_ms":hybrid_ms,
        "ssm_inference_ms":ssm_ms,"inference_cost_ratio":inference_ratio,
        "source_target_causal_leak_negative_pass":True,
        "peak_process_RSS_gib":rss(),
        "frozen_dev_eligible":eligible,"cal_trace":trace,
    }


def run(*,f3:Path,f4:Path,receipt:Path):
    owner=owner_admission()
    freeze=locked()
    start=time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if os.getenv("CUDA_VISIBLE_DEVICES")!="-1":
        raise PermissionError("F5_GPU_NOT_APPROVED")
    fit,cal,source=read_source_zip(f3)
    if len(fit)!=128 or len(cal)!=64:raise ValueError("F5_F3_SOURCE_COUNT_DRIFT")
    strong,metrics=source_controls(f4)
    sampler=frozen_indices(len(fit))
    if sha(sampler.contiguous().numpy().tobytes())!=SAMPLE_SHA:
        raise ValueError("F5_SAMPLER_DRIFT")
    receipt.parent.mkdir(parents=True,exist_ok=True)
    out_dir=receipt.parent/"p85mr_f5_native_models"
    out_dir.mkdir(mode=0o700,exist_ok=False)
    result={
        "schema":"ttt-hybrid.p85mr-f5-native-vs-competent-conventional-dev-result.v1",
        "status":"RUNNING_NO_SCIENCE_PASS","scientific_pass":False,
        "data_source_F3_sha256":SOURCE_F3_ZIP_SHA,
        "strong_control_F4_sha256":CONTROLS_ZIP_SHA,
        "source_cohorts":source,"frozen_protocol_blob":FREEZE_BLOB,
        "sampler_sha256":SAMPLE_SHA,"owner":owner,"models":[],
        "no_ood":True,"no_external_simulator_use":True,"no_forgerunner":True,
    }
    atomic_json(receipt,result)
    try:
        for seed in SEEDS:
            row=train_one(seed,fit,cal,sampler,strong[seed],metrics[seed],out_dir)
            result["models"].append(row)
            atomic_json(receipt,result)
        result["all_three_dev_eligible"]=all(m["frozen_dev_eligible"] for m in result["models"])
        result["status"]=("DEV_CAL_ONLY_PASS_OOD_STILL_SEALED"
                          if result["all_three_dev_eligible"] else
                          "DEV_FAIL_NATIVE_HYBRID_VS_STRONG_CONTROLS")
    except BaseException as e:
        result["status"]="DEVELOPMENT_TRAINING_ABORTED_OOD_SEALED"
        result["error_class"]=type(e).__name__
        atomic_json(receipt,result)
        raise
    result["wall_process_s"]=time.perf_counter()-start
    result["peak_process_RSS_gib"]=rss()
    atomic_json(receipt,result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--f3",required=True,type=Path)
    parser.add_argument("--f4",required=True,type=Path)
    parser.add_argument("--receipt",required=True,type=Path)
    args=parser.parse_args()
    res=run(f3=args.f3, f4=args.f4, receipt=args.receipt)
    print("F5_NATIVE_DEV_RESULT="+res["status"])
    print("F5_COMPLETED_SEEDS="+str(len(res["models"])))
    print("F5_SCIENTIFIC_PASS=false")


if __name__=="__main__":
    main()
