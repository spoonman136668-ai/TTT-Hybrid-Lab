"""P85MR-F4: first actual FIT/CAL training of frozen conventional dynamics controls.

Only the already-collected F3 artifact is admitted. Never imports/runs upstream
simulator, obtains OOD, retrains F1/F2/F3 collection, or promotes a model.
The one-owner first-run GitHub workflow is the sole executable entry point.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import signal
try:
    import resource  # POSIX only; actual training is guarded to owner WSL Linux
except ImportError:
    resource = None
import subprocess
import sys
import time
import zipfile

import numpy as np
import torch

from p80mr_causalgym_trace_adapter import adapt_capture
from p85mr_dynamics_baselines import KINDS, SEEDS, new_baseline, model_metadata
from p85mr_t1_fitcal_engine import (
    BATCH, MAX_UPDATES, SAMPLE_SEED, CHECK_EVERY, LR,
    _check_fit_cal, _padded, _physical_loss, _best_checkpoint,
    calibration_mse, frozen_indices,
)

PROJECT = "spoonman136668-ai/TTT-Hybrid-Lab"
BRANCH = "research/p85mr-f4-actual-fitcal-controls-20261111"
WORKFLOW = ".github/workflows/p85mr-f4-train-controls-once.yml"
FROZEN_BLOB = "d8136313dd90cc406f00c49770ed3beeff4a1ee4"
PARENT_SHA = "00b3f45f47660af4c1016c64b656160975982227"
RUN_ID = "43073165842"
F3_SHA = "00b3f45f47660af4c1016c64b656160975982227"
ZIP_SHA = "a04697bdfba1eb93548244329f4f76fb807489aa7766f4a343de897a56caceb5"
CAPTURE_SHAS = {
    "fit": "b495d6c41fd54c6955fd83ead7569282146af45bc68b3a3dba2a63808e1d6f7b",
    "cal": "15bcbe25b492ad095fca35ee37ec97d0d141c9beb9899549a30558659ded3aef",
}
CAPTURE_N = {"fit": 128, "cal": 64}
CAPTURE_EVENTS = {"fit": 2923, "cal": 1524}
MAX_PER_ARM_SECONDS = 3600
MAX_RSS_GIB = 3.0
ROOT_IN_ZIP = "evidence/p85mr_fitcal_f3/ONCE/"
EXACT_ENTRIES = {
    "local_runs/p85mr_f3_owner_result.json",
    "local_runs/p85mr_windows_cpu.json",
    "local_runs/p85mr_wsl_cpu.json",
    *{ROOT_IN_ZIP + v for v in (
        "ATTEMPT.json", "RESULT.json",
        "fit.json", "cal.json",
        "fit.ndjson", "cal.ndjson", "fit.steps.ndjson", "cal.steps.ndjson",
    )},
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str, root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True, check=False, text=True, timeout=15)
    if completed.returncode:
        raise ValueError("F4_UNQUALIFIED_GIT_SOURCE")
    return completed.stdout.strip()


def locked_protocol(root: Path) -> dict:
    raw = (root / "experiments/p85mr_f4_controls_freeze.json").read_bytes()
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != FROZEN_BLOB:
        raise ValueError("F4_FROZEN_BLOB_MISMATCH")
    f = json.loads(raw)
    if (f["schema"] != "ttt-hybrid.p85mr-f4-actual-fitcal-conventional-controls-freeze.v1"
        or f["parent"]["sha"] != PARENT_SHA
        or f["source"]["fit_sha256"] != CAPTURE_SHAS["fit"]
        or f["source"]["cal_sha256"] != CAPTURE_SHAS["cal"]
        or f["source"]["fit_events"] != CAPTURE_EVENTS["fit"]
        or f["source"]["cal_events"] != CAPTURE_EVENTS["cal"]
        or f["parent"]["artifact_sha256"] != ZIP_SHA
        or f["training"]["max_updates"] != MAX_UPDATES
        or f["training"]["batch_size"] != BATCH
        or f["training"]["sampler_seed"] != SAMPLE_SEED
        or f["controls"]["kinds"] != list(KINDS)
        or f["controls"]["seeds"] != list(SEEDS)
        or f["execution"]["branch"] != BRANCH
        or f["execution"]["workflow"] != WORKFLOW
        or f["metrics"]["quality_competence_max_ratio_vs_zero"] != .75):
        raise ValueError("F4_FROZEN_PROTOCOL_DRIFT")
    return f


def owner_admission(root: Path) -> dict:
    expected = {
        "GITHUB_REPOSITORY": PROJECT,
        "GITHUB_ACTOR": "spoonman136668-ai",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/" + BRANCH,
        "GITHUB_WORKFLOW_REF": PROJECT + "/" + WORKFLOW + "@refs/heads/" + BRANCH,
        "GITHUB_RUN_NUMBER": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "TTT-HYBRID-GPU-WIN",
        "RUNNER_OS": "Windows",
        "RUNNER_ARCH": "X64",
        "P85MR_F4_FITCAL_CONTROLS_ONLY": "1",
    }
    if any(os.environ.get(k) != v for k, v in expected.items()):
        raise PermissionError("F4_NOT_EXACT_OWNER_ONE_SHOT")
    sha = os.environ.get("GITHUB_SHA", "")
    run_id = os.environ.get("GITHUB_RUN_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", sha) or not re.fullmatch(r"[0-9]{1,20}", run_id):
        raise PermissionError("F4_BAD_CANDIDATE_OR_RUN_ID")
    if os.name != "posix" or os.getuid() != 1000 or "microsoft" not in os.uname().release.lower():
        raise PermissionError("F4_NOT_OWNER_WSL_USER")
    if sys.version_info[:2] != (3, 11) or not torch.__version__.startswith("2.5.1"):
        raise PermissionError("F4_UNQUALIFIED_CPU_PYTHON_TORCH")
    if git("rev-parse", "HEAD", root=root) != sha:
        raise PermissionError("F4_SOURCE_SHA_DRIFT")
    if git("status", "--porcelain", "--untracked-files=all", root=root):
        raise PermissionError("F4_DIRTY_PROJECT")
    if git("remote", "get-url", "origin", root=root) not in (
        "https://github.com/" + PROJECT,
        "https://github.com/" + PROJECT + ".git",
    ):
        raise PermissionError("F4_WRONG_PROJECT_ORIGIN")
    locked_protocol(root)
    return {"repo": PROJECT, "run_id": run_id, "sha": sha,
            "runner": expected["RUNNER_NAME"], "source_run_id": RUN_ID}


def read_source_zip(path: Path) -> tuple[list[dict], list[dict], dict]:
    if not path.is_file() or sha256(path.read_bytes()) != ZIP_SHA:
        raise ValueError("F4_SOURCE_ARTIFACT_ZIP_SHA_DRIFT")
    with zipfile.ZipFile(path) as z:
        if (len(z.namelist()) != len(EXACT_ENTRIES)
            or set(z.namelist()) != EXACT_ENTRIES
            or any(i.file_size > 18_000_000 or i.compress_size > 18_000_000
                   for i in z.infolist())):
            raise ValueError("F4_UNEXPECTED_ARTIFACT_MEMBERS")
        result = json.loads(z.read(ROOT_IN_ZIP + "RESULT.json"))
        if (result["owner"]["project_sha"] != F3_SHA
            or result["owner"]["run_id"] != RUN_ID
            or result["owner"]["runner"] != "TTT-HYBRID-GPU-WIN"
            or result["owner"]["source_route"] != "P85MR-F3-INDEPENDENT-ONCE"
            or result["status"] != "SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING"
            or result["executed_actions"] != 4447
            or result["science_pass"] is not False
            or result["ood_created"] is not False
            or result["external_dev_created"] is not False
            or result["trained_baselines"] is not False
            or result["original_f1_f2_runs_replayed"] is not False):
            raise ValueError("F4_F3_RESULT_CUSTODY_FAIL")
        cohorts = []
        info = {}
        refs = set()
        for split in ("fit", "cal"):
            raw = z.read(ROOT_IN_ZIP + split + ".json")
            if sha256(raw) != CAPTURE_SHAS[split]:
                raise ValueError("F4_CAPTURE_SHA_MISMATCH")
            data = json.loads(raw)
            if (data["source_id"] != "p85mr-f3-" + split + "-" + RUN_ID
                or data["custody_ref"] != "ghrun:" + RUN_ID
                or data["producer_id"] != "ttt-hybrid-f3-owner"
                or len(data["worlds"]) != CAPTURE_N[split]
                or result["cohorts"][split]["sha256"] != CAPTURE_SHAS[split]
                or result["cohorts"][split]["valid"] != CAPTURE_N[split]
                or result["cohorts"][split]["short_episode_indices"]):
                raise ValueError("F4_UNQUALIFIED_COHORT_PROVENANCE")
            adapted = adapt_capture(data)
            if (adapted["custody"]["factual_event_count"] != CAPTURE_EVENTS[split]
                or adapted["custody"]["scientific_pass"] is not False):
                raise ValueError("F4_UNQUALIFIED_ADAPTED_SOURCE")
            worlds = []
            for w in adapted["worlds"]:
                values = w["events"]
                for event in values:
                    ref = event["receipt_ref"]
                    if ref in refs:
                        raise ValueError("F4_DUPLICATED_SOURCE_EVENT_RECEIPT")
                    refs.add(ref)
                worlds.append({
                    "action": torch.tensor([ev["action"] for ev in values], dtype=torch.float32),
                    "effect": torch.tensor([ev["effect"] for ev in values], dtype=torch.float32),
                })
            cohorts.append(worlds)
            info[split] = {"worlds": len(worlds),
                           "events": CAPTURE_EVENTS[split],
                           "sha256": CAPTURE_SHAS[split]}
    if len(refs) != 4447:
        raise ValueError("F4_NON_UNIQUE_INTERVENTION_RECEIPTS")
    # Snapshot original data; equal-world CAL weights and fixed update indices.
    fit, cal = _check_fit_cal(cohorts[0], cohorts[1])
    return fit, cal, info


def memory_gib() -> float:
    # Linux ru_maxrss is reported in KiB. No background RSS telemetry.
    if resource is None:
        raise PermissionError("F4_NONPOSIX_TRAINING_DENIED")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def selected_fingerprint(model) -> str:
    h = hashlib.sha256()
    for key, tensor in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_json(path: Path, body: dict) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("x", encoding="utf-8") as fd:
        json.dump(body, fd, sort_keys=True, indent=2, allow_nan=False)
        fd.flush()
        os.fsync(fd.fileno())
    os.replace(temp, path)


def train_one(kind, seed, train, cal, sampler, checkpoint_root: Path):
    model = new_baseline(kind, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR[kind], weight_decay=.01)
    origin = selected_fingerprint(model)
    best = float("inf")
    best_step = None
    best_state = None
    nonimproving = 0
    trace = []
    start = time.perf_counter()
    steps_completed = 0
    for step in range(1, MAX_UPDATES + 1):
        if time.perf_counter() - start > MAX_PER_ARM_SECONDS:
            raise TimeoutError("F4_CONVENTIONAL_ARM_TIME_LIMIT")
        if memory_gib() > MAX_RSS_GIB:
            raise MemoryError("F4_PROCESS_RSS_OVER_3GIB")
        picked = [train[i] for i in sampler[step - 1].tolist()]
        data, valid = _padded(picked)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, delta = _physical_loss(model, data, valid)
        loss.backward()
        grads = [p.grad for p in model.parameters()]
        if not all(x is not None and bool(torch.isfinite(x).all()) for x in grads):
            raise ValueError("F4_NONFINITE_GRADIENT")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        steps_completed = step
        if step % CHECK_EVERY == 0:
            model.eval()
            cal_result = calibration_mse(model, cal)
            current = float(cal_result["cal_delta_mse"])
            new_best, winner = _best_checkpoint(current, best, best_step, step)
            if winner == step:
                best, best_step = new_best, step
                best_state = copy.deepcopy(model.state_dict())
                nonimproving = 0
            else:
                nonimproving += 1
            trace.append({"step": step, "cal_delta_mse": current})
            if (step >= 400 and nonimproving >= 6 and len(trace) >= 3
                and max(x["cal_delta_mse"] for x in trace[-3:]) -
                    min(x["cal_delta_mse"] for x in trace[-3:]) <= .005):
                break
    if best_state is None or best_step is None:
        raise ValueError("F4_MISSING_TRAINED_CHECKPOINT")
    model.load_state_dict(best_state)
    model.eval()
    cal_result = calibration_mse(model, cal)
    if abs(cal_result["cal_delta_mse"] - best) > 1e-8:
        raise ValueError("F4_COLD_RESTORED_CAL_PARITY_FAIL")
    model_sha = selected_fingerprint(model)
    arrays = {name: value.detach().cpu().numpy()
              for name, value in model.state_dict().items()}
    target = checkpoint_root / f"{kind}-{seed}.npz"
    np.savez_compressed(target, **arrays)
    with np.load(target, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(
            not np.array_equal(saved[n], arrays[n]) for n in arrays):
            raise ValueError("F4_CHECKPOINT_PERSISTENCE_FAIL")
    return {
        "kind": kind, "model_seed": seed, "started_from_sha256": origin,
        "selected_state_sha256": model_sha,
        "selected_checkpoint_update": best_step,
        "updates_completed": steps_completed,
        "cal_delta_mse": cal_result["cal_delta_mse"],
        "zero_cal_delta_mse": cal_result["zero_cal_delta_mse"],
        "competent": cal_result["cal_delta_mse"] <= .75 * cal_result["zero_cal_delta_mse"],
        "fit_worlds": len(train), "cal_worlds": len(cal),
        "state_floats_per_world": model_metadata(model)["persistent_floats_per_world"],
        "static_parameters": model_metadata(model)["trainable_static_parameters"],
        "cpu_wall_seconds": time.perf_counter() - start,
        "process_peak_rss_gib": memory_gib(),
        "cal_checkpoints": trace,
        "npz_sha256": sha256(target.read_bytes()),
        "checkpoint_name": target.name,
    }


def run_training(*, root: Path, archive: Path, receipt: Path):
    context = owner_admission(root)
    f = locked_protocol(root)
    train_start = time.perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available() and os.getenv("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("F4_GPU_EXECUTION_DENIED")
    fit, cal, source = read_source_zip(archive)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_root = receipt.parent / "p85mr_f4_models"
    checkpoint_root.mkdir(mode=0o700, exist_ok=False)
    sampler = frozen_indices(len(fit))
    sampler_sha = sha256(sampler.contiguous().numpy().tobytes())
    result = {
        "schema": "ttt-hybrid.p85mr-f4-actual-fitcal-conventional-controls-result.v1",
        "classification": "EXTERNAL_FIT_CAL_CONTROLS_ONLY",
        "status": "RUNNING_NO_SCIENCE_PASS",
        "scientific_pass": False, "ood_consumed": False, "dev_consumed": False,
        "upstream_executed": False, "new_source_effects": False,
        "source_owner": context, "source_artifact_sha256": ZIP_SHA,
        "source_captures": source, "freeze_git_blob": FROZEN_BLOB,
        "sampler_sha256": sampler_sha,
        "models": [], "control_competence_pass": False,
    }
    atomic_json(receipt, result)
    try:
        for kind in KINDS:
            for seed in SEEDS:
                line = train_one(kind, seed, fit, cal, sampler, checkpoint_root)
                line["sampler_sha256"] = sampler_sha
                result["models"].append(line)
                atomic_json(receipt, result)
        result["control_competence_pass"] = all(x["competent"] for x in result["models"])
        result["status"] = (
            "FIT_CAL_CONVENTIONAL_CONTROLS_COMPETENT_OOD_SEALED"
            if result["control_competence_pass"] else
            "SCIENCE_CONTROL_FAIL_OOD_SEALED"
        )
    except BaseException as e:
        result["status"] = "CONTROL_TRAINING_ABORTED_OOD_SEALED"
        result["error_class"] = type(e).__name__
        atomic_json(receipt, result)
        raise
    result["process_wall_seconds"] = time.perf_counter() - train_start
    result["process_peak_rss_gib"] = memory_gib()
    atomic_json(receipt, result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-checkout", required=True, type=Path)
    parser.add_argument("--source-archive", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    result = run_training(root=args.project_checkout.resolve(),
                          archive=args.source_archive.resolve(),
                          receipt=args.receipt.resolve())
    print("P85MR_F4_STATUS=" + result["status"])
    print("P85MR_F4_ARMS=" + str(len(result["models"])))
    print("P85MR_F4_SCIENTIFIC_PASS=false")
    if not result["control_competence_pass"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
