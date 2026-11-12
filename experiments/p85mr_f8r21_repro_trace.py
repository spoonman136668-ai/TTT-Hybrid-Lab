"""F8-R21: CPU reproducibility attribution, NOT a scientific experiment.

Factual source and native/conventional models reuse registered R20 DEV source.
New trace records first action/effect, backward gradient and optimizer state
without changing the R20 science protocol or calling an external simulator.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
from pathlib import Path
import sys
import time

import torch

from experiments import p85mr_f8r20_factorial_transfer as r20
from experiments import p85mr_f8r19_balanced_return as r19

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "experiments/p85mr_f8r21_repro_trace_freeze.json"
FREEZE_BLOB = "cf54a6a4f502a4fc4cd180f98e0ac03596036202"
SOURCE_BLOB = "78700dabd9b4301318148b4536e9df21c64d2617"
SCHEMA = "ttt-hybrid.f8r21-source-neutral-cpu-repro-trace.v1"
ARMS = ("native_return_focus", "linear_return_only")
SEEDS = (985201, 985203)
MODES = ("original_seeded", "canonical_dyadic")
CHECKPOINTS = (0, 1, 8)
NUM_STEPS = 8
BATCH = 8
REPLICAS = 2
SAMPLER_SEED = 985299


def git_blob(content: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(content)).encode() + b"\0" + content
    ).hexdigest()


def locked() -> dict:
    raw = FREEZE.read_bytes()
    if git_blob(raw) != FREEZE_BLOB:
        raise ValueError("F8R21_FREEZE_BLOB_DRIFT")
    if git_blob((ROOT / "experiments/p85mr_f8r20_factorial_transfer.py").read_bytes()) != SOURCE_BLOB:
        raise ValueError("F8R21_R20_SOURCE_DRIFT")
    cfg = json.loads(raw)
    p, f, s = (cfg["parent"], cfg["fixed_protocol"], cfg["stop"])
    if (cfg["schema"] != SCHEMA
            or cfg["stage"] != "PROSPECTIVE_FROZEN_BEFORE_TRACE_CODE_AND_RESULTS"
            or cfg["classification"] != "ENGINEERING_DIAGNOSTIC_NOT_SCIENTIFIC_EXPERIMENT"
            or p["sha"] != "2f13dfa663634e926e29aa2c6c94845ca8efb7e7"
            or p["r20_source_blob"] != SOURCE_BLOB
            or p["r20_freeze_blob"] != "8a181c6fd71c9dc6d4c4d640396d19920eb74175"
            or f["arms"] != list(ARMS) or f["seeds"] != list(SEEDS)
            or f["modes"] != list(MODES)
            or f["update_steps"] != NUM_STEPS
            or f["record_after_steps"] != list(CHECKPOINTS)
            or f["batch_size"] != BATCH or f["sampler_seed"] != SAMPLER_SEED
            or f["replicas_per_case"] != REPLICAS
            or f["same_host_two_replicas_must_match"] is not True
            or any(x is not True for x in s.values())
            or any(x is not False for x in cfg["authority"].values())):
        raise ValueError("F8R21_SOURCE_OR_SCOPE_DRIFT")
    r20.locked()
    return cfg


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha(t: torch.Tensor) -> str:
    a = t.detach().cpu().contiguous()
    if a.is_floating_point() and not bool(torch.isfinite(a).all()):
        raise ValueError("F8R21_NONFINITE_TENSOR")
    return _digest(a.numpy().tobytes())


def full_weights(model) -> dict:
    layers = {name: tensor_sha(t)
              for name,t in sorted(model.state_dict().items())}
    return {"all":r20.r18.r5.f5.hash_weights(model), "layers":layers}


def canonical_weights(model):
    """Precisely representable source-neutral counterfactual initialization."""
    with torch.no_grad():
        for name, p in sorted(model.named_parameters()):
            if not p.is_floating_point():
                raise ValueError("F8R21_NONFLOAT_PARAMETER")
            offset = hashlib.sha256(name.encode()).digest()[0]
            values = [((i + offset) % 17 - 8) / 128.0
                      for i in range(p.numel())]
            p.copy_(torch.tensor(values, dtype=p.dtype, device="cpu").reshape(p.shape))
    return model


def model_for(arm: str, seed: int, mode: str):
    if (arm not in ARMS or type(seed) is not int or seed not in SEEDS
            or mode not in MODES):
        raise ValueError("F8R21_UNREGISTERED_ARM_SEED_MODE")
    m = r19._model(r20.MAPPING[arm], seed).cpu()
    if mode == "canonical_dyadic":
        canonical_weights(m)
    return m


def environment() -> dict:
    capability = "UNAVAILABLE"
    if hasattr(torch._C, "_get_cpu_capability"):
        capability = str(torch._C._get_cpu_capability())
    return {
        "platform":platform.platform(),
        "machine":platform.machine(),
        "processor":platform.processor(),
        "python":sys.version.split()[0],
        "torch":str(torch.__version__),
        "torch_config_sha256":_digest(torch.__config__.show().encode("utf-8")),
        "cpu_capability":capability,
        "torch_mkl":bool(torch.backends.mkl.is_available()),
        "torch_mkldnn":bool(torch.backends.mkldnn.is_available()),
        "mkldnn_enabled":bool(torch.backends.mkldnn.enabled),
        "threads":torch.get_num_threads(),
        "interop_threads":torch.get_num_interop_threads(),
        "deterministic_algorithms":bool(torch.are_deterministic_algorithms_enabled()),
        "cpu_count":os.cpu_count(),
    }


def _data(fit: list[dict], indices: list[int]) -> tuple[dict,torch.Tensor]:
    batch, valid = r20.r18.r5.f4._padded(
        [r20.factual(fit[i]) for i in indices])
    if valid.shape != (BATCH,10) or not bool(valid[:,-1].all()):
        raise ValueError("F8R21_FACTUAL_BATCH_DRIFT")
    return batch, valid


def _physical_objective(arm, model, batch, valid) -> torch.Tensor:
    if arm == "native_return_focus":
        return r19._objective("native_return_focus", model, batch, valid)
    pred = r19._pred("linear_return_focus", model, batch)
    return (pred[:,-1,:4]-batch["effect"][:,-1,:4]).square().mean()


def trace(arm: str, seed: int, mode: str,
          fit: list[dict], batches: list[list[int]]) -> dict:
    model = model_for(arm, seed, mode)
    original = full_weights(model)
    rate = .002 if arm.startswith("native_") else .001
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=rate, weight_decay=.01)
    states = [{"update":0, **original}]
    first = None
    for step, ix in enumerate(batches, start=1):
        model.train()
        batch, valid = _data(fit,ix)
        optimizer.zero_grad(set_to_none=True)
        if step == 1:
            # Inspection-only first forward; it cannot supply the current
            # factual effect to the model, as causal inputs are shifted.
            with torch.no_grad():
                predicted = r19._pred(r20.MAPPING[arm], model, batch)
            pred_signature = tensor_sha(predicted)
            input_signature = {k:tensor_sha(v) for k,v in batch.items()}
            mask_signature = tensor_sha(valid)
        loss = _physical_objective(arm, model, batch, valid)
        if not bool(torch.isfinite(loss)):
            raise ValueError("F8R21_NONFINITE_LOSS")
        loss.backward()
        gradients = {
            name:(tensor_sha(p.grad) if p.grad is not None else "NONE")
            for name,p in sorted(model.named_parameters())
        }
        if (not any(v != "NONE" for v in gradients.values())
                or all(v == "NONE" for v in gradients.values())):
            raise ValueError("F8R21_MISSING_GRADIENTS")
        if step == 1:
            first = {
                "input":input_signature, "valid_mask":mask_signature,
                "predicted_first8_sha256":pred_signature,
                "loss_float64_repr":repr(float(loss.detach())),
                "loss_tensor_sha256":tensor_sha(loss),
                "preclip_layer_grad_sha256":gradients
            }
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        if step in CHECKPOINTS:
            states.append({"update":step, **full_weights(model)})
    return {
        "arm":arm, "seed":seed, "mode":mode,
        "first":first, "weights":states,
        "changed_parameters":original["all"] != states[-1]["all"]
    }


def run() -> dict:
    locked()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    fit = r20.cohort("fit")
    if len(fit) != 192:
        raise ValueError("F8R21_INCOMPLETE_SOURCE_NEUTRAL_FIT")
    rng = torch.Generator(device="cpu")
    rng.manual_seed(SAMPLER_SEED)
    batches = torch.randint(0,len(fit),(NUM_STEPS,BATCH),
                            generator=rng).tolist()
    indices_sha = _digest(json.dumps(batches,sort_keys=True).encode())
    start=time.perf_counter()
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            for mode in MODES:
                independent=[trace(arm,seed,mode,fit,batches)
                             for _ in range(REPLICAS)]
                parity = independent[0] == independent[1]
                rows.append({"arm":arm,"seed":seed,"mode":mode,
                             "within_process_replica_parity":parity,
                             "trace":independent[0]})
    if not all(x["within_process_replica_parity"] for x in rows):
        raise ValueError("F8R21_WITHIN_PROCESS_NONDETERMINISM")
    if not all(x["trace"]["changed_parameters"] for x in rows):
        raise ValueError("F8R21_NO_OPTIMIZER_PROGRESS")
    return {
        "schema":SCHEMA,
        "classification":"ENGINEERING_DIAGNOSTIC_NOT_SCIENTIFIC_EXPERIMENT",
        "parent_sha":"2f13dfa663634e926e29aa2c6c94845ca8efb7e7",
        "freeze_blob":FREEZE_BLOB,
        "r20_original_source_blob":SOURCE_BLOB,
        "environment":environment(),
        "batch_indices_sha256":indices_sha,
        "cases":rows,
        "models_traced":len(rows), "total_short_optimizer_updates":
            len(rows)*REPLICAS*NUM_STEPS,
        "all_same_host_replica_parity":True,
        "original_external_effects":0,
        "spent_F7_replay":False,
        "independent_scientific_pass":False,
        "deployment":"NONE",
        "diagnostic_elapsed_seconds":time.perf_counter()-start,
    }


def retain_ci_receipt(result: dict) -> None:
    if os.environ.get("GITHUB_ACTIONS") != "true":
        return
    path = ROOT / "local_runs/p85mr_f8r21_repro_trace.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        raise FileExistsError("F8R21_NO_TRACE_OVERWRITE")
    path.write_text(json.dumps(result,sort_keys=True,indent=2,
                               allow_nan=False)+"\n")


def refuse_original_source(*_args,**_kwargs):
    raise PermissionError("F8R21_ORIGINAL_SOURCE_FIRSTUSE_NOT_AUTHORIZED")
