"""C4 measured DEV-only capability review of C3 factual checkpoint ancestry.

Reuses the preexisting P3 native causal gradient trainer, C2's isolated
candidate loader, and P3's already-tested capability registry. Actually
measures both old/post-reversal performance from saved learned states.
The P2 algebraic stream overlaps training; NOT scientific heldout evidence.
No promotion, no original-world first-use, no runner/controller activation.
"""
from __future__ import annotations

from copy import deepcopy
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from native_continuous_state import ContinuousState
from native_capability_registry import (
    SCHEMA as POLICY_SCHEMA, REVIEW_SCHEMA, freeze_policy,
    read_policy, review_challenger,
)
from native_transactional_resume import prepare_isolated_candidate, _rng_snapshot, _rng_restore
from native_training_checkpoint import validate_chain
from scripts.native_prequential_gate import NativeCausalEpisode
from scripts import native_p3_continuity as factual

FROZEN = ROOT / "experiments" / "native_c4_measured_retention_dev_frozen.json"
SCHEMA = "ttt-hybrid.c4.measured-dev-capability-review.v1"


def _canonical(x: object) -> bytes:
    return (json.dumps(x, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode()


def _require(ok: bool, reason: str):
    if not ok:
        raise AssertionError("C4_" + reason)


def _spec():
    raw = FROZEN.read_bytes()
    doc = json.loads(raw)
    _require(doc["schema"] == "ttt-hybrid.c4.measured-dev-retention-protocol.v1"
             and doc["source_parent_sha"] == "78644c0548b63d0d0b2fcc807ab4af8748670cf4"
             and doc["facts"]["train_steps"] == factual.TOTAL_EVENTS == 80
             and doc["facts"]["champion_steps"] == factual.CHECKPOINT_AT == 40
             and doc["initializations"] == list(factual.MODEL_INITIALIZATIONS)
             and doc["facts"]["evaluation_actions_per_regime"] == 16
             and doc["decisions"]["scientific_first_use"] is False
             and doc["decisions"]["accepted_model_promotion"] is False,
             "FROZEN_CONFIG_INVALID")
    return doc


def _peak_process_rss_bytes() -> int:
    """OS-reported process high-watermark, not total GPU/external storage cost."""
    if os.name == "nt":
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        data = Counters()
        data.cb = ctypes.sizeof(data)
        # Win32 APIs return pointer-width HANDLEs. The implicit ctypes
        # c_int default truncates GetCurrentProcess() on 64-bit Windows and
        # makes the otherwise-valid PSAPI query fail with an invalid handle.
        kernel32 = ctypes.windll.kernel32
        psapi = ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = (
            ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong)
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        ok = psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(data), data.cb)
        if not ok:
            raise RuntimeError("C4_PEAK_WORKING_SET_UNAVAILABLE")
        return int(data.PeakWorkingSetSize)
    import resource
    # Linux/WSL ru_maxrss is KiB; Windows uses the explicit Win32 API above.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _measure(model, native: dict, regime: str, samples: int) -> list[float]:
    """Read-only copy of the saved native state per query; no training labels used."""
    _require(regime in ("old_regime_mse", "postreversal_mse"),
             "UNKNOWN_EVALUATION_CAPABILITY")
    _require(samples == 16, "EVALUATION_SAMPLE_DRIFT")
    saved = deepcopy(native)
    parameters = deepcopy(model.state_dict())
    errors = []
    with torch.no_grad():
        for i in range(samples):
            t = i if regime == "old_regime_mse" else 40 + i
            action = torch.tensor([[1.0 if i % 2 else -1.0]], dtype=torch.float32)
            sensor = torch.zeros((1, 1), dtype=torch.float32)
            dt = torch.tensor([[0.2]], dtype=torch.float32)
            state = ContinuousState(
                native["recurrent"].detach().clone(),
                native["fast_weights"].detach().clone(),
                native["history"].detach().clone())
            predicted, _ = model.step(sensor, action, dt, state)
            observed = factual.analytic_consequence(action, t)
            error = float((predicted - observed).square().item())
            _require(math.isfinite(error) and error >= 0.0,
                     "NONFINITE_MEASUREMENT")
            errors.append(error)
    _require(factual._exact_equal(native, saved)
             and factual._exact_equal(model.state_dict(), parameters),
             "EVALUATION_ALTERED_SAVED_LEARNING")
    return errors


def _policy(spec: dict, identity: dict, champion_sha: str) -> dict:
    evaluation_def = {
        "actions_per_regime": spec["facts"]["evaluation_actions_per_regime"],
        "sequence": spec["facts"]["action_sequence"],
        "labels": spec["facts"]["labels"],
        "evaluation": spec["facts"]["evaluation"],
        "data_quality": spec["facts"]["evaluation_data"],
    }
    return {
        "schema": POLICY_SCHEMA,
        "champion_checkpoint_sha256": champion_sha,
        "protocol_sha256": identity["protocol_sha256"],
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "evaluation_split_sha256": hashlib.sha256(_canonical(evaluation_def)).hexdigest(),
        "capabilities": deepcopy(spec["capabilities"]),
        "budget": deepcopy(spec["budget"]),
    }


def measure_one(initialization: int, scratch: Path, spec: dict | None = None) -> dict:
    """Evaluate actual P3 step-40 and step-80 learned checkpoints read-only."""
    spec = _spec() if spec is None else spec
    _require(initialization in spec["initializations"],
             "UNREGISTERED_DEV_INITIALIZATION")
    _require(scratch.is_dir() and not any(scratch.iterdir()),
             "SCRATCH_MUST_BE_EMPTY")
    started = time.perf_counter()
    identity = factual.identity()
    trained, optimizer = factual.new_model_and_optimizer(initialization)
    episode = NativeCausalEpisode(trained)
    root = factual.save_completed(scratch, identity, trained, optimizer, episode, 0, None)
    for t in range(factual.CHECKPOINT_AT):
        factual.transact(trained, optimizer, episode, t)
    champion_sha = factual.save_completed(
        scratch, identity, trained, optimizer, episode, factual.CHECKPOINT_AT, root)
    for t in range(factual.CHECKPOINT_AT, factual.TOTAL_EVENTS):
        factual.transact(trained, optimizer, episode, t)
    challenger_sha = factual.save_completed(
        scratch, identity, trained, optimizer, episode, factual.TOTAL_EVENTS, champion_sha)
    _require(validate_chain(scratch, challenger_sha, identity)["steps"] == 80,
             "CHALLENGER_LINEAGE")

    champion = prepare_isolated_candidate(
        scratch, tip_sha256=champion_sha, identity=identity,
        expected_position=40, expected_parent_sha256=root,
        minimum_steps=40, minimum_consumed_examples=40,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        protected_model=trained, protected_optimizer=optimizer)
    challenger = prepare_isolated_candidate(
        scratch, tip_sha256=challenger_sha, identity=identity,
        expected_position=80, expected_parent_sha256=champion_sha,
        minimum_steps=80, minimum_consumed_examples=80,
        make_fresh_pair=lambda: factual.new_model_and_optimizer(initialization),
        protected_model=champion.model, protected_optimizer=champion.optimizer)
    _require(champion.sampler == {"position": 40, "event_index": 40, "pending": False}
             and challenger.sampler == {"position": 80, "event_index": 80, "pending": False},
             "TRAINING_CURSOR_MISMATCH")
    _require(bool(torch.count_nonzero(champion.native_state["fast_weights"]))
             and bool(torch.count_nonzero(challenger.native_state["fast_weights"])),
             "NO_FACTUAL_NATIVE_MEMORY")

    policy = _policy(spec, identity, champion_sha)
    policy_dir = scratch.parent / (scratch.name + "-c4-policy")
    _require(not policy_dir.exists(), "POLICY_SCRATCH_COLLISION")
    policy_dir.mkdir()
    digest = freeze_policy(policy_dir, policy)
    frozen = read_policy(policy_dir, digest)
    measurements = []
    vectors = {}
    for cap in spec["capabilities"]:
        capid = cap["id"]
        k = spec["facts"]["evaluation_actions_per_regime"]
        left = _measure(champion.model, champion.native_state, capid, k)
        right = _measure(challenger.model, challenger.native_state, capid, k)
        mean_left = sum(left) / len(left)
        mean_right = sum(right) / len(right)
        evidence = {
            "id": capid, "initialization": initialization,
            "champion_checkpoint_sha256": champion_sha,
            "challenger_checkpoint_sha256": challenger_sha,
            "champion_squared_errors": left, "challenger_squared_errors": right,
        }
        measurements.append({
            "id": capid, "champion": mean_left, "challenger": mean_right,
            "samples": len(left),
            "evidence_sha256": hashlib.sha256(_canonical(evidence)).hexdigest(),
        })
        vectors[capid] = evidence
    wall = time.perf_counter() - started
    rss = _peak_process_rss_bytes()
    _require(wall <= spec["budget"]["max_wall_seconds"],
             "RUNTIME_BUDGET_EXCEEDED")
    receipt = {
        "schema": REVIEW_SCHEMA, "policy_sha256": digest,
        "champion_checkpoint_sha256": champion_sha,
        "challenger_checkpoint_sha256": challenger_sha,
        "protocol_sha256": frozen["protocol_sha256"],
        "harness_sha256": frozen["harness_sha256"],
        "evaluation_split_sha256": frozen["evaluation_split_sha256"],
        "measurements": measurements,
        "cost": {"wall_seconds": wall, "gpu_seconds": 0, "resident_bytes": rss},
    }
    outcome = review_challenger(frozen, digest, receipt)
    # Negative comparison is derived from ACTUALLY measured champion values,
    # not injected fake performance. No improvement -> must never be eligible.
    unchanged = deepcopy(receipt)
    for row in unchanged["measurements"]:
        row["challenger"] = row["champion"]
    negative = review_challenger(frozen, digest, unchanged)
    _require(negative["status"] == "BLOCKED"
             and "NO_PREDECLARED_MEANINGFUL_GAIN" in negative["reasons"],
             "UNCHANGED_CONTROL_WAS_NOT_REJECTED")
    _require(outcome["promoted"] is False and outcome["scientifically_accepted"] is False
             and outcome["evidence_authenticated"] is False,
             "INADMISSIBLE_AUTOMATIC_PROMOTION")
    return {
        "initialization": initialization,
        "status": "PASS_MEASURED_DEV_POLICY_INTEGRITY_ONLY",
        "champion_tip_sha256": champion_sha,
        "challenger_tip_sha256": challenger_sha,
        "root_sha256": root,
        "frozen_policy_sha256": digest,
        "measurements": measurements,
        "measurement_vectors": vectors,
        "review": outcome,
        "unchanged_control": negative,
        "cost_quality": "OS_PROCESS_PEAK_RSS_ONLY_NO_WHOLE_PROCESS_OR_BACKUP_COST",
        "evaluation_data_quality": "PUBLIC_TRAINING_REGIMES_NOT_HELDOUT",
        "science_first_use": "NOT_RUN",
        "training_activated": False,
        "model_promoted": False,
    }


def run() -> dict:
    _require(os.getenv("NATIVE_C4_ENGINEERING_ONLY", "1") == "1"
             and os.getenv("NATIVE_P3_ENGINEERING_ONLY", "1") == "1",
             "SCIENCE_FIRST_USE_FORBIDDEN")
    started = time.perf_counter()
    torch.set_num_threads(1)
    host_rng = _rng_snapshot()
    try:
        spec = _spec()
        with tempfile.TemporaryDirectory(prefix="hybrid-c4-measured-policy-") as tmp:
            records = []
            for seed in spec["initializations"]:
                d = Path(tmp) / str(seed)
                d.mkdir()
                records.append(measure_one(seed, d, spec))
    finally:
        _rng_restore(host_rng)
    return {
        "schema": SCHEMA,
        "status": "PASS_REAL_MEASUREMENTS_DEV_POLICY_ONLY",
        "source_sha": factual.source_sha(),
        "record_count": len(records),
        "policy_reviews": [r["review"]["status"] for r in records],
        "scientific_first_use": "NOT_RUN",
        "unseen_transfer_qualified": False,
        "evaluation_data_independent": False,
        "scientifically_accepted": False,
        "production_accepted_pointer_changed": False,
        "live_training_activated": False,
        "wall_seconds": round(time.perf_counter() - started, 3),
        "records": records,
    }


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({
        "status": result["status"], "source_sha": result["source_sha"],
        "seed_count": result["record_count"],
        "measured_policy_reviews": result["policy_reviews"],
        "metrics": [{m["id"]: {"champion": m["champion"],
                              "challenger": m["challenger"]}
                     for m in r["measurements"]}
                    for r in result["records"]],
        "scientific_first_use": result["scientific_first_use"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
