"""P79-MR toy-only replay proving causal transfer intake without external first-use.

P78-MR trained three-weight checkpoints and the fixed eta are SHA-pinned.
There is no external trace inference/scoring entrypoint, no network use, and
no science acceptance. Probes NEVER update the state. Role labels and target
outcomes are evaluator-only: neither enters WritePolicy.encode/read.
"""
import hashlib
import json
import math
from pathlib import Path

import torch

from p77_write_policy import WritePolicy
from p79mr_trace_admission import validate_trace

P78_RECEIPT_BLOB = "c180bfa8dab735000360ba2e87f51dbe42ec3d89"
P78_RECEIPT = "p78mr_fitcal_val_dev_result.json"
SEEDS = (978201, 978203, 978207)


def _pinned_receipt():
    p = Path(__file__).with_name(P78_RECEIPT)
    raw = p.read_bytes()
    blob = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii")+raw).hexdigest()
    if blob != P78_RECEIPT_BLOB:
        raise ValueError("P79MR_P78_CHECKPOINT_DRIFT")
    receipt = json.loads(raw)
    if (receipt["schema"] !=
            "ttt-hybrid.p78mr-fit-cal-one-val-complete-dev-receipt.v1" or
            receipt["experiment"]["fast_state_floats_per_world"] != 256):
        raise ValueError("P79MR_INVALID_P78_RECEIPT")
    rows = receipt["experiment"]["seeds"]
    if tuple(r["seed"] for r in rows) != SEEDS:
        raise ValueError("P79MR_P78_SEED_DRIFT")
    return receipt


def pinned_policy(kind, seed=None):
    """Load a frozen model, never train or recalibrate from new world data."""
    receipt = _pinned_receipt()
    if kind == "fixed":
        if seed is not None:
            raise ValueError("P79MR_BAD_POLICY")
        fixed = receipt["experiment"]["selected_fixed"]
        if fixed["rule"] != "constant" or fixed["gain"] != 0.4:
            raise ValueError("P79MR_FIXED_CONTROL_DRIFT")
        return WritePolicy("fixed_eta", eta=fixed["gain"]).eval()
    if kind != "learned3" or seed not in SEEDS:
        raise ValueError("P79MR_BAD_POLICY")
    row = next(r for r in receipt["experiment"]["seeds"] if r["seed"] == seed)
    values = row["no_direction"]["selected_theta"]
    if (not isinstance(values, list) or len(values) != 3
            or not all(type(v) in (int, float) and math.isfinite(v) for v in values)):
        raise ValueError("P79MR_BAD_THETA")
    model = WritePolicy("non_directional3", seed=977201).eval()
    with torch.no_grad():
        model.theta.copy_(torch.tensor(values, dtype=model.theta.dtype))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def _events_as_facts(events):
    """Construct only the factual fields explicitly granted to the learner."""
    return {
        "action": torch.tensor([[e["action"] for e in events]], dtype=torch.float32),
        "effect": torch.tensor([[e["effect"] for e in events]], dtype=torch.float32),
    }


def toy_replay(manifest, worlds, *, kind="learned3", seed=978201):
    """Pure engineering toy replay. All external source prediction is blocked.

    A score here is NEVER evidence of external generalization, transfer or
    scientific success. Exact Git blobs and owner admission are separate.
    """
    admiss = validate_trace(manifest, worlds)
    if admiss["source_kind"] != "toy_engineering":
        raise ValueError("P79MR_OWNER_FIRST_USE_REQUIRED")
    model = pinned_policy(kind, seed if kind == "learned3" else None)
    reports = []
    with torch.no_grad():
        for world in worlds:
            carry = None
            previous = 0
            estimates = {}
            probes = sorted(world["probes"],
                            key=lambda p: (p["after_events"], p["probe_id"]))
            for probe in probes:
                cut = probe["after_events"]
                if cut > previous:
                    facts = _events_as_facts(world["events"][previous:cut])
                    carry = model.encode(facts, carry=carry)
                    previous = cut
                query = torch.tensor([probe["action"]], dtype=torch.float32)
                untouched = carry.clone()
                prediction = model.read(carry, query)[0]
                if not torch.equal(carry, untouched):
                    raise ValueError("P79MR_PROBE_MUTATED_MEMORY")
                # The ground truth is read ONLY after the read-only prediction.
                outcome = torch.tensor(probe["realized_effect"],dtype=torch.float32)
                squared_error = float((prediction-outcome).square().mean())
                estimates[probe["probe_id"]] = {
                    "probe_id": probe["probe_id"],
                    "after_events": cut,
                    "group": probe["group"],
                    "estimate": [float(x) for x in prediction],
                    "mse_toy_only": squared_error,
                }
            reports.append({
                "world_id": world["world_id"],
                "probes": [estimates[p["probe_id"]] for p in world["probes"]],
            })
    return {
        "schema": "ttt-hybrid.p79mr.toy-engineering-report.v1",
        "source_sha256": admiss["worlds_sha256"],
        "policy": kind,
        "seed": seed if kind == "learned3" else None,
        "worlds": reports,
        "classification": "TOY_ENGINEERING_ONLY",
        "external_source_evaluated": False,
        "science_pass": False,
        "deployment": "NONE",
    }
