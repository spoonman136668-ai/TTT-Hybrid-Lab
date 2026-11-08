"""P5 isolated engineering rehearsal: abrupt child-process death during P2 copy.

One-run synthetic tensor/optimizer state only. No actual training owner, original
science source, off-host backup, model promotion, or physical power-failure proof.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from native_checkpoint_recovery import RecoveryRejected, mirror_verified_chain
from native_training_checkpoint import (
    CheckpointRejected, load_checkpoint, save_checkpoint, validate_chain
)


def _identity() -> dict:
    fields = ("architecture_sha256", "interface_sha256", "protocol_sha256",
              "data_split_sha256", "harness_sha256", "environment_sha256")
    return {"source_sha": "b" * 40,
            **{key: hashlib.sha256(("p5-dev:" + key).encode()).hexdigest()
               for key in fields}}


def _model():
    net = torch.nn.Linear(2, 1)
    opt = torch.optim.AdamW(net.parameters(), lr=0.01)
    return net, opt


def _same_mapping(a: dict, b: dict) -> bool:
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor):
            if not isinstance(y, torch.Tensor) or not torch.equal(x, y):
                return False
        elif isinstance(x, dict):
            if not isinstance(y, dict) or not _same_mapping(x, y):
                return False
        elif isinstance(x, list):
            if not isinstance(y, list) or x != y:
                return False
        elif x != y:
            return False
    return True


def rehearse(scratch: Path) -> dict:
    """Real process death followed by explicitly verified replica cold restore."""
    scratch = Path(scratch)
    if scratch.exists():
        if any(scratch.iterdir()):
            raise ValueError("P5_SCRATCH_MUST_BE_EMPTY")
    else:
        scratch.mkdir(parents=True)
    primary, replica, staging = (scratch / x for x in
                                 ("primary", "replica", "staging"))
    for item in (primary, replica, staging):
        item.mkdir()
    identity = _identity()
    torch.manual_seed(62911)
    net, opt = _model()
    native = {"recurrent": torch.ones(1, 2),
              "fast_weights": torch.ones(1, 2, 2),
              "history": torch.ones(1, 1, 2)}
    root = save_checkpoint(primary, model=net, optimizer=opt,
                           identity=identity, native_state=native,
                           sampler_state={"position": 0}, steps=0,
                           consumed_examples=0, parent_manifest_sha256=None)
    x = torch.tensor([[.25, .75]])
    opt.zero_grad(set_to_none=True)
    net(x).square().sum().backward()
    opt.step()
    native["fast_weights"].add_(0.125)
    tip = save_checkpoint(primary, model=net, optimizer=opt,
                          identity=identity, native_state=native,
                          sampler_state={"position": 1}, steps=1,
                          consumed_examples=1, parent_manifest_sha256=root)
    expected_model = deepcopy(net.state_dict())
    expected_optimizer = deepcopy(opt.state_dict())
    code = """
import json, os, sys
sys.path.insert(0, sys.argv[5])
import native_checkpoint_recovery as recovery
original = recovery._publish_object
counter = [0]
def abruptly_terminate_after_root(*args):
    n = original(*args)
    counter[0] += 1
    if counter[0] == 2:
        os._exit(73)
    return n
recovery._publish_object = abruptly_terminate_after_root
recovery.mirror_verified_chain(sys.argv[1], sys.argv[2],
    tip_sha256=sys.argv[3], identity=json.loads(sys.argv[4]))
os._exit(95)
"""
    started = time.perf_counter()
    child = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(primary), str(replica),
         tip, json.dumps(identity, sort_keys=True), str(ROOT)],
        cwd=ROOT, capture_output=True, text=True, timeout=60, check=False,
    )
    if child.returncode != 73:
        raise AssertionError("P5_CHILD_DID_NOT_CRASH_AT_EXPECTED_POINT:" +
                             str(child.returncode) + ":" + child.stderr[-500:])
    # Crash published only complete root objects, not a qualified tip.
    if (replica / (tip + ".json")).exists():
        raise AssertionError("P5_PARTIAL_COPY_ADVERTISED_TIP")
    try:
        validate_chain(replica, tip, identity)
    except (CheckpointRejected, FileNotFoundError):
        pass
    else:
        raise AssertionError("P5_PARTIAL_REPLICA_WRONGLY_ACCEPTED")

    copied = mirror_verified_chain(primary, replica, tip_sha256=tip,
                                   identity=identity)
    if copied["state"] != "COPY_VERIFIED_UNAUTHENTICATED":
        raise AssertionError("P5_REPLICA_CUSTODY_MISLABELLED")
    if copied["objects_verified"] != 4:
        raise AssertionError("P5_REPLICA_INCOMPLETE")

    # Damage primary after the replica has been verified. There must be no
    # implicit fallback: an explicit alternative source and tip are required.
    damaged = sorted(primary.glob("*.pt"))[0]
    damaged.write_bytes(damaged.read_bytes() + b"damage")
    try:
        mirror_verified_chain(primary, staging, tip_sha256=tip,
                              identity=identity)
    except (CheckpointRejected, RecoveryRejected):
        pass
    else:
        raise AssertionError("P5_CORRUPTED_PRIMARY_WAS_ACCEPTED")
    if list(staging.iterdir()):
        raise AssertionError("P5_FAILED_PRIMARY_LEFT_RECOVERED_OBJECTS")
    restored = mirror_verified_chain(replica, staging, tip_sha256=tip,
                                     identity=identity)
    cold, cold_opt = _model()
    recovered_native, sampler = load_checkpoint(
        staging, tip, model=cold, optimizer=cold_opt, identity=identity)
    if not _same_mapping(expected_model, cold.state_dict()):
        raise AssertionError("P5_COLD_MODEL_MISMATCH")
    if not _same_mapping(expected_optimizer, cold_opt.state_dict()):
        raise AssertionError("P5_COLD_OPTIMIZER_MISMATCH")
    if not _same_mapping(native, recovered_native) or sampler != {"position": 1}:
        raise AssertionError("P5_NATIVE_MEMORY_OR_SAMPLER_MISMATCH")
    if any((p / marker).exists() for p in (primary, replica, staging)
           for marker in ("accepted", "latest")):
        raise AssertionError("P5_UNAUTHORIZED_POINTER")
    return {
        "schema": "ttt-hybrid.crash-rehearsal-eng.v1",
        "status": "PASS_SYNTHETIC_PROCESS_CRASH_RECOVERY_ONLY",
        "child_exit_code": child.returncode,
        "partial_tip_rejected": True,
        "corrupt_primary_rejected": True,
        "replica_chain_verified": True,
        "cold_model_optimizer_native_sampler_equal": True,
        "tip_sha256": tip,
        "root_sha256": root,
        "objects_verified": restored["objects_verified"],
        "source_device": "same" if os.stat(primary).st_dev == os.stat(replica).st_dev
                         else "different",
        "off_host_failure_domain_verified": False,
        "owner_authenticated": False,
        "scientific_accepted": False,
        "model_promoted": False,
        "training_activated": False,
        "duration_seconds": round(time.perf_counter() - started, 3),
    }


def main() -> int:
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="hybrid-p5-process-crash-") as temp:
        result = rehearse(Path(temp))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    result["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    args.receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps({"status": result["status"], "child_exit_code": 73,
                      "duration_seconds": result["duration_seconds"],
                      "off_host_failure_domain_verified": False,
                      "receipt": str(args.receipt)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
