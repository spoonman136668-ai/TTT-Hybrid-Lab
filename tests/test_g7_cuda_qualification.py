"""Negative controls for resource-safe authentic GPU admission.

These tests run under the existing G4 Linux/Windows full CPU suite.
They cannot establish authentic GPU compatibility.
"""
import json
import sys
import pytest

from scripts.g7_cuda_qualification import (
    admission_policy,
    run,
    source_module_receipt,
)

GIB = 1024**3

def policy(**overrides):
    args=dict(linux=True,cuda=True,free_bytes=8*GIB,
              total_bytes=12*GIB,min_free_gib=4.0,max_peak_gib=2.0)
    args.update(overrides)
    return admission_policy(**args)

def test_clean_resource_snapshot_admitted():
    verdict=policy()
    assert verdict["pass"] and verdict["reason"]=="HOST_ADMITTED"

def test_windows_gpu_alone_is_not_admission():
    v=policy(linux=False)
    assert not v["pass"] and v["reason"]=="REQUIRES_LINUX_OR_WSL2"

def test_linux_cpu_only_blocks():
    v=policy(cuda=False)
    assert not v["pass"] and v["reason"]=="CUDA_NOT_AVAILABLE"

def test_gpu_without_free_budget_is_blocked():
    v=policy(free_bytes=3*GIB)
    assert not v["pass"] and v["reason"]=="INSUFFICIENT_FREE_VRAM"

def test_invalid_gpu_memory_probe_blocks():
    for free,total in ((-1,12*GIB),(13*GIB,12*GIB),(0,12*GIB)):
        v=policy(free_bytes=free,total_bytes=total)
        assert not v["pass"] and v["reason"]=="INVALID_GPU_MEMORY_PROBE"

def test_invalid_resource_bounds_rejected():
    for free,peak in ((1,1),(4,0),(4,5),(-5,1)):
        with pytest.raises(ValueError, match="INVALID_RESOURCE_BOUNDS"):
            policy(min_free_gib=free,max_peak_gib=peak)

def test_unpinned_module_origin_is_rejected(tmp_path):
    with pytest.raises(RuntimeError,match="UNPINNED_MODULE_ORIGIN"):
        source_module_receipt("json", tmp_path/"json.py")

def test_actual_gpu_smoke_is_never_claimed_without_official_upstream(tmp_path):
    # On GitHub-hosted standard runners this blocks on CUDA. On any future
    # qualified GPU platform it must block because neither source exists.
    receipt=tmp_path/"g7_probe.json"
    recorded,code=run(tmp_path/"nonexistent-mamba",
                      tmp_path/"nonexistent-ttt",receipt=receipt,
                      min_free_gib=4.0,max_peak_gib=2.0)
    assert code != 0
    assert recorded["status"].startswith(("BLOCKED_","FAIL_"))
    assert recorded["status"] != "PASS_NATIVE_CUDA_ENGINEERING_ONLY"
    assert recorded["authentic_integration"]=="NOT_RUN"
    assert recorded["scientific_qualification"]=="UNVERIFIED"
    assert json.loads(receipt.read_text())["status"]==recorded["status"]
