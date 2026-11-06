"""Retained G7-R5 one-time CUDA provenance and no-retrigger contracts.

CI only parses source and checks immutable evidence markers. No WSL or CUDA runs.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-r5-single-cuda-kernel.yml"
WRAPPER = ROOT / "scripts/g7_r5_oneshot.py"
KERNEL = ROOT / "scripts/g7_cuda_qualification.py"
RECEIPT = ROOT / "docs/G7_R5_FIRST_CUDA_KERNEL.md"


def test_one_time_workflow_retired_after_first_success():
    assert not WORKFLOW.exists(), "Consumed G7-R5 execution entrypoint must be absent"


def test_r3_environment_and_exact_custody_are_still_frozen():
    wrapper = WRAPPER.read_text(encoding="utf-8")
    ast.parse(wrapper, filename=str(WRAPPER))
    for marker in (
        'R3_SOURCE_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"',
        'R3_PINS_SHA256 = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"',
        'R3_MANIFEST_SHA256 = "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df"',
        'TORCH = "2.9.0+cu128"', 'TRITON = "3.5.0"',
        'guard(package_sha256 == R3_MANIFEST_SHA256',
        'git(root, "hash-object", "--", rel)',
        'git(root, "status", "--porcelain", "--untracked-files=all")',
        "CHILD_TIMEOUT_SECONDS = 175", "MIN_FREE_GIB = 4.0",
        "MAX_PEAK_GIB = 1.0", "kernel_only=True",
        '"PASS_ONE_BOUNDED_NATIVE_CUDA_MATMUL_ONLY"',
        '"scientific": "UNVERIFIED"', '"training": "NOT_RUN"',
        '"native_mamba_forward": "NOT_RUN"',
        '"native_ttt_forward": "NOT_RUN"',
    ):
        assert marker in wrapper


def test_native_kernel_isolated_from_integrated_model_forward():
    script = KERNEL.read_text(encoding="utf-8")
    ast.parse(script, filename=str(KERNEL))
    assert script.index("if kernel_only:") < script.index(
        "from run_gate3 import run as native_gate3")
    for required in (
        '"PASS_NATIVE_CUDA_KERNEL_ONLY"',
        '"BLOCKED_R5_RUNTIME_DEVICE_IDENTITY"',
        '"NOT_PROBED_KERNEL_ONLY"',
        "torch.cuda.set_per_process_memory_fraction",
        'torch.randn((16, 16), device="cuda", dtype=torch.float32)',
        "torch.cuda.max_memory_allocated(0)",
        "if peak > max_peak_gib * (1024**3):",
    ):
        assert required in script
    wrapper = WRAPPER.read_text(encoding="utf-8").lower()
    for forbidden in ("pip install", "apt install", "sudo ", "git fetch",
                      "git clone", "git push", "torch.compile(",
                      "subprocess.popen(", "shell=true", "os.system(",
                      "workflow_dispatch", "wsl --install"):
        assert forbidden not in wrapper


def test_immutable_receipt_and_science_boundary_retained():
    data = RECEIPT.read_text(encoding="utf-8")
    for marker in (
        "42228676232",
        "13506388398",
        "067bf951a89e7f61ad9b50cc1491566129ee0768",
        "1631b00d0cc0bf6b55c34c00e44901abbb3fe4be68a837f5ef6375b460d3500c",
        "8,523,264",
        "PASS_ONE_BOUNDED_NATIVE_CUDA_MATMUL_ONLY",
        "SCIENTIFIC",
        "NOT_AUTHORIZED",
    ):
        assert marker in data.upper() if marker == "SCIENTIFIC" else marker in data
