"""Retained G7-R4 runtime admission provenance and no-kernel contract tests.

Run 42216104266 succeeded; consumed main-push entrypoint is retired.
Never import torch or invoke WSL in CI from this test module.
"""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "scripts/g7_r4_runtime_probe.py"
RETIRED_WORKFLOW = ROOT / ".github/workflows/g7-r4-runtime-admission.yml"


def test_one_shot_entrypoint_retired():
    assert not RETIRED_WORKFLOW.exists()


def test_runtime_code_syntax_and_frozen_source_baseline():
    src = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(PROBE))
    for marker in (
        'R3_QUALIFIED_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"',
        'R3_SOURCE_PINS_SHA256 = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"',
        'R3_PACKAGE_MANIFEST_SHA256 = "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df"',
        'R3_TORCH = "2.9.0+cu128"', 'R3_TRITON = "3.5.0"',
        '"kernel_execution": "NOT_RUN"', '"model_construction": "NOT_RUN"',
        '"gpu_tensor_allocation": "NOT_REQUESTED"',
        '"native_extension_build": "NOT_RUN"',
        '"system_installation": "NOT_RUN"', '"scientific": "UNVERIFIED"',
    ):
        assert marker in src
    embedded = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ("CHECK_PACKAGES_AND_CUDA", "CHECK_SOURCE_IMPORT"):
                embedded[name] = ast.literal_eval(node.value)
    assert set(embedded) == {"CHECK_PACKAGES_AND_CUDA", "CHECK_SOURCE_IMPORT"}
    for value in embedded.values():
        ast.parse(value)
    assert "torch.cuda.is_available()" in embedded["CHECK_PACKAGES_AND_CUDA"]
    assert "torch.cuda.get_device_properties(0)" in embedded["CHECK_PACKAGES_AND_CUDA"]
    assert "importlib.import_module(module)" in embedded["CHECK_SOURCE_IMPORT"]


def test_forbidden_kernel_installation_and_model_execution_absent():
    src = PROBE.read_text(encoding="utf-8").lower()
    for forbidden in (
        "torch.randn", "torch.empty", "torch.ones", "torch.zeros",
        "torch.cuda.synchronize", "cuda.launch", "triton.jit",
        "mamba2(", "tttforcausallm(", "torch.compile(",
        "pip install", "apt install", "sudo ", "git fetch", "git clone",
        "git push", "set_per_process_memory_fraction", "run_gate3",
        "os.system(", "shell=true", "subprocess.popen(",
        "wsl --install", "workflow_dispatch", "invoke-webrequest",
    ):
        assert forbidden not in src
