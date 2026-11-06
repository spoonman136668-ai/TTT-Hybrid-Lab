"""Retained static custody/safety tests for G7-R3 provisioning.

The one-time WSL execution workflow was retired after run 42208814262.
Never re-run provisioning from this qualification suite.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROVISION = ROOT / "scripts/g7_r3_provision.py"
RETIRED_WORKFLOW = ROOT / ".github/workflows/g7-r3-isolated-env.yml"


def test_one_time_execution_entrypoint_is_retired():
    assert not RETIRED_WORKFLOW.exists()


def test_provisioning_script_is_stable_and_syntax_valid():
    script = PROVISION.read_text(encoding="utf-8")
    ast.parse(script, filename=str(PROVISION))
    for required in (
        'UV_VERSION = "0.9.9"',
        "UV_SHA256 =",
        'TORCH_VERSION = "2.9.0"',
        'TRITON_VERSION = "3.5.0"',
        "https://download.pytorch.org/whl/cu128",
        'if os.geteuid() == 0:',
        'if free_bytes < MIN_FREE_GIB * 1024**3:',
        'if root.exists():',
        '"--no-project"',
        '"source_pins_sha256"',
        '"package_manifest_sha256"',
        '"cuda_kernels": "NOT_RUN"',
        '"cuda_runtime": "NOT_PROBED"',
        '"native_extensions": "NOT_BUILT"',
        '"system_packages": "NOT_INSTALLED"',
        '"training": "NOT_RUN"',
        '"scientific": "UNVERIFIED"',
    ):
        assert required in script


def test_no_cuda_execution_system_install_or_controller_changes():
    content = PROVISION.read_text(encoding="utf-8").lower()
    for disallowed in (
        "import torch", "import triton", "torch.cuda", "cuda.launch",
        "sudo ", "apt install", "pip install --system",
        "nvidia-smi", "run_gate3", "git push",
        "os.system(", "shell=true", "subprocess.popen(",
        "wsl --install", "workflow_dispatch",
    ):
        assert disallowed not in content
