"""G7-R5 one-time CUDA qualification admission contracts.

No test invokes WSL, imports Torch/Triton, or executes a CUDA kernel.
"""
from __future__ import annotations

import ast
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-r5-single-cuda-kernel.yml"
WRAPPER = ROOT / "scripts/g7_r5_oneshot.py"
KERNEL = ROOT / "scripts/g7_cuda_qualification.py"


def powershell_body():
    match = re.search(
        r"(?ms)^        run: \|\n(.*?)^      - name: Preserve immutable",
        WORKFLOW.read_text(encoding="utf-8"),
    )
    assert match is not None
    return textwrap.dedent(match.group(1))


def test_exact_first_push_only_and_authorized_dedicated_runner():
    wf = WORKFLOW.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*\n  push:\s*\n    branches: \[main\]\s*$", wf)
    assert 'paths:\n      - ".github/workflows/g7-r5-single-cuda-kernel.yml"' in wf
    for forbidden in ("  workflow_dispatch:", "  pull_request:", "  schedule:",
                      "  repository_dispatch:", "  issue_comment:"):
        assert forbidden not in wf
    for required in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'", "github.event_name == 'push'",
        "github.actor == 'spoonman136668-ai'",
        "github.run_number == 1", "github.run_attempt == 1",
        "permissions:\n  contents: read",
        "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]",
        "timeout-minutes: 7", "persist-credentials: false",
        "ref: ${{ github.sha }}", "cancel-in-progress: false",
        "if: always()", "retention-days: 30",
    ):
        assert required in wf
    ps = powershell_body()
    for required in (
        "TTT-HYBRID-GPU-WIN", "G7_R5_SHA_DRIFT", "G7_R5_DIRTY_CHECKOUT",
        "Ubuntu-24.04", "/usr/bin/timeout", "--kill-after=10s",
        "240s", "g7_r5_oneshot.py", "g7_r5_kernel.json",
    ):
        assert required in ps


def test_exact_r3_baseline_package_manifest_and_source_custody():
    wrapper = WRAPPER.read_text(encoding="utf-8")
    ast.parse(wrapper, filename=str(WRAPPER))
    expected = (
        'R3_SOURCE_SHA = "06a01c2c3a980eeb806a7ecb7cca1979a431f80c"',
        'R3_PINS_SHA256 = "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768"',
        'R3_MANIFEST_SHA256 = "b0f674fd52f9f514d408840d27b969ce2da19ea119d90a6ce8f59fde05c073df"',
        'TORCH = "2.9.0+cu128"', 'TRITON = "3.5.0"',
        "guard(package_sha256 == R3_MANIFEST_SHA256",
        'guard(packages.get("torch") == TORCH and packages.get("triton") == TRITON',
        'git(root, "status", "--porcelain", "--untracked-files=all")',
        'git(root, "hash-object", "--", rel)',
        'kernel_only=True', '"--child"',
        'MAX_PEAK_GIB = 1.0', 'MIN_FREE_GIB = 4.0',
        'CHILD_TIMEOUT_SECONDS = 175',
        '"PASS_ONE_BOUNDED_NATIVE_CUDA_MATMUL_ONLY"',
        '"scientific": "UNVERIFIED"',
        '"training": "NOT_RUN"',
        '"native_mamba_forward": "NOT_RUN"',
        '"native_ttt_forward": "NOT_RUN"',
        '"system_installation": "NOT_RUN"',
        '"native_extension_build": "NOT_RUN"',
    )
    for marker in expected:
        assert marker in wrapper


def test_only_matmul_path_and_no_automatic_integrated_forward():
    source = KERNEL.read_text(encoding="utf-8")
    ast.parse(source, filename=str(KERNEL))
    assert 'if kernel_only:' in source
    assert source.index('if kernel_only:') < source.index(
        'from run_gate3 import run as native_gate3'
    )
    assert '"PASS_NATIVE_CUDA_KERNEL_ONLY"' in source
    assert '"BLOCKED_R5_RUNTIME_DEVICE_IDENTITY"' in source
    assert '"NOT_PROBED_KERNEL_ONLY"' in source
    assert "torch.cuda.set_per_process_memory_fraction" in source
    assert 'torch.randn((16, 16), device="cuda", dtype=torch.float32)' in source
    assert 'if peak > max_peak_gib * (1024**3):' in source
    wrapper = WRAPPER.read_text(encoding="utf-8").lower()
    for disallowed in (
        "pip install", "apt install", "sudo ", "git fetch", "git clone",
        "git push", "torch.compile(", "torch.cuda.synchronize(",
        "subprocess.popen(", "shell=true", "os.system(",
        "workflow_dispatch", "wsl --install",
    ):
        assert disallowed not in wrapper


def test_powershell_parser_only_on_windows():
    if platform.system() != "Windows":
        return
    with tempfile.TemporaryDirectory(prefix="g7-r5-static-") as tmp:
        target = Path(tmp) / "g7-r5.ps1"
        target.write_text(powershell_body(), encoding="utf-8")
        command = (
            "$t=$null;$e=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "'" + str(target).replace("'", "''") + "',[ref]$t,[ref]$e) | Out-Null;"
            "if($e.Count -gt 0){$e | ForEach-Object{Write-Error $_};exit 1}"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True, text=True, timeout=25,
        )
        assert result.returncode == 0, result.stderr[-1500:]
