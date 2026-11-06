"""Static G7-R4 scope and syntax tests. Never execute WSL or CUDA in CI."""
from __future__ import annotations
import ast
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-r4-runtime-admission.yml"
PROBE = ROOT / "scripts/g7_r4_runtime_probe.py"


def workflow():
    return WORKFLOW.read_text(encoding="utf-8")


def powershell_body():
    match = re.search(r"(?ms)^        run: \|\n(.*?)^      - name: Preserve bounded", workflow())
    assert match is not None
    return textwrap.dedent(match.group(1))


def test_one_shot_path_trigger_owner_and_runner_guard():
    s = workflow()
    assert "on:\n  push:\n    branches: [main]" in s
    assert 'paths:\n      - ".github/workflows/g7-r4-runtime-admission.yml"' in s
    for forbidden in ("  workflow_dispatch:", "  schedule:", "  pull_request:",
                      "  repository_dispatch:", "  workflow_call:"):
        assert forbidden not in s
    for required in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        "github.actor == 'spoonman136668-ai'",
        "github.event_name == 'push'",
        "github.run_number == 1", "github.run_attempt == 1",
        "permissions:\n  contents: read", "cancel-in-progress: false",
        "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]",
        "timeout-minutes: 12", "persist-credentials: false",
        "ref: ${{ github.sha }}", "if: always()",
    ):
        assert required in s
    ps = powershell_body()
    for required in ("TTT-HYBRID-GPU-WIN", "G7_R4_CHECKOUT_SHA_DRIFT",
                     "G7_R4_DIRTY_SOURCE", "wsl.exe --distribution Ubuntu-24.04",
                     "g7_r4_runtime_probe.py", "g7_r4_runtime.json"):
        assert required in ps
    assert "forge-codex-isolated" not in s.lower()


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


def test_windows_powershell_syntax_only():
    if platform.system() != "Windows":
        return
    with tempfile.TemporaryDirectory(prefix="g7-r4-ps-parse-") as tmp:
        target = Path(tmp) / "g7-r4.ps1"
        target.write_text(powershell_body(), encoding="utf-8")
        cmd = (
            "$t=$null;$e=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "'" + str(target).replace("'", "''") + "',[ref]$t,[ref]$e) | Out-Null;"
            "if($e.Count -gt 0){$e | ForEach-Object{Write-Error $_};exit 1}"
        )
        result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
                                capture_output=True, text=True, timeout=25)
        assert result.returncode == 0, result.stderr[-1300:]
