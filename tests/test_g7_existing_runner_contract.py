"""The pre-CUDA runner workflow cannot commandeer FORGE or launch WSL.

These tests deliberately inspect the deployment wiring, not GPU execution.
"""
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/g7-existing-runner-admission.yml"
PS = ROOT / "scripts/g7_existing_runner_inventory.ps1"


def test_only_owner_initiated_trusted_main_can_dispatch():
    wf = WF.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*\n\s+workflow_dispatch:\s*$", wf)
    for forbidden in ("  pull_request:", "  push:", "  schedule:", "  issue_comment:",
                      "  repository_dispatch:"):
        assert forbidden not in wf
    assert "github.ref == 'refs/heads/main'" in wf
    assert "github.actor == 'spoonman136668-ai'" in wf
    assert "github.ref_protected == true" in wf
    assert "github.run_attempt == 1" in wf
    assert "permissions:\n  contents: read" in wf
    assert "timeout-minutes: 8" in wf


def test_ttt_only_runner_label_not_forge():
    wf = WF.read_text(encoding="utf-8")
    route = next(x.strip() for x in wf.splitlines() if x.strip().startswith("runs-on:"))
    assert route == "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]"
    assert "forge-codex-isolated" not in route
    assert "ckb-plane" not in route.lower()


def test_runner_entrypoint_cannot_dispatch_cuda():
    wf = WF.read_text(encoding="utf-8")
    assert "scripts\\g7_existing_runner_inventory.ps1" in wf
    assert "ref: ${{ github.sha }}" in wf
    assert "persist-credentials: false" in wf
    assert "shell: powershell" in wf
    assert not re.search(r"(?m)^\s*run:.*(g7_cuda_qualification|run_gate3|wsl\s+-d)", wf)
    assert "local_runs/g7_runner_inventory.json" in wf


def test_powershell_probe_checks_repo_sha_and_workspace():
    s = PS.read_text(encoding="utf-8")
    for required in (
        "UNAUTHORIZED_REPO_REF_ACTOR_OR_RUNNER",
        "$env:GITHUB_REF_PROTECTED -cne 'true'",
        "INVALID_OR_MISMATCHED_GITHUB_SHA",
        "CHECKOUT_SHA_DRIFT",
        "UNAUTHORIZED_WORKTREE_PATH",
        "DIRTY_SOURCE_TREE",
        "HOST_READONLY_INVENTORY",
        "PASS_RUNNER_INVENTORY_ONLY",
        "authentic_cuda = 'NOT_RUN'",
        "scientific = 'UNVERIFIED'",
        "no_install_or_wsl_launch = $true",
    ):
        assert required in s


def test_powershell_probe_does_not_invoke_wsl_install_or_gpu_kernel():
    s = PS.read_text(encoding="utf-8")
    commands = "\n".join(line for line in s.splitlines()
                          if line.strip() and not line.lstrip().startswith("#"))
    assert "wsl.exe" not in commands
    assert not re.search(r"(?i)(Start-Process|Start-Job|Invoke-Expression|Invoke-WebRequest|git\s+push|pip\s+install|wsl\s+--install)", commands)
    assert "nvidia-smi.exe" in commands
    assert "HKEY_CURRENT_USER\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss" in commands


def test_native_powershell_parser_windows():
    if platform.system() != "Windows":
        return
    expression = (
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "'scripts/g7_existing_runner_inventory.ps1',[ref]$t,[ref]$e) | Out-Null;"
        "if ($e.Count -gt 0) {$e | ForEach-Object { Write-Error $_ }; exit 1}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", expression],
        cwd=ROOT, capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stderr[-1500:]
