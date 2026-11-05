"""G7 uses a called workflow on the existing Mind-Palace runner, not a new registration.

This tests static scope and Windows PowerShell syntax only. It NEVER
dispatches the workflow or touches the owner's host/GPU.
"""
from pathlib import Path
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-mind-palace-reusable-inventory.yml"


def source() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_reusable_only_no_native_or_scheduled_dispatch():
    s = source()
    assert re.search(r"(?m)^on:\s*\n\s+workflow_call:\s*$", s)
    for forbidden in ("  workflow_dispatch:", "  pull_request:", "  push:",
                      "  schedule:", "  repository_dispatch:", "  issue_comment:"):
        assert forbidden not in s
    assert "permissions: {}" in s


def test_caller_and_owner_scope():
    s = source()
    for entry in (
        "github.repository == 'spoonman136668-ai/Mind-Palace'",
        "github.ref == 'refs/heads/main'",
        "github.event_name == 'workflow_dispatch'",
        "github.actor == 'spoonman136668-ai'",
        "github.run_attempt == 1",
        "$env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/Mind-Palace'",
        "$env:RUNNER_NAME -cne 'MIND-PALACE-WIN'",
        "$env:RUNNER_OS -cne 'Windows'",
        "$env:G7_EXPECTED_TTT_SHA -cnotmatch '^[0-9a-f]{40}$'",
    ):
        assert entry in s
    assert "runs-on: [self-hosted, Windows, X64]" in s
    assert "timeout-minutes: 6" in s
    assert "FORGE-CODEX-REMOTE" not in s
    assert "forge-codex-isolated" not in s
    assert "ttt-hybrid-gpu-isolated" not in s


def test_no_source_checkout_wsl_activation_or_cuda():
    s = source()
    # Only comments may contain these diagnostic names.
    noncomment = "\n".join(line.strip() for line in s.splitlines()
                           if line.strip() and not line.lstrip().startswith("#"))
    assert not re.search(
        r"(?i)(uses:\s*actions/checkout|\b(wsl(?:\.exe)?|pip|winget|choco)\s+(?:--install|-d|install)"
        r"|Start-Process|Start-Job|Invoke-Expression|Invoke-WebRequest"
        r"|g7_cuda_qualification\.py|run_gate3\.py|git\s+push)",
        noncomment
    )
    assert "PASS_EXISTING_RUNNER_READONLY_ONLY" in s
    assert "native_mamba_ttt_cuda = 'NOT_RUN'" in s
    assert "scientific" not in s or "science = 'UNVERIFIED'" in s
    assert "nvidia-smi.exe" in s
    assert "HKCU:" in s
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in s


def test_readonly_host_requires_real_gpu_capacity():
    s = source()
    for entry in (
        "memory_total_mib -lt 4096",
        "memory_free_mib -lt 4096",
        "memory_free_mib -gt $report.gpu.memory_total_mib",
        "G7_GPU_QUERY_NOT_SINGLE_NUMERIC_DEVICE",
        "G7_INSUFFICIENT_WINDOWS_GPU_MEMORY",
        "G7_CALLER_RUNNER_IDENTITY_MISMATCH",
        "G7_UNAUTHORIZED_CALLER_EVENT_OR_ACTOR",
    ):
        assert entry in s


def test_windows_powershell_block_parses():
    if platform.system() != "Windows":
        return
    src = source()
    marker = "        run: |\n"
    assert marker in src
    code = src.split(marker, 1)[1].split("      - name: Retain scoped", 1)[0]
    ps = "\n".join(line[10:] if line.startswith("          ") else line
                   for line in code.splitlines())
    expression = (
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseInput("
        "$env:G7_REUSE_PS,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count -gt 0){$e|ForEach-Object{Write-Error $_};exit 1}"
    )
    import os
    env = os.environ.copy()
    env["G7_REUSE_PS"] = ps
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", expression],
        env=env, cwd=ROOT, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr[-1500:]
