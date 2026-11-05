"""No-seventh-runner admission must not activate WSL or any CUDA workloads."""
from pathlib import Path
import os
import platform
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/g7-mind-palace-reusable-inventory.yml"


def content():
    return WF.read_text(encoding="utf-8")


def test_reusable_only_never_autoruns():
    s = content()
    assert re.search(r"(?m)^on:\s*\n\s+workflow_call:", s)
    for trigger in ("  pull_request:", "  push:", "  schedule:",
                    "  workflow_dispatch:", "  repository_dispatch:", "  issue_comment:"):
        assert trigger not in s
    assert "permissions:\n  contents: read" in s
    assert "runs-on: [self-hosted, Windows, X64]" in s
    assert "timeout-minutes: 5" in s


def test_mind_palace_caller_and_runner_admission():
    s = content()
    for required in (
        "github.repository == 'spoonman136668-ai/Mind-Palace'",
        "$env:GITHUB_REPOSITORY -cne 'spoonman136668-ai/Mind-Palace'",
        "$env:GITHUB_REF -cne 'refs/heads/main'",
        "$env:GITHUB_ACTOR -cne 'spoonman136668-ai'",
        "$env:GITHUB_EVENT_NAME -cne 'workflow_dispatch'",
        "$env:GITHUB_RUN_ATTEMPT -cne '1'",
        "$env:RUNNER_NAME -cne 'MIND-PALACE-WIN'",
        "$env:RUNNER_OS -cne 'Windows'",
        "$env:EXPECTED_TTT_CANDIDATE_SHA -cnotmatch '^[0-9a-f]{40}$'",
    ):
        assert required in s
    assert "forge-codex-isolated" not in s
    assert "ttt-hybrid-gpu-isolated" not in s


def test_host_only_no_install_checkout_wsl_kernel_or_source_execution():
    s = content()
    commands = "\n".join(line.strip() for line in s.splitlines()
                         if line.strip() and not line.lstrip().startswith("#"))
    assert not re.search(
        r"(?i)(uses:\s*actions/checkout|\b(?:wsl(?:\.exe)?|pip|winget|choco)\s+"
        r"(?:--install|-d|install)|Start-Process|Start-Job|Invoke-WebRequest"
        r"|Invoke-Expression|g7_cuda_qualification\.py|run_gate3\.py|git\s+push)",
        commands,
    )
    assert "Get-Command -Name nvidia-smi.exe" in s
    assert "G7_GPU_VRAM_UNAVAILABLE" in s
    assert "TTT_G7_NATIVE_CUDA=NOT_RUN" in s
    assert "TTT_G7_SCIENCE=UNVERIFIED" in s


def test_safe_single_device_vram_admission():
    s = content()
    assert "$data.Count -ne 1" in s
    assert "$parts.Count -ne 4" in s
    assert "[int]::TryParse" in s
    assert "$total -lt 4096 -or $free -lt 4096 -or $free -gt $total" in s


def test_windows_powershell_parser():
    if platform.system() != "Windows":
        return
    s = content()
    ps = "\n".join(line[10:] if line.startswith("          ") else line
                   for line in s.split("        run: |\n", 1)[1].splitlines())
    env = os.environ.copy()
    env["G7_REUSE_PS"] = ps
    code = (
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseInput("
        "$env:G7_REUSE_PS,[ref]$t,[ref]$e)|Out-Null;"
        "if($e.Count -gt 0){$e|ForEach-Object{Write-Error $_};exit 1}"
    )
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", code],
        cwd=ROOT, env=env, capture_output=True, text=True, timeout=25,
    )
    assert result.returncode == 0, result.stderr[-1500:]
