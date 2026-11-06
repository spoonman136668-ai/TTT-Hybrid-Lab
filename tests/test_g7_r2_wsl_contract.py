"""Static G7-R2 one-time workflow safety checks. Never execute WSL in CI."""
import ast
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-r2-one-time-wsl-metadata.yml"


def source():
    return WORKFLOW.read_text(encoding="utf-8")


def powershell_body():
    match = re.search(
        r"(?ms)^        run: \|\n(.*?)^      - name: Store immutable",
        source(),
    )
    assert match is not None
    return textwrap.dedent(match.group(1))


def test_once_and_only_owner_main():
    wf = source()
    assert re.search(r"(?m)^on:\s*\n  workflow_dispatch:\s*$", wf)
    for forbidden in ("  push:", "  pull_request:", "  schedule:",
                      "  repository_dispatch:", "  issue_comment:"):
        assert forbidden not in wf
    for required in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        "github.actor == 'spoonman136668-ai'",
        "github.event_name == 'workflow_dispatch'",
        "github.run_number == 1",
        "github.run_attempt == 1",
        "permissions:\n  contents: read",
        "cancel-in-progress: false",
        "timeout-minutes: 8",
        "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]",
        "persist-credentials: false",
        "ref: ${{ github.sha }}",
    ):
        assert required in wf


def test_no_install_cuda_training_or_other_runners():
    wf = source()
    ps = powershell_body()
    for marker in ("$env:RUNNER_NAME -cne 'TTT-HYBRID-GPU-WIN'",
                   "$env:GITHUB_RUN_NUMBER -cne '1'",
                   "G7_R2_SHA_DRIFT", "G7_R2_DIRTY_CHECKOUT",
                   "wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/python3 -I -c $probe",
                   "g7_r2_wsl.json"):
        assert marker in ps
    assert "ttt-g7-r2-wsl-" in wf
    for forbidden in ("pip install", "apt install", "sudo ", "git push",
                      "torch.cuda", "import torch", "import triton",
                      "start-process", "invoke-expression",
                      "invoke-webrequest", "run_gate3", "wsl --install"):
        assert forbidden not in ps.lower()


def test_embedded_python_syntax_no_execution():
    script = powershell_body()
    match = re.search(r"(?ms)^\$probe = @'\n(.*?)^'@\s*$", script)
    assert match is not None
    ast.parse(match.group(1), filename="<g7-r2-metadata>")
    assert "metadata.distributions(" in match.group(1)
    for required in ("'cuda_kernels':'NOT_RUN'", "'third_party_imports':'NOT_RUN'",
                     "'installation':'NOT_RUN'", "'science':'UNVERIFIED'"):
        assert required in match.group(1)


def test_windows_powershell_parser_only():
    if platform.system() != "Windows":
        return
    with tempfile.TemporaryDirectory(prefix="g7-r2-syntax-") as tmp:
        target = Path(tmp) / "probe.ps1"
        target.write_text(powershell_body(), encoding="utf-8")
        cmd = (
            "$t=$null;$e=$null;"
            "[System.Management.Automation.Language.Parser]::ParseFile("
            "'" + str(target).replace("'", "''") + "',[ref]$t,[ref]$e) | Out-Null;"
            "if($e.Count -gt 0){$e | ForEach-Object {Write-Error $_};exit 1}"
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", cmd],
            cwd=ROOT, capture_output=True, text=True, timeout=25,
        )
        assert result.returncode == 0, result.stderr[-1600:]
