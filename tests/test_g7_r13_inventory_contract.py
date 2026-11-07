"""G7-R13 source-only contract: differentiated WSL2 transport, no model."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/".github/workflows/g7-r13-one-shot-compiler-inventory.yml"

def test_one_shot_scope():
    s=WF.read_text(encoding="utf-8")
    for x in (
        "on:\n  push:\n    branches: [main]",
        'paths:\n      - ".github/workflows/g7-r13-one-shot-compiler-inventory.yml"',
        "github.run_number == 1 && github.run_attempt == 1",
        "github.actor == 'spoonman136668-ai'",
        "TTT-HYBRID-GPU-WIN", "ttt-hybrid-gpu-isolated",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "timeout-minutes: 4", "retention-days: 30",
    ): assert x in s, x
    for x in ("actions/checkout@", "workflow_dispatch:", "schedule:",
              "pull_request:", "apt-get", "apt install", "pip install",
              "torch.cuda", "nvidia-smi", "git clone", "Start-Process"):
        assert x not in s, x

def test_transport_phases_and_fail_closed_output_schema():
    s=WF.read_text(encoding="utf-8")
    for x in (
        "Get-Command -Name wsl.exe", "G7_R13_WSL_EXEC_OK",
        "BLOCKED_WSL_EXE_ABSENT", "BLOCKED_WSL_PING_NONZERO",
        "BLOCKED_WSL_PING_MARKER", "BLOCKED_BASH_INVENTORY_NONZERO",
        "BLOCKED_BASH_OUTPUT_LINE_COUNT", "BLOCKED_UNEXPECTED_PROBE_OUTPUT",
        "BLOCKED_NATIVE_INVOCATION_EXCEPTION",
        "probe_phase = 'NOT_STARTED'", "inventory_exit_code = $null",
        "ping_exit_code = $null", "$ErrorActionPreference = 'Continue'",
        "$ErrorActionPreference = $originalPreference", "G7_R13_PHASE=",
        "G7_R13_INVENTORY_EXIT=", "G7_R13_EXCEPTION_TYPE=",
        "command -v", "/usr/bin/sha256sum",
        "PRESENT_UNQUALIFIED", "BLOCKED_NO_INSTALLED_COMPILER_FOUND",
        "cuda = 'NOT_STARTED'", "model = 'NOT_RUN'",
        "installation = 'NOT_AUTHORIZED'", "deployment = 'NOT_AUTHORIZED'",
    ): assert x in s, x
    for x in ("BLOCKED_PROBE_UNVERIFIED", "gcc --version", "clang --version",
              "Write-Output $lines", "Write-Output $probe"):
        assert x not in s, x
