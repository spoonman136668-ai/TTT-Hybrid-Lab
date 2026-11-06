"""G7-R12: pure static scope tests; tests cannot run the WSL2 inventory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/g7-r12-one-shot-compiler-inventory.yml"


def test_owner_single_use_specific_isolated_runner_and_budget():
    s = WF.read_text(encoding="utf-8")
    for marker in (
        "on:\n  push:\n    branches: [main]",
        'paths:\n      - ".github/workflows/g7-r12-one-shot-compiler-inventory.yml"',
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.actor == 'spoonman136668-ai'",
        "github.ref == 'refs/heads/main'",
        "github.run_number == 1 && github.run_attempt == 1",
        "TTT-HYBRID-GPU-WIN", "ttt-hybrid-gpu-isolated",
        "timeout-minutes: 4", "if: always()", "retention-days: 30",
    ):
        assert marker in s, marker
    for banned in (
        "actions/checkout@", "workflow_dispatch:", "schedule:", "pull_request:",
        "git clone", "Start-Process", "apt install", "apt-get",
        "pip install", "torch.cuda", "nvidia-smi", "run_gate3",
    ):
        assert banned not in s


def test_original_failure_sha_identity_and_bounded_inventory_only():
    s = WF.read_text(encoding="utf-8")
    for marker in (
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "42300670454", "42313100239",
        "HOST_C_COMPILER_UNAVAILABLE_CONFIRMED",
        "BLOCKED_ORIGINAL_SOURCE_IDENTITY_DRIFT",
        "wsl.exe --distribution Ubuntu-24.04 --exec /usr/bin/timeout",
        "35s /usr/bin/bash --noprofile --norc -c $probe",
        "command -v", "/usr/bin/sha256sum",
        "PRESENT_UNQUALIFIED", "NOT_FOUND",
        "BLOCKED_NO_INSTALLED_COMPILER_FOUND",
        "cc-environment", "g7_r12_readonly_compiler_inventory.json",
        "cuda = 'NOT_STARTED'", "model = 'NOT_RUN'",
        "training = 'NOT_RUN'", "installation = 'NOT_AUTHORIZED'",
        "deployment = 'NOT_AUTHORIZED'",
    ):
        assert marker in s, marker
    assert "gcc --version" not in s
    assert "clang --version" not in s
    assert "wsl_started_for_inventory_only" in s
    assert "$env:RUNNER_TEMP" in s
