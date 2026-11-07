"""G7-R15 single-use compiler repair and original native forward: CPU-only negative controls."""
from pathlib import Path
import ast
import importlib.util
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/g7-r15-one-shot-compiler-forward.yml"
SRC = ROOT / "scripts/g7_r15_compiler_repair_forward.py"

def test_single_use_owner_runner_exact_head_forward_and_no_separate_controller():
    s=WF.read_text(encoding="utf-8")
    for key in (
        "on:\n  push:\n    branches: [main]",
        'paths:\n      - ".github/workflows/g7-r15-one-shot-compiler-forward.yml"',
        "github.run_number == 1 && github.run_attempt == 1",
        "github.actor == 'spoonman136668-ai'",
        "TTT-HYBRID-GPU-WIN", "ttt-hybrid-gpu-isolated",
        "github.sha", "persist-credentials: false", "clean: true",
        "G7_R15_DIRTY_CHECKOUT", "timeout-minutes: 14",
        "650s /usr/bin/python3 -I -B",
        "g7_r15_compiler_repair_forward.py", "retention-days: 30",
    ):
        assert key in s, key
    for forbidden in (
        "workflow_dispatch:", "schedule:", "pull_request:", "git push",
        "wsl.exe --install", "actions/setup-", "Start-Process",
        "torch.optim", "nvidia-smi",
    ):
        assert forbidden not in s, forbidden

def test_only_gcc_provision_and_no_pinned_runtime_reinstall():
    s=SRC.read_text(encoding="utf-8")
    for key in (
        "SOURCE_PINS_SHA = \"44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768\"",
        "R3_SHA = \"06a01c2c3a980eeb806a7ecb7cca1979a431f80c\"",
        "GCC_SUDO_NONINTERACTIVE_DENIED", "GCC_APT_UPDATE_FAILED",
        "GCC_APT_INSTALL_FAILED", "--no-install-recommends", '"gcc"',
        "HOST_C_SHARED_BUILD_FAILED",
        'env["CC"] = executable', "--require-nonfused",
        "g7_r6_oneshot.py", "SOURCE_PINS_DRIFT", "DIRTY_PROJECT_CHECKOUT",
        "ORIGINAL_BOUNDED_FORWARD_NOT_QUALIFIED",
        '"training": "NOT_RUN"', '"deployment": "NOT_AUTHORIZED"',
    ):
        assert key in s, key
    for forbidden in (
        "pip install", "uv pip install", "torch.optim", "git checkout",
        "apt-get upgrade", "apt-get dist-upgrade", "curl ", "wget ",
        "nvidia-smi", "torch.cuda.",
    ):
        assert forbidden not in s, forbidden
    ast.parse(s)

def test_no_runtime_effects_during_tests():
    tree=ast.parse(SRC.read_text(encoding="utf-8"))
    top_calls=[n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert not top_calls
    assert SRC.is_file()

def test_uses_existing_nonfused_native_acceptance():
    s=SRC.read_text(encoding="utf-8")
    original=(ROOT/"scripts/g7_r6_oneshot.py").read_text(encoding="utf-8")
    assert "CHILD_TIMEOUT_SECONDS = 175" in original
    assert "MAX_PEAK_GIB = 1.0" in original
    assert "MIN_FREE_GIB = 4.0" in original
    assert "valid_native_result(" in original
    assert "native.get(\"status\")" in s
