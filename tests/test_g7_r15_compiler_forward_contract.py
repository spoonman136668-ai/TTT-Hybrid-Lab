"""G7-R15 single-use compiler repair and original native forward: CPU-only negative controls."""
from pathlib import Path
import ast
import importlib.util
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/g7-r15-one-shot-compiler-forward.yml"
SRC = ROOT / "scripts/g7_r15_compiler_repair_forward.py"

def test_consumed_single_use_workflow_removed_and_result_preserved():
    assert not WF.exists()
    report=(ROOT/"docs/G7_R15_RESULT.md").read_text(encoding="utf-8")
    for marker in (
        "42321056057",
        "13553665337",
        "e0edb0ed2e9d3c187ea1aef205cab8ea8275165a",
        "2e9e3f6f6b56a7e32401c017d396d7898c4b997933939227c3477684e59ef329",
        "ef2a5035b5df9f9875c328b618122646f642115d90888e320f71b086ce2f11c4",
        "BLOCKED_GCC_SUDO_NONINTERACTIVE_DENIED",
        "source_custody=PASS",
        "provisioning=NOT_ATTEMPTED",
        "native_forward=NOT_RUN",
        "NO RETRY",
    ):
        assert marker in report, marker


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
