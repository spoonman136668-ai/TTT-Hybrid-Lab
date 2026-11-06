"""G7-R11 pure static constraints for single-use original-message discriminator."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/g7-r11-single-readonly-receipt.yml"


def test_bounded_owner_only_one_shot_on_existing_ttt_runner():
    s = WORKFLOW.read_text(encoding="utf-8")
    for marker in (
        "on:\n  push:\n    branches: [main]",
        'paths:\n      - ".github/workflows/g7-r11-single-readonly-receipt.yml"',
        "github.actor == 'spoonman136668-ai'",
        "github.run_number == 1 && github.run_attempt == 1",
        "github.ref == 'refs/heads/main'",
        "TTT-HYBRID-GPU-WIN", "ttt-hybrid-gpu-isolated",
        "timeout-minutes: 4", "if: always()",
    ):
        assert marker in s
    for banned in (
        "actions/checkout@", "wsl.exe", "python.exe", "pip install",
        "git clone", "Start-Process", "workflow_dispatch:", "pull_request:",
        "schedule:", "GITHUB_REF_PROTECTED",
    ):
        assert banned not in s


def test_exact_original_g7_r9_sha_hash_custody_not_historical_r6():
    s = WORKFLOW.read_text(encoding="utf-8")
    for marker in (
        "original_run_id = 42300670454",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65",
        "c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87",
        "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768",
        "g7_r6_native_internal.json", "g7_native_gate3_subreceipt.json",
        "BLOCKED_NATIVE_DIGEST_MISMATCH", "BLOCKED_G3_DIGEST_MISMATCH",
        "BLOCKED_ORIGINAL_CHECKOUT_REPLACED",
        "PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED",
        "PASS_ORIGINAL_ERROR_CUSTODY_ONLY",
    ):
        assert marker in s
    assert "original_run_id = 42281970624" not in s


def test_message_only_classification_and_full_module_phase():
    s = WORKFLOW.read_text(encoding="utf-8")
    for marker in (
        "$message = [string]$g3.error_message",
        "$traceLines = @($g3.traceback_tail)",
        "$message -match",
        "failed to find C compiler", "HOST_C_COMPILER_MISSING",
        "TRITON_ACTIVE_DRIVER_COUNT", "TRITON_HELPER_MODULE_LOAD",
        "LIBCUDA_DISCOVERY_OR_LOAD", "PTXAS_OR_PTX_COMPILATION",
        "TRITON_COMPILATION_OTHER", "UNCLASSIFIED_OR_INSUFFICIENT_DATA",
        "triton/runtime/build", "triton/runtime/driver",
        "nvidia/backend", "triton/compiler/compiler",
        "precise_root_cause = 'NOT_YET_CONFIRMED'",
        "G7_R11_ROOT_CAUSE=", "NOT_EXPORTED_AMBIGUOUS_BASENAME",
    ):
        assert marker in s, marker
    assert "$detail -match" not in s
    assert "raw_error_message = 'NOT_EXPORTED'" in s
    assert "raw_traceback = 'NOT_EXPORTED'" in s
    assert 'Write-Output $message' not in s
    assert 'Write-Output $trace' not in s
