"""G7-R10 fail-closed static contract for an unexecuted read-only receipt recovery."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_exactly_scoped_one_shot_readonly_on_existing_ttt_runner():
    workflow = (ROOT / ".github/workflows/g7-r10-single-readonly-receipt.yml").read_text()
    assert 'name: G7-R10 one-shot read-only G3 error custody' in workflow
    assert 'on:\n  push:\n    branches: [main]' in workflow
    assert 'paths:\n      - ".github/workflows/g7-r10-single-readonly-receipt.yml"' in workflow
    for marker in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        "github.actor == 'spoonman136668-ai'",
        "github.run_number == 1 && github.run_attempt == 1",
        "TTT-HYBRID-GPU-WIN", "ttt-hybrid-gpu-isolated",
        "timeout-minutes: 4", "if: always()", "retention-days: 30",
    ):
        assert marker in workflow
    for bad in (
        "actions/checkout@", "workflow_dispatch:", "pull_request:",
        "schedule:", "GITHUB_REF_PROTECTED", "wsl.exe", "python.exe",
        "pip install", "Start-Process", "git clone",
    ):
        assert bad not in workflow


def test_original_receipt_identity_cannot_be_bypassed():
    workflow = (ROOT / ".github/workflows/g7-r10-single-readonly-receipt.yml").read_text()
    for required in (
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65",
        "c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87",
        "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768",
        "g7_r6_native_internal.json", "g7_native_gate3_subreceipt.json",
        "Get-FileHash", "BLOCKED_NATIVE_DIGEST_MISMATCH",
        "BLOCKED_G3_DIGEST_MISMATCH", "BLOCKED_ORIGINAL_CHECKOUT_REPLACED",
        "PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED",
        "TRITON_OR_KERNEL_COMPATIBILITY",
        "PASS_ORIGINAL_ERROR_CUSTODY_ONLY",
    ):
        assert required in workflow


def test_whitelisted_aggregate_diagnostic_only():
    workflow = (ROOT / ".github/workflows/g7-r10-single-readonly-receipt.yml").read_text()
    assert "raw_error_message = 'NOT_EXPORTED'" in workflow
    assert "raw_traceback = 'NOT_EXPORTED'" in workflow
    assert "PTX_TOOLCHAIN_OR_COMPILATION" in workflow
    assert "TRITON_COMPILER_EXCEPTION" in workflow
    assert "CUDA_COMPUTE_CAPABILITY_OR_KERNEL_UNSUPPORTED" in workflow
    assert "UNCLASSIFIED_RUNTIME_EXCEPTION" in workflow
    assert "g7_r10_readonly_diagnostic.json" in workflow
    assert "runner.temp" in workflow
    assert "Get-Content -LiteralPath $g3Path -Raw -Encoding UTF8" in workflow
    assert "ConvertTo-Json -Depth 5" in workflow
