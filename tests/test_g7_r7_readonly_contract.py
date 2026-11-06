"""Static no-execution contract for the one-time read-only G7-R7 receipt recovery."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_one_shot_scoped_owner_main_and_isolated_runner():
    data = (ROOT / ".github/workflows/g7-r7-single-readonly-receipt.yml").read_text()
    assert "on:\n  push:\n    branches: [main]" in data
    assert "github.run_number == 1 && github.run_attempt == 1" in data
    assert "github.actor == 'spoonman136668-ai'" in data
    assert "github.ref == 'refs/heads/main'" in data
    assert "ttt-hybrid-gpu-isolated" in data
    assert "TTT-HYBRID-GPU-WIN" in data
    assert "timeout-minutes: 4" in data
    assert "workflow_dispatch:" not in data
    assert "pull_request:" not in data
    assert "schedule:" not in data
    assert "GITHUB_REF_PROTECTED" not in data


def test_never_checkout_or_touch_old_model_runtime():
    data = (ROOT / ".github/workflows/g7-r7-single-readonly-receipt.yml").read_text()
    assert "actions/checkout@" not in data
    assert "wsl.exe" not in data
    assert "python.exe" not in data
    assert "pip install" not in data
    assert "git clone" not in data
    assert "CUDA_VISIBLE_DEVICES" not in data
    assert "Start-Process" not in data
    assert "g7_r6_native_internal.json" in data
    assert "g7_native_gate3_subreceipt.json" in data
    assert "Get-FileHash" in data
    assert "original_source_bound" in data
    assert "raw_error_message = 'NOT_EXPORTED'" in data
    assert "raw_traceback = 'NOT_EXPORTED'" in data
    assert "runner.temp" in data
    assert "if: always()" in data


def test_pinned_original_custody_and_no_trusting_missing_g3():
    data = (ROOT / ".github/workflows/g7-r7-single-readonly-receipt.yml").read_text()
    assert "d028badf39f7e3392adc24329be4152cdcdc00cd" in data
    assert "4d6cd2d716a4ea417d2e5a424e6878c9bad4ab93a0e7e5351ceb0c4158700f29" in data
    assert "44f3e76d2d06ee879fa3222dd3921c7a1f8e08be579b96e03b7545e388a7b768" in data
    assert "BLOCKED_NATIVE_DIGEST_MISMATCH" in data
    assert "BLOCKED_G3_DIGEST_MISMATCH" in data
    assert "PASS_ORIGINAL_ERROR_CUSTODY_ONLY" in data
