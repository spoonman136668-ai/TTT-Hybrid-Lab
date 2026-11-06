"""G7-R10 consumed one-time read-only diagnostic: retirement/evidence contract."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_consumed_one_shot_original_error_recovery_retired():
    # The authorized G7-R10 runner job completed and must not be retriggerable.
    assert not (ROOT / ".github/workflows/g7-r10-single-readonly-receipt.yml").exists()


def test_original_source_custody_and_specific_failure_evidence_is_durable():
    result = (ROOT / "docs/G7_R10_RESULT.md").read_text(encoding="utf-8")
    for marker in (
        "42302765535",
        "13544313274",
        "da7cadb6cd6734619494d7412039a62a37a6d4da",
        "8480387bc23e5ce668800e0481dbaf37957a3582b080316beb85ce54e2f6dcb6",
        "de8cf3cbb3ae7d5a695a36b1d71c4ac1f9fcf34e9ab3aa9741fbccc2b32c5ef6",
        "b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65",
        "c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87",
        "PASS_ORIGINAL_ERROR_CUSTODY_ONLY",
        "TRITON_COMPILER_EXCEPTION",
        "driver.py",
        "STALE_RUN_ID_METADATA",
        "NO GPU RETRY",
    ):
        assert marker in result, marker
