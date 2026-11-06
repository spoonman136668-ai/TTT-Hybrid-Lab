"""G7-R11 original exact-message diagnostic: immutable result and one-shot retirement."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_consumed_one_shot_retired_and_source_only():
    assert not (ROOT / ".github/workflows/g7-r11-single-readonly-receipt.yml").exists()


def test_original_execution_identity_and_result_receipts_durable():
    result=(ROOT / "docs/G7_R11_RESULT.md").read_text(encoding="utf-8")
    for exact in (
        "42313100239", "13549970962",
        "028cd7057e00b7c968ee0f53a8d9074f41771dec",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "ca384d96fd35e7b438c6af86434f641dea65e978fc0f05d93b21cfe03bbf9e20",
        "45a0e3090f936c31af9daefca4d37a8ba4f31c2a8c3cd837a8655fc2b6c1521b",
        "b3c41101ee550603cc73c8e95373081fbd4d59216ac6587f1768d96875cd9d65",
        "c2f81f51ef75c6544ef50d4a56e9e9ee536b1c71bbc66b10015a60ceeb711d87",
        "HOST_C_COMPILER_MISSING",
        "TRITON_HOST_C_HELPER_BUILD",
        "HOST_C_COMPILER_UNAVAILABLE_CONFIRMED",
        "NO GPU RETRY",
    ):
        assert exact in result, exact


def test_scientific_and_install_boundaries_retained():
    result=(ROOT / "docs/G7_R11_RESULT.md").read_text(encoding="utf-8")
    assert "SCIENTIFIC: UNVERIFIED" in result
    assert "DEPLOYMENT: NOT AUTHORIZED" in result
    assert "No installation was performed" in result
    assert "original_run_id=42300670454" in result
    assert "raw_traceback=NOT_EXPORTED" in result
