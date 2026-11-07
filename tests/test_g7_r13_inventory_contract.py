"""G7-R13: consumed one-shot retirement and immutable blocked inventory evidence."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_single_use_wsl_diagnostic_is_retired():
    assert not (ROOT / ".github/workflows/g7-r13-one-shot-compiler-inventory.yml").exists()


def test_source_and_failed_transport_receipt_recorded():
    s = (ROOT / "docs/G7_R13_RESULT.md").read_text(encoding="utf-8")
    for x in (
        "42316405581",
        "13551375790",
        "030e381ad4894d5e1c13a1af28f7df369ce363e9",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "9f4060bd68f92d806263aa98f3e2644fa232fa69af47739444c474dc1401e1da",
        "96fc4d79fb153a2cdfddf00195c877c68ff9beecce0b6bbd3020789664f69768",
        "source_bound=true",
        "BLOCKED_NATIVE_INVOCATION_EXCEPTION",
        "probe_phase=WSL_PING",
        "transport_exception_type=CommandNotFoundException",
        "gcc=UNVERIFIED",
        "clang=UNVERIFIED",
        "NO RETRY",
    ):
        assert x in s, x


def test_no_unmeasured_compiler_and_science_claims():
    s = (ROOT / "docs/G7_R13_RESULT.md").read_text(encoding="utf-8")
    assert "current compiler availability remains UNVERIFIED" in s
    assert "SCIENTIFIC: UNVERIFIED" in s
    assert "DEPLOYMENT: NOT AUTHORIZED" in s
    assert "No compiler installation, CUDA, or model execution" in s
