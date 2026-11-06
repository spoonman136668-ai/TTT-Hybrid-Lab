"""G7-R12 one-use read-only inventory: source-only retirement and receipt integrity."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_consumed_one_use_workflow_retired():
    assert not (ROOT / ".github/workflows/g7-r12-one-shot-compiler-inventory.yml").exists()


def test_g7_r12_blocked_status_provenance_immutable():
    report = (ROOT / "docs/G7_R12_RESULT.md").read_text(encoding="utf-8")
    for marker in (
        "42314692869",
        "13549323708",
        "2edc9d0b78f82c06fa8cc73c40f295a80fd36ce6",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "601cb4a92db315f39a5bcdca308962d4231a87474dfc67491b06a5305164fdc4",
        "56781ed0316ecd155ca3ace4d92bca72115e390590f24362bd790ac0bcb22aef",
        "BLOCKED_PROBE_UNVERIFIED",
        "source_bound=true",
        "wsl_started_for_inventory_only=true",
        "gcc=UNVERIFIED",
        "clang=UNVERIFIED",
        "NO RETRY",
    ):
        assert marker in report, marker


def test_unverified_toolchain_not_treated_as_absent():
    report = (ROOT / "docs/G7_R12_RESULT.md").read_text(encoding="utf-8")
    assert "No reliable absence or presence conclusion" in report
    assert "SCIENTIFIC: UNVERIFIED" in report
    assert "DEPLOYMENT: NOT AUTHORIZED" in report
    assert "No installation, compiler invocation, CUDA, or model execution" in report
