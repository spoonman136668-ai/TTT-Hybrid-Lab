"""Post-consumption G7-R7 static audit; no native or host execution."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_readonly_one_shot_is_retired_not_retriggerable():
    assert not (ROOT / ".github/workflows/g7-r7-single-readonly-receipt.yml").exists()


def test_qualified_original_error_evidence_is_durable_and_scoped():
    report = (ROOT / "docs/G7_R6_RESULT.md").read_text(encoding="utf-8")
    for marker in (
        "G7-R7 diagnosis",
        "PASS_ORIGINAL_ERROR_CUSTODY_ONLY",
        "42285407805",
        "13536149391",
        "94ed67b32c164245c05fd1392be8aad3f2b2881b5c8b0a503aaf76b24a38c3b2",
        "4d6cd2d716a4ea417d2e5a424e6878c9bad4ab93a0e7e5351ceb0c4158700f29",
        "FUSED_CAUSAL_CONV1D_NOT_CALLABLE",
        "ssd_combined.py",
        "raw exception text and traceback",
        "No WSL",
    ):
        assert marker.lower() in report.lower(), marker
