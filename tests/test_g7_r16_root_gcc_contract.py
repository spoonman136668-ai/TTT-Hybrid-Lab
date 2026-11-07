"""G7-R16: consumed root-GCC single use and immutable result negative controls."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/".github/workflows/g7-r16-root-gcc-forward.yml"
PROBE=ROOT/"scripts/g7_r16_compiler_probe.py"
REPORT=ROOT/"docs/G7_R16_RESULT.md"

def test_consumed_one_shot_removed_from_tree():
    assert not WF.exists()

def test_original_run_hashes_and_stages_documented():
    s=REPORT.read_text(encoding="utf-8")
    for marker in (
        "42323786912",
        "13554520874",
        "4a32444ed9ce3b8246aa8b01e41c9ce83f8a5dee",
        "179ab666e1d4368623eb22748b5cd13fe5c59352ff22e0c559e2e0986f07c19d",
        "e24e53005e4afd9328b641eef2d58c9f8734080b148460a3eb7d2c5f6673fea9",
        "c769c5b6c5ec7d564caf557e5d335dbc647c3b778504b88ac4f255f49719b6d2",
        "source_custody=PASS",
        "root_context=ROOT_UID_ZERO_CONFIRMED",
        "apt_update=PASS",
        "apt_install=PASS",
        "compiler_after=PRESENT_UNQUALIFIED",
        "BLOCKED_HOST_C_SHARED_BUILD_FAILED",
        "native_forward=NOT_RUN",
        "NO RETRY",
    ):
        assert marker in s, marker

def test_no_gpu_science_or_additional_privilege_claims():
    s=REPORT.read_text(encoding="utf-8")
    for marker in (
        "SCIENTIFIC: UNVERIFIED",
        "DEPLOYMENT: NOT AUTHORIZED",
        "No GPU forward was launched",
        "libc6-dev",
        "hypothesis, not an independently observed root cause",
    ):
        assert marker in s, marker

def test_retained_compiler_selection_probe_does_not_mutate():
    s=PROBE.read_text(encoding="utf-8")
    ast.parse(s)
    assert "native.source_custody(args.expected_sha)" in s
    assert "native.selected_compiler()" in s
    assert "sys.path.insert(0" in s
    for forbidden in ("apt-get", "sudo", "torch.cuda", "wsl.exe", "git checkout"):
        assert forbidden not in s
