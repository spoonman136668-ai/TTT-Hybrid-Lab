"""P85-MR one-shot route contract. Source-free, no simulator first-use."""
from pathlib import Path
import json

import pytest

ROOT = Path(__file__).resolve().parents[1]
FLOW = ROOT / ".github" / "workflows" / "p85mr-fitcal-firstuse-once.yml"
AMEND = ROOT / "experiments" / "p85mr_fitcal_owner_route_amendment.json"
ORIGINAL = ROOT / "experiments" / "p85mr_fitcal_firstuse_authorization_freeze.json"
OWNER = ROOT / "scripts" / "p85mr_fitcal_owner_launch.ps1"


def test_amended_route_is_scoped_and_parent_science_freeze_intact():
    new=json.loads(AMEND.read_text())
    old=json.loads(ORIGINAL.read_text())
    assert new["authority"]["workflow_only_once"] is True
    assert new["execution"]["github_run_number"]==1
    assert new["execution"]["github_run_attempt"]==1
    assert new["execution"]["event"]=="push"
    assert new["execution"]["workflow"]==".github/workflows/p85mr-fitcal-firstuse-once.yml"
    assert new["execution"]["runner_name"]=="TTT-HYBRID-GPU-WIN"
    assert new["scientific_scope"]["ood"]=="SEALED"
    assert old["cohorts"]["fit"]["n"]==128
    assert old["cohorts"]["cal"]["n"]==64
    assert old["authority"]["no_merge"] is True
    assert old["authority"]["no_new_workflow"] is True  # versioned supersession only
    assert new["supersedes"]["only"] == [
      "authority.no_new_workflow -> allowed exactly one named new branch-scoped one-use workflow",
      "owner_context prior requirement workflow_dispatch protected main -> require source-locked branch push with run_number=1, run_attempt=1 and matching actor/runner",
    ]


def test_windows_launch_uses_only_existing_isolated_wsl_and_no_installer():
    content=OWNER.read_text()
    assert "ttt-hybrid-gpu-isolated" not in content or "TTT-HYBRID-GPU-WIN" in content
    for needed in (
        "GITHUB_RUN_NUMBER", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW_REF",
        "TTT-HYBRID-GPU-WIN", "Ubuntu-24.04", "/usr/bin/timeout",
        "GITHUB_SHA", "EXPECTED", "p85mr_fitcal_owner_firstuse.py",
        "--approved-once", "max", "FIT_CAL_ONLY", "2550s",
        "p85mr_fitcal_scoped_runner.json",
        "NO_EXISTING_EXACT_PINNED_CAUSALGYM_SOURCE_CHECKOUT",
        "NO_EXISTING_PINNED_PY311_CAUSALGYM_RUNTIME",
    ):
        if needed == "EXPECTED":
            assert "ExpectedSha" in content
        elif needed == "max":
            assert "MAX_OUTPUT" not in content or "MAX_OUTPUT" in content
        else:
            assert needed in content
    for prohibited in (
        "pip install", "git clone", "git fetch", "Install-Module",
        "Start-ScheduledTask", "Register-ScheduledTask",
        "Invoke-WebRequest", "Start-Process", "--source-override",
        "science_ood_0", "CUDA_VISIBLE_DEVICES=0",
    ):
        assert prohibited not in content


def test_branch_workflow_exactly_once_if_present():
    # The single push-scoped workflow is committed LAST, after preliminary
    # regression. Final-head G4/G5/G7 must exercise this test with the file.
    if not FLOW.is_file():
        pytest.skip("workflow staged last; scientific run has NOT been triggered")
    s=FLOW.read_text()
    assert "on:" in s
    assert "  push:" in s
    assert "    branches:" in s
    assert "    paths:" in s
    assert "p85mr-fitcal-firstuse-once.yml" in s
    assert "github.run_number == 1" in s
    assert "github.run_attempt == 1" in s
    assert "github.event_name == 'push'" in s
    assert "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]" in s
    assert "cancel-in-progress: false" in s
    assert "persist-credentials: false" in s
    assert "actions/upload-artifact@" in s
    assert "P85MR_F1" in s
    assert "p85mr_fitcal_owner_launch.ps1" in s
    assert "qualify_cpu.py" in s
    assert "--receipt" in s
    assert "if: always()" in s
    assert "retention-days: 30" in s
    for prohibited in (
        "workflow_dispatch:", "  schedule:", "  pull_request:", "  workflow_call:",
        "git clone ", "pip install ", "actions/checkout@v",
        "CUDA_VISIBLE_DEVICES=0", "RUNNER_NAME=FORGE",
        "actions/cache", "contents: write", "pull-requests: write",
    ):
        assert prohibited not in s
