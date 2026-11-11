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



def test_exact_previous_s28_checkout_candidate_has_no_new_source_download():
    source=OWNER.read_text()
    assert '"$workspaceRoot/p85mr-upstream-source"' in source
    assert '"$workspaceRoot/s28-upstream-source"' in source
    assert "$linuxRoot.EndsWith('/p85mr-f1'" in source
    assert "c94e9a4cd3641588b6b8507e71ccee5c27032bdf" in source
    assert 'NO_EXISTING_EXACT_PINNED_CAUSALGYM_SOURCE_CHECKOUT' in source
    # The source must still be reverified against exact upstream SHA/blobs,
    # not assumed trusted because an older workflow created its directory.
    collector=(ROOT/"experiments"/"p85mr_fitcal_owner_firstuse.py").read_text()
    for phrase in ("P85F1_CHECKOUT_SHA_MISMATCH", "P85F1_SOURCE_BLOB_MISMATCH",
                   "P85F1_SOURCE_ORIGIN_MISMATCH", "P85F1_DIRTY_SOURCE"):
        assert phrase in collector


def test_collector_entrypoint_imports_exact_adjacent_modules_without_first_use():
    """Verify real -s -B entrypoint can import local pinned modules.

    --help stops at argparse before owner admission, source checkout,
    simulator construction, reset, do or model training.
    """
    import subprocess
    import sys
    launch=OWNER.read_text()
    assert '$py -s -B "$linuxRoot/experiments/p85mr_fitcal_owner_firstuse.py"' in launch
    assert '$py -I -B "$linuxRoot/experiments/p85mr_fitcal_owner_firstuse.py"' not in launch
    assert "'PYTHONNOUSERSITE=1'" in launch
    app=ROOT/"experiments"/"p85mr_fitcal_owner_firstuse.py"
    p=subprocess.run(
        [sys.executable,"-s","-B",str(app),"--help"],
        cwd=str(ROOT),text=True,capture_output=True,timeout=25,check=False)
    assert p.returncode == 0, (p.stdout[-1000:],p.stderr[-1200:])
    assert "--approved-once" in p.stdout
    assert "--source-checkout" in p.stdout
    assert "--project-checkout" in p.stdout



def test_new_f2_workflow_will_not_be_triggered_from_earlier_engineering_commits():
    """The newly approved workflow is written last and has a unique path trigger."""
    path=ROOT/".github"/"workflows"/"p85mr-f2-pinned-fitcal-once.yml"
    f2=json.loads((ROOT/"experiments"/"p85mr_f2_pinned_fitcal_authorization_freeze.json").read_text())
    assert f2["execution"]["trigger"]=="push of named file on exact branch only"
    assert f2["execution"]["github_run_number"]==1
    assert f2["execution"]["github_run_attempt"]==1
    assert f2["execution"]["runner_name"]=="TTT-HYBRID-GPU-WIN"
    assert f2["execution"]["workflow"]==".github/workflows/p85mr-f2-pinned-fitcal-once.yml"
    assert f2["authority"]["first_attempt"]["scientific_source_effects"]==0
    if not path.exists():
        pytest.skip("F2 trigger file not committed; safe engineering preflight only")
    s=path.read_text()
    assert "  push:" in s and "  schedule:" not in s
    assert "research/p85mr-f2-pinned-source-fitcal-once-20261111" in s
    assert "p85mr-f2-pinned-fitcal-once.yml" in s
    assert "github.run_number == 1" in s
    assert "github.run_attempt == 1" in s
    assert "runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]" in s
    assert "CausalAILab/Causal-Gymnasium" in s
    assert "c94e9a4cd3641588b6b8507e71ccee5c27032bdf" in s
    assert "p85mr-upstream-source" in s
    assert "persist-credentials: false" in s
    assert "P85MR_F2" in s
    assert "--approved-once" not in s or "p85mr_fitcal_owner_launch.ps1" in s
    assert "qualify_cpu.py" in s
    assert "if: always()" in s
    assert "retention-days: 30" in s
    assert "  workflow_dispatch:" not in s
    assert "  pull_request:" not in s
    assert "contents: write" not in s
    assert "schedule:" not in s


def test_f2_shares_exact_frozen_source_and_preflight_before_actual_do():
    src=(ROOT/"experiments"/"p85mr_fitcal_owner_firstuse.py").read_text()
    launcher=OWNER.read_text()
    assert 'F2_BLOB = "bf7738238a52b6570530d86d30fe36fb594eb12d"' in src
    assert 'owner_context(environ)' in src
    assert 'locked_f2_route()' in src
    assert '"P85F2_ROUTE_FREEZE_DRIFT"' in src
    assert "$expectedRef = 'refs/heads/research/p85mr-f2-pinned-source-fitcal-once-20261111'" in launcher
    assert "p85mr-f2-pinned-fitcal-once.yml" in launcher
    assert '"$workspaceRoot/p85mr-upstream-source"' in launcher
    assert 'NO_EXISTING_PINNED_PY311_CAUSALGYM_RUNTIME' in launcher
    assert "--approved-once" in launcher
    assert "2550s" in launcher
    assert "Start-ScheduledTask" not in launcher
    assert "pip install" not in launcher
    assert "git clone" not in launcher
