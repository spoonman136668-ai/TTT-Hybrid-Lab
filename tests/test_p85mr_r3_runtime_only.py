"""P85MR-R3 environment-only engineering tests; never provision or use scientific source."""
import json
from pathlib import Path
import runpy

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/p85mr_r3_runtime_only.py"
FROZEN = ROOT / "experiments/p85mr_r3_runtime_only_frozen.json"


def module():
    return runpy.run_path(str(SCRIPT), run_name="p85mr_r3_test_import")


def test_r3_frozen_scope_and_exact_dependency_versions():
    p = json.loads(FROZEN.read_text(encoding="utf-8"))
    m = module()
    assert p["schema"] == "ttt-hybrid.p85mr-r3-wsl-runtime-only.v1"
    assert p["parent_sha"] == "55faf21d05de4a6b62af52666130517e3aeb260e"
    assert p["pinned_distributions"] == m["PINS"]
    assert p["new_runtime_relative"] == m["NEW"]
    assert p["single_use"]["run_number"] == 1
    assert p["single_use"]["run_attempt"] == 1
    assert p["single_use"]["no_replay"] is True
    assert "no F1 or F2 replay" in p["prohibitions"]
    assert p["runner"] == "TTT-HYBRID-GPU-WIN"


def test_r3_no_science_or_cross_project_execution_paths():
    code = SCRIPT.read_text(encoding="utf-8")
    # Existing one-use collector stays immutable; this runner repair cannot call it.
    for blocked in (
        "p85mr_fitcal_owner_firstuse.py", "p85mr_fitcal_owner_launch.ps1",
        "causal_gym", "Causal-Gymnasium", "torch.cuda",
        "FORGE-F139-UnifiedController", "workflow_dispatch",
        "apt-get", "sudo ", "github.event_name == 'pull_request'",
    ):
        assert blocked not in code
    assert '"status": "PRE_EFFECT_BLOCKED"' in code
    assert '"source_first_use": False' in code
    assert '"simulator_reset": False' in code
    assert '"training_started": False' in code
    assert '"reserved_data_access": False' in code


def test_r3_owner_environment_denied_before_filesystem_or_installer(monkeypatch, tmp_path):
    m = module()
    for key in ("GITHUB_REPOSITORY", "GITHUB_ACTOR", "GITHUB_EVENT_NAME",
                "GITHUB_REF", "GITHUB_RUN_NUMBER", "GITHUB_RUN_ATTEMPT",
                "RUNNER_NAME", "P85MR_R3_RUNTIME_ONLY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(RuntimeError, match="R3_OWNER_ROUTE_DENIED"):
        m["admission"](tmp_path)


def test_r3_missing_runtime_does_not_try_to_install(tmp_path):
    assert module()["qualified"](tmp_path / "absent/bin/python") is False
