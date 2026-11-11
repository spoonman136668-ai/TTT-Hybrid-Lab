"""F3 one-shot external route engineering-only static guards.

Tests never check out/import CartPoleWind, never create source worlds, reset
or issue real interventions. F1/F2 historical routes remain unchanged.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / ".github/workflows/p85mr-f3-repaired-fitcal-once.yml"
LAUNCH = ROOT / "scripts/p85mr_f3_owner_launch.ps1"


def test_f3_launcher_is_dedicated_one_shot_and_spent_f2_unmodified():
    s = LAUNCH.read_text(encoding="utf-8")
    assert "refs/heads/research/p85mr-f3-repaired-fitcal-once-20261111" in s
    assert "p85mr-f3-repaired-fitcal-once.yml" in s
    assert "P85MR_F3_OWNER_SCOPE=FIT_CAL_ONLY" in s
    assert "p85mr-causalgym-cpu311-r3/venv/bin/python" in s
    assert "p85mr-upstream-f3" in s
    assert "ATTEMPT.json" in s
    assert "ATTEMPT_CONSUMED_NO_REPLAY" in s
    assert "NO_REPLAY" in s
    assert "'F3_RUNTIME_IMPORT_PASS'" in s
    assert "F3_UPSTREAM_FROZEN_BLOB_MISMATCH" in s
    assert "SOURCE_COLLECTED_INDEPENDENT_WITNESS_PENDING" in s
    for disallowed in ("p85mr_fitcal_owner_launch.ps1", "p85mr_fitcal_owner_firstuse.py",
                       "apt-get", "pip install", "wsl --install",
                       "FORGE-F139-UnifiedController", "run_codex"):
        assert disallowed not in s


def test_f3_future_workflow_frozen_trigger_contract():
    # The trigger file is intentionally created only AFTER final parent
    # G4/G5/G7 qualification, so absent before that and tests must allow it.
    if not WF.exists():
        return
    s = WF.read_text(encoding="utf-8")
    assert re.search(r"(?m)^on:\s*\n\s+push:\s*$", s)
    assert "research/p85mr-f3-repaired-fitcal-once-20261111" in s
    assert "p85mr-f3-repaired-fitcal-once.yml" in s
    assert "github.run_number == 1" in s and "github.run_attempt == 1" in s
    assert "TTT-HYBRID-GPU-WIN" in s
    assert "ttt-hybrid-gpu-isolated" in s
    assert "persist-credentials: false" in s
    assert "c94e9a4cd3641588b6b8507e71ccee5c27032bdf" in s
    assert "p85mr_f3_owner_launch.ps1" in s
    assert "source sha" not in s.lower() or "EXACT_SHA" in s
    for disallowed in ("workflow_dispatch:", "schedule:", "pull_request:",
                       "repository_dispatch:", "secrets.", "apt-get", "pip install"):
        assert disallowed not in s
