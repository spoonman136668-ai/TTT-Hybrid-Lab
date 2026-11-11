"""P85MR-F3: frozen owner-only route, negative and no-source-consumption tests."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))
import p85mr_f3_fitcal_once as f3
import p85mr_fitcal_owner_firstuse as legacy


def permitted():
    return {
        "GITHUB_REPOSITORY": "spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_ACTOR": "spoonman136668-ai",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF": "refs/heads/research/p85mr-f3-repaired-fitcal-once-20261111",
        "GITHUB_WORKFLOW_REF":
            "spoonman136668-ai/TTT-Hybrid-Lab/.github/workflows/"
            "p85mr-f3-repaired-fitcal-once.yml@refs/heads/"
            "research/p85mr-f3-repaired-fitcal-once-20261111",
        "GITHUB_RUN_NUMBER": "1", "GITHUB_RUN_ATTEMPT": "1",
        "RUNNER_NAME": "TTT-HYBRID-GPU-WIN",
        "RUNNER_OS": "Windows", "RUNNER_ARCH": "X64",
        "P85MR_F3_OWNER_SCOPE": "FIT_CAL_ONLY",
        "GITHUB_RUN_ID": "43099999911",
        "GITHUB_SHA": "a" * 40,
    }


def test_f3_exact_new_prospective_scope_and_parent_custody():
    f = f3.locked_f3()
    assert f["parent"]["sha"] == "b4c273251a3a8e24fb22ccd850579111e5a73527"
    assert f["parent"]["f2_original_collector_blob"] == f3.ORIGINAL_COLLECTOR_BLOB
    assert legacy.blob((EXPERIMENTS / "p85mr_fitcal_owner_firstuse.py").read_bytes()) == f3.ORIGINAL_COLLECTOR_BLOB
    assert f["authority"]["one_run_number"] == 1
    assert f["authority"]["one_run_attempt"] == 1
    assert f["execution"]["workflow"] == f3.WORKFLOW
    assert f["runtime"]["package_pins"] == f3.DEPS
    assert f["runtime"]["path_alias"] == "p85mr-causalgym-cpu311-r3"
    assert f["source"]["source_blobs"] == legacy.SOURCE_BLOBS
    assert f["scientific"]["no_ood"] is True
    assert f["scientific"]["no_training"] is True
    assert f["scientific"]["need_independent_witness"] is True
    assert f3.COHORTS == (("fit", 128), ("cal", 64))
    assert f3.MAX_ACTIONS == 18432
    assert f3.MAX_SECONDS == 2400
    assert legacy.locked_f2_route()["authority"]["second_scoped_attempt"].startswith("ONE")


@pytest.mark.parametrize("field,value", [
    ("GITHUB_REPOSITORY", "spoonman136668-ai/FORGE"),
    ("GITHUB_ACTOR", "outside-actor"),
    ("GITHUB_EVENT_NAME", "pull_request"),
    ("GITHUB_REF", "refs/heads/main"),
    ("GITHUB_WORKFLOW_REF", "different.yml"),
    ("GITHUB_RUN_NUMBER", "2"),
    ("GITHUB_RUN_ATTEMPT", "2"),
    ("RUNNER_NAME", "FORGE-CODEX-REMOTE"),
    ("RUNNER_OS", "Linux"),
    ("RUNNER_ARCH", "ARM64"),
    ("P85MR_F3_OWNER_SCOPE", "OOD"),
    ("GITHUB_RUN_ID", "none"),
    ("GITHUB_SHA", "unverified"),
])
def test_untrusted_owner_context_rejected(field, value):
    env = permitted()
    env[field] = value
    with pytest.raises(PermissionError):
        f3.owner_context(env)


def test_f3_only_new_owner_scoped_context_accepted():
    observed = f3.owner_context(permitted())
    assert observed["source_route"] == "P85MR-F3-INDEPENDENT-ONCE"
    assert observed["project_sha"] == "a" * 40
    assert observed["run_id"] == "43099999911"
    assert observed["attempt"] == 1


def test_f3_never_invokes_original_owner_or_effect_entrypoints():
    source = (EXPERIMENTS / "p85mr_f3_fitcal_once.py").read_text(encoding="utf-8")
    assert "legacy.collect_one_shot(" not in source
    assert "legacy.owner_context(" not in source
    assert "legacy._load_scoped_source(src)" in source
    assert 'attempt.mkdir(mode=0o700, exist_ok=False)' in source
    assert '"ATTEMPT.json"' in source
    assert source.index('"ATTEMPT.json"') < source.index("legacy._load_scoped_source(src)")
    assert '"ood_created": False' in source
    assert '"trained_baselines": False' in source
    assert '"science_pass": False' in source
    assert '"original_f1_f2_runs_replayed": False' in source


def test_no_external_source_opened_from_refused_owner(monkeypatch, tmp_path):
    def effect_is_forbidden(*args, **kwargs):
        raise AssertionError("EXTERNAL_IMPORT_OR_SIMULATOR_EFFECT_FORBIDDEN")
    monkeypatch.setattr(legacy, "_load_scoped_source", effect_is_forbidden)
    monkeypatch.setattr(f3, "preflight", effect_is_forbidden)
    env = permitted()
    env["GITHUB_RUN_ATTEMPT"] = "2"
    with pytest.raises(PermissionError, match="F3_EXACT_ONCE_OWNER_ROUTE_DENIED"):
        f3.collect_one_shot(source_root=tmp_path, project_root=tmp_path, environ=env)
    assert not (tmp_path / "evidence").exists()


def test_frozen_seed_partition_and_existing_causal_controls_unchanged():
    assert legacy.preflight_partition()["cross_split_collisions"] == 0
    assert legacy.preflight_partition()["new_unique_seeds"] == 448
    scope = legacy.locked_scope()
    assert scope["cohorts"]["fit"]["n"] == 128
    assert scope["cohorts"]["cal"]["n"] == 64
    assert legacy.planned_discrete_action("fit", 0, 1) in (0, 1)
    assert legacy.proposed_seed("fit", 0) != legacy.proposed_seed("cal", 0)
