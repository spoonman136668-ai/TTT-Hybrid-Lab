"""P85MR-F4 exact contract negative controls; no external data during default CI."""
import json
from pathlib import Path
import sys
import zipfile
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
import p85mr_f4_train_controls as f4


def test_f4_freeze_bound_and_no_data_opened():
    p = f4.locked_protocol(ROOT)
    assert p["schema"] == "ttt-hybrid.p85mr-f4-actual-fitcal-conventional-controls-freeze.v1"
    assert p["parent"]["sha"] == "00b3f45f47660af4c1016c64b656160975982227"
    assert p["parent"]["source_owner_run"] == 43073165842
    assert p["source"]["fit_events"] + p["source"]["cal_events"] == 4447
    assert p["source"]["artifact_zip_entries"] == 11
    assert p["training"]["max_updates"] == 800
    assert p["controls"]["seeds"] == [985201, 985203, 985207]
    assert p["execution"]["runner"] == "TTT-HYBRID-GPU-WIN"
    assert p["stop"]["no_ood_unseal"] is True
    assert len(f4.EXACT_ENTRIES) == 11


def test_f4_owner_denied_before_archive_access(monkeypatch):
    for key in ("GITHUB_REPOSITORY", "GITHUB_ACTOR", "GITHUB_EVENT_NAME",
                "GITHUB_REF", "GITHUB_WORKFLOW_REF", "GITHUB_RUN_NUMBER",
                "GITHUB_RUN_ATTEMPT", "RUNNER_NAME", "RUNNER_OS",
                "RUNNER_ARCH", "P85MR_F4_FITCAL_CONTROLS_ONLY"):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(PermissionError, match="F4_NOT_EXACT_OWNER_ONE_SHOT"):
        f4.owner_admission(ROOT)


def test_f4_invalid_zip_rejected_before_loading_records(tmp_path):
    file = tmp_path / "invalid-source.zip"
    with zipfile.ZipFile(file, "w") as z:
        z.writestr("evidence/p85mr_fitcal_f3/ONCE/fit.json", "{}")
    with pytest.raises(ValueError, match="F4_SOURCE_ARTIFACT_ZIP_SHA_DRIFT"):
        f4.read_source_zip(file)


def test_f4_no_simulation_or_model_promotion_code():
    src = (ROOT / "experiments/p85mr_f4_train_controls.py").read_text()
    for forbidden in (
        "import causal_gym", "env.reset(", "env.do(", "Causal-Gymnasium",
        "torch.load(", "torch.save(", "workflow_dispatch", "schedule.create",
        "pull_request_target", "subprocess.Popen", "subprocess.call",
    ):
        assert forbidden not in src, forbidden
    assert "allow_pickle=False" in src
    assert 'scientific_pass": False' in src
    assert '"ood_consumed": False' in src
    assert '"new_source_effects": False' in src
    assert "MAX_UPDATES" in src and "frozen_indices" in src


def test_f4_original_baseline_contract_still_unmodified():
    import hashlib
    src = (ROOT / "experiments/p85mr_strong_baseline_freeze.json").read_bytes()
    blob = hashlib.sha1(f"blob {len(src)}\0".encode() + src).hexdigest()
    assert blob == "85ae6d1eee3332ef55a653b678733e012cb57158"
    old = (ROOT / "experiments/p85mr_t1_fitcal_engine_freeze.json").read_bytes()
    old_blob = hashlib.sha1(f"blob {len(old)}\0".encode() + old).hexdigest()
    assert old_blob == "7745855deaafdb80117e68ba437af27059005524"


def test_f4r1_only_postblock_authorized_route():
    r = f4.locked_r1_route(ROOT)
    assert r["original_f4"]["run_id"] == 43079777841
    assert r["original_f4"]["model_training_updates"] == 0
    assert r["original_f4"]["original_attempt_consumed"] is True
    assert f4.WORKFLOW == ".github/workflows/p85mr-f4r1-postblock-fitcal-once.yml"
    assert f4.FROZEN_WORKFLOW == ".github/workflows/p85mr-f4-train-controls-once.yml"


def test_f4r1_python_isolated_cli_import_only_no_source_use():
    # This exact -I -B invocation previously failed BEFORE training on owner WSL.
    # It must import the original P80/P85 adapters in both Linux and Windows.
    import subprocess
    cmd = [sys.executable, "-I", "-B",
           str(ROOT / "experiments/p85mr_f4_train_controls.py"), "--help"]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    assert completed.returncode == 0, completed.stderr[-1500:]
    assert "--source-archive" in completed.stdout
    assert "--receipt" in completed.stdout
