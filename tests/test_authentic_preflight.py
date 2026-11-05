"""Exact-source identity and fail-closed admission negative controls."""
import json
from pathlib import Path
import subprocess
import sys
from authentic_preflight import source_check, module_location_check, hardware_check, evaluate


def git(repo, *cmd):
    p = subprocess.run(["git", "-C", str(repo), *cmd], text=True, capture_output=True)
    assert p.returncode == 0, (cmd, p.stderr)
    return p.stdout.strip()


def make_repo(tmp_path):
    repo = tmp_path / "upstream"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "test")
    git(repo, "config", "user.email", "test@example.invalid")
    (repo / "ttt.py").write_text("# exact contents\n")
    git(repo, "add", "ttt.py")
    git(repo, "commit", "-qm", "freeze")
    return repo, {"sha": git(repo, "rev-parse", "HEAD"),
                  "required_files": ["ttt.py"],
                  "observed_blob_sha": {"ttt.py": git(repo,"rev-parse","HEAD:ttt.py")}}


def test_pin_positive_exact_clean(tmp_path):
    repo, pin = make_repo(tmp_path)
    assert source_check(repo, pin)["reason"] == "EXACT_SOURCE_CLEAN"


def test_sha_mismatch_fail_closed(tmp_path):
    repo, pin = make_repo(tmp_path)
    pin["sha"] = "0"*40
    assert source_check(repo, pin)["reason"] == "SHA_MISMATCH_OR_UNREADABLE"


def test_dirty_tracked_file_rejected(tmp_path):
    repo, pin = make_repo(tmp_path)
    (repo / "ttt.py").write_text("# uncommitted mutation\n")
    assert source_check(repo, pin)["reason"] == "DIRTY_OR_UNREADABLE_TREE"


def test_untracked_file_rejected(tmp_path):
    repo, pin = make_repo(tmp_path)
    (repo / "unexpected.py").write_text("# untracked")
    assert source_check(repo, pin)["reason"] == "DIRTY_OR_UNREADABLE_TREE"


def test_wrong_blob_rejected_even_if_commit_matches(tmp_path):
    repo, pin = make_repo(tmp_path)
    pin["observed_blob_sha"]["ttt.py"] = "f"*40
    assert source_check(repo, pin)["reason"] == "BLOB_MISMATCH"


def test_missing_upstream_rejected(tmp_path):
    repo = tmp_path / "absent"
    pin = {"sha": "f"*40, "required_files": ["ttt.py"], "observed_blob_sha": {"ttt.py": "a"*40}}
    assert source_check(repo, pin)["reason"] == "SHA_MISMATCH_OR_UNREADABLE"


def test_unavailable_import_rejected(tmp_path):
    assert module_location_check("an_unimportable_module_zzzzxxx", tmp_path)["reason"] == "MODULE_UNAVAILABLE"


def test_nonmatching_import_origin_rejected(tmp_path):
    assert module_location_check("json", tmp_path)["reason"] == "UNPINNED_INSTALL"


def test_local_cpu_environment_is_not_source_runtime_pass():
    # The test is portable: on a CUDA Linux runner it should instead report PASS.
    h = hardware_check()
    assert isinstance(h["pass"], bool)
    assert h["pass"] == (h.get("reason") == "LINUX_CUDA_PRESENT")


def test_full_gate_fails_without_sources(tmp_path):
    root = tmp_path / "missing"
    result = evaluate(root, root)
    assert result["status"] == "BLOCKED"
    assert result["scientific_qualification"] == "UNVERIFIED"
