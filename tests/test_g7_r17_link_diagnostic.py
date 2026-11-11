"""G7-R17 pure source-only controls; NEVER launches local WSL or a GPU."""
from __future__ import annotations
import ast
import importlib.util
from pathlib import Path
import sys
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "scripts/g7_r17_link_diagnostic.py"

# Tests intentionally import the diagnostic with an inert stand-in of the old
# qualification module: no actual local host source or compiler access.
class Blocked(Exception):
    pass

old = types.ModuleType("g7_r15_compiler_repair_forward")
old.Blocked = Blocked
old.source_custody = lambda _sha: None
old.selected_compiler = lambda: None
# The stand-in must not contaminate other G4/G5/G7 test modules in the same
# pytest process. Only this diagnostic's module retains a reference to it.
_previous = sys.modules.get(old.__name__)
sys.modules[old.__name__] = old
try:
    spec = importlib.util.spec_from_file_location("g7_r17_link_diagnostic", SRC)
    assert spec and spec.loader
    subject = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(subject)
finally:
    if _previous is None:
        sys.modules.pop(old.__name__, None)
    else:
        sys.modules[old.__name__] = _previous

@pytest.mark.parametrize("message,expected", [
    ("/usr/bin/ld: cannot find crti.o", "C_RUNTIME_STARTUP_OBJECT_MISSING"),
    ("ld: cannot find -lc", "LIBC_LINKER_INPUT_MISSING"),
    ("ld: libc.so not found", "LIBC_LINKER_INPUT_MISSING"),
    ("ld: cannot find libgcc_s.so", "GCC_RUNTIME_LINKER_INPUT_MISSING"),
    ("fatal error: Permission denied", "HOST_BUILD_PERMISSION_DENIED"),
    ("no space left on device", "HOST_BUILD_STORAGE_EXHAUSTED"),
    ("cannot execute cc1", "HOST_COMPILER_EXECUTION_FAILED"),
    ("the original raw secret token appeared", "UNCLASSIFIED_HOST_LINK_FAILURE"),
])
def test_sanitized_classification(message, expected):
    result = subject.classify_linker_error(message)
    assert result == expected
    assert message not in result

@pytest.mark.parametrize("code,status,expected", [
    (0,"install ok installed","INSTALLED"),
    (1,"","NOT_INSTALLED_OR_QUERY_FAILED"),
    (0,"deinstall ok config-files","REMOVED_CONFIG_ONLY"),
    (0,"unknown unexpected value","UNKNOWN_STATUS"),
])
def test_package_status_no_inference(code,status,expected):
    assert subject.classify_dev_package(code,status) == expected


def test_script_is_inert_on_import_and_sanitizes_receipt():
    code = SRC.read_text(encoding="utf-8")
    tree = ast.parse(code)
    top_calls = [n.value for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    assert [ast.unparse(call.func) for call in top_calls] == ["sys.path.insert"]
    for forbidden in ("apt-get", "sudo", "torch.cuda", "wsl.exe", "git checkout", "pip install", "cudaMalloc", "requests.post"):
        assert forbidden not in code
    for required in ("RECEIPT_ALREADY_EXISTS_NO_REPLAY", "native.source_custody(expected_sha)",
                     "g7_r15_compiler_repair_forward", "NOT_ACCESSED", "NOT_RUN",
                     'output.open("x"', "MAX_STDERR"):
        assert required in code


def test_no_compiler_fails_closed(monkeypatch, tmp_path):
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(old, "source_custody", lambda sha: None)
    monkeypatch.setattr(old, "selected_compiler", lambda: None)
    # This exercise uses only mocks; creates a local JSON receipt, no external commands.
    report, code = subject.run("a"*40, tmp_path / "local_runs" / "g7r17.json")
    assert code == 2
    assert report["status"] == "BLOCKED_NO_COMPILER"
    assert report["gpu"] == "NOT_ACCESSED"
    assert (tmp_path / "local_runs" / "g7r17.json").is_file()
    report2, code2 = subject.run("a"*40, tmp_path / "local_runs" / "g7r17.json")
    assert code2 == 2
    assert report2["status"] == "BLOCKED_RECEIPT_ALREADY_EXISTS_NO_REPLAY"


def test_receipt_outside_project_rejected(monkeypatch,tmp_path):
    monkeypatch.setattr(subject,"ROOT",tmp_path/"repo")
    report, code=subject.run("a"*40,tmp_path/"elsewhere.json")
    assert code == 2
    assert report["status"] == "BLOCKED_UNAUTHORIZED_RECEIPT_DESTINATION"
    assert not (tmp_path / "elsewhere.json").exists()

@pytest.mark.parametrize("build_ok,expected_status,expected_category", [
    (True, "PASS_HOST_C_SHARED_LINK_ONLY", None),
    (False, "FAIL_HOST_C_SHARED_LINK", "C_RUNTIME_STARTUP_OBJECT_MISSING"),
])
def test_host_shared_probe_with_mocked_compiler(monkeypatch, tmp_path, build_ok,
                                                expected_status, expected_category):
    from subprocess import CompletedProcess
    monkeypatch.setattr(subject, "ROOT", tmp_path)
    monkeypatch.setattr(old, "source_custody", lambda sha: None)
    monkeypatch.setattr(old, "selected_compiler", lambda: ("/usr/bin/true", "gcc"))
    def fake_probe(argv, *, input_text=None, timeout=15):
        if argv[0].endswith("dpkg-query"):
            return CompletedProcess(argv, 0, "install ok installed", "")
        assert argv[0] == "/usr/bin/true"
        assert input_text == subject.C_SOURCE
        if build_ok:
            Path(argv[-1]).write_bytes(b"fixed-inert-fixture")
        return CompletedProcess(argv, 0 if build_ok else 1, "", "" if build_ok else "ld: cannot find crti.o")
    monkeypatch.setattr(subject, "fixed_probe", fake_probe)
    result, exit_code=subject.run("a"*40,tmp_path/"local_runs"/"one-shot.json")
    assert result["status"] == expected_status
    assert result["link_failure_category"] == expected_category
    assert result["libc6_dev"] == "INSTALLED"
    assert result["gpu"] == "NOT_ACCESSED"
    assert result["model_training"] == "NOT_RUN"
    assert exit_code == (0 if build_ok else 1)
    text=(tmp_path/"local_runs"/"one-shot.json").read_text()
    assert "cannot find crti.o" not in text
