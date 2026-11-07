"""G7-R14: real hosted Windows PowerShell syntax test; no isolated runner."""
import json
import platform
import subprocess
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
PS=ROOT/"scripts/g7_r14_windows_cmd_invocation.ps1"

def test_inert_script_and_original_failed_receipt():
    source=PS.read_text(encoding="utf-8")
    for token in (
        "Get-Command -Name 'cmd.exe' -CommandType Application",
        "$app.Source /d /c",
        "$exe = [string]$app.Definition",
        "$o = @(& $exe /d /c",
        "[IO.File]::Exists($exe)",
        "PASS_COMMAND_NOT_FOUND",
        "original_runner='NOT_TOUCHED'",
        "cuda='NOT_RUN'",
    ):
        assert token in source, token
    for banned in (
        "wsl.exe", "nvidia-smi", "nvcc", "torch.cuda",
        "apt install", "apt-get", "pip install",
        "Start-Process", "Invoke-Expression", "git push",
    ):
        assert banned not in source, banned
    record=(ROOT/"docs/G7_R13_RESULT.md").read_text(encoding="utf-8")
    assert "probe_phase=WSL_PING" in record
    assert "transport_exception_type=CommandNotFoundException" in record

@pytest.mark.skipif(platform.system()!="Windows", reason="PowerShell native call test is Windows-only")
def test_actual_windows_cmd_invocation_and_negative_control():
    proc=subprocess.run(
        ["powershell.exe","-NoProfile","-NonInteractive",
         "-ExecutionPolicy","Bypass","-File",str(PS)],
        cwd=ROOT,capture_output=True,text=True,timeout=30,check=False,
    )
    assert proc.returncode==0, f"rc={proc.returncode} stdout={proc.stdout[-1200:]} stderr={proc.stderr[-500:]}"
    receipt=json.loads(proc.stdout.strip())
    assert receipt["schema"]=="ttt-hybrid.g7-r14.windows-native-launch.v1"
    assert receipt["status"]=="PASS_HOSTED_WINDOWS_SYNTAX_ONLY"
    assert receipt["source_info"]=="NONEMPTY"
    assert receipt["definition_info"]=="NONEMPTY"
    assert receipt["explicit_path_call"]=="PASS"
    assert receipt["negative_missing_command"]=="PASS_COMMAND_NOT_FOUND"
    assert receipt["legacy_call"] in ("PASS","FAILED_EXCEPTION","FAIL_OUTPUT")
    assert receipt["original_runner"]=="NOT_TOUCHED"
    assert receipt["cuda"]=="NOT_RUN"
    assert receipt["linux"]=="NOT_RUN"
    assert receipt["model"]=="NOT_RUN"
    out=ROOT/"local_runs/g7_r14_windows_native_launcher.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n",encoding="utf-8")
