"""Verify safety and PowerShell syntax for the optional Windows inventory probe.

Never execute the probe on CI: local WSL startup may consume user resources.
"""
from __future__ import annotations
import os
from pathlib import Path
import platform
import subprocess

ROOT=Path(__file__).resolve().parents[1]
FILE=ROOT/"scripts"/"g7_readonly_host_probe.ps1"


def test_probe_never_starts_linux_or_installs_programs():
    content=FILE.read_text(encoding="utf-8")
    assert "READ_ONLY_MACHINE_DISCOVERY" in content
    assert "NOT_PROBED_REQUIRES_SEPARATE_APPROVAL" in content
    assert "native_cuda_smoke = 'NOT_RUN'" in content
    command_lines=[
        line.strip().lower()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any(line.startswith(("wsl.exe --install", "wsl --install",
                                   "pip install", "sudo ", "winget install",
                                   "gh workflow run", "git push"))
                   for line in command_lines)
    assert "Invoke-Probe 'wsl.exe' @('--list','--verbose')" in content


def test_powershell_parser_on_windows():
    if platform.system() != "Windows":
        return
    env=os.environ.copy()
    env["G7_HOST_PROBE_FILE"]=str(FILE)
    cmd=(
        "$t=$null;$e=$null;"
        "[System.Management.Automation.Language.Parser]::ParseFile("
        "$env:G7_HOST_PROBE_FILE,[ref]$t,[ref]$e) | Out-Null;"
        "if($e.Count -gt 0){$e | ForEach-Object{Write-Error $_};exit 1}"
    )
    result=subprocess.run(["powershell.exe","-NoProfile","-NonInteractive","-Command",cmd],
                          cwd=ROOT,env=env,capture_output=True,text=True,timeout=30)
    assert result.returncode == 0, result.stderr[-1200:]
