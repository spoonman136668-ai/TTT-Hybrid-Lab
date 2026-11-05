"""Contract for the bounded, operator-initiated TTT-only runner registration helper."""
from pathlib import Path
import platform
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "Register-TTT-GpuRunner.ps1"
DOC = ROOT / "docs" / "DEDICATED_WINDOWS_GPU_RUNNER.md"

def test_runner_registration_source_is_bounded_and_manual():
    src = SCRIPT.read_text(encoding="utf-8")
    for required in (
        "spoonman136668-ai/TTT-Hybrid-Lab", "TTT-HYBRID-GPU-WIN",
        "ttt-hybrid-gpu-isolated", "C:\\actions-runner-ttt-hybrid",
        "Get-FileHash", "SHA256", "TTT_RUNNER_ALREADY_REGISTERED_NO_REPLACEMENT",
        "TTT_RUNNER_DIR_NOT_EMPTY_NO_REPLACEMENT", "Get-Command -Name 'nvidia-smi.exe'",
        "TTT_INSUFFICIENT_FREE_GPU_MEMORY", "Read-Host", "-AsSecureString",
        "--url $repo --token $plain --name $name --labels $label",
        "--unattended", "ZeroFreeBSTR", "TTT_RUNNER_REGISTERED=CHECK_GITHUB_UI"
    ):
        assert required in src
    for forbidden in (
        "Set-ExecutionPolicy", "--replace", "svc install", "schtasks", "Start-Service",
        "wsl.exe", "pip install", "git push", "FORGE-CODEX-REMOTE",
        "WINGLESS-UP-B", "YGG-A-LINKDEADKB", "Invoke-Expression",
    ):
        assert forbidden not in src

def test_document_does_not_claim_runtime_or_cuda_acceptance():
    doc = DOC.read_text(encoding="utf-8")
    assert "currently unprotected" in doc
    assert "do not isolate GPU memory" in doc
    assert "read-only" in doc
    assert "registration token" in doc

def test_windows_script_parser_only_no_registration():
    if platform.system() != "Windows":
        return
    cmd = ("$t=$null;$e=$null;"
           "[System.Management.Automation.Language.Parser]::ParseFile("
           "'scripts/Register-TTT-GpuRunner.ps1',[ref]$t,[ref]$e)|Out-Null;"
           "if($e.Count -gt 0){$e|ForEach-Object{Write-Error $_};exit 1}")
    result = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive",
                             "-Command", cmd], cwd=ROOT, capture_output=True,
                             text=True, timeout=20)
    assert result.returncode == 0, result.stderr[-2000:]
