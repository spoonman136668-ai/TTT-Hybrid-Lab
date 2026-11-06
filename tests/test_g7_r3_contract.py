"""Static G7-R3 authorization and non-CUDA-execution contracts."""
from __future__ import annotations

import ast
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / '.github/workflows/g7-r3-isolated-env.yml'
PROVISION = ROOT / 'scripts/g7_r3_provision.py'


def script():
    return PROVISION.read_text(encoding='utf-8')


def workflow():
    return WORKFLOW.read_text(encoding='utf-8')


def embedded_windows_script():
    match = re.search(r'(?ms)^        run: \|\n(.*?)^      - name: Preserve provenance', workflow())
    assert match is not None
    return textwrap.dedent(match.group(1))


def test_one_shot_owner_only_runner_gates():
    wf = workflow()
    assert re.search(r'(?m)^on:\s*\n  push:\s*\n    branches: \[main\]\s*
    for disallowed in ('  workflow_dispatch:', '  pull_request:', '  schedule:',
                       '  repository_dispatch:', '  issue_comment:'):
        assert disallowed not in wf
    for necessary in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        "github.actor == 'spoonman136668-ai'",
        "github.run_number == 1", "github.run_attempt == 1",
        'permissions:\n  contents: read', 'cancel-in-progress: false',
        'runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]',
        'timeout-minutes: 55', 'persist-credentials: false',
        'ref: ${{ github.sha }}', 'if: always()',
    ):
        assert necessary in wf
    ps = embedded_windows_script()
    for necessary in ('TTT-HYBRID-GPU-WIN', 'G7_R3_SHA_MISMATCH',
                      'G7_R3_DIRTY_CHECKOUT', 'wsl.exe --distribution Ubuntu-24.04',
                      'scripts/g7_r3_provision.py', 'g7_r3_env.json'):
        assert necessary in ps
    assert 'forge-codex-isolated' not in wf.lower()


def test_python_syntax_and_static_execution_scope():
    contents = script()
    ast.parse(contents, filename=str(PROVISION))
    for required in ('UV_SHA256 =', 'TORCH_VERSION = "2.9.0"',
                     'TRITON_VERSION = "3.5.0"',
                     'https://download.pytorch.org/whl/cu128',
                     'git", "-C", str(dest), "hash-object"',
                     '"cuda_kernels": "NOT_RUN"',
                     '"training": "NOT_RUN"',
                     '"native_extensions": "NOT_BUILT"',
                     '"system_packages": "NOT_INSTALLED"',
                     'if os.geteuid() == 0:',
                     'if free_bytes < MIN_FREE_GIB * 1024**3:',
                     '"--no-project"'):
        assert required in contents
    for forbidden in ('import torch', 'import triton', 'sudo ', 'apt install',
                      'pip install --system', 'torch.cuda', 'nvidia-smi',
                      'run_gate3', 'MAMBA_KEEP_CUDA_BUILD=TRUE',
                      'git push', 'os.system(', 'shell=True'):
        assert forbidden not in contents


def test_powershell_parser_on_windows_no_runner_invocation():
    if platform.system() != 'Windows':
        return
    with tempfile.TemporaryDirectory(prefix='g7-r3-parser-') as tmp:
        f = Path(tmp) / 'g7-r3.ps1'
        f.write_text(embedded_windows_script(), encoding='utf-8')
        cmd = (
            '$t=$null;$e=$null;'
            '[System.Management.Automation.Language.Parser]::ParseFile('
            "'" + str(f).replace("'", "''") + "',[ref]$t,[ref]$e) | Out-Null;"
            'if($e.Count -gt 0){$e | ForEach-Object{Write-Error $_};exit 1}'
        )
        r = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr[-1200:]
, wf)
    assert 'paths:\n      - ".github/workflows/g7-r3-isolated-env.yml"' in wf
    for disallowed in ('  push:', '  pull_request:', '  schedule:',
                       '  repository_dispatch:', '  issue_comment:'):
        assert disallowed not in wf
    for necessary in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        "github.actor == 'spoonman136668-ai'",
        "github.run_number == 1", "github.run_attempt == 1",
        'permissions:\n  contents: read', 'cancel-in-progress: false',
        'runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]',
        'timeout-minutes: 55', 'persist-credentials: false',
        'ref: ${{ github.sha }}', 'if: always()',
    ):
        assert necessary in wf
    ps = embedded_windows_script()
    for necessary in ('TTT-HYBRID-GPU-WIN', 'G7_R3_SHA_MISMATCH',
                      'G7_R3_DIRTY_CHECKOUT', 'wsl.exe --distribution Ubuntu-24.04',
                      'scripts/g7_r3_provision.py', 'g7_r3_env.json'):
        assert necessary in ps
    assert 'forge-codex-isolated' not in wf.lower()


def test_python_syntax_and_static_execution_scope():
    contents = script()
    ast.parse(contents, filename=str(PROVISION))
    for required in ('UV_SHA256 =', 'TORCH_VERSION = "2.9.0"',
                     'TRITON_VERSION = "3.5.0"',
                     'https://download.pytorch.org/whl/cu128',
                     'git", "-C", str(dest), "hash-object"',
                     '"cuda_kernels": "NOT_RUN"',
                     '"training": "NOT_RUN"',
                     '"native_extensions": "NOT_BUILT"',
                     '"system_packages": "NOT_INSTALLED"',
                     'if os.geteuid() == 0:',
                     'if free_bytes < MIN_FREE_GIB * 1024**3:',
                     '"--no-project"'):
        assert required in contents
    for forbidden in ('import torch', 'import triton', 'sudo ', 'apt install',
                      'pip install --system', 'torch.cuda', 'nvidia-smi',
                      'run_gate3', 'MAMBA_KEEP_CUDA_BUILD=TRUE',
                      'git push', 'os.system(', 'shell=True'):
        assert forbidden not in contents


def test_powershell_parser_on_windows_no_runner_invocation():
    if platform.system() != 'Windows':
        return
    with tempfile.TemporaryDirectory(prefix='g7-r3-parser-') as tmp:
        f = Path(tmp) / 'g7-r3.ps1'
        f.write_text(embedded_windows_script(), encoding='utf-8')
        cmd = (
            '$t=$null;$e=$null;'
            '[System.Management.Automation.Language.Parser]::ParseFile('
            "'" + str(f).replace("'", "''") + "',[ref]$t,[ref]$e) | Out-Null;"
            'if($e.Count -gt 0){$e | ForEach-Object{Write-Error $_};exit 1}'
        )
        r = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],
                           capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr[-1200:]
