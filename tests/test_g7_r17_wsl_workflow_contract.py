"""Source-only controls: manually gated protected-main one-shot, never executed."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
WF = ROOT / '.github/workflows/g7-r17-one-shot-wsl-link-diagnostic.yml'


def source():
    return WF.read_text(encoding='utf-8')


def test_only_manual_dispatch_no_implicit_or_scheduled_run():
    src = source()
    assert re.search(r'(?m)^on:\n  workflow_dispatch:\s*\npermissions:', src)
    for pattern in (r'(?m)^  schedule:', r'(?m)^  push:', r'(?m)^  pull_request:'):
        assert not re.search(pattern, src)
    assert 'cancel-in-progress: false' in src


def test_owner_protected_main_and_no_retry_or_replay():
    src = source()
    for required in (
        "github.repository == 'spoonman136668-ai/TTT-Hybrid-Lab'",
        "github.ref == 'refs/heads/main'",
        'github.ref_protected == true',
        "github.event_name == 'workflow_dispatch'",
        "github.actor == 'spoonman136668-ai'",
        'github.run_number == 1',
        'github.run_attempt == 1',
        "$env:GITHUB_REF_PROTECTED -cne 'true'",
        "$env:RUNNER_NAME -cne 'TTT-HYBRID-GPU-WIN'",
        "$env:RUNNER_OS -cne 'Windows'",
        "$env:RUNNER_ARCH -cne 'X64'",
        '$head -cne $env:GITHUB_SHA',
        'G7_R17_DIRTY_SOURCE_REFUSED',
    ):
        assert required in src


def test_existing_hybrid_runner_only_and_low_privilege_checkout():
    src = source()
    for required in (
        'runs-on: [self-hosted, Windows, X64, ttt-hybrid-gpu-isolated]',
        'timeout-minutes: 6',
        'permissions:\n  contents: read',
        'actions/checkout@11d5960a326750d5838078e36cf38b85af677262',
        'ref: ${{ github.sha }}',
        'persist-credentials: false',
        'clean: true',
    ):
        assert required in src


def test_fixed_wsl_ubuntu_2404_invocation_diagnostic_only():
    src = source()
    assert src.count('wsl.exe --distribution Ubuntu-24.04') == 2
    for required in (
        'g7_r17_link_diagnostic.py',
        '/usr/bin/python3 -I -B',
        '--expected-sha "$env:GITHUB_SHA"',
        '--kill-after=10s 80s',
    ):
        assert required in src
    # Reject active privileges / CUDA / original one-shot replay in runnable
    # PowerShell, not benign mention of the words in comments.
    runnable = '\n'.join(x for x in src.splitlines() if not x.lstrip().startswith('#'))
    for forbidden in ('--user root', 'apt-get', 'sudo', 'pip install', 'torch.cuda',
                      'g7_r15_compiler_repair_forward.py', 'g7_cuda_qualification.py',
                      'git checkout', 'FORGE-F139-UnifiedController'):
        assert forbidden not in runnable


def test_export_only_allowlisted_sanitized_receipt():
    src = source()
    assert src.count('uses: actions/upload-artifact@') == 1
    assert 'if: always()' in src
    assert 'path: local_runs/g7_r17_link_diagnostic.json' in src
    assert 'retention-days: 30' in src
    assert 'if-no-files-found: warn' in src
