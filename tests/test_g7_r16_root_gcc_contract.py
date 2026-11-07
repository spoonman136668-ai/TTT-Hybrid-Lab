"""G7-R16: source-only negative controls for root GCC + one native forward."""
from pathlib import Path
import ast

ROOT=Path(__file__).resolve().parents[1]
WF=ROOT/".github/workflows/g7-r16-root-gcc-forward.yml"
PROBE=ROOT/"scripts/g7_r16_compiler_probe.py"

def test_one_use_only_and_exact_original_runner():
    s=WF.read_text(encoding="utf-8")
    for token in (
        "on:\n  push:\n    branches: [main]",
        'paths:\n      - ".github/workflows/g7-r16-root-gcc-forward.yml"',
        "github.run_number == 1 && github.run_attempt == 1",
        "github.actor == 'spoonman136668-ai'",
        "TTT-HYBRID-GPU-WIN",
        "ttt-hybrid-gpu-isolated",
        "persist-credentials: false",
        "clean: true",
        "G7_R16_CANDIDATE_SOURCE_DRIFT",
        "timeout-minutes: 14",
        "retention-days: 30",
    ): assert token in s, token
    for token in ("workflow_dispatch:", "schedule:", "pull_request:",
                  "wsl.exe --install", "gh workflow run", "Start-Process",
                  "git push", "actions/setup-python"):
        assert token not in s, token

def test_root_only_explicit_gcc_apt_install():
    s=WF.read_text(encoding="utf-8")
    assert s.count("--user root --exec /usr/bin/id -u")==1
    assert s.count("--user root --exec /usr/bin/timeout")==2
    assert s.count("/usr/bin/apt-get -o Acquire::Retries=0 update")==1
    assert s.count("/usr/bin/apt-get install -y --no-install-recommends gcc")==1
    for banned in (
        "apt-get upgrade", "apt-get dist-upgrade", "apt-get autoremove",
        "apt-get install -y build-essential", "pip install", "uv pip",
        "torch.optim", "nvidia-smi", "nvcc", "wsl.exe --set-default",
        "CUDA_VISIBLE_DEVICES=", "CC=", "workflow_dispatch",
    ):
        assert banned not in s, banned

def test_original_user_forward_and_negative_stops():
    s=WF.read_text(encoding="utf-8")
    for token in (
        "BLOCKED_UBUNTU_ROOT_CONTEXT_UNAVAILABLE",
        "BLOCKED_GCC_APT_UPDATE_FAILED", "BLOCKED_GCC_APT_INSTALL_FAILED",
        "BLOCKED_COMPILER_NOT_VISIBLE_TO_UNPRIVILEGED_USER",
        "BLOCKED_NATIVE_FORWARD_OR_HOST_COMPILER_PROBE",
        "G7_R16_COMPILER=PRESENT", "G7_R16_COMPILER=ABSENT",
        "g7_r15_compiler_repair_forward.py",
        "--expected-sha", "g7_r15_compiler_forward.json",
        "g7_r16_root_gcc_forward.json",
        "cuda_runtime_change = 'NONE'",
        "training = 'NOT_RUN'", "scientific = 'UNVERIFIED'",
    ): assert token in s, token
    assert s.count("g7_r15_compiler_repair_forward.py")==1  # only one call site
    assert s.count("wsl.exe --distribution Ubuntu-24.04 --user root") == 3
    assert "g7_r6_oneshot.py" in (ROOT/"scripts/g7_r15_compiler_repair_forward.py").read_text(encoding="utf-8")

def test_readonly_compiler_probe_no_root_or_cuda():
    s=PROBE.read_text(encoding="utf-8")
    ast.parse(s)
    assert "native.source_custody(args.expected_sha)" in s
    assert "native.selected_compiler()" in s
    for expected in ("G7_R16_COMPILER=PRESENT", "G7_R16_COMPILER=ABSENT"):
        assert expected in s
    for banned in ("apt-get", "sudo", "torch.cuda", "wsl.exe", "git checkout"):
        assert banned not in s
