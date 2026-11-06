"""G7-R6 pure negative controls: no CUDA, host, network, or WSL access."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_r6_oneshot as r6


def fixture():
    return {
        "project_sha": "a" * 40,
        "mode": "AUTHENTIC_FORWARD_ONLY",
        "candidate_identity": {"pass": True},
        "preflight": {"status": "PASS_SOURCE_RUNTIME_ADMISSION_ONLY"},
        "admission": {"pass": True},
        "status": "PASS_NATIVE_CUDA_FORWARD_ONLY",
        "native_gate3_status": "PASS_AUTHENTIC_GPU_FORWARD_ONLY",
        "authentic_integration": "PASS",
        "scientific_qualification": "UNVERIFIED",
        "torch": "2.9.0+cu128",
        "torch_cuda_build": "12.8",
        "gpu": {"capability": [12, 0], "peak_allocated_bytes": 20000000},
        "modules": [{"exact_source_path": True}] * 4,
        "native_forward_checks": {
            "forward_finite": True, "shape": [2, 16, 64],
            "episode_reset_deterministic": True, "prefix_invariance": True,
            "suffix_negative_control": True, "gradient_execution": "NOT_RUN",
            "optimizer_steps": 0, "training": "NOT_RUN",
        },
    }


def test_qualified_forward_acceptance_and_child_failure():
    receipt = fixture()
    assert r6.valid_native_result(receipt, "a" * 40, 0)
    assert not r6.valid_native_result(receipt, "a" * 40, 1)


def test_fail_closed_on_source_mode_coverage_memory_and_leakage():
    for path, value in [
        (("project_sha",), "b" * 40),
        (("mode",), "CUDA_KERNEL_ONLY"),
        (("native_gate3_status",), "BLOCKED"),
        (("gpu", "capability"), [9, 0]),
        (("gpu", "peak_allocated_bytes"), 2 * 1024**3),
        (("native_forward_checks", "episode_reset_deterministic"), False),
        (("native_forward_checks", "prefix_invariance"), False),
        (("native_forward_checks", "suffix_negative_control"), False),
        (("native_forward_checks", "gradient_execution"), "RUN"),
        (("native_forward_checks", "optimizer_steps"), 1),
        (("native_forward_checks", "training"), "RUN"),
        (("scientific_qualification",), "PASS"),
        (("modules",), [{"exact_source_path": True}] * 3),
    ]:
        data = fixture()
        target = data
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        assert not r6.valid_native_result(data, "a" * 40, 0), path


def test_one_time_workflow_owner_main_only_without_global_branch_protection():
    path = ROOT / ".github/workflows/g7-r6-one-shot-forward.yml"
    data = path.read_text(encoding="utf-8")
    assert "on:\n  push:\n    branches: [main]" in data
    assert "github.ref_protected" not in data
    assert "GITHUB_REF_PROTECTED" not in data
    assert "github.actor == 'spoonman136668-ai'" in data
    assert "GITHUB_ACTOR -cne 'spoonman136668-ai'" in data
    assert "github.event_name == 'push'" in data
    assert "github.ref == 'refs/heads/main'" in data
    assert "github.run_number == 1 && github.run_attempt == 1" in data
    assert "ttt-hybrid-gpu-isolated" in data
    assert "TTT-HYBRID-GPU-WIN" in data
    assert "timeout-minutes: 7" in data
    assert "--expected-sha \"$env:GITHUB_SHA\"" in data
    assert "actions/checkout@" in data and "persist-credentials: false" in data
    assert "workflow_dispatch:" not in data
    assert "pull_request:" not in data
    assert "schedule:" not in data


def test_forward_script_reuses_pinned_r3_custody_without_training_or_installation():
    source = (ROOT / "scripts/g7_r6_oneshot.py").read_text(encoding="utf-8")
    assert "forward_only=True" in source
    assert "custody.source_custody(" in source
    assert "custody.R3_MANIFEST_SHA256" in source
    assert "MIN_FREE_GIB = 4.0" in source
    assert "MAX_PEAK_GIB = 1.0" in source
    assert "CHILD_TIMEOUT_SECONDS = 175" in source
    assert "optimizer_steps" in source
    assert "subprocess.run(" in source
    assert "pip install" not in source
    assert "git clone" not in source
    assert "torch.optim" not in source
