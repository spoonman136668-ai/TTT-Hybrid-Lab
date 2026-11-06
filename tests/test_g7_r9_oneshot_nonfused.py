"""G7-R9 admission contract: pure CPU negative controls and owner-only workflow."""

from pathlib import Path
import sys
import inspect

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import g7_r6_oneshot as custody


def record():
    return {
        "project_sha": "a"*40, "mode": "AUTHENTIC_FORWARD_ONLY",
        "candidate_identity": {"pass": True},
        "preflight": {"status": "PASS_SOURCE_RUNTIME_ADMISSION_ONLY"},
        "admission": {"pass": True},
        "status": "PASS_NATIVE_CUDA_FORWARD_ONLY",
        "native_gate3_status": "PASS_AUTHENTIC_GPU_FORWARD_ONLY",
        "authentic_integration": "PASS",
        "native_mamba_execution_path": "PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED",
        "scientific_qualification": "UNVERIFIED",
        "torch": "2.9.0+cu128", "torch_cuda_build": "12.8",
        "gpu": {"capability": [12, 0], "peak_allocated_bytes": 20000000},
        "modules": [{"exact_source_path": True} for _ in range(4)],
        "native_forward_checks": {
            "forward_finite": True, "shape": [2, 16, 64],
            "episode_reset_deterministic": True, "prefix_invariance": True,
            "suffix_negative_control": True,
            "gradient_execution": "NOT_RUN", "optimizer_steps": 0,
            "training": "NOT_RUN",
        },
    }


def test_nonfused_exact_source_native_acceptance_requires_real_route_and_controls():
    value = record()
    assert custody.valid_native_result(value, "a"*40, 0, require_nonfused=True)
    assert not custody.valid_native_result(value, "a"*40, 1, require_nonfused=True)
    for path, wrong in [
        (("native_mamba_execution_path",), "PINNED_OFFICIAL_FUSED_ENGINEERING_UNVERIFIED"),
        (("native_mamba_execution_path",), None),
        (("status",), "FAIL_NATIVE_GATE3_INTEGRATION"),
        (("native_gate3_status",), "FAIL_AUTHENTIC_GPU_INTEGRATION"),
        (("authentic_integration",), "NOT_RUN"),
        (("project_sha",), "b"*40),
        (("gpu", "peak_allocated_bytes"), 2*1024**3),
        (("native_forward_checks", "forward_finite"), False),
        (("native_forward_checks", "prefix_invariance"), False),
        (("native_forward_checks", "suffix_negative_control"), False),
        (("native_forward_checks", "episode_reset_deterministic"), False),
        (("native_forward_checks", "optimizer_steps"), 1),
        (("native_forward_checks", "training"), "RUN"),
    ]:
        mutated = record()
        sub=mutated
        for part in path[:-1]:
            sub=sub[part]
        sub[path[-1]]=wrong
        assert not custody.valid_native_result(mutated,"a"*40,0,require_nonfused=True), path


def test_consumed_g7_r9_gpu_workflow_retired_after_single_execution():
    # The original run and hash-bound sanitized failure are preserved in docs.
    assert not (ROOT/".github/workflows/g7-r9-one-shot-nonfused.yml").exists()
    report=(ROOT/"docs/G7_R9_RESULT.md").read_text(encoding="utf-8")
    for marker in (
        "42300670454",
        "cf207e3d2d3e02b2803548a0ebc74e3bd6511744",
        "13543572395",
        "c08de830e8d2cbd672bdbdf805ed8bd86c23865a6478b1b5c9f6411c4b9e533b",
        "TRITON_OR_KERNEL_COMPATIBILITY",
        "PINNED_OFFICIAL_NONFUSED_FORWARD_UNVERIFIED",
        "NO AUTOMATIC RETRY",
    ):
        assert marker in report, marker


def test_existing_exact_source_custody_reused_with_budgets_and_failure_categories():
    w=(ROOT/"scripts/g7_r6_oneshot.py").read_text()
    n=(ROOT/"scripts/g7_cuda_qualification.py").read_text()
    g3=(ROOT/"run_gate3.py").read_text()
    assert "MIN_FREE_GIB = 4.0" in w
    assert "MAX_PEAK_GIB = 1.0" in w
    assert "CHILD_TIMEOUT_SECONDS = 175" in w
    assert "custody.source_custody(" in w
    assert "custody.R3_MANIFEST_SHA256" in w
    assert "require_nonfused=args.require_nonfused" in w
    assert "native_gate3_error_category" in n
    assert "native_gate3_receipt_sha256" in n
    assert "mamba_mem_eff_path=not forward_only" in g3
    assert "pip install" not in w
    assert "torch.optim" not in w
