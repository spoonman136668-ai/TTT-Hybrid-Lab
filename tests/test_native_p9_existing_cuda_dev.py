"""P9 G1 offline admission/denial tests; no real GPU from pytest."""
import json
from pathlib import Path
import pytest
from scripts.native_p9_existing_cuda_dev import (
    SCHEMA,BASE_SHA,MIN_FREE_GIB,MAX_TORCH_ALLOCATED_MIB,
    MAX_GPU_UTILIZATION_PERCENT,STEPS,BATCH,REPETITIONS,
    AdmissionBlocked,demand
)


def test_bounded_params_and_exact_existing_r3_identity():
    assert MIN_FREE_GIB==8.
    assert MAX_TORCH_ALLOCATED_MIB==256
    assert MAX_GPU_UTILIZATION_PERCENT==20
    assert STEPS==64 and BATCH==4 and REPETITIONS==3
    assert BASE_SHA=="083438e51212c1e41a213001263fdd0d002dd4cc"
    with pytest.raises(AdmissionBlocked,match="DENIED"):
        demand(False,"DENIED")


def test_protocol_no_science_or_new_install():
    root=Path(__file__).resolve().parents[1]
    x=json.loads((root/"experiments/native_p9_existing_cuda_admission_frozen.json").read_text())
    assert x["schema"]=="ttt-hybrid.native-p9.existing-r3-cuda-frozen.v1"
    assert x["source"]["parent_p9_sha"]==BASE_SHA
    assert x["science"]["authorized"] is False
    assert x["science"]["world_seeds"]=={}
    assert x["science"]["first_use_consumed"] is False
    assert x["conditions"]["min_free_gib"]==MIN_FREE_GIB
    assert x["conditions"]["max_torch_allocated_mib"]==MAX_TORCH_ALLOCATED_MIB
    assert x["conditions"]["no_package_install"] is True


def test_gpu_admission_script_is_stdlib_before_child():
    root=Path(__file__).resolve().parents[1]
    script=(root/"scripts/native_p9_existing_cuda_dev.py").read_text()
    assert "from scripts import g7_r5_oneshot as custody" in script
    assert "from scripts.native_p9_parallel_refinement_dev import run" in script
    assert "if args.child:" in script
    assert '"CHILD_R3_PACKAGE_GRAPH_DRIFT"' in script
    assert "MAX_CHILD_WALL_SECONDS=130" in script
    assert "--query-compute-apps=pid" in script
    assert "torch.cuda.set_per_process_memory_fraction" in script
