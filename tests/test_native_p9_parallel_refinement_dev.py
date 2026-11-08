"""P9 regression for actual skipped compute and uninterrupted state."""
import json
from pathlib import Path
import subprocess
import sys
import pytest
import torch
from scripts.native_p9_parallel_refinement_dev import (
    SEEDS,WINDOW,ContinuousDualStream,trace,exercise,run
)


def test_gate_uses_only_present_observation_and_random_has_same_budget():
    for seed in SEEDS:
        s,a,dt,y,gate,rnd=trace(seed,32,4)
        assert s.shape==(32,4,2) and y.shape==(32,4,2)
        assert sum(gate)==sum(rnd)==16
        assert torch.allclose(y-s,a*torch.tensor([[[.18,-.18]]]))
        assert all(bool(x.var(-1,unbiased=False).mean()>.1)==v
                   for x,v in zip(s,gate))
    with pytest.raises(ValueError,match="TRACE_BUDGET"):
        trace(7207,32,4)


def test_bypass_does_not_invoke_refiner_but_updates_state():
    torch.manual_seed(613)
    m=ContinuousDualStream().eval()
    s=torch.tensor([[.85,-.85]])
    a=torch.ones(1,1)
    dt=torch.full((1,1),.1)
    before=m.initial(s)
    called=[]
    hook=m.refine.register_forward_hook(lambda *_:called.append(1))
    try:
        with torch.no_grad():
            p0,n0=m.step(s,a,dt,before,mode="parallel",active=False)
            p1,n1=m.step(s,a,dt,before,mode="parallel",active=True)
        assert called==[1] and not torch.equal(p0,p1)
        assert torch.equal(n0.h,n1.h)
        assert torch.equal(n0.cache,n1.cache)
        assert n0.events==n1.events==1
    finally:
        hook.remove()


def test_full_scan_identical_with_all_refinement_skipped():
    torch.manual_seed(613)
    m=ContinuousDualStream().eval()
    t=trace(SEEDS[0],16,2)
    with torch.no_grad():
        a,sa,_=exercise(m,t,"parallel",[True]*16)
        b,sb,_=exercise(m,t,"parallel",[False]*16)
        c,sc,_=exercise(m,t,"sequential",[True]*16)
    assert sa.events==sb.events==sc.events==16
    assert sa.cache.shape==(2,WINDOW,8)
    assert torch.equal(sa.h,sb.h) and torch.equal(sa.h,sc.h)
    assert torch.equal(sa.cache,sb.cache) and torch.equal(sa.cache,sc.cache)
    assert not torch.equal(a,b)


def test_invalid_elapsed_and_scientific_execution_denied(monkeypatch):
    m=ContinuousDualStream()
    sensor=torch.zeros(1,2)
    with pytest.raises(ValueError,match="INVALID_INPUT_OR_STATE"):
        m.step(sensor,torch.ones(1,1),-torch.ones(1,1),m.initial(sensor),
               mode="parallel",active=True)
    monkeypatch.setenv("NATIVE_P9_DEV_ONLY","0")
    with pytest.raises(RuntimeError,match="SCIENTIFIC_FIRST_USE_FORBIDDEN"):
        run()


def test_actual_dev_control_counts_do_not_require_performance_success():
    r=run("cpu",steps=24,batch=2,repetitions=1)
    assert r["status"]=="PASS_DEV_INTEGRITY_NOT_SCIENCE"
    assert r["scientific_first_use"]=="NOT_RUN"
    assert r["gpu_benchmark_verified"] is False
    assert r["fused_Mamba2_exercised"] is False
    assert r["scientific_seeds_consumed"]==[]
    assert len(r["records"])==3
    for q in r["records"]:
        assert all(q["checks"].values())
        assert q["measurements"]["parallel_gate"]["refine_calls"]==12
        assert q["measurements"]["parallel_random"]["refine_calls"]==12
        assert q["measurements"]["parallel_gate"]["avoided_refine_linear_MACs_estimate"]>0
    assert all(0<=v<=3 for v in r["hypothesis_support_of_three"].values())


def test_actual_isolated_cli_and_frozen_scope():
    root=Path(__file__).resolve().parents[1]
    p=subprocess.run([sys.executable,"-I","-B",
                      str(root/"scripts/native_p9_parallel_refinement_dev.py"),
                      "--help"],cwd=root,text=True,capture_output=True,timeout=20)
    assert p.returncode==0,p.stderr[-1000:]
    frozen=json.loads((root/"experiments/native_p9_parallel_refinement_dev_frozen.json").read_text())
    assert frozen["schema"]=="ttt-hybrid.native-p9.parallel-refinement-frozen.v1"
    assert frozen["source"]["parent_p8_sha"]=="a0d749d3ec28cbc15ff212a6fdd9752799cc8578"
    assert frozen["science"]["authorized"] is False
    assert frozen["science"]["seeds"]=={}
