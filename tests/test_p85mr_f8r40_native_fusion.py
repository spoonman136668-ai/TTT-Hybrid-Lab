"""F8-R40: frozen binder Native residual isolation, no science promotion."""
import inspect
import math

import pytest
import torch
from experiments import p85mr_f8r40_native_fusion as x

torch.set_num_threads(1)


def test_frozen_protocol_and_execution_authority():
    f=x.locked()
    assert f["parent"]["sha"]=="7fb94ecc43bf9ee0ab98f2647f35c86e15096386"
    assert f["fusion_arms"]["arms"]==["naive_sum","zero_gated_residual"]
    assert f["frozen_judgment"]["independent_scientific_pass"] is False
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_zero_gate_is_exact_binder_and_same_native_seed_initialization():
    binder=x.r38.TrainableBinding(x.SEEDS[0])
    naive=x.FrozenBinderResidual(x.SEEDS[0],"naive_sum",binder)
    gated=x.FrozenBinderResidual(x.SEEDS[0],"zero_gated_residual",binder)
    assert x._hash(naive.native)==x._hash(gated.native)
    a,e=x.r38.candidate_inputs(x.r38.worlds("dev")[0],"active_information")
    assert a.shape==(2,14,16)
    gated.eval()
    with torch.no_grad():
        before=gated(a,e)
        off=gated(a,e,zero_native=True)
        assert torch.equal(before,off)
        poison=e.clone()
        poison[:,-1,:]=19.
        assert torch.equal(before,gated(a,poison))
    assert x._hash(binder)==x._hash(gated.binder)


def test_new_law_probe_policies_and_return_effect_fence():
    for p in ("oscillatory_even","nonlinear_odd"):
        w=x.r39.worlds(p)[0]
        for policy in x.POLICIES:
            a,e=x._candidate_new(w,policy)
            assert a.shape==(2,14 if p=="oscillatory_even" else 22,16)
            assert e.shape==a.shape
            assert torch.count_nonzero(e[:,-1,:])==0
        erased=x._candidate_new(w,"active_information",erase=True)
        assert torch.count_nonzero(erased[1][:,4,:])==0


def test_full_native_residual_training_and_frozen_cold_restore():
    output=x.run()
    assert output["pretrain_replay_seed_count"]==2
    assert output["pretrain_optimizer_updates"]==1024
    assert output["native_arms_trained"]==4
    assert output["native_optimizer_updates"]==1024
    assert output["external_firstuse"] is False
    assert output["spent_F7_replay"] is False
    assert output["deployment"]=="NONE"
    assert not output["native_improvement_proven"]
    assert not output["independent_scientific_pass"]
    assert isinstance(output["judgment"]["source_neutral_nonregression_development_pass"],bool)
    for rec in output["models"]:
        assert rec["optimizer_updates"]==256
        assert rec["frozen_binder_unchanged"]
        assert rec["initial_native_sha256"]!=rec["final_native_sha256"]
        assert rec["cold_restore_equal"]
        assert [x["updates"] for x in rec["checkpoints"]]==[128,256]
        for p in x.PANELS:
            cell=rec["panels"][p]
            assert set(cell["policies"])==set(x.POLICIES)
            assert len(cell["erased"]["choices"])==64
            assert len(cell["zero_native"]["choices"])==64
            assert len(cell["zero_backbone"]["choices"])==64
            for pol in x.POLICIES:
                assert len(cell["policies"][pol]["choices"])==64
                assert math.isfinite(cell["policies"][pol]["mean_regret"])
    x.retain_ci_receipt(output)


def test_no_evaluator_return_outcome_or_generator_hidden_q_supervision():
    for fn in (x.FrozenBinderResidual.forward,x.train_arm,x._validation):
        raw=inspect.getsource(fn)
        for forbidden in ("q_evaluator_only","regret_evaluator_only(",
                          "r39._generator_law(","r38.physical("):
            assert forbidden not in raw
    full=inspect.getsource(x)
    for forbidden in ("workflow_dispatch(","subprocess.run(","requests.get(",
                      "urlopen(","from causal_gym","import causal_gym"):
        assert forbidden not in full
