"""F8-R21 source-neutral CPU attribution and custody."""
import inspect

import pytest
import torch

from experiments import p85mr_f8r21_repro_trace as x


def test_frozen_scope_and_owner_firstuse_denial():
    f=x.locked()
    assert f["stage"]=="PROSPECTIVE_FROZEN_BEFORE_TRACE_CODE_AND_RESULTS"
    assert f["classification"]=="ENGINEERING_DIAGNOSTIC_NOT_SCIENTIFIC_EXPERIMENT"
    assert f["fixed_protocol"]["update_steps"]==8
    assert f["authority"]["external_scientific_firstuse"] is False
    with pytest.raises(PermissionError,match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_source()
    for arm,seed,mode in (("external",985201,"original_seeded"),
                          ("native_return_focus",985202,"original_seeded"),
                          ("native_return_focus",985201,"unknown")):
        with pytest.raises(ValueError,match="UNREGISTERED_ARM_SEED_MODE"):
            x.model_for(arm,seed,mode)


def test_exact_dyadic_initialization_and_replay():
    x.locked()
    a=x.model_for("native_return_focus",985201,"canonical_dyadic")
    b=x.model_for("native_return_focus",985201,"canonical_dyadic")
    assert x.full_weights(a)==x.full_weights(b)
    c=x.model_for("linear_return_only",985201,"canonical_dyadic")
    d=x.model_for("linear_return_only",985201,"canonical_dyadic")
    assert x.full_weights(c)==x.full_weights(d)
    assert x.full_weights(a)["all"] != x.full_weights(c)["all"]


def test_registered_short_full_original_gradient_audit():
    result=x.run()
    assert result["classification"]=="ENGINEERING_DIAGNOSTIC_NOT_SCIENTIFIC_EXPERIMENT"
    assert result["models_traced"]==8
    assert result["total_short_optimizer_updates"]==128
    assert result["all_same_host_replica_parity"] is True
    assert result["original_external_effects"]==0
    assert result["spent_F7_replay"] is False
    assert result["independent_scientific_pass"] is False
    assert result["deployment"]=="NONE"
    assert len(result["cases"])==8
    assert len(result["batch_indices_sha256"])==64
    assert result["environment"]["deterministic_algorithms"]
    assert result["environment"]["threads"]==1
    for row in result["cases"]:
        assert row["within_process_replica_parity"]
        assert row["trace"]["changed_parameters"]
        assert [x["update"] for x in row["trace"]["weights"]]==[0,1,8]
        assert len(row["trace"]["first"]["loss_tensor_sha256"])==64
        assert row["trace"]["first"]["preclip_layer_grad_sha256"]
    x.retain_ci_receipt(result)


def test_no_external_runner_source_or_model_promotion():
    s=inspect.getsource(x)
    for pattern in ("import causal_gym","from causal_gym","env.reset(",
                    "env.do(","ctf_do(","workflow_dispatch(",
                    "subprocess.run(","requests.get(","urlopen("):
        assert pattern not in s
