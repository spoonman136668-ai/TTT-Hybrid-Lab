"""R29 source-neutral calibrated information-gathering task controls."""
import inspect
import math
import pytest
from experiments import p85mr_f8r29_information_calibration as x


def test_immutable_registered_protocol_and_no_authority():
    f = x.locked()
    assert f["stage"] == "FROZEN_BEFORE_IMPLEMENTATION_OR_NUMERIC_RESULT"
    assert f["parent"]["pr"] == 236
    assert f["parent"]["head"] == "98fead8094f1bdacae418ccc4d54800c561bf827"
    assert all(value is False for value in f["authority"].values())
    assert f["next_neural_gate"]["require_train_convergence"]
    with pytest.raises(PermissionError, match="FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_external_firstuse()


def test_four_factual_probes_cannot_reveal_q_and_selected_fifth_can():
    for panel in x.PANELS:
        for context in (0, 1):
            low = x.world(panel, 2, context, -1)
            high = x.world(panel, 2, context, 1)
            assert low["early_actual"] == high["early_actual"]
            assert low["b_history"] == high["b_history"]
            assert x.after(x.PROBE_STATE, context, context, low["hidden_q_evaluator_only"]) != \\
                   x.after(x.PROBE_STATE, context, context, high["hidden_q_evaluator_only"])
            other = 1-context
            assert x.after(x.PROBE_STATE, other, context, low["hidden_q_evaluator_only"]) == \\
                   x.after(x.PROBE_STATE, other, context, high["hidden_q_evaluator_only"])


def test_all_panels_balanced_and_unregistered_inputs_denied():
    for panel in x.PANELS:
        rows = x.worlds(panel)
        assert len(rows) == 48
        assert all(len([w for w in rows if w["group"] == g]) == 4
                   for g in range(x.GROUPS))
        for g in range(x.GROUPS):
            assert {(w["public_context"], math.copysign(1,w["hidden_q_evaluator_only"]))
                    for w in rows if w["group"]==g} == {(0,-1.0),(0,1.0),(1,-1.0),(1,1.0)}
            assert len({w["b_history"] for w in rows if w["group"]==g}) == 1
    with pytest.raises(ValueError,match="UNREGISTERED_WORLD"):
        x.world("ood", 12, 0, 1)
    with pytest.raises(ValueError,match="INVALID_PUBLIC_PROBE_INPUT"):
        x.probe_action("active_information", "ood", 0, 2)


def test_factual_nonoracle_choice_fails_closed_on_invalid_observation():
    for panel in x.PANELS:
        for c in (0,1):
            for sign in (-1,1):
                w=x.world(panel, 1, c, sign)
                choice=x.physical_plan(w, "active_information")
                assert choice["fifth_action"]==c
                assert choice["factual_early_count"]==4
                assert choice["factual_fifth_count"]==1
                assert choice["B_count"]==x.B_STEPS[panel]
                assert choice["return_regret"]<1e-10
                without=x.physical_plan(w,"active_information",erase=True)
                assert without["selected_return_action"]==1-c
                with pytest.raises(ValueError,match="INCONSISTENT_FACTUAL_OBSERVATION"):
                    x.return_choice(c,panel,c,12345.0)
    with pytest.raises(ValueError,match="INVALID_PHYSICAL_INPUT"):
        x.after(float("nan"),0,0,1.0)


def test_precommitted_paired_information_value_and_erasure():
    result=x.run()
    assert result["analytic_task_calibration_pass"]
    assert result["native_model_training_updates"]==0
    assert result["original_external_effects"]==0
    assert result["independent_scientific_pass"] is False
    for panel in x.PANELS:
        z=result["panels"][panel]
        assert z["world_count"]==48
        assert z["factual_early_effects_each_policy"]==192
        assert z["factual_selected_fifth_effects_each_policy"]==48
        assert z["B_effects_each_policy"]==48*x.B_STEPS[panel]
        assert z["task_calibration_pass"]
        assert z["best_fixed_gain"]>=.08
        assert z["paired_bootstrap_lower_gain"]>.05
        assert z["erased_active_regret"]>.08
    x.retain_ci_receipt(result)


def test_no_source_first_use_gpu_workflows_or_hidden_model_input():
    code=inspect.getsource(x)
    for bad in ("import causal_gym","from causal_gym","env.reset(",
                "env.do(","ctf_do(","workflow_dispatch(","subprocess.run(",
                "requests.get(","urlopen("):
        assert bad not in code
    assert "def return_choice(public_context" in code
