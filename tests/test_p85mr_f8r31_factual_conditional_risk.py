"""F8-R31 source-neutral factual-only conditional-risk development controls."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r31_factual_conditional_risk as x


def test_frozen_preimplementation_protocol_and_authority():
    f=x.locked()
    assert f["precommitted_development_thresholds"]["independent_scientific_pass"] is False
    assert not any(f["authority"].values())
    assert f["design"]["fit_groups_training"] == "group 0..47"
    with pytest.raises(PermissionError,match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()


def test_factual_only_target_and_uncertainty_gates():
    fit=x.b.fit_cohort()
    row=fit[0]
    assert len(fit)==3072 and set(row)=={
        "action","effect","physical_target","direct_target"}
    inputs,bucket=x._feature_from_factual(row["action"],row["effect"])
    assert len(inputs)==5 and bucket in x.BUCKETS
    w=x.b.r29.worlds("dev")[0]
    a,e=x.b._candidate_inputs(w,"active_information",False)
    aa,ee=x.b._candidate_inputs(w,"active_information",True)
    assert torch.equal(a,aa)
    assert torch.count_nonzero(ee[:,4]) == 0
    assert torch.equal(e[:,:4],ee[:,:4])
    assert torch.equal(e[:,5:],ee[:,5:])
    # Erasure produces an explicitly uncertain matching-return candidate.
    actual=[x._feature_from_factual(a[i],e[i])[1] for i in (0,1)]
    missing=[x._feature_from_factual(aa[i],ee[i])[1] for i in (0,1)]
    assert "informed" in actual and "uncertain" in missing
    with pytest.raises(ValueError,match="INVALID_FACTUAL_HISTORY"):
        x._feature_from_factual(a[0,:3],e[0,:3])


def test_train_only_on_selected_factual_FIT_and_closed_form_convergence():
    fit=x.b.fit_cohort()
    models=[x.fit_factual_only(fit,g) for g in x.TRAIN_GROUPS]
    assert [m["examples"] for m in models] == [576,1152,2304]
    assert all(m["normal_equation_gradient_max"] < 1e-6 for m in models)
    assert all(m["conditional_variance"]["uncertain"] >
               m["conditional_variance"]["safe"] for m in models)
    assert all(math.isfinite(x._validation(fit,m)) for m in models)
    with pytest.raises(ValueError,match="UNREGISTERED_FIT"):
        x.fit_factual_only(fit,64)


def test_full_frozen_dev_and_required_fail_closed_receipt():
    result=x.run()
    assert result["neural_optimizer_updates"] == 0
    assert result["same_factual_R29B_episodes"] == 3072
    assert result["cold_restore_actions_equal"]
    assert not result["independent_scientific_pass"]
    assert not result["independent_neural_baseline_convergence_qualified"]
    assert not result["matched_neural_FLOPs"]
    assert result["original_external_effects"] == 0
    assert result["deployment"] == "NONE"
    assert set(result["panels"]) == set(x.b.PANELS)
    assert len(result["factual_fit_curve"]) == 3
    assert isinstance(result["judgment"]["source_neutral_development_pass"],bool)
    for panel in x.b.PANELS:
        s=result["panels"][panel]
        assert set(s["risk_aware"]) == set(x.b.POLICIES)
        for variant in ("risk_aware","mean_only"):
            for policy in x.b.POLICIES:
                t=s[variant][policy]
                assert len(t["choices"]) == 48
                assert len(t["pergroup_regret"]) == 12
                assert math.isfinite(t["mean_regret"])
    x.retain_ci_receipt(result)


def test_no_counterfactual_or_hidden_labels_to_fitter_or_reader():
    for fn in (x._feature_from_factual,x._fit_row,x.fit_factual_only,
               x._estimate,x._choose):
        src=inspect.getsource(fn)
        for forbidden in ("hidden_q_evaluator_only","r29.after(",
                          "actual_regret(","r29.return_choice("):
            assert forbidden not in src
    src=inspect.getsource(x)
    for forbidden in ("import causal_gym","from causal_gym",
                      "workflow_dispatch(","requests.get(",
                      "subprocess.run(","urlopen("):
        assert forbidden not in src
