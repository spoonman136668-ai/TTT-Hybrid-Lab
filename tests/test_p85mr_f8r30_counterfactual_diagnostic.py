"""Tests for R30 frozen neural prediction/decision failure diagnostics."""
import inspect
import math

import pytest
import torch
from experiments import p85mr_f8r30_counterfactual_diagnostic as x

torch.set_num_threads(1)


def test_frozen_protocol_and_external_authority_negative():
    frozen = x.locked()
    assert frozen["parent"]["sha"] == "dfb2176f04e661372aa8d250b1fafc154f6f552f"
    assert not any(frozen["authority"].values())
    with pytest.raises(PermissionError, match="EXTERNAL_FIRSTUSE"):
        x.refuse_external_firstuse()
    with pytest.raises(ValueError, match="INVALID_ARM"):
        x.diagnose("unqualified_oracle", None)


def test_factual_only_ridge_and_preaction_effect_mask():
    fit = x.b.fit_cohort()
    assert len(fit) == 3072
    assert set(fit[0]) == {"action", "effect", "physical_target", "direct_target"}
    ridge = x.fit_only_ridge(fit)
    assert ridge.ndim == 1 and bool(torch.isfinite(ridge).all())
    w = x.b.r29.worlds("dev")[0]
    acts, obs = x.b._candidate_inputs(w, "active_information", False)
    assert obs.shape[-1] == 16 and not torch.count_nonzero(obs[:, -1])
    predicted = x.pair_raw("fit_only_ridge", ridge, "dev", "active_information")
    assert len(predicted) == 48 and all(len(p)==2 for p in predicted)


def test_evaluator_only_negative_constant_predictions():
    worlds = x.b.r29.worlds("dev")
    fake = [(0., 0.)] * len(worlds)
    scored = x.score_evaluator_only("native_dynamics", worlds, fake, fake)
    assert scored["constant_action_collapse"]
    assert scored["decision_changes_on_erasure"] == 0
    assert scored["choice_accuracy_vs_oracle"] < 1
    assert scored["action_gap_correlation"] is None
    assert scored["physical_effect_mse"] > 0
    with pytest.raises(ValueError, match="UNBALANCED_SCORER"):
        x.score_evaluator_only("native_dynamics", worlds[:1], fake, fake)


def test_retrain_exact_frozen_budget_and_report_all_diagnostics():
    out = x.run()
    assert out["real_optimizer_updates"] == 768
    assert len(out["neural_models"]) == 6
    assert out["factual_fit_episodes"] == 3072
    assert not out["independent_scientific_pass"]
    assert not out["conventional_independent_convergence_qualified"]
    assert out["original_external_effects"] == 0 and not out["spent_F7_replay"]
    assert out["deployment"] == "NONE"
    for arm in out["neural_models"]:
        assert arm["initial_weight_hash"] != arm["trained_weight_hash"]
        assert arm["optimizer_updates"] == 128
    for arm in out["neural_models"] + [out["fit_only_ridge"]]:
        for panel in x.b.PANELS:
            for policy in x.b.POLICIES:
                cell = arm["panels"][panel][policy]
                assert cell["worlds"] == 48 and len(cell["choices"]) == 48
                assert 0 <= cell["choice_accuracy_vs_oracle"] <= 1
                assert math.isfinite(cell["action_value_mse"])
                assert math.isfinite(cell["mean_regret"])
                if cell["physical_effect_mse"] is not None:
                    assert math.isfinite(cell["physical_effect_mse"])
    x.retain_ci_receipt(out)


def test_training_and_fit_never_touch_oracle_or_prohibited_firstuse():
    for fn in (x._train, x._features, x.fit_only_ridge, x.pair_raw):
        text = inspect.getsource(fn)
        assert "hidden_q_evaluator_only" not in text
        assert "r29.after(" not in text
        assert "actual_regret(" not in text
    src = inspect.getsource(x)
    for forbidden in ("import causal_gym", "from causal_gym", "workflow_dispatch(",
                      "subprocess.run(", "requests.get(", "urlopen("):
        assert forbidden not in src
