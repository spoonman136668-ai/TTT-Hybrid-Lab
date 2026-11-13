"""R29B source-neutral neural factual-information uptake and negatives."""
import inspect
import math

import pytest
import torch

from experiments import p85mr_f8r29b_neural_information as x

torch.set_num_threads(1)


def test_prospectively_frozen_protocol_and_prohibited_external_firstuse():
    f = x.locked()
    assert f["parent"]["head"] == "bc220a22495c2b4c718e3737c616f783e4f1f28a"
    assert f["stage"] == "FROZEN_BEFORE_LINKED_IMPLEMENTATION_TRAINING_AND_RESULTS"
    assert f["train"]["total_updates"] == 768
    assert f["score"]["independent_scientific_pass"] is False
    assert all(v is False for v in f["authority"].values())
    with pytest.raises(PermissionError, match="EXTERNAL_FIRSTUSE_NOT_AUTHORIZED"):
        x.refuse_original_external_firstuse()
    with pytest.raises(ValueError, match="UNREGISTERED_MODEL"):
        x.model_for("native_direct_value", 985203)


def test_new_factual_FIT_factorial_and_early_information_barrier():
    samples = x.fit_cohort()
    assert len(samples) == 3072
    assert all(set(row) == {"action", "effect", "physical_target", "direct_target"}
               for row in samples)
    assert samples[0]["action"].shape == (14, 16)
    assert samples[0]["effect"].shape == (14, 16)
    # Opposite hidden q creates identical pre-fifth actual histories; only
    # the selected, context-matching fifth effect reveals the hidden sign.
    lo = x._episode(3, 0, -1, 1., 0, 0)
    hi = x._episode(3, 0, 1, 1., 0, 0)
    assert torch.equal(lo["action"], hi["action"])
    assert torch.equal(lo["effect"][:4], hi["effect"][:4])
    assert not torch.equal(lo["effect"][4], hi["effect"][4])
    assert torch.equal(lo["effect"][5:13], hi["effect"][5:13])
    # Mismatched fifth probe does not identify q; q remains evaluator-only.
    mismatch0 = x._episode(3, 0, -1, 1., 1, 0)
    mismatch1 = x._episode(3, 0, 1, 1., 1, 0)
    assert torch.equal(mismatch0["effect"][4], mismatch1["effect"][4])
    with pytest.raises(ValueError, match="UNREGISTERED_FACTUAL_FIT"):
        x._episode(64, 0, -1, 1., 0, 0)


def test_heldout_observation_erasure_retains_actions_and_B():
    w = x.r29.worlds("ood")[0]
    a, y = x._candidate_inputs(w, "active_information", False)
    ae, ye = x._candidate_inputs(w, "active_information", True)
    assert a.shape == ae.shape == (2, 22, 16)
    assert y.shape == ye.shape == (2, 22, 16)
    assert torch.equal(a, ae)
    assert torch.equal(y[:, :4], ye[:, :4])
    assert not torch.equal(y[:, 4], ye[:, 4])
    assert torch.count_nonzero(ye[:, 4]) == 0
    assert torch.equal(y[:, 5:], ye[:, 5:])
    assert torch.count_nonzero(y[:, -1]) == 0
    with pytest.raises(ValueError, match="INVALID_SCORE_PANEL_OR_POLICY"):
        x._choices("native_dynamics", x.model_for("native_dynamics", 985201),
                   "fit", "active_information")


def test_native_GRU_SSM_preaction_current_effect_is_masked():
    world = x.r29.worlds("dev")[0]
    action, obs = x._candidate_inputs(world, "active_information", False)
    for arm in ("native_dynamics", "gru_dynamics", "ssm_dynamics"):
        model = x.model_for(arm, 985201).eval()
        with torch.no_grad():
            original = x._predict(arm, model,
                                  {"action": action, "effect": obs})[:, -1, 0]
            contaminated = obs.clone()
            contaminated[:, -1] = 0.8
            changed = x._predict(arm, model,
                                 {"action": action, "effect": contaminated})[:, -1, 0]
        assert torch.allclose(original, changed, atol=1e-7, rtol=0), arm


def test_six_models_actually_update_score_and_cold_restore_without_falsifying_failures():
    result = x.run()
    assert result["neural_model_instances"] == 6
    assert result["real_optimizer_updates"] == 768
    assert result["factual_fit_episodes"] == 3072
    assert result["factual_fit_effects"] == 3072 * 14
    assert len(result["trained_models"]) == len(result["judgments"]) == 6
    assert isinstance(result["native_development_pass_2_of_3"], bool)
    assert 0 <= result["native_development_winning_seeds"] <= 3
    assert result["independent_scientific_pass"] is False
    assert result["conventional_independent_convergence_qualified"] is False
    assert result["matched_capacity_and_compute"] is False
    assert result["original_external_effects"] == 0
    assert result["spent_F7_replay"] is False
    assert result["deployment"] == "NONE"
    for row in result["trained_models"]:
        assert row["optimizer_updates"] == 128
        assert row["parameter_count"] > 0
        assert row["initial_weight_hash"] != row["trained_weight_hash"]
        assert row["cold_actions_equal"]
        for panel in x.PANELS:
            s = row["panels"][panel]
            assert set(s["policies"]) == set(x.POLICIES)
            assert math.isfinite(s["fifth_effect_erasure_penalty"])
            for pol in x.POLICIES:
                cell = s["policies"][pol]
                assert cell["selected_fifth_effects"] == 48
                assert len(cell["choices"]) == 48
                assert len(cell["pergroup_regret"]) == 12
                assert math.isfinite(cell["mean_regret"])
    for panel in x.PANELS:
        assert result["analytic_scripted_ceiling"][panel]["active_information"] <= 1e-10
    x.retain_ci_receipt(result)


def test_no_new_source_or_workflow_or_hidden_oracle_label_pipeline():
    src = inspect.getsource(x)
    for forbidden in ("import causal_gym", "from causal_gym", "env.reset(",
                      "env.do(", "ctf_do(", "workflow_dispatch(",
                      "subprocess.run(", "requests.get(", "urlopen("):
        assert forbidden not in src
    assert "hidden_q_evaluator_only" not in inspect.getsource(x._record)
    assert "hidden_q_evaluator_only" not in inspect.getsource(x._batch)
    assert "hidden_q_evaluator_only" not in inspect.getsource(x._predict)
