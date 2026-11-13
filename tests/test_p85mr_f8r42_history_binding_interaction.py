"""F8-R42 prospective source-neutral falsifier: controls and genuine training."""
import pytest
import torch

from experiments import p85mr_f8r42_history_binding_interaction as r42


def test_preregistration_is_exact_source_bound_and_no_external_firstuse():
    frozen = r42.locked()
    assert frozen["parent"]["sha"] == r42.PARENT
    assert frozen["prospective_score"]["independent_scientific_pass"] is False
    assert len(r42.ARMS) == 6
    assert set(r42.PANELS) == {"dev", "ood"}
    with pytest.raises(PermissionError, match="FIRSTUSE_DENIED"):
        r42.refuse_external_firstuse("synthetic-does-not-authorize-external")


def test_new_generator_hides_gate_from_early_fifth_and_fails_closed():
    left = r42.world("fit", 0, 1, 1, 1)
    right = r42.world("fit", 0, 1, 1, -1)
    a, e = r42.factual_history(left)
    aa, ee = r42.factual_history(right)
    assert torch.equal(a, aa)
    assert torch.equal(e[:5], ee[:5])
    m = 5+r42.marker("fit",0)
    assert not torch.equal(e[m],ee[m])
    assert torch.equal(torch.cat((e[:m],e[m+1:])),
                       torch.cat((ee[:m],ee[m+1:])))
    with pytest.raises(ValueError, match="UNREGISTERED_WORLD"):
        r42.world("test",0,0,1,1)
    with pytest.raises(ValueError, match="INVALID_PANEL"):
        r42.worlds("external")
    with pytest.raises(ValueError, match="INVALID_MODEL"):
        r42.PairGate("unregistered",985201,None)


def test_factual_only_labels_and_masked_candidate_return():
    fit, labels = r42.factual_fit()
    assert len(fit) == 768 and len(labels) == 384
    assert set(labels) == {0,1}
    assert labels.count(0) == labels.count(1) == 192
    aa,ee = r42.candidates(r42.world("dev",0,0,1,1))
    assert aa.shape == ee.shape == (2,14,16)
    assert torch.equal(ee[:,-1,:],torch.zeros_like(ee[:,-1,:]))
    swapped = r42.candidates(r42.world("dev",0,0,1,-1))
    assert torch.equal(ee[:,:5,:],swapped[1][:,:5,:])


def test_frozen_factual_native_and_strong_controls_no_science_promotion():
    receipt = r42.run()
    assert receipt["schema"] == r42.SCHEMA
    assert receipt["freeze_blob"] == r42.FREEZE_BLOB
    assert receipt["parent"] == r42.PARENT
    assert receipt["original_binder_replay_updates"] == 1024
    assert receipt["new_neural_updates"] == 2304
    assert receipt["factual_fit_episodes"] == 768
    assert receipt["factual_train_pairs"] == 256
    assert receipt["factual_heldout_pairs"] == 128
    assert len(receipt["models"]) == len(r42.ARMS)
    assert receipt["independent_scientific_pass"] is False
    assert receipt["synthetic_development_only"] is True
    assert receipt["spent_external_firstuse"] is False
    assert receipt["GPU"] is False
    assert receipt["deployment"] == "NONE"
    assert receipt["judgment"]["independent_scientific_pass"] is False
    assert len(receipt["judgment"]["seeds"]) == 2
    for row in receipt["models"]:
        assert row["optimizer_updates"] == r42.STEPS
        assert row["cold_restore"] is True
        assert row["history_initial_hash"] != row["history_trained_hash"]
        assert len(row["checkpoints"]) == 2
        assert set(row["panels"]) == {"dev","ood"}
        for p in r42.PANELS:
            assert set(row["panels"][p]) == {
                "active","erase_B","swap_B","erase_fifth",
                "zero_backbone","zero_gate","zero_binder"}
            assert len(row["panels"][p]["active"]["choices"]) == 128
    for p in r42.PANELS:
        assert len(receipt["positive_diagnostic"][p]["choices"]) == 128
    r42.retain_ci_receipt(receipt)



def test_positive_control_uses_only_observed_B_history_not_oracle_marker(monkeypatch):
    fit, _ = r42.factual_fit()
    for panel in r42.PANELS:
        panel_worlds = r42.worlds(panel)
        observed = {id(w): r42.candidates(w) for w in panel_worlds}
        for w in panel_worlds:
            measured = r42._observed_b_peak(observed[id(w)][1][0])
            expected = w["b_history"][r42.marker(panel, w["group"])][1]
            assert measured == pytest.approx(expected, abs=2e-6)
        with monkeypatch.context() as patch:
            patch.setattr(r42, "worlds", lambda requested, panel=panel, panel_worlds=panel_worlds:
                          panel_worlds if requested == panel else
                          pytest.fail("unexpected diagnostic panel"))
            patch.setattr(r42, "candidates", lambda w, observed=observed: observed[id(w)])
            patch.setattr(r42, "marker", lambda *_args, **_kwargs:
                          pytest.fail("held-out oracle marker lookup"))
            diagnostic = r42.positive_control(fit, panel)
        # This diagnostic learned no alternate goal target. It must not be
        # relabeled a zero-regret positive under the corrected factual objective.
        factual = [r42._regret(w, c) for w, c in
                   zip(panel_worlds, diagnostic["choices"])]
        assert diagnostic["mean_regret"] == pytest.approx(
            sum(factual)/len(factual), abs=1e-12)
        assert diagnostic["mean_regret"] > 0.0
        assert diagnostic["history_gate_pair_flips"] == 64
    with pytest.raises(ValueError, match="INVALID_OBSERVED_B_HISTORY"):
        r42._observed_b_peak(torch.zeros(5, 16))
    broken = torch.zeros(14, 16)
    broken[5, 0] = float("nan")
    with pytest.raises(ValueError, match="NONFINITE_OBSERVED_B_HISTORY"):
        r42._observed_b_peak(broken)


def test_r42_regret_matches_post_action_physical_state_and_factual_goal():
    for panel in r42.PANELS:
        for group in (0, 1):
            for context in (0, 1):
                for q in (-1, 1):
                    for gate in (-1, 1):
                        w = r42.world(panel, group, context, q, gate)
                        s = w["return_state"]
                        observed_return_costs = []
                        for action in (0, 1):
                            selected_effect = r42.r41._law(
                                s, action, context,
                                w["q_evaluator_only"] * w["gate_evaluator_only"],
                                panel)
                            # Match factual_fit(): state + decoded selected effect.
                            encoded = r42.r33.b._record(
                                s, action, selected_effect, context)[1]
                            result_state = s + r42.r31._physical_delta(encoded[0])
                            observed_return_costs.append(
                                (result_state - w["goal"]) ** 2)
                        for action in (0, 1):
                            actual = observed_return_costs[action] - min(
                                observed_return_costs)
                            assert r42._regret(w, action) == pytest.approx(
                                max(0.0, actual), abs=1e-6)
    with pytest.raises(ValueError, match="INVALID_RETURN_ACTION"):
        r42._regret(r42.world("dev", 0, 0, 1, 1), -1)


def test_old_delta_only_evaluation_can_invert_optimal_action():
    w = r42.world("dev", 1, 0, 1, 1)
    s = w["return_state"]
    change = [r42.r41._law(s, a, w["public_context"],
              w["q_evaluator_only"] * w["gate_evaluator_only"], "dev")
              for a in (0, 1)]
    incorrectly_scored = [(d-w["goal"])**2 for d in change]
    correctly_scored = [(s+d-w["goal"])**2 for d in change]
    assert incorrectly_scored.index(min(incorrectly_scored)) != (
           correctly_scored.index(min(correctly_scored)))
    wrong_action = incorrectly_scored.index(min(incorrectly_scored))
    assert r42._regret(w, wrong_action) > 0.0
