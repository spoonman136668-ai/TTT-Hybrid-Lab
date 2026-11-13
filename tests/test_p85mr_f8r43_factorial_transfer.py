"""F8-R43 prospective factorial transfer and learned causal salience negatives."""
import pytest
import torch

from experiments import p85mr_f8r43_factorial_transfer as r43


def test_preregistration_pinned_before_new_science_or_original_first_use():
    freeze = r43.locked()
    assert freeze["lineage"]["parent_sha"] == r43.PARENT
    assert freeze["source"]["new_salts"] == r43.SALT
    assert len(r43.PANELS) == 8
    assert len(r43.ARMS) == 10
    assert r43.STEPS == 256
    assert freeze["prospective_decision"]["independent_scientific_pass"] is False
    assert freeze["authority"]["external_original_simulator_first_use"] is False
    with pytest.raises(PermissionError, match="FIRST_USE_NOT_AUTHORIZED"):
        r43.refuse_external_scientific_first_use("not-allowed")
    with pytest.raises(ValueError, match="UNREGISTERED_PANEL"):
        r43.worlds("novel_external_original")
    with pytest.raises(ValueError, match="UNREGISTERED_WORLD"):
        r43.world("fit", 49, 0, 1, 1)
    with pytest.raises(ValueError, match="UNREGISTERED_SALIENCE_MODEL"):
        r43.SalienceGate("unregistered", 985201, None)


def test_factorial_source_is_independently_controlled_and_stays_factual():
    for g in range(r43.GROUPS):
        for h in (-1, 1):
            base = r43.world("e8_seen_quartic", g, 0, 1, h)
            extended = r43.world("e16_seen_quartic", g, 0, 1, h)
            new_position = r43.world("e8_novel_quartic", g, 0, 1, h)
            new_law = r43.world("e8_seen_cubic", g, 0, 1, h)
            assert base["b_history"] == extended["b_history"][:8]
            assert base["b_history"] == new_law["b_history"]
            assert len(new_position["b_history"]) == 8
            assert r43.marker(g, "seen") <= 4
            assert 5 <= r43.marker(g, "novel") <= 7
            a8, e8 = r43.candidates(base)
            a16, e16 = r43.candidates(extended)
            aq, eq = r43.candidates(new_law)
            assert a8.shape == e8.shape == (2, 14, 16)
            assert a16.shape == e16.shape == (2, 22, 16)
            assert torch.equal(e8[:, :13, :], e16[:, :13, :])
            assert torch.equal(e8, eq)
            assert torch.equal(a8, aq)
            assert torch.equal(e8[:, -1, :], torch.zeros_like(e8[:, -1, :]))
            flipped = r43.world("e8_seen_quartic", g, 0, 1, -h)
            _, ef = r43.candidates(flipped)
            assert torch.equal(e8[:, :5, :], ef[:, :5, :])
            assert torch.any(e8[:, 5:-1, :] != ef[:, 5:-1, :])
            assert r43._return_law(base, 0) != r43._return_law(new_law, 0)


def test_post_action_goal_matches_decoded_selected_factual_consequences():
    for panel in ("fit", "e8_seen_quartic", "e16_novel_cubic"):
        for g in (0, 1):
            for c in (0, 1):
                for q in (-1, 1):
                    for h in (-1, 1):
                        w = r43.world(panel, g, c, q, h)
                        costs = []
                        for a in (0, 1):
                            following = r43._return_law(w, a)
                            selected = r43.r33.b._record(
                                w["return_state"], a, following, c)
                            decoded_following = (w["return_state"] +
                                r43.r31._physical_delta(selected[1][0]))
                            assert decoded_following == pytest.approx(
                                following, abs=1e-6)
                            costs.append((decoded_following - w["goal"])**2)
                        for a in (0, 1):
                            assert r43._regret(w, a) == pytest.approx(
                                max(0., costs[a] - min(costs)), abs=1e-6)
    with pytest.raises(ValueError, match="INVALID_ACTION"):
        r43._regret(r43.world("e8_seen_quartic", 0, 0, 1, 1), -1)


def test_observational_control_feature_never_reads_hidden_marker(monkeypatch):
    w = r43.world("e16_novel_cubic", 1, 1, -1, 1)
    a, e = r43.candidates(w)
    with monkeypatch.context() as patch:
        patch.setattr(r43, "marker", lambda *_args, **_kwargs:
            pytest.fail("leaked generator marker index"))
        values = r43._control_feature_from_observations(
            a, e, w["return_state"], w["public_context"])
    assert len(values) == 8
    assert all(torch.isfinite(torch.tensor(values)))
    with pytest.raises(ValueError, match="BAD_CONTROL_OBSERVATIONS"):
        r43._control_feature_from_observations(
            a[:, :3, :], e, w["return_state"], w["public_context"])


def test_full_factorial_factual_training_cold_retention_and_nonpromotion():
    out = r43.run()
    assert out["schema"] == r43.SCHEMA
    assert out["freeze_blob"] == r43.FREEZE_BLOB
    assert out["parent"] == r43.PARENT
    assert out["factual_fit_episodes"] == 768
    assert out["train_pairs"] == 256
    assert out["heldout_pairs"] == 128
    assert out["original_binder_replay_updates"] == 1024
    assert out["new_neural_updates"] == 2560
    assert set(out["factored_panels"]) == set(r43.PANELS)
    assert len(out["models"]) == 10
    assert out["synthetic_development_only"]
    assert not out["independent_scientific_pass"]
    assert not out["external_scientific_first_use"]
    assert not out["GPU"]
    assert out["deployment"] == "NONE"
    assert not out["judgment"]["independent_scientific_pass"]
    assert out["judgment"]["native_specific_science"].startswith("UNVERIFIED")
    assert len(out["judgment"]["seeds"]) == 2
    assert out["observational_control"]["calibration_fit_worlds"] == 256
    assert not out["observational_control"]["oracle_metadata_used"]
    assert set(out["observational_control"]["panels"]) == set(r43.PANELS)
    for row in out["models"]:
        assert row["optimizer_updates"] == r43.STEPS
        assert len(row["checkpoints"]) == 2
        assert row["cold_restored"] is True
        assert set(row["panels"]) == set(r43.PANELS)
        for panel in r43.PANELS:
            assert len(row["panels"][panel]["active"]["choices"]) == 128
            assert row["panels"][panel]["active"]["mean_regret"] >= 0.0
        assert "erase_B" in row["panels"][r43.HARD]
        assert "zero_salience" in row["panels"][r43.HARD]
    r43.retain_ci_receipt(out)
