"""R101 development-only negatives. No FIT/DEV physical first use here."""
import numpy as np
import pytest
import torch

from experiments import r85_delayed_composition as r85
from experiments import r101_unknown_operator_diagnostic as r101


def test_freeze_sha_north_star_and_science_fences():
    f = r101.frozen()
    assert r85.git_blob(r101.FREEZE.read_bytes()) == r101.FREEZE_BLOB
    assert f["parent"]["r100_sealed_parent_SHA"] == r101.PARENT
    assert not f["parent"]["prior_scored_source_or_weights_reused"]
    assert f["source"]["support_factual_interventions"] == 6
    assert f["source"]["no_prior_cohort_replay"]
    assert set(f["source"]["known_function_families"]).isdisjoint(
        f["source"]["unseen_function_families"])
    assert len(f["arms"]) == 9
    assert f["evaluation"]["independent_seeds"] == 3
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert not f["authority"]["protected_scored_first_use"]
    assert not f["authority"]["official_GPU_first_use"]
    assert not f["authority"]["merge"]
    assert not f["authority"]["FORGE_integration"]


@pytest.mark.parametrize("panel", ["FIT", "VALID", "DEV_FAMILIAR", "DEV_UNKNOWN",
                                     "OOD_SHIFT", "DELAYED_COLD", "REVERSAL"])
def test_all_new_scored_panels_denied_even_with_plausible_sha(panel):
    with pytest.raises(PermissionError, match="R101_SCORED_PHYSICAL_SOURCE_NOT_ADMITTED"):
        r101._world(panel, r101.SEEDS[0], 0, "a" * 40)


@pytest.mark.parametrize("panel", ["R100_DEV_COMPOSE", "R99_FIT", "R85_CALIBRATION", "PROTECTED"])
def test_old_spent_and_protected_source_denied(panel):
    with pytest.raises(PermissionError, match="R101_OLD_SCORED_OR_PROTECTED_SOURCE_DENIED"):
        r101._world(panel, r101.SEEDS[0], 0, "a" * 40)


def test_unknown_seed_index_and_original_first_use_always_denied():
    with pytest.raises(PermissionError, match="R101_UNFROZEN_SOURCE_INDEX_OR_SEED"):
        r101._world("FIT", 42, 0)
    with pytest.raises(PermissionError, match="R101_UNFROZEN_SOURCE_INDEX_OR_SEED"):
        r101._world("DEV_UNKNOWN", r101.SEEDS[0], 128)
    with pytest.raises(PermissionError, match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r101._admit("original")
    with pytest.raises(PermissionError, match="R101_PROTECTED_ORIGINAL_FIRST_USE_NOT_AUTHORIZED"):
        r101.original_ci()


@pytest.mark.parametrize("family", r101.KNOWN + r101.UNSEEN)
def test_private_operator_is_evaluator_only_and_deterministic(family):
    factors = np.array([1, -1, 1, -1], dtype=np.int64)
    got = r101._operator_truth(factors, (0, 1, 2), family)
    assert got in (-1, 1)
    assert got == r101._operator_truth(factors.copy(), (0, 1, 2), family)
    with pytest.raises(ValueError, match="R101_UNKNOWN_PHYSICAL_OPERATOR"):
        r101._operator_truth(factors, (0, 1, 2), "FORBIDDEN")


@pytest.mark.parametrize("index", [0, 1, 5, 6, 7, 13])
def test_unit_only_physical_observations_have_seven_events_no_hidden_law(index):
    obj = r101.unit_fixture(index)
    public, evaluator = obj["public"], obj["evaluator_only"]
    assert set(public) == {"support_slots", "support_rewards", "support_actions",
                           "query0", "query2"}
    assert public["support_slots"].shape == (6, 5, 13)
    assert public["support_rewards"].shape == (6,)
    assert public["query0"].shape == public["query2"].shape == (5, 13)
    assert public["support_slots"].dtype == np.float32
    assert np.all(np.isfinite(public["support_rewards"]))
    assert np.all(np.isfinite(public["support_slots"]))
    assert set(public["support_actions"]) == {r85.CALIBRATE, r85.EXPLOIT_POS}
    assert np.sum(public["support_actions"] == r85.CALIBRATE) == 3
    assert evaluator["family"] == (r101.KNOWN + r101.UNSEEN)[index % 7]
    assert evaluator["private_law"] in (-1, 1)
    assert np.array_equal(public["query0"][:4], public["query2"][:4])
    assert np.array_equal(public["query0"][-1, 4:], public["query2"][-1, 4:])
    assert public["query0"][-1, r85.CALIBRATE] == 1.
    assert public["query2"][-1, r85.EXPLOIT_POS] == 1.
    # The public role tags carry a permutation of three roles plus distractor,
    # but NO hidden family opcode, precomputed law sign or private truth label.
    for slots in list(public["support_slots"]) + [public["query0"]]:
        assert sorted(slots[:4, 8].tolist()) == [0., .25, .5, .75]
    with pytest.raises(PermissionError, match="R101_UNIT_ONLY_FIXTURE_RANGE"):
        r101.unit_fixture(14)


def test_unit_support_shuffle_preserves_action_and_time_and_has_no_fixed_points():
    public = r101.unit_fixture(3)["public"]
    perm = r101.support_shuffle_indices(public["support_actions"],
                                        public["support_slots"], 3)
    assert len(set(perm.tolist())) == 6
    assert not np.any(perm == np.arange(6))
    np.testing.assert_array_equal(public["support_actions"][perm],
                                  public["support_actions"])
    np.testing.assert_array_equal(public["support_slots"][perm, -1, 12],
                                  public["support_slots"][:, -1, 12])
    with pytest.raises(PermissionError, match="R101_UNIT_ONLY_SHUFFLE"):
        r101.support_shuffle_indices(public["support_actions"],
                                     public["support_slots"], 999)


def test_native_negative_controls_identically_initialized_and_all_nine_forward():
    torch.set_num_threads(1)
    models = r101.make_models(r101.SEEDS[0])
    assert set(models) == set(r101.ARMS)
    fixture = r101.unit_fixture(4)
    for name, model in models.items():
        a = r101.unit_value(model, fixture)
        assert a.shape == (1,), name
        assert torch.isfinite(a).all(), name
    for name in (r101.SHUFFLED, r101.NO_OUTCOME, r101.RESET):
        a = dict(models[r101.NATIVE].named_parameters())
        b = dict(models[name].named_parameters())
        assert a.keys() == b.keys()
        assert all(torch.equal(a[key], b[key]) for key in a)


def test_native_support_reward_gradient_can_reach_memory_and_native_stem():
    torch.set_num_threads(1)
    model = r101.make_models(r101.SEEDS[0])[r101.NATIVE]
    fixture = r101.unit_fixture(0)
    public = fixture["public"]
    x = torch.from_numpy(public["support_slots"])[None]
    y = torch.from_numpy(public["support_rewards"])[None].clone().requires_grad_(True)
    q = torch.from_numpy(public["query0"])[None]
    value = model(x, y, q)
    assert value.shape == (1,)
    value.square().mean().backward()
    assert y.grad is not None and torch.isfinite(y.grad).all()
    assert sum(float(p.grad.abs().sum()) for p in model.support.parameters()
               if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in model.core.stem.parameters()
               if p.grad is not None) > 0


def test_no_outcome_and_reset_negative_control_cannot_see_reward():
    torch.set_num_threads(1)
    models = r101.make_models(r101.SEEDS[0])
    fixture = r101.unit_fixture(1)["public"]
    x = torch.from_numpy(fixture["support_slots"])[None]
    y = torch.from_numpy(fixture["support_rewards"])[None]
    q = torch.from_numpy(fixture["query0"])[None]
    with torch.no_grad():
        for name in (r101.NO_OUTCOME, r101.RESET):
            m = models[name]
            assert torch.allclose(m(x, y, q), m(x, y + 100., q))
        # The real arm receives true factual feedback into the state.
        m = models[r101.NATIVE]
        assert not torch.allclose(m(x, y, q), m(x, y + 100., q), atol=1e-7)


def test_no_original_science_or_production_write_from_dev_module():
    assert r101.STEPS == 3072 and r101.BATCH == 16
    assert not hasattr(r101, "run_scored")
    assert not hasattr(r101, "promote")
