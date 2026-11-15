"""R85 calibration: negative controls use synthetic-only, non-science worlds.

No pytest code may call R85 official CALIBRATION, QUALIFICATION, DEV or OOD.
Only the original SHA-pinned GitHub-hosted PR job can first-use CALIBRATION.
"""
import copy

import numpy as np
import pytest

from experiments import r85_delayed_composition as r85


def _synthetic(i, seed=231):
    return r85._make_synthetic_world(
        i, np.random.default_rng(np.random.SeedSequence([0, seed, i])))


def test_prospective_source_freeze_is_immutable_and_distinct_from_spent():
    f = r85.frozen()
    assert r85.git_blob(r85.FREEZE.read_bytes()) == r85.FREEZE_BLOB
    assert r85.git_blob(r85.R84_FREEZE.read_bytes()) == r85.R84_FREEZE_BLOB
    assert f["parent"]["R84_candidate_SHA"] == r85.PARENT
    assert f["parent"]["R84_original_science"] == "DEV_LEARNED_INTERVENTION_NO_GO_BOTH_SEEDS"
    salts = f["source"]["independent_salts"]
    assert salts["CALIBRATION"] == r85.SALT
    assert set(salts.values()).isdisjoint(
        set(f["parent"]["R84_spent_salts"])
        | set(f["parent"]["R83_sealed_salts_still_untouched"])
        | {f["parent"]["R83_CALIBRATION_spent"]})
    assert f["source"]["seeds"] == list(r85.SEEDS)
    assert f["source"]["calibration_worlds_per_seed"] == 512
    assert f["classification"].endswith("NO_MODEL_TRAINING")
    assert f["gates"]["scientific_learning_pass"] is False
    assert all(v is False for v in f["authority"].values())


def test_no_protected_or_original_calibration_source_available_in_pytest():
    seed = r85.SEEDS[0]
    for forbidden in ("QUALIFICATION_SEALED", "DEV_SEALED", "OOD_SEALED",
                      "R84_DEV", "R83_CALIBRATION", "FIT"):
        with pytest.raises(PermissionError, match="R85_SEALED_OR_SPENT_PANEL_ACCESS_DENIED"):
            r85._world(forbidden, seed, 0)
    with pytest.raises(PermissionError, match="R85_CALIBRATION_SOURCE_NOT_ADMITTED"):
        r85._world("CALIBRATION", seed, 0)
    with pytest.raises(PermissionError, match="R85_UNFROZEN_SEED_OR_WORLD_INDEX"):
        r85._world("CALIBRATION", seed, 512)
    with pytest.raises(PermissionError, match="R85_UNFROZEN_SEED_OR_WORLD_INDEX"):
        r85._world("CALIBRATION", 7, 0)
    with pytest.raises(PermissionError, match="R85_NOT_ORIGINAL_QUALIFIED_HOSTED_PR_FIRST_ATTEMPT"):
        r85._admit()


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 14, 15])
def test_no_hidden_factor_or_current_reward_in_legal_prefix(idx):
    world = _synthetic(idx)
    a, b, r0, r2 = r85._paired_public(world)
    pivot = world["pivot"]
    assert a.shape == b.shape == (pivot + 1, 13)
    assert a.dtype == b.dtype == np.float32
    np.testing.assert_array_equal(a[:pivot], b[:pivot])
    np.testing.assert_array_equal(a[pivot, 4:13], b[pivot, 4:13])
    assert a[pivot, 0] == b[pivot, 2] == 1.
    assert a[pivot, 8:12].shape == (4,)
    assert np.isfinite([r0, r2]).all()
    assert world["law_a"] == int(np.prod(world["factors_a"]))
    assert world["law_b"] == int(np.prod(world["factors_b"]))
    # Selected probe roles are publicly marked; hidden signs never directly
    # appear outside the noisy past factual probe observations.
    a_probes = r85._public_probes(a, 1)
    assert len(a_probes) == 4
    assert {k for k, _ in a_probes} == {0, 1, 2, 3}
    if pivot == r85.COLD:
        b_probes = r85._public_probes(a, -1)
        assert len(b_probes) == 4
        assert {k for k, _ in b_probes} == {0, 1, 2, 3}
    else:
        assert not np.any(a[:, 11] == -1)


def test_exact_future_only_perturbation_cannot_reach_public_prefix():
    for idx in (0, 1, 2, 3):
        world = _synthetic(idx)
        altered = copy.deepcopy(world)
        p = world["pivot"]
        altered["innovations"][p + 1:p + 9, 0] += 30.
        altered["actions"][p + 1:p + 9] = 3
        for choice in (r85.CALIBRATE, r85.EXPLOIT_POS):
            prefix, reward = r85._rollout(world, choice)
            future_changed, new_reward = r85._rollout(altered, choice)
            np.testing.assert_array_equal(prefix, future_changed)
            assert abs(new_reward - reward) > 1.0


def test_compositional_public_rule_reconstructs_a_across_cold_return():
    for idx in (0, 1, 2, 3):
        world = _synthetic(idx)
        # Perfect public A cues for controlled TEST-only negative, not official data.
        for t in r85.A_PROBES:
            world["innovations"][t, 2] = 0.0
        # B cues are independent distractors after reversal.
        for t in r85.B_PROBES:
            world["innovations"][t, 2] = 0.0
        a, _, r0, r2 = r85._paired_public(world)
        choice = r85._policy(a)
        assert choice == (r85.CALIBRATE if world["law_a"] > 0 else r85.EXPLOIT_POS)
        if world["law_a"] > 0:
            assert r0 > r2
        else:
            assert r2 > r0
        if world["pivot"] == r85.COLD:
            changed_b = copy.deepcopy(world)
            changed_b["factors_b"] = -changed_b["factors_b"]
            changed_b["innovations"][r85.B_PROBES, 2] += 10
            changed, _, _, _ = r85._paired_public(changed_b)
            assert r85._policy(changed) == choice
            np.testing.assert_array_equal(
                a[r85.A_PROBES[-1] + 1], changed[r85.A_PROBES[-1] + 1])


def test_long_delay_is_not_single_probe_shortcut():
    world = _synthetic(1)
    world["factors_a"] = np.array([-1, -1, 1, 1], dtype=np.int64)
    world["law_a"] = 1
    for t in r85.A_PROBES:
        world["innovations"][t, 2] = 0.0
    a, _, _, _ = r85._paired_public(world)
    assert r85._policy(a) == r85.CALIBRATE
    assert r85._policy(a, "RECENT_B") in (r85.CALIBRATE, r85.EXPLOIT_POS)
    with pytest.raises(ValueError, match="R85_UNREGISTERED_PUBLIC_RULE"):
        r85._policy(a, "ORACLE_HIDDEN_LAW")


def test_frozen_information_thresholds_and_sealed_source_rule():
    f = r85.frozen()
    t = f["gates"]
    assert t["paired_world_bootstrap_resamples"] == 2000
    assert t["public_minus_best_fixed_reward_min"] == 1.75
    assert t["public_minus_shuffled_A_reward_min"] == 1.75
    assert t["public_minus_last_A_cue_reward_min"] == 1.75
    assert t["public_minus_recent_B_reward_cold_min"] == 1.5
    assert t["early_public_minus_best_fixed_min"] == 1.5
    assert t["cold_public_minus_best_fixed_min"] == 1.5
    assert t["classification_fail"] == "ENGINEERING_R85_BENCHMARK_INFORMATION_NO_GO"
    assert f["source"]["access_now"].startswith("CALIBRATION ONLY")
    assert f["next_distinct_frozen_learner_study"].startswith("Only if R85 calibration passes")
