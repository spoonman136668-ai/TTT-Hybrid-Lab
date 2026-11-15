"""R86 source and leakage guards; fixtures never access any R86 research panel."""
import copy

import numpy as np
import pytest
import torch

from experiments import r86_native_compositional_value as r86


def _test_world(index):
    return r86._test_only_world(index)


def test_original_immutable_freeze_and_source_separation():
    f = r86.frozen()
    assert r86.r85.git_blob(r86.FREEZE.read_bytes()) == r86.FREEZE_BLOB
    assert f["parent"]["R85_exact_SHA"] == r86.PARENT
    assert f["parent"]["R85_freeze_blob"] == r86.r85.FREEZE_BLOB
    assert f["parent"]["R84_original_status"] == "DEV_LEARNED_INTERVENTION_NO_GO_BOTH_SEEDS"
    assert set(r86.SALT.values()).isdisjoint(
        set(f["parent"]["R85_reserved_salts_still_sealed"]) |
        set(f["parent"]["R84_spent_salts"]) |
        {f["parent"]["R85_CALIBRATION_salt_spent"]})
    assert f["worlds"]["model_observed_timesteps"] == 48
    assert f["models"]["arms"] == list(r86.ARMS)
    assert f["models"]["training"]["native_updates"] == 1024
    assert f["models"]["training"]["GRU64_updates"] == 8192
    assert f["models"]["training"]["GRU256_updates"] == 8192
    assert not f["models"]["training"]["strict_equal_FLOPs"]
    assert not f["evaluation"]["thresholds"]["independent_scientific_pass"]
    assert all(x is False for x in f["authority"].values())


def test_no_original_scoring_or_factual_world_in_full_regression():
    seed = r86.SEEDS[0]
    for blocked in ("R85_CALIBRATION", "R85_DEV", "R84_FIT",
                    "R85_QUALIFICATION", "INVALID", "TEST_ONLY"):
        with pytest.raises(PermissionError, match="R86_PRIOR_OR_PROTECTED_PANEL_DENIED"):
            r86._fresh_world(blocked, seed, 0)
    with pytest.raises(PermissionError, match="R86_ORIGINAL_SOURCE_NOT_ADMITTED"):
        r86._fresh_world("FIT", seed, 0)
    with pytest.raises(PermissionError, match="R86_ORIGINAL_SOURCE_NOT_ADMITTED"):
        r86._fresh_world("DEV", seed, 0)
    with pytest.raises(PermissionError, match="R86_SOURCE_SEED_OR_INDEX_DENIED"):
        r86._fresh_world("DEV", seed, 128)
    with pytest.raises(PermissionError, match="R86_SOURCE_SEED_OR_INDEX_DENIED"):
        r86._fresh_world("FIT", 2, 0)
    with pytest.raises(PermissionError, match="R86_ONLY_FIT_AND_VALIDATION_ALLOWED_FOR_LEARNING"):
        r86._factual("DEV", seed, custody_sha="fake")
    with pytest.raises(PermissionError, match="R86_ONLY_FRESH_PROTECTED_EVALUATION_ALLOWED"):
        r86._paired_scoring("FIT", seed, custody_sha="fake")
    with pytest.raises(PermissionError, match="R86_ORIGINAL_PR_HOSTED_CPU_FIRST_ATTEMPT_ONLY"):
        r86._admit(seed)


@pytest.mark.parametrize("index", [0, 1, 4, 5, 14, 15])
def test_event_compression_retains_exact_public_causal_probes(index):
    w = _test_world(index)
    p = w["pivot"]
    x0, r0 = r86._view(w, r86.r85.CALIBRATE)
    x2, r2 = r86._view(w, r86.r85.EXPLOIT_POS)
    assert x0.shape == x2.shape == (48, 13)
    assert x0.dtype == x2.dtype == np.float32
    np.testing.assert_array_equal(x0[:-1], x2[:-1])
    np.testing.assert_array_equal(x0[-1, 4:], x2[-1, 4:])
    assert x0[-1, 0] == x2[-1, 2] == 1.
    assert x0[-1, 12] == pytest.approx(p / 223)
    assert len(r86.r85._public_probes(x0, 1)) == 4
    if p == r86.r85.COLD:
        assert len(r86.r85._public_probes(x0, -1)) == 4
    else:
        assert not np.any(x0[:, 11] == -1)
    assert r0 != r2
    assert r86._blind(x0)[..., 4:12].sum() == 0
    np.testing.assert_array_equal(r86._blind(x0)[..., :4], x0[..., :4])
    np.testing.assert_array_equal(r86._blind(x0)[..., 12], x0[..., 12])


def test_no_selected_current_reward_future_or_hidden_law_leakage():
    for index in (0, 1, 2, 3):
        w = _test_world(index)
        later = copy.deepcopy(w)
        p = w["pivot"]
        later["innovations"][p + 1:p + 9, 0] += 50
        for chosen in (r86.r85.CALIBRATE, r86.r85.EXPLOIT_POS):
            original, reward = r86._view(w, chosen)
            changed, reward_changed = r86._view(later, chosen)
            np.testing.assert_array_equal(original, changed)
            assert abs(reward_changed - reward) > 1.


def test_ood_shift_uses_different_probe_clocks_and_preserves_public_only_observation():
    for index in (0, 1):
        w = _test_world(index)
        shifted = r86._shift_public_world(w, index)
        assert w["pivot"] == (64 if index == 0 else 192)
        assert shifted["pivot"] == (68 if index == 0 else 208)
        assert sorted(t for t, (phase, _, _) in shifted["probe"].items() if phase == "A") == [12, 22, 30, 40]
        assert sorted(t for t, (phase, _, _) in shifted["probe"].items() if phase == "B") == [110, 120, 136, 148]
        assert shifted["law_a"] == w["law_a"]
        assert shifted["law_b"] == w["law_b"]
        assert not np.array_equal(shifted["innovations"], w["innovations"])
        x0, r0 = r86._view(shifted, r86.r85.CALIBRATE)
        x2, r2 = r86._view(shifted, r86.r85.EXPLOIT_POS)
        assert x0.shape == (48, 13)
        np.testing.assert_array_equal(x0[:-1], x2[:-1])
        assert len(r86.r85._public_probes(x0, 1)) == 4
        if index == 1:
            assert len(r86.r85._public_probes(x0, -1)) == 4
        assert r0 != r2


def test_correct_a_factor_parity_on_pure_test_source_even_after_b_regime():
    for index in (0, 1, 2, 3):
        w = _test_world(index)
        for t in list(w["probe"]):
            w["innovations"][t, 2] = 0  # Perfect public fixture only.
        x, _ = r86._view(w, r86.r85.CALIBRATE)
        expected = r86.r85.CALIBRATE if w["law_a"] > 0 else r86.r85.EXPLOIT_POS
        assert r86.r85._policy(x) == expected
        if index % 2 == 1:
            other = copy.deepcopy(w)
            other["factors_b"] = -other["factors_b"]
            x2, _ = r86._view(other, r86.r85.CALIBRATE)
            assert r86.r85._policy(x2) == expected


def test_frozen_native_model_is_identical_at_initialization_and_causal():
    torch.set_num_threads(1)
    ms = r86._models(r86.SEEDS[0])
    assert set(ms) == set(r86.ARMS)
    p = dict(ms["NATIVE_FULL"].named_parameters())
    q = dict(ms["NATIVE_OBSERVATION_BLIND"].named_parameters())
    assert sum(t.numel() for t in p.values()) == 16818178
    assert p.keys() == q.keys()
    assert all(torch.equal(p[k], q[k]) for k in p)
    x = torch.from_numpy(r86._view(_test_world(0), r86.r85.CALIBRATE)[0])
    for name, model in ms.items():
        model.eval()
        with torch.no_grad():
            original = model(x[None])[:, -1]
            shortened = model(x[None, :-1])
        assert original.shape == (1, 1)
        assert shortened.shape == (1, 47, 1)
