"""R87 frozen source, legal representation and mechanistic negative controls.

Every test world is a synthetic independent fixture with salt 25299911.
No original R87 FIT/VALIDATION/DEV/OOD world is created in pytest.
"""
import copy

import numpy as np
import pytest
import torch

from experiments import r87_public_composition_diagnostic as r87


def fixture_world(index):
    rng = np.random.default_rng(np.random.SeedSequence([25299911, index]))
    return r87.r85._make_synthetic_world(index, rng)


def test_immutable_preimplementation_freeze_and_previous_negative_preservation():
    f = r87.frozen()
    assert r87.r85.git_blob(r87.FREEZE.read_bytes()) == r87.FREEZE_BLOB
    assert f["parent"]["R86_exact_SHA"] == r87.PARENT
    assert f["parent"]["R86_freeze_blob"] == r87.r86.FREEZE_BLOB
    assert f["parent"]["R86_two_seed_result"] == "DEV_NATIVE_COMPOSITIONAL_VALUE_NO_GO_BOTH_SEEDS_ALL_11_FROZEN_CHECKS_FAILED"
    assert f["sources"]["salts"] == r87.SALT
    assert f["sources"]["seeds"] == list(r87.SEEDS)
    assert f["arms"] == list(r87.ARMS)
    assert not f["evaluation"]["thresholds"]["independent_scientific_pass"]
    assert f["representations"]["no_native_retraining"]
    assert all(a is False for a in f["authority"].values())
    spent = set(f["parent"]["R86_FIT_DEV_OOD_salts_spent"])
    spent |= set(f["parent"]["R85_QUAL_DEV_OOD_reserved_untouched"])
    spent.add(f["parent"]["R85_CALIBRATION_spent"])
    assert set(r87.SALT.values()).isdisjoint(spent)


def test_every_original_panel_closed_outside_first_attempt_scoped_job():
    seed = r87.SEEDS[0]
    for panel in ("R85_CALIBRATION", "R86_FIT", "R86_DEV", "R83_SEALED", "TEST"):
        with pytest.raises(PermissionError, match="R87_PRIOR_OR_SEALED_SCIENCE_SOURCE_DENIED"):
            r87._world(panel, seed, 0)
    for panel in r87.SALT:
        with pytest.raises(PermissionError, match="R87_ORIGINAL_SOURCE_ACCESS_NOT_ADMITTED"):
            r87._world(panel, seed, 0)
    with pytest.raises(PermissionError, match="R87_NONFROZEN_SEED_OR_INDEX"):
        r87._world("FIT", seed, r87.COUNT["FIT"])
    with pytest.raises(PermissionError, match="R87_NONFROZEN_SEED_OR_INDEX"):
        r87._world("DEV", 77, 0)
    with pytest.raises(PermissionError, match="R87_TRAINING_ON_SCORED_PANEL_DENIED"):
        r87._assemble("DEV", seed, custody_sha="not-admitted")
    with pytest.raises(PermissionError, match="R87_SCORING_ONLY_FRESH_PROTECTED_PANELS"):
        r87._evaluation("FIT", seed, custody_sha="not-admitted")
    with pytest.raises(PermissionError, match="R87_ORIGINAL_FIRST_ATTEMPT_HOSTED_PR_ADMISSION_DENIED"):
        r87._admit(seed)


@pytest.mark.parametrize("idx", [0, 1, 2, 3, 14, 15])
def test_observed_cues_legal_actions_and_timestamps(idx):
    w = fixture_world(idx)
    x0, r0 = r87._view(w, r87.r85.CALIBRATE)
    x2, r2 = r87._view(w, r87.r85.EXPLOIT_POS)
    assert x0.shape == x2.shape == (48, 13)
    assert x0.dtype == x2.dtype == np.float32
    np.testing.assert_array_equal(x0[:-1], x2[:-1])
    np.testing.assert_array_equal(x0[-1,4:], x2[-1,4:])
    f0,p0 = r87._legal_features(x0)
    f2,p2 = r87._legal_features(x2)
    assert f0.shape == (8,)
    assert f0.dtype == np.float32
    np.testing.assert_array_equal(f0[:4],f2[:4])
    assert p0 == p2 and p0 in (-1.0, 1.0)
    assert f0[4:6].tolist() == [1.0, 0.0]
    assert f2[4:6].tolist() == [0.0, 1.0]
    assert f0[7] == f2[7] == 0
    assert f0[6] == pytest.approx(w["pivot"]/223)
    assert np.isfinite([r0,r2]).all() and r0 != r2
    assert len(r87.r85._public_probes(x0,1)) == 4
    if idx % 2:
        assert len(r87.r85._public_probes(x0,-1)) == 4


def test_private_truth_not_accessible_through_public_preintervention_inputs():
    for idx in (0,1,2,3):
        w=fixture_world(idx)
        w["innovations"][list(r87.r85.A_PROBES),2]=0.
        x, _ = r87._view(w,r87.r85.CALIBRATE)
        _, parity = r87._legal_features(x)
        assert parity == w["law_a"]
        # Future-only perturbation changes true physical returns, not legal past.
        future=copy.deepcopy(w)
        p=w["pivot"]
        future["innovations"][p+1:p+9,0]+=33.
        for act in (r87.r85.CALIBRATE,r87.r85.EXPLOIT_POS):
            a,rew =r87._view(w,act)
            b,newrew=r87._view(future,act)
            np.testing.assert_array_equal(a,b)
            assert newrew != rew


def test_stratum_preserving_composition_shuffle_must_break_correct_association():
    n=r87.COUNT["FIT"]
    raw=np.zeros((n,8),np.float32)
    raw[::2,6] = 64/223
    raw[1::2,6] = 192/223
    parity=np.tile(np.array([1,-1,-1,1],np.float32),n//4)
    swapped=r87._shuffle_parity(raw,parity,r87.SEEDS[0],"FIT")
    again=r87._shuffle_parity(raw,parity,r87.SEEDS[0],"FIT")
    np.testing.assert_array_equal(swapped,again)
    assert swapped.shape==parity.shape
    assert not np.array_equal(swapped,parity)
    for rows in (np.arange(0,n,2),np.arange(1,n,2)):
        assert sorted(swapped[rows].tolist()) == sorted(parity[rows].tolist())
    with pytest.raises(PermissionError,match="R87_SHUFFLED_NEGATIVE_ONLY_ON_REGISTERED_PANEL"):
        r87._shuffle_parity(raw,parity,r87.SEEDS[0],"INVALID")


def test_frozen_composition_signal_is_not_hidden_label_and_same_width():
    torch.set_num_threads(1)
    models=r87._models(r87.SEEDS[0])
    assert set(models)==set(r87.ARMS)
    params=[sum(p.numel() for p in models[a].parameters()) for a in r87.ARMS[:3]]
    assert len(set(params))==1
    raw=np.stack([r87._legal_features(r87._view(fixture_world(i),r87.r85.CALIBRATE)[0])[0]
                  for i in (0,1)],axis=0)
    parity=np.array([r87._legal_features(r87._view(fixture_world(i),r87.r85.CALIBRATE)[0])[1]
                     for i in (0,1)],np.float32)
    base={"features":raw,"parity":parity,"shuffled_parity":parity[::-1],
          "sequence":np.stack([r87._view(fixture_world(i),r87.r85.CALIBRATE)[0]
                               for i in (0,1)])}
    a=r87._arm_inputs(base,"MLP64_RAW")
    b=r87._arm_inputs(base,"MLP64_COMPOSED")
    c=r87._arm_inputs(base,"MLP64_COMPOSED_SHUFFLED")
    assert a.shape==b.shape==c.shape==(2,8)
    np.testing.assert_array_equal(a[:,:7],b[:,:7])
    np.testing.assert_array_equal(a[:,:7],c[:,:7])
    assert np.all(a[:,7]==0)
    np.testing.assert_array_equal(b[:,7],parity)
    np.testing.assert_array_equal(c[:,7],parity[::-1])
    assert r87._arm_inputs(base,"GRU64_PUBLIC").shape==(2,48,13)
    for name,model in models.items():
        inputs=r87._arm_inputs(base,name)
        with torch.no_grad():
            value=model(torch.from_numpy(inputs))
        if name=="GRU64_PUBLIC":
            assert value.shape == (2,48,1)
        else:
            assert value.shape == (2,)
        assert bool(torch.isfinite(value).all())


def test_original_no_postresult_tuning_thresholds():
    g=r87.frozen()["evaluation"]["thresholds"]
    assert g["DEV_composed_minus_strongest_raw_control_min"]==0.30
    assert g["DEV_composed_minus_shuffled_min"]==0.75
    assert g["DEV_composed_minus_best_fixed_min"]==1.25
    assert g["DEV_composed_cold_minus_strongest_raw_min"]==0.30
    assert g["DEV_composed_legal_rule_action_agreement_min"]==0.85
    assert g["DEV_composed_oracle_regret_max"]==0.70
    assert g["OOD_composed_minus_DEV_selected_raw_min"]==0.20
    assert g["OOD_cold_composed_minus_DEV_selected_raw_min"]==0.20
    assert g["DEV_legal_A_parity_probe_agreement_min"]==0.90
    assert g["independent_scientific_pass"] is False
    assert r87.frozen()["training"]["same_update_count"]
