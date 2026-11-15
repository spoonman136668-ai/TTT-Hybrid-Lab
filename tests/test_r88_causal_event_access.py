"""R88 source custody and mechanism-level negative/control tests, TEST ONLY.

Never generate R88 FIT, validation, DEV or shifted OOD in regression.
"""
import copy

import numpy as np
import pytest
import torch

from experiments import r88_causal_event_access as r88


def _public(i,chosen=None):
    w = r88._test_only_world(i)
    a = r88.r85.CALIBRATE if chosen is None else chosen
    return r88._view(w,a)[0]


def test_freeze_immutable_and_scored_source_separation():
    f = r88.frozen()
    assert r88.r85.git_blob(r88.FREEZE.read_bytes()) == r88.FREEZE_BLOB
    assert f["parent"]["R87_source_sha"] == r88.PARENT
    assert f["parent"]["R87_freeze_blob"] == r88.r87.FREEZE_BLOB
    assert f["parent"]["R87_science"] == "R87_COMPOSITION_BOTTLENECK_NO_GO_BOTH_SEEDS_7_OF_10_FROZEN_CHECKS_PASS"
    assert set(r88.SALT.values()).isdisjoint(
        set(f["parent"]["R87_spent_salts"]) |
        set(f["parent"]["R86_spent_salts"]) |
        set(f["parent"]["R85_spent"]) |
        set(f["parent"]["R85_reserved_untouched"]))
    assert f["arms"] == list(r88.ARMS)
    assert f["training"]["all_arms_steps"] == 4096
    assert f["training"]["batch"] == 16
    assert not f["training"]["whole_process_compute_equal"]
    assert not f["evaluation"]["thresholds"]["independent_scientific_pass"]
    assert all(v is False for v in f["authority"].values())


def test_original_factual_and_scored_source_denied_outside_admitted_job():
    seed = r88.SEEDS[0]
    for bad in ("R87_DEV", "R86_FIT", "R85_CALIBRATION", "OLD", "TEST_ONLY"):
        with pytest.raises(PermissionError,match="R88_OLD_OR_SEALED_PANEL_DENIED"):
            r88._world(bad,seed,0)
    for panel in r88.SALT:
        with pytest.raises(PermissionError,match="R88_ORIGINAL_SOURCE_NOT_ADMITTED"):
            r88._world(panel,seed,0)
    with pytest.raises(PermissionError,match="R88_NONFROZEN_OR_OUT_OF_RANGE_PANEL_INDEX"):
        r88._world("FIT",seed,r88.COUNT["FIT"])
    with pytest.raises(PermissionError,match="R88_NONFROZEN_OR_OUT_OF_RANGE_PANEL_INDEX"):
        r88._world("DEV",123,0)
    with pytest.raises(PermissionError,match="R88_TRAIN_DATA_CANNOT_INCLUDE_DEV_OOD"):
        r88._assemble("DEV",seed,custody_sha="fake")
    with pytest.raises(PermissionError,match="R88_ONLY_NEW_HELDOUT_EVALUATOR_SOURCE"):
        r88._paired("FIT",seed,custody_sha="fake")
    with pytest.raises(PermissionError,match="R88_FIRST_ATTEMPT_ORIGINAL_G4_HOSTED_JOB_REQUIRED"):
        r88._admit(seed)


@pytest.mark.parametrize("idx",[0,1,4,5,30,31])
def test_public_slots_are_only_actual_a_probe_rows_and_current_action(idx):
    w=r88._test_only_world(idx)
    x0,ret0 = r88._view(w,r88.r85.CALIBRATE)
    x2,ret2 = r88._view(w,r88.r85.EXPLOIT_POS)
    assert x0.shape==x2.shape==(48,13)
    np.testing.assert_array_equal(x0[:-1],x2[:-1])
    np.testing.assert_array_equal(x0[-1,4:],x2[-1,4:])
    e0,e2=r88._public_slots(x0),r88._public_slots(x2)
    assert e0.shape==e2.shape==(5,13)
    np.testing.assert_array_equal(e0[:4],e2[:4])
    np.testing.assert_array_equal(e0[-1],x0[-1])
    np.testing.assert_array_equal(e2[-1],x2[-1])
    assert e0[-1,0]==1 and e2[-1,2]==1
    assert ret0!=ret2
    assert set(np.round(e0[:4,9]*4).astype(int).tolist())=={1,2,3,4}
    assert np.all(e0[:4,5]==1) and np.all(e0[:4,11]==1)
    for role in range(4):
        assert e0[role,9]==pytest.approx((role+1)/4)
    assert np.all(e0[:4,12]<x0[-1,12])


def test_phase_mask_drops_non_a_observations_but_no_actions_times_or_a_cues():
    for idx in (0,1,2,3):
        x=_public(idx)
        masked=r88._public_A_mask(x)
        a=(x[:,5]==1)&(x[:,11]==1)
        assert a.sum()==4
        assert np.all(masked[~a,8:12]==0)
        np.testing.assert_array_equal(masked[a],x[a])
        np.testing.assert_array_equal(masked[:,:8],x[:,:8])
        np.testing.assert_array_equal(masked[:,12],x[:,12])
        assert not np.array_equal(masked,x)


def test_physical_future_only_randomness_never_changes_any_public_predecision_view():
    for idx in (0,1):
        world=r88._test_only_world(idx)
        later=copy.deepcopy(world)
        p=world["pivot"]
        later["innovations"][p+1:p+9,0]+=44
        for act in (r88.r85.CALIBRATE,r88.r85.EXPLOIT_POS):
            x0,reward0=r88._view(world,act)
            x1,reward1=r88._view(later,act)
            np.testing.assert_array_equal(x0,x1)
            assert abs(reward1-reward0)>1
            np.testing.assert_array_equal(r88._public_slots(x0),r88._public_slots(x1))


def test_legal_probe_factor_parity_from_test_only_world_without_hidden_truth():
    for idx in (0,1,2,3):
        w=r88._test_only_world(idx)
        w["innovations"][list(r88.r85.A_PROBES),2]=0
        x,_=r88._view(w,r88.r85.CALIBRATE)
        slot=r88._public_slots(x)
        sign=int(np.prod(np.where(slot[:4,10]>0,1,-1)))
        assert sign==w["law_a"]
        assert r88.r85._policy(x)==(
            r88.r85.CALIBRATE if sign>0 else r88.r85.EXPLOIT_POS)


def test_stratum_preserving_shuffled_four_event_quartets_never_mix_own_action():
    n=r88.COUNT["FIT"]
    slots=np.zeros((n,5,13),np.float32)
    slots[:,4,0]=1
    slots[::2,4,12]=64/223
    slots[1::2,4,12]=192/223
    slots[:,:4,10]=np.arange(n,dtype=np.float32)[:,None]
    other=r88._shuffle_slots(slots,r88.SEEDS[0],"FIT")
    again=r88._shuffle_slots(slots,r88.SEEDS[0],"FIT")
    np.testing.assert_array_equal(other,again)
    np.testing.assert_array_equal(other[:,4,:],slots[:,4,:])
    assert not np.array_equal(other[:,:4],slots[:,:4])
    for stride in (0,1):
        selected=np.arange(stride,n,2)
        assert sorted(other[selected,0,10].tolist())==sorted(slots[selected,0,10].tolist())
    with pytest.raises(PermissionError,match="R88_SHUFFLE_ONLY_FROZEN_SOURCE_SHAPE"):
        r88._shuffle_slots(slots,r88.SEEDS[0],"OLD")


def test_same_gru_parameter_geometry_and_trainable_role_event_attention():
    torch.set_num_threads(1)
    models=r88._models(r88.SEEDS[0])
    assert set(models)==set(r88.ARMS)
    num=[sum(p.numel() for p in models[a].parameters()) for a in r88.GRUS]
    assert len(set(num))==1
    assert sum(p.numel() for p in models[r88.LEARNED].parameters())>num[0]
    for arm in r88.ARMS:
        x=np.stack([_public(0),_public(1)])
        if arm==r88.MASK:
            x=np.stack([r88._public_A_mask(y) for y in x])
        elif arm in (r88.SLOTS,r88.SHUFFLE):
            x=np.stack([r88._public_slots(y) for y in x])
        elif arm==r88.MLP:
            x=np.stack([r88.r87._legal_features(y)[0] for y in x])
        with torch.no_grad():
            val=r88._prediction(models[arm],arm,torch.from_numpy(x))
        assert val.shape==(2,)
        assert bool(torch.isfinite(val).all())
    learned=models[r88.LEARNED]
    input=torch.from_numpy(np.stack([_public(0),_public(1)]))
    val=learned(input)
    assert val.shape==(2,)
    (val.square().mean()).backward()
    assert learned.queries.grad is not None
    assert bool(torch.isfinite(learned.queries.grad).all())
    assert learned.embed[0].weight.grad is not None
    assert learned.gru.weight_ih_l0.grad is not None
    assert learned.pivot_projection.weight.grad is not None
    assert learned.role_penalty_raw.grad is not None


def test_frozen_go_nogo_not_adjusted_from_original_negative():
    t=r88.frozen()["evaluation"]["thresholds"]
    assert t["DEV_attention_minus_raw_GRU_min"]==.75
    assert t["DEV_attention_minus_masked_GRU_min"]==.40
    assert t["DEV_attention_minus_shuffled_A_slots_min"]==1.25
    assert t["DEV_attn_minus_best_fixed_min"]==1.25
    assert t["DEV_attn_cold_minus_raw_GRU_min"]==.75
    assert t["DEV_attention_legal_rule_choice_agreement_min"]==.85
    assert t["DEV_attention_oracle_regret_max"]==.70
    assert t["DEV_deterministic_slots_minus_raw_GRU_min"]==1.25
    assert t["DEV_deterministic_slots_minus_shuffled_slots_min"]==1.25
    assert t["OOD_attention_minus_raw_GRU_min"]==.50
    assert t["OOD_attention_cold_minus_raw_GRU_min"]==.50
    assert t["DEV_legal_A_sign_identifiability_min"]==.90
    assert t["DEV_paired_bootstrap_lower95_attention_minus_raw_positive"]
    assert t["independent_scientific_pass"] is False
