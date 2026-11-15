"""R89 full-regression CPU tests. All examples come from TEST-only source salt.

Never generate R89 original FIT/VALID/DEV/OOD in pytest; verify custody
admission denies all scientific sources before original qualified job.
"""
import copy
import numpy as np
import pytest
import torch

from experiments import r89_native_event_access as r89


def view(index, action=None):
    w=r89._test_world(index)
    a=r89.r85.CALIBRATE if action is None else action
    return r89._view(w,a)[0]


def test_immutable_freeze_original_and_no_previous_data_or_promotion():
    f=r89.frozen()
    assert r89.r85.git_blob(r89.FREEZE.read_bytes()) == r89.FREEZE_BLOB
    assert f["parent"]["R88_qualified_head"] == r89.PARENT
    assert f["parent"]["R88_freeze_blob"] == r89.r88.FREEZE_BLOB
    assert f["parent"]["R88_status"] == "R88_LEARNED_ROLE_EVENT_READOUT_DIAGNOSTIC_SIGNAL_ONLY_BOTH_SEEDS_13_OF_13"
    assert f["sources"]["seeds"]==list(r89.SEEDS)
    assert f["sources"]["independent_salts"]==r89.SALT
    spent=set(f["parent"]["R88_spent_salts"])|set(f["parent"]["R87_spent"])
    spent|=set(f["parent"]["R86_spent"])|set(f["parent"]["R85_still_sealed"])
    spent.add(f["parent"]["R85_CALIBRATION_spent"])
    assert set(r89.SALT.values()).isdisjoint(spent)
    assert f["arms"]==list(r89.ARMS)
    assert f["training"]["original_native_core_parameter_count"]==16818178
    assert f["training"]["native_updates_each"]==1024
    assert f["training"]["GRU64_and_GRU256_updates_each"]==8192
    assert f["training"]["first_1024_identical_seeded_FIT_index_pairs_all_arms"]
    assert not f["evaluation"]["thresholds"]["independent_scientific_pass"]
    assert f["authority"]["cpu_scoped_factual_training_allowed_without_merge"]
    assert not any(value for key,value in f["authority"].items()
                   if key!="cpu_scoped_factual_training_allowed_without_merge")


def test_all_original_sources_strictly_denied_outside_exact_science_job():
    seed=r89.SEEDS[0]
    for old in ("R88_DEV","R87_FIT","R86_OOD","R85_CALIBRATION","OLD"):
        with pytest.raises(PermissionError,match="R89_OLD_SPENT_OR_SEALED_PANEL_DENIED"):
            r89._world(old,seed,0)
    for panel in r89.SALT:
        with pytest.raises(PermissionError,match="R89_ORIGINAL_FIRST_USE_SOURCE_NOT_ADMITTED"):
            r89._world(panel,seed,0)
    with pytest.raises(PermissionError,match="R89_NONFROZEN_SEED_OR_INDEX"):
        r89._world("FIT",7,0)
    with pytest.raises(PermissionError,match="R89_NONFROZEN_SEED_OR_INDEX"):
        r89._world("DEV",seed,r89.COUNT["DEV"])
    with pytest.raises(PermissionError,match="R89_NO_SCORED_PANEL_IN_FIT"):
        r89._factual("DEV",seed,custody_sha="never")
    with pytest.raises(PermissionError,match="R89_SCORING_ONLY_IN_NEW_DEV_OOD"):
        r89._scored("FIT",seed,custody_sha="never")
    with pytest.raises(PermissionError,match="R89_NOT_ORIGINAL_FIRST_ATTEMPT_CLEAN_HOSTED_JOB"):
        r89._admit(seed)


@pytest.mark.parametrize("index",[0,1,2,3,10,11])
def test_selected_public_probe_roles_and_causal_intervention_nonleak(index):
    w=r89._test_world(index)
    xa,ra=r89._view(w,r89.r85.CALIBRATE)
    xb,rb=r89._view(w,r89.r85.EXPLOIT_POS)
    assert xa.shape==xb.shape==(48,13)
    np.testing.assert_array_equal(xa[:-1],xb[:-1])
    np.testing.assert_array_equal(xa[-1,4:],xb[-1,4:])
    a,b=r89._slots(xa),r89._slots(xb)
    assert a.shape==b.shape==(5,13)
    np.testing.assert_array_equal(a[:-1],b[:-1])
    np.testing.assert_array_equal(a[-1],xa[-1])
    np.testing.assert_array_equal(b[-1],xb[-1])
    assert a[-1,0]==1 and b[-1,2]==1
    assert ra!=rb
    assert set(np.round(a[:4,9]*4).astype(int).tolist())=={1,2,3,4}
    assert np.all(a[:4,5]==1)
    assert np.all(a[:4,11]==1)
    assert np.all(a[:4,12]<xa[-1,12])
    f0,f2=r89.r87._legal_features(xa),r89.r87._legal_features(xb)
    assert f0[0].shape==f2[0].shape==(8,)
    assert f0[1] in (-1.0,1.0)
    assert f0[1]==f2[1]
    np.testing.assert_array_equal(f0[0][:4],f2[0][:4])


def test_future_action_consequences_not_public_before_decision():
    for idx in (0,1,2,3):
        world=r89._test_world(idx)
        changed=copy.deepcopy(world)
        pivot=world["pivot"]
        changed["innovations"][pivot+1:pivot+9,0]+=50
        for action in (r89.r85.CALIBRATE,r89.r85.EXPLOIT_POS):
            x0,v0=r89._view(world,action)
            x1,v1=r89._view(changed,action)
            np.testing.assert_array_equal(x0,x1)
            np.testing.assert_array_equal(r89._slots(x0),r89._slots(x1))
            assert abs(v0-v1)>1


def test_original_native_max_width_hard_router_and_identical_core_tensors():
    torch.set_num_threads(1)
    ms=r89._models(r89.SEEDS[0])
    assert set(ms)==set(r89.ARMS)
    raw=ms[r89.RAW]
    hard=ms[r89.HARD]
    learned=ms[r89.LEARN]
    assert isinstance(learned,r89.NativeLearnedRoleEventAccess)
    p0=dict(raw.named_parameters())
    p1=dict(hard.named_parameters())
    p2=dict(learned.native.named_parameters())
    assert sum(p.numel() for p in raw.parameters())==16818178
    assert sum(p.numel() for p in hard.parameters())==16818178
    assert sum(p.numel() for p in learned.native.parameters())==16818178
    assert p0.keys()==p1.keys()==p2.keys()
    assert all(torch.equal(p0[k],p1[k]) and torch.equal(p0[k],p2[k])
               for k in p0)
    assert sum(p.numel() for p in learned.parameters())>16818178
    g=[sum(p.numel() for p in ms[k].parameters())
       for k in (r89.GRU_RAW,r89.GRU_HARD,r89.GRU_SHUFFLE)]
    assert len(set(g))==1
    assert sum(p.numel() for p in ms[r89.GRU256].parameters())>g[0]


def test_trainable_native_role_query_gradients_and_no_privileged_feature():
    torch.set_num_threads(1)
    native=r89._models(r89.SEEDS[0])[r89.LEARN]
    x=torch.from_numpy(np.stack([view(0),view(1)]))
    slots=native.legal_rows(x)
    assert slots.shape==(2,5,13)
    np.testing.assert_array_equal(slots.detach().numpy()[:,-1],x.numpy()[:,-1])
    assert torch.all(slots[:,:4,11]>.4)
    values=native(x)
    assert values.shape==(2,5,1)
    loss=values[:,-1,0].square().mean()
    loss.backward()
    assert native.query.grad is not None
    assert native.key[0].weight.grad is not None
    assert native.role_strength_raw.grad is not None
    assert native.native.head.weight.grad is not None if hasattr(native.native,"head") else any(
        p.grad is not None for p in native.native.parameters())
    assert bool(torch.isfinite(native.query.grad).all())
    assert bool(torch.isfinite(native.key[0].weight.grad).all())
    assert bool(torch.isfinite(native.role_strength_raw.grad).all())


def test_synthetic_shuffled_role_slots_deterministic_stratum_and_own_action():
    n=r89.COUNT["FIT"]
    x=np.zeros((n,5,13),np.float32)
    x[:,4,0]=1
    x[::2,4,12]=64/223
    x[1::2,4,12]=192/223
    x[:,:4,10]=np.arange(n,dtype=np.float32)[:,None]
    sh=r89._shuffle(x,r89.SEEDS[0],"FIT")
    np.testing.assert_array_equal(sh,r89._shuffle(x,r89.SEEDS[0],"FIT"))
    assert not np.array_equal(sh[:,:4],x[:,:4])
    np.testing.assert_array_equal(sh[:,-1],x[:,-1])
    for offset in (0,1):
        group=np.arange(offset,n,2)
        assert sorted(sh[group,0,10].tolist())==sorted(x[group,0,10].tolist())
    with pytest.raises(PermissionError,match="R89_SHUFFLE_ONLY_FROZEN_PANEL_GEOMETRY"):
        r89._shuffle(x,r89.SEEDS[0],"OLD")


def test_exact_input_geometry_full_native_control_and_action_value_arms():
    torch.set_num_threads(1)
    ms=r89._models(r89.SEEDS[0])
    public=np.stack([view(0),view(1)])
    event=np.stack([r89._slots(x) for x in public])
    cues=np.stack([r89.r87._legal_features(x)[0] for x in public])
    for arm in r89.ARMS:
        x=cues if arm==r89.MLP else (
            event if arm in (r89.HARD,r89.GRU_HARD,r89.GRU_SHUFFLE,r89.GRU256)
            else public)
        with torch.no_grad():
            value=r89._prediction(ms[arm],torch.from_numpy(x),arm)
        assert value.shape==(2,) and bool(torch.isfinite(value).all())


def test_original_precommitted_science_thresholds_and_no_positive_relabel():
    f=r89.frozen()
    t=f["evaluation"]["thresholds"]
    assert t["DEV_learned_native_minus_raw_native_min"]==1.
    assert t["DEV_learned_native_minus_native_deterministic_slot_min"]==-.25
    assert t["DEV_learned_native_minus_best_trained_slot_GRU_min"]==-.35
    assert t["DEV_learned_native_minus_shuffled_slot_GRU_min"]==1.25
    assert t["DEV_learned_native_minus_best_fixed_min"]==1.25
    assert t["DEV_cold_native_learned_minus_raw_min"]==1.
    assert t["DEV_lawful_parity_rule_action_agreement_min"]==.85
    assert t["DEV_learned_native_oracle_regret_max"]==.75
    assert t["OOD_native_learned_minus_raw_min"]==.75
    assert t["OOD_cold_native_learned_minus_raw_min"]==.75
    assert t["OOD_native_learned_minus_trained_GRU256_slots_min"]==-.50
    assert t["DEV_public_A_sign_identifiability_min"]==.90
    assert t["DEV_bootstrap_lower95_native_learned_minus_raw_positive"]
    assert t["independent_scientific_pass"] is False
