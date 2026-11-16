"""R93 exact-SHA TEST-only native exposure/readout-condition regression controls.

Never opens original R93 TRAIN or DEV_TEST in local/non-original G4.
Never replays spent R91 or R92 scientific or diagnostic source.
"""
import copy
import os

import numpy as np
import pytest
import torch

from experiments import r93_headcap_exposure_diagnostic as r93


def test_prospective_immutable_frozen_protocol_and_reuse_denials():
    f=r93.frozen()
    assert r93.r85.git_blob(r93.FREEZE.read_bytes())==r93.FREEZE_BLOB
    assert f["schema"]==r93.SCHEMA
    assert f["parent"]["R92_exact_SHA"]==r93.PARENT
    assert f["parent"]["R92_frozen_blob"]==r93.r92.FREEZE_BLOB
    assert f["parent"]["R92_final"]=="R92_NATIVE_DIVERSITY_LR_NO_GO_DIAGNOSTIC_ONLY_TWO_FRESH_SEEDS"
    assert f["source"]["seeds"]==list(r93.SEEDS)
    assert f["source"]["train_salt"]==r93.TRAIN_SALT
    assert f["source"]["dev_test_salt"]==r93.DEV_SALT
    assert f["source"]["batch_permutation_salt"]==r93.BATCH_SALT
    assert f["source"]["type"]=="FRESH_TEST_ONLY_NONPROTECTED_DIAGNOSTIC"
    assert f["parent"]["no_old_training_weights_or_scored_source_reuse"] is True
    assert f["design"]["fullwidth_native_params"]==16818178
    assert f["frozen_evaluation"]["mandatory_level"]=="w32_n256_e8"
    assert not f["frozen_evaluation"]["independent_scientific_pass"]
    assert f["authority"]["CPU_test_only_diagnostic_allowed"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="CPU_test_only_diagnostic_allowed")
    assert f["execution"]["new_workflow_or_job"] is False


def test_source_admission_denies_previous_and_new_scored_panels_outside_runtime():
    for old in ("R92_DEV_TEST","R91_DEV_TEST","R89_DEV","R85_FIT","OLD"):
        with pytest.raises(PermissionError,match="R93_ALL_SPENT_PROTECTED_AND_OLD_PANELS_DENIED"):
            r93._world(old,r93.SEEDS[0],0,"invalid")
    for panel,index in (("TRAIN",0),("DEV_TEST",1000)):
        with pytest.raises(PermissionError,match="R93_NEW_PANEL_SOURCE_BEFORE_EXACT_SHA_ADMISSION_DENIED"):
            r93._world(panel,r93.SEEDS[0],index,"definitely-not-an-exact-SHA")
    with pytest.raises(PermissionError,match="R93_UNREGISTERED_SEED_OR_WORLD_INDEX"):
        r93._world("TRAIN",42,0,"invalid")
    with pytest.raises(PermissionError,match="R93_UNREGISTERED_SEED_OR_WORLD_INDEX"):
        r93._world("DEV_TEST",r93.SEEDS[0],255,"invalid")
    with pytest.raises(PermissionError,match="R93_UNREGISTERED_SEED_OR_WORLD_INDEX"):
        r93._world("TRAIN",r93.SEEDS[0],256,"invalid")
    if os.getenv("GITHUB_HEAD_REF")!=r93.BRANCH or os.getenv("RUNNER_OS")!="Linux":
        with pytest.raises(PermissionError,match="R93_ONLY_ORIGINAL_CLEAN_HOSTED_LINUX_G4_FIRST_ATTEMPT"):
            r93._admit()


def test_public_causal_rows_selected_from_actual_unit_only_physical_observations():
    for i in range(4):
        w,action=r93.unit_test_world(i)
        x,y=r93.r89._view(w,action)
        slots=r93.r88._public_slots(x)
        assert x.dtype==np.float32 and x.shape==(48,13)
        assert slots.dtype==np.float32 and slots.shape==(5,13)
        np.testing.assert_array_equal(slots[-1],x[-1])
        assert slots[-1,action]==1
        assert set(np.rint(slots[:4,9]*4).astype(np.int64))=={1,2,3,4}
        assert np.all(slots[:4,5]==1) and np.all(slots[:4,11]==1)
        other=copy.deepcopy(w)
        pivot=w["pivot"]
        other["innovations"][pivot+1:pivot+9,0]+=41
        xx,yy=r93.r89._view(other,action)
        np.testing.assert_array_equal(x,xx)
        assert abs(y-yy)>1.0
    with pytest.raises(PermissionError,match="R93_NONFROZEN_UNIT_FIXTURE_INDEX"):
        r93.unit_test_world(4)


def test_exact_epoch_balancing_and_reproducible_matched_arm_batches():
    for seed in r93.SEEDS:
        for n,epochs in ((256,1),(256,8),(16,1),(16,8)):
            p=r93.balanced_batches(seed,n,epochs)
            assert p.shape==(n*epochs//2,2)
            assert np.array_equal(np.bincount(p.ravel(),minlength=n),
                                  np.full(n,epochs))
            np.testing.assert_array_equal(p,r93.balanced_batches(seed,n,epochs))
            for e in range(epochs):
                observed=p.reshape(-1)[e*n:(e+1)*n]
                assert sorted(observed.tolist())==list(range(n))
    with pytest.raises(ValueError,match="R93_OUTSIDE_FROZEN_EXACT_BALANCED_EXPOSURE_DESIGN"):
        r93.balanced_batches(r93.SEEDS[0],64,16)


def test_causal_A_shuffle_preserves_own_actual_chosen_pivot_action_and_role_rows():
    public=[]
    for i in range(4):
        w,act=r93.unit_test_world(i)
        public.append(r93.r88._public_slots(r93.r89._view(w,act)[0]))
    rows=np.stack(public).astype(np.float32)
    shuffled=r93._shuffle_A(rows,r93.SEEDS[0],4)
    np.testing.assert_array_equal(shuffled[:,-1],rows[:,-1])
    assert not np.array_equal(shuffled[:,:4],rows[:,:4])
    np.testing.assert_array_equal(shuffled,r93._shuffle_A(rows,r93.SEEDS[0],4))
    with pytest.raises(ValueError,match="R93_SHUFFLE_EXPECTS_5_PUBLIC_EVENT_ROWS_AND_PAIRED_STRATA"):
        r93._shuffle_A(rows[:3],r93.SEEDS[0],4)


def test_width32_native_headcap_changes_gradients_not_initial_forward_or_weights():
    torch.set_num_threads(1)
    seed=r93.SEEDS[0]
    a=r93._model(r93.HARD,32,seed)
    b=r93._model(r93.HEADCAPPED,32,seed)
    c=r93._model(r93.NO_DEPTH,32,seed)
    aa=dict(a.named_parameters())
    bb=dict(b.named_parameters())
    cc=dict(c.named_parameters())
    assert aa.keys()==bb.keys()
    assert all(torch.equal(aa[k],bb[k]) for k in aa)
    assert all(torch.equal(aa[k],cc[k]) for k in cc)
    rows=[]
    for i in range(4):
        w,act=r93.unit_test_world(i)
        rows.append(r93.r88._public_slots(r93.r89._view(w,act)[0]))
    x=torch.from_numpy(np.stack(rows).astype(np.float32))
    with torch.no_grad():
        torch.testing.assert_close(r93._value(a,x),r93._value(b,x),atol=0,rtol=0)
        assert r93._value(c,x).shape==(4,)
        assert r93._value(r93._model(r93.GRU,32,seed),x).shape==(4,)
    assert len(r93.LEVELS)==5
    assert r93.LEVEL_ARMS[0]==r93.NATIVE
    assert r93.LEVEL_ARMS[1]==r93.ARMS


def test_frozen_double_seed_thresholds_and_dev_only_statuses():
    f=r93.frozen()
    t=f["frozen_evaluation"]["thresholds"]
    assert t["headcap_vs_global_DEV_MSE_relative_gain_min"]==.08
    assert t["headcap_vs_constant_TRAIN_mean_DEV_MSE_relative_gain_min"]==.05
    assert t["headcap_vs_global_TRAIN_MSE_relative_gain_min"]==.10
    assert t["headcap_vs_GRU64_DEV_MSE_relative_gain_min"]==-.10
    assert t["headcap_head_clip_activation_fraction_min"]==.25
    assert t["headcap_head_grad_reduction_fraction_min"]==.20
    assert r93._gain(10,9)==pytest.approx(.1)
    assert r93._gain(10,11)==pytest.approx(-.1)
    assert f["frozen_evaluation"]["both_independent_seeds_required"]
    assert f["training"]["batch"]==2
    assert f["execution"]["official_GPU_first_use"] is False
    assert f["execution"]["protected_science_first_use"] is False


def test_only_original_single_full_width_test_only_diagnostic_in_same_sha_linux_g4(monkeypatch):
    if not (os.getenv("GITHUB_ACTIONS")=="true"
            and os.getenv("GITHUB_HEAD_REF")==r93.BRANCH
            and os.getenv("RUNNER_OS")=="Linux"
            and os.getenv("EXPECTED_CANDIDATE_SHA")==r93.ORIGINAL_R93_SPENT_SHA
            and os.getenv("GITHUB_RUN_ID")==r93.ORIGINAL_R93_SPENT_RUN_ID):
        # Scored R93 DEV_TEST is already consumed at b024f4a / run 43787112857.
        # New SHA engineering qualification MUST NOT re-open those worlds.
        pytest.skip("R93 new diagnostic original first-use only on exact first-attempt Linux G4")
    def denied(*a,**k):
        raise AssertionError("R93_REOPENED_OLD_R89_R91_R92_SOURCE")
    monkeypatch.setattr(r93.r89,"_world",denied)
    monkeypatch.setattr(r93.r89,"_test_world",denied)
    monkeypatch.setattr(r93.r92,"development_ci",denied)
    monkeypatch.setattr(r93.r92.r91,"development_ci",denied)
    result=r93.development_ci()
    assert result["schema"]==r93.SCHEMA
    assert result["frozen_blob"]==r93.FREEZE_BLOB
    assert result["R92_parent_SHA"]==r93.PARENT
    assert result["candidate_SHA"]==os.environ["EXPECTED_CANDIDATE_SHA"]
    assert result["scientific_pass"] is False
    assert not result["protected_or_previous_scored_source_used"]
    assert result["status"] in ("R93_HEAD_GRAD_CONDITIONING_DEV_SIGNAL_ONLY",
                                "R93_EXPOSURE_HEADCAP_DEV_NO_GO",
                                "R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO")
    if result["status"]!="R93_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO":
        assert result["all_arms_frozen_before_DEV_TEST"]
        assert result["DEV_TEST_original_first_use"]
        assert len(result["train_arms"])==len(r93.LEVELS)
        assert set(result["frozen_checks"])==set(map(str,r93.SEEDS))
        for x in result["frozen_checks"].values():
            assert len(x["checks"])==6
            assert all(isinstance(q,bool) for q in x["checks"].values())
            assert "headcap_vs_global_DEV_MSE_relative_gain" in x["scores"]
    else:
        assert not result["DEV_TEST_original_first_use"]
