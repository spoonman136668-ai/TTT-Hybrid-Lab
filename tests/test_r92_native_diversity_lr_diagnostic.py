"""R92: resource-bounded 32/1024-width TEST-only native optimizer diagnostic.

No R91 scored-world replay; all new R92 original TRAIN/DEV_TEST cohorts
are hard-denied from ordinary pytest and non-original CI branches.
"""
import copy
import os

import numpy as np
import pytest
import torch

from experiments import r92_native_diversity_lr_diagnostic as r92


def test_immutable_prospective_R92_contract_and_historical_source_fences():
    f=r92.frozen()
    assert r92.r74.git_blob(r92.FREEZE.read_bytes())==r92.FREEZE_BLOB
    assert f["schema"]==r92.SCHEMA
    assert f["parent"]["R91_sha"]==r92.PARENT
    assert f["parent"]["R91_freeze_blob"]==r92.r91.FREEZE_BLOB
    assert f["parent"]["R91_original_status"]=="R91_SOFT_ROUTING_DEV_NO_GO_BOTH_OS"
    assert f["source"]["seeds"]==list(r92.SEEDS)
    assert r92.TRAIN_SALT==25330111 and r92.DEV_SALT==25330211
    assert f["design"]["world_counts_by_width"]=={"1024":[256,16],"32":[256,64,16,4]}
    assert f["design"]["model_arms"]==list(r92.ARMS)
    assert f["training"]["native_updates_per_arm"]==96
    assert f["training"]["GRU_updates_per_arm"]==768
    assert f["analysis"]["frozen_thresholds"]["replication_required_both_seeds"]
    assert not f["analysis"]["frozen_thresholds"]["independent_scientific_pass"]
    assert f["execution"]["R91_original_DEV_TEST_replay_denied_outside_original_R91_branch"]
    assert f["authority"]["CPU_test_only_development_allowed"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="CPU_test_only_development_allowed")


def test_original_train_and_test_salts_not_admitted_in_normal_or_unrelated_G4():
    with pytest.raises(PermissionError,match="R92_ORIGINAL_EXACT_HEAD_LINUX_G4_FIRST_ATTEMPT_ONLY"):
        if os.getenv("GITHUB_HEAD_REF")!=r92.BRANCH:
            r92._admit()
        else:
            # The original R92 branch runs only in the original CI G4.
            pytest.skip("Original admission is tested through the full diagnostic")
    for panel in ("R89_DEV","R91_DEV_TEST","R85_FIT","OLD"):
        with pytest.raises(PermissionError,match="R92_PROTECTED_OR_PRIOR_TEST_SOURCE_DENIED"):
            r92._world(panel,r92.SEEDS[0],0,"fake")
    for panel,index in (("TRAIN",0),("DEV_TEST",1000)):
        with pytest.raises(PermissionError,match="R92_ORIGINAL_NEW_TEST_SOURCE_NOT_ADMITTED"):
            r92._world(panel,r92.SEEDS[0],index,"fake")
    with pytest.raises(PermissionError,match="R92_NONFROZEN_NEW_TEST_INDEX_OR_SEED"):
        r92._world("TRAIN",r92.SEEDS[0],257,"fake")
    with pytest.raises(PermissionError,match="R92_NONFROZEN_NEW_TEST_INDEX_OR_SEED"):
        r92._world("DEV_TEST",r92.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R92_NONFROZEN_NEW_TEST_INDEX_OR_SEED"):
        r92._world("TRAIN",1,0,"fake")


def test_legal_test_only_public_events_and_future_effects_never_leak():
    for idx in (0,1,2,3):
        world,chosen=r92._test_only_physics_for_unit_tests(idx)
        view,reward=r92.r89._view(world,chosen)
        assert view.shape==(48,13) and view.dtype==np.float32
        slots=r92.r88._public_slots(view)
        assert slots.shape==(5,13)
        np.testing.assert_array_equal(slots[-1],view[-1])
        assert slots[-1,chosen]==1
        assert set((slots[:4,9]*4).round().astype(int).tolist())=={1,2,3,4}
        assert np.all(slots[:4,5]==1) and np.all(slots[:4,11]==1)
        shifted=copy.deepcopy(world)
        shifted["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=35
        revised,other_reward=r92.r89._view(shifted,chosen)
        np.testing.assert_array_equal(revised,view)
        assert abs(other_reward-reward)>1
    with pytest.raises(PermissionError,match="R92_NONFROZEN_NONSCIENTIFIC_UNIT_TEST_WORLD"):
        r92._test_only_physics_for_unit_tests(5)


def test_model_initialization_exact_native_width32_and_negative_shuffle_preserves_action():
    torch.set_num_threads(1)
    seed=r92.SEEDS[0]
    a=r92._model(r92.ARMS[0],32,seed)
    b=r92._model(r92.ARMS[1],32,seed)
    c=r92._model(r92.ARMS[2],32,seed)
    pa,pb,pc=(dict(m.named_parameters()) for m in (a,b,c))
    assert pa.keys()==pb.keys()==pc.keys()
    assert all(torch.equal(pa[k],pb[k]) and torch.equal(pa[k],pc[k]) for k in pa)
    rows=[]
    for i in range(4):
        w,action=r92._test_only_physics_for_unit_tests(i)
        rows.append(r92.r88._public_slots(r92.r89._view(w,action)[0]))
    data=np.stack(rows).astype(np.float32)
    with torch.no_grad():
        for model in (a,b,c,r92._model(r92.GRUS[0],32,seed)):
            v=r92._value(model,torch.from_numpy(data))
            assert v.shape==(4,) and bool(torch.isfinite(v).all())
    shuffled=r92._shuffle(data,seed,4)
    np.testing.assert_array_equal(shuffled[:,-1],data[:,-1])
    assert not np.array_equal(shuffled[:,:4],data[:,:4])
    np.testing.assert_array_equal(shuffled,r92._shuffle(data,seed,4))
    with pytest.raises(ValueError,match="R92_SHUFFLE_REQUIRES_PUBLIC_EVENT_INPUT"):
        r92._shuffle(data[:3],seed,4)


def test_frozen_comparison_and_learning_rate_controls_remain_separate_from_science():
    f=r92.frozen()
    assert f["analysis"]["frozen_thresholds"]["width1024_world256_lowLR_hard_vs_highLR_hard_DEV_factual_MSE_relative_gain_min"]==.05
    assert f["analysis"]["frozen_thresholds"]["width1024_world256_lowLR_hard_vs_constant_DEV_factual_MSE_relative_gain_min"]==.05
    assert f["analysis"]["frozen_thresholds"]["width1024_world256_lowLR_hard_train_MSE_relative_gain_min"]==.10
    assert f["analysis"]["frozen_thresholds"]["width1024_world256_softLR_vs_hardLR_DEV_factual_MSE_relative_gain_min"]==.05
    assert r92.STEPS[r92.NATIVE[0]]==r92.STEPS[r92.NATIVE[2]]==96
    assert r92.STEPS[r92.GRUS[0]]==r92.STEPS[r92.GRUS[1]]==768
    assert r92._relative_gain(10.,9.)==pytest.approx(.1)
    assert r92._relative_gain(10.,11.)==pytest.approx(-.1)
    assert f["execution"]["new_workflow_or_job_created"] is False
    assert f["design"]["persistent_model_checkpoints"] is False
    assert f["execution"]["no_GPU_first_use"]


def test_original_single_first_use_on_exact_G4_Linux_only(monkeypatch):
    if not (os.getenv("GITHUB_ACTIONS")=="true"
            and os.getenv("GITHUB_HEAD_REF")==r92.BRANCH
            and os.getenv("RUNNER_OS")=="Linux"):
        pytest.skip("Original frozen R92 test-only high-first run only on exact Linux G4")
    def deny(*args,**kwargs):
        raise AssertionError("R92_REUSED_OLD_R89_SPENT_OR_PROTECTED_SOURCE")
    monkeypatch.setattr(r92.r89,"_world",deny)
    monkeypatch.setattr(r92.r89,"_test_world",deny)
    monkeypatch.setattr(r92.r91,"development_ci",deny)
    result=r92.development_ci()
    assert result["schema"]==r92.SCHEMA
    assert result["freeze_blob"]==r92.FREEZE_BLOB
    assert result["candidate_SHA"]==os.environ["EXPECTED_CANDIDATE_SHA"]
    assert result["parent_R91_SHA"]==r92.PARENT
    assert result["scientific_pass"] is False
    assert not result["protected_or_previous_DEV_OOD_access"]
    assert result["status"] in (
        "R92_NATIVE_DIVERSITY_LR_DEV_SIGNAL_ONLY",
        "R92_NATIVE_DIVERSITY_LR_NO_GO_DIAGNOSTIC_ONLY",
        "R92_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO")
    if result["status"]!="R92_INCOMPLETE_RESOURCE_NOT_SCIENTIFIC_NO_GO":
        assert result["DEV_TEST_FIRST_USE_AFTER_ALL_TRAINING"]
        assert len(result["completed"])==len(r92.LEVELS)
        assert set(result["comparison_checks"])==set(map(str,r92.SEEDS))
        for item in result["comparison_checks"].values():
            assert len(item["checks"])==4
            assert all(isinstance(v,bool) for v in item["checks"].values())
    else:
        assert not result["DEV_TEST_FIRST_USE_AFTER_ALL_TRAINING"]
