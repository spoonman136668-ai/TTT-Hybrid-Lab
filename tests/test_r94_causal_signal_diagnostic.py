"""R94 frozen causal public-observation signal vs action-residual objective checks.

Unit tests NEVER open new R94 FIT/VALID/DEV/OOD worlds. R94 fresh source
first use is admitted ONLY in exact SHA branch-scoped Ubuntu CPU science jobs
after dual-OS full G4 regression. Old R93 DEV_TEST is already spent.
"""
import copy
import os

import numpy as np
import pytest
import torch

from experiments import r94_causal_signal_diagnostic as r94


def test_r94_original_prospective_freeze_and_factual_authority():
    f=r94.frozen()
    assert r94.r85.git_blob(r94.FREEZE.read_bytes())==r94.FREEZE_BLOB
    assert f["schema"]==r94.SCHEMA
    assert f["parent"]["R93_engineering_SHA"]==r94.PARENT
    assert f["parent"]["R93_frozen_blob"]==r94.r93.FREEZE_BLOB
    assert f["parent"]["R93_original_scored_SHA"]==r94.r93.ORIGINAL_R93_SPENT_SHA
    assert f["sources"]["seeds"]==list(r94.SEEDS)
    assert f["sources"]["world_counts"]==r94.COUNT
    assert f["arms"]==list(r94.ARMS)
    assert f["training"]["all_arms_steps"]==r94.STEPS
    assert f["training"]["batch"]==r94.BATCH
    assert f["evaluation"]["two_seed_requirement"] is True
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["sources"]["previous_scored_or_reserved_source_access"] is False
    assert not f["parent"]["older_model_weights_or_scored_cohorts_reused"]
    assert f["authority"]["cpu_scoped_test_only_development_allowed"] is True
    assert not any(v for k,v in f["authority"].items()
                   if k!="cpu_scoped_test_only_development_allowed")


def test_pre_first_use_balances_actual_action_pivot_law_index_all_frozen_panels():
    f=r94.frozen()
    counts=r94.balance_preflight()
    for seed in r94.SEEDS:
        for panel,n in r94.COUNT.items():
            groups={}
            for i in range(n):
                key=(i%2,((i%r94.r85.N)//2)%2,r94._action(seed,i))
                groups[key]=groups.get(key,0)+1
            assert len(groups)==8
            assert set(groups.values())=={n//8}
            assert counts[str(seed)][panel]=={"eight_cells":8,"worlds_per_cell":n//8}
            assert n//8==(256 if panel=="FIT" else 32)
    with pytest.raises(PermissionError,match="R94_UNFROZEN_SEED_OR_WORLD_INDEX"):
        r94._action(10,0)


def test_protected_and_spent_sources_denied_outside_exact_first_use_scoped_job():
    for panel in ("R93_DEV_TEST","R92_TRAIN","R89_DEV","R85_QUALIFICATION","OLD"):
        with pytest.raises(PermissionError,match="R94_PRIOR_SPENT_OR_SEALED_PANEL_DENIED"):
            r94._world(panel,r94.SEEDS[0],0,"invalid")
    with pytest.raises(PermissionError,match="R94_NONFROZEN_PANEL_SEED_OR_INDEX"):
        r94._world("FIT",r94.SEEDS[0],r94.COUNT["FIT"],"invalid")
    with pytest.raises(PermissionError,match="R94_NONFROZEN_PANEL_SEED_OR_INDEX"):
        r94._world("DEV",42,0,"invalid")
    for panel in r94.SALT:
        with pytest.raises(PermissionError,match="R94_NEW_SOURCE_NOT_ADMITTED_BEFORE_ORIGINAL_CI"):
            r94._world(panel,r94.SEEDS[0],0,"NOT_ORIGINAL_CANDIDATE_SHA")
    with pytest.raises(PermissionError,match="R94_NO_TRAINING_ON_EVALUATOR_ONLY_PANELS"):
        r94._factual("DEV",r94.SEEDS[0],"invalid")
    with pytest.raises(PermissionError,match="R94_EVALUATOR_CANNOT_LEARN_ON_NEW_SCORED_PANELS"):
        r94._paired("FIT",r94.SEEDS[0],"invalid")
    if os.getenv("GITHUB_JOB")!="r94-causal-signal":
        with pytest.raises(PermissionError,match="R94_ORIGINAL_FIRST_ATTEMPT_SCOPED_G4_HOSTED_JOB_REQUIRED"):
            r94._admit(r94.SEEDS[0])


@pytest.mark.parametrize("i",[0,1,2,3])
def test_legally_observed_A_feature_composition_and_action_are_only_public(i):
    world,action=r94.unit_only_world(i)
    other=copy.deepcopy(world)
    x,r=r94.r86._view(world,action)
    x0,r0=r94.r86._view(world,r94.r85.CALIBRATE)
    x2,r2=r94.r86._view(world,r94.r85.EXPLOIT_POS)
    assert x.shape==(48,13) and x.dtype==np.float32
    np.testing.assert_array_equal(x0[:-1],x2[:-1])
    np.testing.assert_array_equal(x0[-1,4:],x2[-1,4:])
    raw,composed,sign=r94._features(x)
    assert raw.shape==(8,) and composed.shape==(8,)
    assert sign in (-1.,1.)
    assert sign==float(np.prod(np.where(raw[:4]>0,1.,-1.)))
    assert composed[0]==sign and np.array_equal(composed[1:4],np.zeros(3,np.float32))
    np.testing.assert_array_equal(composed[4:],raw[4:])
    assert (raw[4],raw[5]) == ((1.,0.) if action==r94.r85.CALIBRATE else (0.,1.))
    assert np.isfinite(r) and np.isfinite(r0) and np.isfinite(r2)
    other["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=30.
    changed,r_new=r94.r86._view(other,action)
    np.testing.assert_array_equal(x,changed)
    assert abs(r_new-r)>1.


def test_train_only_action_mean_is_selected_by_slot_four_not_by_parity():
    f=np.zeros((4,8),np.float32)
    f[:,0]=np.asarray([1,-1,-1,1],np.float32)
    f[:2,4]=1.;f[2:,5]=1.
    baseline={r94.r85.CALIBRATE:9.,r94.r85.EXPLOIT_POS:-3.}
    np.testing.assert_array_equal(r94._offset(f,baseline),
                                  np.asarray([9,9,-3,-3],np.float32))
    comp=f.copy()
    comp[:,0]*=-1
    np.testing.assert_array_equal(r94._offset(comp,baseline),
                                  np.asarray([9,9,-3,-3],np.float32))


def test_shuffled_causal_sign_only_preserves_action_and_timing_without_dev_access():
    n=r94.COUNT["FIT_VALIDATION"]
    i=np.arange(n)
    pivot=(i%2).astype(np.float32)
    chosen=np.asarray([r94._action(r94.SEEDS[0],int(v)) for v in i],np.int8)
    move=r94._shuffle_index(pivot,chosen,r94.SEEDS[0],"FIT_VALIDATION")
    np.testing.assert_array_equal(pivot[move],pivot)
    np.testing.assert_array_equal(chosen[move],chosen)
    assert np.any(move!=i)
    np.testing.assert_array_equal(move,
          r94._shuffle_index(pivot,chosen,r94.SEEDS[0],"FIT_VALIDATION"))
    with pytest.raises(PermissionError,match="R94_SHUFFLED_PRIOR_OR_SEALED_DATASET_REFUSED"):
        r94._shuffle_index(pivot,chosen,r94.SEEDS[0],"R93_DEV_TEST")


def test_same_initialized_trained_conventional_MLP_capacity_and_prediction():
    torch.set_num_threads(1)
    models=r94._models(r94.SEEDS[0])
    assert list(models)==list(r94.ARMS)
    all_params=[sum(p.numel() for p in m.parameters()) for m in models.values()]
    assert all_params==[4801]*len(r94.ARMS)
    base=dict(models[r94.RAW_DIRECT].named_parameters())
    for m in models.values():
        current=dict(m.named_parameters())
        assert base.keys()==current.keys()
        assert all(torch.equal(base[k],current[k]) for k in base)
    samples=[]
    for i in range(4):
        w,a=r94.unit_only_world(i)
        samples.append(r94._features(r94.r86._view(w,a)[0])[0])
    rows=np.stack(samples)
    for m in models.values():
        out=r94._predict(m,rows)
        assert out.shape==(4,) and bool(np.isfinite(out).all())


def test_frozen_two_seeds_feature_vs_objective_gates_and_stop_states():
    f=r94.frozen()
    feat=f["evaluation"]["feature_checks"]
    obj=f["evaluation"]["objective_checks"]
    assert feat["composed_residual_minus_shuffled_DEV_reward_min"]==.75
    assert feat["composed_residual_minus_best_fixed_DEV_reward_min"]==1.25
    assert feat["composed_residual_oracle_DEV_regret_max"]==.75
    assert feat["composed_residual_legal_rule_DEV_action_agreement_min"]==.85
    assert feat["composed_residual_minus_shuffled_bootstrap_lower95_min"]==.10
    assert feat["composed_residual_minus_shuffled_shifted_OOD_reward_min"]==.50
    assert feat["composed_residual_minus_shuffled_shifted_OOD_cold_reward_min"]==.50
    assert obj["composed_residual_minus_composed_direct_DEV_reward_min"]==.15
    assert obj["composed_residual_minus_raw_residual_DEV_reward_min"]==.15
    assert obj["composed_residual_minus_composed_direct_shifted_OOD_reward_min"]==.10
    assert f["training"]["required_all_5_arms_full_updates"] is True
    assert f["execution"]["G4_full_Linux_Windows_regression_required_before_new_scoring"] is True
    assert f["execution"]["new_workflow_file"] is False
    assert f["execution"]["no_GPU_first_use"] is True


def test_original_first_use_source_is_not_run_as_part_of_full_regression():
    # Original new test-source training belongs exclusively to branch-scoped
    # downstream gated G4 seed jobs, NOT the reusable full G4 pytest suite.
    assert r94.BRANCH=="research/r94-causal-feature-residual-value-diagnostic-20261115"
    assert r94.BATCH==16 and r94.STEPS==4096
    assert r94.COUNT["DEV"]==256 and r94.COUNT["OOD_SHIFT"]==256
    with pytest.raises(PermissionError,match="R94_ONLY_INDEPENDENT_UNIT_FIXTURES"):
        r94.unit_only_world(5)


def test_original_r94_testonly_scored_sources_are_permanently_nonreplayable(monkeypatch):
    assert r94.ORIGINAL_R94_SPENT_SHA=="1ea70bc3b06872e737238cc94bc3efecdd3debe3"
    assert r94.ORIGINAL_R94_SPENT_RUN_ID=="43789681024"
    env={
      "GITHUB_ACTIONS":"true",
      "GITHUB_EVENT_NAME":"pull_request",
      "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
      "GITHUB_HEAD_REF":r94.BRANCH,
      "GITHUB_JOB":"r94-causal-signal",
      "R94_SEED":str(r94.SEEDS[0]),
      "RUNNER_OS":"Linux",
      "RUNNER_ENVIRONMENT":"github-hosted",
      "GITHUB_RUN_ATTEMPT":"1",
      "GITHUB_REF":"refs/pull/290/merge",
      "EXPECTED_CANDIDATE_SHA":"f"*40,
      "GITHUB_RUN_ID":"43799999999",
    }
    for key,value in env.items():
        monkeypatch.setenv(key,value)
    with pytest.raises(PermissionError,match="R94_ALREADY_CONSUMED_FIRST_USE_DENIED"):
        r94._admit(r94.SEEDS[0])
    with pytest.raises(PermissionError,match="R94_NEW_SOURCE_NOT_ADMITTED_BEFORE_ORIGINAL_CI"):
        r94._world("FIT",r94.SEEDS[0],0,"f"*40)
