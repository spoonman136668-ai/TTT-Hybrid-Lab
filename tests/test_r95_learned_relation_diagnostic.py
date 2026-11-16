"""R95 raw-cue learned relation: immutable source, implementation and negative controls.

No original R95 FIT/VALID/DEV/OOD is opened by pytest. Only admitted,
post-dual-OS first-attempt original seed jobs may start source first use.
"""
import copy
import os

import numpy as np
import pytest
import torch
from torch import nn

from experiments import r95_learned_relation_diagnostic as r95


def test_prospective_frozen_R95_and_authority():
    f=r95.frozen()
    assert r95.r85.git_blob(r95.FREEZE.read_bytes())==r95.FREEZE_BLOB
    assert f["schema"]==r95.SCHEMA
    assert f["stage"]=="PROSPECTIVE_FROZEN_BEFORE_R95_IMPLEMENTATION_TESTS_CI_OR_NEW_SOURCE_FIRST_USE"
    assert f["parent"]["R94_final_source_sha"]==r95.PARENT
    assert f["parent"]["R94_freeze_blob"]==r95.r94.FREEZE_BLOB
    assert f["parent"]["R94_original_result"]=="R94_PUBLIC_CAUSAL_FEATURE_SIGNAL_ONLY"
    assert f["source"]["salts"]=={**r95.SALT,"SHUFFLE":r95.SHUFFLE_SALT,
        "BATCHES":r95.BATCH_SALT,"BOOTSTRAP":r95.BOOTSTRAP_SALT,"UNIT_ONLY":r95.UNIT_SALT}
    assert f["source"]["seeds"]==list(r95.SEEDS)
    assert f["source"]["world_counts"]==r95.COUNT
    assert f["arms"]==list(r95.ARMS)
    assert f["training"]["per_arm_steps"]==r95.STEPS
    assert f["training"]["batch"]==r95.BATCH
    assert f["execution"]["two_new_seed_jobs_in_existing_G4_only"]
    assert f["architecture"]["full_native_width1024_not_claimed"]
    assert f["architecture"]["isolated_experimental_additive_sidecar_only"]
    assert f["architecture"]["no_structural_change_to_main_native"]
    assert f["evaluation"]["independent_scientific_pass"] is False
    assert f["authority"]["scoped_cpu_testonly_diagnostic_allowed"]
    assert not any(v for k,v in f["authority"].items()
                   if k!="scoped_cpu_testonly_diagnostic_allowed")


def test_balanced_public_training_action_vs_pivot_and_unseen_source_law_strata():
    summary=r95.balance_preflight()
    for seed in r95.SEEDS:
        for panel,n in r95.COUNT.items():
            counts={}
            for i in range(n):
                key=(i%2,((i%r95.r85.N)//2)%2,r95._chosen(seed,i))
                counts[key]=counts.get(key,0)+1
            assert len(counts)==8 and set(counts.values())=={n//8}
            assert summary[str(seed)][panel]=={"joint_cells":8,"worlds_per_cell":n//8}
    with pytest.raises(PermissionError,match="R95_OUTSIDE_FROZEN_SELECTED_ACTION"):
        r95._chosen(2,0)


def test_old_and_new_source_first_use_denied_in_all_ordinary_regressions():
    for p in ("R94_DEV","R93_DEV_TEST","R89_FIT","R85_QUALIFICATION","OLD"):
        with pytest.raises(PermissionError,match="R95_PREVIOUS_OR_PROTECTED_PANEL_DENIED"):
            r95._world(p,r95.SEEDS[0],0,"fake")
    with pytest.raises(PermissionError,match="R95_NONFROZEN_SEED_OR_PANEL_INDEX"):
        r95._world("FIT",42,0,"fake")
    with pytest.raises(PermissionError,match="R95_NONFROZEN_SEED_OR_PANEL_INDEX"):
        r95._world("DEV",r95.SEEDS[0],256,"fake")
    for p in r95.SALT:
        with pytest.raises(PermissionError,match="R95_SOURCE_NOT_ADMITTED_FOR_ORIGINAL_SCIENCE"):
            r95._world(p,r95.SEEDS[0],0,"not-valid-SHA")
    with pytest.raises(PermissionError,match="R95_TRAINING_CANNOT_OPEN_SCORED_DEV_OOD"):
        r95._factual("DEV",r95.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R95_EVALUATOR_ONLY_FRESH_DEV_OR_SHIFTED_OOD"):
        r95._paired("FIT",r95.SEEDS[0],"fake")
    with pytest.raises(PermissionError,match="R95_ORIGINAL_HOSTED_FIRST_ATTEMPT_G4_JOB_REQUIRED"):
        r95._admit(r95.SEEDS[0])


@pytest.mark.parametrize("index",[0,1,2,3])
def test_public_cues_are_raw_noisy_role_observations_not_calculated_private_parity(index):
    world,act=r95.unit_world(index)
    prefix,return_value=r95.r86._view(world,act)
    assert prefix.shape==(48,13) and prefix.dtype==np.float32
    slots=r95.r88._public_slots(prefix)
    features=r95._raw_from_slots(slots)
    assert slots.shape==(5,13)
    assert features.shape==(8,) and features.dtype==np.float32
    np.testing.assert_array_equal(features[:4],slots[:4,10])
    np.testing.assert_array_equal(features[4:7],np.array([
        float(act==r95.r85.CALIBRATE),float(act==r95.r85.EXPLOIT_POS),
        float(slots[-1,12])],np.float32))
    assert features[7]==0.
    upper=r95._upper(features[None,:])[0]
    assert upper[0] in (-1.,1.)
    assert upper[0]==np.where(features[:4]>0,1.,-1.).prod()
    assert np.all(upper[1:4]==0)
    np.testing.assert_array_equal(upper[4:],features[4:])
    assert np.isfinite(return_value)
    # A future-only physical intervention can change factual return but never
    # a lawfully observable pre-intervention cue/action prefix.
    shifted=copy.deepcopy(world)
    shifted["innovations"][world["pivot"]+1:world["pivot"]+9,0]+=35
    new_prefix,new_reward=r95.r86._view(shifted,act)
    np.testing.assert_array_equal(new_prefix,prefix)
    assert abs(return_value-new_reward)>1.


def test_unit_source_is_not_original_training_source():
    with pytest.raises(PermissionError,match="R95_ONLY_UNIT_NONSCORING_SOURCE"):
        r95.unit_world(4)


def test_learnable_pair_merger_has_no_explicit_parity_target_or_zero_gradients():
    torch.set_num_threads(1)
    model=r95.LearnedPairTree()
    f=torch.tensor([[.5,-.6,.7,-.8,1.,0.,.5,0.],
                    [-.4,.2,-1.1,.6,0.,1.,.9,0.]],dtype=torch.float32)
    y=model(f)
    assert y.shape==(2,) and bool(torch.isfinite(y).all())
    assert all(isinstance(x,nn.Module) for x in [model.pair,model.merge,model.readout])
    loss=((y-torch.tensor([3.,-3.]))**2).mean()
    loss.backward()
    assert model.pair[0].weight.grad is not None
    assert float(model.pair[0].weight.grad.abs().sum())>0
    assert model.merge[0].weight.grad is not None
    assert float(model.merge[0].weight.grad.abs().sum())>0
    # Permuting four roles does not change parity. This task therefore cannot
    # establish role-identity learning or discover genuinely novel relations.
    assert r95.frozen()["null"].find("commutative")>=0


def test_three_native_initial_original_cores_are_equal_and_sidecar_is_isolated():
    torch.set_num_threads(1)
    models=r95._models(r95.SEEDS[0])
    assert set(models)==set(r95.ARMS)  # training order is frozen by explicit ARMS iteration
    a,b,c=models[r95.NATIVE_RAW],models[r95.NATIVE_FLAT].native,models[r95.NATIVE_TREE].native
    pa,pb,pc=[dict(m.named_parameters()) for m in (a,b,c)]
    assert pa.keys()==pb.keys()==pc.keys()
    assert all(torch.equal(pa[k],pb[k]) and torch.equal(pa[k],pc[k]) for k in pa)
    assert sum(x.numel() for x in models[r95.MLP_RAW].parameters())==4801
    assert sum(x.numel() for x in models[r95.UPPER].parameters())==4801
    assert sum(x.numel() for x in models[r95.GRU].parameters())==15233
    slots=[];raw=[]
    for i in range(4):
        w,act=r95.unit_world(i)
        selected=r95.r88._public_slots(r95.r86._view(w,act)[0])
        slots.append(selected)
        raw.append(r95._raw_from_slots(selected))
    x=torch.from_numpy(np.stack(slots))
    f=torch.from_numpy(np.stack(raw))
    with torch.no_grad():
        for arm in r95.ARMS:
            if arm==r95.UPPER:
                inp=torch.from_numpy(r95._upper(f.numpy()))
            else:
                inp=f
            result=r95._predict_tensor(models[arm],arm,x,inp)
            assert result.shape==(4,)
            assert bool(torch.isfinite(result).all())


def test_shuffle_negative_preserves_actual_current_action_and_time():
    panel="FIT_VALIDATION"
    seed=r95.SEEDS[0]
    n=r95.COUNT[panel]
    arr=np.zeros((n,8),np.float32)
    arr[:,6]=np.where(np.arange(n)%2==0,.3,.9)
    choices=np.asarray([r95._chosen(seed,int(i)) for i in range(n)],np.int8)
    arr[:,4]=(choices==r95.r85.CALIBRATE)
    arr[:,5]=(choices==r95.r85.EXPLOIT_POS)
    index=r95._shuffle_index(arr,seed,panel,choices)
    assert (index!=np.arange(n)).all()
    np.testing.assert_array_equal(arr[index,6],arr[:,6])
    np.testing.assert_array_equal(choices[index],choices)
    np.testing.assert_array_equal(index,
        r95._shuffle_index(arr,seed,panel,choices))
    with pytest.raises(PermissionError,match="R95_SHUFFLE_ONLY_FRESH_FROZEN_PANEL_GEOMETRY"):
        r95._shuffle_index(arr,seed,"R94_DEV",choices)


def test_precommitted_thresholds_and_without_scored_data_we_do_not_claim_science():
    f=r95.frozen()
    c=f["evaluation"]["conventional_checks"]
    n=f["evaluation"]["native_sidecar_checks"]
    assert len(c)==6 and len(n)==4
    assert c["MLP_tree_minus_raw_DEV_reward_min"]==.2
    assert c["MLP_tree_minus_shuffled_DEV_reward_min"]==.75
    assert c["MLP_tree_minus_shuffled_OOD_cold_reward_min"]==.5
    assert c["MLP_tree_vs_explicit_public_parity_DEV_reward_gap_max"]==.35
    assert c["MLP_tree_legal_DEV_action_agreement_min"]==.9
    assert c["MLP_tree_minus_shuffled_DEV_paired_bootstrap_lower95_min"]==.1
    assert n["native_tree_minus_native_flat_DEV_reward_min"]==.2
    assert n["native_tree_minus_native_raw_DEV_reward_min"]==.75
    assert n["native_tree_minus_native_flat_OOD_cold_reward_min"]==.2
    assert n["native_tree_vs_conventional_MLP_tree_DEV_reward_gap_max"]==.3
    assert r95.STEPS[r95.NATIVE_TREE]==1024
    assert r95.STEPS[r95.MLP_TREE]==4096
    assert f["training"]["matched_params_compute"] is False
    assert f["evaluation"]["two_independent_seeds_required"]
    assert f["evaluation"]["independent_scientific_pass"] is False


def test_r95_spent_first_use_is_denied_on_any_new_commit_or_run(monkeypatch):
    assert r95.ORIGINAL_R95_SPENT_SHA=="bc80edd63101cb8f293769a1e6345689acca0705"
    assert r95.ORIGINAL_R95_SPENT_RUN_ID=="43792098794"
    values={
        "GITHUB_ACTIONS":"true",
        "GITHUB_EVENT_NAME":"pull_request",
        "GITHUB_REPOSITORY":"spoonman136668-ai/TTT-Hybrid-Lab",
        "GITHUB_HEAD_REF":r95.BRANCH,
        "GITHUB_JOB":"r95-learned-relation",
        "R95_SEED":str(r95.SEEDS[0]),
        "RUNNER_OS":"Linux",
        "RUNNER_ENVIRONMENT":"github-hosted",
        "GITHUB_RUN_ATTEMPT":"1",
        "GITHUB_REF":"refs/pull/291/merge",
        "EXPECTED_CANDIDATE_SHA":"a"*40,
        "GITHUB_RUN_ID":"43799999999",
    }
    for key,val in values.items():
        monkeypatch.setenv(key,val)
    with pytest.raises(PermissionError,match="R95_SPENT_TEST_ONLY_FIRST_USE_REPLAY_DENIED"):
        r95._admit(r95.SEEDS[0])
    with pytest.raises(PermissionError,match="R95_SOURCE_NOT_ADMITTED_FOR_ORIGINAL_SCIENCE"):
        r95._world("FIT",r95.SEEDS[0],0,"a"*40)


def test_known_r95_shuffled_negative_has_index_law_anticorrelation_without_opening_scored_data():
    # SOURCE-DESIGN AUDIT, not old world readback:
    # Under the immutable R85 pure constructor the physical A-law index
    # stratum alternates by index//2; the fallback np.roll inside a public
    # pivot group maps to a neighboring index with often opposite law.
    # Thus cross-world "shuffle" is NOT a valid uninformative causal negative.
    # Keep its observed result frozen and do NOT rescore its spent worlds.
    for seed in r95.SEEDS:
        for panel in ("DEV","OOD_SHIFT"):
            n=r95.COUNT[panel]
            ix=np.arange(n)
            raw=np.zeros((n,8),np.float32)
            raw[:,6]=np.where(ix%2==0,.3,.9)
            move=r95._shuffle_index(raw,seed,panel,None)
            orig=((ix//2)%2)
            moved=((move//2)%2)
            anticorrelated=float(np.mean(moved!=orig))
            assert anticorrelated>0.70
            # This negative has a predictable relation to the simulator's
            # index-based law. It cannot establish causal cue destruction.
            assert (move!=ix).all()
